using ReTest
using Kaimon
using Kaimon.Database
using SQLite
using DBInterface

# ── cancel_eval honesty ──────────────────────────────────────────────────────
# Regression: cancel_eval used to write status='cancelled' (plus finished_at)
# at REQUEST time — before the eval had stopped. A job whose code never
# observed is_cancelled() stayed 'cancelled' in the DB forever while still
# burning CPU, and a job that later finished normally had its row rewritten to
# 'completed'. The fix records the request without ending the job: status stays
# 'running' (so restart reconciliation still finds the row and no consumer has
# to learn a new status), `cancelled` + `cancel_requested_at` carry the request,
# check_eval reports it as pending, and the collector writes the terminal
# 'cancelled' (with the result) only when the code actually stops.

const _CANCEL_TEST_CODE = "for i in 1:10^9; KaimonGate.is_cancelled() && break; end"

function _cancel_test_fixture(eval_id::String; status::Symbol = :promoted)
    Database.init_db!(tempname() * ".db")
    mgr = Kaimon.ConnectionManager(sock_dir = tempdir())
    Database.persist_job!(eval_id, "sess1234", _CANCEL_TEST_CODE, time(), time())
    rec = Kaimon.EvalRecord(eval_id, "sess1234", _CANCEL_TEST_CODE,
                            time(), 0.0, 0.0, time(), status, "", "", true,
                            Dict{String, String}())
    push!(mgr.eval_history, rec)
    mgr, rec
end

function with_conn_mgr(f::Function, mgr)
    old = Kaimon.GATE_CONN_MGR[]
    Kaimon.GATE_CONN_MGR[] = mgr
    try
        f()
    finally
        Kaimon.GATE_CONN_MGR[] = old
    end
end

@testset "cancel_eval marks the request, not the result" begin
    eval_id = "cafe0001"
    mgr, rec = _cancel_test_fixture(eval_id)
    with_conn_mgr(mgr) do
        # No REPLConnection exists for the session, so the :cancel_job notify is
        # skipped — this tests the record + DB transition.
        resp = Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id))

        @test rec.cancel_requested_at > 0.0
        @test rec.status == :promoted          # not ended by the request
        @test rec.finished_at == 0.0           # the job has NOT finished
        @test occursin("Cancellation requested", string(resp))

        job = Database.get_job(eval_id)
        @test job !== nothing
        @test job["status"] == "running"       # still reconcilable after a restart
        @test job["cancelled"] == 1
        @test job["cancel_requested_at"] > 0.0
        @test job["finished_at"] == 0.0        # not stamped at request time
        # A pending request is not a cancelled job: stats count only stopped rows.
        @test Database.get_job_stats()["cancelled"] == 0
    end
end

@testset "check_eval reports a pending cancel request" begin
    eval_id = "cafe0002"
    mgr, rec = _cancel_test_fixture(eval_id)
    with_conn_mgr(mgr) do
        Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id))
        resp = string(Kaimon.check_eval_tool.handler(Dict("eval_id" => eval_id)))
        @test occursin("running", resp)
        @test occursin("cancel requested", resp)
    end
end

@testset "check_eval reports it from the DB for a previous session" begin
    eval_id = "cafe0003"
    mgr, _ = _cancel_test_fixture(eval_id)   # in-memory record dropped below
    empty!(mgr.eval_history)
    with_conn_mgr(mgr) do
        Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id))
        resp = string(Kaimon.check_eval_tool.handler(Dict("eval_id" => eval_id)))
        @test occursin("running", resp)
        @test occursin("cancel requested", resp)
    end
end

@testset "terminal status: a pending request wins over completed/failed" begin
    eval_id = "cafe0004"
    mgr, rec = _cancel_test_fixture(eval_id)
    @test Kaimon._job_terminal_status(mgr, eval_id, false) == :completed
    @test Kaimon._job_terminal_status(mgr, eval_id, true) == :failed

    rec.cancel_requested_at = time()
    @test Kaimon._job_terminal_status(mgr, eval_id, false) == :cancelled
    @test Kaimon._job_terminal_status(mgr, eval_id, true) == :cancelled
end

@testset "cancel_eval leaves a finished job alone" begin
    eval_id = "cafe0005"
    mgr, rec = _cancel_test_fixture(eval_id)
    Kaimon._record_eval_done!(mgr, eval_id, :completed, "done")
    Database.update_job!(eval_id; status = "completed", result = "done", finished_at = time())
    with_conn_mgr(mgr) do
        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id)))
        @test occursin("already completed", resp)
        @test rec.cancel_requested_at == 0.0
        @test rec.status == :completed
        job = Database.get_job(eval_id)
        @test job["status"] == "completed"
        @test job["cancelled"] == 0
        @test job["cancel_requested_at"] == 0.0
        @test Database.get_job_stats()["cancelled"] == 0
    end
end

@testset "cancel_eval leaves a finished job alone (DB only)" begin
    eval_id = "cafe0006"
    mgr, _ = _cancel_test_fixture(eval_id)
    empty!(mgr.eval_history)   # a job from a previous session: only its row is known
    Database.update_job!(eval_id; status = "completed", finished_at = time())
    with_conn_mgr(mgr) do
        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id)))
        @test occursin("already completed", resp)
        @test Database.get_job(eval_id)["cancelled"] == 0

        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => "cafe9999")))
        @test occursin("No running job matching", resp)
    end
end

@testset "cancel by prefix uses the full id" begin
    eval_id = "cafe0abc"
    mgr, rec = _cancel_test_fixture(eval_id)
    with_conn_mgr(mgr) do
        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => "cafe0a")))
        @test occursin("Cancellation requested for $eval_id", resp)
        @test rec.cancel_requested_at > 0.0
        job = Database.get_job(eval_id)
        @test job["cancelled"] == 1
        @test job["cancel_requested_at"] > 0.0
    end
end

@testset "cancel refuses an ambiguous prefix" begin
    Database.init_db!(tempname() * ".db")
    mgr = Kaimon.ConnectionManager(sock_dir = tempdir())
    ids = ["cafe0a01", "cafe0a02"]
    for id in ids
        Database.persist_job!(id, "sess1234", _CANCEL_TEST_CODE, time(), time())
        push!(mgr.eval_history, Kaimon.EvalRecord(id, "sess1234", _CANCEL_TEST_CODE,
                                                  time(), 0.0, 0.0, time(), :promoted,
                                                  "", "", true, Dict{String, String}()))
    end
    with_conn_mgr(mgr) do
        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => "cafe0a")))
        @test occursin("matches 2 running", resp)
        @test all(r -> r.cancel_requested_at == 0.0, mgr.eval_history)
        @test all(id -> Database.get_job(id)["cancelled"] == 0, ids)
    end
end

@testset "cancel refuses an ambiguous prefix (DB only)" begin
    Database.init_db!(tempname() * ".db")
    mgr = Kaimon.ConnectionManager(sock_dir = tempdir())
    ids = ["cafe0b01", "cafe0b02"]
    for id in ids
        Database.persist_job!(id, "sess1234", _CANCEL_TEST_CODE, time(), time())
    end
    with_conn_mgr(mgr) do
        resp = string(Kaimon.cancel_eval_tool.handler(Dict("eval_id" => "cafe0b")))
        @test occursin("matches 2 running", resp)
        @test all(id -> Database.get_job(id)["cancelled"] == 0, ids)
    end
end

@testset "the collector's write records a stopped cancelled job" begin
    eval_id = "cafe0007"
    mgr, _ = _cancel_test_fixture(eval_id)
    with_conn_mgr(mgr) do
        Kaimon.cancel_eval_tool.handler(Dict("eval_id" => eval_id))
        @test Database.get_job_stats()["cancelled"] == 0   # pending: not counted

        # What the collector does once the code actually stops.
        Database.update_job!(eval_id; status = "cancelled", result = "stopped",
                            finished_at = time())
        job = Database.get_job(eval_id)
        @test job["status"] == "cancelled"
        @test job["result"] == "stopped"
        @test job["finished_at"] > 0.0
        @test Database.get_job_stats()["cancelled"] == 1   # stopped: counted
    end
end

@testset "cancel_requested_at is added to a pre-existing database" begin
    # A DB created before the column existed must gain it rather than erroring
    # on the first cancel_eval.
    path = tempname() * ".db"
    db = SQLite.DB(path)
    DBInterface.execute(db, """CREATE TABLE background_jobs (
        eval_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, code TEXT NOT NULL,
        started_at REAL NOT NULL, finished_at REAL DEFAULT 0.0, promoted_at REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'running', result TEXT DEFAULT '',
        result_preview TEXT DEFAULT '', cancelled INTEGER DEFAULT 0)""")
    DBInterface.execute(db, """INSERT INTO background_jobs
        (eval_id, session_key, code, started_at, promoted_at, status)
        VALUES ('cafe0008', 'sess1234', 'x', 0.0, 0.0, 'completed')""")
    SQLite.close(db)

    Database.init_db!(path)
    Database.update_job!("cafe0008"; cancelled=true, cancel_requested_at = 123.0)
    job = Database.get_job("cafe0008")
    @test job["cancel_requested_at"] == 123.0
    @test job["cancelled"] == 1
    @test job["status"] == "completed"   # the request columns are all it touched

    Database.init_db!(path)              # idempotent: no duplicate-column error
    @test Database.get_job("cafe0008")["cancel_requested_at"] == 123.0
end
