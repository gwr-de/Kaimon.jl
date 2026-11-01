using ReTest
using Kaimon

const KL = Kaimon

# A fake Ollama store: one manifest naming a weights blob, and the blob.
function _fake_ollama_store(dir; ns = "library", name = "qwen3-embedding", tag = "0.6b",
                            digest = "sha256:" * "ab"^32)
    mf = joinpath(dir, "manifests", "registry.ollama.ai", ns, name)
    mkpath(mf)
    write(joinpath(mf, tag), KL.JSON.json(Dict("layers" => [
        Dict("mediaType" => "application/vnd.ollama.image.model", "digest" => digest, "size" => 4),
        Dict("mediaType" => "application/vnd.ollama.image.license", "digest" => "sha256:" * "cd"^32, "size" => 1),
    ])))
    mkpath(joinpath(dir, "blobs"))
    blob = joinpath(dir, "blobs", replace(digest, ":" => "-"))
    write(blob, "GGUF")
    return blob
end

@testset "Managed llama.cpp embeddings" begin
    @testset "Ollama model names" begin
        @test KL.split_model_name("qwen3-embedding:0.6b") == ("library", "qwen3-embedding", "0.6b")
        @test KL.split_model_name("nomic-embed-text") == ("library", "nomic-embed-text", "latest")
        @test KL.split_model_name("someone/model:q8") == ("someone", "model", "q8")
    end

    @testset "model file: an Ollama store's copy, else a downloaded one" begin
        store, models = mktempdir(), mktempdir()
        try
            blob = _fake_ollama_store(store)
            withenv("OLLAMA_MODELS" => store, "KAIMON_MODEL_DIR" => models) do
                @test KL.ollama_store_model_file("qwen3-embedding:0.6b") == blob   # the weights layer
                @test KL.embedding_model_file("qwen3-embedding:0.6b") == blob
                @test KL.embedding_model_file("qwen3-embedding:4b") === nothing
                @test KL.downloaded_model_path("qwen3-embedding:4b") ==
                      joinpath(models, "library+qwen3-embedding+4b.gguf")
                write(KL.downloaded_model_path("qwen3-embedding:4b"), "GGUF")
                @test KL.embedding_model_file("qwen3-embedding:4b") == KL.downloaded_model_path("qwen3-embedding:4b")
                @test sort(KL.llamacpp_local_models()) == ["qwen3-embedding:0.6b", "qwen3-embedding:4b"]
            end
        finally
            rm(store; recursive = true, force = true); rm(models; recursive = true, force = true)
        end
    end

    @testset "server: loopback, the launch's key, pooling from the model file" begin
        args = KL.llamacpp_server_args("/m.gguf", 23456; threads = 4)
        @test args[findfirst(==("--host"), args) + 1] == "127.0.0.1"
        @test args[findfirst(==("--port"), args) + 1] == "23456"
        # The key is passed in the environment, never on the command line.
        cmd = KL.llamacpp_server_command("/bin/llama-server", "/m.gguf", 23456, "k789")
        @test !any(a -> occursin("k789", a), cmd.exec)
        @test "LLAMA_API_KEY=k789" in cmd.env
        @test "--embeddings" in args
        @test !("--pooling" in args)
        @test args[findfirst(==("-ub"), args) + 1] == string(KL.LLAMACPP_BATCH)
        # The longest text sent fits one batch even at two characters to a token.
        for (name, cfg) in KL.EMBEDDING_CONFIGS
            startswith(name, "qwen3-embedding") && @test cfg.context_chars ÷ 2 <= KL.LLAMACPP_BATCH
        end
        @test occursin(gethostname(), KL.llamacpp_record_path("qwen3-embedding:0.6b"))
        @test !occursin(':', basename(KL.llamacpp_record_path("qwen3-embedding:0.6b")))
    end

    @testset "a configured llama-server replaces the JLL's" begin
        d = mktempdir()
        try
            bin = joinpath(d, "llama-server"); write(bin, "#!/bin/sh\n")
            withenv("KAIMON_LLAMACPP_SERVER" => bin, "KAIMON_CACHE_DIR" => mktempdir()) do
                @test KL.llamacpp_installed()                     # no JLL needed
                @test KL.llamacpp_server_cmd() == Cmd([bin])
            end
            withenv("KAIMON_LLAMACPP_SERVER" => joinpath(d, "missing"), "KAIMON_CACHE_DIR" => mktempdir()) do
                @test !KL.llamacpp_installed()
                @test (@test_logs (:warn,) KL.llamacpp_server_cmd()) === nothing
            end
        finally
            rm(d; recursive = true, force = true)
        end
    end

    @testset "not installed: nothing is started" begin
        withenv("KAIMON_CACHE_DIR" => mktempdir()) do
            @test !KL.llamacpp_installed()
            @test KL.ensure_llamacpp!("qwen3-embedding:0.6b") === nothing
            @test isempty(KL._LLAMACPP_PROCS)
        end
    end

    @testset "model names match by tag" begin
        @test KL._model_matches("qwen3-embedding:0.6b", "qwen3-embedding:0.6b")
        @test !KL._model_matches("qwen3-embedding:0.6b", "qwen3-embedding:8b")   # another model
        @test KL._model_matches("nomic-embed-text:latest", "nomic-embed-text")
        @test !KL._model_matches("nomic-embed-text:v1.5", "nomic-embed-text")
    end

    @testset "the backend setting picks what embeds" begin
        store = mktempdir()
        try
            _fake_ollama_store(store)
            withenv("KAIMON_EMBEDDING_BACKEND" => "llamacpp", "OLLAMA_MODELS" => store,
                    "KAIMON_CACHE_DIR" => mktempdir()) do
                @test KL.embedding_backend() === :llamacpp
                @test KL.embedder_name() == "llama.cpp"
                @test !KL.ping_embedder()                                  # not installed here
                @test KL.embedding_model_available("qwen3-embedding:0.6b")  # the file exists
                @test KL.embedding_model_available("missing:1b") == false
                msg = KL._require_services(need_qdrant = false, need_embedder = true,
                                           model = "qwen3-embedding:0.6b")
                @test msg !== nothing && occursin("llama.cpp", msg)
            end
        finally
            rm(store; recursive = true, force = true)
        end
    end
end
