"""
Qdrant Vector Database Client

Simple HTTP client for interacting with Qdrant vector database.

Where Qdrant is, and the API key it takes, come from `ENDPOINT[]`, which Kaimon installs: it answers
from the search settings and the managed instance's record (`qdrant_endpoint`). The answer is
cached for a few seconds, since indexing makes many requests.
"""
module QdrantClient

using HTTP
using JSON

# Qdrant server configuration
const QDRANT_URL = Ref("http://localhost:6333")
const API_KEY = Ref("")
const ENDPOINT = Ref{Any}(nothing)   # () -> (url, key); installed by Kaimon
const _RESOLVED_AT = Ref(0.0)
const _RESOLVE_S = 5.0

"""
    set_url(url::String)

Set the Qdrant server URL (default: http://localhost:6333). An installed `ENDPOINT` resolver
replaces it on its next refresh.
"""
function set_url(url::String)
    QDRANT_URL[] = url
end

"The Qdrant base URL, refreshed from `ENDPOINT[]` when the cached answer is older than `_RESOLVE_S`."
function endpoint()
    f = ENDPOINT[]
    if f !== nothing && time() - _RESOLVED_AT[] > _RESOLVE_S
        try
            url, key = f()
            QDRANT_URL[] = url
            API_KEY[] = key
        catch e
            @debug "Qdrant endpoint resolution failed" exception = e
        end
        _RESOLVED_AT[] = time()
    end
    return QDRANT_URL[]
end

"Make the next request re-resolve the endpoint (after starting or stopping a managed instance)."
refresh!() = (_RESOLVED_AT[] = 0.0; nothing)

"Point the client at `url` with `key` now, as the resolver would."
configure!(url::AbstractString, key::AbstractString) =
    (QDRANT_URL[] = String(url); API_KEY[] = String(key); _RESOLVED_AT[] = time(); nothing)

"Request headers: the API key when there is one, then `extra`."
_hdrs(extra::Pair...) = isempty(API_KEY[]) ? Pair{String,String}[extra...] :
                         Pair{String,String}["api-key" => API_KEY[], extra...]

_json_hdrs() = _hdrs("Content-Type" => "application/json")

_get(path::AbstractString; kw...) = HTTP.get("$(endpoint())$path", _hdrs(); kw...)

# Every POST used here (search, scroll, count, delete by id or filter) is safe to repeat,
# so let HTTP.jl retry it as it would a GET. Otherwise a pooled keep-alive connection
# that Qdrant closed while idle fails the request with ECONNRESET instead of reconnecting.
_post(path::AbstractString, body) = HTTP.post("$(endpoint())$path",
    _json_hdrs(), JSON.json(body); retry_non_idempotent = true)

"""
    list_collections() -> Vector{String}

List all collections in the Qdrant instance.
"""
function list_collections()
    try
        response = _get("/collections")
        data = JSON.parse(String(response.body))

        if haskey(data, "result") && haskey(data["result"], "collections")
            return [c["name"] for c in data["result"]["collections"]]
        end
        return String[]
    catch e
        @error "Failed to list collections" exception = e
        return String[]  # empty → callers check length; _friendly_error used at tool layer
    end
end

"""
    get_collection_info(collection::String) -> Dict

Get information about a specific collection.
"""
function get_collection_info(collection::String)
    try
        response = _get("/collections/$(collection)")
        data = JSON.parse(String(response.body))
        return get(data, "result", Dict())
    catch e
        @error "Failed to get collection info" collection = collection exception = e
        return Dict("error" => _friendly_error(e))
    end
end

"""
    search(collection::String, vector::Vector{Float64}; limit::Int=5, score_threshold::Float64=0.0, filter::Union{Dict,Nothing}=nothing) -> Vector{Dict}

Search for similar vectors in a collection.

# Arguments
- `collection::String`: Name of the collection to search
- `vector::Vector{Float64}`: Query vector
- `limit::Int`: Maximum number of results (default: 5)
- `score_threshold::Float64`: Minimum similarity score (default: 0.0)
- `filter::Union{Dict,Nothing}`: Qdrant filter for metadata (e.g., Dict("must" => [Dict("key" => "type", "match" => Dict("value" => "function"))]))

# Returns
Vector of result dictionaries containing id, score, and payload.
"""
function search(
    collection::String,
    vector::Vector{Float64};
    limit::Int = 5,
    score_threshold::Float64 = 0.0,
    filter::Union{Dict,Nothing} = nothing,
)
    try
        body = Dict(
            "vector" => vector,
            "limit" => limit,
            "score_threshold" => score_threshold,
            "with_payload" => true,
            "with_vector" => false,
        )

        # Add filter if provided
        if filter !== nothing
            body["filter"] = filter
        end

        response = _post("/collections/$(collection)/points/search", body)

        data = JSON.parse(String(response.body))
        return get(data, "result", [])
    catch e
        @error "Search failed" collection = collection exception = e
        return [Dict("error" => _friendly_error(e))]
    end
end

"""
    search_with_text(collection::String, query_text::String; limit::Int=5, embed_func::Function) -> Vector{Dict}

Search using text by first converting to embedding.

# Arguments
- `collection::String`: Name of the collection to search
- `query_text::String`: Text query to search for
- `limit::Int`: Maximum number of results (default: 5)
- `embed_func::Function`: Function that takes text and returns Vector{Float64} embedding
"""
function search_with_text(
    collection::String,
    query_text::String;
    limit::Int = 5,
    embed_func::Function,
)
    try
        # Get embedding for query text
        embedding = embed_func(query_text)

        # Search using the embedding
        return search(collection, embedding; limit = limit)
    catch e
        @error "Text search failed" collection = collection exception = e
        return [Dict("error" => _friendly_error(e))]
    end
end

"""
    scroll_points(collection::String; limit::Int=10, offset=nothing) -> Dict

Retrieve points from a collection with pagination.

# Arguments
- `collection::String`: Name of the collection
- `limit::Int`: Number of points to retrieve (default: 10)
- `offset`: Offset for pagination (optional)
"""
function scroll_points(collection::String; limit::Int = 10, offset = nothing, with_vector::Bool = false)
    try
        body = Dict{String,Any}("limit" => limit, "with_payload" => true, "with_vector" => with_vector)

        if offset !== nothing
            # Qdrant accepts both integer and UUID string offsets
            body["offset"] = offset isa Integer ? offset : string(offset)
        end

        response = _post("/collections/$(collection)/points/scroll", body)

        data = JSON.parse(String(response.body))
        return get(data, "result", Dict())
    catch e
        @error "Scroll failed" collection = collection exception = e
        return Dict("error" => _friendly_error(e))
    end
end

"""
    get_point(collection::String, point_id) -> Dict

Retrieve a specific point by ID.
"""
function get_point(collection::String, point_id)
    try
        response = _get("/collections/$(collection)/points/$(point_id)")
        data = JSON.parse(String(response.body))
        return get(data, "result", Dict())
    catch e
        @error "Failed to get point" collection = collection point_id = point_id exception =
            e
        return Dict("error" => _friendly_error(e))
    end
end

"""
    point_payload_field(collection, ids, field) -> Dict{String,Any}

The payload `field` of each point among `ids` that exists, keyed by ID (`nothing` where a point lacks
the field). Empty on any failure, so a caller treats every point as missing.
"""
function point_payload_field(collection::String, ids::AbstractVector, field::String)
    out = Dict{String,Any}()
    isempty(ids) && return out
    try
        body = Dict("ids" => map(_coerce_point_id, ids), "with_payload" => [field], "with_vector" => false)
        data = JSON.parse(String(_post("/collections/$(collection)/points", body).body))
        for p in get(data, "result", [])
            pl = get(p, "payload", nothing)
            out[string(p["id"])] = pl isa AbstractDict ? get(pl, field, nothing) : nothing
        end
    catch e
        @debug "Point lookup failed" collection = collection exception = e
        empty!(out)
    end
    return out
end

"""Transient connection failures worth retrying for an idempotent request — a stale
HTTP keepalive connection Qdrant already closed (→ `unexpected EOF` / ECONNRESET) or a
brief gateway hiccup. Surfaces under sustained upsert load."""
_is_transient_http(e) =
    e isa HTTP.ParseError || e isa Base.IOError || e isa Base.SystemError ||
    e isa EOFError || e isa HTTP.ConnectError || e isa HTTP.TimeoutError ||
    (e isa HTTP.StatusError && e.status in (502, 503, 504))

"""Run `f()` with a few retries on transient connection errors (exponential backoff).
Used for idempotent requests (Qdrant upserts carry explicit point IDs, so a retried,
partially-applied batch just overwrites the same points)."""
function _with_http_retry(f; tries::Int=4)
    local last_err
    for attempt in 1:tries
        try
            return f()
        catch e
            last_err = e
            (_is_transient_http(e) && attempt < tries) || rethrow()
            sleep(0.1 * 2.0^(attempt - 1))   # 0.1s, 0.2s, 0.4s
        end
    end
    throw(last_err)
end

"""
    upsert_points(collection::String, points::Vector{Dict}) -> Bool

Upsert points into a collection.

# Arguments
- `collection::String`: Name of the collection
- `points::Vector{Dict}`: Vector of point dictionaries with keys:
  - `id`: Point ID (string UUID or integer)
  - `vector`: Vector{Float64} embedding
  - `payload`: Dict with metadata

# Returns
true on success, false on failure.
"""
function upsert_points(collection::String, points::Vector{Dict})
    try
        body = Dict("points" => points)

        response = _with_http_retry() do
            HTTP.put(
                "$(endpoint())/collections/$(collection)/points",
                _json_hdrs(),
                JSON.json(body),
            )
        end

        data = JSON.parse(String(response.body))
        return get(data, "status", "") == "ok"
    catch e
        @error "Upsert failed" collection = collection reason = sprint(showerror, e)
        return false
    end
end

"""
    delete_collection(collection::String) -> Bool

Delete a collection.
"""
function delete_collection(collection::String)
    try
        response = HTTP.delete("$(endpoint())/collections/$(collection)", _hdrs())
        data = JSON.parse(String(response.body))
        return get(data, "result", false) == true
    catch e
        @error "Delete collection failed" collection = collection exception = e
        return false
    end
end

"""
    create_collection(collection::String; vector_size::Int=768, distance::String="Cosine") -> Bool

Create a new collection.
"""
function create_collection(
    collection::String;
    vector_size::Int = 768,
    distance::String = "Cosine",
)
    try
        body = Dict("vectors" => Dict("size" => vector_size, "distance" => distance))

        response = HTTP.put(
            "$(endpoint())/collections/$(collection)",
            _json_hdrs(),
            JSON.json(body),
        )

        data = JSON.parse(String(response.body))
        return get(data, "result", false) == true
    catch e
        # 409 Conflict = the collection already exists. Qdrant's PUT-create isn't
        # idempotent, but the desired end state (collection present) is met — so
        # treat it as success. This keeps concurrent indexers racing to create the
        # same collection (common against a fresh instance) from spamming errors.
        if e isa HTTP.StatusError && e.status == 409
            return true
        end
        @error "Create collection failed" collection = collection exception = e
        return false
    end
end

"""
    _coerce_point_id(id)

Normalize a point ID to Qdrant's accepted forms: an unsigned integer or a UUID
string. Numeric values — and all-digit strings like `"1"` — become integers so
they match points stored with integer IDs; UUID/other strings pass through
unchanged. (Qdrant treats `1` and `"1"` as different IDs.)
"""
function _coerce_point_id(id)
    id isa Integer && return id
    id isa Real && return isinteger(id) ? Int(id) : id   # JSON numbers as Float64
    if id isa AbstractString
        n = tryparse(Int, id)
        return n !== nothing ? n : String(id)
    end
    return id
end

"""
    delete_points(collection::String, point_ids::AbstractVector) -> Bool

Delete specific points by their IDs. IDs are normalized via `_coerce_point_id`
so numeric IDs delete points stored with integer IDs.

# Returns
true on success, false on failure.
"""
function delete_points(collection::String, point_ids::AbstractVector)
    isempty(point_ids) && return true
    try
        body = Dict("points" => map(_coerce_point_id, point_ids))
        response = _post("/collections/$(collection)/points/delete", body)
        data = JSON.parse(String(response.body))
        return get(data, "status", "") == "ok"
    catch e
        @error "Delete points failed" collection = collection exception = e
        return false
    end
end

"""
    collection_exists(collection::String) -> Bool

Check whether a collection exists in Qdrant.
"""
function collection_exists(collection::String)
    return collection in list_collections()
end

"""
    delete_by_filter(collection::String, filter::Dict) -> Bool

Delete points matching a filter condition.

# Arguments
- `collection::String`: Name of the collection
- `filter::Dict`: Qdrant filter condition (e.g., Dict("must" => [Dict("key" => "file", "match" => Dict("value" => "/path/to/file.jl"))]))

# Returns
true on success, false on failure.
"""
function delete_by_filter(collection::String, filter::Dict)
    try
        body = Dict("filter" => filter)

        response = _post("/collections/$(collection)/points/delete", body)

        data = JSON.parse(String(response.body))
        return get(data, "status", "") == "ok"
    catch e
        @error "Delete by filter failed" collection = collection exception = e
        return false
    end
end

"""
    delete_by_file(collection::String, file_path::String) -> Bool

Delete all points from a specific file.

# Arguments
- `collection::String`: Name of the collection
- `file_path::String`: Path to the file whose chunks should be deleted

# Returns
true on success, false on failure.
"""
function delete_by_file(collection::String, file_path::String)
    filter = Dict("must" => [Dict("key" => "file", "match" => Dict("value" => file_path))])
    return delete_by_filter(collection, filter)
end

"""
    count_by_file(collection::String, file_path::String) -> Int

Count points from a specific file.

# Returns
Number of points from the file, or -1 on error.
"""
function count_by_file(collection::String, file_path::String)
    try
        filter =
            Dict("must" => [Dict("key" => "file", "match" => Dict("value" => file_path))])
        body = Dict("filter" => filter, "exact" => true)

        response = _post("/collections/$(collection)/points/count", body)

        data = JSON.parse(String(response.body))
        return get(get(data, "result", Dict()), "count", 0)
    catch e
        @error "Count by file failed" collection = collection exception = e
        return -1
    end
end

"""
    probe() -> Symbol

`:ok` when Qdrant answers a request that needs the API key, `:refused` when it answers but rejects
the key (it is not the instance this key belongs to), `:down` otherwise. `/healthz` is no use for
this: Qdrant serves it without a key.
"""
function probe()::Symbol
    try
        r = _get("/collections"; connect_timeout = 2, request_timeout = 3,
                 status_exception = false, retry = false)
        r.status == 200 && return :ok
        r.status in (401, 403) && return :refused
        return :down
    catch
        return :down
    end
end

"""
    ping() -> Bool

Check if Qdrant is reachable and accepts our API key.
"""
ping() = probe() === :ok

"""
    _friendly_error(e) -> String

Classify an exception into a user-friendly error message.
"""
function _friendly_error(e)
    msg = string(e)
    if e isa HTTP.ConnectError ||
       contains(msg, "ConnectError") ||
       contains(msg, "connection refused")
        return "Qdrant is not reachable at $(QDRANT_URL[]). Is it running?"
    elseif e isa HTTP.StatusError
        status = e.status
        if status == 404
            return "Collection not found (404)"
        elseif status in (401, 403)
            return "Qdrant at $(QDRANT_URL[]) rejected the API key ($status)"
        elseif status == 400
            return "Bad request to Qdrant (400): $(first(msg, 200))"
        else
            return "Qdrant returned HTTP $status"
        end
    elseif contains(msg, "TimeoutError") || contains(msg, "timeout")
        return "Qdrant request timed out"
    else
        return "Qdrant error: $(first(msg, 300))"
    end
end

end # module
