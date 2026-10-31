# (Kaimon Qdrant indexer — split into qdrant_indexer_*.jl; this one loads FIRST)
"""
Code Indexer for Qdrant Vector Search

Indexes Julia source files into Qdrant for semantic code search.
Uses Ollama for embeddings (qwen3-embedding model).
"""

# Uses QdrantClient, get_embedding, and DEFAULT_EMBEDDING_MODEL from parent scope

# ── Collection prefix for shared Qdrant instances ────────────────────────────
# Set via KAIMON_QDRANT_PREFIX env var or config.json "qdrant_prefix" field.
# When set, all collection names are prefixed: "myprefix_projectname"
const _QDRANT_COLLECTION_PREFIX = Ref{String}("")

"""
    set_collection_prefix!(prefix::AbstractString)

Set a prefix for all Qdrant collection names. Useful when multiple users
share a single Qdrant instance. Set to "" to disable.
"""
function set_collection_prefix!(prefix::AbstractString)
    _QDRANT_COLLECTION_PREFIX[] = String(prefix)
end

"""
    get_collection_prefix() -> String

Return the current Qdrant collection prefix (empty string if none).
"""
get_collection_prefix() = _QDRANT_COLLECTION_PREFIX[]

# Cross-project search queries the project collections themselves (see `search_code`), so no
# vector is stored twice. Older versions kept a merged copy of every project in one collection;
# this is its name, so it can be skipped where collections are enumerated and removed.
"""Name of the merged cross-project collection older versions kept."""
global_collection_name() = _prefixed("kaimon_all")

"""
    drop_legacy_global_collection!() -> Bool

Delete the merged cross-project collection older versions kept: a copy of vectors the project
collections already hold. True if there was one.
"""
function drop_legacy_global_collection!()::Bool
    gc = global_collection_name()
    try
        QdrantClient.collection_exists(gc) || return false
        QdrantClient.delete_collection(gc)
        with_index_logger(() -> @info "Removed the merged cross-project collection; search queries project collections directly" collection = gc)
        return true
    catch
        return false
    end
end

"""Apply the collection prefix to a name, if configured."""
function _prefixed(name::String)
    prefix = _QDRANT_COLLECTION_PREFIX[]
    isempty(prefix) ? name : "$(prefix)_$(name)"
end


# Extensible PDF text extraction — implemented by KaimonPDFIOExt when PDFIO is loaded.
function _extract_pdf_text end

# Embedding model configuration. `context_chars` is the longest text sent for one embedding; a
# longer chunk is split. The time and memory an embedding takes grow with the square of its length,
# so this sits well under what the model accepts, which keeps one embedding to seconds on a CPU.
const EMBEDDING_CONFIGS = Dict(
    "embeddinggemma:latest" => (dims=768, context_tokens=2048, context_chars=4000),
    "qwen3-embedding:0.6b" => (dims=1024, context_tokens=8192, context_chars=4000),
    "qwen3-embedding:4b" => (dims=2560, context_tokens=8192, context_chars=4000),
    "qwen3-embedding:8b" => (dims=4096, context_tokens=8192, context_chars=4000),
    "snowflake-arctic-embed:latest" => (dims=1024, context_tokens=512, context_chars=1000),
    "nomic-embed-text" => (dims=768, context_tokens=512, context_chars=1000),
)

const CHUNK_SIZE = 1500  # Target chunk size in characters
const CHUNK_OVERLAP = 200  # Overlap between chunks

# Default file extensions to index. Broad on purpose: an extension costs nothing for a
# project that has no files of that type, whereas a narrow default silently indexes
# NOTHING for a project in a language it omits. Per-project `extensions` config (or
# auto-detection) narrows this where it matters.
#
# `.js`/`.mjs` are deliberately absent — in a repo that ships JS, the committed `.js` is
# usually build output; TypeScript/JSX sources are indexed instead. Data and lock formats
# (`.json`, `.xml`, `.lock`) are excluded as bulk without search value. Both can be opted
# into explicitly per project.
const DEFAULT_INDEX_EXTENSIONS = [
    ".jl",                                    # Julia
    ".py", ".pyi",                            # Python
    ".rs",                                    # Rust
    ".go",                                    # Go
    ".ts", ".tsx", ".jsx",                    # TypeScript / JSX
    ".c", ".h", ".cpp", ".cc", ".hpp", ".hh", # C / C++
    ".java", ".kt", ".kts", ".scala",         # JVM
    ".cs", ".fs",                             # .NET
    ".rb", ".php", ".swift", ".m", ".mm",     # Ruby / PHP / Apple
    ".ex", ".exs", ".erl", ".hrl",            # BEAM
    ".hs", ".ml", ".mli", ".clj", ".cljs",    # Functional
    ".lua", ".zig", ".nim", ".r", ".R",       # Other languages
    ".sh", ".bash", ".zsh",                   # Shell
    ".sql", ".proto", ".graphql",             # Schemas / IDL
    ".vue", ".svelte",                        # Component frameworks
    ".md", ".mdx", ".rst",                    # Prose / docs
    ".toml",                                  # Project metadata (Project/Cargo/pyproject)
]

# Extensions that auto-detection will never select on its own. `.js` is the important
# one: in a JS/TS project the committed `.js` is usually build output, not source (the
# same reason it's absent from DEFAULT_INDEX_EXTENSIONS). Data/lock formats add bulk
# without helping code search. A user can still index any of these by listing them
# explicitly in the project's `extensions` config.
const AUTO_DETECT_EXTENSION_DENY = Set([".js", ".json", ".xml", ".lock", ".map"])

# Known source file extensions for auto-detection (superset of per-language lists)
const SOURCE_EXTENSIONS = Set([
    ".jl", ".py", ".rs", ".go", ".ts", ".tsx", ".js", ".jsx",
    ".c", ".cpp", ".cc", ".h", ".hpp", ".java", ".kt", ".kts",
    ".rb", ".ex", ".exs", ".zig", ".nim", ".lua", ".swift", ".m",
    ".cs", ".fs", ".scala", ".clj", ".cljs", ".erl", ".hrl",
    ".hs", ".ml", ".mli", ".r", ".R", ".jl", ".sh", ".bash",
    ".md", ".mdx", ".rst", ".toml", ".yaml", ".yml", ".json",
    ".xml", ".html", ".css", ".scss", ".sass", ".less",
    ".sql", ".graphql", ".gql", ".proto", ".tf", ".hcl",
    ".vue", ".svelte",
])

# Directories to skip during walkdir and auto-detection, even if tracked by git.
# `.gitignore` already excludes most of these in a well-kept repo, but plenty of
# projects commit their build output, and a single committed JS bundle directory is
# enough to swamp a collection — so the name list is the second line of defence.
const IGNORED_DIRS = Set([
    # Dependencies / language caches
    "node_modules", "bower_components", "jspm_packages", "vendor",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".eggs", ".egg-info", "venv", ".venv", "env", ".tox",
    # Generic build output
    "dist", "build", "_build", "out", "target", "bin", "obj",
    # JS/TS toolchain output — bundlers, frameworks, and transpile targets
    ".next", ".nuxt", ".output", ".svelte-kit", ".astro", ".docusaurus",
    ".parcel-cache", ".turbo", ".vite", ".webpack", ".rollup.cache",
    ".angular", ".vercel", ".netlify", "storybook-static",
    "es", "esm", "cjs", "umd", "amd",
    # Misc caches and reports
    "coverage", ".nyc_output", ".gradle", ".cache", "site-packages",
])
# Deliberately NOT ignored: `deps/` is real source in a Julia package (build.jl), and
# `lib/` holds sub-packages rather than build output.

# Files that are generated/compiled artifacts regardless of which directory they sit
# in. Minified bundles are the pathological case: a single line megabytes long, which
# is pure noise in a semantic index and expensive to embed.
const GENERATED_FILE_SUFFIXES = [
    ".min.js", ".min.css", ".min.mjs",
    ".bundle.js", "-bundle.js", ".bundle.mjs",
    ".chunk.js", "-chunk.js",
    ".js.map", ".css.map", ".mjs.map",
    ".d.ts", ".d.mts", ".d.cts",
    "_pb2.py", "_pb2_grpc.py", ".pb.go", "_generated.go",
]

# Hard ceilings for a single indexed file. Anything past these is treated as generated
# content: real source rarely trips them, bundles and vendored blobs always do.
const MAX_INDEX_FILE_BYTES = 1_500_000
const MAX_INDEX_LINE_BYTES = 5_000

# Get embedding config for a model
function get_embedding_config(model::String)
    return get(EMBEDDING_CONFIGS, model, (dims=768, context_tokens=512, context_chars=2000))
end

# Logging and error tracking for background indexing
const INDEX_LOGGER = Ref{Union{LoggingExtras.TeeLogger,Nothing}}(nothing)
const INDEX_ERROR_COUNT = Ref{Int}(0)
const INDEX_USER_NOTIFIED = Ref{Bool}(false)
const INDEX_FAILED_FILES = Ref{Dict{String,Int}}(Dict{String,Int}())  # file -> consecutive fail count

"""
    setup_index_logging(project_path::String=pwd())

Setup rotating log file for background indexing operations.
Log file is stored in ~/.cache/kaimon/indexer.log with 10MB max size and 3 file rotation (30MB total).
"""
function setup_index_logging(project_path::String=pwd())
    log_file = joinpath(kaimon_cache_dir(), "indexer.log")

    # Create rotating file logger (10MB max, 3 files = 30MB total)
    file_logger = LoggingExtras.MinLevelLogger(
        LoggingExtras.FileLogger(log_file; append=true, always_flush=true),
        Logging.Info
    )

    INDEX_LOGGER[] = file_logger

    @info "Indexer logging initialized" log_file = log_file
    return log_file
end

"""
    with_index_logger(f::Function)

Execute function with indexer logger active, then restore original logger.
"""
function with_index_logger(f::Function)
    if INDEX_LOGGER[] === nothing
        return f()
    end

    old_logger = global_logger()
    try
        global_logger(INDEX_LOGGER[])
        return f()
    finally
        global_logger(old_logger)
    end
end

"""
    check_and_notify_index_errors()

Check error count and notify user (once) if persistent indexing problems detected.
Shows warning after 5+ consecutive failures, resets on success.
"""
function check_and_notify_index_errors()
    if INDEX_ERROR_COUNT[] >= 5 && !INDEX_USER_NOTIFIED[]
        printstyled(
            "\n⚠️  Semantic search indexing is experiencing issues. Check ~/.cache/kaimon/indexer.log for details.\n",
            color=:yellow
        )
        INDEX_USER_NOTIFIED[] = true
    end
end

# ── Search Config (user preferences) ─────────────────────────────────────────
# User search configuration lives in ~/.config/kaimon/search.json so it
# survives cache clears.  Only regenerable index state stays in the cache
# at ~/.cache/kaimon/projects.json.

"""Path to the search config JSON file (user preferences)."""
_search_config_path() = joinpath(kaimon_config_dir(), "search.json")

"""Path to the index cache JSON file (regenerable state)."""
_project_registry_path() = joinpath(kaimon_cache_dir(), "projects.json")


"""
    load_search_config() -> Dict

Load search configuration from `~/.config/kaimon/search.json`.
On first call, migrates config fields from the old cache-only
`projects.json` if `search.json` doesn't exist yet.
"""
function load_search_config()
    path = _search_config_path()
    if isfile(path)
        try
            parsed = JSON.parse(read(path, String))
            if !haskey(parsed, "version")
                parsed["version"] = 1
            end
            if !haskey(parsed, "projects")
                parsed["projects"] = Dict{String,Any}()
            end
            return parsed
        catch e
            @warn "Failed to load search config, starting fresh" exception = e
            return Dict("version" => 1, "projects" => Dict{String,Any}())
        end
    end

    # No search.json yet — start fresh (projects get added as you use them)
    return Dict("version" => 1, "projects" => Dict{String,Any}())
end

"""
    save_search_config(config::AbstractDict)

Write search configuration to `~/.config/kaimon/search.json`.
"""
function save_search_config(config::AbstractDict)
    path = _search_config_path()
    try
        write(path, JSON.json(config, 2))
    catch e
        @error "Failed to save search config" exception = e
    end
end

# ── Per-collection embedding model (search.json `collection_models` map) ───────
# Vectors from different embedding models are incompatible — and NOT just at
# different dimensions: two 1024-dim models (e.g. qwen3-embedding:0.6b vs
# snowflake-arctic) live in different spaces, so a wrong-model query returns
# silent garbage with no error. So we record the model a collection was built
# with and always query with that same model.

"""Record the embedding model `collection` was indexed with."""
function set_collection_model!(collection::String, model::String)
    cfg = load_search_config()
    models = get!(() -> Dict{String,Any}(), cfg, "collection_models")
    models[collection] = model
    save_search_config(cfg)
    return nothing
end

"""The embedding model `collection` was indexed with, or `nothing` if unrecorded
(legacy collections predating model-stamping)."""
function get_collection_model(collection::String)
    models = get(load_search_config(), "collection_models", nothing)
    models isa AbstractDict || return nothing
    m = get(models, collection, nothing)
    m === nothing ? nothing : String(m)
end

"""The embedding model configured in search.json, else `DEFAULT_EMBEDDING_MODEL`.

This is what a caller gets when it asks to embed without naming a model. It is NOT
interchangeable with `resolve_search_model` — that answers "what was this collection
built with", which is a property of the stored vectors and must not follow config.
"""
configured_embedding_model() =
    String(get(load_search_config(), "embedding_model", DEFAULT_EMBEDDING_MODEL))

"""Resolve which embedding model to query `collection` with: the model it was
indexed with if recorded, else `DEFAULT_EMBEDDING_MODEL` (the long-standing
default that legacy/unstamped collections were almost certainly built with —
backwards-compatible, and avoids querying with a mismatched model)."""
resolve_search_model(collection::String) =
    something(get_collection_model(collection), DEFAULT_EMBEDDING_MODEL)

