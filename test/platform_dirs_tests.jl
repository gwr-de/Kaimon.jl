using ReTest
using Kaimon

# Every location Kaimon uses, its own and the clients it registers with, is built from the base
# directories in Kaimon.jl, so writing a client's config and reading it back agree.
@testset "Platform directories" begin
    @testset "an empty variable counts as unset" begin
        withenv("XDG_CONFIG_HOME" => "", "XDG_CACHE_HOME" => "", "APPDATA" => "", "LOCALAPPDATA" => "") do
            @test Kaimon.xdg_config_home() == joinpath(homedir(), ".config")
            @test Kaimon.xdg_cache_home() == joinpath(homedir(), ".cache")
            @test Kaimon.windows_appdata() == joinpath(homedir(), "AppData", "Roaming")
            @test Kaimon.windows_localappdata() == joinpath(homedir(), "AppData", "Local")
        end
    end

    @testset "XDG_CONFIG_HOME reaches Kaimon's config and the clients'" begin
        d = mktempdir()
        try
            withenv("XDG_CONFIG_HOME" => d, "KAIMON_CONFIG_DIR" => nothing) do
                @test Kaimon.opencode_config_dir() == joinpath(d, "opencode")   # every platform
                if !Sys.iswindows() && !Sys.isapple()
                    @test Kaimon.vscode_user_dir() == joinpath(d, "Code", "User")
                    @test Kaimon.kaimon_config_dir() == joinpath(d, "kaimon")
                end
            end
        finally
            rm(d; recursive = true, force = true)
        end
        if Sys.isapple()
            @test Kaimon.vscode_user_dir() ==
                  joinpath(homedir(), "Library", "Application Support", "Code", "User")
        end
    end

    @testset "service subprocesses use the shared load-path joiner" begin
        lp = only(filter(startswith("JULIA_LOAD_PATH="), Kaimon.service_julia_cmd(`julia`).env))
        @test lp == "JULIA_LOAD_PATH=" * Kaimon._join_load_path("@", "@v#.#", "@stdlib")
    end
end
