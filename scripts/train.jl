# train.jl — CLI entry point; see src/train.jl for the implementation.
#
# Usage:
#   pixi run julia --project scripts/train.jl path/to/settings.toml [--continue|--overwrite]

flags   = filter(a -> startswith(a, "--"), ARGS)
posargs = filter(a -> !startswith(a, "--"), ARGS)

length(posargs) == 1 && length(flags) <= 1 ||
    error("Usage: julia scripts/train.jl <settings.toml> [--continue|--overwrite]\nGot ARGS = $(ARGS)")

on_existing_run = isempty(flags)          ? :error :
                  flags[1] == "--continue"  ? :continue :
                  flags[1] == "--overwrite" ? :overwrite :
                  error("Unknown flag $(flags[1]); expected --continue or --overwrite")

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

train(posargs[1]; on_existing_run)
