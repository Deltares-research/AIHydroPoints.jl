# train.jl — CLI entry point; see src/train.jl for the implementation.
#
# Usage:
#   pixi run julia --project scripts/train.jl path/to/settings.toml

length(ARGS) == 1 ||
    error("Usage: julia scripts/train.jl <settings.toml>\nGot ARGS = $(ARGS)")

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

train(ARGS[1])
