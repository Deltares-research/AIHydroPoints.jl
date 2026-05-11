# predict.jl — CLI entry point; see src/predict.jl for the implementation.
#
# Usage:
#   pixi run julia --project scripts/predict.jl path/to/settings.toml

length(ARGS) == 1 ||
    error("Usage: julia scripts/predict.jl <settings.toml>\nGot ARGS = $(ARGS)")

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

predict(ARGS[1])
