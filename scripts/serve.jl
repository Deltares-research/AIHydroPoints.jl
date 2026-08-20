# serve.jl — CLI entry point; see src/serve.jl for the implementation.
#
# Usage:
#   MODEL_DIR=path/to/model_dir pixi run julia --project scripts/serve.jl

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
ENV["GKSwstype"] = "nul"   # allow plotting in headless environments
using AIHydroPoints

serve()
