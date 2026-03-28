using Test
using Dates
using hatyan_core

const TEST_DATA_DIR = joinpath(@__DIR__, "..", "test_data")

include("test_constituents_donar.jl")
include("test_constituent_list.jl")
include("test_doodson_schureman.jl")
include("test_prediction.jl")
include("test_analysis.jl")
include("test_foreman.jl")
