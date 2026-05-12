using Test
using AIHydroPoints

# Paths relative to this file
examples_dir = joinpath(@__DIR__, "..", "examples")
train_toml   = joinpath(examples_dir, "LinearSurgeModel.toml")
predict_toml = joinpath(examples_dir, "predict_LinearSurgeModel.toml")
model_dir    = joinpath(examples_dir, "training_output", "example_LinearSurgeModel")

@testset "train/predict pipeline (LinearSurgeModel)" begin

    @testset "train" begin
        train(train_toml)

        @test isdir(model_dir)
        @test isfile(joinpath(model_dir, "params.jld2"))
        @test isfile(joinpath(model_dir, "model_settings.toml"))
        @test isfile(joinpath(model_dir, "run_settings.toml"))
        @test isfile(joinpath(model_dir, "losses.png"))

        # output_settings defaults are written into run_settings.toml
        saved = toml_read(joinpath(model_dir, "run_settings.toml"))
        @test haskey(saved, "output_settings")
        outputs = saved["output_settings"]["outputs"]
        @test length(outputs) == 1
        @test outputs[1]["split"] == "testing"
        @test outputs[1]["timeseries"] == true

        # timeseries subfolder is created with one PNG per station
        ts_dir = joinpath(model_dir, "testing_timeseries")
        @test isdir(ts_dir)
        @test !isempty(readdir(ts_dir))
    end

    @testset "predict" begin
        predict(predict_toml)
    end

end
