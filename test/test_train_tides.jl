using Test
using AIHydroPoints
using Dates

@testset "tide model training pipeline" begin
    data_dir = joinpath(@__DIR__, "..", "test_data")
    train_file = joinpath(data_dir, "DCSM-FM_0_5nm_2008_3yr_5stations_his.jld2")
    test_file  = joinpath(data_dir, "DCSM-FM_0_5nm_2011_5stations_his.jld2")

    @test isfile(train_file)
    @test isfile(test_file)

    ts_train = JLD2TimeSeries(train_file, varname="waterlevel")
    ts_test  = JLD2TimeSeries(test_file,  varname="waterlevel")

    train_dict = Dict("waterlevel" => ts_train)
    test_dict  = Dict("waterlevel" => ts_test)

    nstations = length(get_names(ts_train))
    @test nstations > 0

    mktempdir() do model_dir
        settings = TideSettings(
            model_name = "test_tide_model",
            model_dir  = model_dir,
            use_gpu    = false,
            nstations  = nstations,
            model_pars = Dict(
                "nlayers_branch" => 1,
                "nhidden_branch" => 8,
                "nlayers_trunk"  => 0,
                "nhidden_trunk"  => 4,
                "nlayers_down"   => 1,
            )
        )
        train_settings = TrainingSettings(nepochs=2, nbatches=64, learning_rate=1.0e-3)

        model = TideModel(settings)
        @test !isnothing(model)

        model, acc_losses, train_losses, test_losses =
            train_model(model, settings, train_settings, train_dict, test_dict)

        @test length(train_losses) == train_settings.nepochs
        @test all(isfinite, train_losses)
        @test all(isfinite, test_losses)

        # Predict on test set
        prediction = predict(model, settings, ts_test)
        @test size(prediction, 1) == nstations
        @test size(prediction, 2) == length(get_times(ts_test))
        @test any(isfinite, prediction)
    end
end
