using Test
using AIHydroPoints
using Dates

@testset "tide model training pipeline" begin
    data_dir   = joinpath(@__DIR__, "..", "test_data")
    train_file = joinpath(data_dir, "tides_schureman_2011.nc")
    test_file  = joinpath(data_dir, "tides_schureman_2012.nc")

    @test isfile(train_file)
    @test isfile(test_file)

    ts_train = TimeSeries(NetCDFTimeSeries(train_file, "waterlevel"))
    ts_test  = TimeSeries(NetCDFTimeSeries(test_file,  "waterlevel"))

    train_dict = Dict("waterlevel" => ts_train)
    test_dict  = Dict("waterlevel" => ts_test)

    nstations = length(get_names(ts_train))
    @test nstations > 0

    model_dir = joinpath(temp_dir, "test_tide_model")
    mkpath(model_dir)

    settings = Dict{String, Any}(
        "model_name" => "test_tide_model",
        "model_dir"  => model_dir,
        "freqs"      => ["M2", "S2", "K1"],
        "model_pars" => Dict{String, Any}(
            "nlayers_branch" => 1,
            "nhidden_branch" => 8,
            "nlayers_trunk"  => 0,
            "nhidden_trunk"  => 4,
            "nlayers_down"   => 1,
        ),
    )
    train_settings = TrainingSettings(nepochs=2, batch_size=64, learning_rate=1.0e-3)

    model = DeepONetTideModel(settings)
    @test model isa DeepONetTideModel
    @test model isa AbstractTideModel

    train_losses, val_losses = train_model!(model, train_settings, train_dict, train_dict)

    @test length(train_losses) == train_settings.nepochs
    @test all(isfinite, train_losses)
    @test isempty(val_losses)

    output = predict(model, test_dict)
    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "waterlevel")
    @test size(output["waterlevel"].values, 1) == nstations
    @test any(isfinite, output["waterlevel"].values)
end
