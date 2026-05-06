using Test
using AIHydroPoints
using Dates

@testset "surge model training pipeline" begin
    data_dir = joinpath(@__DIR__, "..", "test_data")
    train_surge_file = joinpath(data_dir, "surge_schureman_2011.nc")
    train_wind_file  = joinpath(data_dir, "era5_wind_stress_2011_testing.jld2")
    test_surge_file  = joinpath(data_dir, "surge_schureman_2012.nc")
    test_wind_file   = joinpath(data_dir, "era5_wind_stress_2012_validation.jld2")

    @test isfile(train_surge_file)
    @test isfile(train_wind_file)
    @test isfile(test_surge_file)
    @test isfile(test_wind_file)

    function load_dict(surge_file, wind_file)
        ts_h      = NetCDFTimeSeries(surge_file, "surge")
        ts_wind_x = JLD2TimeSeries(wind_file, varname="stress_x")
        ts_wind_y = JLD2TimeSeries(wind_file, varname="stress_y")
        ts_press  = JLD2TimeSeries(wind_file, varname="pressure")
        t_start = max(get_times(ts_h)[1],   get_times(ts_wind_x)[1])
        t_end   = min(get_times(ts_h)[end], get_times(ts_wind_x)[end])
        return Dict{String, TimeSeries}(
            "waterlevel" => select_timespan(ts_h,      t_start, t_end),
            "stress_x"   => select_timespan(ts_wind_x, t_start, t_end),
            "stress_y"   => select_timespan(ts_wind_y, t_start, t_end),
            "pressure"   => select_timespan(ts_press,  t_start, t_end),
        )
    end

    train_dict = load_dict(train_surge_file, train_wind_file)
    test_dict  = load_dict(test_surge_file,  test_wind_file)

    nstations = length(get_names(train_dict["waterlevel"]))
    nwind     = length(get_names(train_dict["stress_x"]))
    @test nstations > 0
    @test nwind > 0

    train_input  = Dict{String, TimeSeries}(k => train_dict[k] for k in ("stress_x","stress_y","pressure"))
    train_target = Dict{String, TimeSeries}("surge" => train_dict["waterlevel"])
    test_input   = Dict{String, TimeSeries}(k => test_dict[k]  for k in ("stress_x","stress_y","pressure"))
    test_target  = Dict{String, TimeSeries}("surge" => test_dict["waterlevel"])

    mktempdir() do model_dir
        model_settings = Dict{String, Any}(
            "model_name" => "test_surge_model",
            "model_dir"  => model_dir,
            "nstations"  => nstations,
            "nwind"      => nwind,
            "nlags"      => 4,
        )
        train_settings = TrainingSettings(nepochs=2, nbatches=64, learning_rate=1e-3)

        model = LinearSurgeModel(model_settings)
        @test model isa AbstractSurgeModel

        train_losses, val_losses = train_model!(model, train_settings, train_input, train_target)

        @test length(train_losses) == train_settings.nepochs
        @test all(isfinite, train_losses)
        @test isempty(val_losses)

        # predict produces the correct output shape
        output = predict(model, test_input)
        @test haskey(output, "surge")
        nvalid = length(get_times(test_dict["stress_x"])) - 4 + 1
        @test size(output["surge"].values) == (nstations, nvalid)
    end
end
