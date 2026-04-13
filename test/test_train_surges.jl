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
        # Align time ranges (surge and ERA5 may differ by one step)
        t_start = max(get_times(ts_h)[1],   get_times(ts_wind_x)[1])
        t_end   = min(get_times(ts_h)[end], get_times(ts_wind_x)[end])
        return Dict(
            "waterlevel" => select_timespan(ts_h,      t_start, t_end),
            "wind_x"     => select_timespan(ts_wind_x, t_start, t_end),
            "wind_y"     => select_timespan(ts_wind_y, t_start, t_end),
            "pressure"   => select_timespan(ts_press,  t_start, t_end),
        )
    end

    train_dict = load_dict(train_surge_file, train_wind_file)
    test_dict  = load_dict(test_surge_file,  test_wind_file)

    nstations = length(get_names(train_dict["waterlevel"]))
    nwind     = length(get_names(train_dict["wind_x"]))
    @test nstations > 0
    @test nwind > 0

    lats_in  = get_latitudes(train_dict["wind_x"])
    lons_in  = get_longitudes(train_dict["wind_x"])
    lats_out = get_latitudes(train_dict["waterlevel"])
    lons_out = get_longitudes(train_dict["waterlevel"])
    gn = GraphNetwork(collect(zip(lats_in, lons_in)), collect(zip(lats_out, lons_out)),
                      max_distance=1e5)

    mktempdir() do model_dir
        settings = SurgeSettings(
            model_name = "test_surge_model",
            model_dir  = model_dir,
            use_gpu    = false,
            nstations  = nstations,
            nwind      = nwind,
            nlags      = 4,
            model_pars = Dict(
                "theta"          => 10000.0,
                "nheads"         => 2,
                "nlayers_branch" => 1,
                "nlayers_trunk"  => 0,
                "nhidden_trunk"  => 8,
                "nembed"         => 8,
            ),
        )
        train_settings = TrainingSettings(nepochs=2, nbatches=64, learning_rate=1.0e-3)

        model = SurgeModel(gn, settings)
        @test !isnothing(model)

        model, acc_losses, train_losses, test_losses =
            train_model(model, settings, train_settings, train_dict, test_dict)

        @test length(train_losses) == train_settings.nepochs
        @test all(isfinite, train_losses)
        @test all(isfinite, test_losses)
    end
end
