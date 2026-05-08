using Test
using AIHydroPoints
using Dates

const DATA_DIR = joinpath(@__DIR__, "..", "test_data")

@testset "load_data" begin

    @testset "netcdf + jld2 (surge splits)" begin
        data_settings = Dict{String,Any}(
            "files" => [
                Dict("path"      => joinpath(DATA_DIR, "surge_schureman_2011.nc"),
                     "format"    => "netcdf",
                     "split"     => "training",
                     "variables" => ["surge"]),
                Dict("path"      => joinpath(DATA_DIR, "era5_wind_stress_2011_testing.jld2"),
                     "format"    => "jld2",
                     "split"     => "training",
                     "variables" => ["stress_x", "stress_y", "pressure"]),
                Dict("path"      => joinpath(DATA_DIR, "surge_schureman_2012.nc"),
                     "format"    => "netcdf",
                     "split"     => "testing",
                     "variables" => ["surge"]),
                Dict("path"      => joinpath(DATA_DIR, "era5_wind_stress_2012_validation.jld2"),
                     "format"    => "jld2",
                     "split"     => "testing",
                     "variables" => ["stress_x", "stress_y", "pressure"]),
            ],
            "model_io" => Dict("input" => ["stress_x", "stress_y", "pressure"],
                               "target" => ["surge"]),
        )

        data = load_data(data_settings)

        @test haskey(data, "training")
        @test haskey(data, "testing")

        for split in ("training", "testing")
            d = data[split]
            @test haskey(d.input, "stress_x")
            @test haskey(d.input, "stress_y")
            @test haskey(d.input, "pressure")
            @test haskey(d.target, "surge")

            # All series in a split share the same time axis (intersect applied)
            ntimes_input  = size(get_values(d.input["stress_x"]), 2)
            ntimes_target = size(get_values(d.target["surge"]), 2)
            @test ntimes_input == ntimes_target
            @test ntimes_input > 0

            t_in  = get_times(d.input["stress_x"])
            t_out = get_times(d.target["surge"])
            @test t_in[1]   == t_out[1]
            @test t_in[end] == t_out[end]
        end
    end

    @testset "location filtering" begin
        data_settings = Dict{String,Any}(
            "files" => [
                Dict("path"      => joinpath(DATA_DIR, "surge_schureman_2011.nc"),
                     "format"    => "netcdf",
                     "split"     => "training",
                     "variables" => ["surge"],
                     "locations" => ["VLISSGN"]),
            ],
            "model_io" => Dict("input" => String[], "target" => ["surge"]),
        )

        data = load_data(data_settings)
        ts = data["training"].target["surge"]
        @test length(get_names(ts)) == 1
        @test get_names(ts)[1] == "VLISSGN"
    end

    @testset "variable aliasing" begin
        data_settings = Dict{String,Any}(
            "files" => [
                Dict("path"      => joinpath(DATA_DIR, "era5_wind_stress_2011_testing.jld2"),
                     "format"    => "jld2",
                     "split"     => "training",
                     "variables" => [Dict("name" => "stress_x", "as" => "wind_x")]),
            ],
            "model_io" => Dict("input" => ["wind_x"], "target" => String[]),
        )

        data = load_data(data_settings)
        @test haskey(data["training"].input, "wind_x")
        @test !haskey(data["training"].input, "stress_x")
    end

    @testset "timerange split from single file" begin
        data_settings = Dict{String,Any}(
            "files" => [
                Dict("path"      => joinpath(DATA_DIR, "surge_schureman_2011.nc"),
                     "format"    => "netcdf",
                     "split"     => "first_half",
                     "timerange" => ["2011-01-01", "2011-06-30"],
                     "variables" => ["surge"]),
                Dict("path"      => joinpath(DATA_DIR, "surge_schureman_2011.nc"),
                     "format"    => "netcdf",
                     "split"     => "second_half",
                     "timerange" => ["2011-07-01", "2011-12-31"],
                     "variables" => ["surge"]),
            ],
            "model_io" => Dict("input" => String[], "target" => ["surge"]),
        )

        data = load_data(data_settings)
        t1 = get_times(data["first_half"].target["surge"])
        t2 = get_times(data["second_half"].target["surge"])
        @test t1[end] < t2[1]
        @test t1[1]  >= DateTime(2011, 1, 1)
        @test t2[end] <= DateTime(2011, 12, 31, 23, 59, 59)
    end

    @testset "noos collection (wave splits)" begin
        wave_dir = joinpath(DATA_DIR, "waves_2021")

        data_settings = Dict{String,Any}(
            "files" => [
                Dict("path"      => wave_dir,
                     "format"    => "noos",
                     "source"    => "knmi_harmonie40_wind",
                     "split"     => "training",
                     "timerange" => ["2021-01-01", "2021-09-30T23:00:00"],
                     "variables" => ["wind_speed"]),
                Dict("path"      => wave_dir,
                     "format"    => "noos",
                     "source"    => "swan_dcsm_harmonie",
                     "split"     => "training",
                     "timerange" => ["2021-01-01", "2021-09-30T23:00:00"],
                     "variables" => ["wave_height"]),
                Dict("path"      => wave_dir,
                     "format"    => "noos",
                     "source"    => "knmi_harmonie40_wind",
                     "split"     => "testing",
                     "timerange" => ["2021-10-01", "2021-12-31T23:00:00"],
                     "variables" => ["wind_speed"]),
                Dict("path"      => wave_dir,
                     "format"    => "noos",
                     "source"    => "swan_dcsm_harmonie",
                     "split"     => "testing",
                     "timerange" => ["2021-10-01", "2021-12-31T23:00:00"],
                     "variables" => ["wave_height"]),
            ],
            "model_io" => Dict("input"  => ["wind_speed"],
                               "target" => ["wave_height"]),
        )

        data = load_data(data_settings)

        for split in ("training", "testing")
            d = data[split]
            @test haskey(d.input,  "wind_speed")
            @test haskey(d.target, "wave_height")

            nstations_wind = size(get_values(d.input["wind_speed"]), 1)
            nstations_wave = size(get_values(d.target["wave_height"]), 1)
            @test nstations_wind == 3
            @test nstations_wave == 3

            t_in  = get_times(d.input["wind_speed"])
            t_out = get_times(d.target["wave_height"])
            @test t_in[1]   == t_out[1]
            @test t_in[end] == t_out[end]
        end

        # Training window ends before testing window starts
        @test get_times(data["training"].input["wind_speed"])[end] <
              get_times(data["testing"].input["wind_speed"])[1]
    end

end
