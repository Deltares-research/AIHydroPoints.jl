# test_series_zarr.jl

using Rasters
using ZarrDatasets
using Dates
using HTTP

function test_zarr_timeseries()
    filename = joinpath(pwd(), "..","test_data", "estuary_his.zarr")
    series = ZarrTimeSeries(filename,"waterlevel")

    @test typeof(series) == ZarrTimeSeries

    # test getters
    values = get_values(series)
    @test size(values) == (3, 34561) # 3 locations, 34561 time steps
    @test typeof(values) == Array{Float32, 2}
    times = get_times(series)
    @test length(times) == 34561 # 10 time steps
    @test times[1] == DateTime(1991, 1, 1) # check first time step
    @test times[end] == DateTime(1991, 8, 29)
    names = get_names(series)
    @test length(names) == 3 # 3 locations
    @test names == ["station01", "station02", "station03"]
    longitudes = get_longitudes(series)
    @test length(longitudes) == 3 # 3 locations
    latitudes = get_latitudes(series)
    quantity = get_quantity(series)
    source = get_source(series)
    @test source=="Zarr file: $(filename)"

    # test selection by location index
    selected_series = select_location_by_id(series, 1)
    @test get_names(selected_series) == ["station01"]
    @test size(get_values(selected_series)) == (1, 34561) # 1 location, 34561 time steps
    @test get_times(selected_series) == times
    @test get_longitudes(selected_series) == [longitudes[1]]
    @test get_latitudes(selected_series) == [latitudes[1]]
    @test get_quantity(selected_series) == quantity
    @test get_source(selected_series) == source
    # test selection by multiple location indices
    selected_series_multi = select_locations_by_ids(series, [1, 2])
    @test get_names(selected_series_multi) == ["station01", "station02"]
    @test size(get_values(selected_series_multi)) == (2, 34561) # 2 locations, 34561 time steps
    @test get_times(selected_series_multi) == times
    @test get_longitudes(selected_series_multi) == longitudes[1:2]
    @test get_latitudes(selected_series_multi) == latitudes[1:2]

    # test selection by names
    selected_series_names = select_locations_by_names(series, ["station01", "station03"])
    @test get_names(selected_series_names) == ["station01", "station03"]
    @test size(get_values(selected_series_names)) == (2, 34561) # 2 locations, 34561 time steps
    @test get_times(selected_series_names) == times
    @test get_longitudes(selected_series_names) == longitudes[[1,3]]
    @test get_latitudes(selected_series_names) == latitudes[[1,3]]
    @test get_quantity(selected_series_names) == quantity
    @test get_source(selected_series_names) == source
    # test select single location by name
    selected_single = select_location_by_name(series, "station02")
    @test get_names(selected_single) == ["station02"]
    @test size(get_values(selected_single)) == (1, 34561) # 1 location, 34561 time steps
    @test get_times(selected_single) == times
    @test get_longitudes(selected_single) == [longitudes[2]]
    @test get_latitudes(selected_single) == [latitudes[2]]

    # test selection by timespan
    start_date = DateTime(1991, 1, 1, 12, 0, 0)
    end_date = DateTime(1991, 1, 2, 12, 0, 0)
    series_timespan = select_timespan(series, start_date, end_date)
    @test length(get_times(series_timespan)) == 145 # 145 hours in the range
    selected_times = get_times(series_timespan)
    @test selected_times[1] == DateTime(1991, 1, 1, 12, 0, 0) # check first time step in timespan
    @test selected_times[end] == DateTime(1991, 1, 2, 12, 0, 0) # check last time step in timespan
    @test size(get_values(series_timespan)) == (3, 145) # 3 locations, 145 time steps
    @test get_names(series_timespan) == names
    @test get_longitudes(series_timespan) == longitudes
    @test get_latitudes(series_timespan) == latitudes
    @test get_quantity(series_timespan) == quantity
    @test get_source(series_timespan) == source

    # Test showing the series
    io = IOBuffer()
    show(io, series)
    output_str = String(take!(io))
    @test occursin("station01", output_str)
    @test occursin("Number of locations: 3", output_str)
    @test occursin("ZarrTimeSeries", output_str)
end

function test_zarr_timeseries_over_https()
    urlname = "https://nx7384.your-storageshare.de/apps/sharingpath/wetwin/public/DCSM-FM_0_5nm_0000_his.zarr"
    series = ZarrTimeSeries(urlname, "waterlevel")

    @test typeof(series) == ZarrTimeSeries

    # short subset of the tests
    times = get_times(series)
    @test length(times) ==  315649# number of time steps
    @test times[1] == DateTime(2012, 1, 1) # check first time step
    @test times[end] == DateTime(2018, 1, 1)
    names = get_names(series)
    @test length(names) == 378 # number of locations
    @test names[1:3] == ["A12", "A2", "ABDN"]

    finalize(series) # close the Zarr dataset
end

function test_zarr_timeseries_over_s3()
    urlname = "s3://s3.deltares.nl/emodnet/DCSM-FM_0_5nm_1980-2023_his.zarr?profile=minio_deltares"
    series = ZarrTimeSeries(urlname, "waterlevel")
    try
        @test typeof(series) == ZarrTimeSeries

        # short subset of the tests
        times = get_times(series)
        @test length(times) ==  2319985# number of time steps
        @test times[1] == DateTime(1979, 12, 22) # check first time step
        @test times[end] == DateTime(2024, 1, 31)
        names = get_names(series)
        @test length(names) == 317 # number of locations
        @test names[1:3] == ["A12", "A2", "ABDN"]
    finally
        finalize(series)
        # Give background tasks time to finish
        sleep(2)
    end
end

function test_zarr_timeseries_over_s3_scaleway()
    urlname = "s3://s3.nl-ams.scw.cloud/ai-hydro/DCSM-FM_0_5nm_1980-2023_his.zarr?profile=minio_scaleway"
    series = ZarrTimeSeries(urlname, "waterlevel")
    try

        @test typeof(series) == ZarrTimeSeries

        # short subset of the tests
        times = get_times(series)
        @test length(times) ==  2319985# number of time steps
        @test times[1] == DateTime(1979, 12, 22) # check first time step
        @test times[end] == DateTime(2024, 1, 31)
        names = get_names(series)
        @test length(names) == 317 # number of locations
        @test names[1:3] == ["A12", "A2", "ABDN"]
    finally
        finalize(series) # close the Zarr dataset
        sleep(2)
    end
end


function check_server(server)
    try
        response = HTTP.get(server; connect_timeout=2,readtimeout=2)
        return response.status == 200
    catch e
        println("Failed to connect to $server: $e")
        return false
    end
end

# run tests
test_zarr_timeseries()

if check_server("https://nx7384.your-storageshare.de") # points to login for the account
    test_zarr_timeseries_over_https()
else
    println("Skipping test_zarr_timeseries_over_https: Server not reachable.")
end

#
# NOTE: The S3 tests may give an error when run in multiple threads. We can usually ignore this.
# If you want to check in sequential mode, then you can run: `julia --threads 1 --project "using Pkg; Pkg.test()"`
#

if check_server("https://s3.deltares.nl/")
    try # TODO somehow zarr tests can give an error within a unit test, so we catch it here
        test_zarr_timeseries_over_s3()
    catch e
        println("Error during test_zarr_timeseries_over_s3: $e")
    end
else
    println("Skipping test_zarr_timeseries_over_s3: Deltares S3 server not reachable.")
end

if check_server("http://s3.nl-ams.scw.cloud/") && AIHydroPoints.has_aws_credentials() # Points to Scaleway S3 server which responds with a simple text
    try # TODO somehow zarr tests can give an error within a unit test, so we catch it here
        test_zarr_timeseries_over_s3_scaleway()
    catch e
        println("Error during test_zarr_timeseries_over_s3_scaleway: $e")
    end
    test_zarr_timeseries_over_s3_scaleway()
else 
    println("Skipping test_zarr_timeseries_over_s3_scaleway: Scaleway S3 server not reachable.")
end
