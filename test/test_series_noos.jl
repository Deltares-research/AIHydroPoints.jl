# test_series_noos.jl

using Dates

const test_data_dir = joinpath(@__DIR__, "..", "test_data")

function test_noos_single_series_readwrite() #basics: create a noos file and read it back
    times1 = collect(DateTime(2020, 1, 1):Minute(60):DateTime(2020, 1, 3))
    longitudes1 = [4.0, 5.0, 6.0]
    latitudes1 = [52.0, 53.0, 54.0]
    names1 = ["Station A", "Station B", "Station C"]
    values1 = randn(length(longitudes1),length(times1)) # random values for testing
    quantity1 = "water level"
    source1 = "Test Source"
    series1 = TimeSeries(values1, times1, names1, longitudes1, latitudes1, quantity1, source1)
    # write to noos file
    filename1 = joinpath(temp_dir, "test_series_noos1.txt")
    write_single_noos_file(filename1, series1,1)
    filename2 = joinpath(temp_dir, "test_series_noos2.txt")
    write_single_noos_file(filename2, series1,2)
    # read back
    series1_read1 = read_single_noos_file(filename1)
    @test size(get_values(series1_read1)) == (1, length(times1))
    @test length(get_times(series1_read1)) == length(times1)
    @test length(get_names(series1_read1)) == 1
    @test get_quantity(series1_read1) == "water level"
    @test get_source(series1_read1) == "Test Source"
    @test get_longitudes(series1_read1) == longitudes1[1:1]
    @test get_latitudes(series1_read1) == latitudes1[1:1]
    @test isapprox(get_values(series1_read1)[1,1], values1[1,1], atol=1e-4)

    # read a file produced by the matroos database
    filename3 = joinpath(test_data_dir, "waves_20240101_20240102","20240101","wave_height__swan_dcsm_harmonie__K13a.noos")
    series3_read = read_single_noos_file(filename3)
    # times from 202312101000 to 202401020000 hourly
    ntimes = length(collect(DateTime(2024,1,1):Hour(1):DateTime(2024,1,2,0)))
    @test size(get_values(series3_read)) == (1, ntimes)
    @test length(get_times(series3_read)) == ntimes
    @test length(get_names(series3_read)) == 1
    @test get_quantity(series3_read) == "wave_height"
    @test get_source(series3_read) == "swan_dcsm_harmonie"
    @test get_names(series3_read) == ["K13a"]
    @test get_longitudes(series3_read) ≈ [3.219036]
    @test get_latitudes(series3_read) ≈ [53.218117]
    @test get_values(series3_read)[1,1] ≈ 2.155f0
    @test get_values(series3_read)[1,end] ≈ 1.88700f0
end

function test_noos_read()
    # read all files in a directory and its subdirectories
    input_dir = joinpath(test_data_dir, "waves_20240101_20240102")
    collection=NoosTimeSeriesCollection(input_dir)
    # check the contents
    sources=get_sources(collection)
    @show sources
    @test length(sources) == 2
    @test "knmi_harmonie40_wind" in sources
    @test "swan_dcsm_harmonie" in sources
    quantities=get_quantities(collection)
    @show quantities
    @test length(quantities) == 3
    @test "wind_speed" in quantities
    @test "wind_direction" in quantities
    @test "wave_height" in quantities
    keys=get_source_quantity_keys(collection)
    # @show keys
    @test length(keys) == 3
    @test ("knmi_harmonie40_wind","wind_speed") in keys
    @test ("knmi_harmonie40_wind","wind_direction") in keys
    @test ("swan_dcsm_harmonie","wave_height") in keys
    # get a series from the collection
    swh_series=get_series_from_collection(collection,"swan_dcsm_harmonie","wave_height")
    # @show swh_series
    locations=get_names(swh_series)
    @test length(locations) == 2
    @test "K13a" in locations
    @test "Europlatform" in locations
    times=get_times(swh_series)
    @test length(times) == 49 # from 20240101 00:00 to 20240103 00:00 hourly
    @test times[1] == DateTime(2024,1,1,0,0)
    @test times[2] == DateTime(2024,1,1,1,0)
    @test times[end] == DateTime(2024,1,3,0,0)
    values=get_values(swh_series)
    @test size(values) == (length(locations), length(times))
end

test_noos_single_series_readwrite()

test_noos_read()