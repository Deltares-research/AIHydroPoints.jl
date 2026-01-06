# test_series.jl

using Dates

function test_basics() #basics
    times1 = collect(DateTime(2020, 1, 1):Minute(60):DateTime(2020, 1, 3))
    longitudes1 = [4.0, 5.0, 6.0]
    latitudes1 = [52.0, 53.0, 54.0]
    names1 = ["Station A", "Station B", "Station C"]
    values1 = randn(length(longitudes1),length(times1)) # random values for testing
    quantity1 = "water level"
    source1 = "Test Source"
    series1 = TimeSeries(values1, times1, names1, longitudes1, latitudes1, quantity1, source1)
    @test size(get_values(series1)) == (3, length(times1))
    @test length(get_times(series1)) == length(times1)
    @test length(get_names(series1)) == 3
    @test get_quantity(series1) == "water level"
    @test get_source(series1) == "Test Source"
    @test get_longitudes(series1) == longitudes1
    @test get_latitudes(series1) == latitudes1
    @test get_values(series1)[1,1] ≈ values1[1,1]

   # Test selecting a single location
    series_loc_from_index = select_location_by_id(series1, 1)
    series_loc_from_name = select_location_by_name(series1, "Station A")
    @test series_loc_from_index.names == series_loc_from_name.names

    # Test selecting multiple locations
    series_multi_from_index = select_locations_by_ids(series1, [1, 2])
    series_multi_from_names = select_locations_by_names(series1, ["Station A", "Station B"])
    @test get_names(series_multi_from_index) == get_names(series_multi_from_names)
    @test size(get_values(series_multi_from_index)) == (2, length(times1))
    @test size(get_values(series_multi_from_names)) == (2, length(times1))

    # Test selection by start and end dates
    start_date = DateTime(2020, 1, 1, 12, 0, 0)
    end_date = DateTime(2020, 1, 2, 12, 0, 0)
    series_timespan = select_timespan(series1, start_date, end_date)
    @test length(get_times(series_timespan)) == 25 # 25 hours in the range
    @test size(get_values(series_timespan)) == (3, 25)

    # Test showing the series
    io = IOBuffer()
    show(io, series1)
    output_str = String(take!(io))
    @test occursin("Station A", output_str)
    @test occursin("Number of locations: 3", output_str)
end

function test_merge_times()
    # create a time series with two locations and hourly data
    times1 = collect(DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 2))
    longitudes1 = [4.0, 5.0]
    latitudes1 = [52.0, 53.0]
    names1 = ["Station A", "Station B"]
    values1 = randn(length(longitudes1),length(times1)) # random values for testing
    quantity1 = "water level"
    source1 = "Test Source"
    # create timeseries for times 1:12 and 13:end
    series1 = TimeSeries(values1[:,1:12], times1[1:12], names1, longitudes1, latitudes1, quantity1, source1)
    series2 = TimeSeries(values1[:,13:end], times1[13:end], names1, longitudes1, latitudes1, quantity1, source1)
    # merge the two series
    merged_series = merge_by_times(series1, series2)
    # now check the merged series
    @test size(get_values(merged_series)) == (2, length(times1))
    @test length(get_times(merged_series)) == length(times1)
    @test get_names(merged_series) == names1
    @test get_longitudes(merged_series) == longitudes1
    @test get_latitudes(merged_series) == latitudes1
    @test get_quantity(merged_series) == quantity1
    @test get_source(merged_series) == source1
    @test get_values(merged_series)[1,1] ≈ values1[1,1]
    @test get_values(merged_series)[2,13] ≈ values1[2,13]
end

function test_select_timerange_with_fill()
    times1 = collect(DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 2))
    longitudes1 = [4.0, 5.0]
    latitudes1 = [52.0, 53.0]
    names1 = ["Station A", "Station B"]
    values1 = randn(length(longitudes1),length(times1)) # random values for testing
    quantity1 = "water level"
    source1 = "Test Source"
    series1 = TimeSeries(values1, times1, names1, longitudes1, latitudes1, quantity1, source1)
    # select a time range with gaps
    time_range = DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 3)
    filled_series = select_timerange_with_fill(series1, time_range; fill_value=-999.0)
    @test length(get_times(filled_series)) == length(time_range)
    @test size(get_values(filled_series)) == (2, length(time_range))
    # Check that the original values are in the correct positions
    for (i, t) in enumerate(get_times(series1))
        idx = findfirst(x -> x == t, get_times(filled_series))
        @test idx !== nothing
        @test get_values(filled_series)[1, idx] ≈ get_values(series1)[1, i]
        @test get_values(filled_series)[2, idx] ≈ get_values(series1)[2, i]
    end
    # Check that the fill values are correctly placed
    for (i, t) in enumerate(get_times(filled_series))
        if t < minimum(get_times(series1)) || t > maximum(get_times(series1))
            @test get_values(filled_series)[1, i] == -999.0
            @test get_values(filled_series)[2, i] == -999.0
        end
    end
end

function test_merge_locations()
    # create two time series with each a location and hourly data
    times1 = collect(DateTime(2020, 1, 1):Hour(1):DateTime(2020, 1, 2))
    longitudes1 = [4.0]
    latitudes1 = [52.0]
    names1 = ["Station A"]
    values1 = randn(length(longitudes1),length(times1)) # random values for testing
    quantity = "water_level"
    source = "test_source"
    series1 = TimeSeries(values1, times1, names1, longitudes1, latitudes1, quantity, source)
    names2 = ["Station B"]
    longitudes2 = [5.0]
    latitudes2 = [53.0]
    values2 = randn(length(longitudes2),length(times1)) # random values for testing
    series2 = TimeSeries(values2, times1, names2, longitudes2, latitudes2, quantity, source)
    # merge the two series
    merged_series = merge_by_locations([series1, series2])
    # now check the merged series
    @test size(get_values(merged_series)) == (2, length(times1))
    @test length(get_times(merged_series)) == length(times1)
    @test get_names(merged_series) == ["Station A", "Station B"]
    @test get_longitudes(merged_series) == [4.0, 5.0]
    @test get_latitudes(merged_series) == [52.0, 53.0]
    @test get_quantity(merged_series) == quantity
    @test get_source(merged_series) == source
    @test get_values(merged_series)[1,1] ≈ values1[1,1]
    @test get_values(merged_series)[2,1] ≈ values2[1,1] 
end
# Run all the working tests
test_basics()
test_merge_times()
test_select_timerange_with_fill()
test_merge_locations()
