# test_series_jld2.jl
# Test JLD2 import and export functionality
using Test
using Dates

function test_reading_jld2_timeseries()
    filename = joinpath(pwd(), "..","test_data", "estuary_his.jld2")
    series = JLD2TimeSeries(filename)
    
    @test typeof(series) == TimeSeries

    # test if data was read correctly
    values = get_values(series)
    @test size(values) == (3, 34561) # 3 locations, 34561 time steps
    @test values isa Matrix
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
    @test source=="my-source"
end

function test_writing_jld2_timeseries()
    filename = joinpath(temp_dir, "test_write.jld2")
    # Check if the file already exists and remove it
    if isfile(filename)
        rm(filename)
    end
    # Create a dummy TimeSeries object
    values = rand(Float32, 3, 10)
    times = DateTime(2021, 1, 1):Dates.Hour(1):DateTime(2021, 1, 10)
    names = ["station01", "station02", "station03"]
    longitudes = [5.0, 6.0, 7.0]
    latitudes = [50.0, 51.0, 52.0]
    quantity = "waterlevel"
    source = "test-source"
    
    ts = TimeSeries(values, times, names, longitudes, latitudes, quantity, source)
    
    write_to_jld2(ts, filename)
    
    # Read back the file
    ts_read = JLD2TimeSeries(filename)
    
    @test get_values(ts_read) == get_values(ts)
    @test get_times(ts_read) == get_times(ts)
    @test get_names(ts_read) == get_names(ts)
    @test get_longitudes(ts_read) == get_longitudes(ts)
    @test get_latitudes(ts_read) == get_latitudes(ts)
    @test get_quantity(ts_read) == get_quantity(ts)
    @test get_source(ts_read) == get_source(ts)

end 


# Run the tests
test_reading_jld2_timeseries()
test_writing_jld2_timeseries()