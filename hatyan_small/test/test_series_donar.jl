@testset "read_donar_timeseries: VLISSGN_obs19.txt" begin
    filename = joinpath(TEST_DATA_DIR, "VLISSGN_obs19.txt")
    ts = read_donar_timeseries(filename)

    @testset "metadata" begin
        @test get_names(ts) == ["Vlissingen"]
        @test get_source(ts) == "VLISSGN"
        @test occursin("WATHTE", get_quantity(ts))
        @test occursin("NAP", get_quantity(ts))
    end

    @testset "time vector" begin
        @test get_times(ts)[1]   == DateTime(1976, 1, 1, 0, 0, 0)
        @test get_times(ts)[end] == DateTime(1994, 12, 31, 23, 0, 0)
        @test get_times(ts)[2] - get_times(ts)[1] == Minute(60)
        @test length(get_times(ts)) == 166560
    end

    @testset "values" begin
        vals = get_values(ts)
        @test size(vals) == (1, 166560)
        @test count(isnan, vals) == 0
        # physically plausible water levels at Vlissingen in metres relative to NAP
        @test minimum(vals) > -5.0
        @test maximum(vals) <  5.0
    end
end
