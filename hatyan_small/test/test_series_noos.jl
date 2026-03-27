@testset "read_single_noos_file: VLISSGN_waterlevel_20180101_20180401.noos" begin
    filename = joinpath(TEST_DATA_DIR, "VLISSGN_waterlevel_20180101_20180401.noos")
    ts = read_single_noos_file(filename)

    @testset "metadata" begin
        @test get_names(ts)      == ["vlissingen"]
        @test get_source(ts)     == "observed"
        @test get_quantity(ts)   == "waterlevel"
        @test get_longitudes(ts) ≈  [3.597577]  atol=1e-6
        @test get_latitudes(ts)  ≈  [51.443861] atol=1e-6
    end

    @testset "time vector" begin
        @test get_times(ts)[1]   == DateTime(2018, 1, 1, 0, 0, 0)
        @test get_times(ts)[end] == DateTime(2018, 4, 1, 0, 0, 0)
        @test get_times(ts)[2] - get_times(ts)[1] == Minute(10)
        @test length(get_times(ts)) == 12752
    end

    @testset "values" begin
        vals = get_values(ts)
        @test size(vals) == (1, 12752)
        @test count(isnan, vals) == 0
        @test vals[1, 1]   ≈ 2.5f0
        @test vals[1, end] ≈ 1.05f0
        # physically plausible water levels at Vlissingen in metres
        @test minimum(vals) > -5.0
        @test maximum(vals) <  5.0
    end
end
