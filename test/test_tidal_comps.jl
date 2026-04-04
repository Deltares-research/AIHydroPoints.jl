# test_tidal_comps.jl

using Dates

@testset "tidal_comps.jl" begin

    @testset "robust_timedelta_sec" begin
        # Reference date itself should give 0 seconds
        refdate = DateTime(1900, 1, 1)
        @test robust_timedelta_sec(refdate) ≈ 0.0 atol=1e-6

        # One day after reference = 86400 seconds
        @test robust_timedelta_sec(DateTime(1900, 1, 2)) ≈ 86400.0 atol=1e-6

        # One hour after reference = 3600 seconds
        @test robust_timedelta_sec(DateTime(1900, 1, 1, 1, 0, 0)) ≈ 3600.0 atol=1e-6

        # Vector method returns the same values as scalar
        dates = [DateTime(1900, 1, 1), DateTime(1900, 1, 2)]
        result = robust_timedelta_sec(dates)
        @test result[1] ≈ 0.0 atol=1e-6
        @test result[2] ≈ 86400.0 atol=1e-6

        # Custom reference date
        @test robust_timedelta_sec(DateTime(2000, 1, 2); refdate_dt=DateTime(2000, 1, 1)) ≈ 86400.0 atol=1e-6
    end

    @testset "lunar2solar" begin
        # Identity-like: pure T constituent should map [1,0,0,0,0,0] -> [1,-1,1,0,0,0]
        # L=1, S=0, H=0 -> T=L=1, S=S-L=-1, H=H+L=1
        result = lunar2solar([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test result == [1.0, -1.0, 1.0, 0.0, 0.0, 0.0]

        # P, N, P1 are passed through unchanged
        result = lunar2solar([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        @test result[4] ≈ 1.0
        @test result[5] ≈ 2.0
        @test result[6] ≈ 3.0

        # M2: lunar [2,0,0,0,0,0] -> solar [2,-2,2,0,0,0]
        result = lunar2solar([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test result == [2.0, -2.0, 2.0, 0.0, 0.0, 0.0]

        # S2: lunar [2,2,-2,0,0,0] -> solar [2,0,0,0,0,0]
        result = lunar2solar([2.0, 2.0, -2.0, 0.0, 0.0, 0.0])
        @test result == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    @testset "get_doodson_eqvals - scalar" begin
        d = get_doodson_eqvals(DateTime(1900, 1, 1, 12, 0, 0))

        # Should return 6 values
        @test length(d) == 6

        # T: at 12:00 on Jan 1 1900 (exactly at reference noon), dood_Tj ≈ 0
        # dood_T_rad = deg2rad(180 + 12*15) = deg2rad(360) = 2π
        @test d[1] ≈ 2π atol=1e-6

        # All values should be finite
        @test all(isfinite, d)
    end

    @testset "get_doodson_eqvals - vector" begin
        dates = [DateTime(1900, 1, 1, 12, 0, 0), DateTime(2000, 1, 1, 0, 0, 0)]
        result = get_doodson_eqvals(dates)

        @test size(result) == (2, 6)
        @test all(isfinite, result)

        # First row should match scalar call
        scalar_result = get_doodson_eqvals(dates[1])
        @test result[1, :] ≈ scalar_result atol=1e-10
    end

    @testset "primary_frequencies_as_doodson" begin
        # Single constituent
        result = primary_frequencies_as_doodson(["M2"])
        @test size(result) == (6, 1)
        @test result[:, 1] == constituents["M2"]

        # Multiple constituents
        result = primary_frequencies_as_doodson(["M2", "S2", "K1"])
        @test size(result) == (6, 3)
        @test result[:, 1] == constituents["M2"]
        @test result[:, 2] == constituents["S2"]
        @test result[:, 3] == constituents["K1"]

        # Unknown constituent throws
        @test_throws KeyError primary_frequencies_as_doodson(["UNKNOWN"])
    end

    @testset "constituents dictionary" begin
        # All expected constituents are present
        for name in ["M2", "S2", "N2", "K2", "K1", "O1", "Q1", "P1", "SSA"]
            @test haskey(constituents, name)
            @test length(constituents[name]) == 6
        end

        # SSA is long-period (zero T, zero S)
        # lunar [0,0,2,0,0,0] -> solar [0,-0,2,0,0,0] = [0,0,2,0,0,0]
        @test constituents["SSA"][1] ≈ 0.0  # T
        @test constituents["SSA"][2] ≈ 0.0  # S

        # M2 semidiurnal: T component should be 2
        @test constituents["M2"][1] ≈ 2.0

        # S2 solar semidiurnal: in solar doodson should have T=2, S=0, H=0
        @test constituents["S2"] ≈ [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # K1 diurnal: T component should be 1
        @test constituents["K1"][1] ≈ 1.0
    end

end
