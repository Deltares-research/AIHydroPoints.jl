# test_prediction.jl
# Unit tests for prediction.jl
#
# Reference values generated from Python hatyan prediction_singleperiod()
# using VLISSGN_ana.txt constituents, 10-min steps on 2019-01-01 (naive UTC+1).

@testset "_extract_method_from_source" begin
    @test hatyan_small._extract_method_from_source("VLISSGN") == "schureman"
    @test hatyan_small._extract_method_from_source("VLISSGN | analysis(schureman)") == "schureman"
    @test hatyan_small._extract_method_from_source("X | analysis(foreman)") == "foreman"
    @test hatyan_small._extract_method_from_source("A | analysis(schureman) | prediction") == "schureman"
end

@testset "prediction: VLISSGN Jan 2019 (fu_alltimes=true)" begin
    # Read constituents from file — source will be "VLISSGN" (no analysis token)
    # so prediction falls back to method="schureman"
    tc = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))

    # 10-minute steps, 2019-01-01 00:00 to 23:50 (naive, 144 steps)
    times = [DateTime(2019, 1, 1) + Minute(10 * i) for i in 0:143]

    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)
    ts = prediction(tc, times, settings)

    # ── basic structure ────────────────────────────────────────────────────────
    @test ts isa TimeSeries
    @test length(get_times(ts)) == 144
    @test get_times(ts)[1]  == DateTime(2019, 1, 1, 0, 0)
    @test get_times(ts)[end] == DateTime(2019, 1, 1, 23, 50)
    @test size(get_values(ts)) == (1, 144)
    @test get_names(ts) == ["VLISSGN"]
    @test occursin("prediction", get_source(ts))

    # ── values match Python reference (tolerance 1e-4 m) ──────────────────────
    h = vec(get_values(ts))
    @test h[1]   ≈  1.01851115f0  atol=1e-4   # t=0
    @test h[2]   ≈  0.90534129f0  atol=1e-4   # t=10 min
    @test h[6]   ≈  0.38741335f0  atol=1e-4   # t=50 min
    @test h[end] ≈  1.69676010f0  atol=1e-4   # t=23:50

    # ── physical range for Vlissingen ─────────────────────────────────────────
    @test minimum(h) ≈ -1.68145835f0  atol=1e-3
    @test maximum(h) ≈  1.89444336f0  atol=1e-3
    @test minimum(h) > -3.0f0
    @test maximum(h) <  3.0f0
end

@testset "prediction: fu_alltimes=false" begin
    tc = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    times = [DateTime(2019, 1, 1) + Minute(10 * i) for i in 0:143]

    settings = HatyanSettings(nodalfactors=true, fu_alltimes=false, xfac=false)
    ts = prediction(tc, times, settings)

    h = vec(get_values(ts))
    @test h[1]   ≈  1.01853528f0  atol=1e-4
    @test h[73]  ≈  1.13158086f0  atol=1e-4   # index 73 = 720 min = t[72] in 0-based = h[73] in 1-based
    @test size(get_values(ts)) == (1, 144)
end

@testset "prediction: nodalfactors=false" begin
    tc = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    times = [DateTime(2019, 1, 1) + Minute(10 * i) for i in 0:143]

    ts_nf_off = prediction(tc, times, HatyanSettings(nodalfactors=false))
    ts_nf_on  = prediction(tc, times, HatyanSettings(nodalfactors=true))

    h_off = vec(get_values(ts_nf_off))
    h_on  = vec(get_values(ts_nf_on))

    # Should be close but not identical — nodal factors are near 1
    @test maximum(abs.(h_off .- h_on)) < 0.5f0
    @test maximum(abs.(h_off .- h_on)) > 1f-6
end

@testset "prediction: StepRange times" begin
    tc    = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    start = DateTime(2019, 1, 1)
    stop  = DateTime(2019, 1, 1, 23, 50)
    times_range = start:Minute(10):stop

    ts = prediction(tc, times_range)
    @test length(get_times(ts)) == 144
    @test get_times(ts)[1]  == start
    @test get_times(ts)[end] == stop
end

@testset "prediction: source provenance" begin
    tc = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    times = [DateTime(2019, 1, 1) + Minute(10 * i) for i in 0:9]

    ts = prediction(tc, times)
    @test get_source(ts) == "VLISSGN | prediction"

    # Simulate constituents that went through analysis
    tc2 = TidalConstituents(
        tc.amplitudes, tc.phases, tc.constituent_names,
        tc.names, tc.longitudes, tc.latitudes, tc.quantity,
        "VLISSGN | analysis(schureman)",
    )
    ts2 = prediction(tc2, times)
    @test get_source(ts2) == "VLISSGN | analysis(schureman) | prediction"
end

@testset "prediction: compare with VLISSGN_pre.txt" begin
    # Both Julia and Python predictions should agree to within ~0.02 m;
    # VLISSGN_pre.txt values are stored at 1 cm resolution so some rounding is expected.
    tc    = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    ts_ref = read_donar_timeseries(joinpath(TEST_DATA_DIR, "VLISSGN_pre.txt"))

    # First day of 2019 at 10-minute steps (UTC+1 naive, 144 steps)
    times = [DateTime(2019, 1, 1) + Minute(10 * i) for i in 0:143]
    ts = prediction(tc, times, HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false))

    h_julia = vec(get_values(ts))
    # Select the first 144 timesteps from the reference file
    h_ref   = vec(get_values(ts_ref)[1, 1:144])

    # Max difference should be within rounding (VLISSGN_pre.txt is stored in cm)
    @test maximum(abs.(h_julia .- h_ref)) < 0.03f0
end
