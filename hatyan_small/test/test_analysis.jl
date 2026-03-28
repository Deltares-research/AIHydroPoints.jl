# test_analysis.jl
# Unit tests for analysis.jl
#
# Strategy:
#   1. Synthetic roundtrip — build a clean 3-constituent signal with known A/φ
#      using prediction(), analyse it, verify recovery.
#   2. Year roundtrip     — predict a full year from VLISSGN_ana.txt, analyse
#      the prediction, verify constituents match the originals.
#   3. NaN handling       — drop NaN rows, result should still converge.
#   4. Source provenance  — source string is extended correctly.
#   5. Condition check    — too-short series raises an error.

# ── helper: build a TidalConstituents with three known constituents ────────────
function _make_synthetic_tc(; A0=0.5f0, A_M2=1.2f0, phi_M2=30.0f0,
                                         A_S2=0.4f0, phi_S2=120.0f0)
    amplitudes = Float32[A0    A_M2    A_S2 ]   # 1 × 3
    phases     = Float32[0.0f0 phi_M2  phi_S2]
    TidalConstituents(
        amplitudes, phases,
        ["A0", "M2", "S2"],
        ["SYN"], [0.0], [0.0],
        "water level", "SYN",
    )
end

@testset "analysis: synthetic 3-constituent roundtrip" begin
    tc_in  = _make_synthetic_tc()
    # 1 month of 10-min data (4 464 steps)
    times  = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4463]
    ts     = prediction(tc_in, times, HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false))

    tc_out = analysis(ts, ["A0", "M2", "S2"], "schureman",
                      HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false))

    A   = tc_out.amplitudes[1, :]
    phi = tc_out.phases[1, :]

    # Values should be recovered to Float32 storage precision (~1e-4)
    @test A[1]   ≈ 0.5f0   atol=1e-4   # A0
    @test A[2]   ≈ 1.2f0   atol=1e-4   # M2
    @test A[3]   ≈ 0.4f0   atol=1e-4   # S2

    @test phi[1] ≈ 0.0f0   atol=1e-3   # A0
    @test phi[2] ≈ 30.0f0  atol=1e-3   # M2
    @test phi[3] ≈ 120.0f0 atol=1e-3   # S2
end

@testset "analysis: return type and shape" begin
    tc_in = _make_synthetic_tc()
    times = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4463]
    ts    = prediction(tc_in, times)

    tc_out = analysis(ts, ["A0", "M2", "S2"])

    @test tc_out isa TidalConstituents
    @test size(tc_out.amplitudes) == (1, 3)
    @test size(tc_out.phases)     == (1, 3)
    @test tc_out.constituent_names == ["A0", "M2", "S2"]
    @test tc_out.names == ["SYN"]
end

@testset "analysis: source provenance" begin
    tc_in = _make_synthetic_tc()
    times = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4463]
    ts    = prediction(tc_in, times)

    tc_out = analysis(ts, ["A0", "M2", "S2"])
    @test get_source(tc_out) == "SYN | prediction | analysis(schureman)"
end

@testset "analysis: NaN handling" begin
    tc_in = _make_synthetic_tc()
    times = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4463]
    ts    = prediction(tc_in, times)

    # Inject NaNs into ~10% of values
    vals = copy(get_values(ts))
    vals[1, 1:10:end] .= NaN32
    ts_nan = TimeSeries(vals, get_times(ts), get_names(ts),
                        get_longitudes(ts), get_latitudes(ts),
                        get_quantity(ts), get_source(ts))

    tc_out = analysis(ts_nan, ["A0", "M2", "S2"])
    A = tc_out.amplitudes[1, :]

    # Should still recover within looser tolerance (fewer clean observations)
    @test A[1] ≈ 0.5f0 atol=1e-3
    @test A[2] ≈ 1.2f0 atol=1e-3
    @test A[3] ≈ 0.4f0 atol=1e-3
end

@testset "analysis: year roundtrip on VLISSGN" begin
    # Predict a full year from the reference constituents, then analyse.
    # Differences should be within Float32 storage precision (~1e-4 m / ~1e-3 deg).
    tc_ref = read_donar_constituents(joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt"))
    times  = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:52559]   # 1 year, 10-min
    ts     = prediction(tc_ref, times, HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false))

    const_list = constituent_list("year")
    tc_out = analysis(ts, const_list, "schureman",
                      HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false))

    A_ref = vec(tc_ref.amplitudes)
    A_out = vec(tc_out.amplitudes)
    p_ref = vec(tc_ref.phases)
    p_out = vec(tc_out.phases)

    # All amplitudes recovered within 1e-3 m
    @test maximum(abs.(A_out .- A_ref)) < 1e-3

    # Phase differences: wrap to (−180, 180] before comparing
    dp = mod.(p_out .- p_ref .+ 180f0, 360f0) .- 180f0
    @test maximum(abs.(dp)) < 0.01f0   # < 0.01 degree
end

@testset "analysis: two locations" begin
    # Build a two-location TimeSeries from two different synthetic signals,
    # analyse together, verify each location is fitted independently.
    times = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4463]   # 1 month

    # Location 1: A0=0.5, M2=1.2/30°, S2=0.4/120°
    tc1 = _make_synthetic_tc(A0=0.5f0, A_M2=1.2f0, phi_M2=30.0f0,
                                        A_S2=0.4f0, phi_S2=120.0f0)
    # Location 2: different amplitudes and phases
    tc2 = _make_synthetic_tc(A0=0.2f0, A_M2=0.8f0, phi_M2=75.0f0,
                                        A_S2=0.6f0, phi_S2=210.0f0)

    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)
    ts1 = prediction(tc1, times, settings)
    ts2 = prediction(tc2, times, settings)

    # Combine into a single two-location TimeSeries
    ts_both = TimeSeries(
        vcat(get_values(ts1), get_values(ts2)),
        times,
        ["LOC1", "LOC2"],
        [3.0, 4.0], [51.0, 52.0],
        "water level", "SYN",
    )

    tc_out = analysis(ts_both, ["A0", "M2", "S2"], "schureman", settings)

    @test size(tc_out.amplitudes) == (2, 3)
    @test size(tc_out.phases)     == (2, 3)
    @test tc_out.names == ["LOC1", "LOC2"]

    # Location 1 recovery
    @test tc_out.amplitudes[1, 1] ≈ 0.5f0 atol=1e-4   # A0
    @test tc_out.amplitudes[1, 2] ≈ 1.2f0 atol=1e-4   # M2
    @test tc_out.phases[1, 2]     ≈ 30.0f0 atol=1e-3

    # Location 2 recovery
    @test tc_out.amplitudes[2, 1] ≈ 0.2f0 atol=1e-4   # A0
    @test tc_out.amplitudes[2, 2] ≈ 0.8f0 atol=1e-4   # M2
    @test tc_out.phases[2, 2]     ≈ 75.0f0 atol=1e-3

    # The two locations must differ
    @test tc_out.amplitudes[1, :] != tc_out.amplitudes[2, :]
end

@testset "analysis: condition error on too-short series" begin
    tc_in = _make_synthetic_tc()
    # Only 5 timesteps — far too few for 3 constituents; xTx will be ill-conditioned
    times = [DateTime(2019, 1, 1) + Minute(10*i) for i in 0:4]
    ts    = prediction(tc_in, times)

    @test_throws ErrorException analysis(ts, ["A0", "M2", "S2"])
end
