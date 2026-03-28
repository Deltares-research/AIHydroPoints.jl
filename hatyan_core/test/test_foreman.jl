# test_foreman.jl
# Tests for the Foreman (2004) tidal method:
#   - table loading
#   - get_foreman_freqs / get_foreman_v0
#   - get_foreman_uf
#   - prediction with method="foreman"
#   - analysis with method="foreman"
#   - analysis/prediction roundtrip

# ── helpers ───────────────────────────────────────────────────────────────────

function _make_foreman_tc(; A0=0.5f0, A_M2=1.2f0, phi_M2=30.0f0,
                                      A_S2=0.4f0, phi_S2=120.0f0)
    TidalConstituents(
        Float32[A0 A_M2 A_S2],
        Float32[0.0f0 phi_M2 phi_S2],
        ["A0", "M2", "S2"],
        ["SYN"], [0.0], [0.0],
        "water level", "SYN | analysis(foreman)",
    )
end

# ── table loading ─────────────────────────────────────────────────────────────

@testset "get_foreman_table: loading" begin
    t = get_foreman_table()

    @test t isa hatyan_core.ForemanTable
    # Standard harmonic constituents must be present
    @test haskey(t.harmonic, "M2")
    @test haskey(t.harmonic, "S2")
    @test haskey(t.harmonic, "K1")
    @test haskey(t.harmonic, "O1")
    @test haskey(t.harmonic, "N2")

    # Shallow constituents must be present
    @test "M4"  ∈ t.shallow_set
    @test "MS4" ∈ t.shallow_set
    @test "M6"  ∈ t.shallow_set

    # Reasonable count
    @test length(t.harmonic_names) >= 40
    @test length(t.satellites) >= 100
    @test length(t.shallow) >= 50

    # Caching: same object returned on second call
    @test get_foreman_table() === get_foreman_table()
end

@testset "get_foreman_table: M2 Doodson numbers" begin
    t = get_foreman_table()
    hm = t.harmonic["M2"]
    # M2 solar Doodson: T=2, S=−2, H=2, P=0, N=0, P1=0 (after lunar→solar conversion)
    @test hm.T  ==  2
    @test hm.S  == -2
    @test hm.H  ==  2
    @test hm.P  ==  0
    @test hm.N  ==  0
    @test hm.P1 ==  0
end

# ── frequencies ───────────────────────────────────────────────────────────────

@testset "get_foreman_freqs: known frequencies" begin
    d = [DateTime(2020, 1, 1, 12)]
    freqs = get_foreman_freqs(["M2", "S2", "K1", "O1", "M4"], d)

    # Reference values (cycles/hr, Schureman/Foreman agree to ~1e-5)
    @test freqs[1] ≈ 0.08051140  atol=1e-5   # M2
    @test freqs[2] ≈ 0.08333333  atol=1e-5   # S2 (exactly 1/12 cycle/hr)
    @test freqs[3] ≈ 0.04178100  atol=1e-5   # K1
    @test freqs[4] ≈ 0.03873065  atol=1e-5   # O1
    @test freqs[5] ≈ 2 * freqs[1]  atol=1e-8  # M4 = 2×M2
end

@testset "get_foreman_freqs: shape and sign" begin
    consts = ["A0", "M2", "S2", "K1", "O1"]
    d = [DateTime(2020, 6, 15, 0)]
    freqs = get_foreman_freqs(consts, d)

    @test length(freqs) == 5
    @test all(freqs .>= 0.0)   # all non-negative
    @test freqs[1] == 0.0      # A0 (zero frequency)
end

@testset "get_foreman_freqs vs get_schureman_freqs" begin
    consts = ["M2", "S2", "N2", "K1", "O1"]
    d = [DateTime(2020, 1, 1, 0)]
    f_for = get_foreman_freqs(consts, d)
    f_sch = get_schureman_freqs(consts, d)
    # Frequencies from both methods agree to within 1e-5 cycles/hr
    @test maximum(abs.(f_for .- f_sch)) < 1e-5
end

# ── initial phase v₀ ──────────────────────────────────────────────────────────

@testset "get_foreman_v0: shape" begin
    consts = ["M2", "S2", "K1"]
    dates  = [DateTime(2020, 1, 1), DateTime(2020, 7, 1)]
    v0 = get_foreman_v0(consts, dates)

    @test size(v0) == (3, 2)
    @test eltype(v0) == Float64
end

@testset "get_foreman_v0: S2 phase evolves at solar rate" begin
    # S2 has T=2, S=0, H=0: v0 changes only with T (hour of day)
    # Adding 12 hours should add 2*π (one full S2 cycle = 12 hr)
    d1 = [DateTime(2020, 1, 1,  0)]
    d2 = [DateTime(2020, 1, 1, 12)]
    v1 = get_foreman_v0(["S2"], d1)[1, 1]
    v2 = get_foreman_v0(["S2"], d2)[1, 1]
    @test mod(v2 - v1, 2π) ≈ 0.0  atol=1e-8
end

@testset "get_foreman_v0: shallow M4 = 2×M2" begin
    d  = [DateTime(2020, 3, 15, 6)]
    v0 = get_foreman_v0(["M2", "M4"], d)
    # M4 v0 should equal 2 * M2 v0 (modulo 2π wrapping doesn't matter for linear check)
    @test v0[2, 1] ≈ 2 * v0[1, 1]  atol=1e-8
end

# ── nodal factors ─────────────────────────────────────────────────────────────

@testset "get_foreman_uf: shape" begin
    consts = ["M2", "S2", "K1", "O1"]
    dates  = [DateTime(2020, 1, 1) + Hour(h) for h in 0:23]
    u, f = get_foreman_uf(consts, dates)

    @test size(u) == (4, 24)
    @test size(f) == (4, 24)
    @test eltype(u) == Float64
    @test eltype(f) == Float64
end

@testset "get_foreman_uf: f positive, f near 1" begin
    consts = ["M2", "S2", "K1", "O1", "N2"]
    d = [DateTime(2020, 1, 1, 12)]
    u, f = get_foreman_uf(consts, d)

    @test all(f .> 0.0)
    # Nodal factors are near 1 (within ±15% for standard constituents)
    @test all(abs.(f .- 1.0) .< 0.15)
end

@testset "get_foreman_uf: A0 has f=1 and u=0" begin
    d = [DateTime(2020, 6, 1, 0)]
    u, f = get_foreman_uf(["A0"], d)
    @test f[1, 1] ≈ 1.0  atol=1e-10
    @test u[1, 1] ≈ 0.0  atol=1e-10
end

@testset "get_foreman_uf: shallow M4 f = f_M2^2" begin
    d = [DateTime(2020, 3, 15, 6)]
    u_all, f_all = get_foreman_uf(["M2", "M4"], d)
    @test f_all[2, 1] ≈ f_all[1, 1]^2  atol=1e-10
    @test u_all[2, 1] ≈ 2 * u_all[1, 1]  atol=1e-10
end

# ── prediction with method="foreman" ─────────────────────────────────────────

@testset "prediction: foreman basic shape and source" begin
    tc    = _make_foreman_tc()
    times = [DateTime(2020, 1, 1) + Minute(10*i) for i in 0:143]

    ts = prediction(tc, times)

    @test ts isa TimeSeries
    @test size(get_values(ts)) == (1, 144)
    @test occursin("foreman", get_source(ts))
    @test occursin("prediction", get_source(ts))
end

@testset "prediction: foreman physical range" begin
    tc    = _make_foreman_tc()
    times = [DateTime(2020, 1, 1) + Minute(10*i) for i in 0:4463]  # 1 month

    ts = prediction(tc, times)
    h  = vec(get_values(ts))

    # Signal bounded by sum of amplitudes: |h| ≤ A0 + A_M2 + A_S2 = 2.1 m (allow small nodal overshoot)
    @test maximum(abs.(h)) ≤ 2.2f0
    @test minimum(h) < 0.0f0   # tidal signal crosses zero
    @test maximum(h) > 0.0f0
end

@testset "prediction: foreman vs schureman close for short window" begin
    # Over a short window (a few days) the two methods should agree closely
    # since nodal factors and phases are nearly identical.
    tc_sch = TidalConstituents(
        Float32[0.5 1.2 0.4],
        Float32[0.0 30.0 120.0],
        ["A0", "M2", "S2"],
        ["SYN"], [0.0], [0.0],
        "water level", "SYN | analysis(schureman)",
    )
    tc_for = TidalConstituents(
        tc_sch.amplitudes, tc_sch.phases, tc_sch.constituent_names,
        tc_sch.names, tc_sch.longitudes, tc_sch.latitudes,
        tc_sch.quantity, "SYN | analysis(foreman)",
    )

    times    = [DateTime(2020, 6, 1) + Minute(10*i) for i in 0:143]
    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)

    h_sch = vec(get_values(prediction(tc_sch, times, settings)))
    h_for = vec(get_values(prediction(tc_for, times, settings)))

    # Should agree to within 1 cm over a one-day window
    @test maximum(abs.(h_sch .- h_for)) < 0.01f0
end

# ── analysis with method="foreman" ───────────────────────────────────────────

@testset "analysis: foreman return type and shape" begin
    tc_in = _make_foreman_tc()
    times = [DateTime(2020, 1, 1) + Minute(10*i) for i in 0:4463]
    ts    = prediction(tc_in, times)

    tc_out = analysis(ts, ["A0", "M2", "S2"], "foreman")

    @test tc_out isa TidalConstituents
    @test size(tc_out.amplitudes) == (1, 3)
    @test size(tc_out.phases)     == (1, 3)
    @test tc_out.constituent_names == ["A0", "M2", "S2"]
    @test occursin("foreman", get_source(tc_out))
end

@testset "analysis: foreman synthetic roundtrip" begin
    tc_in  = _make_foreman_tc()
    times  = [DateTime(2020, 1, 1) + Minute(10*i) for i in 0:4463]
    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)

    ts     = prediction(tc_in, times, settings)
    tc_out = analysis(ts, ["A0", "M2", "S2"], "foreman", settings)

    A   = tc_out.amplitudes[1, :]
    phi = tc_out.phases[1, :]

    @test A[1]   ≈ 0.5f0    atol=1e-4   # A0
    @test A[2]   ≈ 1.2f0    atol=1e-4   # M2
    @test A[3]   ≈ 0.4f0    atol=1e-4   # S2
    @test phi[1] ≈ 0.0f0    atol=1e-3
    @test phi[2] ≈ 30.0f0   atol=1e-3
    @test phi[3] ≈ 120.0f0  atol=1e-3
end

@testset "analysis: foreman year roundtrip" begin
    # Predict a full year from known constituents, re-analyse, check recovery.
    tc_ref = TidalConstituents(
        Float32[0.0 1.2 0.4 0.3],
        Float32[0.0 30.0 120.0 45.0],
        ["A0", "M2", "S2", "K1"],
        ["SYN"], [0.0], [0.0],
        "water level", "SYN | analysis(foreman)",
    )
    times    = [DateTime(2020, 1, 1) + Minute(10*i) for i in 0:52559]   # 1 year
    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)

    ts     = prediction(tc_ref, times, settings)
    tc_out = analysis(ts, ["A0", "M2", "S2", "K1"], "foreman", settings)

    A_ref = vec(tc_ref.amplitudes)
    A_out = vec(tc_out.amplitudes)
    p_ref = vec(tc_ref.phases)
    p_out = vec(tc_out.phases)

    @test maximum(abs.(A_out .- A_ref)) < 1e-3
    dp = mod.(p_out .- p_ref .+ 180f0, 360f0) .- 180f0
    @test maximum(abs.(dp)) < 0.01f0
end
