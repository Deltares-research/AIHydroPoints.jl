# test_doodson_schureman.jl
# Unit tests for doodson.jl and schureman.jl
# Reference values generated from Python hatyan (schureman.py / hatyan_core.py)

@testset "robust_timedelta_sec" begin
    @test robust_timedelta_sec([DateTime(1900, 1, 1, 0, 0, 0)])[1] ≈ 0.0
    @test robust_timedelta_sec([DateTime(2000, 1, 1, 12, 0, 0)])[1] ≈ 3155716800.0
    @test robust_timedelta_sec([DateTime(2020, 6, 15, 6, 30, 0)])[1] ≈ 3801191400.0
end

@testset "get_doodson_eqvals: position mode" begin
    # 1900-01-01 00:00 — epoch, Tj=0
    d1 = [DateTime(1900, 1, 1, 0, 0, 0)]
    dood1 = get_doodson_eqvals(d1)
    @test size(dood1) == (6, 1)
    @test dood1[DOOD_T,  1] ≈  3.1415926536  atol=1e-8
    @test dood1[DOOD_S,  1] ≈  4.8349946532  atol=1e-7
    @test dood1[DOOD_H,  1] ≈  4.8902293956  atol=1e-7
    @test dood1[DOOD_P,  1] ≈  5.8361247840  atol=1e-7
    @test dood1[DOOD_N,  1] ≈  4.5231394899  atol=1e-7
    @test dood1[DOOD_P1, 1] ≈  4.9082299108  atol=1e-7

    # 2020-06-15 06:30 — arbitrary modern date
    d2 = [DateTime(2020, 6, 15, 6, 30, 0)]
    dood2 = get_doodson_eqvals(d2)
    @test dood2[DOOD_T,  1] ≈  4.8432886743  atol=1e-7
    @test dood2[DOOD_S,  1] ≈  10122.4937567751  atol=1e-4
    @test dood2[DOOD_H,  1] ≈  761.7316965841    atol=1e-4
    @test dood2[DOOD_P,  1] ≈  91.3788602342     atol=1e-4
    @test dood2[DOOD_N,  1] ≈ -36.1381300605     atol=1e-4
    @test dood2[DOOD_P1, 1] ≈  4.9443835030      atol=1e-7
end

@testset "get_doodson_eqvals: freq mode" begin
    d = [DateTime(1900, 1, 1)]
    dood = get_doodson_eqvals(d; mode=:freq)
    @test size(dood) == (6, 1)
    @test dood[DOOD_T,  1] ≈ 0.2617993878  atol=1e-9
    @test dood[DOOD_S,  1] ≈ 0.0095821461  atol=1e-9
    @test dood[DOOD_H,  1] ≈ 0.0007167830  atol=1e-9
    @test dood[DOOD_P,  1] ≈ 0.0000810153  atol=1e-9
    @test isnan(dood[DOOD_N, 1])
    @test dood[DOOD_P1, 1] ≈ 3.423e-8  atol=1e-10
end

@testset "get_schureman_table" begin
    tbl = get_schureman_table()
    @test length(tbl.names) == 221
    @test tbl.names[1] == "A0"
    @test "M2" in tbl.names
    @test "S2" in tbl.names
    @test "K1" in tbl.names
    @test "MS4" in tbl.names   # shallow-water component
    @test "M4"  in tbl.names   # shallow-water component

    # M2 v0u row: T=2, S=-2, H=2, P=0, N=0, P1=0, EDN=0, DKSI=-2, DNU=2, rest 0
    m2_row = tbl.v0u[tbl.index["M2"], :]
    @test m2_row[1] ==  2   # T
    @test m2_row[2] == -2   # S
    @test m2_row[3] ==  2   # H
    @test m2_row[8] ==  2   # DKSI
    @test m2_row[9] == -2   # DNU

    # M2 f row: DND78=1 (everything else 0 for M2)
    m2_f = tbl.f[tbl.index["M2"], :]
    @test m2_f[6] ≈ 1.0    # DND78 (index 6 in 1-based f columns)
    @test sum(m2_f) ≈ 1.0  # only one non-zero f entry for M2

    # size sanity
    @test size(tbl.v0u, 2) == 14
    @test size(tbl.f,   2) == 12
end

@testset "get_schureman_freqs" begin
    const_list = ["A0", "M2", "S2", "K1", "O1", "M4", "MS4"]
    d = [DateTime(1900, 1, 1)]
    freqs = get_schureman_freqs(const_list, d)  # cycles/hr

    @test length(freqs) == 7
    @test freqs[1] ≈ 0.0               atol=1e-12  # A0
    @test freqs[2] ≈ 0.080511400603    atol=1e-10  # M2
    @test freqs[3] ≈ 0.083333333333    atol=1e-10  # S2
    @test freqs[4] ≈ 0.041780746219    atol=1e-10  # K1
    @test freqs[5] ≈ 0.038730654384    atol=1e-10  # O1
    @test freqs[6] ≈ 2 * freqs[2]      atol=1e-10  # M4 ≈ 2·M2
    @test freqs[7] ≈ freqs[2] + freqs[3]  atol=1e-10  # MS4 ≈ M2+S2
end

@testset "get_schureman_v0" begin
    const_list = ["A0", "M2", "S2", "K1", "O1", "M4", "MS4"]
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    v0 = get_schureman_v0(const_list, d)   # n_const × 1

    @test size(v0) == (7, 1)
    @test v0[1, 1] ≈  0.0                 atol=1e-8   # A0
    @test v0[2, 1] ≈ -18711.8375430336    atol=1e-4   # M2
    @test v0[3, 1] ≈   9.6865773486       atol=1e-6   # S2
    @test v0[4, 1] ≈  765.0041889316      atol=1e-4   # K1
    @test v0[5, 1] ≈ -19476.8417319651    atol=1e-4   # O1
    @test v0[6, 1] ≈ -37423.6750860672    atol=1e-3   # M4  ≈ 2·v0(M2)
    @test v0[7, 1] ≈ -18702.1509656850    atol=1e-4   # MS4
end

@testset "get_schureman_constants" begin
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    C = get_schureman_constants(d)

    @test C.DOMEGA ≈ 0.4093197552  atol=1e-8
    @test C.DIKL   ≈ 0.0898037573  atol=1e-8
    @test C.DC5023 ≈ 0.5022605478  atol=1e-8
    @test C.DC1681[1] ≈ 0.1680994609  atol=1e-8
    @test C.DC0365[1] ≈ 0.0364626808  atol=1e-8
end

@testset "get_schureman_u" begin
    const_list = ["A0", "M2", "S2", "K1", "O1", "M4", "MS4"]
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    u = get_schureman_u(const_list, d)

    @test size(u) == (7, 1)
    @test u[1, 1] ≈  0.0             atol=1e-10  # A0
    @test u[2, 1] ≈ -0.0373060275    atol=1e-8   # M2
    @test u[3, 1] ≈  0.0             atol=1e-10  # S2
    @test u[4, 1] ≈ -0.1532974088    atol=1e-8   # K1
    @test u[5, 1] ≈  0.1847541176    atol=1e-8   # O1
    @test u[6, 1] ≈ -0.0746120550    atol=1e-8   # M4  ≈ 2·u(M2)
    @test u[7, 1] ≈ -0.0373060275    atol=1e-8   # MS4

    # Wrapped to (−π, π]
    @test all(-π .< u .<= π)
end

@testset "HatyanSettings" begin
    s = HatyanSettings()
    @test s.nodalfactors == true
    @test s.fu_alltimes  == true
    @test s.xfac         == false

    s2 = HatyanSettings(nodalfactors=false, fu_alltimes=false, xfac=true)
    @test s2.nodalfactors == false
    @test s2.fu_alltimes  == false
    @test s2.xfac         == true
end

@testset "get_schureman_f" begin
    const_list = ["A0", "M2", "S2", "K1", "O1", "M4", "MS4"]
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    f = get_schureman_f(const_list, d, false)

    @test size(f) == (7, 1)
    @test f[1, 1] ≈ 1.0000000000  atol=1e-8   # A0
    @test f[2, 1] ≈ 0.9998078412  atol=1e-8   # M2
    @test f[3, 1] ≈ 1.0000000000  atol=1e-8   # S2
    @test f[4, 1] ≈ 1.0158269809  atol=1e-8   # K1
    @test f[5, 1] ≈ 1.0250882511  atol=1e-8   # O1
    @test f[6, 1] ≈ 0.9996157193  atol=1e-8   # M4
    @test f[7, 1] ≈ 0.9998078412  atol=1e-8   # MS4

    # All f values should be positive
    @test all(f .> 0)
end

@testset "get_freqv0_generic" begin
    const_list = ["A0", "M2", "S2", "K1"]
    d_mid   = [DateTime(2020, 6, 15, 6, 30, 0)]
    d_start = [DateTime(2020, 1, 1, 0, 0, 0)]

    freq, v0 = get_freqv0_generic(const_list, d_mid, d_start, "schureman")

    # freq matches get_schureman_freqs directly
    @test freq ≈ get_schureman_freqs(const_list, d_mid)  atol=1e-12
    # v0 matches get_schureman_v0 at d_start
    @test v0   ≈ get_schureman_v0(const_list, d_start)   atol=1e-8

    @test length(freq) == 4
    @test size(v0) == (4, 1)

    # Foreman also works
    freq_f, v0_f = get_freqv0_generic(const_list, d_mid, d_start, "foreman")
    @test length(freq_f) == 4
    @test size(v0_f) == (4, 1)
    # M2 frequency should be close between methods (< 1e-5 cycles/hr)
    @test abs(freq_f[2] - freq[2]) < 1e-5

    @test_throws ErrorException get_freqv0_generic(const_list, d_mid, d_start, "unknown")
end

@testset "get_uf_generic: nodalfactors=true" begin
    const_list = ["A0", "M2", "S2", "K1"]
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)

    u, f = get_uf_generic(const_list, d, settings, "schureman")

    @test size(u) == (4, 1)
    @test size(f) == (4, 1)
    # Results should match direct Schureman calls
    @test u ≈ get_schureman_u(const_list, d)            atol=1e-10
    @test f ≈ get_schureman_f(const_list, d, false)     atol=1e-10
end

@testset "get_uf_generic: nodalfactors=false" begin
    const_list = ["A0", "M2", "S2", "K1"]
    d = [DateTime(2020, 6, 15, 6, 30, 0)]
    settings = HatyanSettings(nodalfactors=false, fu_alltimes=true, xfac=false)

    u, f = get_uf_generic(const_list, d, settings, "schureman")

    @test size(u) == (4, 1)
    @test size(f) == (4, 1)
    @test all(u .== 0.0)
    @test all(f .== 1.0)
end

@testset "get_uf_generic: fu_alltimes (multiple dates)" begin
    const_list = ["M2", "S2"]
    dates = [DateTime(2020, 1, 1) + Hour(i) for i in 0:23]   # 24 hourly steps
    settings = HatyanSettings(nodalfactors=true, fu_alltimes=true, xfac=false)

    u, f = get_uf_generic(const_list, dates, settings, "schureman")

    @test size(u) == (2, 24)
    @test size(f) == (2, 24)
    # S2 has f=1 and u=0 for all times (no nodal dependence)
    @test all(f[2, :] .≈ 1.0)
    @test all(u[2, :] .≈ 0.0)
end
