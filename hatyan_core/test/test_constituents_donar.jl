@testset "read_donar_constituents: VLISSGN_ana.txt" begin
    filename = joinpath(TEST_DATA_DIR, "VLISSGN_ana.txt")
    tc = read_donar_constituents(filename)

    @testset "metadata" begin
        @test get_names(tc)   == ["VLISSGN"]
        @test get_source(tc)  == "VLISSGN"
        @test occursin("WATHTE", get_quantity(tc))
        @test occursin("NAP",    get_quantity(tc))
    end

    @testset "constituent list" begin
        cnames = get_constituent_names(tc)
        # A0 prepended from MIDD line, plus 94 COMP lines
        @test length(cnames) == 95
        @test cnames[1]      == "A0"
        @test "M2" in cnames
        @test "S2" in cnames
        @test "K1" in cnames
        @test "O1" in cnames
    end

    @testset "data shape" begin
        @test size(get_amplitudes(tc)) == (1, 95)
        @test size(get_phases(tc))     == (1, 95)
    end

    @testset "A0 (mean water level)" begin
        i = findfirst(==("A0"), get_constituent_names(tc))
        # MIDD 1.000 cm → 0.01 m
        @test get_amplitudes(tc)[1, i] ≈ 0.01f0  atol=1e-4
        @test get_phases(tc)[1, i]     ≈ 0.0f0
    end

    @testset "M2 constituent" begin
        i = findfirst(==("M2"), get_constituent_names(tc))
        # COMP 65  28.984104  174.666  59.47  M2  → 174.666 cm = 1.74666 m, phase 59.47°
        @test get_amplitudes(tc)[1, i] ≈ 1.74666f0  atol=1e-4
        @test get_phases(tc)[1, i]     ≈ 59.47f0    atol=1e-2
    end

    @testset "physically plausible values" begin
        amps = get_amplitudes(tc)
        @test all(amps .>= 0.0f0)   # amplitudes are non-negative
        @test maximum(amps) < 5.0f0 # no implausible amplitude for Vlissingen
        phases = get_phases(tc)
        @test all(phases .>= 0.0f0)
        @test all(phases .< 360.0f0)
    end
end
