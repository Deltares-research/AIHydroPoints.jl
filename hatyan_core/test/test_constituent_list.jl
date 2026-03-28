# test_constituent_list.jl

@testset "constituent_list: known presets" begin
    @test constituent_list("year")       isa Vector{String}
    @test constituent_list("halfyear")   isa Vector{String}
    @test constituent_list("month")      isa Vector{String}
    @test constituent_list("month_deepwater") isa Vector{String}
    @test constituent_list("springneap") isa Vector{String}
    @test constituent_list("day")        isa Vector{String}
    @test constituent_list("tidalcycle") isa Vector{String}
end

@testset "constituent_list: lengths" begin
    @test length(constituent_list("year"))       == 95
    @test length(constituent_list("halfyear"))   == 89
    @test length(constituent_list("month"))      == 22
    @test length(constituent_list("month_deepwater")) == 22
    @test length(constituent_list("springneap")) == 15
    @test length(constituent_list("day"))        == 11
    @test length(constituent_list("tidalcycle")) == 7
end

@testset "constituent_list: content spot-checks" begin
    year = constituent_list("year")
    @test "A0"  in year
    @test "M2"  in year
    @test "S2"  in year
    @test "K1"  in year
    @test "M4"  in year
    @test year[1] == "A0"   # A0 is always first

    month = constituent_list("month")
    @test "M2"  in month
    @test "S2"  in month
    @test !("SA" in month)  # SA needs a full year to resolve

    # deep-water variant swaps 2MN2 for L2
    @test "L2"    in constituent_list("month_deepwater")
    @test !("2MN2" in constituent_list("month_deepwater"))
    @test "2MN2"  in constituent_list("month")
end

@testset "constituent_list: all" begin
    all_list = constituent_list("all")
    @test all_list isa Vector{String}
    @test length(all_list) > 100
    @test "M2" in all_list
    @test "A0" in all_list
end

@testset "constituent_list: returns independent copy" begin
    a = constituent_list("day")
    b = constituent_list("day")
    push!(a, "FAKE")
    @test length(b) == 11   # original unaffected
end

@testset "constituent_list: unknown name errors" begin
    @test_throws ErrorException constituent_list("decade")
    @test_throws ErrorException constituent_list("")
end
