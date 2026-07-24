using Test
using AIHydroPoints

@testset "check_format_version" begin
    cfv = AIHydroPoints.check_format_version
    V   = AIHydroPoints.CURRENT_FORMAT_VERSION
    @test V == 2

    # Current version passes.
    @test cfv(Dict{String,Any}("format_version" => V)) === nothing
    # Missing key → treated as version 1 (old format) → rejected.
    @test_throws ErrorException cfv(Dict{String,Any}())
    # Explicit older version → rejected.
    @test_throws ErrorException cfv(Dict{String,Any}("format_version" => 1))
    # Newer than this build supports → rejected.
    @test_throws ErrorException cfv(Dict{String,Any}("format_version" => V + 1))
end

@testset "TrainingSettings unknown-key rejection" begin
    # Known keys construct fine.
    ts = TrainingSettings(Dict("nepochs" => 3, "batch_size" => 8))
    @test ts.nepochs == 3
    @test ts.batch_size == 8

    # A stale/old or misspelled key errors instead of silently falling back.
    @test_throws ErrorException TrainingSettings(Dict("nbatches" => 8))
    @test_throws ErrorException TrainingSettings(Dict("bogus_key" => 1))
end

@testset "train errors when no training split" begin
    ex    = joinpath(@__DIR__, "..", "examples", "LinearSurgeModel.toml")
    exdir = dirname(abspath(ex))
    s     = toml_read(ex)
    # Absolutise data paths (they are relative to the example dir) and typo the
    # training split so no "training" split remains.
    for f in s["data_settings"]["files"]
        f["path"] = abspath(joinpath(exdir, f["path"]))
        f["split"] == "training" && (f["split"] = "trianing")
    end
    mkpath(joinpath(@__DIR__, "temp"))
    tmp = joinpath(@__DIR__, "temp", "no_training_split.toml")
    toml_write(tmp, s; overwrite=true)
    @test_throws ErrorException train(tmp)
end
