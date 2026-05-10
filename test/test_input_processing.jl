using Test
using AIHydroPoints
using Dates

# ── Helpers ───────────────────────────────────────────────────────────────────

function ip_make_ts(nlocations, ntimes, quantity)
    values = randn(Float32, nlocations, ntimes)
    times  = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names  = ["loc$i" for i in 1:nlocations]
    lons   = Float64.(1:nlocations)
    lats   = Float64.(51 .+ (1:nlocations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

function ip_make_all_settings(; model_name="ConvSurgeModel", include_model_dir=false)
    ms = Dict{String,Any}("model_name" => model_name, "nlags" => 4)
    include_model_dir && (ms["model_dir"] = "custom/dir")
    return Dict{String,Any}(
        "run_info"       => Dict{String,Any}("runid" => "test_run", "description" => "test"),
        "model_settings" => ms,
        "train_settings" => Dict{String,Any}("nepochs" => 10),
        "data_settings"  => Dict{String,Any}(
            "model_io" => Dict("input" => ["stress_x"], "target" => ["surge"]),
        ),
    )
end

# ── Happy path: surge-style (non-empty input + target) ────────────────────────

@testset "validate_and_augment_settings! surge-style" begin
    all_settings = ip_make_all_settings()
    train_input  = Dict{String,TimeSeries}("stress_x" => ip_make_ts(3, 50, "stress_x"))
    train_target = Dict{String,TimeSeries}("surge"    => ip_make_ts(2, 50, "surge"))

    validate_and_augment_settings!(all_settings, train_input, train_target)

    ms = all_settings["model_settings"]

    # output-side keys populated
    @test haskey(ms, "out_quantities")
    @test haskey(ms, "out_names")
    @test haskey(ms, "out_lons")
    @test haskey(ms, "out_lats")
    @test haskey(ms, "nlocations_output")
    @test ms["out_quantities"]    == ["surge"]
    @test ms["out_names"]         == ["loc$i" for i in 1:2]
    @test ms["nlocations_output"] == 2

    # input-side keys populated
    @test haskey(ms, "in_quantities")
    @test haskey(ms, "in_names")
    @test haskey(ms, "in_lons")
    @test haskey(ms, "in_lats")
    @test haskey(ms, "nlocations_input")
    @test ms["in_quantities"]    == ["stress_x"]
    @test ms["nlocations_input"] == 3

    # model_dir derived from runid + model_name
    @test ms["model_dir"] == joinpath("training_output", "test_run_ConvSurgeModel")
end

# ── Happy path: tide-style (empty input) ──────────────────────────────────────

@testset "validate_and_augment_settings! empty input (tide-style)" begin
    all_settings = ip_make_all_settings(model_name="DeepONetTideModel")
    train_input  = Dict{String,TimeSeries}()
    train_target = Dict{String,TimeSeries}("waterlevel" => ip_make_ts(5, 100, "waterlevel"))

    validate_and_augment_settings!(all_settings, train_input, train_target)

    ms = all_settings["model_settings"]

    @test haskey(ms, "out_names")
    @test ms["nlocations_output"] == 5

    # in_* keys must NOT be present for empty input
    @test !haskey(ms, "in_names")
    @test !haskey(ms, "nlocations_input")
end

# ── model_dir not overwritten when already present ────────────────────────────

@testset "validate_and_augment_settings! preserves existing model_dir" begin
    all_settings = ip_make_all_settings(include_model_dir=true)
    train_input  = Dict{String,TimeSeries}("stress_x" => ip_make_ts(3, 20, "stress_x"))
    train_target = Dict{String,TimeSeries}("surge"    => ip_make_ts(2, 20, "surge"))

    validate_and_augment_settings!(all_settings, train_input, train_target)

    @test all_settings["model_settings"]["model_dir"] == "custom/dir"
end

# ── Idempotent: existing keys not overwritten ─────────────────────────────────

@testset "validate_and_augment_settings! idempotent" begin
    all_settings = ip_make_all_settings()
    all_settings["model_settings"]["out_names"] = ["fixed_name"]
    train_input  = Dict{String,TimeSeries}("stress_x" => ip_make_ts(3, 20, "stress_x"))
    train_target = Dict{String,TimeSeries}("surge"    => ip_make_ts(2, 20, "surge"))

    validate_and_augment_settings!(all_settings, train_input, train_target)

    @test all_settings["model_settings"]["out_names"] == ["fixed_name"]
end

# ── Error paths ───────────────────────────────────────────────────────────────

@testset "validate_and_augment_settings! errors" begin
    ts_in  = Dict{String,TimeSeries}("stress_x" => ip_make_ts(2, 10, "stress_x"))
    ts_out = Dict{String,TimeSeries}("surge"    => ip_make_ts(2, 10, "surge"))

    # missing model_settings
    s = Dict{String,Any}("run_info" => Dict("runid"=>"r"), "data_settings" => Dict("model_io"=>Dict("input"=>[],"target"=>[])))
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, ts_out)

    # missing model_name
    s = ip_make_all_settings()
    delete!(s["model_settings"], "model_name")
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, ts_out)

    # missing run_info
    s = ip_make_all_settings()
    delete!(s, "run_info")
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, ts_out)

    # missing model_io
    s = ip_make_all_settings()
    delete!(s["data_settings"], "model_io")
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, ts_out)

    # missing model_io "input" key
    s = ip_make_all_settings()
    delete!(s["data_settings"]["model_io"], "input")
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, ts_out)

    # empty train_target
    s = ip_make_all_settings()
    @test_throws ErrorException validate_and_augment_settings!(s, ts_in, Dict{String,TimeSeries}())
end
