ENV["GKSwstype"] = "nul"   # headless GR — no display needed

# Shared minimal TimeSeries for plot helper tests
let
    global _plot_ts_pred, _plot_ts_true
    ntimes  = 50
    times   = DateTime(2012,1,1):Hour(1):DateTime(2012,1,1)+Hour(ntimes-1)
    times   = collect(times)
    names   = ["StationA", "StationB"]
    lons    = [3.0, 4.0]; lats = [51.0, 52.0]
    pred_vals = randn(Float32, 2, ntimes)
    true_vals = pred_vals .+ 0.1f0 .* randn(Float32, 2, ntimes)
    _plot_ts_pred = Dict("surge" => TimeSeries(pred_vals, times, names, lons, lats, "surge", "test"))
    _plot_ts_true = Dict("surge" => TimeSeries(true_vals, times, names, lons, lats, "surge", "test"))
end

@testset "_plot_station_series" begin
    subdir = joinpath(temp_dir, "series_test")
    mkpath(subdir)
    AIHydroPoints._plot_station_series(_plot_ts_pred, _plot_ts_true, subdir)
    @test isfile(joinpath(subdir, "StationA.png"))
    @test isfile(joinpath(subdir, "StationB.png"))
end

@testset "_plot_station_fft" begin
    subdir = joinpath(temp_dir, "fft_test")
    mkpath(subdir)
    AIHydroPoints._plot_station_fft(_plot_ts_pred, _plot_ts_true, subdir)
    @test isfile(joinpath(subdir, "StationA.png"))
    @test isfile(joinpath(subdir, "StationB.png"))
end

@testset "timerange filtering" begin
    subdir = joinpath(temp_dir, "timerange_test")
    mkpath(subdir)
    # restrict to first 10 hours — files should still be produced
    t0 = Dates.format(DateTime(2012,1,1),        "yyyy-mm-ddTHH:MM:SS")
    t1 = Dates.format(DateTime(2012,1,1)+Hour(9), "yyyy-mm-ddTHH:MM:SS")
    AIHydroPoints._plot_station_series(_plot_ts_pred, _plot_ts_true, subdir;
                                       timerange=[t0, t1])
    @test isfile(joinpath(subdir, "StationA.png"))
end

@testset "_plot_station_scatter" begin
    subdir = joinpath(temp_dir, "scatter_test")
    mkpath(subdir)
    AIHydroPoints._plot_station_scatter(_plot_ts_pred, _plot_ts_true, subdir)
    @test isfile(joinpath(subdir, "StationA.png"))
    @test isfile(joinpath(subdir, "StationB.png"))
end

@testset "_plot_station_scatter with fit + qq overlays" begin
    subdir = joinpath(temp_dir, "scatter_overlay_test")
    mkpath(subdir)
    AIHydroPoints._plot_station_scatter(_plot_ts_pred, _plot_ts_true, subdir;
                                        add_fit=true, add_qq=true)
    @test isfile(joinpath(subdir, "StationA.png"))
    @test isfile(joinpath(subdir, "StationB.png"))
end

@testset "_write_station_stats" begin
    path = joinpath(temp_dir, "stats_test.csv")
    AIHydroPoints._write_station_stats(_plot_ts_pred, _plot_ts_true, path)
    @test isfile(path)
    lines = readlines(path)
    @test length(lines) == 3          # header + 2 stations
    @test startswith(lines[1], "location_id")
    @test occursin("StationA", lines[2])
    @test occursin("StationB", lines[3])
end

@testset "_write_station_series" begin
    @testset "netcdf" begin
        AIHydroPoints._write_station_series(_plot_ts_pred, _plot_ts_true,
                                            temp_dir, "test", "netcdf")
        @test isfile(joinpath(temp_dir, "series_test.nc"))
    end

    @testset "jld2" begin
        AIHydroPoints._write_station_series(_plot_ts_pred, _plot_ts_true,
                                            temp_dir, "test", "jld2")
        @test isfile(joinpath(temp_dir, "series_test.jld2"))
    end

    @testset "noos" begin
        AIHydroPoints._write_station_series(_plot_ts_pred, _plot_ts_true,
                                            temp_dir, "test", "noos")
        subdir = joinpath(temp_dir, "series_test")
        @test isdir(subdir)
        @test isfile(joinpath(subdir, "StationA.noos"))
        @test isfile(joinpath(subdir, "StationB.noos"))
    end

    @testset "overwrite" begin
        # second call should not error (overwrites existing files)
        AIHydroPoints._write_station_series(_plot_ts_pred, _plot_ts_true,
                                            temp_dir, "test", "netcdf")
        @test isfile(joinpath(temp_dir, "series_test.nc"))
    end

    @testset "unknown format" begin
        @test_throws ErrorException AIHydroPoints._write_station_series(
            _plot_ts_pred, _plot_ts_true, temp_dir, "test", "csv")
    end
end

@testset "save_loss_plot" begin
    train_losses = [1.0, 0.8, 0.6]
    val_losses   = [1.1, 0.9, 0.7]
    path = joinpath(temp_dir, "losses.png")

    save_loss_plot(path, train_losses, val_losses)
    @test isfile(path)

    # error if file exists and overwrite=false (default)
    @test_throws ErrorException save_loss_plot(path, train_losses, val_losses)

    # overwrite=true replaces the file
    save_loss_plot(path, train_losses, val_losses; overwrite=true)
    @test isfile(path)

    # error if parent directory does not exist
    bad_path = joinpath(temp_dir, "nonexistent_dir", "losses.png")
    @test_throws ErrorException save_loss_plot(bad_path, train_losses)

    # works without val_losses
    path2 = joinpath(temp_dir, "losses_train_only.png")
    save_loss_plot(path2, train_losses)
    @test isfile(path2)
end

@testset "_check_and_align_locations" begin
    ntimes = 10
    times  = collect(DateTime(2020,1,1) .+ Hour.(0:ntimes-1))
    vals_ab = ones(Float32, 2, ntimes)
    vals_abc = ones(Float32, 3, ntimes)

    ts_ab  = TimeSeries(vals_ab,  times, ["A","B"],   [1.0,2.0], [51.0,52.0], "q", "test")
    ts_ba  = TimeSeries(vals_ab,  times, ["B","A"],   [2.0,1.0], [52.0,51.0], "q", "test")
    ts_abc = TimeSeries(vals_abc, times, ["A","B","C"],[1.0,2.0,3.0],[51.0,52.0,53.0],"q","test")

    # correct order: no-op
    aligned = AIHydroPoints._check_and_align_locations(ts_ab, ["A","B"], "test")
    @test get_names(aligned) == ["A","B"]

    # wrong order: reorders to expected
    aligned = AIHydroPoints._check_and_align_locations(ts_ba, ["A","B"], "test")
    @test get_names(aligned) == ["A","B"]

    # extra locations: silently dropped
    aligned = AIHydroPoints._check_and_align_locations(ts_abc, ["A","B"], "test")
    @test get_names(aligned) == ["A","B"]

    # missing location: readable error
    err = @test_throws ErrorException AIHydroPoints._check_and_align_locations(
        ts_ab, ["A","B","C"], "test_label")
    @test occursin("test_label", err.value.msg)
    @test occursin("missing", err.value.msg)

    # plot helpers accept reordered target without error
    subdir = joinpath(temp_dir, "align_series")
    mkpath(subdir)
    pred_vals = randn(Float32, 2, ntimes)
    true_vals = randn(Float32, 2, ntimes)
    ts_pred = Dict("surge" => TimeSeries(pred_vals, times, ["A","B"], [1.0,2.0], [51.0,52.0], "surge", "p"))
    ts_true_rev = Dict("surge" => TimeSeries(true_vals, times, ["B","A"], [2.0,1.0], [52.0,51.0], "surge", "t"))
    AIHydroPoints._plot_station_series(ts_pred, ts_true_rev, subdir)
    @test isfile(joinpath(subdir, "A.png"))
end
