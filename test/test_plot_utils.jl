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

@testset "_plot_station_scatter" begin
    subdir = joinpath(temp_dir, "scatter_test")
    mkpath(subdir)
    AIHydroPoints._plot_station_scatter(_plot_ts_pred, _plot_ts_true, subdir)
    @test isfile(joinpath(subdir, "StationA.png"))
    @test isfile(joinpath(subdir, "StationB.png"))
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
