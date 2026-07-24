# test_train_waves.jl
#
# Smoke-test for the wave model training pipeline using the new ConvWaveModel interface.
# Uses the small 2021 test dataset in test_data/waves_2021/.

using Dates
using DataFrames

@testset "wave model training pipeline" begin

    data_folder = joinpath(@__DIR__, "..", "test_data", "waves_2021")

    # ── Load data ──────────────────────────────────────────────────────────
    series_collection = NoosTimeSeriesCollection(data_folder)
    @test !isempty(get_sources(series_collection))
    @test !isempty(get_quantities(series_collection))

    u10  = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_speed")
    udir = get_series_from_collection(series_collection, "knmi_harmonie40_wind", "wind_direction")
    swh  = get_series_from_collection(series_collection, "swan_dcsm_harmonie",   "wave_height")

    output_locations = get_names(swh)
    input_locations  = get_names(u10)
    @test length(output_locations) >= 1
    @test length(input_locations)  >= 1

    time_selection = DateTime(2021,1,1):Hour(1):DateTime(2021,12,31,23)
    u10  = select_timerange_with_fill(u10,  time_selection, fill_value=0.0f0)
    udir = select_timerange_with_fill(udir, time_selection, fill_value=0.0f0)
    swh  = select_timerange_with_fill(swh,  time_selection, fill_value=0.0f0)

    @test length(get_times(u10)) == length(time_selection)

    # ── Train / test split ─────────────────────────────────────────────────
    t_split = DateTime(2021, 10, 1)

    train_dict = Dict(
        "wind_speed"     => select_timespan(u10,  time_selection[1], t_split),
        "wind_direction" => select_timespan(udir, time_selection[1], t_split),
        "wave_height"    => select_timespan(swh,  time_selection[1], t_split),
    )
    test_dict = Dict(
        "wind_speed"     => select_timespan(u10,  t_split, time_selection[end]),
        "wind_direction" => select_timespan(udir, t_split, time_selection[end]),
        "wave_height"    => select_timespan(swh,  t_split, time_selection[end]),
    )

    # nlags=4 → nchannel length must be 2 (2^2=4)
    nstations = length(output_locations)
    nwind     = length(input_locations)

    model_dir = joinpath(temp_dir, "test_wave_model")
    mkpath(model_dir)

    settings = Dict{String, Any}(
        "model_name"        => "test_wave_model",
        "model_dir"         => model_dir,
        "nlocations_output"         => nstations,
        "nlocations_input"             => nwind,
        "nlags"             => 4,
        "wind_scale"        => 0.5,
        "wave_scale"        => 3.0,
        "n_input_channels"  => 4,
        "model_pars"        => Dict{String,Any}("nchannel" => [4, 1], "activation" => "swish"),
    )
    train_settings = TrainingSettings(nepochs=2, batch_size=8, learning_rate=1e-3)

    # ── Model creation ─────────────────────────────────────────────────────
    model = ConvWaveModel(settings)
    @test model isa AbstractWaveModel

    # ── Training ───────────────────────────────────────────────────────────
    train_losses, val_losses = train_model!(model, train_settings, train_dict, train_dict)
    @test length(train_losses) == train_settings.nepochs
    @test all(isfinite, train_losses)
    @test isempty(val_losses)

    # ── Prediction ─────────────────────────────────────────────────────────
    output = predict(model, test_dict)
    @test output isa Dict{String, TimeSeries}
    @test haskey(output, "wave_height")
    pred_values = get_values(output["wave_height"])
    @test size(pred_values, 1) == nstations
    @test any(isfinite, pred_values)

    # ── Statistics ─────────────────────────────────────────────────────────
    # Align times: predict trims the first nlags-1 steps
    swh_trimmed = select_timespan(test_dict["wave_height"],
                                  get_times(output["wave_height"])[1],
                                  get_times(output["wave_height"])[end])
    stats = stats_skipnan(swh_trimmed, output["wave_height"])
    @test nrow(stats) == nstations
    @test all(isfinite, stats.rmse)
    @test all(stats.count .> 0)

end
