# test_train_waves.jl
#
# Smoke-test for the wave model training pipeline.
# Uses the small 2021 test dataset in test_data/waves_2021/.
# The goal is fast feedback that the pipeline runs end-to-end,
# NOT that the resulting model is accurate.

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

    # ── Train / validation split ───────────────────────────────────────────
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

    # ── Minimal settings for speed ─────────────────────────────────────────
    # nlags must equal 2^length(nchannel): 4 = 2^2
    mktempdir() do model_dir
        settings = WaveSettings(
            model_name       = "test_wave_model",
            model_dir        = model_dir,
            nepochs          = 2,
            nbatches         = 8,
            learning_rate    = 0.001,
            weight_reg       = 1.0e-4,
            use_gpu          = false,
            nstations        = length(output_locations),
            nwind            = length(input_locations),
            nlags            = 4,
            n_input_channels = 4,
            wind_scale       = 0.5,
            wave_scale       = 3.0,
            input_noise_std  = 0.0,
            model_pars       = Dict("nchannel" => [4, 1], "activation" => "swish"),
        )

        # ── Model creation ─────────────────────────────────────────────────
        model = create_wave_model(settings)
        @test !isnothing(model)

        # ── Training ───────────────────────────────────────────────────────
        model, acc_losses, train_losses, test_losses =
            train_model(model, settings, train_dict, test_dict)
        @test length(train_losses) == settings.nepochs
        @test length(test_losses)  == settings.nepochs
        @test all(isfinite, train_losses)

        # ── Prediction ─────────────────────────────────────────────────────
        full_dict = Dict("wind_speed" => u10, "wind_direction" => udir, "wave_height" => swh)
        swh_predicted = predict(model, settings, full_dict)

        pred_values = get_values(swh_predicted)
        @test size(pred_values, 1) == length(output_locations)
        @test size(pred_values, 2) == length(time_selection)
        # After the initial NaN-filled lags, values should be finite
        @test any(isfinite, pred_values)

        # ── Statistics ─────────────────────────────────────────────────────
        stats = stats_skipnan(swh, swh_predicted)
        @test nrow(stats) == length(output_locations)
        @test all(isfinite, stats.rmse)
        @test all(stats.count .> 0)
    end

end
