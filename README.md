
# A series approach to forecasting tides and storm-surges

The purpose of this time-series based AI model is to have a simple and fast benchmark model.
The inputs are time-series of winds, air-pressure, tides, and surge from ERA5 / DCSM-FM.
The outputs are time-series of tides, surge, waves, and their interaction at a number of locations.

The AI model contains four modules:

- **Tides** — takes time and location identity as input; trained on multi-year tidal records.
- **Surge** — takes lagged wind stress and pressure at a few locations as input.
- **Waves** — takes lagged wind speed and direction at input locations; encodes output stations via one-hot vectors.
- **Tide–surge interaction** — takes lagged tide and surge at each station as input and predicts the non-linear residual.

Summing all four outputs gives the total water level.  The architecture captures both local and generic dynamics: a one-hot station vector selects station-specific parameters while shared layers encode the common physics.

The model has no internal state, which makes it equally suitable for hindcasts (reanalysis winds) and forecasts (NWP winds).

## Status

All four models (tides, surges, waves, interaction) are fully implemented and tested.

- Model hierarchy: `AbstractModel → AbstractFluxModel → Abstract{Surge,Tide,Wave,Interaction}Model → concrete models`
- Settings split: model architecture lives in a `Dict{String,Any}`; training hyperparameters live in `TrainingSettings` — see `docs/settings.md`
- 530 unit tests pass (`pixi run julia --project -e "using Pkg; Pkg.test()"`)
- All training and prediction scripts smoke-tested end-to-end via `check_training_scripts.sh`

## Workflow

Training and inference are fully driven by TOML config files.  Example configs live in `examples/`.

```bash
# Train a model
pixi run julia --project scripts/train.jl examples/ConvSurgeModel.toml

# Run inference with a trained model
pixi run julia --project scripts/predict.jl examples/predict_ConvSurgeModel.toml
```

Each example TOML has `nepochs = 5` (or similar) for a fast smoke-test; increase for a real run.

To smoke-test all examples end-to-end:

```bash
bash check_training_scripts.sh
```

## Data downloads

We aim to have the main datasets available in Zarr format in the cloud.
For now the scripts download data and save it locally.
For S3 storage you need `~/.aws/credentials` and `~/.aws/config`.

## Source code (`src/`)

All library code lives in `src/` and is exposed as the `AIHydroPoints` Julia package.
The top-level functions `train(toml)` and `predict(toml)` are exported and can be called
directly after `using AIHydroPoints`.

### Model hierarchy

| File | Contents |
|---|---|
| `models/abstract_model.jl` | `AbstractModel` — common interface (`predict`, `train_model!`, `save_params`, `load_params!`, `write_outputs`) |
| `models/abstract_flux_model.jl` | `AbstractFluxModel` — generic `predict`/`save_params`/`load_params!`/`write_outputs` via `preprocess`/`forward`/`postprocess!` |
| `models/training_settings.jl` | `TrainingSettings` — epochs, learning rate, batch size, validation split, checkpoints, etc. |
| `models/AbstractSurgeModel.jl` | Shared surge `preprocess` (wind-stress lags), `postprocess!`, `train_model!` |
| `models/LinearSurgeModel.jl` | Single Dense layer surge baseline |
| `models/ConvSurgeModel.jl` | Conv1D over lag dimension |
| `models/AttentionSurgeModel.jl` | Transformer branch + dense trunk + graph adjacency |
| `models/AbstractTideModel.jl` | Shared tide `preprocess` (Doodson phases), `postprocess!`, `train_model!` |
| `models/DeepONetTideModel.jl` | DeepONet branch/trunk architecture |
| `models/ProductTideModel.jl` | Station×Doodson product + residual gating |
| `models/AbstractWaveModel.jl` | Shared wave `preprocess` (one-hot station + lagged wind-stress), `postprocess!`, `train_model!` |
| `models/ConvWaveModel.jl` | `WaveInputLayer` exp gate + strided Conv1D |
| `models/DeepONetWaveModel.jl` | Conv branch + dot-product station merge |
| `models/AbstractInteractionModel.jl` | Shared interaction `preprocess` (one-hot + tide/surge lags, Z-score), `postprocess!`, `train_model!` |
| `models/ConvInteractionModel.jl` | `InteractionInputLayer` gate + strided Conv1D |

### Utilities

| File | Contents |
|---|---|
| `train.jl` | `train(toml)` — full training pipeline: load → augment → create → train → save → outputs |
| `predict.jl` | `predict(toml)` — inference pipeline: load weights → run → outputs |
| `input_processing.jl` | `validate_and_augment_settings!`, `MODEL_REGISTRY`, `create_model` |
| `data_loading.jl` | `load_data` — loads any split configuration from TOML data settings |
| `tidal_comps.jl` | Doodson phases, lunar-to-solar conversion, named tidal constituents |
| `wind_stress.jl` | Convert 10 m winds to stress components |
| `wave_stats.jl` | `stats_skipnan`, `average_stats` — per-station wave statistics |
| `graph_network.jl` | Graph network building blocks |
| `attention.jl` | Transformer / attention building blocks |
| `toml_utils.jl` | `toml_write` — save/load settings dicts as TOML |
| `plot_utils.jl` | Output helpers: `save_loss_plot`, time-series, FFT, scatter, stats, tidal analysis plots |

Time-series I/O (NetCDF, Zarr, JLD2, NOOS) is provided by
[MultiTimeSeries.jl](https://github.com/robot144/MultiTimeSeries.jl).

See `docs/settings.md` for a full reference of all settings fields.
See `docs/output_settings.md` for output configuration.
See `docs/data_input_settings.md` for data loading configuration.

## Analysis scripts

- `analyse_tides_schureman.jl` — Harmonic tidal analysis (Schureman, 95 constituents) on a
  yearly DCSM-FM 5-station JLD2 dataset.  Produces tides and surge as NetCDF, per-station
  plots, statistics CSV, and constituent CSV.  Takes the year as a command-line argument
  (default 2010):
  ```
  julia --project analyse_tides_schureman.jl 2011
  ```
  Output is written to `output_tides_<year>/`.

## Design

- All settings are plain `Dict{String,Any}`.  Training-only fields live in `TrainingSettings`.
- A trained model's inference settings are saved as `model_settings.toml`; model weights are
  saved as `params.jld2` (final epoch) and `params_best.jld2` (best validation loss, when
  validation data is provided).  The `model_weights` key in `model_settings.toml` records
  which file to load for inference (defaults to `params_best.jld2` when available).
- Continuing training from saved weights is automatic: if the weights file already exists in
  `model_dir` when `train()` is called, it is loaded before training begins.
- Output is fully configurable via `[output_settings]` in the TOML — see `docs/output_settings.md`.
