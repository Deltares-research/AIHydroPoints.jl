
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
- 406 unit tests pass (`pixi run julia --project -e "using Pkg; Pkg.test()"`)
- All training scripts smoke-tested end-to-end via `check_training_scripts.sh`

## Intended workflow for training

```bash
# Tides
julia --project train_tide.jl

# Surge
julia --project train_surge.jl

# Waves
julia --project train_waves.jl

# Tide–surge interaction
julia --project train_interaction.jl
```

Each script has `nepochs = 2` for a fast smoke-test; increase for a real run.

## Data downloads

We aim to have the main datasets available in Zarr format in the cloud.
For now the scripts download data and save it locally.
For S3 storage you need `~/.aws/credentials` and `~/.aws/config`.

## Source code (`src/`)

All library code lives in `src/` and is exposed as the `AIHydroPoints` Julia package.

### Model hierarchy

| File | Contents |
|---|---|
| `models/abstract_model.jl` | `AbstractModel` — common interface (`predict`, `train_model!`, `save_params`, `load_params!`, `plot_series`) |
| `models/abstract_flux_model.jl` | `AbstractFluxModel` — generic `predict`/`save_params`/`load_params!` via `preprocess`/`forward`/`postprocess!` |
| `models/training_settings.jl` | `TrainingSettings` — epochs, learning rate, batch size, validation split, etc. |
| `models/AbstractSurgeModel.jl` | Shared surge `preprocess` (wind-stress lags), `postprocess!`, `train_model!`, `plot_series` |
| `models/LinearSurgeModel.jl` | Single Dense layer surge baseline |
| `models/ConvSurgeModel.jl` | Conv1D over lag dimension |
| `models/AttentionSurgeModel.jl` | Transformer branch + dense trunk + graph adjacency |
| `models/AbstractTideModel.jl` | Shared tide `preprocess` (Doodson phases), `postprocess!`, `train_model!`, `plot_series` |
| `models/DeepONetTideModel.jl` | DeepONet branch/trunk architecture |
| `models/ProductTideModel.jl` | Station×Doodson product + residual gating |
| `models/AbstractWaveModel.jl` | Shared wave `preprocess` (one-hot station + lagged wind-stress), `postprocess!`, `train_model!`, `plot_series` |
| `models/ConvWaveModel.jl` | `WaveInputLayer` exp gate + strided Conv1D |
| `models/DeepONetWaveModel.jl` | Conv branch + dot-product station merge |
| `models/AbstractInteractionModel.jl` | Shared interaction `preprocess` (one-hot + tide/surge lags, Z-score), `postprocess!`, `train_model!`, `plot_series` |
| `models/ConvInteractionModel.jl` | `InteractionInputLayer` gate + strided Conv1D |

### Utilities

| File | Contents |
|---|---|
| `tidal_comps.jl` | Doodson phases, lunar-to-solar conversion, named tidal constituents |
| `wind_stress.jl` | Convert 10 m winds to stress components |
| `wave_stats.jl` | `stats_skipnan`, `average_stats` — per-station wave statistics |
| `graph_network.jl` | Graph network building blocks |
| `attention.jl` | Transformer / attention building blocks |
| `training.jl` | Legacy training loop for old `AbstractModelSettings`-based models |
| `toml_utils.jl` | `toml_write` — save model settings dict as TOML |
| `plot_utils.jl` | `save_loss_plot`, `_plot_station_series`, `plot_fft` |

Time-series I/O (NetCDF, Zarr, JLD2, NOOS) is provided by
[MultiTimeSeries.jl](https://github.com/robot144/MultiTimeSeries.jl).

See `docs/settings.md` for a full reference of all settings fields.

## Analysis scripts

- `analyse_tides_schureman.jl` — Harmonic tidal analysis (Schureman, 95 constituents) on a
  yearly DCSM-FM 5-station JLD2 dataset.  Produces tides and surge as NetCDF, per-station
  plots, statistics CSV, and constituent CSV.  Takes the year as a command-line argument
  (default 2010):
  ```
  julia --project analyse_tides_schureman.jl 2011
  ```
  Output is written to `output_tides_<year>/`.

## Other

- `test_minio_zarr_with_julia.ipynb` — test script for downloading a subset of the 1980–2023 DCSM run
- `hatyan_core.py` — copy of basic tide routines from Hatyan2

## Design

- All settings are plain `Dict{String,Any}`.  Training-only fields live in `TrainingSettings`.
- A trained model's inference settings are saved as a flat `settings.toml`; model weights are saved as `model_params.jld2`.
- Long-term goal: generic `ai_hydro_train.jl` / `ai_hydro_predict.jl` fully driven by a config TOML.

## Statistics and hyperparameters

### Tide model

- tide layers: 3 — channels per layer: 64 — regularization: 0.0001
- batch size: 1024 — stations: 314 (all) — epochs: 20
- train period: 2008–2010 — test period: 2011
- mean RMSE train: 0.216 — mean RMSE test: 0.230
