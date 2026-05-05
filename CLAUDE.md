# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Julia is managed via [pixi](https://pixi.sh) — prefix all `julia` commands with `pixi run`:

**Run all tests:**
```bash
pixi run julia --project -e "using Pkg; Pkg.test()"
```

**Run a single test file:**
```bash
pixi run julia --project -e 'include("test/models/test_LinearSurgeModel.jl")'
```

**Start a Julia REPL with the project loaded:**
```bash
pixi run julia --project
```

**Install/update dependencies:**
```julia
# Inside REPL
using Pkg; Pkg.instantiate()
```

**Smoke-test all training scripts:**
```bash
bash check_training_scripts.sh
```

Julia version: `>=1.12.5, <1.13` (pinned in `pixi.toml`).

## Architecture

AIHydroPoints is a Julia package for ML-based time-series forecasting of oceanographic phenomena (tides, storm surge, waves) using neural networks (Flux.jl). Data flows as `Dict{String,TimeSeries}` where keys are variable names (`"wind_x"`, `"surge"`, etc.) and values are `TimeSeries` from MultiTimeSeries.jl (`values::Matrix{Float32}` with shape `(locations, times)`).

### Model Type Hierarchy

```
AbstractModel                    # src/models/abstract_model.jl
└── AbstractFluxModel            # src/models/abstract_flux_model.jl
        └── LinearSurgeModel     # src/models/LinearSurgeModel.jl
```

The older models (`TideSettings`, `SurgeSettings`, `WaveSettings` in `src/tides.jl`, `src/surge.jl`, `src/waves.jl`) use typed `AbstractModelSettings` structs and are being progressively migrated to the new design.

### AbstractModel Interface

All models must implement:
- `predict(m, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}`
- `get_settings(m) -> Dict{String,Any}`
- `save_params(m, file::String)` / `load_params!(m, file::String)` — JLD2 serialization
- `train_model!(m, train_settings, input, target)` — mutates model in-place

Unimplemented methods throw descriptive `ErrorException`s (not `MethodError`s).

### AbstractFluxModel Customization Points

`AbstractFluxModel` provides generic implementations of `predict`, `save_params`, and `load_params!`. Domain models inherit these by implementing three customization points:
- `preprocess(m, input) -> (Array{Float32,4}, Dict{String,TimeSeries})` — build input tensor + pre-allocate output dict
- `forward(m, x) -> Array{Float32,3}` — Flux forward pass
- `postprocess!(output, m, y)` — fill output dict in-place
- `get_flux_model(m)` — return the Flux chain
- `get_settings(m)` — return `Dict{String,Any}`

### Tensor Layout Convention

- Input: `(1, features*locations, nlags, ntimes)` — 4D, time is the batch dimension
- Output: `(locations, 1, ntimes)` — 3D

### Settings Convention

New models use `Dict{String,Any}` for settings (not typed structs). Required keys are validated at construction time. Metadata like `"out_names"`, `"out_lons"`, `"out_lats"`, `"out_quantity"` is populated during `train_model!` from the target `TimeSeries`.

Training hyperparameters are kept separate in `TrainingSettings` (`src/models/training_settings.jl`): `nepochs`, `nbatches`, `learning_rate`, `lr_decay_factor`, `patience`, etc.

### After each change

- Update `docs/` (e.g. `docs/settings.md`) when the public API or settings change.
- Check `README.md` is still accurate.
- Update `plan.md` status checkboxes.

### Known Issues

`new_train_LinearSurgeModel_issues.md` documents active design gaps:
1. `save_settings`/`load_settings` only dispatch on old `AbstractModelSettings` structs — not yet implemented for `Dict{String,Any}`.
2. `plot_losses` and `plot_series` in `src/training.jl` don't accept `AbstractFluxModel` instances.
