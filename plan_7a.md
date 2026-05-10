# Plan: Step 7a — Formalise input checking and augmentation

## Goal

Replace the ad-hoc `get!` augmentation blocks in the four `train_*.jl` scripts with
a single library function in `src/input_processing.jl`. This is a prerequisite for
the generic `train.jl` (step 7c).

## Naming convention (decided)

All location counts use two canonical keys:

| Key | Meaning |
|---|---|
| `"nlocations_output"` | number of output locations (grid points, not necessarily measurement sites) |
| `"nlocations_input"`  | number of input locations (forcing grid points, not necessarily measurement sites) |

Old keys to be retired:

| Old key | Replacement | Models affected |
|---|---|---|
| `"nstations"` | `"nlocations_output"` | all four model families |
| `"nwind"`     | `"nlocations_input"`  | surge + wave only |

Interaction model has no `"nlocations_input"` (input and output share locations).
Tide model has no `"nlocations_input"` (inputs are computed from time, not loaded).
`"n_input_channels"` in `ConvWaveModel` is an architecture parameter — unchanged.

## New file: `src/input_processing.jl`

### Function signature

```julia
validate_and_augment_settings!(
    all_settings::Dict{String,Any},
    train_input::Dict{String,TimeSeries},
    train_target::Dict{String,TimeSeries},
)
```

Mutates `all_settings["model_settings"]` in-place. Returns nothing.

### Two-level validation contract

Validation is split across two layers; `input_processing.jl` only owns the
generic layer:

1. **Generic** (`validate_and_augment_settings!`) — everything that is true for
   all models regardless of type:
   - Structural checks: `model_name` present, `data_settings` has `model_io`
     with `input` and `target` keys, `run_info` has `runid`.
   - Derive `model_settings["model_dir"]` from `runid` + `model_name` if absent.
   - Augment `model_settings` with location metadata from data (see below).

2. **Per-model** (each model constructor) — model-specific required keys
   (`"freqs"` for tide, `"nlags"` for surge/wave/interaction, `"model_pars"`
   structure, etc.) remain the constructor's responsibility, as per the existing
   design convention in `design.md`. `input_processing.jl` never dispatches on
   model type.

### Augmentation logic

Populate missing keys in `model_settings` from the loaded data:

```julia
# Output-side (all models)
get!(model_settings, "out_quantities",     collect(keys(train_target)))
get!(model_settings, "out_names",          get_names(first_target))
get!(model_settings, "out_lons",           get_longitudes(first_target))
get!(model_settings, "out_lats",           get_latitudes(first_target))
get!(model_settings, "nlocations_output",  length(model_settings["out_names"]))

# Input-side (only when train_input is non-empty — tide model has no loaded inputs)
if !isempty(train_input)
    get!(model_settings, "in_quantities",    collect(keys(train_input)))
    get!(model_settings, "in_names",         get_names(first_input))
    get!(model_settings, "in_lons",          get_longitudes(first_input))
    get!(model_settings, "in_lats",          get_latitudes(first_input))
    get!(model_settings, "nlocations_input", length(model_settings["in_names"]))
end
```

The tide model's `model_io["input"]` variables are not loaded from files (they
are computed from time and coordinates), so `train_input` will be empty for tide
models and the `in_*` block is skipped automatically.

## Complete inventory of occurrences

### Source files — settings key reads (must change)

| File | Keys | Where |
|---|---|---|
| `src/models/AbstractSurgeModel.jl`      | `"nstations"`, `"nwind"` | `preprocess` |
| `src/models/LinearSurgeModel.jl`        | `"nstations"`, `"nwind"` | constructor |
| `src/models/ConvSurgeModel.jl`          | `"nstations"`, `"nwind"` | constructor |
| `src/models/AttentionSurgeModel.jl`     | `"nstations"`, `"nwind"` | constructor + `preprocess` |
| `src/models/AbstractWaveModel.jl`       | `"nstations"`, `"nwind"` | `preprocess` + `train_model!` (sets them) |
| `src/models/ConvWaveModel.jl`           | `"nstations"`, `"nwind"` | constructor |
| `src/models/DeepONetWaveModel.jl`       | `"nstations"`, `"nwind"` | constructor |
| `src/models/AbstractInteractionModel.jl`| `"nstations"`            | `preprocess` + `train_model!` (sets it) |
| `src/models/ConvInteractionModel.jl`    | `"nstations"`            | constructor |

Note: `AbstractWaveModel.train_model!` and `AbstractInteractionModel.train_model!`
currently *write* `"nstations"`/`"nwind"` into settings from data. This logic
moves to `validate_and_augment_settings!` and must be removed from `train_model!`.

### Test files — settings dicts (must change)

| File | Keys |
|---|---|
| `test/models/test_LinearSurgeModel.jl`    | `"nstations"`, `"nwind"` |
| `test/models/test_ConvSurgeModel.jl`      | `"nstations"`, `"nwind"` |
| `test/models/test_AttentionSurgeModel.jl` | `"nstations"`, `"nwind"` |
| `test/models/test_ConvWaveModel.jl`       | `"nstations"`, `"nwind"` (incl. `haskey` checks) |
| `test/models/test_DeepONetWaveModel.jl`   | `"nstations"`, `"nwind"` (incl. `haskey` checks) |
| `test/models/test_ConvInteractionModel.jl`| `"nstations"` |
| `test/test_train_surges.jl`               | `"nstations"`, `"nwind"` |
| `test/test_train_waves.jl`                | `"nstations"`, `"nwind"` |

### Training scripts (must change)

| File | Keys | Note |
|---|---|---|
| `train_surge.jl`       | `"nstations"`, `"nwind"` | already has TODO comments |
| `train_tide.jl`        | `"nstations"`            | already has TODO comment |
| `train_waves.jl`       | `"nstations"`, `"nwind"` | already has TODO comments |
| `train_interaction.jl` | `"nstations"`            | already has TODO comment |

These `get!` blocks are replaced wholesale by `validate_and_augment_settings!`.

### TOML config files (must change)

| File | Keys | Note |
|---|---|---|
| `test_data/config_files/settings_surgemodel.toml` | `nstations`, `nwind` | test fixture |
| `test_data/config_files/settings_wavemodel.toml`  | `nstations`, `nwind` | test fixture |
| `test_data/config_files/settings_tidemodel.toml`  | `nstations`          | test fixture |

### Documentation (must change)

| File | Note |
|---|---|
| `src/models/design.md`         | settings tables + tensor shape comments + code examples |
| `docs/settings.md`             | settings key tables for all model types |
| `docs/data_input_settings.md`  | shape comments using `nwind`/`nstations` |

### Files to leave unchanged

| File | Reason |
|---|---|
| `src/wave_stats.jl` + `score.md` | `nstations` is a DataFrame column name in statistics output, not a settings key |
| `wave_model_10to11_*/settings.toml` | old training output, not under active use |
| `run_settings_augmented.toml`       | generated file from a past run |

## Changes to source files

### `src/AIHydroPoints.jl`
Add `include("input_processing.jl")` and export `validate_and_augment_settings!`.

### `src/models/design.md`
Add a section documenting the naming convention and the two-level validation contract.

## Tests

Add tests in `test/test_input_processing.jl`:
- Happy path: augments correctly for a surge-style (input + target) case.
- Tide path: skips `in_*` keys when `train_input` is empty.
- Error path: missing `model_name` raises an error.
- Error path: missing `model_io` raises an error.
