
# Next steps for working on this project

## Main goal
The main goal of this project is to develop a machine learning model for predicting tides, surges and their interaction. The model will be trained on historical data and will be able to make predictions for future events. The model will be implemented in Julia and will be designed to be easily extensible and adaptable to different datasets and configurations

## Next steps
1. [x] Make the unit tests work again.
2. [x] Create a small test dataset
3. [x] Create unit tests for each of the models (tides, surges, interaction, waves) that run fast.
4. [x] Create additional tests for code that is not tested at all.
    a. [x] figure out which code is not tested at all, and which code is tested but not well enough.
    b. [x] write tests for the code that is not tested at all, and improve the tests for the code that is tested but not well enough.
5. [x] Clean up the code and make it more modular and reusable.
6. [x] Make training scripts more similar and consistent — all four scripts use `load_data`, a shared `train_model!` interface, unified plotting, `runid`/`description` metadata, and write `run_settings.toml` for reproducibility.
7. Make training fully controllable from the toml input file, and make it possible to run training from the command line with a specified toml file.
    a. [x] Formalize input checking and augmentation. 
    b. [x] Add a `MODEL_REGISTRY` and `get_model_type(settings)` that converts `settings["model_name"]` to a Julia type once, enabling type dispatch everywhere instead of if/elseif on strings. Includes `validate_model_settings!` hook (default no-op) called from `validate_and_augment_settings!`, and `create_model` factory dispatching on model type.
    c. [x] Add a `create_model(settings, train_input)` factory function in the library that dispatches on `settings["model_name"]`.
         - `AttentionSurgeModel` needs a `GraphNetwork` built from `train_input`; the factory builds it automatically.
         - The interaction script's `_synthesize_waterlevel` should be removed: instead load the surge file directly as the target variable, renamed to `"interaction"` via the `"as"` alias in `data_settings`. No special cases needed in the generic script. Note: this is a temporary placeholder — a meaningful interaction target (residual = observed − tide_pred − surge_pred) requires a well-trained surge model and observed waterlevel data, which are not yet available.
    d. [x] Create a generic `train.jl` that reads all settings from a TOML file passed as `ARGS[1]` and runs the shared skeleton: load data → augment settings → create model → train → save → plot.
    e. [x] Create example TOML files (one per model type) in `examples/`. All 8 smoke-test via `train.jl` in `check_training_scripts.sh`. Fixed `AbstractInteractionModel` hardcoded `"waterlevel"` key to use `out_quantities` from settings.
    f. [x] Clean up dead code: removed duplicate `model_type =` assignments; replaced if/elseif model construction blocks with `create_model`; removed redundant `test_output = predict(...)` lines; refactored `train_interaction.jl` to load surge as `"interaction"` target instead of synthesising it.
    g. [x] Replace `plot_series` with `write_outputs(model, data, output_settings)`: moves all output logic into the model, controlled by `[output_settings]` TOML section (`plot_train`, `plot_test`, `plot_fft`). Default: test only. Defaults are filled into `all_settings` by `validate_and_augment_settings!` so they appear in `run_settings.toml`.
    h. [x] Move pipeline logic into `src/train.jl` (`train(toml)`) and `src/predict.jl` (`predict(toml)`), exported from the package. Root `train.jl` and `predict.jl` become thin CLI wrappers. Users can call `train`/`predict` directly after `using AIHydroPoints`.
    i. [x] Improve documentation of settings in `docs/settings.md` and `docs/data_input_settings.md`: clarified train vs predict required tables, moved `model_name` to required keys, fixed stale code examples, resolved TBD on path resolution.
8. [x] Write a separate script for inference: `predict.jl` mirrors `train.jl` — reads a TOML with `[model_settings]` (model_dir), `[data_settings]`, and `[output_settings]`; loads trained weights and runs `write_outputs`. Example TOML at `examples/predict_ConvSurgeModel.toml`. Smoke-tested via `check_training_scripts.sh`.
9. [x] Improve output during training — all output types (timeseries, scatter, fft, stats, series, tidal analysis, summary) work generically for all model types via `write_outputs`; explicit validation splits; location alignment at inference; explicit `model_weights` key; `params_best.jld2` and epoch checkpoints. See `docs/output_settings.md`.
10. [x] Create leaderboard — `src/leaderboard.jl` (find_run_dirs, load_leaderboard, sort_leaderboard); `experiments/leaderboard.ipynb` (ranked table, CSV, PNG); `experiments/leaderboard.qmd` + `render_leaderboard.sh` (Quarto HTML via engine: julia). Per-station stats CSV excluded due to inconsistent schemas across model families.
11. Create a baseline for each model type
    a. [x] Check difference between old and new JLD2 formats before generating data
    b. [x] surge baselines 1yr, 5yr 20yr (determine timespans)
    c. [ ] interaction datasets, first testing
        - Issue: ConvInteractionModel (simplified Conv1D, no station gate) does not learn. Data is hourly so nlags=16 = 16h covers a full tidal cycle — nlags is not the problem. Root cause unknown; needs further investigation.
    d. [ ] interaction baselines 1yr, 5yr 20yr (determine timespans)
    e. [x] tide baselines
    f. [ ] improve scatter plot
12. Create script for real-time forecasts
13. Create an environment for online demos
14. Add experiments for waves
15. Scale surge model to a large number of stations
16. Try to remove separate training loop for the AttentionSurgeModel
17. Create a presentation
    a. [x] create first draft
    b. [x] update methods, results and refine
18. Improve documentation
    a. [x] start quarto template
    b. [x] test publication on github
    c. [x] Expand `docs/notation.md` to support `docs/background.md`: dense + ReLU
       walkthrough with Flux & PyTorch side-by-side; column-major vs row-major
       intermezzo; index convention table (capital = output, lowercase = input);
       generic convolution notation with kernel offset `Δi` and per-direction size
       `N_i, K_i → N_{Δi}`; summary table of three conv variants with parameter
       counts; `$\star$` reduced to "apply weights" (subscripts moved into the
       index pattern of `W`).
    d. [ ] Extend `notation.md` to cover the *realistic* 4-axis input/output
       (point, quantity, time-lag, batch-time) once the per-model layout
       refactor (step 20) lands — see `docs/notes_dimensions.md`.
19. Add dvc for data storage
20. Tensor-layout review and refactor (working notes in `docs/notes_dimensions.md`)
    a. [x] Survey actual tensor shapes used in each model (preprocess vs.
       layer-ready) — Notes 1–4.
    b. [x] Identify the design issue: a "unified" abstract input tensor
       layout is an illusion; only data extraction is genuinely shared,
       the tensor arrangement is layer-specific — Note 5.
    c. [x] Decide call-signature convention: every Flux model gets a
       `(m::ModelFlux)(x::Tuple) = m(x...)` wrapper so `forward` can call
       `flux_model(x)` uniformly — Note 6.
    d. [x] Decide output convention: drop the trailing
       `reshape(y, size(y, 1), 1, ntimes)` placeholder; each `forward`
       returns its natural 2D `(nstations, ntimes)` shape; `postprocess!`
       consumes 2D directly — Note 7.
    e. [ ] Fix the `ConvSurgeModel.jl:70` reshape bug: `reshape(x, nlags,
       n_in, size(x,2))` doesn't permute memory and silently scrambles
       channel/lag positions whenever `nlags ≠ 3·nwind`. Verify with a
       synthetic test, then fix via per-model `preprocess` (Note 5
       recommendation) — not via `permutedims` in `forward`.
    f. [ ] Refactor `preprocess` along the principle in the conclusion of
       `notes_dimensions.md`: shared data-extraction helper +
       per-model assembly step. Update `LinearSurgeModel`, `ConvSurgeModel`,
       `AttentionSurgeModel` accordingly; verify wave and interaction
       models too.
    g. [ ] Update `src/models/design.md` to reflect the new convention
       (each model declares its own input/output tensor layout; the
       abstract pipeline only standardises the `Dict{String,TimeSeries}`
       boundary).

## Checklist for each step:
- all source should eventually be in src/ and all tests should be in test/ and test data should be in test_data/
- make code compilable and runnable
- consider to add new unit tests for the new code
- fix all unit tests `pixi run julia --project -e "using Pkg; Pkg.test()"`
- make sure that the code is well documented and that the documentation is up to date.
- update docs/ (e.g. docs/settings.md) when the public API or settings change.
- Check if README.md is up to date and update it if necessary.
- Adapt the status in plan.md
- run the unit tests and make sure that they all pass.
- run check_training_scripts.sh to smoke-test all training scripts

## General notes
- we use pixi to install python and julia etc. But we use julia packages in Project.toml.
- we think before we write code. We raise potential issues with the user before continuing
- we discuss design decisions with the user before implementing them, and we make sure that the user is happy with the design before proceeding.
- we keep output of unit tests in test/temp. We clean this folder before running the tests, and leave files for inspection after the tests have run. We make sure that the output of the tests is informative and useful for debugging.

## Status

Steps 1–6 are complete. The new model hierarchy (`AbstractModel → AbstractFluxModel →
AbstractSurgeModel / AbstractTideModel / AbstractWaveModel / AbstractInteractionModel →`
concrete models) is fully implemented, tested, and all legacy source files removed.
406 unit tests pass. Training scripts: `train_surge.jl`, `train_tide.jl`,
`train_waves.jl`, `train_interaction.jl`. All use `load_data`, a shared `train_model!`
interface, unified plotting, `runid`/`description` metadata, and write `run_settings.toml`
for reproducibility. Smoke-tested via `check_training_scripts.sh`.

Steps 7 and 8 are complete. `validate_and_augment_settings!`, model registry, `create_model`,
`train.jl`, `predict.jl`, 8 example TOMLs in `examples/`. All smoke-test clean (11 PASS).
501 tests pass.

Steps 7–9 are complete. `validate_and_augment_settings!`, model registry, `create_model`,
generic `train`/`predict` pipelines, 8 example TOMLs, full output suite, location alignment
at inference, explicit model weights with best-val and epoch checkpoints.
543 tests, 11 smoke tests pass.

Steps 11a, 11b, 11e are complete. JLD2 format differences documented. Surge and tide
baselines created: 3 training spans (1yr/5yr/20yr) × 3 surge models (Linear, Conv,
Attention) + 3 tide models (ProductTideModel, with nodal cycle `"N"` added for 20yr).
Data downloaded from Deltares S3, tidal analysis via `analyse_tides_schureman.jl`.
Training loop switched to `Flux.DataLoader` (proper full-epoch shuffling) across all
model families. `AttentionSurgeModel` val_input bug fixed. Leaderboard extended with
tide table; `leaderboard.ipynb` removed in favour of `leaderboard.qmd`.


