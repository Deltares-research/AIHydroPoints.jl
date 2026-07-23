
# Next steps for working on this project

## Main goal
The main goal of this project is to develop a machine learning model for predicting tides, surges and their interaction. The model will be trained on historical data and will be able to make predictions for future events. The model will be implemented in Julia and will be designed to be easily extensible and adaptable to different datasets and configurations

## Current status
The core architecture, TOML-driven pipeline, output suite, leaderboard, and the
tensor-layout refactor are all complete; surge and tide baselines exist. Last
full run: ~591 unit tests + 11/11 training smoke tests green. The remaining work
is baselines for the interaction model, a few product/scaling features, and
data/infra follow-ups — see **Open tasks**.

## Completed

Foundation and pipeline
- **Foundation** — unit tests restored; small test dataset; fast per-model tests;
  coverage-gap analysis + new tests; code modularised and made reusable.
- **Consistent training scripts** — all four families share `load_data`, a common
  `train_model!` interface, unified plotting, `runid`/`description` metadata, and
  write `run_settings.toml` for reproducibility.
- **TOML-driven pipeline** — `validate_and_augment_settings!`, `MODEL_REGISTRY` +
  `get_model_type`, `create_model` factory, generic `train`/`predict` in `src/`
  with thin CLI wrappers, 8 example TOMLs in `examples/`, `write_outputs`
  controlled by the `[output_settings]` table; settings docs in `docs/`.
- **Inference script** — `predict.jl` mirrors `train.jl`
  (`examples/predict_ConvSurgeModel.toml`), smoke-tested.
- **Output suite** — timeseries, scatter, fft, stats, series, tidal analysis, and
  summary outputs all work generically across models; explicit validation splits;
  location alignment at inference; `params_best.jld2` + epoch checkpoints. See
  `docs/output_settings.md`.
- **Leaderboard** — `src/leaderboard.jl` + `experiments/leaderboard.qmd` /
  `render_leaderboard.sh` (Quarto via the Julia engine): ranked table, CSV, PNG.

Models and refactors
- **Tensor-layout review & refactor** — fixed the `ConvSurgeModel` reshape
  scramble; adopted the always-tuple `preprocess → (x::Tuple, output)` convention
  with a shared `_surge_lag_windows` extraction helper; collapsed all four
  families into one generic `train_model!(::AbstractFluxModel)` + generic
  `forward` (uniform `m(x::Tuple)` flux call, 3-arg `preprocess` train form);
  removed the separate `AttentionSurgeModel` training loop. Working notes in
  `docs/notes_dimensions.md`.
- **ConvSurgeModel architecture** — strided Conv1D (`stride = filtersize`,
  non-overlapping lag tiling → smaller `Dense`) and a configurable activation
  defaulting to `swish`.
- **Surge & tide baselines** — JLD2 old/new format differences documented; surge
  (Linear/Conv/Attention) and tide (ProductTideModel, nodal cycle `"N"` for 20yr)
  baselines across 1yr/5yr/20yr spans; data from Deltares S3; training switched to
  `Flux.DataLoader` full-epoch shuffling.
  *(ConvSurgeModel surge baselines still need regenerating — see Open task 1.)*

Docs and comms
- **Presentation** — first draft plus refined methods/results.
- **Documentation** — Quarto template + GitHub publication; `notation.md` and
  `background.md` expanded (dense/ReLU walkthrough, generic convolution notation,
  the realistic 4-axis surge example, the `$\odot$` operator, and a
  ProductTideModel-led tide section); `docs/design.md` updated to the post-refactor
  conventions.

## Open tasks

1. [ ] **Finalize the TOML input format** (breaking changes) — do this now, while
   only ~34 hand-authored config files exist, before more baselines lock the
   format in and make migration painful. Detailed in [`plan_1.md`](plan_1.md).
   Introduces a `format_version` key (v2 = new format; v1/missing rejected with a
   migration message) — **document the key and a version-history section** in
   `docs/settings.md` (anchor `#format-versions`) and reference it from
   `docs/data_input_settings.md`, so future format changes have a place to record
   the version bump.
2. [ ] **Implement dead `[train_settings]` knobs.** Three settings are documented
   but never applied; the format renames (`plan_1.md`) only fix names, not
   behaviour:
   - **Early stopping** (`early_stopping_epochs`, ex-`patience`, Change 3): the
     loop always runs the full `1:nepochs`. Add an epochs-since-improvement
     counter and `break` when it exceeds the threshold, reusing the existing
     `best_val_rmse` / per-epoch val logic.
   - **Weight decay** (`weight_decay`, ex-`weight_reg`, Change 4): the optimiser
     is bare `Adam(learning_rate)`. Wrap it —
     `OptimiserChain(WeightDecay(weight_decay), Adam(learning_rate))` — skipping
     the wrapper when the coefficient is `0`. Also change the default `1.0e-4` →
     `0.0` (off), so regularisation is opt-in rather than silently applied to
     every existing config.
   - **Input noise** (`input_noise_std`, name unchanged): Gaussian data-augmentation
     noise on the inputs. It was applied in the pre-step-20h wave loop and lost in
     the `train_model!` unification (a silent regression). Re-add it to the generic
     loop — per-batch `x .+ input_noise_std .* randn`-style, gated on `> 0`. No
     rename (name is accurate), so this is implementation only, not a format change.
   All three are behaviour changes; do them alongside the format renames.
3. [ ] **Regenerate ConvSurgeModel baselines** (1yr/5yr/20yr). The existing Conv
   baselines are stale — trained before the reshape fix and before the strided
   Conv1D + `swish` activation changes. Linear (Dense is permutation-invariant)
   and Attention baselines remain valid. *(User to rerun.)*
4. [ ] **Interaction model — datasets & first testing.** `ConvInteractionModel`
   (simplified Conv1D, no station gate) does not learn. Data is hourly so
   `nlags=16` (16 h) covers a full tidal cycle — `nlags` is not the cause; root
   cause unknown and needs investigation. Note: the current interaction target is
   a temporary placeholder (surge loaded and renamed via the `"as"` alias); a
   meaningful target (residual = observed − tide_pred − surge_pred) requires a
   well-trained surge model and observed waterlevel data, which are not yet
   available.
5. [ ] **Interaction baselines** 1yr/5yr/20yr (determine timespans). Blocked on
   task 4.
6. [ ] **Improve scatter plot.**
7. [ ] **Create script for real-time forecasts.**
8. [ ] **Create an environment for online demos.**
9. [ ] **Add experiments for waves.**
10. [ ] **Scale surge model to a large number of stations.**
11. [ ] **Add DVC for data storage.**
12. [ ] **Deferred documentation follow-ups** (non-blocking): (a) the interaction
    models have no `background.md` writeup yet; (b) the DeepONet tide model is a
    one-line placeholder — its code merge is FiLM-style scale/shift, not a dot
    product; correct if/when needed.

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
