
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

1. [x] **Finalize the TOML input format** (breaking changes). **DONE.** Phase A
   (renames, `format_version` gate, unknown-key rejection, data-settings validation
   hardening; 39 TOMLs migrated; docs updated) and Phase B (task 2) both complete.
   612 unit + 11 smoke pass.
   Introduces a `format_version` key (v2 = new format; v1/missing rejected with a
   migration message) — **document the key and a version-history section** in
   `docs/settings.md` (anchor `#format-versions`) and reference it from
   `docs/data_input_settings.md`, so future format changes have a place to record
   the version bump.
2. [x] **Implement dead `[train_settings]` knobs.** **DONE** — all three
   implemented in the generic `train_model!`, tested
   (`test/test_training_features.jl`), 612 unit + 11 smoke pass. `early_stopping_epochs`
   is now active whenever validation data is present (set `nothing` to disable).
   Original scope:
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
3. [x] **Improve scatter plot.** **DONE.** Density modes (`density=:auto`:
   transparent points for small series, log-count heatmap for large) + `linear_fit!`
   / `qq!` overlay helpers added to MultiTimeSeries.jl; AIHydroPoints scatter output
   gained `scatter_add_fit` / `scatter_add_qq` overlays (default `true`) and a
   `<station> RMSE = …` title matching the time-series plots. 614 unit + smoke pass.
4. [x] **Combined `SurgeInteractionModel` series.** **First model DONE:**
   `BiLinearSurgeInteractionModel` (`AbstractSurgeInteractionModel <:
   AbstractSurgeModel`) — linear surge × a per-station tide modulation
   `1 + a·σ(V ⋆ tide)`, forcing all-to-all + tide one-to-one at output stations;
   documented in `docs/background.md`. Registry + example TOML + tests + smoke added.
   Also fixed two latent issues surfaced by the mixed input grids:
   `_surge_lag_windows` must not align `tide`, and `validate_and_augment_settings!`
   now derives `in_names` from a non-`tide` input. 654 unit + smoke pass.
   Experiments (5/317-station, 1–20yr): interaction gives ~5–6% aggregate storm gain,
   **~+20% at shallow estuaries/Wadden**, hurts deep sites — **next: gate the
   interaction per station** (see [surge-interaction findings] memory). `a≈0.5`;
   `tanh`/`exp` variants still earmarked.
5. [x] **Map/chart outputs for per-station statistics.** **DONE.** Two layers:
   `plot_map(bbox; …)` composable
   primitive (cached EMODnet WMS bathymetry background, request-hashed cache +
   graceful offline fallback; `src/maps.jl`) and a `plot_stats` output flag →
   per-station **RMSE + bias** maps (lon/lat scatter on the bathymetry;
   `_plot_station_stats`, wired into `write_outputs`). Added `FileIO`/`ImageIO`/`Downloads` deps, docs, and
   offline tests. **661 unit pass.** (Single-run only; two-run improvement maps
   deferred to item 6.) Motivation: skill / interaction gain is **spatially
   localized** (shallow estuaries/Wadden; see [surge-interaction findings] memory).
6. [x] **Parameter-sweep script** — design in [`plan_6.md`](plan_6.md). **DONE (simple
   version).** `scripts/parameter_sweep.jl`: from a reference training TOML, varies
   **one** parameter (dotted path) over a value list with **N repeats** + the
   unmodified baseline, into `sweeps/<experiment>/<setting>_rep<k>/` (stats-only,
   absolute paths, stale-`params` cleared). After all runs it prints + writes
   `results.csv` — per setting the **mean ± std across repeats** (consistency) and the
   **% RMSE reduction vs the unmodified baseline** (testing + storm). Validated on a
   5-station sweep. Deferred: multi-param grids, comparison **maps** (task-5 deferral),
   parallel execution, leaderboard integration.
7. [ ] **Determine robust training settings for the existing experiments.** The 317
   5yr/20yr runs are **undertrained** — more training data gave *worse* RMSE on every
   split (incl. the Linear baseline; train RMSE rises too), because the configs use
   `nepochs=20` with `lr_decay_factor=0.5, lr_decay_epochs=10` + early stopping, vs
   `nepochs=50` at 1yr. Use the sweep script (6) + charts (5) to find `nepochs` / LR
   schedule / `early_stopping_epochs` / `batch_size` that **converge consistently
   across 1/5/20yr and 5/317 stations**, then apply them to the experiment configs.
8. [ ] **Regenerate ConvSurgeModel baselines** (1yr/5yr/20yr). The existing Conv
   baselines are stale — trained before the reshape fix and before the strided
   Conv1D + `swish` activation changes. Linear (Dense is permutation-invariant)
   and Attention baselines remain valid. *(User to rerun.)*
9. [ ] **Interaction model — datasets & first testing.** `ConvInteractionModel`
   (simplified Conv1D, no station gate) does not learn. Data is hourly so
   `nlags=16` (16 h) covers a full tidal cycle — `nlags` is not the cause; root
   cause unknown and needs investigation. Note: the current interaction target is
   a temporary placeholder (surge loaded and renamed via the `"as"` alias); a
   meaningful target (residual = observed − tide_pred − surge_pred) requires a
   well-trained surge model and observed waterlevel data, which are not yet
   available.
10. [ ] **Interaction baselines** 1yr/5yr/20yr (determine timespans). Blocked on
    task 9.
11. [ ] **Create script for real-time forecasts.**
12. [ ] **Create an environment for online demos.**
13. [ ] **Add experiments for waves.**
14. [ ] **Scale surge model to a large number of stations.**
15. [ ] **Add DVC for data storage.**
16. [ ] **Deferred documentation follow-ups** (non-blocking): (a) the interaction
    models have no `background.md` writeup yet; (b) the DeepONet tide model is a
    one-line placeholder — its code merge is FiLM-style scale/shift, not a dot
    product; correct if/when needed.
17. [ ] **Return the best model, not the last epoch.** After training with early
    stopping, `train_model!` leaves the in-memory model at the *final* epoch, while
    the best weights are saved only to `params_best.jld2`. So predicting straight
    from the returned model can use worse-than-best weights. Reload `params_best`
    before returning (or document the current behaviour). Pre-existing behaviour,
    unchanged by the Phase B early-stopping work.
18. [ ] **Stale `run_settings.toml` for pre-format-v2 baselines** (records only; not
    replayable against the new reader unless migrated).
19. [ ] **Minor data-settings robustness (deferred):** (a) warn when a variable is
    provided by more than one file in the same split — `flat[as] = …` currently
    overwrites silently; (b) detect mixed sampling grids — `_intersect_times` clips
    the range only and does not verify a common timestamp grid, so e.g. hourly input
    + 10-min target would pass and mis-align downstream (detect-and-error for now; a
    resample/align policy is a larger change).
20. [ ] **Lake training** — extend beyond the North Sea coastal stations to lakes.
    Scope (data sources, target quantities, whether existing surge/tide models
    apply as-is) still to be worked out.
21. [ ] **World-wide GTSM surrogate (FUTURA)** — train a global surrogate of GTSM
    (Global Tide and Surge Model) using this same ML approach, rather than the
    North Sea only. Likely depends on task 15 (DVC) for handling the much larger
    data volume.

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
