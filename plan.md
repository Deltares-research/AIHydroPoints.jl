
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
7. [x] **Determine robust training settings for the existing experiments.** **DONE.**
   The 317 5yr/20yr runs were **undertrained** — more training data gave *worse*
   RMSE on every split (incl. the Linear baseline; train RMSE rises too), because
   the configs used `nepochs=20` with `lr_decay_factor=0.5, lr_decay_epochs=10` +
   early stopping, vs `nepochs=50` at 1yr. Using the sweep script (6) +
   mlflow-tracked sweeps (`pybridge/`), all 24 `surge_*.toml` (5/317 stations ×
   1/5/20yr × Linear/Conv/Attention/BiLinear) were unified to a common,
   evidence-based `[train_settings]`/`nlags` baseline (`nlags=48` for
   Linear/Conv/BiLinear, `nlags=24` for Attention specifically; `batch_size=128`;
   `nepochs=50`; `lr_decay_factor=0.4`/`lr_decay_epochs=15`; `weight_decay=1.0e-4`)
   — see [surge hyperparameter consolidation] memory. **Deferred to task 22**: the
   317-station/20yr `AttentionSurgeModel` run itself won't complete — it reliably
   OOMs regardless of these settings, a pipeline scaling issue rather than a
   hyperparameter one. Giving up on tuning around it; config renamed to
   `experiments/317stations/surge_20yr_AttentionSurgeModel.toml_CRASHES` (non-`.toml`
   extension, deliberately so it's skipped by any `*.toml` glob/loop) and left
   out of scope until task 22 lands and it can be safely rerun.
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
15. [ ] **Add DVC for data storage, with a MinIO remote** — move larger
    training datasets out of git (currently several `data/*`/`test_data/*`
    files live directly in the working tree/git history) onto DVC-tracked
    storage backed by MinIO (S3-compatible), rather than committing them.
    Status: `dvc` is already in `pixi.toml` (`pixi run dvc` available,
    commit "add dvc to pixi"), and a throwaway scratch repo
    (`dvc_test.git/`, untracked, not part of this repo) already validated
    the basic `dvc add`/`push`/`pull` flow end-to-end with a plain
    local-filesystem remote (`.dvc/config` → `remote "myremote"` pointing
    at a local `dvcstore/` dir) — next step is pointing a remote at actual
    MinIO instead of a local dir. See `brainstorm_long_term_design.md`
    (task × tool matrix) for the earlier DVC-vs-Airflow/Snakemake/Ray
    reasoning behind choosing DVC in the first place.
    Open questions to settle before implementing: MinIO endpoint/bucket/
    credentials and how they're supplied (env vars vs. `.dvc/config.local`);
    which directories actually move to DVC (`data/` and `test_data/` are the
    obvious candidates — `test_data/` is small and used by CI-style unit
    tests, so may be worth keeping in git for simplicity even if `data/`
    moves); whether to rewrite git history to remove already-committed large
    files or just stop adding new ones going forward (rewriting affects
    every clone/collaborator, worth a deliberate decision rather than
    default `git filter-repo`).
16. [ ] **Deferred documentation follow-ups** (non-blocking): (a) the interaction
    models have no `background.md` writeup yet; (b) the DeepONet tide model is a
    one-line placeholder — its code merge is FiLM-style scale/shift, not a dot
    product; correct if/when needed.
17. [x] **Return the best model, not the last epoch.** **DONE.** After
    training with validation data and a `model_dir`, `train_model!` now
    reloads `params_best.jld2` into the model before returning (previously
    it left the in-memory model at the *final* epoch, which for some model
    families — notably `BiLinearSurgeInteractionModel` at 317 stations —
    could be substantially worse than the best validation epoch found along
    the way; `LinearSurgeModel` was accidentally immune since its training
    converges cleanly, final ≈ best). No-op when there's no validation data
    or no `model_dir` (nothing was ever persisted to reload). Two new tests
    in `test/test_training_features.jl`: the reload actually happens and
    matches `params_best.jld2` exactly, and the no-`model_dir` case still
    runs without error. 679 unit + 12 smoke pass.

    **Why this was bumped up:** task 23 found this wasn't a minor accuracy
    nit — `summary.toml` (computed from the pre-fix final-epoch model) made
    `BiLinearSurgeInteractionModel` at 317 stations look far worse than it
    is, and manufactured a spurious −61% storm-RMSE "finding" for the `full`
    modulation variant that vanished once re-evaluated on `params_best.jld2`.
    Any *past* comparison that used `summary.toml`'s recorded RMSE rather
    than `params_best.jld2` directly for a model where final ≠ best should
    be treated with caution — this fix only prevents the issue going
    forward, it doesn't retroactively correct old `summary.toml` files. See
    `brainstorm_surge_model_status.md` items 4/8.
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
22. [ ] **Unbatched forward pass causes OOM at scale** — related to task 14.
    `train_model!`'s epoch-end loss (`abstract_flux_model.jl:248,254`,
    `flux_model(x)`/`flux_model(x_val)`) and the generic `predict()`
    (`abstract_flux_model.jl:96`, used by `write_outputs` at lines 534 and 641,
    and hence by both training-time output generation and standalone
    `predict.jl`/`bin/predict`) all run the model on the **entire** input tensor
    in a single unbatched call — only the training step itself is chunked, via
    `Flux.DataLoader(...; batchsize=train_settings.batch_size)`. Confirmed as the
    likely cause of a real crash: `surge_20yr_AttentionSurgeModel` at 317
    stations was reliably `SIGKILL`ed (exit -9, i.e. OOM) even running alone —
    the branch network embeds 951 wind/pressure channels down to `nembed=32`,
    then **de-embeds back up to 951** per lag step, so the per-timestep
    activation footprint is non-trivial and gets multiplied across the full
    ~175,000-timestep training split in one shot (train tensor itself is
    already ~16GB; the unbatched forward pass plausibly adds tens of GB more,
    transiently, on top). `write_outputs` also calls `predict()` on the full
    split **twice** per split touched (once for output generation, once again
    in the summary block just for `rmse_<split>`) and does so even for
    narrow-`timerange` entries (e.g. `storm_eunice_2022` forward-passes the
    whole testing split before `select_timespan` narrows the result
    afterward) — redundant and wasteful, not just unscaled. Likely fix: chunk
    `predict()`/`forward()` (and the epoch-end loss calls) over the batch-time
    axis the same way training already does, accumulating outputs/loss across
    chunks — one change point that would benefit training, output-writing, and
    standalone inference at once. Not yet scoped into a concrete implementation
    plan; discuss approach before implementing.

23. [x] **Full-breadth tide modulation** (`brainstorm_surge_model_status.md`
    item 8). **DONE.** Added `model_pars["modulation_type"]` (`"local"`/`"full"`,
    `FullTideModulation`) to `BiLinearSurgeInteractionModel`, plus a small
    `scripts/parameter_sweep.jl` fix to sweep string-valued params. 5-station
    pass (1/5/20yr) was a wash within noise. The decisive 317-station/5yr
    pass initially found `full` a real loss on storm RMSE (−61%) — but that
    number was computed from `summary.toml` (task 17's final-epoch model, not
    `params_best.jld2`), and a same-checkpoint re-run showed `local` and
    `full` are actually statistically tied on both `rmse_testing` and
    `rmse_storm_eunice_2022`. **Verdict: no evidence *for* `full`, at either
    scale — deprioritized.** Doesn't settle whether neighboring-station tide
    genuinely carries physical information (no test here rules that out),
    but there's a structural reason not to expect this specific approach to
    find it even if so: `full`'s regressor stacks spatial collinearity on
    top of `local`'s already-considerable temporal collinearity (tide's
    harmonic-constituent structure), and a plain all-to-all weight matrix
    under MSE + `weight_decay` responds to that by shrinking toward a bland,
    near-zero solution rather than a sharper one — observed directly in both
    the 5- and 317-station weight patterns. If the physical question is ever
    revisited, replacing the raw 48-lag tide window with a handful of
    harmonic-constituent features (the same sin/cos basis `ProductTideModel`
    already uses) would remove the degeneracy by construction and give an
    actually well-conditioned test — not a reason to retry this same
    implementation at a different scale. Side findings along the way:
    317-station BiLinear
    has real run-to-run variance even under fixed settings (task-item 12 of
    the brainstorm doc), most of which turned out to trace back to the same
    checkpoint-selection issue (task 17) rather than a real architectural
    problem — an explicit modulation-on/off ablation confirms the branch is
    functionally beneficial despite noisy-looking weights, and tide's
    harmonic-constituent collinearity explains why the weights look noisy in
    the first place. Full tables/discussion in `brainstorm_surge_model_status.md`
    (items 4, 8, 12). Task 17 (return best model, not last epoch) is now
    fixed — see task 17 itself.

    **On-disk artifacts patched to match.** All 4 sweeps (317-station 5yr,
    5-station 1/5/20yr; 36 runs) had their `summary.toml` recomputed from
    `params_best.jld2` in place and `results.csv` regenerated via
    `parameter_sweep.jl --continue` (skips retraining, just re-aggregates),
    so the on-disk record now matches what's logged in the brainstorm doc —
    no more silent mismatch between the two. Checking the 5-station sweeps
    this way (not previously checked) surfaced one more real correction:
    the 20yr storm cell, originally flagged live as a "high-variance, one
    repeat early-stopped" outlier at −6.87%, collapses to a noise-level
    −0.74% once corrected — confirming that flag was right. No other
    5-station cell moved enough to change a conclusion.

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
