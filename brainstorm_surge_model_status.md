# Surge model status brainstorm

**Just some initial thoughts** — early-stage, exploratory, and incomplete.

Open notes on the current status of the surge model track (experiments,
results, open issues), ahead of committing anything to `plan.md`. Not a plan,
not decisions — just captured reasoning so we don't re-derive it later.

## Test priority table

All logged tests, sorted by **Priority** (descending) — Priority is a
judgment call informed by the other three scores, not a mechanical average
of them; kept independently adjustable rather than computed. Item 5's test
is identical to item 1's (omitted here — see item 5); item 6's test isn't
fully scored yet (omitted — see item 6).

| Item | Test | Complexity | Likelihood | Potential gain | Priority | Result |
|---|---|---|---|---|---|---|
| 8 | Full-breadth tide modulation | 1 | 2 | 3 | **4** | Negative — also see item 4 |
| 11 (probe) | MSE → Huber loss swap | 1 | 4 | 2 | **4** | — |
| 9 | Literature scan (surrogate-model papers) | 2 | 3 | 4 | **4** | — |
| 17 | Charnock sensitivity check (smaller constant) | 1 | 4 | 3 | **4** | — |
| 2a | EOF/spectral analysis of ERA5 field | 1 | 4 | 3 | **3** | Postponed — superseded by in-progress shallow-water adjoint, expected more informative |
| 2 | Deep MLP (wind+pressure ± tide) | 2 | 2 | 3 | **3** | No bias signal (wind+pressure only); +tide is the largest single gain found in this doc at 5yr/20yr (~16–28%, a wash at 1yr) — 5-station `rmse_testing` floor now characterized at ~0.082 |
| 3 | Deep MLP, multi-seed (bias test) | 2 | 2 | 2 | **3** | — |
| 4 | BiLinear/317 rerun + parameter stability | 2 | 3 | 3 | **3** | Negative (also see item 4) |
| 7 | Nested tide-modulation correction | 2 | 4 | 2 | **3** | — |
| 2b | Point-density sweep | 3 | 2 | 3 | **3** | Done — real but secondary ~7–8% `rmse_testing` gain, transfers to `DenseSurgeInteractionModel`; see `input_sensitivity.md` |
| 11 (full) | Curriculum + self-paced learning study | 4 | 3 | 4 | **3** | — |
| 1 | Long slow-LR training run, multi-seed | 1 | 1 | 4 | **2** | — |
| 10 | Extend DCSM-FM/ERA5 toward 1950 | 5 | 1 | 2 | **1** | — |

## Argument list

Flat list of items debated so far, opposing sides shown together, plus
flagged categories not yet argued. Numbering matches "Debate summary" below.
Next step: work through these one at a time — look for a concrete way to
test each hypothesis, then assign priority. From item 7 onward, test scores
also include **Potential gain** (magnitude of improvement *if* the test
succeeds, 1–5) alongside Complexity/Likelihood of success — split out
because Likelihood of success alone conflates "does it help at all" with
"how much"; retrofitted onto items 1–6 as we revisit each.

1. **Model convergence across architectures.** *Cannot:* several different
   models already land at similar accuracy. *Can:* could be an artifact —
   task 7's undertraining precedent, and aggregate RMSE masking per-station
   gains.
   - **Test:** run one config much longer under a non-decaying/slow LR
     schedule (+multi-seed) to check for a late "weekend-training" breakthrough.
   - Complexity: 1/5 · Likelihood of success: 1/5 · Potential gain: 4/5 · Priority: 2/5.
2. **Universal approximation vs. input information.** *Can:* ML can in
   theory represent the wind/pressure→surge mapping. *Cannot:* our inputs
   are a sparse point-sample of what DCSM-FM actually integrates — an
   information bottleneck, not a capacity question.
   - **Test:** unstructured deep MLP, two variants (wind+pressure only vs.
     +tide), to separate "does depth help" from "does unstructured tide access help."
   - Complexity: 2/5 · Likelihood of success: 2/5 · Potential gain: 3/5 · Priority: 3/5.
   - **Test 2a:** EOF/spectral analysis of the full ERA5 field — how many
     spatial modes explain most of the variance vs. our 9/317 sampled points.
     User's prior, from other results seen elsewhere: likely ~10–40
     significant modes — a real answer worth confirming, but not a
     surprising/decisive one, hence the lower priority below.
   - Complexity: 1/5 · Likelihood of success: 4/5 · Potential gain: 3/5 · Priority: 3/5.
   - **Postponed (user decision):** deprioritized in favor of an adjoint for
     the shallow-water model currently in development, which will give
     sensitivities directly rather than the EOF's indirect proxy (how many
     ERA5 modes explain variance) — expected to be more informative once
     available. Not abandoned, just superseded as the preferred route to
     the same "how much input information are we missing" question.
   - **Test 2b:** point-density sweep — fix output stations, vary input
     point count, retrain, watch `rmse_testing` vs. density.
   - Complexity: 3/5 · Likelihood of success: 2/5 · Potential gain: 3/5 · Priority: 3/5 (contingent on 2a).
   - **Result (test 2b, done without waiting on 2a — full writeup in
     `input_sensitivity.md`):** a controlled 9→24-point layout sweep (5
     output stations held fixed throughout) found a real, monotonic
     ~7–8% `rmse_testing` gain (0.1103→0.1018 at the best 23-point
     config), confirmed to transfer to `DenseSurgeInteractionModel` at
     similar relative size (~6.5%) — real but clearly secondary next to
     this item's own +16–28% tide-concatenation gain below (5yr/20yr; a
     wash at 1yr, see below). Along the
     way: local wind clustering near the shallow Wadden stations didn't
     help (spread points did, contradicting the shallow-water-setup
     prior); Wick's outsized weight behaves like a large-scale "external
     surge" entry-point signal that points further upstream capture
     better, up to a point (plateaus rather than climbing indefinitely);
     a reproducible ~2× weight asymmetry between two near-identical
     (corr ≈0.99) points near Blankenberge turned out to be real across
     seeds, not noise — cause still open; and a follow-up "flank the
     hotspots with nearby points" branch underperformed "push further
     outward," reinforcing that crowded/nearby points mostly redistribute
     weight rather than adding information. Item 2a (EOF/spectral analysis)
     would have been the cheaper, more direct way to answer the same
     underlying "how much input information are we missing" question, but
     is postponed in favor of the in-progress adjoint (see above).
   - **Result (wind+pressure-only deep MLP, `DenseSurgeModel`, 5-station,
     `experiments/5outputs9inputs/surge_{1,5,20}yr_DenseSurgeModel.toml`, vs. the
     `LinearSurgeModel`/`ConvSurgeModel` baselines at the same timespans):**

     | timespan | Linear rmse_testing | Conv rmse_testing | Dense rmse_testing | Conv rmse_storm | Dense rmse_storm |
     |---|---|---|---|---|---|
     | 1yr | 0.1165 | 0.1198 | 0.1255 | 0.2241 | 0.2404 |
     | 5yr | 0.1101 | 0.1048 | 0.1053 | 0.2063 | 0.2019 |
     | 20yr | 0.1093 | 0.1038 | 0.1031 | 0.2064 | 0.1984 |

     (Linear and Conv numbers above are corrected from an earlier pass that
     used `early_stopping_epochs=10` against `lr_decay_epochs=15` — the same
     before-the-first-decay trap documented below and in `input_sensitivity.md`,
     just not yet ported to these two baselines. Re-run with
     `early_stopping_epochs=25`: Linear improved 2–9% at every timespan
     (largest at 5yr, −8.7%), Conv barely moved at 1yr/20yr (its 5yr config
     was already on `25`). Dense numbers are unaffected — `DenseSurgeModel`
     already used `early_stopping_epochs=25` throughout.)

     (`n_params`: Linear 6,485 · Conv 4,661 · Dense 41,669 — Dense's single
     `Dense(32)` hidden layer costs ~9× Conv's parameter count for
     effectively the same accuracy.)

     `DenseSurgeModel` (a genuinely unstructured MLP: no convolution, no
     lag-window locality, no weight sharing across stations) tracks
     `ConvSurgeModel` within ~0.5–4.8% at every timespan, in both directions
     (Dense wins at 20yr, Conv wins at 1yr and 5yr) — well inside the
     run-to-run noise measured elsewhere in this doc (item 8's ~5–15% seed
     std). Conv/Dense beat `LinearSurgeModel` at 5yr/20yr, but — now that
     Linear's own early-stopping bug (above) is fixed — by only ~5–6%, not
     the ~9–13% an earlier pass of this doc reported; most of that original
     gap was the same measurement artifact already flagged elsewhere in this
     doc (item 4's checkpoint-selection bug, item 8's pre-consolidation
     hyperparameters), not a real architecture effect. All three are
     roughly tied at 1yr (not enough data for the extra capacity to pay
     off — both Conv and Dense land slightly worse than Linear there). If
     anything, fixing the artifact makes the "no inductive-bias signal"
     reading below *stronger*: Linear, Conv, and Dense track each other
     even more closely than previously thought.

     Partial answer to item 2: **no signal that Conv's inductive bias (or
     Linear's, or presumably Attention's/BiLinear's) is what's capping
     accuracy** — an architecture with none of that structure reaches the
     same ceiling, not a lower or higher one. This experiment is inherently
     hard to make definitive (see the noise caveat below): it doesn't
     distinguish "the bias is a good match to the problem" from "the bias
     doesn't matter because the real ceiling is elsewhere" (e.g. item 9's
     input-information ceiling) — only that loosening it further doesn't buy
     anything, the same conclusion item 3 already reached comparing
     Conv→Attention→BiLinear. The `+tide` half of this test
     (`DenseSurgeInteractionModel`, same unstructured-MLP idea but
     concatenating tide lag windows with the forcing instead of `BiLinear`'s
     multiplicative gate) exists in code and trains end-to-end but hasn't
     been run as a scaled/repeated experiment yet — still open.

     Caveat carried over from the `hidden_dims` sweep that produced the
     `[32]` setting used above: the sweep's first pass
     (`early_stopping_epochs = 10`) was confounded by early stopping firing
     before the first scheduled LR decay in about half its runs (best-epoch
     as low as 3, purely seed-dependent) — inflating apparent noise and
     shrinking effective training length unpredictably. Re-run with
     `early_stopping_epochs = 25` (past all three decay steps) fixed it:
     per-run `rmse_testing` spread (20yr, 3 seeds × 5 `hidden_dims` settings)
     tightened from `[0.1022, 0.1067]` to `[0.1022, 0.1044]`, and every
     `hidden_dims` variant from `[32]` to `[256,128,64,32]` came out
     statistically tied (Δ ≤ 0.54%, within the corrected noise band) —
     capacity/depth genuinely doesn't matter in that range, it isn't a
     training-budget artifact. `[32]` was picked over the original `[64,32]`
     mainly on parameter economy plus a consistent storm-RMSE edge (+6.26%
     at 20yr), not because it was the clear overall-accuracy winner — it
     wasn't.
   - **Result (`+tide`, `DenseSurgeInteractionModel`, 5-station,
     `experiments/5outputs9inputs/surge_{1,5,20}yr_DenseSurgeInteractionModel.toml`,
     concatenating flattened forcing + tide lag windows into the same
     unstructured MLP, vs. no-tide `DenseSurgeModel` and
     `BiLinearSurgeInteractionModel`'s multiplicative gate):**

     | timespan | DenseSurgeModel (no tide) | BiLinearSurgeInteractionModel | `DenseSurgeInteractionModel` |
     |---|---|---|---|
     | 1yr rmse_testing | 0.1255 | 0.1140 | 0.1151 |
     | 5yr rmse_testing | 0.1053 | 0.1071 | **0.0919** |
     | 20yr rmse_testing | 0.1031 | 0.1057 | **0.0826** |
     | 1yr rmse_storm | 0.2404 | **0.2151** | 0.2127 |
     | 5yr rmse_storm | 0.2019 | 0.1929 | **0.2110** |
     | 20yr rmse_storm | 0.1984 | 0.1920 | **0.1624** |

     (BiLinear's 1yr/20yr numbers are corrected the same way as Linear/Conv
     above — `early_stopping_epochs` 10→25 — its 5yr config was already on
     `50`, unaffected.)

     **Still the single largest gain found anywhere in this doc at 5yr/20yr**
     — concatenating tide beats both the no-tide model and BiLinear's gate by
     ~16–28% on `rmse_testing` there, and at 20yr it's the one case where the
     storm-specific metric improves *with* the overall metric rather than
     decoupling from it (contrast item 8's `full`-modulation result, or the
     flat/negative storm deltas throughout this doc). **At 1yr the picture is
     now a wash, not a win**: BiLinear's corrected number (0.1140) is
     marginally *better* than `DenseSurgeInteractionModel` (0.1151, +0.96%)
     and its storm RMSE clearly better (0.2151 vs 0.2127 — this direction was
     already a near-tie pre-fix); with only 1 year of training data, neither
     architecture has enough signal for the tide-concatenation idea to show a
     real edge over BiLinear's narrower gate. Still directly complicates item
     8's conclusion that widening BiLinear's tide modulation doesn't help at
     5yr/20yr: the failure there wasn't that neighbouring-station tide
     information is useless, it's that BiLinear's *multiplicative gate* is a
     comparatively poor way to use even the one-to-one tide signal it already
     had — plain concatenation into an MLP extracts substantially more from
     the same input, once there's enough data (5yr+) for it to matter.

     **Architecture search on top of this (20yr, `scripts/parameter_sweep.jl`,
     3 seeds per setting) converged cleanly, unlike the no-tide sweep:**

     | `hidden_dims` | rmse_testing (mean±std) | Δ vs `[32]` |
     |---|---|---|
     | `[16]` | 0.08948 ± 0.00060 | −5.8% (worse) |
     | `[32]` | 0.08457 ± 0.00037 | — |
     | `[64]` | 0.08455 ± 0.00062 | ~0% |
     | `[128]` | 0.08444 ± 0.00036 | ~0% |
     | `[64,32]` | 0.08216 ± 0.00050 | **+2.9%** |
     | `[128,64]` | 0.08192 ± 0.00075 | **+3.1%** |
     | `[64,64,32]` | 0.08273 ± 0.00061 | −0.7% (vs `[64,32]` baseline) |
     | `[64,64,64,32]` | 0.08316 ± 0.00079 | −1.2% (vs `[64,32]`; storm −6.1%) |

     Unlike the no-tide `DenseSurgeModel` sweep (where every `hidden_dims`
     from `[32]` to `[256,128,64,32]` was statistically tied), depth genuinely
     matters here: going too narrow (`[16]`) clearly hurts, width alone
     (`[64]`, `[128]`) doesn't help, but adding a second layer (`[64,32]`,
     `[128,64]`) buys a real ~3% beyond the noise band — and going deeper
     still (3–4 layers) plateaus and mildly *reverses* it, consistently across
     both metrics. `[64,32]` was adopted as the production setting: same gain
     as `[128,64]` (tied within noise) at roughly half the parameters, and a
     genuine local optimum rather than an under-explored point — width,
     depth, and (separately, see below) training length were all pushed past
     it without further gain.

     **Patience check (20yr, `nepochs` 50 vs. 100, 3 seeds each,
     `hidden_dims=[32]`):** `+1.22%` from doubling the epoch budget
     (0.08546 ± 0.00046 → 0.08442 ± 0.00053) — real (~2× the noise std) but
     small, confirming the model wasn't starved for training time, just
     mildly under-budgeted. `nepochs=100` was adopted; a detailed look at one
     run's loss curve (`hidden_dims=[64,32]`, 20yr) shows a clean, non-overfit
     picture: LR-decay-aligned improvement steps at epochs 15/30/60, best
     epoch at 66, then a genuine flat plateau (val RMSE 0.0915–0.0923) until
     early stopping fires correctly at epoch 91 — not another training-budget
     artifact, and the train–val gap (~0.011–0.013) stays stable rather than
     widening, so no emerging overfitting either.

     **Final 5-station production settings**
     (`hidden_dims=[64,32]`, `activation="swish"`, `nepochs=100`,
     `early_stopping_epochs=25`, `nlags=48`, `lr_decay_epochs=15`,
     `lr_decay_factor=0.4`, `weight_decay=1e-4`), all three timespans
     retrained and consistent (`n_params=100,613` throughout):

     | timespan | rmse_testing | rmse_storm_eunice_2022 |
     |---|---|---|
     | 1yr | 0.1151 | 0.2127 |
     | 5yr | 0.0919 | 0.2110 |
     | 20yr | **0.0826** | 0.1624 |

     Taken together (repeat-run std ~0.0005, i.e. ~0.6% relative, at 20yr;
     width/depth/patience all swept past their optimum without further gain),
     this is a reasonably well-characterized **`rmse_testing` floor for the
     5-station case at ≈0.082–0.083** given the current input configuration
     (9-point ERA5 wind/pressure + one-to-one output-station tide, 48 lags,
     20yr training). "Floor for this input configuration" is the operative
     caveat: it says architecture search on top of *this* input set has
     plateaued, not that the floor is fixed — item 9's literature scan, item
     2a/2b's input-density questions, and item 17's Charnock check all bear on
     whether a richer or corrected input could push the floor itself lower.
3. **Inductive bias.** *Can:* current architectures constrain how inputs
   combine — a less restrictive one might help. *Cannot:* already spanned a
   wide bias range (Linear→Conv→Attention→BiLinear); loosening it hasn't
   reliably helped empirically.
   - **Test:** shares item 2's variant-1 (wind+pressure MLP) — run with 2–3
     seeds so it's decisive, not another single-run anecdote.
   - Complexity: 2/5 · Likelihood of success: 2/5 · Potential gain: 2/5 · Priority: 3/5.
4. **Parameter count vs. noise.** *Cannot:* more parameters → more noise, no
   net benefit. *Can:* confounded — BiLinear at 317 stations wasn't given
   the same nepochs/nlags/weight_decay as the others.
   - **Test:** rerun BiLinear/317 (unified settings) with 2–3 seeds; compare
     learned `V`/`surge.weight` across seeds (no permutation ambiguity —
     fully labeled params) for parameter-level, not just RMSE-level, stability.
   - Complexity: 2/5 · Likelihood of success: 3/5 · Potential gain: 3/5 (diagnostic — reads as "how much would this update our picture if the confound is real," not a direct accuracy gain) · Priority: 3/5.
   - **Result (revised — a second confound, checkpoint selection, explains
     most of that gap):** `train_model!` returns the *final*-epoch model
     (task 17), and `summary.toml` is computed from that, not
     `params_best.jld2`. For Linear, final ≈ best (`~0.0839` either way) —
     never mattered. For BiLinear/317 it does: evaluated at
     `params_best.jld2` instead, `local`'s `rmse_testing` drops to
     `0.095 ± 0.008` — much closer to Linear. Most of "BiLinear has a
     genuine scaling problem at 317 stations" was this artifact, not a real
     architectural gap.

     A direct ablation (same best-epoch checkpoint, forcing `a=0` to switch
     the modulation off) shows the branch isn't dead weight despite its
     noisy-looking `V`: `rmse_testing` +29–42% and `rmse_storm_eunice_2022`
     +55–72% worse in *every* seed with it off. "Noisy weights" ≠
     "functionally useless" — only the weight-inspection plots tested the
     first claim.

     Why the weights look noisy anyway: tide is a sum of a handful of
     harmonic constituents, so the 48-lag regressor is collinear — many
     different `V` fit almost equally well, and seeds land in different
     points of that near-flat manifold. Same reasoning explains why `full`'s
     weights were far *tighter* across seeds than `local`'s despite being
     more degenerate (adds spatial collinearity on top): `weight_decay`
     picks the unique minimum-norm solution, more consistently so the more
     degenerate the problem. Cuts both ways: this collinearity is a
     structural headwind for identifying *either* variant's weights from
     data, so neither's learned shape is strong evidence about which is
     physically correct — the ablation-based functional effect is the
     trustworthy signal, not the weight shapes.
5. **Patience / more training.** *Can:* maybe it just needs more epochs.
   *Cannot:* loss curves already show converge-and-flatten / an LR-decay
   staircase.
   - **Test:** see item 1 (same run — long non-decaying/slow-LR schedule, multi-seed).
6. **Untried regularization tricks.** *Can:* dropout/DropConnect etc. haven't
   been tried. *Cannot:* train-test gap is already near zero; the real gap
   is storm-specific.
   - **Test:** thinking along similar lines to items 1 and 5 (same kind of
     extended/varied training run) — not yet fully worked out.
   - Potential gain: 2/5 (provisional — no train-test gap for dropout-style
     tricks to close, per the "cannot" side).
7. **Small modification of Linear (BiLinear).** *Can:* already does well —
   try more small, targeted modifications. *Cannot:* its gain is tied to one
   specific known mechanism and doesn't generalize (worse than Linear at 317
   stations).
   - **Test:** nested modulation `U·s·(1+V·t·(1+Dense(t)))` — a learned
     correction added *on top of* the existing local `V` term, not replacing it.
   - Complexity: 2/5 · Likelihood of success: 4/5 · Potential gain: 2/5 · Priority: 3/5.
8. **Spatial breadth of the tide modulation.** *Can:* modulation is
   local-only; neighboring stations might help. *Cannot:* surge branch is
   already all-to-all, tide is spatially redundant, and widening repeats
   argument 4's capacity risk.
   - **Test:** full-breadth modulation `Dense(nstations·nlags => nstations)`
     over all-station tide — same pattern as the surge branch, ~1/3 its weights.
   - Complexity: 1/5 · Likelihood of success: 2/5 · Potential gain: 3/5 · Priority: 4/5.
   - **Result (5-station pass, `local` vs `full`, 3 repeats, `scripts/parameter_sweep.jl`;
     re-patched to `params_best.jld2` after the task-17 fix — see below):**

     | timespan | rmse_testing Δ | rmse_storm_eunice_2022 Δ |
     |---|---|---|
     | 1yr | +0.02% (wash) | −1.53% (worse, within noise) |
     | 5yr | −0.20% (wash) | −1.85% (worse, within noise) |
     | 20yr | +0.62% (wash) | −0.74% (worse, within noise — was −6.87% pre-patch) |

     No evidence `full` helps at 5-station scale; both metrics are a wash at
     every timespan. 5-station training turned out to be only mildly exposed
     to the task-17 checkpoint bug (small shifts, no reversed conclusion) —
     *except* the 20yr storm number, originally flagged live as "high-variance,
     one repeat early-stopped at epoch 13/50": that flag was right, and
     correcting for it collapses −6.87% to a noise-level −0.74%, consistent
     with every other cell in this table. Matches the "cannot" side's
     prediction (tide is spatially redundant at only 5 stations). Weak-power
     test by design (see the caveat in `plan.md` task 23) — doesn't settle
     the hypothesis, just gives no reason to skip the decisive 317-station run.
   - **Result (revised — every run's `summary.toml` patched to
     `params_best.jld2` in place, `results.csv` regenerated via
     `parameter_sweep.jl --continue`, both arms):**

     | setting | rmse_testing | rmse_storm_eunice_2022 |
     |---|---|---|
     | local | 0.0950 ± 0.0095 | 0.1561 ± 0.0153 |
     | full | 0.0931 ± 0.0005 | 0.1554 ± 0.0019 |

     **The −61% storm result evaporates entirely.** Evaluated fairly,
     `local` and `full` are statistically tied on both metrics — well within
     each other's noise. `full` is actually the *more consistent* of the two
     across seeds (std ~5–8× tighter), but that's not an accuracy win, and
     per item 4's collinearity finding it's an expected side-effect of
     `full`'s more degenerate weight space being pulled harder toward
     `weight_decay`'s minimum-norm solution, not evidence it learned
     something `local` didn't.
     **Conclusion: no evidence *for* `full`** — a wash on
     `rmse_testing` and statistically tied on `rmse_storm_eunice_2022` at
     both 5- and 317-station, consistently, after correcting for the
     checkpoint bug. Two separate claims worth keeping apart: this does
     *not* settle whether neighboring-station tide genuinely carries useful
     information for the physical interaction (no test here rules that out);
     it *does* mean we don't currently expect further gains along this
     specific path. The reason is structural, not just an unlucky result:
     `full`'s regressor stacks spatial collinearity on top of `local`'s
     already-considerable temporal collinearity (tide's harmonic-constituent
     structure, item 4), and a plain all-to-all weight matrix trained under
     MSE + `weight_decay` is a poor tool for identifying anything in a
     regressor space that degenerate — it visibly responds to more
     degeneracy by shrinking toward a bland, near-zero-or-averaged-out
     solution rather than by finding a sharper one. If the physical question
     is ever revisited, the collinearity itself points at the fix: replace
     the raw 48-lag tide window with a handful of harmonic-constituent
     features (sin/cos at the known tidal frequencies — the same basis
     `ProductTideModel`'s Doodson forcing already uses, `docs/background.md`)
     instead of feeding 48 collinear raw lags. That would remove the
     degeneracy by construction and give a genuinely well-conditioned test of
     spatial breadth, rather than repeating this same estimation difficulty
     at a different scale. Task 23 in `plan.md` reopened accordingly.
9. **Literature precedent.** *Can:* published papers report higher accuracy
   for ML storm-surge emulators. *Cannot:* those typically use synthetic
   storm ensembles + richer spatial inputs — a different data regime.
   - **Test:** literature scan restricted to ML **surrogates of a numerical
     surge model** (target = simulated output, like ours, not observations)
     in comparable extratropical/shelf-sea settings — our task is reproducing
     DCSM-FM, not matching measured sea levels, so only that category is a fair comparison.
   - Complexity: 2/5 · Likelihood of success: 3/5 · Potential gain: 4/5 · Priority: 4/5.
10. **More training data (temporal extent).** *Can:* ERA5 goes back to 1950.
    *Cannot:* the DCSM-FM target only covers 2000–2023 (extendable, but
    costly), the 5yr→20yr comparison already shows diminishing/negative
    returns, and pre-~1980 ERA5 is less homogeneous (sparser satellite record).
    - **Test:** extend DCSM-FM + ERA5 toward 1950 — technically feasible, but
     deprioritized given the evidence above.
    - Complexity: 5/5 · Likelihood of success: 1/5 · Potential gain: 2/5 · Priority: 1/5.
11. **Training focus / hard-example reweighting.** *Can:* reweight toward
    where the model does poorly. *Cannot:* ERA5 likely underestimates storm
    peak intensity (an input ceiling for storms — weakened since DCSM-FM's
    own forcing is confirmed by the user to also be ERA5, same weak input
    → correspondingly weak target, self-consistent; one known difference is
    now split out as item 17), and reweighting risks overfitting to a
    handful of storm events.
    - **Probe (cheap, do first):** swap MSE → Huber loss (`Flux.Losses.huber_loss`,
      one-line change) on 1-2 existing configs; compare storm vs. overall
      RMSE delta vs. the MSE baseline — gauges whether loss-weighting of
      large-error/storm cases matters at all before committing to the full study below.
    - Complexity: 1/5 · Likelihood of success: 4/5 · Potential gain: 2/5 (diagnostic, gates the study below) · Priority: 4/5.
    - **Full study (contingent on probe):** curriculum learning (easy→hard
      schedule, targets tail/storm gains) + self-paced learning (model-driven
      difficulty, targets general-optimization gains) — distinct mechanisms
      in the literature, evaluate separately via `rmse_storm_eunice_2022`
      (curriculum) vs. `rmse_testing` (self-paced), not one aggregate number.
      Needs a staged-sampling feature in the shared `train_model!` loop
      (`abstract_flux_model.jl`/`training.jl`), not a one-off script.
      ([Imbalanced regression and extreme value prediction](https://link.springer.com/article/10.1007/s10994-020-05900-9),
      [Dynamic Curriculum Learning for imbalanced/long-tail data](https://arxiv.org/pdf/2210.14724),
      [Self-Paced Learning: an Implicit Regularization Perspective](https://www.researchgate.net/publication/303750070_Self-Paced_Learning_an_Implicit_Regularization_Perspective),
      [Curriculum Learning: A Survey](https://arxiv.org/pdf/2101.10382)).
    - Complexity: 4/5 · Likelihood of success: 3/5 · Potential gain: 4/5 · Priority: 3/5 (contingent on probe).
12. **Statistical significance / run-to-run variance** *(flagged, not yet
    argued)* — seeding infrastructure already exists (`scripts/parameter_sweep.jl`'s
    `Random.seed!(rep)` before each in-process `train()` call — genuinely
    reproducible, not new code needed), but it's scoped to repeats of *one*
    model's hyperparameter values, never applied across different
    architectures — so no repeat-run variance has actually been measured for
    the model-vs-model comparisons this debate relies on.
    - **First real data point (from item 8's decisive-pass baseline, 3
      seeds):** BiLinear, 317-station 5yr, `local` modulation — `rmse_testing`
      std/mean ≈ 25%, `rmse_storm_eunice_2022` std/mean ≈ 15%. That's large —
      large enough that several single-run comparisons elsewhere in this
      debate involving 317-station BiLinear specifically should be treated
      with real caution until repeated. Not yet known whether this variance
      is BiLinear-specific (317-station scale, or the modulation branch) or
      general at this scale — worth checking against a repeat-run Linear/317
      baseline before generalizing.
13. **Ensembling / model combination** *(flagged)* — combining the
    already-trained models, untried.
14. **Feature engineering** *(flagged)* — derived quantities (pressure
    gradient, inverse-barometer term, wind-stress curl) from the same raw
    inputs.
15. **Target/label definition** *(flagged)* — the tide/surge decomposition
    used to build the target may carry its own label noise.
16. **Physics-informed / hybrid approaches** *(flagged)* — ML residual on a
    reduced physical model, rather than end-to-end learning.
17. **Wind-stress Charnock parameterization mismatch.** DCSM-FM's forcing is
    confirmed ERA5 (per item 11), but likely used ERA5's own space/time-
    varying Charnock parameter, while `src/wind_stress.jl` uses a constant
    `α_ch = 0.041` (its own docstring: "a first relatively simple version").
    *User's estimate:* impact is probably small. *Literature counterpoint:*
    [Özkan, Verlaan, Muis & Zijl 2025, Ocean Dynamics](https://pmc.ncbi.nlm.nih.gov/articles/PMC12276138/)
    found constant vs. variable Charnock changed storm Xaver's RMSE from
    0.21m to 0.14m and North Sea surge locally by up to 0.20m — potentially
    a large share of our own `rmse_storm_eunice_2022` (0.10–0.23m range) if
    the effect transfers, in tension with the user's prior.
    - **Test:** cheap sensitivity check first — rerun with a smaller constant
      `α_ch` (no new data needed) and see how much storm RMSE moves, before
      sourcing/integrating ERA5's actual varying `chnk` field.
    - Complexity: 1/5 · Likelihood of success: 4/5 · Potential gain: 3/5 (literature suggests real effect; user expects smaller) · Priority: 4/5.
18. **Tide/surge trained separately** *(flagged)* — tide and surge are two
    independently-trained models; even the interaction models (item 2) only
    feed a frozen, pretrained tide into surge one-directionally, with no
    gradient flowing back into the tide model. Distinct from item 15 (label
    noise in the decomposition) and item 16 (physics-residual framing) —
    this is about the *training procedure* itself, not the data or the
    architecture family.

## Debate summary

The debate started from an observation: current surge and surge+interaction
experiments (Linear / Conv / Attention / BiLinear families, 5- and
317-station) converge to similar `rmse_testing`, and it wasn't obvious
whether that meant accuracy was near-maxed-out or whether real headroom
remained. Working through it argument by argument (11 rounds, alternating
"can" and "cannot," each checked against the actual code and experiment
artifacts rather than argued in the abstract) turned up a consistent shape:
several of the *architecture*-side explanations for the plateau didn't hold
up well under inspection — loosening inductive bias (Conv → Attention →
BiLinear) hasn't reliably beaten the plain `Dense`-only `LinearSurgeModel`,
loss curves are already flat under the current LR-decay schedule with no
sign of a hidden late-training breakthrough, and train/test RMSE gaps are
already near zero, leaving little for classic regularization tricks
(dropout, DropConnect) to fix. At the same time, several of the *apparent*
capacity/data-quantity failures turned out to be measurement artifacts
rather than real findings: BiLinear at 317 stations was compared against
Linear/Attention while still running pre-consolidation hyperparameters
(`nepochs=20`/`nlags=16`/no weight decay vs. the unified `50`/`48`/`1e-4`
baseline), and the 5yr→20yr "more data" comparison, while a fair
apples-to-apples test on its own terms, doesn't by itself say much about
extending to 1950 given DCSM-FM's target only covers 2000–2023 in this repo
and pre-1980 ERA5 is less homogeneous besides.

Where the debate consistently landed, instead, was the *input/data* side and
one specific, very consistent failure mode: every model tested, regardless
of architecture or station count, degrades ~70–95% on the one extreme storm
event checked (`storm_eunice_2022`) relative to its overall testing RMSE —
by far the most stable pattern found anywhere in this exercise. The leading
candidate explanation is that ERA5's sparse point-sampling and known
peak-wind underestimation during extreme events caps what any architecture
can extract for storms specifically — though that's weakened, not
eliminated, by the user's confirmation that DCSM-FM's own forcing is indeed
ERA5, making the model's input and its target self-consistently weak rather
than genuinely information-starved for the raw wind field itself. One
confirmed remaining difference (item 17): DCSM-FM likely used ERA5's actual
space/time-varying Charnock parameter for wind stress, while this repo's own
`wind_stress.jl` uses a constant value — literature on exactly this
mechanism reports real storm-specific effects (up to 0.20m locally in the
North Sea), in tension with the user's own estimate that the impact here is
small; a cheap sensitivity test is queued to settle it. None of this got
resolved outright; instead it converged on a ranked list of concrete, checkable tests
(the "Test priority table" and per-item tests above) rather than a verdict,
which was always the actual goal — the can/cannot framing was scaffolding to
surface and stress-test each hypothesis against the repo's real experiment
data, not a debate to be won.

Notes worth keeping in mind going forward:
- `LinearSurgeModel` (and BiLinear's surge branch) is literally a single
  `Dense` layer with identity activation — no hidden layers, no
  nonlinearity. `DenseSurgeModel` now provides the unstructured-deep-MLP test
  (item 2) and tracks `ConvSurgeModel` closely at every timespan checked — no
  inductive-bias signal so far. `DenseSurgeInteractionModel` (same
  unstructured idea, concatenating tide with forcing instead of BiLinear's
  narrow physically-motivated gate) is now fully run and swept (item 2): it's
  the single largest gain found anywhere in this doc at 5yr/20yr (~16–28%
  over both no-tide Dense and BiLinear; a wash at 1yr), and after sweeping
  width/depth/patience a
  5-station `rmse_testing` floor of ≈0.082–0.083 (20yr) is reasonably well
  characterized for the current input configuration.
- Item 2b (point-density/layout sweep) is now closed out too: full writeup
  in `input_sensitivity.md`. A controlled 9→24-point sweep, output stations
  held fixed, found input layout is a real (~7–8%) but secondary lever —
  well behind tide's +16–28% at 5yr/20yr (item 2 above) and still dwarfed by the
  still-unexplained storm-specific gap below. Confirmed to transfer to
  `DenseSurgeInteractionModel`. Item 2a (EOF/spectral analysis of the full
  ERA5 field) is postponed by user decision — it's being superseded by an
  adjoint for the shallow-water model, currently in development, which will
  give input sensitivities directly rather than the EOF's indirect variance
  proxy, once available.
- The generic training loop itself has no `Random.seed!` call, but
  `scripts/parameter_sweep.jl` already demonstrates a working, reproducible
  seeding pattern (`Random.seed!(rep)` before each in-process `train()`
  call) — it's just never been applied across different architectures, only
  within one model's hyperparameter sweeps. So zero repeat-run variance has
  been measured for any of the architecture-vs-architecture RMSE comparisons
  this whole debate is built on (item 12), but closing that gap is small
  scripting work (reuse the existing pattern over {configs} × {seeds}), not
  a training-loop code change. Several of the smaller reported "wins" (a few
  percent) are plausibly
  within seed-to-seed noise, unverified either way.
- BiLinearSurgeInteractionModel at 317 stations is not yet a controlled
  comparison against Linear/Attention — it was left on pre-consolidation
  training settings, confounding its "more capacity, more noise" result.
- The storm-specific ~70–95% RMSE degradation (vs. overall testing RMSE) is
  the single most consistent, unexplained finding in the whole exercise —
  present in every model and scale checked.
- Six categories were flagged as gaps but never argued through at all:
  ensembling existing trained models, feature engineering on the same raw
  inputs, the tide/surge decomposition's own potential label noise,
  physics-informed/hybrid ML+numerical-model approaches, tide and surge
  being trained as separate models rather than jointly, and (see above)
  run-to-run statistical variance (items 12–16, 18).
- Three corrections surfaced mid-debate and are folded into the items above
  rather than left as open contradictions: DCSM-FM extension is costly, not
  blocked (item 10); DCSM-FM's own forcing is confirmed ERA5, weakening (not
  eliminating) the ERA5-storm-underestimation ceiling (item 11); and the one
  confirmed remaining forcing difference — constant vs. ERA5's actual
  varying Charnock parameter — is split out as its own item with a cheap
  sensitivity test queued (item 17).
