# Notes on tensor dimensions

## Purpose

These notes are a working document while we cross-check the **tensor
layouts actually used in the code** against the **notation and background
material** in `notation.md` (and related documents).

The goal is twofold:

1. Make explicit which axes (point `p`, quantity `q`, time-lag `Δt`,
   batch-time `t`, etc.) each model expects, and in which order.
2. Identify inconsistencies — between models, between code and
   documentation, or between code and the conceptual story we want to
   tell — and decide whether to (a) update the docs to match reality, or
   (b) refactor the code towards a more consistent convention.

These notes are not yet a recommendation; they are a survey. Conclusions
and any clean-up actions will be drawn at the end.

## Note 1 — `preprocess` shape vs. layer-ready shape (ConvSurgeModel)

There are **two distinct shapes per model** that we have to keep
separate:

- the **preprocess shape**: the tensor that `preprocess` returns. This
  is a generic 4D scratch shape, intentionally layer-agnostic, that the
  abstract pipeline can hand to any subclass.
- the **layer-ready shape**: the shape the Flux layer actually consumes,
  after any `reshape` performed inside `forward` / inside the `Chain`.

For `ConvSurgeModel` (`src/models/ConvSurgeModel.jl:70`):

```julia
chain = Chain(
    x -> reshape(x, nlags, n_in, size(x, 2)),   # (1, q·p, lag, t) → (lag, q·p, t)
    [Conv((filtersize,), ch_seq[i] => ch_seq[i+1], relu; pad=SamePad())
     for i in 1:length(ch_seq)-1]...,
    Flux.flatten,
    Dense(nlags * channels[end] => nstations),
)
```

| Stage | Shape | Axis meaning |
|---|---|---|
| `preprocess` output | `(1, q·p, lag, t)` | leading `1` is a placeholder; `q·p` flattened channels; lag steps; batch-time |
| After `Chain[1]` reshape | `(lag, q·p, t)` | spatial (slid over by Conv); channels; batch |
| Conv input expectation (Flux 1D) | `(spatial, channel, batch)` | matches the reshape |

So for the **convolution itself** the axis ordering is unambiguous and
matches Flux's `(spatial, channel, batch)` convention: lag is the
spatial axis, `q·p` are channels, `t` is batch.

The leading singleton `1` in the preprocess output is **not** a
conceptual axis — it is a storage convention that makes the abstract
pipeline simpler (every model receives a 4D tensor). Each model reshapes
it to whatever its layer wants.

### Consequence for the notation doc

When we describe the model in index notation we should talk about the
**layer-ready shape** (`lag, q·p, t` here), not the preprocess shape.
The fact that `preprocess` returns a 4D scratch tensor is an
implementation detail of the abstract pipeline and belongs in the
design doc, not the notation doc.

When we describe the tensor layout used in code (e.g. for a "memory
layout" table) we have to be explicit about *which* shape we mean.

## Note 2 — The `Chain[1]` reshape in ConvSurgeModel looks like a bug

Following on from Note 1, the *intended* mapping is

```
preprocess output      (1, q·p, lag, t)
forward flatten        (q·p · lag, t)
Chain[1] reshape       (lag, q·p, t)        ← "conv-ready"
Conv1D                 slides over lag, q·p as channels, t as batch
```

The concern is that Julia's `reshape` **does not permute memory**; it
reinterprets the same flat buffer with new dimensions. So if the
underlying memory order doesn't already match the new shape, the result
is a *scrambled* tensor, not a transposed one.

### Memory trace

`preprocess` (`src/models/AbstractSurgeModel.jl:126-131,144`) builds

```julia
x = zeros(Float32, 3*nwind, nlags, nvalid)
for (i, t) in enumerate(valid_range)
    x[1:nwind,           :, i] = stress_x[:, t-nlags+1:t]
    x[nwind+1:2*nwind,   :, i] = stress_y[:, t-nlags+1:t]
    x[2*nwind+1:3*nwind, :, i] = press[   :, t-nlags+1:t]
end
tensor = reshape(x, 1, 3*nwind, nlags, nvalid)
```

So `x` has shape `(q·p, lag, t)` and (Julia is column-major) memory
order **q·p fastest → lag → t**. Adding the leading singleton via
`reshape` doesn't move any data.

`forward` (`src/models/ConvSurgeModel.jl:88-93`) does

```julia
_, nfeatures, nlags_dim, ntimes = size(x)
x_flat = reshape(x, nfeatures * nlags_dim, ntimes)
```

So `x_flat` is `(q·p · lag, t)` with the inner index running `q·p`
inside `lag` (still memory order q·p fastest within each column).

Then `Chain[1]` (`src/models/ConvSurgeModel.jl:70`) does

```julia
x -> reshape(x, nlags, n_in, size(x, 2))
```

claiming to produce `(lag, q·p, t)`. But because `reshape` preserves
the flat buffer, the new axis interpretation reads memory with `nlags`
varying fastest, while the **data** was stored with `q·p` varying
fastest.

### Concrete example

Take `nlags = 4`, `n_in = q·p = 3`. The "lag axis at b=1" in the new
shape reads, in order:

| New index `(a, b=1)` | flat pos | Original `(c, l)` |
|---|---|---|
| `a=1` | 1 | `c=1, l=1` |
| `a=2` | 2 | `c=2, l=1` |
| `a=3` | 3 | `c=3, l=1` |
| `a=4` | 4 | `c=1, l=2` |

The conv kernel would be sliding over a **mixture of channel and lag
positions**, not a coherent lag sequence. Numerically the code runs;
physically the 1D convolution is being asked to find structure in
interleaved-channel data.

The bug is invisible iff `nlags == 3·nwind` — then the two axes have
the same length and the "swap" looks like a no-op (it still
scrambles, but the scramble happens to be a valid permutation of
indices of the same length, and the conv kernel learns *something*).

### Two clean fixes

1. **Use `permutedims`** instead of `reshape` — actually moves the
   data.
   ```julia
   x -> permutedims(dropdims(x; dims=1), (2, 1, 3))   # (lag, q·p, t)
   ```
   Costs a copy each forward pass.

2. **Build the preprocess output in the conv-ready order to start
   with.** I.e. make `x = zeros(Float32, nlags, 3*nwind, nvalid)` and
   assign with the axes swapped:
   ```julia
   x[:, 1:nwind,           i] = transpose(stress_x[:, t-nlags+1:t])
   ...
   tensor = reshape(x, 1, nlags, 3*nwind, nvalid)
   ```
   No runtime copy; the only cost is the transpose at preprocess time.

Fix 2 is cheaper and more honest, but it changes the contract of
`preprocess` — and the same scratch tensor is consumed by
`LinearSurgeModel` and `AttentionSurgeModel` too. Before changing,
we should check whether the other consumers are compatible (or
whether they also rely on the q·p-first ordering for a flatten-into-
Dense pattern, in which case the right answer is for *each* model's
`forward` to choose its own layout via an honest `permutedims`).

### Status — RESOLVED (step 20e/f)

- [x] Verified by a synthetic test (`stress_x[p,t] = 100*p + t`): the old
  `reshape` made the Conv spatial axis read `[101, 201, 301, 401]` (a feature
  ramp) instead of `[101, 102, 103, 104]` (a lag ramp). Now a permanent
  regression test, "ConvSurgeModel conv-ready layout (no scramble)".
- [x] Decided: **neither fix 1 (`permutedims` in `forward`) nor a global fix 2**,
  but the Note-5 design — `ConvSurgeModel.preprocess` builds the conv-ready
  `(nlags, 3·nwind, nvalid)` layout directly (transposing each lag window with
  `permutedims` at preprocess time, not per forward pass), and the `Chain[1]`
  reshape is gone.
- [x] Check that `LinearSurgeModel` is not silently affected by either
  fix — see Note 3.
- [x] Check `AttentionSurgeModel` — see Note 4.

## Note 3 — LinearSurgeModel is unaffected by the ordering issue

`LinearSurgeModel` consumes the same `preprocess` output as
`ConvSurgeModel` but does **not** suffer from the Note 2 bug, and
understanding why is informative for how we plan the fix.

### Trace

```julia
# preprocess output: (1, q·p, lag, t)
# memory order: q·p fastest → lag → t

# forward (src/models/LinearSurgeModel.jl:60-63)
_, nfeatures, nlags_dim, ntimes = size(x)
x_flat = reshape(x, nfeatures * nlags_dim, ntimes)   # (q·p · lag, t)
y      = model.flux_model(x_flat)                     # Dense(3*nwind*nlags => nstations)
```

The flattened input vector has structure **`q·p` varying fastest inside
`lag`**: `[all q·p at lag=1, all q·p at lag=2, …]`. There is no second
reshape, and the Dense layer eats the flat vector directly.

### Why this is fine

`Dense` is **permutation-invariant on its input vector**: whatever
ordering the flat vector has, the Dense weight matrix simply learns to
associate each scalar position with the right output. So as long as
training and inference use the *same* ordering (they do — both go
through the same `preprocess` + `forward`), the model is correct.

In contrast, `Conv` is **not** permutation-invariant: it assumes the
spatial axis is a coherent 1D sequence. That is exactly why the broken
reshape in `ConvSurgeModel` is a bug for `Conv` but not for `Dense`.

### Implication for the fix in Note 2

If we adopt **Fix 2** — change `preprocess` to produce
`(1, lag, q·p, t)` directly — then `LinearSurgeModel` keeps working,
because the flatten would just yield a different but still consistent
ordering (now `lag` fastest inside `q·p`). The Dense weights would
permute themselves at training time and the model would be equivalent.

So `LinearSurgeModel` does **not** constrain the choice between fix 1
and fix 2.

### General rule

| Layer type | Cares about flat-input ordering? | Cares about axis ordering of a multi-D input? |
|---|---|---|
| Dense (after flatten) | no | n/a |
| Conv | n/a (input is multi-D) | **yes** — first axis is spatial |
| Attention | depends on which axis is contracted | **yes** |

The implication for the package as a whole: it is only the *spatial-
aware* models (Conv, attention, anything that interprets a particular
axis as having structure) that pin down the layout convention. Dense
models inherit whatever the spatial-aware models decide.

## Note 4 — AttentionSurgeModel

`AttentionSurgeModel` is a different case because it **overrides**
`preprocess` and produces a tuple of two tensors with their own
layout. So it is not consuming the same scratch tensor as
`ConvSurgeModel` and `LinearSurgeModel`.

### preprocess output

(`src/models/AttentionSurgeModel.jl:151-208`)

```julia
x_wind    = (3*nwind, nlags, nvalid)        # same fill pattern as AbstractSurgeModel
x_station = (6, nstations, nvalid)          # cos/sin of lat, lon, day-of-year
```

The fill pattern for `x_wind` is identical to the abstract
`preprocess`:

```julia
x_wind[1:nwind,           :, i] = stress_x[:, t-nlags+1:t]
x_wind[nwind+1:2*nwind,   :, i] = stress_y[:, t-nlags+1:t]
x_wind[2*nwind+1:3*nwind, :, i] = press[   :, t-nlags+1:t]
```

So in memory: **point fastest inside quantity → lag → batch-time**.

### branch_net

(`src/models/AttentionSurgeModel.jl:105-111`)

```julia
Chain(
    embed,                                  # (3*nwind, nlags, t) → (nembed, nlags, t)
    pos_embed,                              # adds positional encoding along nlags
    [Transformer(nembed, nheads) ...]...,   # operates on axis 2 = nlags (sequence)
    deembed,                                # → (3*nwind, nlags, t)
    x -> reshape(x, (nwind, 3, nlags, :)),  # split q·p into (p, q)
)
```

Two points to check:

1. **Transformer treats axis 2 as the sequence axis.** Axis 2 holds
   `nlags`, which is a meaningful 1D sequence (history time steps).
   Physically correct.
2. **The reshape `(3·nwind, …) → (nwind, 3, …)`** — does it scramble
   like the one in Note 2?

For the reshape to mean `(point, quantity)` correctly, memory must
already be laid out as `(p=1, q=1), (p=2, q=1), …, (p=nwind, q=1),
(p=1, q=2), …`. That is exactly what the `preprocess` fill pattern
produces (point varies fastest within quantity). So the reshape **is**
valid here — memory and intended interpretation agree.

### AttentionSurgeFlux

(`src/models/AttentionSurgeModel.jl:37-47`)

```julia
trunk_out  = trunk_net(x_station)                 # (nwind, nstations, t)
branch_out = (nwind, 3, nlags, t)
merged = batched_mul(
    batched_transpose(trunk_out .* m.adjacency),  # (nstations, nwind, t)
    reshape(branch_out, (nwind, :, t)),           # (nwind, 3*nlags, t)
)                                                 # → (nstations, 3*nlags, t)
downsample = Conv((1,), 3*nlags => nlags, …)      # 1×1 conv on channel axis
# output: (nstations, nlags, t)
```

The `reshape(branch_out, (nwind, :, t))` only collapses two adjacent
axes — memory ordering preserved, no scramble.

### Verdict

`AttentionSurgeModel` is **not affected** by the Note 2 bug:

- The Transformer sees `nlags` as a coherent sequence axis (correct).
- The only `reshape` that splits a previously merged axis
  (`3·nwind → (nwind, 3)`) is consistent with the fill order of
  `preprocess`.
- All other reshapes only collapse adjacent axes (always safe).

### Implication for the fix

`AttentionSurgeModel.preprocess` is an **override** — independent of
`AbstractSurgeModel.preprocess`. So **Fix 2** from Note 2 (changing
the abstract preprocess to produce a conv-ready order) would *not*
touch `AttentionSurgeModel`. The two pipelines are decoupled.

Fix scope is therefore:

- `AbstractSurgeModel.preprocess` (the abstract one)
- `LinearSurgeModel.forward` (still works regardless — Note 3)
- `ConvSurgeModel.forward` (this is the model being fixed)

`AttentionSurgeModel` requires no changes.

### Latent concern (convention)

`AttentionSurgeModel.preprocess` *also* relies on the
"point-fastest-inside-quantity" fill pattern, because the downstream
`reshape(x, (nwind, 3, nlags, :))` only works under that ordering. If
we ever change the fill pattern (e.g. switch to "quantity-fastest-
inside-point" by writing `x_wind[(p-1)*3+1:p*3, :, i] = [stress_x[p,:]; stress_y[p,:]; press[p,:]]`),
this reshape would silently scramble. Worth pinning down as part of
the layout convention before any refactor.

## Note 5 — Cost of `permutedims` and what that implies for the design

A natural question after Note 2 is: "if we unify `preprocess` and
fix layout per-model with `permutedims` in `forward`, how expensive
is that?" The answer matters because it tells us whether *one* shared
`preprocess` plus per-model permutes is acceptable, or whether each
model needs its own `preprocess`.

### What `permutedims` actually does

- In Julia, `permutedims(A, perm)` **always materialises a new array**
  — it is an O(N) memory-bound copy.
- It is **not** a view. The lazy alternative is `PermutedDimsArray(A,
  perm)`, but downstream consumers (especially cuDNN / NNlib `Conv`)
  typically force materialisation anyway for cache- and SIMD-friendly
  access. So the lazy version is rarely a real win in practice.
- The cost is roughly memory bandwidth, not compute, but the access
  pattern is **strided**, so it runs slower than a contiguous
  `memcpy` (typically 2–10× slower depending on stride pattern).

### Rough numbers

For a surge-shaped tensor `(27, 16, 1024)` ≈ 1.7 MB Float32:

| Operation | CPU | GPU |
|---|---|---|
| Contiguous copy | ~50 µs | ~5 µs |
| `permutedims` (swap axes 1↔2) | ~200 µs | ~20 µs |
| One `Conv` forward | low ms | ~100 µs |

So **per forward pass**, a `permutedims` is roughly **5–20% overhead**
on top of the conv itself. Per backward pass roughly the same.

### When does it actually hurt?

- **Inside `preprocess`** (called once per dataset / per batch
  construction): cost is irrelevant. Run `permutedims` freely.
- **Inside `forward`** (every iteration of training): cost is real but
  not catastrophic. For 100 epochs × 1000 batches × 200 µs ≈ 20 s
  extra. For 20-year experiments with much larger tensors it scales
  linearly.
- **On GPU**: smaller absolute overhead, but Flux/cuDNN sometimes
  inserts permutations internally anyway when the data layout doesn't
  match what cuDNN expects. So the apparent "savings" of avoiding
  `permutedims` may already be eaten elsewhere.

### Three design options

Ranked by per-pass performance:

1. **Per-model `preprocess`, no `forward`-time permute.** Each model
   produces exactly the layout its layer needs. Fastest. Cost: code
   duplication and a less unified abstract pipeline.
2. **Unified `preprocess` + `permutedims` in `forward`.** Clean API,
   ~5–20% per-pass overhead. Acceptable for prototyping and small /
   medium runs.
3. **Unified `preprocess` + `PermutedDimsArray` (lazy) in `forward`.**
   Looks free, but usually forces a materialised copy downstream
   anyway. Rarely worth the indirection over option 2.

### Recommendation

For this package — where models are heterogeneous (Dense, Conv,
Attention, Transformer, DeepONet) and each really wants a specific
axis order — **option 1 (per-model `preprocess`) is the honest
design**. The "unification" the abstract pipeline currently suggests
is partly an illusion: the only thing genuinely shared is the **data
extraction** (`stress_x`, `stress_y`, `press`, slicing the lag
windows). The **arrangement** into a tensor is layer-specific.

A clean refactor would split `preprocess` into:

- a shared **data-extraction helper** that returns
  `stress_x`, `stress_y`, `press` already sliced into lag windows
  (i.e. arrays of shape `(nwind, nlags, nvalid)`); and
- a **per-model assembly step** that arranges those into the model's
  preferred tensor layout, with no `permutedims` needed at all.

That gives the best of both: no duplication of the expensive
data-extraction work, and each model is honest about its own layout.
The current `ConvSurgeModel` bug (Note 2) essentially exists because
the abstract design tried to skip the per-model assembly step and
relied on a `reshape` to do a transpose's job.

## Note 6 — Unifying the Flux-model call signature (tensor vs. tuple)

Some Flux models in this package take a **single tensor** as input
(`LinearSurgeModel`, `ConvSurgeModel`), and some take a **tuple of
tensors** (`AttentionSurgeModel`'s
`AttentionSurgeFlux(x_station, x_wind)`, and likewise the wave and
interaction models). This is a further departure from a unified
preprocessing — and it changes the call signature seen by `forward`.

The natural question: can we have a single `y = flux_model(x)` call
that works whether `x` is a tensor or a tuple of tensors? Julia's
multiple dispatch makes this a one-liner.

### The pattern

For each Flux model with a multi-arg call, add a tuple-splat wrapper:

```julia
# The "real" method (e.g. AttentionSurgeFlux already has this):
(m::AttentionSurgeFlux)(x_station, x_wind) = ...

# One-line tuple wrapper:
(m::AttentionSurgeFlux)(x::Tuple) = m(x...)
```

For single-tensor Flux models the existing `Array`-typed method is
already enough. Now `y = flux_model(x)` works uniformly — Julia
dispatches on the type of `x`.

### Two levels at which to apply this

1. **At the `flux_model` level** (preferred). Add the tuple wrapper to
   every multi-arg Flux model. Then `forward` can be a single unified
   definition in the abstract pipeline:
   ```julia
   function forward(model::AbstractFluxModel, x)
       y = get_flux_model(model)(x)         # works for tensor or tuple
       return reshape(y, size(y, 1), 1, size(y)[end])
   end
   ```
   No more `forward(model, x::Tuple)` override is needed for
   `AttentionSurgeModel`.

2. **At the `forward` level** (current approach). Keep distinct
   `forward(model, x::Array)` / `forward(model, x::Tuple)` methods.
   More boilerplate, and the variability is spread across multiple
   methods instead of being localised.

### Compatibility with autodiff

Zygote differentiates through tuples natively, so
`Flux.withgradient(m) do model; loss(model(x), y); end` works
whether `x::Array` or `x::Tuple`. Either level is safe for training.

### Subtlety with `Flux.DataLoader`

`DataLoader` iterates whatever you give it. For a tuple-input model
the current code passes `(x_st, x_w, y)` and unpacks three items in
the inner loop. Under level 1, the cleanest pattern is to package the
inputs as a tuple and pass `((x_st, x_w), y)` so the loader yields
`(x, y)` pairs uniformly. Small adjustment, not a real obstacle.

### Preferred option

**Option 1** (tuple-splat wrapper on every Flux model) is preferred.
It pushes the input-arity variability into one well-defined extension
point — the Flux model's own call signature — rather than into
multiple `forward` overrides on the model wrapper. This is exactly the
place where heterogeneity belongs: each model declares what it eats,
and the abstract pipeline calls it uniformly.

Concrete actions implied:

- Add `(m::ModelFlux)(x::Tuple) = m(x...)` to every multi-arg Flux
  model (`AttentionSurgeFlux`, and the equivalent in wave/interaction
  models).
- Collapse the per-model `forward(model, x::Tuple)` overrides into a
  single generic `forward(model::AbstractFluxModel, x)` in
  `abstract_flux_model.jl`.
- Adjust `Flux.DataLoader` usage to yield `(x, y)` pairs, with `x`
  being a tuple where appropriate.

## Note 7 — The output-side singleton and the emerging design

Looking at the output side mirrors what Note 5 found for the input
side. Every model's `forward` currently ends with

```julia
return reshape(y, size(y, 1), 1, ntimes)         # add singleton "feature" axis
```

purely so the abstract `postprocess!` can do `y[:, 1, :]`. The `1` is
a placeholder for a future multi-feature output axis. For now it
exists only to satisfy the contract.

### Two options

**Option A — move the reshape into `postprocess!`.** Conservative:

```julia
function postprocess!(output, model::AbstractSurgeModel, y::AbstractArray)
    output["surge"].values .= reshape(y, size(y, 1), size(y)[end])
end
```

`forward` can then return whatever shape it likes; `postprocess!`
adapts.

**Option B — abandon the singleton.** Each `forward` returns its
natural shape, `postprocess!` consumes it directly:

```julia
# in each forward:
return y                          # (nstations, ntimes) for Linear/Conv
return y[:, end, :]               # (nstations, ntimes) for Attention

# in postprocess!:
function postprocess!(output, model::AbstractSurgeModel, y::Array{Float32, 2})
    output["surge"].values .= y
end
```

### Future multi-feature outputs

The singleton placeholder isn't really "saving work" for the future
multi-feature case — that case needs custom dispatch anyway:

```julia
function postprocess!(output, model::MultiOutputModel, y::Array{Float32, 3})
    output["surge"].values .= y[:, 1, :]
    output["tide"].values  .= y[:, 2, :]
end
```

So the reserved axis buys us nothing concrete.

### Decision

**Option B** — abandon the singleton — is the chosen approach. It is
the output-side mirror of Note 5: each model honest about its own
shape; the abstract pipeline only asserts what is *actually* shared,
which is "we produce predictions per station per time" (a 2D array),
not "we produce a 3D tensor with a reserved feature axis".

Concrete actions implied:

- Drop the trailing `reshape(y, size(y, 1), 1, ntimes)` from each
  `forward` (`LinearSurgeModel`, `ConvSurgeModel`, `AttentionSurgeModel`,
  …) — return the natural 2D shape `(nstations, ntimes)` directly.
- Update `postprocess!(::AbstractSurgeModel, …)` to accept a 2D `y`
  and assign directly to `output["surge"].values`.
- For future multi-feature output models, add a new
  `postprocess!` dispatch on the 3D shape — no contract change needed
  on the existing 2D-output models.

## Conclusion — emerging design principle

Taking Notes 1–7 together, the cleanest design pattern is:

> **Each Flux model declares what it wants as input and output to
> run efficiently. `preprocess` and `postprocess!` convert between
> the *standardized* external interface (`Dict{String, TimeSeries}`)
> and the model-specific *optimised* tensor(s).**

The standardisation happens at the **package boundary** — the
`Dict{String, TimeSeries}` format that callers see — not inside the
tensor pipeline. Between `preprocess` and `postprocess!`, each model
is free to use whatever tensor shapes, tuples of tensors, or other
data structures it needs for efficient computation.

Concretely this means:

1. **External contract (stable):** `predict(model, input::Dict{String,TimeSeries}) -> Dict{String,TimeSeries}`.
2. **`preprocess`** (per model): standardised Dict → model-specific
   tensor(s), arranged in the layout the model's layers actually need.
3. **`forward`** (per model): model-specific tensor(s) →
   model-specific output tensor(s). No imposed shape contract.
4. **`postprocess!`** (per model): model-specific output tensor(s) →
   standardised Dict.
5. **Shared code** lives in helpers called by `preprocess` /
   `postprocess!` (e.g. lag-window extraction, stress conversion,
   `TimeSeries` allocation), not in shape contracts on the tensor
   pipeline.

This is honest about what is genuinely shared (the external Dict
format and the data-extraction utilities) and what is not (every
single aspect of the tensor pipeline itself). It also makes the bug
in Note 2 structurally impossible: there's no abstract
"layout-agnostic" tensor for a layer-specific reshape to silently
scramble.
