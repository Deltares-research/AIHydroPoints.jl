# ProductTideModel.jl
#
# Concrete subtype of AbstractTideModel using a multiplicative product of station
# and Doodson encodings, followed by residual gating layers.
#
# Inspired by TideInputLayer / TideLayer from the old tides.jl design, adapted
# to use the 4-feature (cos/sin lat, cos/sin lon) station encoding from
# AbstractTideModel.preprocess rather than one-hot station indices.
#
# Inherits from AbstractTideModel without override:
#   preprocess, postprocess!, train_model!, plot_series, save_params, load_params!

using Flux

# ──────────────────────────────────────────────────────────────────────────────
# Internal Flux layers
# ──────────────────────────────────────────────────────────────────────────────

"""
    ProductInputLayer

Encodes station coordinates and Doodson arguments into a shared feature space,
then forms their element-wise product as the initial representation.

- `station_proj`: `Dense(4 → nfeats, identity; bias=false)` — applied to each
  station's `[cos(lat), sin(lat), cos(lon), sin(lon)]` encoding.
- `doodson_proj`: `Dense(2*nfreqs → nfeats, identity; bias=false)` — applied to
  the `[cos(doodson); sin(doodson)]` time-step encoding.

Input shapes:
- `x_station :: (4, nstations, ntimes)`
- `x_doodson :: (2*nfreqs, ntimes)`

Output shape: `(nfeats, nstations, ntimes)` — the element-wise product,
broadcast over stations.
"""
struct ProductInputLayer{S, D}
    station_proj :: S
    doodson_proj :: D
end

function (l::ProductInputLayer)(x_station, x_doodson)
    s = l.station_proj(x_station)                   # (nfeats, nstations, ntimes)
    d = l.doodson_proj(x_doodson)                   # (nfeats, ntimes)
    d = reshape(d, size(d, 1), 1, size(d, 2))       # (nfeats, 1, ntimes)
    return s .* d                                    # broadcast → (nfeats, nstations, ntimes)
end

@Flux.layer ProductInputLayer

"""
    ProductGatingLayer

Residual gating layer: `output = x + gate(x) * x` where `gate` is a Dense(relu)
layer.  Operates on `(nfeats, nstations, ntimes)` — Flux's Dense applies over
dim 1, so each station and time step is processed independently.
"""
struct ProductGatingLayer{T}
    gate :: T
end

(l::ProductGatingLayer)(x) = x .+ l.gate(x) .* x

@Flux.layer ProductGatingLayer

"""
    ProductTideFlux

Internal Flux model combining `ProductInputLayer`, a chain of
`ProductGatingLayer`s, and a final `Dense` output projection.

Not exported — construct via `ProductTideModel`.
"""
struct ProductTideFlux{I, G, O}
    input_layer   :: I
    gating_layers :: G
    output_layer  :: O
end

function (m::ProductTideFlux)(x_station, x_doodson)
    x = m.input_layer(x_station, x_doodson)   # (nfeats, nstations, ntimes)
    x = m.gating_layers(x)                    # (nfeats, nstations, ntimes)
    y = m.output_layer(x)                     # (1, nstations, ntimes)
    return y[1, :, :]                         # (nstations, ntimes)
end

# Tuple-call form so the generic train_model!/forward can invoke every flux model
# uniformly as `m(x)`.
(m::ProductTideFlux)(x::Tuple) = m(x...)

@Flux.layer ProductTideFlux

# ──────────────────────────────────────────────────────────────────────────────
# ProductTideModel struct and constructor
# ──────────────────────────────────────────────────────────────────────────────

"""
    ProductTideModel <: AbstractTideModel

Tide model that predicts water levels by forming the element-wise product of
a learned station encoding and a learned Doodson-argument encoding, then
processing the result through residual gating layers.

## Constructor

```julia
model = ProductTideModel(settings::Dict{String, Any})
```

Required key in `settings`: `"freqs"` (vector of tidal constituent names).

Optional key `"model_pars"` (Dict):
- `"nfeats"`  — feature dimension for all layers (default `64`)
- `"nlayers"` — number of `ProductGatingLayer`s (default `3`)

## Architecture

```
preprocess → x_station (4, nstations, ntimes)   [cos/sin lat, cos/sin lon]
             x_doodson (2*nfreqs, ntimes)        [cos/sin Doodson arguments]

ProductInputLayer:
    station_proj = Dense(4 → nfeats, identity; bias=false)
    doodson_proj = Dense(2*nfreqs → nfeats, identity; bias=false)
    output = station_proj(x_station) .* doodson_proj(x_doodson)
           → (nfeats, nstations, ntimes)

ProductGatingLayer × nlayers:
    output = x + Dense(nfeats → nfeats, relu)(x) * x
           → (nfeats, nstations, ntimes)

Dense(nfeats → 1) → (1, nstations, ntimes) → (nstations, 1, ntimes)
```
"""
mutable struct ProductTideModel <: AbstractTideModel
    flux_model :: ProductTideFlux
    settings   :: Dict{String, Any}
end

"""
    ProductTideModel(settings::Dict{String, Any}) -> ProductTideModel

Construct a `ProductTideModel` from `settings`.

Required key: `"freqs"`.  Optional key `"model_pars"` with `"nfeats"` and `"nlayers"`.
"""
function ProductTideModel(settings::Dict{String, Any})
    nfreqs  = length(settings["freqs"])
    mp      = get(settings, "model_pars", Dict{String, Any}())
    nfeats  = get(mp, "nfeats",  64)
    nlayers = get(mp, "nlayers", 3)

    input_layer   = ProductInputLayer(
        Dense(4 => nfeats, identity; bias=false),
        Dense(2 * nfreqs => nfeats, identity; bias=false),
    )
    gating_layers = Chain([ProductGatingLayer(Dense(nfeats => nfeats, relu))
                           for _ in 1:nlayers]...)
    output_layer  = Dense(nfeats => 1)

    flux_model = ProductTideFlux(input_layer, gating_layers, output_layer)
    return ProductTideModel(flux_model, settings)
end

get_flux_model(m::ProductTideModel) = m.flux_model
get_settings(m::ProductTideModel)   = m.settings

# forward, postprocess!, and train_model! are inherited from AbstractTideModel.
