# hatyan_core

A Julia package for tidal analysis and prediction, ported from the Python
[hatyan](https://github.com/Deltares/hatyan) package developed at Deltares.

> **Research repository — not for production use.**
> This is an experimental Julia port for research and exploration purposes only.
> For operational and production work, use the original Python
> [hatyan](https://github.com/Deltares/hatyan) package.
>
> **Disclaimer:** This port may contain errors in the computation of tidal
> constituents, nodal factors, or harmonic summation. Results have been spot-checked
> against Python hatyan reference values, but no guarantee is made as to their
> correctness. Always validate against the authoritative source before drawing
> any conclusions.

## Acknowledgements

`hatyan_core` is a Julia port of [hatyan](https://github.com/Deltares/hatyan),
developed and maintained by the Deltares team. All credit for the underlying
algorithms, constituent tables, and methodology belongs to the original authors.
If you use tidal analysis in your work, please refer to and cite the original
hatyan package.


## Features
- Perform harmonic tidal analysis using the Schureman method
- Compute Doodson astronomical arguments and Schureman nodal factors
- Reconstruct water levels with the harmonic cosine summation formula:
  `h(t) = Σ fᵢ · Aᵢ · cos(ωᵢ · Δt + v₀ᵢ + uᵢ − φᵢ)`
- Read and write time series in DONAR and NOOS formats


## Quick start: tidal prediction

```julia
using hatyan_core
using Dates

# Load tidal constituents from a DONAR analysis file
tc = read_donar_constituents("test_data/VLISSGN_ana.txt")
# => TidalConstituents: water level from VLISSGN, 1 location, 94 constituents

# Define the prediction period (10-minute steps over one day)
times = DateTime(2019, 1, 1) : Minute(10) : DateTime(2019, 1, 1, 23, 50)

# Run prediction with default settings (nodal factors on, evaluated at mid-point)
ts = prediction(tc, times)

# Access results
h = vec(get_values(ts))         # water levels in metres (Float32)
t = get_times(ts)               # Vector{DateTime}
println("Station : ", get_names(ts)[1])
println("Max tide: ", maximum(h), " m")
println("Min tide: ", minimum(h), " m")
println("Source  : ", get_source(ts))
# => Source: VLISSGN | prediction
```

### Prediction settings

`HatyanSettings` controls nodal factor behaviour:

| Field          | Default | Description |
|----------------|---------|-------------|
| `nodalfactors` | `true`  | Apply nodal amplitude (f) and phase (u) corrections |
| `fu_alltimes`  | `true`  | Recompute f/u at every timestep (more accurate) |
| `xfac`         | `false` | Apply x-factor correction to f |

```julia
# Nodal factors evaluated once at the midpoint of the period (faster, ~equivalent)
settings = HatyanSettings(nodalfactors=true, fu_alltimes=false, xfac=false)
ts = prediction(tc, times, settings)

# No nodal factor corrections at all
ts_nf = prediction(tc, times, HatyanSettings(nodalfactors=false))
```

## Quick start: tidal analysis

```julia
using hatyan_core
using Dates

# Load observed water levels from a DONAR file
obs = read_donar_timeseries("test_data/VLISSGN_obs19.txt")

# Choose a constituent set appropriate for the record length
# "year" = 95 constituents (A0 + 94), requires ~1 year of data
const_list = constituent_list("year")

# Fit the constituents by least-squares harmonic analysis
tc = analysis(obs, const_list)
# => TidalConstituents: water level from VLISSGN | analysis(schureman)

# Inspect results
A   = get_amplitudes(tc)           # Matrix{Float32} [locations × constituents]
phi = get_phases(tc)               # phases in degrees
names = get_constituent_names(tc)  # ["A0", "SA", "SM", "Q1", ...]

i_M2 = findfirst(==("M2"), names)
println("M2 amplitude : ", A[1, i_M2], " m")
println("M2 phase     : ", phi[1, i_M2], " °")
println("Source       : ", get_source(tc))
# => Source: VLISSGN | analysis(schureman)
```

### Analysis → prediction workflow

```julia
# Analyse observations to get constituents ...
tc = analysis(obs, constituent_list("year"))

# ... then predict water levels for any period
times_future = DateTime(2020, 1, 1) : Minute(10) : DateTime(2020, 12, 31, 23, 50)
ts_pred = prediction(tc, times_future)
# => Source: VLISSGN | analysis(schureman) | prediction
```

### Constituent list presets

| Name              | Data needed  | N constituents |
|-------------------|--------------|----------------|
| `"year"`          | ~1 year      | 95             |
| `"halfyear"`      | ~6 months    | 89             |
| `"month"`         | ~1 month     | 22             |
| `"springneap"`    | ~15 days     | 15             |
| `"day"`           | ~1 day       | 11             |
| `"tidalcycle"`    | ~12.4 hours  | 7              |
| `"all"`           | —            | all Schureman  |

## Reading and writing time series

```julia
# Read a DONAR water-level file
obs = read_donar_timeseries("test_data/VLISSGN_obs19.txt")

# Read a NOOS file collection
col = read_single_noos_file("test_data/VLISSGN_waterlevel_20180101_20180401.noos")

# Select a sub-period
ts_sub = select_timespan(obs, DateTime(2019, 1, 1), DateTime(2019, 1, 2))

# Write to NOOS format
write_single_noos_file("output.noos", ts_sub)
```

## Key types

| Type | Description |
|------|-------------|
| `TimeSeries` | Water-level time series: `values[locations × times]`, `times`, `names`, etc. |
| `TidalConstituents` | Harmonic constituents: `amplitudes[locations × constituents]`, `phases`, `constituent_names` |
| `HatyanSettings` | Prediction configuration |

### `TimeSeries` getters

```julia
get_values(ts)        # Matrix{Float32} [locations × times]
get_times(ts)         # Vector{DateTime}
get_names(ts)         # Vector{String}  station names
get_longitudes(ts)    # Vector{Float64}
get_latitudes(ts)     # Vector{Float64}
get_quantity(ts)      # String, e.g. "water level"
get_source(ts)        # provenance string, e.g. "VLISSGN | prediction"
```

### `TidalConstituents` getters

```julia
get_amplitudes(tc)        # Matrix{Float32} [locations × constituents], metres
get_phases(tc)            # Matrix{Float32} [locations × constituents], degrees
get_constituent_names(tc) # Vector{String}, e.g. ["A0", "M2", "S2", ...]
```

## Provenance tracking

Every operation appends a token to the `source` string so the data lineage is
always traceable:

```
"VLISSGN"
  └─ read_donar_timeseries    →  "VLISSGN"
  └─ analysis(schureman)      →  "VLISSGN | analysis(schureman)"
  └─ prediction               →  "VLISSGN | analysis(schureman) | prediction"
```

Loading pre-computed constituents and predicting skips the analysis token:

```
"VLISSGN"
  └─ read_donar_constituents  →  "VLISSGN"
  └─ prediction               →  "VLISSGN | prediction"
```

## Running the tests

```bash
cd hatyan_core
julia --project=. -e "using Pkg; Pkg.test()"
```
