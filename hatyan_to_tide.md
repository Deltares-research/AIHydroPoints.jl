
# Convert the Hatyan core computational routines from python to julia

The Hatyan core computational routines are currently implemented in python. We want to convert them to julia for better performance and ease of use from julia scripts. There are many extra options in the hatyan package that do not need in the julia version.

## Steps
1. create a summary of the python code [hatyan_summary.md](hatyan_summary.md)
2. select the core computational routines that are needed for the julia version
3. identify the key data structures used in the core routines (see below)
4. implement reading and writing routines, so we can test the julia code against the python code on the same data
5. determine the function signatures for the core routines, and implement them in julia
6. implement the core routines bottom-up (see implementation order below)

## Implementation for tidal analysis

Building bottom-up from the call graph, implement in this order:

6. **Constituent list utility** (`constituents.jl`) — add a `constituent_list(name::String) -> Vector{String}`
   function that returns predefined sets of constituent names. The standard North Sea set is
   `"year"` (94 components including `"A0"`). The list is hardcoded in
   `hatyan.git/hatyan/hatyan_core.py` (`get_const_list_hatyan`). This is a prerequisite so
   callers don't have to supply constituent names manually.

7. **Design matrix and least-squares solve** (`analysis.jl`) — implement `analysis_singleperiod`.
   For `m` observations and `N` constituents build the `m × 2N` design matrix:
   ```
   xmat[1:m, 1:N]    = f_i(t) .* cos.(ω_i .* t_s .+ v0_i .+ u_i(t))   # cosine columns
   xmat[1:m, N+1:2N] = f_i(t) .* sin.(ω_i .* t_s .+ v0_i .+ u_i(t))   # sine columns
   ```
   Solve `(xᵀx) β = xᵀy` with `LinearAlgebra.\\`, then convert to amplitude/phase:
   ```julia
   A_i = sqrt.(β[N+1:end].^2 .+ β[1:N].^2)
   φ_i = rad2deg.(mod.(atan.(β[N+1:end], β[1:N]), 2π))
   ```
   Special cases:
   - `"A0"` (mean water level) uses an all-ones sine column and a zero cosine column, and
     Python corrects `xTx[N, N] = m` to improve matrix conditioning.
   - If the solved `A0` phase is 180°, negate the amplitude and set phase to 0°.
   - Add `LinearAlgebra` to `[deps]` in `Project.toml`.

8. **Matrix condition check** (`analysis.jl`) — after forming `xTx`, compute
   `cond(xTx)` with `LinearAlgebra.cond`. Raise an error (or warning) if it exceeds a
   threshold (Python default: 20). This catches series that are too short for the requested
   constituent list or that contain duplicate constituents.

9. **`analysis` top-level function** (`analysis.jl`) — thin wrapper that:
   - Drops timesteps where `values` are `NaN` before the solve.
   - Calls `analysis_singleperiod` for each location independently.
   - Appends the method to the provenance trail:
     `"VLISSGN" → "VLISSGN | analysis(schureman)"`
   - Returns a `TidalConstituents` with `source` containing the `analysis(method)` token
     so that `prediction` can recover the method automatically.

   Function signature (see also [Function signatures](#function-signatures) below):
   ```julia
   analysis(ts, const_list, method="schureman", settings=HatyanSettings()) -> TidalConstituents
   ```

### Optional extensions (lower priority)

| Feature | Description |
|---|---|
| Rayleigh criterion check | Warn when two constituents are closer in frequency than `1 / T_obs`; catches unresolvable pairs before the solve |
| `vectoravg` | Vector-average A and φ across multiple analysis sub-periods (e.g. per year), then return the mean; needed for multi-year data |
| Component splitting | Derive satellite components from a parent constituent after the main solve; matches Python `split_components` |

## Implementation order for tidal prediction

Building bottom-up from the call graph, implement in this order:

1. **Schureman tables** (`schureman.jl`) — load `data_schureman_harmonic.csv` and
   `data_schureman_shallowrelations.csv` from `hatyan_small/data/`. Pure data loading,
   no computation. Copy the CSV files from `hatyan.git/hatyan/data/`. Verify by
   comparing the loaded table against Python output.

2. **Doodson equations** (`doodson.jl`) — implement `robust_timedelta_sec()` and
   `get_doodson_eqvals()` to compute the six astronomical arguments T, S, H, P, N, P1
   from a `DateTime`. Test against Python output for known dates.

3. **Schureman constituent calculations** (`schureman.jl`) — implement
   `get_schureman_freqs()`, `get_schureman_v0()`, `get_schureman_constants()`,
   `get_schureman_u()`, `get_schureman_f()`. Each takes the tables and Doodson values
   as input and can be unit-tested individually against Python for a known date and
   constituent list.

4. **Generic wrappers** (`schureman.jl`) — implement `get_freqv0_generic()` and
   `get_uf_generic()`. Thin dispatch layer; trivial once step 3 works.

5. **Prediction summation** (`prediction.jl`) — implement the cosine summation
   `h(t) = Σ f_i · A_i · cos(ω_i · Δt + v₀_i + u_i − φ_i)`. Validate end-to-end:
   read constituents from `VLISSGN_ana.txt` with `read_donar_constituents`, predict,
   and compare against `VLISSGN_pre.txt`.

## Function signatures

### Settings struct

`HatyanSettings` groups the numerical options that control the harmonic calculation.
It does **not** include the constituent table method (`"schureman"` / `"foreman"`);
that is passed as a separate argument to `analysis` and then stored inside the
resulting `TidalConstituents.source` so `prediction` can recover it automatically.

```julia
struct HatyanSettings
    nodalfactors :: Bool   # apply nodal factor corrections (f and u); default true
    fu_alltimes  :: Bool   # compute f/u at every timestep rather than at period centre; default true
    xfac         :: Bool   # apply x-factor amplitude correction to f; default false
end

# Convenience constructor with defaults
HatyanSettings(;
    nodalfactors = true,
    fu_alltimes  = true,
    xfac         = false,
) -> HatyanSettings
```

---

### `analysis`

Performs a least-squares harmonic analysis on one or more water-level timeseries and
returns the fitted tidal constituent amplitudes and phases.

```julia
analysis(
    ts         :: TimeSeries,                    # input water levels [locations × times], metres
    const_list :: Vector{String},                # constituent names, e.g. ["M2", "S2", "K1"]
    method     :: String = "schureman",          # constituent table: "schureman" or "foreman"
    settings   :: HatyanSettings = HatyanSettings(),
) -> TidalConstituents                           # amplitudes (m) and phases (°) [locations × constituents]
```

**`TidalConstituents.source` is built by appending the analysis step to the input
`TimeSeries.source`**, forming a provenance trail:

```
"VLISSGN" → "VLISSGN | analysis(schureman)"
```

The method name is embedded in the source string, so `prediction` can recover it by
parsing the last `analysis(...)` token — the caller does not need to repeat it.

**Other notes:**
- `const_list` can be built from a predefined set using `constituent_list(name::String)`,
  e.g. `constituent_list("year")` returns the standard 94-component North Sea set.
- If `ts` contains multiple locations, each is analysed independently and the results
  are merged into a multi-location `TidalConstituents`.
- `NaN` values in `ts` are dropped before the least-squares solve for each location.
- Include `"A0"` in `const_list` to fit the mean water level.

**Mapping from Python:**

| Python `analysis()` kwarg | Julia |
|---|---|
| `nodalfactors=True` | `settings.nodalfactors=true` |
| `fu_alltimes=True` | `settings.fu_alltimes=true` |
| `xfac=False` | `settings.xfac=false` |
| `source='schureman'` | `method="schureman"` |

---

### `prediction`

Reconstructs a water-level timeseries from tidal constituents using the harmonic
summation formula `h(t) = Σ f_i · A_i · cos(ω_i · Δt + v₀_i + u_i − φ_i)`.

```julia
prediction(
    tc       :: TidalConstituents,              # amplitudes and phases [locations × constituents]
    times    :: Vector{DateTime},               # output times at which to evaluate h(t)
    settings :: HatyanSettings = HatyanSettings(),
) -> TimeSeries                                 # predicted water levels [locations × times], metres
```

The constituent table method is recovered automatically from `tc.source`
(the `"schureman"` or `"foreman"` suffix written by `analysis`), so it does not need
to be passed again.

**`TimeSeries.source` extends the provenance trail** by appending the prediction step:

```
"VLISSGN | analysis(schureman)" → "VLISSGN | analysis(schureman) | prediction"
```

**Other notes:**
- `times` can also be passed as a `StepRange{DateTime}`, which is collected internally.
- If `tc` contains multiple locations, each is predicted independently and the results
  are merged into a multi-location `TimeSeries`.
- Δt is measured in seconds from `times[1]` (the start of the prediction period).

## Julia design elements
- write to [hatyan_small](hatyan_small) — Julia package with only the core computational routines and minimal dependencies (e.g. DataFrames.jl, CSV.jl)
- src/

## Keep or forget
- Keep:
    - core computational routines for tidal analysis and prediction
    - data structures for time-series and tidal constituents
    - astronomical constants and Schureman tables
    - default constituent set (e.g. M2, S2, N2, K1, O1, etc.) and the ability to specify custom sets
- Forget:
    - file I/O and plotting routines
    - high/low water detection and numbering
    - astronomical event calculations (moon phases, sunrise/sunset, etc.)
    - metadata management and Dutch national data portal helpers
    - CLI entry point and deprecated backward-compatibility wrappers
    - Foreman method (Schureman is the default; add later if needed)
    - quality-check helpers (Nyquist folding, Rayleigh check) — re-implement in Julia as needed

## Key data structures in Hatyan.git

### Time-series

A time-series is a `pandas.DataFrame` with a `pd.DatetimeIndex` as the row index.

**Equidistant (regular) time-series** — used as input to `analysis()` and output of `prediction()`:

| Column | Type | Description |
|---|---|---|
| `values` | `float64` | Water level in metres |

**Extremes time-series** — output of `calc_HWLW()`, used for high/low water analysis:

| Column | Type | Description |
|---|---|---|
| `values` | `float64` | Water level in metres at the extreme |
| `HWLWcode` | `int` | 1 = high water, 2 = low water, 3 = first LW (double HW), 4 = agger, 5 = second LW |
| `HWLWno` | `int` | Sequential tidal wave number (added by `calc_HWLWnumbering()`) |

Both variants carry an `.attrs` dict (pandas DataFrame metadata) with station/unit/vertical-reference fields. The only field used inside the core computational routines is `tzone` (timezone of the index).

**Key invariants:**
- Index is always monotonically increasing (enforced at entry points).
- Values are in **metres** throughout the Python code (conversion from cm happens on file read).
- `NaN` is used for missing values; analysis drops them before the least-squares solve.

---

### Tidal constituents (components)

A component set is a `pandas.DataFrame` with constituent names (e.g. `'A0'`, `'M2'`, `'S2'`) as the row index.

| Column | Type | Description |
|---|---|---|
| `A` | `float64` | Amplitude in metres (`A0` encodes mean water level) |
| `phi_deg` | `float64` | Phase in degrees, range [0, 360) |

The `.attrs` dict carries analysis provenance: `nodalfactors`, `xfac`, `fu_alltimes`, `source`, `tstart`, `tstop`, `tzone`.

**Key conventions:**
- `A0` is the mean water level (DC offset). Its `phi_deg` is always 0; a negative amplitude is used when the mean level is negative.
- Phases are referenced to the start of the analysis period (`dood_date_start`) and include the initial astronomical phase v₀ and nodal correction u.
- The prediction formula is `h(t) = Σ f_i · A_i · cos(ω_i · Δt + v₀_i + u_i − φ_i)` where Δt is seconds from `dood_date_start`.

---

## Core computational routines needed for Julia

The tidal analysis/prediction computation is a least-squares harmonic fit (analysis) and a cosine summation (prediction). The needed routines fall into four layers.

### Layer 1 — Astronomical constants

| Routine | File | Role |
|---|---|---|
| `robust_timedelta_sec()` | `hatyan_core.py` | Seconds since 1900-01-01 — the time reference for all Doodson calculations |
| `get_doodson_eqvals()` | `hatyan_core.py` | Computes T, S, H, P, N, P1 from a date — the six astronomical arguments |

### Layer 2 — Schureman constituent tables

| Routine | File | Role |
|---|---|---|
| `get_schureman_shallowrelations()` | `schureman.py` | Reads `data_schureman_shallowrelations.csv` — shallow-water component arithmetic |
| `get_schureman_table()` | `schureman.py` | Builds the full coefficient table by evaluating shallow-water expressions on the harmonic table |
| `get_schureman_freqs()` | `schureman.py` | Frequencies ω (rad/hr) from Doodson numbers |
| `get_schureman_v0()` | `schureman.py` | Initial astronomical phase v₀ at start of period |
| `get_schureman_constants()` | `schureman.py` | Intermediate astronomical constants (ξ, ν, etc.) from Doodson values |
| `get_schureman_u()` | `schureman.py` | Phase correction u (nodal modulation angle) |
| `get_schureman_f()` | `schureman.py` | Amplitude factor f (nodal modulation factor) |
| `correct_fwith_xfac()` | `schureman.py` | Applies optional x-factor correction to f |

### Layer 3 — Generic wrappers

| Routine | File | Role |
|---|---|---|
| `get_freqv0_generic()` | `hatyan_core.py` | Returns freq and v₀ for a constituent list |
| `get_uf_generic()` | `hatyan_core.py` | Returns u and f; handles the `nodalfactors=False` shortcut (f=1, u=0) |

### Layer 4 — Analysis and prediction

| Routine | File | Role |
|---|---|---|
| `analysis_singleperiod()` | `analysis_prediction.py` | Core least-squares solve: builds design matrix, computes xTx, solves for A and φ |
| `vectoravg()` | `analysis_prediction.py` | Vector-averages A and φ across multiple analysis periods |
| `split_components()` | `analysis_prediction.py` | Derives satellite components from a parent constituent (optional) |
| `prediction_singleperiod()` | `analysis_prediction.py` | Core prediction: `h(t) = Σ f_i · A_i · cos(ω_i·t + v₀_i + u_i − φ_i)` |

### Call graph

```
analysis() / prediction()
 ├─ get_freqv0_generic()
 │   ├─ get_schureman_freqs()  ──┐
 │   └─ get_schureman_v0()    ──┤─── get_doodson_eqvals()
 ├─ get_uf_generic()              │       └─ robust_timedelta_sec()
 │   ├─ get_schureman_f()   ──┐  │
 │   └─ get_schureman_u()   ──┤──┘
 │       └─ get_schureman_constants()
 └─ [solve xTx / summation]
     └─ get_schureman_table()        (loaded once, cached)
         └─ get_schureman_shallowrelations()  (loaded once, cached)
```

### Data files

The two CSV files in `hatyan.git/hatyan/data/` are the only external data dependencies and must be bundled with the Julia package:

- `data_schureman_harmonic.csv` — Doodson and nodal factor coefficients for harmonic constituents
- `data_schureman_shallowrelations.csv` — arithmetic relations defining shallow-water constituents

### Not needed

- All file I/O (`read_dia`, `write_dia`, `read_noos`, `write_netcdf`, etc.)
- Plotting (`plot_timeseries`, `plot_components`, etc.)
- High/low water detection (`calc_HWLW*`, `calc_HWLWnumbering`)
- Astronomical events (`astrog.py` — moon phases, sunrise/set, perigee/apogee)
- Metadata management (`metadata.py`)
- Dutch national data portal helpers (`ddlpy_helpers.py`)
- CLI entry point (`cli.py`)
- Deprecated backward-compatibility wrappers (`deprecated.py`)
- Foreman method (`foreman.py`) — Schureman is the default; add later if needed
- Quality-check helpers (`nyquist_folding`, `check_rayleigh`) — re-implement in Julia as needed