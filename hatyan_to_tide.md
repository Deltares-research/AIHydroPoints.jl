
# Convert the Hatyan core computational routines from python to julia

The Hatyan core computational routines are currently implemented in python. We want to convert them to julia for better performance and ease of use from julia scripts. There are many extra options in the hatyan package that do not need in the julia version.

## Steps
1. create a summary of the python code [hatyan_summary.md](hatyan_summary.md)
2. select the core computational routines that are needed for the julia version
3. identify the key data structures used in the core routines (see below)
4. implement reading and writing routines, so we can test the julia code against the python code on the same data


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