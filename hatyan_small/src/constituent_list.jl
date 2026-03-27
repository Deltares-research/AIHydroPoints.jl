# constituent_list.jl — Predefined tidal constituent sets
#
# Ported from get_const_list_hatyan() in hatyan_core.py (Python hatyan / Deltares).
# Lists originate from tidegui initializetide.m (Koos Doekes).
#
# Each preset is sized to match the minimum time series length needed to resolve all
# its constituents via the Rayleigh criterion (1/T_obs frequency separation):
#   "year"       → ~1 year   of data  (94 constituents + A0)
#   "halfyear"   → ~6 months of data  (88 constituents + A0)
#   "month"      → ~1 month  of data  (21 constituents + A0)
#   "month_deepwater" → ~1 month, deep-water variant  (21 + A0)
#   "springneap" → ~15 days  of data  (14 constituents + A0)
#   "day"        → ~1 day    of data  (10 constituents + A0)
#   "tidalcycle" → ~12.4 hrs of data  ( 6 constituents + A0)
#   "all"        → all constituents available in the Schureman table

const _CONSTITUENT_LISTS = Dict{String, Vector{String}}(

    # ── Standard North Sea set (recommended for ~1-year records) ──────────────
    # 94 components + A0; shallow-water components: 2MS6, 2SM2, 3MS4, 3MS8,
    # 4MS10, M4, M6, M8, MS4.
    "year" => [
        "A0","SA","SM",
        "Q1","O1","M1C","P1","S1","K1",
        "3MKS2","3MS2","OQ2","MNS2","2ML2S2","NLK2","MU2","N2","NU2",
        "MSK2","MPS2","M2","MSP2","MKS2","LABDA2","2MN2","T2","S2","K2",
        "MSN2","2SM2","SKM2",
        "NO3","2MK3","2MP3","SO3","MK3","SK3",
        "4MS4","2MNS4","3MS4","MN4","2MLS4","2MSK4","M4","3MN4","MS4",
        "MK4","2MSN4","S4",
        "MNO5","3MK5","2MP5","3MO5","MSK5","3KM5",
        "3MNS6","2NM6","4MS6","2MN6","2MNU6","3MSK6","M6","MSN6","MKNU6",
        "2MS6","2MK6","3MSN6","2SM6","MSK6",
        "2MNO7","M7","2MSO7",
        "2(MN)8","3MN8","M8","2MSN8","2MNK8","3MS8","3MK8","2(MS)8","2MSK8",
        "3MNK9","4MK9","3MSK9",
        "4MN10","M10","3MSN10","4MS10","2(MS)N10","3M2S10",
        "4MSK11",
        "M12","4MSN12","5MS12","4M2S12",
    ],

    # ── ~6-month set (year minus S1, MSK2, MPS2, MSP2, MKS2, T2) ─────────────
    # 88 components + A0.
    "halfyear" => [
        "A0","SA","SM",
        "Q1","O1","M1C","P1","K1",
        "3MKS2","3MS2","OQ2","MNS2","2ML2S2","NLK2","MU2","N2","NU2",
        "M2","LABDA2","2MN2","S2","K2","MSN2","2SM2","SKM2",
        "NO3","2MK3","2MP3","SO3","MK3","SK3",
        "4MS4","2MNS4","3MS4","MN4","2MLS4","2MSK4","M4","3MN4","MS4",
        "MK4","2MSN4","S4",
        "MNO5","3MK5","2MP5","3MO5","MSK5","3KM5",
        "3MNS6","2NM6","4MS6","2MN6","2MNU6","3MSK6","M6","MSN6","MKNU6",
        "2MS6","2MK6","3MSN6","2SM6","MSK6",
        "2MNO7","M7","2MSO7",
        "2(MN)8","3MN8","M8","2MSN8","2MNK8","3MS8","3MK8","2(MS)8","2MSK8",
        "3MNK9","4MK9","3MSK9",
        "4MN10","M10","3MSN10","4MS10","2(MS)N10","3M2S10",
        "4MSK11",
        "M12","4MSN12","5MS12","4M2S12",
    ],

    # ── ~1-month set ──────────────────────────────────────────────────────────
    # 21 components + A0.
    "month" => [
        "A0","Q1","O1","K1",
        "3MS2","MNS2","MU2","N2","M2","2MN2","S2","2SM2",
        "3MS4","MN4","M4","MS4",
        "2MN6","M6","2MS6",
        "M8","3MS8",
        "4MS10",
    ],

    # ── ~1-month deep-water variant (L2 instead of 2MN2) ─────────────────────
    "month_deepwater" => [
        "A0","Q1","O1","K1",
        "3MS2","MNS2","MU2","N2","M2","L2","S2","2SM2",
        "3MS4","MN4","M4","MS4",
        "2MN6","M6","2MS6",
        "M8","3MS8",
        "4MS10",
    ],

    # ── ~15-day (spring–neap) set ─────────────────────────────────────────────
    # 14 components + A0. Note: N2 and MN4 cannot be resolved at this length.
    "springneap" => [
        "A0","O1","K1",
        "MU2","M2","S2","2SM2",
        "3MS4","M4","MS4",
        "M6","2MS6",
        "M8","3MS8",
        "4MS10",
    ],

    # ── ~1-day set ────────────────────────────────────────────────────────────
    # 10 components + A0 (M-family overtides).
    "day" => [
        "A0","M1","M2","M3","M4","M5","M6","M7","M8","M10","M12",
    ],

    # ── ~1-tidal-cycle set (~12h 25min) ──────────────────────────────────────
    # 6 components + A0.
    "tidalcycle" => [
        "A0","M2","M4","M6","M8","M10","M12",
    ],
)

"""
    constituent_list(name::String) -> Vector{String}

Return a predefined list of tidal constituent names for harmonic analysis.

# Available presets

| Name             | Length needed | N constituents (incl. A0) |
|------------------|---------------|---------------------------|
| `"year"`         | ~1 year       | 95                        |
| `"halfyear"`     | ~6 months     | 89                        |
| `"month"`        | ~1 month      | 22                        |
| `"month_deepwater"` | ~1 month   | 22 (L2 instead of 2MN2)   |
| `"springneap"`   | ~15 days      | 15                        |
| `"day"`          | ~1 day        | 11                        |
| `"tidalcycle"`   | ~12.4 hrs     | 7                         |
| `"all"`          | —             | all Schureman constituents |

# Example

```julia
const_list = constituent_list("year")   # 95-element Vector{String}
tc = analysis(ts, const_list)
```
"""
function constituent_list(name::String)
    if name == "all"
        return get_schureman_table().names
    end
    if !haskey(_CONSTITUENT_LISTS, name)
        valid = join(sort(collect(keys(_CONSTITUENT_LISTS))), ", ")
        error("Unknown constituent list \"$name\". Valid options: $valid, \"all\"")
    end
    return copy(_CONSTITUENT_LISTS[name])
end
