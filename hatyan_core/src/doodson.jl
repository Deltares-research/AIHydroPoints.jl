# doodson.jl — Doodson astronomical arguments
#
# Implements the six Doodson astronomical arguments T, S, H, P, N, P1 used in
# tidal analysis. Ported from hatyan_core.py (Python hatyan package).
#
# Reference: Schureman (1941), "Manual of Harmonic Analysis and Prediction of Tides"

# Row indices in the 6-row Doodson matrix returned by get_doodson_eqvals
const DOOD_T  = 1   # T  – solar hour angle (corrected for epoch)
const DOOD_S  = 2   # S  – mean longitude of moon
const DOOD_H  = 3   # H  – mean longitude of sun
const DOOD_P  = 4   # P  – longitude of lunar perigee
const DOOD_N  = 5   # N  – longitude of lunar node
const DOOD_P1 = 6   # P1 – longitude of solar perigee


"""
    robust_timedelta_sec(dates) -> Vector{Float64}

Return seconds elapsed since 1900-01-01 00:00:00 for each `DateTime` in `dates`.
This is the universal time reference used in all Doodson astronomical calculations.
"""
function robust_timedelta_sec(dates::AbstractVector{DateTime})
    ref = DateTime(1900, 1, 1, 0, 0, 0)
    return [Dates.value(d - ref) * 1e-3 for d in dates]   # ms → s
end


"""
    get_doodson_eqvals(dates; mode=:position) -> Matrix{Float64}

Compute the six Doodson astronomical arguments for each date.

Returns a **6 × n** matrix where rows are [T, S, H, P, N, P1] (use the `DOOD_*`
constants to index them) and columns correspond to `dates`.

# Arguments
- `dates`: vector of `DateTime` values
- `mode`:
  - `:position` (default) — values in radians (positions / phases)
  - `:freq` — angular speeds in rad/hr; N is not used and is set to `NaN`

# Notes
The formula follows the hatyan convention: T = deg2rad(180 + hour*15 + min*15/60).
Doodson coefficients from Schureman (1941), p. 162.
"""
function get_doodson_eqvals(dates::AbstractVector{DateTime}; mode::Symbol=:position)
    DNUJE = 24.0 * 36525.0            # hours per Julian century
    t_sec = robust_timedelta_sec(dates)
    Tj    = @. (t_sec / 3600.0 + 12.0) / DNUJE

    if mode === :freq
        T  = fill(deg2rad(15.0), length(Tj))
        S  = @. (8399.7092745 + 0.0000346 * Tj * 2) / DNUJE
        H  = @. ( 628.3319500 + 0.0000052 * Tj * 2) / DNUJE
        P  = @. (  71.0180412 - 0.0001801 * Tj * 2) / DNUJE
        N  = fill(NaN, length(Tj))
        P1 = @. (   0.0300053 + 0.0000079 * Tj * 2) / DNUJE
    else
        h  = Float64.(Dates.hour.(dates))
        m  = Float64.(Dates.minute.(dates))
        T  = @. deg2rad(180.0 + h * 15.0 + m * 15.0 / 60.0)
        S  = @. 4.7200089 + 8399.7092745 * Tj + 0.0000346 * Tj ^ 2
        H  = @. 4.8816280 +  628.3319500 * Tj + 0.0000052 * Tj ^ 2
        P  = @. 5.8351526 +   71.0180412 * Tj - 0.0001801 * Tj ^ 2
        N  = @. 4.5236016 -   33.7571463 * Tj + 0.0000363 * Tj ^ 2
        P1 = @. 4.9082295 +    0.0300053 * Tj + 0.0000079 * Tj ^ 2
    end

    return vcat(T', S', H', P', N', P1')   # 6 × n matrix
end
