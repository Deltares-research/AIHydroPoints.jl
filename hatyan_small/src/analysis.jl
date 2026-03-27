# analysis.jl — Tidal harmonic analysis (least-squares fit)
#
# Implements the inverse of prediction:
#   given observed h(t) and a constituent list, solve for amplitudes A and phases φ.
#
# Design matrix (m × 2N):
#   xmat[:, 1:N]    = f_i(t) .* cos(ω_i·Δt + v₀_i + u_i(t))   [cosine columns]
#   xmat[:, N+1:2N] = f_i(t) .* sin(ω_i·Δt + v₀_i + u_i(t))   [sine columns]
#
# Solve the normal equations  (xᵀx) β = xᵀy,  then
#   A_i   = √(β_a² + β_b²)
#   φ_i   = mod(atan(β_b, β_a), 2π)   in degrees
#
# Ported from analysis_singleperiod() in analysis_prediction.py (Python hatyan).

using LinearAlgebra


# ── Public API ────────────────────────────────────────────────────────────────

"""
    analysis(ts, const_list, method=\"schureman\", settings=HatyanSettings();
             max_matrix_condition=20.0) -> TidalConstituents

Perform a least-squares harmonic analysis on a water-level `TimeSeries` and
return the fitted tidal constituent amplitudes (m) and phases (°).

# Arguments
- `ts`: `TimeSeries` — observed water levels `[locations × times]`
- `const_list`: constituent names to fit, e.g. `constituent_list("year")`
- `method`: constituent table (`"schureman"` only for now)
- `settings`: `HatyanSettings` — same nodal-factor options as `prediction`
- `max_matrix_condition`: abort if `cond(xᵀx)` exceeds this (default 20)

# Notes
- `NaN` values in `ts` are dropped before the solve for each location.
- Include `"A0"` in `const_list` to fit the mean water level.
- Δt is measured in **seconds** from `times[1]`.
- Provenance: `"VLISSGN" → "VLISSGN | analysis(schureman)"`

# Returns
A `TidalConstituents` (Float32) with the same station metadata as `ts`.
"""
function analysis(
    ts                   :: TimeSeries,
    const_list           :: Vector{String},
    method               :: String        = "schureman",
    settings             :: HatyanSettings = HatyanSettings();
    max_matrix_condition :: Float64       = 20.0,
)
    n_locs  = length(get_names(ts))
    n_const = length(const_list)
    times   = get_times(ts)

    amp_out = Matrix{Float32}(undef, n_locs, n_const)
    phi_out = Matrix{Float32}(undef, n_locs, n_const)

    for loc in 1:n_locs
        values = Float64.(get_values(ts)[loc, :])
        A_i, phi_i = _analysis_singleperiod(
            times, values, const_list, method, settings, max_matrix_condition,
        )
        amp_out[loc, :] = Float32.(A_i)
        phi_out[loc, :] = Float32.(phi_i)
    end

    new_source = get_source(ts) * " | analysis($method)"
    quantity   = isempty(get_quantity(ts)) ? "water level" : get_quantity(ts)

    return TidalConstituents(
        amp_out,
        phi_out,
        const_list,
        get_names(ts),
        get_longitudes(ts),
        get_latitudes(ts),
        quantity,
        new_source,
    )
end


# ── Internal: single-location solve ──────────────────────────────────────────

"""
    _analysis_singleperiod(times, values, const_list, method, settings,
                           max_condition) -> (A, phi_deg)

Least-squares harmonic fit for one location.

- `times`  : full time vector (NaN positions included, used for mid/start dates)
- `values` : observed water levels (NaN entries are dropped before the solve)

Returns `(A_i, phi_deg_i)` — Float64 vectors of length N.
"""
function _analysis_singleperiod(
    times         :: Vector{DateTime},
    values        :: Vector{Float64},
    const_list    :: Vector{String},
    method        :: String,
    settings      :: HatyanSettings,
    max_condition :: Float64,
)
    n_all = length(times)
    N     = length(const_list)

    # ── Drop NaN ─────────────────────────────────────────────────────────────
    valid = .!isnan.(values)
    sum(valid) >= 2 || error(
        "fewer than 2 valid (non-NaN) observations after dropping NaNs; " *
        "analysis not possible"
    )
    times_nonan  = times[valid]
    values_nonan = values[valid]
    m = length(times_nonan)

    # ── Reference dates ───────────────────────────────────────────────────────
    # mid  : middle of the *full* time vector (matches Python ts_pd.index[n//2])
    # start: first timestep (determines v₀ and Δt origin)
    mid_idx        = n_all ÷ 2 + 1          # Python: n_all // 2  (0-indexed) → same element
    dood_date_mid   = times[mid_idx:mid_idx]
    dood_date_start = times[1:1]

    # ── Δt in seconds from times[1], for non-NaN observations ─────────────────
    ref = times[1]
    t_s = [Dates.value(t - ref) * 1e-3 for t in times_nonan]   # (m,)

    # ── Frequencies and initial phase ────────────────────────────────────────
    freq, v0 = get_freqv0_generic(const_list, dood_date_mid, dood_date_start, method)
    # freq : (N,)    cycles/hr
    # v0   : (N, 1)  radians

    omega_rads = freq .* (2π / 3600.0)   # (N,) rad/s

    # ── Nodal factors ─────────────────────────────────────────────────────────
    # fu_alltimes=true  → evaluate at every non-NaN observation
    # fu_alltimes=false → evaluate once at the mid-point
    dood_date_fu = settings.fu_alltimes ? times_nonan : dood_date_mid
    u, f = get_uf_generic(const_list, dood_date_fu, settings, method)
    # u, f : (N, m) or (N, 1)

    # ── Design matrix  xmat : (m × 2N) ───────────────────────────────────────
    # omega_t[i, t] = ω_i · Δtₜ   →  (N, m)
    omega_t = omega_rads .* t_s'

    # v_u[i, t] = v₀_i + u_i(t)   →  (N, m) via broadcast
    v_u = v0 .+ u

    # cos/sin columns, each (N, m), transposed to (m, N)
    cos_terms = (f .* cos.(omega_t .+ v_u))'   # (m, N)
    sin_terms = (f .* sin.(omega_t .+ v_u))'   # (m, N)

    xmat = hcat(cos_terms, sin_terms)           # (m, 2N)

    # ── Normal equations ──────────────────────────────────────────────────────
    xTx = xmat' * xmat   # (2N, 2N)

    # A0 correction: A0 has freq=0, v₀=0, u=0, f=1, so its sine column is all
    # zeros → xTx diagonal entry = 0 → singular. Set it to m (number of
    # observations) to regularise, matching Python's xTxmat[N, N] = m fix.
    if "A0" in const_list
        i_A0 = findfirst(==("A0"), const_list)
        xTx[N + i_A0, N + i_A0] = Float64(m)
    end

    # ── Condition check ───────────────────────────────────────────────────────
    cond_val = cond(xTx)
    cond_val <= max_condition || error(
        "xTx matrix condition ($(@sprintf("%.2f", cond_val))) exceeds maximum " *
        "($max_condition). Try a shorter constituent list or a longer time series."
    )

    xTy  = xmat' * values_nonan   # (2N,)
    beta = xTx \ xTy              # (2N,)  least-squares solution

    # ── Recover amplitude and phase ───────────────────────────────────────────
    # beta[1:N]    = cosine coefficients  (a_i)
    # beta[N+1:2N] = sine   coefficients  (b_i)
    a = beta[1:N]
    b = beta[N+1:end]

    A_i     = sqrt.(a .^ 2 .+ b .^ 2)
    phi_deg = rad2deg.(mod.(atan.(b, a), 2π))   # wrapped to [0°, 360°)

    # A0 phase correction: a phase of 180° means the amplitude should be
    # negative (encodes a negative mean water level).
    if "A0" in const_list
        i_A0 = findfirst(==("A0"), const_list)
        if phi_deg[i_A0] ≈ 180.0
            A_i[i_A0]     = -A_i[i_A0]
            phi_deg[i_A0] = 0.0
        end
    end

    return A_i, phi_deg
end
