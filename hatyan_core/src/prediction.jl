# prediction.jl — Tidal prediction (harmonic cosine summation)
#
# Implements h(t) = Σ fᵢ · Aᵢ · cos(ωᵢ · Δt + v₀ᵢ + uᵢ − φᵢ)
# Ported from prediction_singleperiod() in analysis_prediction.py (Python hatyan).

"""
    prediction(tc, times, settings=HatyanSettings()) -> TimeSeries

Reconstruct water levels from tidal constituents using the harmonic summation
formula `h(t) = Σ fᵢ · Aᵢ · cos(ωᵢ · Δt + v₀ᵢ + uᵢ − φᵢ)`.

# Arguments
- `tc`: `TidalConstituents` — amplitudes (m) and phases (°) for one or more locations
- `times`: output times; accepts `Vector{DateTime}` or any `AbstractRange{DateTime}`
- `settings`: `HatyanSettings` controlling nodal factors etc. (defaults apply)

# Notes
- Δt is measured in **seconds** from `times[1]` (start of the prediction period).
- The constituent table method is read from `tc.source` by parsing the last
  `analysis(...)` token (e.g. `"VLISSGN | analysis(schureman)"`). Falls back to
  `"schureman"` when no such token is found.
- When `settings.fu_alltimes = true`, nodal factors f and u are re-evaluated at
  every timestep; otherwise they are evaluated once at the mid-point of `times`.
- Multiple locations in `tc` are predicted independently and merged into the
  returned `TimeSeries`.
- The returned `TimeSeries.source` extends the provenance trail:
  `"VLISSGN | analysis(schureman)" → "VLISSGN | analysis(schureman) | prediction"`

# Returns
A `TimeSeries` (Float32 values, metres) with the same locations and metadata as `tc`.
"""
function prediction(
    tc       :: TidalConstituents,
    times    :: AbstractVector{DateTime},
    settings :: HatyanSettings = HatyanSettings(),
)
    # Collect StepRange / AbstractRange into a plain Vector
    times_vec = collect(times)

    method = _extract_method_from_source(tc.source)

    n_times = length(times_vec)
    n_locs  = length(tc.names)

    # Doodson reference dates
    mid_idx        = (n_times + 1) ÷ 2
    dood_date_mid   = times_vec[mid_idx:mid_idx]    # single-element Vector
    dood_date_start = times_vec[1:1]

    # Δt in seconds from the first timestep
    ref  = times_vec[1]
    t_s  = [Dates.value(t - ref) * 1e-3 for t in times_vec]   # n_times, seconds

    # Frequencies and initial phase (shared across all locations)
    const_list = tc.constituent_names
    freq, v0 = get_freqv0_generic(const_list, dood_date_mid, dood_date_start, method)

    # Angular frequency in rad/s
    omega_rads = freq .* (2π / 3600.0)   # cycles/hr × 2π / 3600 s/hr = rad/s

    # Nodal factors (same across all locations)
    dood_date_fu = settings.fu_alltimes ? times_vec : dood_date_mid
    u, f = get_uf_generic(const_list, dood_date_fu, settings, method)

    # Pre-compute omeg_t [n_const × n_times]: ωᵢ · Δtₜ
    omeg_t = omega_rads .* t_s'   # (n_const,) .* (1 × n_times) → n_const × n_times

    # Accumulate predictions for each location
    out_values = Matrix{Float32}(undef, n_locs, n_times)

    for loc in 1:n_locs
        A_col   = reshape(Float64.(tc.amplitudes[loc, :]), :, 1)   # n_const × 1 (m)
        phi_col = reshape(deg2rad.(Float64.(tc.phases[loc, :])), :, 1)  # n_const × 1 (rad)

        # v_u_phi[i, t] = v₀ᵢ + uᵢ(t) − φᵢ
        # v0: n_const × 1, u: n_const × n_dates_fu, phi_col: n_const × 1
        # Broadcasting: v0 .+ u broadcasts u to n_const × n_times_fu
        v_u_phi = v0 .+ u .- phi_col   # n_const × n_times_fu (or n_const × 1)

        # f_A[i, t] = fᵢ(t) · Aᵢ
        f_A = f .* A_col               # n_const × n_times_fu (or n_const × 1)

        # h[t] = Σᵢ f_A[i,t] · cos(omeg_t[i,t] + v_u_phi[i,t])
        h = vec(sum(f_A .* cos.(omeg_t .+ v_u_phi), dims=1))   # n_times

        out_values[loc, :] = Float32.(h)
    end

    new_source = tc.source * " | prediction"
    quantity   = isempty(tc.quantity) ? "water level" : tc.quantity

    return TimeSeries(
        out_values,
        times_vec,
        tc.names,
        tc.longitudes,
        tc.latitudes,
        quantity,
        new_source,
    )
end


# ── Internal helper ───────────────────────────────────────────────────────────

"""
    _extract_method_from_source(source) -> String

Parse the constituent table method from a provenance source string.
Looks for the last `analysis(...)` token; defaults to `"schureman"` if absent.
"""
function _extract_method_from_source(source::String)
    matches = collect(eachmatch(r"analysis\((\w+)\)", source))
    isempty(matches) && return "schureman"
    return String(matches[end].captures[1])
end
