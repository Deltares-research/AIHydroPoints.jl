# foreman.jl — Foreman constituent tables and nodal factor calculations
#
# Implements loading of the Foreman (2004) harmonic/shallow-water tables and the
# constituent-specific calculations for frequency, v₀, u, and f.
# Ported from foreman.py (Python hatyan package).
#
# Reference: Foreman (2004), "Manual for Tidal Heights Analysis and Prediction",
#            Institute of Ocean Sciences, Patricia Bay, Sidney B.C., Canada

# ── Data structures ───────────────────────────────────────────────────────────

"""
    ForemanHarmonic

Solar-convention Doodson numbers and EDN phase for one harmonic constituent.
Fields T, S, H, P, N, P1 are integers; EDN is in fraction-of-cycle (×2π = rad).
"""
struct ForemanHarmonic
    T   :: Int
    S   :: Int
    H   :: Int
    P   :: Int
    N   :: Int
    P1  :: Int
    EDN :: Float64
end

"""
    ForemanSatellite

One satellite correction row for the nodal factor calculation.
dP, dN, dP1 are integer changes; dEDN is in fraction-of-cycle; factor is amplitude ratio.
"""
struct ForemanSatellite
    const_name :: String
    dP         :: Int
    dN         :: Int
    dP1        :: Int
    dEDN       :: Float64
    factor     :: Float64
end

"""
    ForemanTable

Complete Foreman tidal constituent table (harmonic + shallow-water).
"""
struct ForemanTable
    harmonic_names :: Vector{String}
    harmonic       :: Dict{String, ForemanHarmonic}   # name → doodson+edn
    satellites     :: Vector{ForemanSatellite}          # all satellite rows
    sat_index      :: Dict{String, Vector{Int}}         # name → indices in satellites
    shallow        :: Vector{Tuple{String, Vector{Tuple{Float64, String}}}}  # ordered
    shallow_set    :: Set{String}
end

# Module-level lazy cache (keyed by latitude so different latitudes are supported)
const _FOREMAN_TABLE_CACHE = Dict{Float64, ForemanTable}()

"""
    get_foreman_table(lat_deg=51.45) -> ForemanTable

Return the Foreman constituent table for the given latitude (degrees North).
Built on first call per latitude and cached for subsequent calls.
"""
function get_foreman_table(lat_deg::Float64 = 51.45)
    if !haskey(_FOREMAN_TABLE_CACHE, lat_deg)
        _FOREMAN_TABLE_CACHE[lat_deg] = _build_foreman_table(lat_deg)
    end
    return _FOREMAN_TABLE_CACHE[lat_deg]
end


# ── File parsing ──────────────────────────────────────────────────────────────

function _parse_foreman_harmonic_file(filepath::String, lat_deg::Float64)
    lat_rad = deg2rad(lat_deg)
    R1 = 0.36309 * (1.0 - 5.0 * sin(lat_rad)^2) / sin(lat_rad)
    R2 = 2.59808 * sin(lat_rad)

    # Accumulate: first occurrence = main line, later = satellite lines
    seen          = Dict{String, Bool}()
    harmonic_names = String[]
    harmonic       = Dict{String, ForemanHarmonic}()
    # raw satellite rows before grouping: (name, dP, dN, dP1, dEDN, factor)
    raw_sats = ForemanSatellite[]

    for raw_line in readlines(filepath)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue

        # Strip inline comment
        cidx = findfirst('#', line)
        if !isnothing(cidx)
            line = strip(line[1:cidx-1])
        end
        isempty(line) && continue

        parts = split(line)
        name  = String(parts[1])
        vals  = parts[2:end]

        if !get(seen, name, false)
            # ── Main constituent line: T S H P N P1 EDN nsats ──────────────
            length(vals) >= 8 || continue
            T_lun = parse(Int,     vals[1])
            S_lun = parse(Int,     vals[2])
            H_lun = parse(Int,     vals[3])
            P     = parse(Int,     vals[4])
            N     = parse(Int,     vals[5])
            P1    = parse(Int,     vals[6])
            EDN   = parse(Float64, vals[7])
            # Convert lunar → solar convention: S -= T, H += T
            T_sol = T_lun
            S_sol = S_lun - T_lun
            H_sol = H_lun + T_lun
            push!(harmonic_names, name)
            harmonic[name] = ForemanHarmonic(T_sol, S_sol, H_sol, P, N, P1, EDN)
            seen[name] = true
        else
            # ── Satellite line: up to 3 × (dP dN dP1 dEDN factor) ──────────
            # Groups of 5 tokens; trailing NaN slots are simply absent
            n = length(vals)
            istart = 1
            while istart + 3 <= n
                dP_s   = parse(Int, vals[istart])
                dN_s   = parse(Int, vals[istart+1])
                dP1_s  = parse(Int, vals[istart+2])
                dEDN_s = parse(Float64, vals[istart+3])
                istart += 4

                istart > n && break
                fac_str = String(vals[istart])
                istart += 1

                # Apply R1/R2 latitude correction
                fac_val = if endswith(fac_str, "R1")
                    parse(Float64, fac_str[1:end-2]) * R1
                elseif endswith(fac_str, "R2")
                    parse(Float64, fac_str[1:end-2]) * R2
                else
                    parse(Float64, fac_str)
                end

                push!(raw_sats, ForemanSatellite(name, dP_s, dN_s, dP1_s, dEDN_s, fac_val))
            end
        end
    end

    # Build satellite index: name → Vector{Int} of positions in raw_sats
    sat_index = Dict{String, Vector{Int}}()
    for (i, sat) in enumerate(raw_sats)
        push!(get!(Vector{Int}, sat_index, sat.const_name), i)
    end

    return harmonic_names, harmonic, raw_sats, sat_index
end


function _parse_foreman_shallow_file(filepath::String, harmonic_names::Vector{String})
    harmonic_set = Set(harmonic_names)
    shallow = Tuple{String, Vector{Tuple{Float64, String}}}[]

    for raw_line in readlines(filepath)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue

        cidx = findfirst('#', line)
        if !isnothing(cidx)
            line = strip(line[1:cidx-1])
        end
        isempty(line) && continue

        parts = split(line)
        length(parts) < 4 && continue
        sname  = String(parts[1])
        ndeps  = parse(Int, parts[2])

        deps = Tuple{Float64, String}[]
        for k in 1:ndeps
            fac  = parse(Float64, parts[2 + 2*k - 1])
            dep  = String(parts[2 + 2*k])
            dep ∈ harmonic_set || error("Foreman shallow '$sname': dependency '$dep' not in harmonic table")
            push!(deps, (fac, dep))
        end
        push!(shallow, (sname, deps))
    end

    return shallow
end


function _build_foreman_table(lat_deg::Float64)
    data_dir = joinpath(@__DIR__, "..", "data")
    harm_file    = joinpath(data_dir, "data_foreman_harmonic.txt")
    shallow_file = joinpath(data_dir, "data_foreman_shallowrelations.txt")

    harmonic_names, harmonic, raw_sats, sat_index =
        _parse_foreman_harmonic_file(harm_file, lat_deg)
    shallow = _parse_foreman_shallow_file(shallow_file, harmonic_names)
    shallow_set = Set(s[1] for s in shallow)

    return ForemanTable(harmonic_names, harmonic, raw_sats, sat_index, shallow, shallow_set)
end


# ── Helper ────────────────────────────────────────────────────────────────────

function _foreman_check_const(name::String, table::ForemanTable)
    haskey(table.harmonic, name) || name ∈ table.shallow_set ||
        error("Constituent '$name' not in Foreman table (harmonic or shallow)")
end


# ── Frequency and v₀ ─────────────────────────────────────────────────────────

"""
    get_foreman_freqs(const_list, dates, lat_deg=51.45) -> Vector{Float64}

Angular frequencies in cycles/hr for each constituent in `const_list`.
Evaluated at `dates[1]` (typically the mid-period date).
"""
function get_foreman_freqs(
    const_list :: Vector{String},
    dates      :: AbstractVector{DateTime},
    lat_deg    :: Float64 = 51.45,
)
    table = get_foreman_table(lat_deg)
    for n in const_list; _foreman_check_const(n, table); end

    # Doodson angular speeds (mode=:freq): 6-row, ignores N
    dood_freq = get_doodson_eqvals(dates; mode=:freq)   # 6 × n_dates
    # Use only T,S,H,P,P1 (rows 1,2,3,4,6) at first date
    freq_vars = dood_freq[[DOOD_T, DOOD_S, DOOD_H, DOOD_P, DOOD_P1], 1]  # 5-vector

    # Precompute frequencies for all harmonic constituents
    harm_freq = Dict{String, Float64}()
    for (name, hm) in table.harmonic
        coefs = Float64[hm.T, hm.S, hm.H, hm.P, hm.P1]
        harm_freq[name] = dot(coefs, freq_vars) / (2π)   # rad/hr → cycles/hr
    end

    freqs = Vector{Float64}(undef, length(const_list))
    for (i, name) in enumerate(const_list)
        if haskey(table.harmonic, name)
            freqs[i] = harm_freq[name]
        else
            # Shallow: linear combination of harmonic frequencies
            idx = findfirst(s -> s[1] == name, table.shallow)
            _, deps = table.shallow[idx]
            freqs[i] = sum(fac * harm_freq[dep] for (fac, dep) in deps)
        end
    end
    return freqs
end


"""
    get_foreman_v0(const_list, dates, lat_deg=51.45) -> Matrix{Float64}

Initial astronomical phase v₀ (rad) for each constituent at each date.
Returns an n_const × n_dates matrix.
"""
function get_foreman_v0(
    const_list :: Vector{String},
    dates      :: AbstractVector{DateTime},
    lat_deg    :: Float64 = 51.45,
)
    table = get_foreman_table(lat_deg)
    for n in const_list; _foreman_check_const(n, table); end

    dood = get_doodson_eqvals(dates)   # 6 × n_dates (position mode)
    T_v  = dood[DOOD_T,  :]
    S_v  = dood[DOOD_S,  :]
    H_v  = dood[DOOD_H,  :]
    P_v  = dood[DOOD_P,  :]
    N_v  = dood[DOOD_N,  :]
    P1_v = dood[DOOD_P1, :]

    n_dates = length(dates)

    # Precompute v0 for all harmonic constituents (n_dates-vector per constituent)
    harm_v0 = Dict{String, Vector{Float64}}()
    for (name, hm) in table.harmonic
        v = @. hm.T * T_v + hm.S * S_v + hm.H * H_v +
               hm.P * P_v + hm.N * N_v + hm.P1 * P1_v +
               hm.EDN * 2π
        harm_v0[name] = v
    end

    V0 = Matrix{Float64}(undef, length(const_list), n_dates)
    for (i, name) in enumerate(const_list)
        if haskey(table.harmonic, name)
            V0[i, :] = harm_v0[name]
        else
            idx = findfirst(s -> s[1] == name, table.shallow)
            _, deps = table.shallow[idx]
            row = zeros(Float64, n_dates)
            for (fac, dep) in deps
                row .+= fac .* harm_v0[dep]
            end
            V0[i, :] = row
        end
    end
    return V0
end


# ── Nodal factors ─────────────────────────────────────────────────────────────

"""
    get_foreman_uf(const_list, dates, lat_deg=51.45) -> (u, f)

Nodal phase correction u (rad) and amplitude factor f (dimensionless) for each
constituent at each date.  Returns two n_const × n_dates matrices.

Shallow-water constituents are derived from their harmonic dependencies:
  f_shallow = ∏ f_dep^|coeff|,   u_shallow = Σ coeff * u_dep
"""
function get_foreman_uf(
    const_list :: Vector{String},
    dates      :: AbstractVector{DateTime},
    lat_deg    :: Float64 = 51.45,
)
    table    = get_foreman_table(lat_deg)
    for n in const_list; _foreman_check_const(n, table); end
    n_dates  = length(dates)

    dood = get_doodson_eqvals(dates)   # 6 × n_dates
    P_v  = dood[DOOD_P,  :]
    N_v  = dood[DOOD_N,  :]
    P1_v = dood[DOOD_P1, :]

    # Determine which harmonic constituents we need (own + shallow dependencies)
    needed_harmonics = Set{String}(n for n in const_list if haskey(table.harmonic, n))
    for name in const_list
        if name ∈ table.shallow_set
            idx = findfirst(s -> s[1] == name, table.shallow)
            _, deps = table.shallow[idx]
            for (_, dep) in deps
                push!(needed_harmonics, dep)
            end
        end
    end

    # Compute f and u for all needed harmonic constituents
    harm_f = Dict{String, Vector{Float64}}()
    harm_u = Dict{String, Vector{Float64}}()

    for name in needed_harmonics
        sat_idxs = get(table.sat_index, name, Int[])
        if isempty(sat_idxs)
            # No satellites → f=1, u=0
            harm_f[name] = ones(Float64, n_dates)
            harm_u[name] = zeros(Float64, n_dates)
        else
            fj_left  = zeros(Float64, n_dates)
            fj_right = zeros(Float64, n_dates)
            for si in sat_idxs
                sat = table.satellites[si]
                # phase angle for this satellite: dP*P + dN*N + dP1*P1 + dEDN*2π
                delta = @. sat.dP * P_v + sat.dN * N_v + sat.dP1 * P1_v + sat.dEDN * 2π
                fj_left  .+= sat.factor .* cos.(delta)
                fj_right .+= sat.factor .* sin.(delta)
            end
            harm_f[name] = @. sqrt((1.0 + fj_left)^2 + fj_right^2)
            harm_u[name] = @. -atan(fj_right, 1.0 + fj_left)   # sign matches hatyan convention
        end
    end

    # Assemble output arrays
    f_out = ones(Float64,  length(const_list), n_dates)
    u_out = zeros(Float64, length(const_list), n_dates)

    for (i, name) in enumerate(const_list)
        if haskey(table.harmonic, name)
            f_out[i, :] = harm_f[name]
            u_out[i, :] = harm_u[name]
        else
            # Shallow: f = ∏ f_dep^|coeff|,  u = Σ coeff * u_dep
            idx = findfirst(s -> s[1] == name, table.shallow)
            _, deps = table.shallow[idx]
            f_row = ones(Float64, n_dates)
            u_row = zeros(Float64, n_dates)
            for (fac, dep) in deps
                f_row .*= harm_f[dep] .^ abs(fac)
                u_row .+= fac .* harm_u[dep]
            end
            f_out[i, :] = f_row
            u_out[i, :] = u_row
        end
    end

    return u_out, f_out
end
