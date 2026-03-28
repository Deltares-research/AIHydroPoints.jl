# schureman.jl — Schureman constituent tables and nodal factor calculations
#
# Implements loading of the Schureman harmonic/shallow-water tables and the
# constituent-specific calculations for frequency, v₀, u, and f.
# Ported from schureman.py (Python hatyan package).
#
# Reference: Schureman (1941), "Manual of Harmonic Analysis and Prediction of Tides"

# ── Column indices within the v0u matrix (14 columns) ────────────────────────
# T, S, H, P, N, P1, EDN, DKSI, DNU, DQ, DQU, DR, DUK1, DUK2
const _SCOL_T    = 1   # T  – solar time coefficient
const _SCOL_S    = 2   # S  – mean lunar longitude coefficient
const _SCOL_H    = 3   # H  – mean solar longitude coefficient
const _SCOL_P    = 4   # P  – lunar perigee coefficient
const _SCOL_N    = 5   # N  – lunar node coefficient (in table only)
const _SCOL_P1   = 6   # P1 – solar perigee coefficient
const _SCOL_EDN  = 7   # EDN – Extended Doodson Number (phase offset, degrees)
const _SCOL_DKSI = 8   # ξ coefficient for u calculation
const _SCOL_DNU  = 9   # ν coefficient for u calculation
const _SCOL_DQ   = 10  # Q  coefficient for u calculation
const _SCOL_DQU  = 11  # Qu coefficient for u calculation
const _SCOL_DR   = 12  # R  coefficient for u calculation
const _SCOL_DUK1 = 13  # K1 u-correction coefficient
const _SCOL_DUK2 = 14  # K2 u-correction coefficient

# ── Schureman table struct ────────────────────────────────────────────────────

"""
    SchuremanTable

Stores the full Schureman constituent table (harmonic + shallow-water components).

Fields:
- `names`: constituent names in row order
- `index`: `Dict` mapping name → row index (1-based)
- `v0u`: `n × 14` integer matrix: T, S, H, P, N, P1, EDN, DKSI, DNU, DQ, DQU, DR, DUK1, DUK2
- `f`: `n × 12` float matrix: DND73–DND79, DFM1, DFK1, DFL2, DFK2, DFM1C
"""
struct SchuremanTable
    names :: Vector{String}
    index :: Dict{String, Int}
    v0u   :: Matrix{Int}       # n × 14
    f     :: Matrix{Float64}   # n × 12
end

# Module-level lazy cache
const _TABLE_CACHE = Ref{Union{Nothing, SchuremanTable}}(nothing)

"""
    get_schureman_table() -> SchuremanTable

Return the full Schureman constituent table (harmonic + shallow-water).
Built on first call and cached for all subsequent calls.
"""
function get_schureman_table()
    if isnothing(_TABLE_CACHE[])
        _TABLE_CACHE[] = _build_schureman_table()
    end
    return _TABLE_CACHE[]
end


# ── CSV loading ───────────────────────────────────────────────────────────────

function _load_harmonic_csv(filepath::String)
    lines = readlines(filepath)

    # Find the header line (first non-comment, non-empty line)
    header_idx = findfirst(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), lines)
    isnothing(header_idx) && error("No header found in $filepath")

    names    = String[]
    v0u_rows = Vector{Vector{Int}}()
    f_rows   = Vector{Vector{Float64}}()

    for line in lines[(header_idx + 1):end]
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue

        # Strip inline comment
        cidx = findfirst('#', s)
        if !isnothing(cidx)
            s = strip(s[1:cidx - 1])
        end
        isempty(s) && continue

        parts = split(s, ',')
        length(parts) < 27 && continue   # need 1 name + 14 ints + 12 floats

        name = strip(strip(parts[1]), '"')
        isempty(name) && continue

        ints   = [parse(Int,     strip(parts[i])) for i in  2:15]   # T..DUK2
        floats = [parse(Float64, strip(parts[i])) for i in 16:27]   # DND73..DFM1C

        push!(names, name)
        push!(v0u_rows, ints)
        push!(f_rows, floats)
    end

    n = length(names)
    V = Matrix{Int}(undef, n, 14)
    F = Matrix{Float64}(undef, n, 12)
    for i in 1:n
        V[i, :] = v0u_rows[i]
        F[i, :] = f_rows[i]
    end

    return names, V, F
end


function _load_shallow_relations(filepath::String)
    # Returns an ordered Vector of (component_name, expression_string) pairs.
    # Order matters: later relations may reference results of earlier ones.
    pairs = Tuple{String, String}[]
    for line in readlines(filepath)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue

        # Strip inline comment
        cidx = findfirst('#', s)
        if !isnothing(cidx)
            s = strip(s[1:cidx - 1])
        end
        isempty(s) && continue

        comma = findfirst(',', s)
        isnothing(comma) && continue

        name = strip(s[1:comma - 1])
        expr = strip(s[comma + 1:end])
        isempty(name) && continue

        push!(pairs, (name, expr))
    end
    return pairs
end


# ── Expression parser ─────────────────────────────────────────────────────────
#
# Parses arithmetic expressions like "3*M2 - K2 - S2" or "2*S2 + N2 - (M2 + K2)"
# into a list of (coefficient, component_name) pairs.
#
# pos_only=true replaces all '-' with '+' before parsing, which is used for the
# f-factor columns (f uses power products, never division, so signs are all positive).

function _parse_linear_expr(expr::String; pos_only::Bool = false)
    if pos_only
        expr = replace(expr, "-" => "+")
    end
    result = Tuple{Int, String}[]
    _collect_terms!(result, strip(expr), +1)
    return result
end

# Split `expr` at top-level '+' and '-' operators (respecting parentheses).
# Returns a Vector of (sign_char::Char, token::String) pairs.
function _split_toplevel(expr::AbstractString)
    segments     = Tuple{Char, String}[]
    depth        = 0
    buf          = IOBuffer()
    current_sign = '+'

    for c in expr
        if c == '('
            depth += 1
            write(buf, c)
        elseif c == ')'
            depth -= 1
            write(buf, c)
        elseif depth == 0 && (c == '+' || c == '-')
            tok = strip(String(take!(buf)))
            if !isempty(tok)
                push!(segments, (current_sign, tok))
            end
            current_sign = c
            buf = IOBuffer()
        else
            write(buf, c)
        end
    end
    tok = strip(String(take!(buf)))
    if !isempty(tok)
        push!(segments, (current_sign, tok))
    end
    return segments
end

function _collect_terms!(result::Vector{Tuple{Int, String}}, expr::AbstractString, outer_sign::Int)
    for (sign_char, token) in _split_toplevel(expr)
        sign  = (sign_char == '+') ? outer_sign : -outer_sign
        token = strip(token)

        if startswith(token, "(") && endswith(token, ")")
            # Parenthesised group — recurse with the sign distributed inward
            _collect_terms!(result, token[2:end - 1], sign)
        elseif (star = findfirst('*', token); !isnothing(star))
            n    = parse(Int, strip(token[1:star - 1]))
            name = String(strip(token[star + 1:end]))
            push!(result, (sign * n, name))
        else
            push!(result, (sign, String(token)))
        end
    end
end


# ── Table construction ────────────────────────────────────────────────────────

function _build_schureman_table()
    data_dir = joinpath(@__DIR__, "..", "data")
    names, V, F = _load_harmonic_csv(joinpath(data_dir, "data_schureman_harmonic.csv"))
    shallow     = _load_shallow_relations(joinpath(data_dir, "data_schureman_shallowrelations.csv"))

    # Work with mutable copies so we can append shallow-water rows
    names_v = copy(names)
    V_m     = copy(V)
    F_m     = copy(F)
    idx     = Dict(name => i for (i, name) in enumerate(names_v))

    for (sname, expr) in shallow
        haskey(idx, sname) && continue   # already in harmonic table

        terms_vu = _parse_linear_expr(expr)                 # for v0/u: ± arithmetic
        terms_f  = _parse_linear_expr(expr; pos_only=true)  # for f:    all positive

        new_vu = zeros(Int,     14)
        for (coeff, cname) in terms_vu
            i = get(idx, cname, 0)
            i == 0 && error("Shallow '$sname': component '$cname' not in table")
            new_vu .+= coeff .* V_m[i, :]
        end

        new_f = zeros(Float64, 12)
        for (coeff, cname) in terms_f
            i = get(idx, cname, 0)
            i == 0 && error("Shallow '$sname': component '$cname' not in table")
            new_f .+= Float64(coeff) .* F_m[i, :]
        end

        push!(names_v, sname)
        V_m = vcat(V_m, new_vu')
        F_m = vcat(F_m, new_f')
        idx[sname] = length(names_v)
    end

    return SchuremanTable(names_v, idx, V_m, F_m)
end


# ── Helper ────────────────────────────────────────────────────────────────────

function _get_rows(table::SchuremanTable, const_list::Vector{String})
    rows = Vector{Int}(undef, length(const_list))
    for (k, name) in enumerate(const_list)
        haskey(table.index, name) || error("Constituent '$name' not in Schureman table")
        rows[k] = table.index[name]
    end
    return rows
end


# ── Public calculation functions ──────────────────────────────────────────────

"""
    get_schureman_freqs(const_list, dates) -> Vector{Float64}

Angular frequencies in rad/hr for each constituent in `const_list`.
Evaluated at `dates` (typically a single mid-period date).
Returns an `n_const`-vector (frequency at `dates[1]`).
"""
function get_schureman_freqs(const_list::Vector{String}, dates::AbstractVector{DateTime})
    table = get_schureman_table()
    rows  = _get_rows(table, const_list)

    dood      = get_doodson_eqvals(dates; mode=:freq)         # 6 × n
    freq_vars = dood[[DOOD_T, DOOD_S, DOOD_H, DOOD_P, DOOD_P1], :]   # 5 × n

    # Columns in v0u for T,S,H,P,P1 (skip N at column 5)
    v0u_sel = table.v0u[rows, [_SCOL_T, _SCOL_S, _SCOL_H, _SCOL_P, _SCOL_P1]]   # n_const × 5

    DOMEGA = v0u_sel * freq_vars    # n_const × n_dates (angular velocity rad/hr)
    return DOMEGA[:, 1] ./ (2π)    # rad/hr → cycles/hr, at dates[1]
end


"""
    get_schureman_v0(const_list, dates) -> Matrix{Float64}

Initial astronomical phase v₀ (rad) for each constituent at each date.
Returns an `n_const × n_dates` matrix.
"""
function get_schureman_v0(const_list::Vector{String}, dates::AbstractVector{DateTime})
    table = get_schureman_table()
    rows  = _get_rows(table, const_list)

    dood = get_doodson_eqvals(dates)   # 6 × n, position mode
    vars = dood[[DOOD_T, DOOD_S, DOOD_H, DOOD_P, DOOD_P1], :]   # 5 × n

    v0u_sel = table.v0u[rows, [_SCOL_T, _SCOL_S, _SCOL_H, _SCOL_P, _SCOL_P1]]   # n_const × 5
    EDN     = Float64.(table.v0u[rows, _SCOL_EDN])   # n_const

    # DV0 = v0u_sel * vars  +  deg2rad(EDN)  [broadcast EDN as column vector]
    DV0 = v0u_sel * vars .+ deg2rad.(EDN)   # n_const × n_dates
    return DV0
end


"""
    get_schureman_constants(dates) -> NamedTuple

Time-dependent Schureman astronomical constants used for nodal factor calculations.

Returns a `NamedTuple` with:
- `DOMEGA::Float64`  — obliquity of the ecliptic (fixed, rad)
- `DIKL::Float64`    — inclination of moon's orbit to ecliptic (fixed, rad)
- `DC5023::Float64`  — fixed factor (0.5 + 0.75·e²)
- `DC1681::Vector{Float64}` — time-varying (n_dates)
- `DC0365::Vector{Float64}` — time-varying (n_dates)
"""
function get_schureman_constants(dates::AbstractVector{DateTime})
    # Physical constants (Schureman, p. 162)
    DAGC   = 0.01657                          # mean lunar parallax / mean radius
    DE     = 0.054900489                      # eccentricity of moon's orbit
    DMGE   = 1.0 / 81.53                      # mass of moon / mass of earth
    DSGE   = 82.53 / 81.53 * 327932.0        # mass of sun / mass of earth
    DAGC1  = 0.00004261                       # mean solar parallax / mean radius

    DU     = DMGE * DAGC ^ 3
    DU1    = DSGE * DAGC1 ^ 3
    DSACCE = DU1 / DU                         # solar factor

    DOMEGA = deg2rad(23.0 + 27.0/60.0 +  8.26/3600.0)   # obliquity of ecliptic [rad]
    DIKL   = deg2rad( 5.0 +  8.0/60.0 + 43.3546/3600.0) # inclination of moon's orbit [rad]

    DC5023 = 0.5 + 0.75 * DE * DE

    # Time-dependent quantities
    t_sec  = robust_timedelta_sec(dates)
    Tj     = @. (t_sec / 3600.0 + 12.0) / (24.0 * 36525.0)
    DE1    = @. 0.01675104 - 0.0000418 * Tj   # eccentricity of earth's orbit
    DCOFSI = @. (0.5 + 0.75 * DE1 ^ 2) * DSACCE

    DC0365 = @. DCOFSI * sin(DOMEGA) ^ 2
    DC1681 = @. DCOFSI * sin(2 * DOMEGA)

    return (DOMEGA=DOMEGA, DIKL=DIKL, DC5023=DC5023, DC1681=DC1681, DC0365=DC0365)
end


"""
    get_schureman_u(const_list, dates) -> Matrix{Float64}

Nodal phase correction u (rad) for each constituent at each date.
Returns an `n_const × n_dates` matrix wrapped to (−π, π].
"""
function get_schureman_u(const_list::Vector{String}, dates::AbstractVector{DateTime})
    table = get_schureman_table()
    rows  = _get_rows(table, const_list)

    dood  = get_doodson_eqvals(dates)
    N_rad = dood[DOOD_N, :]
    P_rad = dood[DOOD_P, :]

    C = get_schureman_constants(dates)
    DOMEGA = C.DOMEGA; DIKL = C.DIKL; DC1681 = C.DC1681; DC5023 = C.DC5023

    DHOMI = (DOMEGA - DIKL) * 0.5
    DHOPI = (DOMEGA + DIKL) * 0.5
    DTHN  = @. tan(N_rad * 0.5)
    DATC  = @. atan(cos(DHOMI) * DTHN, cos(DHOPI))
    DATS  = @. atan(sin(DHOMI) * DTHN, sin(DHOPI))
    DIH   = @. acos(cos(DIKL) * cos(DOMEGA) - sin(DIKL) * sin(DOMEGA) * cos(N_rad))
    DKSI  = N_rad .- DATC .- DATS
    DNU   = DATC .- DATS

    DPMKSI = P_rad .- DKSI
    DC2PMK = @. cos(2 * DPMKSI)
    DS2PMK = @. sin(2 * DPMKSI)
    DCIH   = @. cos(DIH)
    DCHIH  = @. cos(DIH * 0.5)
    DCTHIH = @. 1.0 / tan(DIH * 0.5)
    DS2IH  = @. sin(2 * DIH)

    DQU    = @. atan(DS2PMK, 3 * DCIH / (DCHIH * DCHIH) + DC2PMK)
    DQ     = @. atan((5 * DCIH - 1) * tan(DPMKSI), 7 * DCIH + 1)
    DNUACC = @. atan(DS2IH * sin(DNU), DS2IH * cos(DNU) + DC1681 / DC5023)
    DR     = @. atan(DS2PMK, DCTHIH * DCTHIH / 6.0 - DC2PMK)
    D2NU2A = @. atan(sin(DIH) ^ 2 * sin(2 * DNU),
                     sin(DIH) ^ 2 * cos(2 * DNU) + C.DC0365 / DC5023)
    DUK1   = -DNUACC
    DUK2   = -D2NU2A

    # Stack u variables: 7 × n_dates
    mul_vars = vcat(DKSI', DNU', DQ', DQU', DR', DUK1', DUK2')

    # u columns in v0u: columns 8..14 (_SCOL_DKSI.._SCOL_DUK2)
    u_sel = table.v0u[rows, _SCOL_DKSI:_SCOL_DUK2]   # n_const × 7

    DU = u_sel * mul_vars                # n_const × n_dates
    DU = @. mod(DU + π, 2π) - π         # wrap to (−π, π]
    return DU
end


"""
    get_schureman_f(const_list, dates, xfac) -> Matrix{Float64}

Nodal amplitude factor f for each constituent at each date.
Returns an `n_const × n_dates` matrix.

`xfac` can be `false`/`nothing` (no correction), `true` (default xfac values),
or a `Dict{String, Float64}` of custom x-factors keyed by constituent name.
"""
function get_schureman_f(
    const_list :: Vector{String},
    dates      :: AbstractVector{DateTime},
    xfac,
)
    table = get_schureman_table()
    rows  = _get_rows(table, const_list)

    dood  = get_doodson_eqvals(dates)
    N_rad = dood[DOOD_N, :]
    P_rad = dood[DOOD_P, :]

    C = get_schureman_constants(dates)
    DOMEGA = C.DOMEGA; DIKL = C.DIKL
    DC1681 = C.DC1681; DC5023 = C.DC5023; DC0365 = C.DC0365

    DHOMI  = (DOMEGA - DIKL) * 0.5
    DHOPI  = (DOMEGA + DIKL) * 0.5
    DSOMEG = sin(DOMEGA)
    DSIKL  = sin(DIKL)
    DTHN   = @. tan(N_rad * 0.5)
    DATC   = @. atan(cos(DHOMI) * DTHN, cos(DHOPI))
    DATS   = @. atan(sin(DHOMI) * DTHN, sin(DHOPI))
    DIH    = @. acos(cos(DIKL) * cos(DOMEGA) - DSIKL * DSOMEG * cos(N_rad))
    DKSI   = @. mod(N_rad - DATC - DATS, 2π)
    DNU    = DATC .- DATS

    # Scalar factors derived from fixed constants DOMEGA, DIKL
    DCHOM  = cos(DOMEGA * 0.5)
    DSHOM  = sin(DOMEGA * 0.5)
    DSOM2  = DSOMEG ^ 2
    DCHOM2 = DCHOM ^ 2
    DSHOM2 = DSHOM ^ 2
    DCHIKL = cos(DIKL * 0.5)
    DCHIK4 = DCHIKL ^ 4
    D132S2 = 1.0 - 1.5 * DSIKL ^ 2
    DMOF65 = (2.0/3.0 - DSOM2) * D132S2
    DMOF66 = DSOM2 * DCHIK4
    DMOF67 = DSOMEG * DCHOM2 * DCHIK4
    DMOF68 = sin(2 * DOMEGA) * D132S2
    DMOF69 = DSOMEG * DSHOM2 * DCHIK4
    DMOF70 = DCHOM2 ^ 2 * DCHIK4
    DMOF71 = DSOM2 * D132S2

    # Time-varying factors (n_dates vectors)
    DSIH   = @. sin(DIH)
    DCHIH  = @. cos(DIH * 0.5)
    DSHIH  = @. sin(DIH * 0.5)
    DS2IH  = @. sin(2 * DIH)
    DTHIH2 = @. (DSHIH / DCHIH) ^ 2
    DIGHI2 = @. cos(DIH) / DCHIH ^ 2

    DPMKSI = P_rad .- DKSI
    DC2PMK = @. cos(2 * DPMKSI)

    DQA    = @. 1.0 / sqrt(0.25 + 1.5 * DIGHI2 * DC2PMK + 2.25 * DIGHI2 ^ 2)
    DRA    = @. 1.0 / sqrt(1.0 - 12 * DTHIH2 * DC2PMK + 36 * DTHIH2 ^ 2)

    # Equations 73–79 from Schureman + K1, L2, K2, M1C factors
    DND73  = @. (2.0/3.0 - DSIH ^ 2) / DMOF65
    DND74  = @. DSIH ^ 2 / DMOF66
    DND75  = @. DSIH * DCHIH ^ 2 / DMOF67
    DND76  = @. DS2IH / DMOF68
    DND77  = @. DSIH * DSHIH ^ 2 / DMOF69
    DND78  = @. DCHIH ^ 4 / DMOF70
    DND79  = @. DSIH ^ 2 / DMOF71
    DFM1   = @. DND75 / DQA
    DFK1   = @. sqrt(DC5023 ^ 2 * DS2IH ^ 2 +
                     2 * DC5023 * DC1681 * DS2IH * cos(DNU) + DC1681 ^ 2) /
               (DC5023 * DMOF68 + DC1681)
    DFL2   = @. DND78 / DRA
    DFK2   = @. sqrt(DC5023 ^ 2 * DSIH ^ 4 +
                     2 * DC5023 * DC0365 * DSIH ^ 2 * cos(2 * DNU) + DC0365 ^ 2) /
               (DC5023 * DMOF71 + DC0365)
    DFM1C  = @. (1.0 - 10 * DSHIH ^ 2 + 15 * DSHIH ^ 4) * DCHIH ^ 2 /
               ((1.0 - 10 * DSHOM2 + 15 * DSHOM2 ^ 2) * DCHOM2)

    multiply_variables = [DND73, DND74, DND75, DND76, DND77, DND78, DND79,
                          DFM1,  DFK1,  DFL2,  DFK2,  DFM1C]

    n_const = length(rows)
    n_dates = length(dates)
    f_i = ones(Float64, n_const, n_dates)

    for (k, variable) in enumerate(multiply_variables)
        power_vec = table.f[rows, k]
        for (i, p) in enumerate(power_vec)
            p == 0.0 && continue
            @. f_i[i, :] *= variable ^ p
        end
    end

    if xfac !== false && xfac !== nothing && xfac != 0
        rows_M2 = _get_rows(table, ["M2"])
        f_M2    = ones(Float64, 1, n_dates)
        for (k, variable) in enumerate(multiply_variables)
            p = table.f[rows_M2[1], k]
            p == 0.0 && continue
            @. f_M2[1, :] *= variable ^ p
        end
        f_i = correct_fwith_xfac(f_i, f_M2, const_list, xfac)
    end

    return f_i
end


"""
    correct_fwith_xfac(f_i, f_M2, const_list, xfac) -> Matrix{Float64}

Apply x-factor amplitude correction to the nodal factor matrix `f_i`.

`xfac` is `true` for default values or a `Dict{String, Float64}` of custom factors.
`f_M2` is the 1×n_dates M2 nodal factor matrix.
"""
function correct_fwith_xfac(
    f_i        :: Matrix{Float64},
    f_M2       :: Matrix{Float64},
    const_list :: Vector{String},
    xfac,
)
    if isa(xfac, Dict)
        xfac_values = xfac
    else
        xfac_values = Dict(
            "MU2"  =>  0.00,
            "N2"   =>  0.00,
            "NU2"  =>  0.80,
            "M2"   =>  0.53,
            "2MN2" =>  0.20,
            "S2"   => -0.82,
            "M4"   =>  0.70,
            "MS4"  =>  0.00,
            "M6"   =>  0.75,
            "2MS6" =>  0.20,
            "M8"   =>  0.70,
            "3MS8" =>  0.60,
        )
    end

    f_out = copy(f_i)
    for (xname, xval) in xfac_values
        idx = findfirst(==(xname), const_list)
        isnothing(idx) && continue
        if all(f_i[idx, :] .≈ 1.0)
            # Non-nodal-dependent component (f=1 by table, like S2)
            f_out[idx, :] .= xval .* (f_M2[1, :] .- 1.0) .+ 1.0
        else
            f_out[idx, :] .= xval .* (f_i[idx, :] .- 1.0) .+ 1.0
        end
    end
    return f_out
end


# ── Generic wrappers ──────────────────────────────────────────────────────────

"""
    get_freqv0_generic(const_list, dood_date_mid, dood_date_start, method) -> (freq, v0)

Return constituent frequencies and initial phases for the given method.

# Arguments
- `const_list`: constituent names
- `dood_date_mid`: single-element vector — mid-point of the analysis/prediction period
  (frequencies are evaluated here)
- `dood_date_start`: single-element vector — start of the period (v₀ evaluated here)
- `method`: `"schureman"` (only supported method)

# Returns
- `freq::Vector{Float64}` — frequencies in **cycles/hr** (length n_const)
- `v0::Matrix{Float64}` — initial phase in **rad**, shape n_const × n_dates
"""
function get_freqv0_generic(
    const_list      :: Vector{String},
    dood_date_mid   :: AbstractVector{DateTime},
    dood_date_start :: AbstractVector{DateTime},
    method          :: String,
)
    if method == "schureman"
        freq = get_schureman_freqs(const_list, dood_date_mid)
        v0   = get_schureman_v0(const_list,   dood_date_start)
    elseif method == "foreman"
        freq = get_foreman_freqs(const_list, dood_date_mid)
        v0   = get_foreman_v0(const_list,   dood_date_start)
    else
        error("Unknown method \"$method\"; supported: \"schureman\", \"foreman\"")
    end
    return freq, v0
end


"""
    get_uf_generic(const_list, dood_date_fu, settings, method) -> (u, f)

Return nodal phase corrections and amplitude factors.

When `settings.nodalfactors = false`, returns u = 0 and f = 1 for all constituents
and dates, bypassing the nodal factor calculation entirely.

# Arguments
- `const_list`: constituent names
- `dood_date_fu`: dates at which to evaluate f and u (either a single mid-point date
  or the full prediction time vector, depending on `settings.fu_alltimes`)
- `settings`: `HatyanSettings` controlling whether nodal factors are applied
- `method`: `"schureman"` (only supported method)

# Returns
- `u::Matrix{Float64}` — nodal phase correction in **rad**, shape n_const × n_dates
- `f::Matrix{Float64}` — nodal amplitude factor (dimensionless), shape n_const × n_dates
"""
function get_uf_generic(
    const_list   :: Vector{String},
    dood_date_fu :: AbstractVector{DateTime},
    settings     :: HatyanSettings,
    method       :: String,
)
    n_const = length(const_list)
    n_dates = length(dood_date_fu)

    if !settings.nodalfactors
        return zeros(Float64, n_const, n_dates), ones(Float64, n_const, n_dates)
    end

    if method == "schureman"
        f = get_schureman_f(const_list, dood_date_fu, settings.xfac)
        u = get_schureman_u(const_list, dood_date_fu)
    elseif method == "foreman"
        u, f = get_foreman_uf(const_list, dood_date_fu)
    else
        error("Unknown method \"$method\"; supported: \"schureman\", \"foreman\"")
    end
    return u, f
end
