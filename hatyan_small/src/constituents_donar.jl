# constituents_donar.jl
#
# Read tidal constituents from the DONAR/hatyan component file format used by Rijkswaterstaat.
# Analogous to series_donar.jl, but populates a TidalConstituents instead of a TimeSeries.
#
# Format structure (space-delimited, fixed-section lines):
#   * ...          — comment lines (ignored)
#   STAT  code  quantity  vertref  unit  wns
#   PERD  startdate  starttime  enddate  endtime  timestep_min
#   CODE  code
#   MIDD  value          — mean water level (stored as constituent "A0", phase = 0)
#   NCOM  n              — number of COMP lines that follow
#   COMP  id  freq  A  phi  name   — one line per constituent
#
# Amplitudes are in the unit given on the STAT line (typically cm) and are converted to metres.
# Phases are in degrees [0, 360).
# Geographic coordinates are not stored in this format; longitudes and latitudes are set to 0.
#
# Example lines:
#   STAT  VLISSGN       WATHTE            NAP           cm           1
#   MIDD     1.000
#   COMP   65    28.984104   174.666   59.47  M2

"""
    read_donar_constituents(filename) -> TidalConstituents

Read a DONAR/hatyan component file. Returns a TidalConstituents with one location.
Amplitudes are converted to metres. The mean water level (MIDD line) is stored as
constituent "A0" with phase 0. Geographic coordinates are not available in this
format and are stored as 0.
"""
function read_donar_constituents(filename::String)
    if !isfile(filename)
        error("File not found: $filename")
    end

    lines = readlines(filename)

    # Metadata
    station_code = ""
    quantity     = ""
    vertref      = ""
    unit         = "cm"
    midd         = 0.0f0    # mean water level (A0)
    midd_found   = false

    # Constituent data collected from COMP lines
    comp_names = String[]
    comp_amp   = Float32[]
    comp_phase = Float32[]

    for line in lines
        s = strip(line)
        isempty(s)          && continue
        startswith(s, "*")  && continue   # comment

        parts = split(s)    # split on any whitespace

        if parts[1] == "STAT" && length(parts) >= 5
            station_code = parts[2]
            quantity     = parts[3]
            vertref      = parts[4]
            unit         = parts[5]

        elseif parts[1] == "MIDD" && length(parts) >= 2
            midd       = parse(Float32, parts[2])
            midd_found = true

        elseif parts[1] == "COMP" && length(parts) >= 6
            # COMP  id  freq  A  phi  name
            name  = parts[6]
            amp   = parse(Float32, parts[4])
            phase = parse(Float32, parts[5])
            push!(comp_names, name)
            push!(comp_amp,   amp)
            push!(comp_phase, phase)
        end
    end

    # Convert amplitudes from cm to metres
    scale = (unit == "cm") ? 0.01f0 : 1.0f0

    # Prepend A0 (mean water level from MIDD line) as the first constituent
    if midd_found
        constituent_names = vcat(["A0"], comp_names)
        amplitudes        = vcat([midd * scale], comp_amp  .* scale)
        phases            = vcat([0.0f0],         comp_phase)
    else
        constituent_names = comp_names
        amplitudes        = comp_amp .* scale
        phases            = comp_phase
    end

    quantity_str = isempty(vertref) ? quantity : "$quantity ($vertref)"

    # Wrap as (1 × n_constituents) matrices to match TidalConstituents layout
    n = length(constituent_names)
    return TidalConstituents(
        reshape(amplitudes, 1, n),
        reshape(phases,     1, n),
        constituent_names,
        [station_code],
        [0.0], [0.0],
        quantity_str,
        station_code,
    )
end
