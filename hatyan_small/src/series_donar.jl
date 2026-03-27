# series_donar.jl
#
# Read time series in the DONAR format used by Rijkswaterstaat (Dutch water authority).
#
# Format structure:
#   [IDT;...] — identification block
#   [W3H]     — metadata block (PAR, EHD, HDH, LOC, TYP, ...)
#   [RKS]     — time specification block (TYD line with start/end/timestep)
#   [TPS]     — status block
#   [WRD]     — data block: colon-separated value/qualitycode pairs, wrapped over multiple lines
#
# Values are integers in cm; missing values are encoded as -999999999.
# Values are converted to metres (divide by 100) on read.
# Coordinates in the LOC line are in the Dutch RD system and are stored as-is (not converted to lon/lat).
#
# Example header:
#   LOC;VLISSGN;Vlissingen;P;RD;3048000;38522000
#   TYD;20190101;0000;20191231;2350;10;min
#
# Example data line:
#   102/0:90/0:78/0:66/0:52/0:39/0:

using Dates

const DONAR_MISSING_VALUE = -999999999

"""
    read_donar_timeseries(filename) -> TimeSeries

Read a single DONAR-format timeseries file. Returns a TimeSeries with water levels in
metres. Missing values (encoded as $DONAR_MISSING_VALUE in the file) become NaN.
The time vector is reconstructed from the TYD header line; values are read from the
[WRD] block in order.

Station coordinates (RD system) are stored in the `longitudes` and `latitudes` fields
as raw RD x/y values (not converted to geographic coordinates).
"""
function read_donar_timeseries(filename::String)
    if !isfile(filename)
        error("File not found: $filename")
    end

    lines = readlines(filename)

    # Metadata fields
    station_code = ""
    station_name = ""
    quantity     = ""
    unit         = "cm"
    vertref      = ""
    rd_x         = 0.0
    rd_y         = 0.0
    start_dt     = DateTime(1900, 1, 1)
    end_dt       = DateTime(1900, 1, 1)
    timestep_min = 60

    in_wrd       = false
    data_tokens  = String[]

    for line in lines
        s = strip(line)

        if startswith(s, "[WRD]")
            in_wrd = true
            continue
        end

        if in_wrd
            # Collect all value/qc tokens; each line ends with ":" so split produces a trailing empty string
            append!(data_tokens, filter(!isempty, split(s, ":")))
            continue
        end

        # --- metadata lines (before [WRD]) ---
        if startswith(s, "LOC;")
            parts = split(s, ";")
            length(parts) >= 3 && (station_code = strip(parts[2]); station_name = strip(parts[3]))
            if length(parts) >= 7
                rd_x = tryparse(Float64, strip(parts[6]))
                rd_y = tryparse(Float64, strip(parts[7]))
                rd_x === nothing && (rd_x = 0.0)
                rd_y === nothing && (rd_y = 0.0)
            end
        elseif startswith(s, "PAR;")
            parts = split(s, ";")
            length(parts) >= 2 && (quantity = strip(parts[2]))
        elseif startswith(s, "EHD;")
            parts = split(s, ";")
            length(parts) >= 3 && (unit = strip(parts[3]))
        elseif startswith(s, "HDH;")
            parts = split(s, ";")
            length(parts) >= 2 && (vertref = strip(parts[2]))
        elseif startswith(s, "TYD;")
            parts = split(s, ";")
            if length(parts) >= 7
                start_dt = DateTime(strip(parts[2]) * lpad(strip(parts[3]), 4, "0"), dateformat"yyyymmddHHMM")
                end_dt   = DateTime(strip(parts[4]) * lpad(strip(parts[5]), 4, "0"), dateformat"yyyymmddHHMM")
                timestep_val  = parse(Int, strip(parts[6]))
                timestep_unit = strip(parts[7])
                if timestep_unit == "min"
                    timestep_min = timestep_val
                elseif timestep_unit in ("hr", "uur")
                    timestep_min = timestep_val * 60
                else
                    error("Unknown timestep unit in TYD line: $timestep_unit")
                end
            end
        end
    end

    # Build time vector from TYD header
    times   = collect(start_dt:Minute(timestep_min):end_dt)
    n_times = length(times)

    if length(data_tokens) < n_times
        @warn "DONAR file has fewer data tokens ($(length(data_tokens))) than expected timesteps ($n_times): $filename"
    end

    # Parse data values: each token is "value/qualitycode"
    values = fill(Float32(NaN), 1, n_times)
    for i in 1:min(n_times, length(data_tokens))
        token = strip(data_tokens[i])
        slash = findfirst('/', token)
        val_str = slash === nothing ? token : token[1:slash-1]
        val_int = tryparse(Int, strip(val_str))
        if val_int !== nothing && val_int != DONAR_MISSING_VALUE
            if unit == "cm"
                values[1, i] = Float32(val_int / 100.0)   # convert cm → m
            else
                values[1, i] = Float32(val_int)
            end
        end
    end

    # Build quantity string including vertical reference
    quantity_str = isempty(vertref) ? quantity : "$quantity ($vertref)"
    name         = isempty(station_name) ? station_code : station_name

    return TimeSeries(values, times, [name], [rd_x], [rd_y], quantity_str, station_code)
end
