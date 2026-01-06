# series_jld2.jl

using JLD2

"""
JLD2TimeSeries(filename::String)
Create a TimeSeries object from a JLD2 file.
This function reads a JLD2 file and returns a TimeSeries object in memory.

Examples:
```julia
    ts = JLD2TimeSeries("example.jld2")
```
The JLD2 file should contain the following keys:
- `times`: a vector of DateTime objects
- `names`: a vector of station names (Strings)
- `station_x_coordinate`: a vector of station longitudes (Float64)
- `station_y_coordinate`: a vector of station latitudes (Float64)
- `quantity`: a String representing the physical quantity measured (e.g., "waterlevel")
- `<values>`: a matrix of Float32 values with rows as stations and columns as time steps. The name should match the `quantity` key, e.g., "waterlevel".
- `source`: a String representing the source of the data (optional, defaults to "JLD2 file: <filename>")
"""
function JLD2TimeSeries(filename::String)
    # Does the file have jld2 extension?
    if !endswith(filename, ".jld2")
        error("Filename $(filename) does not have a .jld2 extension.")
    end
    # Check if the file exists
    if !isfile(filename)
        error("JLD2 file $(filename) does not exist.")
    end
    # read the JLD2 file
    d=load(filename)
    # Check for variables in the JLD2 file
    quantity="waterlevel"
    values_name="values"
    if !haskey(d,"quantity") # older files may not have this key
        if haskey(d,"waterlevel")
            quantity = "waterlevel"
            values_name="waterlevel"
        else
            error("JLD2 file $(filename) does not contain the 'quantity' key.")
        end
    end
    # look for the values
    if !haskey(d,values_name)
        error("JLD2 file $(filename) does not contain the values at the expected key '$(values_name)'.")
    end
    data= d[values_name][:, :] #load the data into memory
    if !haskey(d,"source")
        source = "JLD2 file: $(filename)"
    else
        source = d["source"]
    end
    if !haskey(d,"times")
        error("JLD2 file $(filename) does not contain the 'times' key.")
    end
    times = d["times"]
    names_key= "station_names"
    if !haskey(d,"station_names")
        if !haskey(d,"names")
            error("JLD2 file $(filename) does not contain the 'names' key.")
        else
            names_key = "names"
        end
    end
    names = d[names_key]
    nstations = length(names)
    if haskey(d,"station_x_coordinate")
        longitudes = d["station_x_coordinate"]
    else
        longitudes = zeros(Float64, nstations)
    end
    if haskey(d,"station_y_coordinate")
        latitudes = d["station_y_coordinate"]
    else
        latitudes = zeros(Float64, nstations)
    end
    # Create in-memory TimeSeries object and return 
    return TimeSeries(data, times, names, longitudes, latitudes, quantity, source)
end

"""Write a TimeSeries to a JLD2 file.
This function writes the time series to a JLD2 file.

Examples:
```julia
    write_to_jld2(ts, "example.jld2")
    write_to_jld2(ts, "example_with_new_source.jld2", replace_source="New source")
```
"""
function write_to_jld2(ts::AbstractTimeSeries, filename::String, replace_source::String="")
    # Chech and refuse if the file already exists
    if isfile(filename)
        error("JLD2 file $(filename) already exists. Please choose a different filename or delete the existing file.")
    end
    # Handle the source
    if replace_source == ""
        source = get_source(ts)
    else
        source = replace_source
    end
    # Write the time series to a JLD2 file
    jldopen(filename, "w") do file
        file["values"] = get_values(ts)
        file["times"] = get_times(ts)
        file["station_names"] = get_names(ts)
        file["station_x_coordinate"] = get_longitudes(ts)
        file["station_y_coordinate"] = get_latitudes(ts)
        file["quantity"] = get_quantity(ts)
        file["source"] = source
    end
end