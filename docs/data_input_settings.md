
# Data input settings

For training, the data input is controlled by the `data_settings` dictionary in the toml file.
We want to have a single `load_data` function that can load any dataset based on the settings
and return input and target time series already separated per split.


## Example: surge model (`train_surge.jl`)

The script loads:
- a NetCDF file with the surge time series (quantity `"surge"`)
- a JLD2 file with wind stress and pressure fields (`"stress_x"`, `"stress_y"`, `"pressure"`)

for two splits (training = 2011, testing = 2012), and feeds them into the model as
`input = {stress_x, stress_y, pressure}` and `target = {surge}`.

### TOML

```toml
[[data_settings.files]]
  path      = "test_data/surge_schureman_2011.nc"
  format    = "netcdf"
  split     = "training"
  variables = ["surge"]
  locations = ["VLISSGN", "HOEKVHLD"]   # omit to load all locations

[[data_settings.files]]
  path      = "test_data/era5_wind_stress_2011_testing.jld2"
  format    = "jld2"
  split     = "training"
  variables = ["stress_x", "stress_y", "pressure"]

[[data_settings.files]]
  path      = "test_data/surge_schureman_2012.nc"
  format    = "netcdf"
  split     = "testing"
  variables = ["surge"]
  locations = ["VLISSGN", "HOEKVHLD"]

[[data_settings.files]]
  path      = "test_data/era5_wind_stress_2012_validation.jld2"
  format    = "jld2"
  split     = "testing"
  variables = ["stress_x", "stress_y", "pressure"]

# Which variables go into the model's input dict vs. target dict.
[data_settings.model_io]
  input  = ["stress_x", "stress_y", "pressure"]
  target = ["surge"]
```

### Parsed Dict (what `TOML.parsefile` returns)

```julia
Dict{String,Any}(
  "data_settings" => Dict{String,Any}(
    "files" => [
      Dict("path"=>"test_data/surge_schureman_2011.nc",
           "format"=>"netcdf", "split"=>"training",
           "variables"=>["surge"],
           "locations"=>["VLISSGN","HOEKVHLD"]),
      Dict("path"=>"test_data/era5_wind_stress_2011_testing.jld2",
           "format"=>"jld2", "split"=>"training",
           "variables"=>["stress_x","stress_y","pressure"]),
      Dict("path"=>"test_data/surge_schureman_2012.nc",
           "format"=>"netcdf", "split"=>"testing",
           "variables"=>["surge"],
           "locations"=>["VLISSGN","HOEKVHLD"]),
      Dict("path"=>"test_data/era5_wind_stress_2012_validation.jld2",
           "format"=>"jld2", "split"=>"testing",
           "variables"=>["stress_x","stress_y","pressure"]),
    ],
    "model_io" => Dict(
      "input"  => ["stress_x","stress_y","pressure"],
      "target" => ["surge"],
    ),
  )
)
```

### Result of `load_data(data_settings)`

```julia
Dict{String, NamedTuple}(
  "training" => (
    input  = Dict("stress_x"=>TimeSeries,  # values: (nwind, ntrain)
                  "stress_y"=>TimeSeries,
                  "pressure"=>TimeSeries),
    target = Dict("surge"=>TimeSeries),    # values: (2, ntrain), names: ["VLISSGN","HOEKVHLD"]
  ),
  "testing" => (
    input  = Dict("stress_x"=>TimeSeries,  # values: (nwind, ntest)
                  "stress_y"=>TimeSeries,
                  "pressure"=>TimeSeries),
    target = Dict("surge"=>TimeSeries),    # values: (2, ntest)
  ),
)
```

The training script then becomes:

```julia
data = load_data(data_settings)
train_model!(model, train_settings, data["training"].input, data["training"].target)
predict(model, data["testing"].input)
```

### Example: single file split by timerange

When training and validation come from the same file, repeat the entry with different
`split` and `timerange` values:

```toml
[[data_settings.files]]
  path      = "test_data/surge_schureman_2010_2012.nc"
  format    = "netcdf"
  split     = "training"
  timerange = ["2010-01-01", "2011-12-31"]
  variables = ["surge"]

[[data_settings.files]]
  path      = "test_data/surge_schureman_2010_2012.nc"
  format    = "netcdf"
  split     = "testing"
  timerange = ["2012-01-01", "2012-12-31"]
  variables = ["surge"]
```

---

## Example: wave model (`train_waves.jl`)

The script loads a directory of NOOS files covering 2021, then splits by time:
training = Jan–Sep, testing = Oct–Dec.  Input is wind speed and direction; target
is significant wave height.  The same files appear twice with different `timerange`
values — one entry per split.

### TOML

```toml
[[data_settings.files]]
  path      = "test_data/waves_2021/wind_speed_*.noos"
  format    = "noos"
  split     = "training"
  timerange = ["2021-01-01", "2021-09-30T23:00:00"]
  variables = ["wind_speed"]

[[data_settings.files]]
  path      = "test_data/waves_2021/wind_direction_*.noos"
  format    = "noos"
  split     = "training"
  timerange = ["2021-01-01", "2021-09-30T23:00:00"]
  variables = ["wind_direction"]

[[data_settings.files]]
  path      = "test_data/waves_2021/wave_height_*.noos"
  format    = "noos"
  split     = "training"
  timerange = ["2021-01-01", "2021-09-30T23:00:00"]
  variables = ["wave_height"]

[[data_settings.files]]
  path      = "test_data/waves_2021/wind_speed_*.noos"
  format    = "noos"
  split     = "testing"
  timerange = ["2021-10-01", "2021-12-31T23:00:00"]
  variables = ["wind_speed"]

[[data_settings.files]]
  path      = "test_data/waves_2021/wind_direction_*.noos"
  format    = "noos"
  split     = "testing"
  timerange = ["2021-10-01", "2021-12-31T23:00:00"]
  variables = ["wind_direction"]

[[data_settings.files]]
  path      = "test_data/waves_2021/wave_height_*.noos"
  format    = "noos"
  split     = "testing"
  timerange = ["2021-10-01", "2021-12-31T23:00:00"]
  variables = ["wave_height"]

[data_settings.model_io]
  input  = ["wind_speed", "wind_direction"]
  target = ["wave_height"]
```

### Result of `load_data(data_settings)`

```julia
Dict{String, NamedTuple}(
  "training" => (
    input  = Dict("wind_speed"     => TimeSeries,  # values: (nwind, ntrain)
                  "wind_direction" => TimeSeries),
    target = Dict("wave_height"    => TimeSeries),  # values: (nstations, ntrain)
  ),
  "testing" => (
    input  = Dict("wind_speed"     => TimeSeries,  # values: (nwind, ntest)
                  "wind_direction" => TimeSeries),
    target = Dict("wave_height"    => TimeSeries),  # values: (nstations, ntest)
  ),
)
```

---

## Key design decisions

**`files[].split`** — arbitrary label (`"training"`, `"testing"`, `"validation"`, …)
used by the training script to select which entries to load.  Split membership is
explicit per entry rather than implied by nesting.

**`files[].timerange`** — optional two-element array of ISO-8601 strings.  When
omitted, the full file is loaded.  Listing the same file twice with different
`split`/`timerange` values is the canonical way to carve training and validation
windows out of a single source file.

**`files[].variables`** — two forms are supported and may be mixed:
- Plain string: `"stress_x"` — the on-disk name is used as-is in the in-memory dict.
- Table with alias: `{ name = "u10", as = "stress_x" }` — use when the on-disk name
  and the model-facing name differ.

**`files[].locations`** — optional list of location IDs to load from the file.  When
omitted, all locations in the file are loaded.

**`files[].path`** — relative to the TOML file's directory (or script working directory;
TBD).  Absolute paths are also accepted.  Glob patterns are expanded before loading;
each matched file must contain the same quantity (e.g. `wind_speed_*.noos` selects one
station per file, all for wind speed).

**`model_io.input` / `model_io.target`** — determines how loaded variables are routed
into the `input` and `target` dicts returned by `load_data`.
