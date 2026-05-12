# Output settings

Output is controlled by the optional `[output_settings]` table in the run TOML file.
It has two levels: global keys that apply to the whole run, and a
`[[output_settings.outputs]]` array whose entries each describe one output block
(a full split or a named event window within a split).

---

## Global keys

| Key | Default | Description |
|---|---|---|
| `series_format` | `"netcdf"` | Format for written time-series files. One of `"netcdf"`, `"jld2"`, `"noos"`. |
| `write_summary` | `true` | Write a `summary.toml` with overall RMSE, model parameter count, and predict time. |

---

## Per-entry keys (`[[output_settings.outputs]]`)

Each entry describes one output block. Entries with no `timerange` cover the full split;
entries with a `timerange` define a named event sub-window. The `name` field is used as
the label in output file names; if omitted, the `split` label is used.

| Key | Default | Description |
|---|---|---|
| `split` | — | **Required.** Which data split to use (e.g. `"test"`, `"train"`). |
| `name` | `split` value | Label used in output file names. Required when the same split appears more than once. |
| `timerange` | full split | Two ISO-8601 strings restricting all outputs to a sub-window. Same syntax as `data_settings.files[].timerange`. |
| `timeseries` | `true` if split is `"testing"`, else `false` | Plot predicted vs observed time series, one PNG per station. |
| `fft` | `false` | Plot FFT spectra (observations, predicted, residual), one PNG per station. |
| `scatter` | `false` | Scatter plot of predicted vs observed (one point per timestep), one PNG per station. |
| `write_stats` | `true` if split is `"testing"`, else `false` | Write per-station statistics (RMSE, bias, correlation) to `stats_<name>.csv`. |
| `write_series` | `false` | Write predicted time series to `series_<name>.<ext>` in `series_format`. |

If no `[[output_settings.outputs]]` entries are given, a single default test entry is used
(equivalent to `split = "test"` with per-entry defaults applied).

---

## Output file naming

All files are written relative to `model_dir`. Per-station plots go into a subfolder
named `<name>_<type>` so that plots for different splits and event windows stay
separated and are easy to browse.

| Output | Path |
|---|---|
| Time-series plot | `<name>_timeseries/<station>.png` |
| FFT plot | `<name>_fft/<station>.png` |
| Scatter plot | `<name>_scatter/<station>.png` |
| Statistics | `stats_<name>.csv` |
| Predicted series | `series_<name>.<ext>` |
| Run summary | `summary.toml` |

Where `<name>` is the entry's `name` value (defaults to `split`), and `<ext>` is
determined by `series_format`.

---

## Example

```toml
[output_settings]
series_format = "netcdf"
write_summary = true

# Full test split: plots, scatter, stats
[[output_settings.outputs]]
split        = "testing"
timeseries   = true
fft          = false
scatter      = true
write_stats  = true
write_series = false

# Full train split: stats only
[[output_settings.outputs]]
split        = "training"
timeseries   = false
fft          = false
scatter      = false
write_stats  = true
write_series = false

# Named event within test split: all outputs including FFT
[[output_settings.outputs]]
split        = "testing"
name         = "storm_jan_2012"
timerange    = ["2012-01-13", "2012-01-16"]
timeseries   = true
fft          = true
scatter      = true
write_stats  = true
write_series = false
```

---

## Current status

The `[[output_settings.outputs]]` design described above is **not yet implemented**.

What is currently implemented (in `src/models/abstract_flux_model.jl`) is a simpler
flat `[output_settings]` with three global boolean keys:

| Key | Default | Description |
|---|---|---|
| `plot_train` | `false` | Plot predictions vs observations for the training split. |
| `plot_test` | `true` | Plot predictions vs observations for the testing split. |
| `plot_fft` | `false` | Add FFT spectral panels to each station plot. |

These produce one PNG per station (2-panel or 4-panel with FFT) via
`_plot_station_series` in `src/plot_utils.jl`. No scatter, no stats, no series output,
no event windows.

Migration to the new design is tracked under step 9 of `plan.md`.
