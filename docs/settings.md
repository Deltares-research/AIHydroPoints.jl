# Settings Reference

A run is configured by a single TOML file.  Every input file must declare a
top-level `format_version` (see [Format versions](#format-versions)):

```toml
format_version = 2
```

The required tables differ between `train.jl` and `predict.jl`:

| Table | `train.jl` | `predict.jl` | Purpose |
|---|---|---|---|
| `[run_info]` | yes | no | [Run identifier and description](run_info_settings.md) |
| `[model_settings]` | yes | yes | [Model directory and architecture](model_settings.md) |
| `[train_settings]` | yes | no | [Training hyperparameters](training_settings.md) |
| `[data_settings]` | yes | yes | [Data files and input/output routing](data_input_settings.md) |
| `[output_settings]` | no | no | [Plots, statistics, and timeseries output](output_settings.md) |

For `predict.jl`, `[model_settings]` only requires `model_dir` — the full architecture
settings are loaded automatically from `model_dir/model_settings.toml`.

## Format versions

`format_version` is a top-level integer key that lets the reader detect and reject
input files written for an older, incompatible format instead of silently
mis-reading them.  A file with **no** `format_version` key is treated as **version
1** and rejected with a message describing the migration; the current version is
**2**.  Bump this number whenever a breaking change is made to the input format,
and add a row below.

| Version | Changes |
|---|---|
| 1 | Original format (implicit — no `format_version` key). |
| 2 | `[train_settings]`: `nbatches` → `batch_size`, `lr_decay_rate` → `lr_decay_epochs`, `patience` → `early_stopping_epochs`, `weight_reg` → `weight_decay` (default `1.0e-4` → `0.0`); `val_daterange` removed. `[[output_settings.outputs]]`: plot flags gained a `plot_` prefix (`timeseries` → `plot_timeseries`, `fft` → `plot_fft`, `scatter` → `plot_scatter`, `tidal_analysis` → `plot_tidal_analysis`) and `residuals` → `write_residuals`. Unknown keys in `[train_settings]` and `[[data_settings.files]]` are now rejected. |

To migrate a version-1 file: add `format_version = 2`, rename the keys per the
table above, and remove `val_daterange`.
