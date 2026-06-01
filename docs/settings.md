# Settings Reference

A run is configured by a single TOML file.  The required tables differ between
`train.jl` and `predict.jl`:

| Table | `train.jl` | `predict.jl` | Purpose |
|---|---|---|---|
| `[run_info]` | yes | no | [Run identifier and description](run_info_settings.md) |
| `[model_settings]` | yes | yes | [Model directory and architecture](model_settings.md) |
| `[train_settings]` | yes | no | [Training hyperparameters](training_settings.md) |
| `[data_settings]` | yes | yes | [Data files and input/output routing](data_input_settings.md) |
| `[output_settings]` | no | no | [Plots, statistics, and timeseries output](output_settings.md) |

For `predict.jl`, `[model_settings]` only requires `model_dir` — the full architecture
settings are loaded automatically from `model_dir/model_settings.toml`.
