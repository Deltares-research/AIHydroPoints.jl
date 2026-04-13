
# A series approach to forecasting tides and storm-surges

The purpose of this time-series based AI model is to have a simple and fast benchmark model. The inputs are a few time-series of winds and air-pressure, from ERA5 in the examples in this repository. The outputs are timeseries of tides, surge and interaction at a number of locations. For now, the model is trained on output of a numerical model. In principle, measurements can also be used for training.

The AI model in this folder contains 3 modules for forecasting, tides, surges and their interaction. 
- Tides: takes only time and location name as input, and should be trained on a multi year dataset of multiple time-series
- Surge: takes winds and pressure at a few points around the North Sea as input, and should be trained with several years of timeseries for a collection of stations.
- Tide-Surge Interation: takes the output of the previous two modules as input and outputs time-series for the non-linear interation. 
This should be summed together result in time-series for the total waterlevel as well as for the individual components. The architecture considers the dynamics to be in part local and in part generic, which is reflected in specific inputs per location and common layers. For example to compute the tide level at the second location a one-hot vector `[0,1,0,...]` is used with as length the number of locations. The other inputs are Doodson phases at that data and time.
The model uses three components and no internal state to achieve a reliable behavior for long lead times. The model is as easily fed with forecasted winds as with winds from a reanalysis for reconstruction of a historical event. Our understanding of the physics of the phenomena has been included in multiple ways into the architecture.

The inputs for winds and air-pressure are sampled at a few relevant locations. In the examples ERA5 fields from the Copernicus Climate Data Store (CDS) are used. Tides require Doodson phases as input, but these are easily computed from the times. The outputs in the examples are from the DCSM-FM model.
Previous values of wind and pressure are taken into account for the surge, and the interaction module also has a time window. You have to make sure that the data provided contains an additional few days to compute the first values. The length is equal to the sum of both windows. It's safe to add a bit extra, so you don't have to change anything in case of a small modification of the model.

## Data downloads

We aim to have the main datasets available in zarr format in the cloud, so they can be easily accessed from the scripts. For now, the scripts download the data and save it locally. You'll need credentials for downloading data. For the S3 storage this amounts to seting the `.aws/credentials` and `.aws/config` files.

## Status

All four models (tides, surges, waves, interaction) use the `AIHydroPoints` library in `src/`.
Unit tests pass for wind stress, tidal constituents, and the wave, tide, and surge training pipelines.
Training and inference settings are split: model-architecture fields live in `TideSettings` /
`SurgeSettings` / `WaveSettings`, while training hyperparameters (epochs, learning rate, etc.)
live in a shared `TrainingSettings` struct — see `docs/settings.md`.
A smoke-test script (`check_training_scripts.sh`) verifies all training scripts end-to-end.

## Intended workflow for training

### ML model Tides
- download sealevel data `get_dcsm_series.jl` - read from 1980-2023 DCSM run stored in the cloud
- train tides `train_tides.jl`.

### ML model Surge
- convert era5 data to datasets for training `get_era_series.jl`
- train surges `train_surges.jl`

### ML model for tide-surge interaction
- train tide-surge interaction model `train_interaction.jl`

### Combined analysis
- make analysis of a trained model for a new input dataset `run_analysis.jl`

## Source code (`src/`)

All library code lives in `src/` and is exposed as the `AIHydroPoints` Julia package.

- `models/abstract_model.jl` — `AbstractModel` abstract type (common interface for all models)
- `models/training_settings.jl` — `TrainingSettings` struct (epochs, learning rate, regularisation, etc.)
- `tidal_comps.jl` — Doodson phases, lunar-to-solar conversion, named tidal constituents
- `wind_stress.jl` — Convert 10 m winds to stress components
- `waves.jl` — Wave model: `WaveSettings`, `create_wave_model`, `train_epoch!`, `predict`, `stats_skipnan`, `plot_series`
- `tides.jl` — Tide model: `TideSettings`, `TideModel`, `prepare_inputs`, `predict`, `plot_series`
- `surge.jl` — Surge model: `SurgeSettings`, `SurgeModel`, `prepare_inputs`, `predict`, `plot_series`
- `interaction.jl` — Tide–surge interaction model: `InteractionSettings`
- `training.jl` — Shared training loop (`train_model`), `save_model`, `load_model`, `save_settings`, `load_settings`
- `graph_network.jl`, `attention.jl` — Graph network and attention building blocks

Time-series I/O (in-memory, NetCDF, Zarr, JLD2, NOOS) is provided by the external
[MultiTimeSeries.jl](https://github.com/robot144/MultiTimeSeries.jl) package.

See `docs/settings.md` for a full reference of all settings fields.
## Analysis scripts

- `analyse_tides_schureman.jl` — Harmonic tidal analysis (Schureman, 95 constituents) on a
  yearly DCSM-FM 5-station JLD2 dataset. Produces tides and surge as NetCDF, per-station
  full-year and Jan 1–15 plots, a statistics CSV, and a constituent amplitude/phase CSV.
  Takes the year as a command-line argument (default 2010):
  ```
  julia --project analyse_tides_schureman.jl 2011
  ```
  Output is written to `output_tides_<year>/`.

## Other
- `test_minio_zarr_with_julia.ipynb`
    Test script for downloading a subset of the 1980-2023 DCSM run
- `hatyan_core.py`
    Copy of basic tide routines from Haytan2

## Design ideas

The different models all need time-series and a configuration as inputs. Each model has different configuration options when studied in more detail. 
- Configurations can use a TOML file, wich maps to a data-structure in memory. During development scripts can override values. Production scripts should be fully comfigurable from the config file
- For the model config we can make the time-span for the computation optional. When not given the model settings and times of the dataset are used to determine the start and end time.
- Different configs should share elements where useful
- Long term goal could be more generic scripts like `ai_hydro_train.jl` and `ai_hydro_predict.jl` with the model settings etc all in a config.

## TODO

### Waves
- [x] wave model architecture and training in `src/waves.jl`
- [x] `train_waves.jl` updated to use `AIHydroPoints` library
- [x] unit test for wave training pipeline (`test/test_train_waves.jl`)
### Tides
- [x] convert DCSM to zarr and store in cloud
- [x] basic routines for tides (`src/tidal_comps.jl`)
- [x] create a few training datasets for tides
- [x] prototype for tide training
- [x] export to netcdf his file
- [x] small test dataset in `test_data/DCSM-FM_0_5nm_*_5stations_his.jld2`
- [x] unit test for tide training pipeline (`test/test_train_tides.jl`)
- [x] `train_tides.jl` updated to use `AIHydroPoints` library
- [ ] check with cpu and gpu. Is gpu faster?
- [ ] rewrite `get_dcsm_series.jl` to use TimeSeries
### Surge
- [x] download ERA5 data — see [DataCollector.jl repo](https://github.com/robot144/DataCollector.jl)
- [x] convert to jld2 and compute stresses
- [x] `train_surges.jl` updated to use `AIHydroPoints` library
- [x] unit test for surge training pipeline (`test/test_train_surges.jl`)
### Tide-Surge Interaction
- [x] create AI model and train
- [ ] update `train_interaction.jl` to use `AIHydroPoints` library
### Cleaner code
- [x] Unit tests (`test/`)
- [x] TimeSeries type via MultiTimeSeries.jl
- [x] Selection of locations and times for TimeSeries
- [x] Read and write time-series (NetCDF, Zarr, JLD2, NOOS)
- [x] `wind_stress.jl` moved to `src/`
- [x] tidal constituent routines moved to `src/tidal_comps.jl`
- [x] `AbstractModel` abstract type in `src/models/abstract_model.jl`
- [x] `TrainingSettings` separated from model settings; documented in `docs/settings.md`
- [x] smoke-test script `check_training_scripts.sh`
- [ ] update `train_interaction.jl` to use `AIHydroPoints` library
- [ ] add unit test for interaction model



## Statistics and hyperparameters

### Tide model
- tide layers: 3
- channels per layer: 64
- regularization: 0.0001
- batch size: 1024
- stations: 314 (all)
- epochs: 20
- train perdiod: 2008, 2009, 2010
- testing period: 2011
- mean RMSE train: 0.216
- mean RMSE test: 0.230