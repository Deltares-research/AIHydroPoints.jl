
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

## Status

Currently the three modules tide, surge and interaction are working. There are a few succesful models. However, the code is still messy. We're working on improvements to the code, but it's still not working again and many scripts still use old routines.

## Intended workflow for training

### ML model Tides
- download sealevel data `get_dcsm_series.jl` - read from 1980-2023 DCSM run stored in the cloud
- train tides `train_tides.jl`.

### ML model Surge
- convert era5 data to datasets for training `prepare_data_for_surge.jl`
- train surges `train_surges.jl`

### ML model for tide-surge interaction
- train tide-surge interaction model `train_interaction.jl`

### Combined analysis
- make analysis of a trained model for a new input dataset `run_analysis.jl`

## Supporting source code
- `tide_time.jl`
    Some tide routines converted from Hatyan, mainly to compute phases for the basic Doodson frequencies
- `wind_stress.jl`
    Convert 10m winds to stresses
- `netcdf_utils.jl`
    Write data in delft3d-fm his format nc files

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

### Tides 
- [x] convert DCSM to zarr and store in cload
- [x] basic routines for tides
- [x] create a few training datassets for tides
- [x] prototype for tide training
- [x] export to netcdf his file
- [ ] rewrite `train_tides.jl` to use TimeSeries datasets
- [ ] check with cpu and gpu. Is gpu faster?
- [ ] rewrite `get_dcsm_series.jl` to use TimeSeries
### Surge
- [x] download ERA5 data # see [DataCollector.jl repo](https://github.com/robot144/DataCollector.jl)
- [x] convert to jld2 and compute stresses
- [ ] rewrite `train_surge.jl` to use TimeSeries
- [ ] test `train_surge.jl` with test-dataset
- ? convert ERA5 to zarr and store in cloud (also in DataCollector.jl)
### Tide-Surge Interaction
- [x] create AI model and train
- [ ] update model to unse TimeSeries
### Cleaner code
- [x] Unit tests
- [x] TimeSeries type based on AbstractTimeSeries
- [x] Selection of locations and times for TimeSeries
- [x] Read and write time-series
    - [x] NetCDF
    - [x] Zarr
    - [x] JLD2
- [ ] move `tide_time.jl` to src and add a test and check train_tides
- [ ] move `wind_stress.jl` to src and add a test



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