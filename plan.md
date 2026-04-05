
# Next steps for working on this project

## Main goal
The main goal of this project is to develop a machine learning model for predicting tides, surges and their interaction. The model will be trained on historical data and will be able to make predictions for future events. The model will be implemented in Julia and will be designed to be easily extensible and adaptable to different datasets and configurations

## Next steps
1. [x] Make the unit tests work again.
2. [x] Create a small test dataset
3. [x] Create unit tests for each of the models (tides, surges, interaction, waves) that run fast.
4. [x] Create additional tests for code that is not tested at all.
    a. [x] figure out which code is not tested at all, and which code is tested but not well enough.
    b. [x] write tests for the code that is not tested at all, and improve the tests for the code that is tested but not well enough.
5. Clean up the code and make it more modular and reusable.
    a. [x] replace netcdf_utils.jl with MultiTimeSeries.jl
    b. Structure the model code, with an explicit AbstractModel type, and a common interface for all models.
    c. Extranct the training settings from the model settings
6. Create a Separate data-structure for the training data, building on the MultiTimeSeries.jl package, and use it in the training code.
7. Make training fully controllable from the toml input file, and make it possible to run training from the command line with a specified toml file.
8. Write a separate script for inference

## Checklist for each step:
- all source should eventually be in src/ and all tests should be in test/ and test data should be in test_data/
- make code compilable and runnable
- consider to add new unit tests for the new code
- fix all unit tests `pixi run julia --project -e "using Pkg; Pkg.test()"`
- make sure that the code is well documented and that the documentation is up to date.
- Check if README.md is up to date and update it if necessary.
- Adapt the status in plan.md
- run the unit tests and make sure that they all pass.
- run check_training_scripts.sh to smoke-test all training scripts

## General notes
- we use pixi to install python and julia etc. But we use julia packages in Project.toml.
- we think before we write code. We raise potential issues with the user before continuing
- we discuss design decisions with the user before implementing them, and we make sure that the user is happy with the design before proceeding.
- we keep output of unit tests in test/temp. We clean this folder before running the tests, and leave files for inspection after the tests have run. We make sure that the output of the tests is informative and useful for debugging.

## Status
- [x] Make the unit tests work again.
- [x] Create a small dataset for waves in test_data/waves_2021
- [x] Create a unit test for the wave model that runs fast, using the small dataset in test_data/waves_2021
- [x] Adapt train_waves.jl to use the new data structures in src/
- [x] Adapt train_waves_don.jl to use AIHydroPoints (replace series_ml)
- [x] Fix train_tides.jl (rm force=true)
- [x] Move DCSM-FM*.jld2 and era5*.jld2 datasets to test_data/
- [x] Create unit test for the tide model (test/test_train_tides.jl)
- [x] Create analyse_tides_schureman.jl to perform harmonic tidal analysis and produce tides/surge NetCDF, statistics and constituent CSVs, and plots (full year + Jan 1–15)
- [x] Create surge test dataset using harmonic analysis of the DCSM-FM_0_5nm_2011_5stations_his.jld2 dataset
- [x] Create unit test for the surge model (test/test_train_surges.jl)
- [x] Adapt train_surges.jl to use the test dataset
- [x] Create check_training_scripts.sh to smoke-test all training scripts
