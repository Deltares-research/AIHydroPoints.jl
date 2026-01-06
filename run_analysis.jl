# run_analysis.jl
# Make plots and compute statistics for outputs of existing rund

# Move to this folder if not already there
cd(@__DIR__)
# activate the environment
using Pkg
Pkg.activate(".")

# Load required packages
using Dates
using Plots
using JLD2
using Statistics
using LinearAlgebra
using BSON
using NetCDF
using DataFrames
using CSV
#include("netcdf_utils.jl")

# config for datasets
#tide_folder="tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_20epochs"
target_file="DCSM-FM_0_5nm_2011_317stations_his.jld2"
surge_data="tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_surge.jld2"
tide_data="tide_model_317stations_3tl_3yr_1024batch_0p0001reg_64nodes_100epochs_testing_tide.jld2"
# load data
surge_data=load(surge_data)
tide_data=load(tide_data)
target_data=load(target_file)