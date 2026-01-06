
using Test
using AIHydroPoints

#clear cache
cache_dir = joinpath(pwd(),".cache")
if isdir(cache_dir)
   rm(cache_dir,recursive=true)
else
   mkdir(cache_dir)
end
@show cache_dir

temp_dir= joinpath(pwd(),"temp")
if isdir(temp_dir) # remove temp directory if it exists
    rm(temp_dir,recursive=true)
end
mkdir(temp_dir) # create a new empty temp directory
@show temp_dir

@testset "Tools for machine learning based on time-series" begin

   @testset "Abstract time series tools" begin
      include("test_abstract_series.jl")
   end

   @testset "Time series tools" begin
      include("test_series.jl")
   end

   @testset "Netcdf time series tools" begin
      include("test_series_netcdf.jl")
   end

   @testset "Zarr time series tools" begin
      include("test_series_zarr.jl")
   end
   
   @testset "JLD2 time series tools" begin
      include("test_series_jld2.jl")
   end

   @testset "noos ascii time series tools" begin
      include("test_series_noos.jl")
   end

end