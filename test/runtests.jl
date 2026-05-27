
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

   @testset "wind stress" begin
      include("test_wind_stress.jl")
   end

   @testset "tidal constituents" begin
      include("test_tidal_comps.jl")
   end

   @testset "waves" begin
      include("test_train_waves.jl")
   end

   @testset "tides" begin
      include("test_train_tides.jl")
   end

   @testset "surges" begin
      include("test_train_surges.jl")
   end

   @testset "settings" begin
      include("test_settings.jl")
   end

   @testset "toml_utils" begin
      include("test_toml_utils.jl")
   end

   @testset "plot_utils" begin
      include("test_plot_utils.jl")
   end

   @testset "data_loading" begin
      include("test_data_loading.jl")
   end

   @testset "input_processing" begin
      include("test_input_processing.jl")
   end

   @testset "model_registry" begin
      include("test_model_registry.jl")
   end

   @testset "pipeline" begin
      include("test_pipeline.jl")
   end

   @testset "abstract model interface" begin
      include("models/test_abstract_model.jl")
   end

   @testset "LinearSurgeModel" begin
      include("models/test_LinearSurgeModel.jl")
   end

   @testset "AttentionSurgeModel" begin
      include("models/test_AttentionSurgeModel.jl")
   end

   @testset "ConvSurgeModel" begin
      include("models/test_ConvSurgeModel.jl")
   end

   @testset "DeepONetTideModel" begin
      include("models/test_DeepONetTideModel.jl")
   end

   @testset "ProductTideModel" begin
      include("models/test_ProductTideModel.jl")
   end

   @testset "ConvWaveModel" begin
      include("models/test_ConvWaveModel.jl")
   end

   @testset "DeepONetWaveModel" begin
      include("models/test_DeepONetWaveModel.jl")
   end

   @testset "ConvInteractionModel" begin
      include("models/test_ConvInteractionModel.jl")
   end

   @testset "ProductInteractionModel" begin
      include("models/test_ProductInteractionModel.jl")
   end
end