using Test
using AIHydroPoints
using Dates

# Minimal LinearSurgeModel fixtures (self-contained).
function _ft_ts(values::Matrix{Float32}, quantity::String)
    nstations, ntimes = size(values)
    times = DateTime(2020, 1, 1) .+ Hour.(0:ntimes-1)
    names = ["s$i" for i in 1:nstations]
    lons  = Float64.(1:nstations)
    lats  = Float64.(51 .+ (1:nstations))
    return TimeSeries(values, times, names, lons, lats, quantity, "test")
end

_ft_input(; nwind=3, ntimes=60) = Dict{String,TimeSeries}(
    "stress_x" => _ft_ts(randn(Float32, nwind, ntimes), "stress_x"),
    "stress_y" => _ft_ts(randn(Float32, nwind, ntimes), "stress_y"),
    "pressure" => _ft_ts(ones(Float32, nwind, ntimes) .* 1.0f5, "pressure"),
)

_ft_target(; nstations=2, ntimes=60) =
    Dict{String,TimeSeries}("surge" => _ft_ts(randn(Float32, nstations, ntimes), "surge"))

_ft_model(; nstations=2, nwind=3, nlags=4) = LinearSurgeModel(Dict{String,Any}(
    "nlocations_output" => nstations, "nlocations_input" => nwind, "nlags" => nlags))

@testset "weight_decay active (+ LR decay on OptimiserChain)" begin
    input, target = _ft_input(), _ft_target()
    # weight_decay > 0 wraps Adam in OptimiserChain; lr_decay exercises adjust!
    # on that chain (the main API risk).
    ts = TrainingSettings(nepochs=4, batch_size=16, learning_rate=1.0e-2,
                          weight_decay=1.0e-2, lr_decay_factor=0.5, lr_decay_epochs=2)
    tl, _ = train_model!(_ft_model(), ts, input, target)
    @test length(tl) == 4
    @test all(isfinite, tl)
end

@testset "input_noise_std active" begin
    input, target = _ft_input(), _ft_target()
    ts = TrainingSettings(nepochs=3, batch_size=16, learning_rate=1.0e-3,
                          input_noise_std=0.1)
    tl, _ = train_model!(_ft_model(), ts, input, target)
    @test length(tl) == 3
    @test all(isfinite, tl)
end

@testset "early stopping halts before nepochs" begin
    input, target = _ft_input(ntimes=80), _ft_target(ntimes=80)
    # Random target is unlearnable → val RMSE plateaus → short patience stops early.
    ts = TrainingSettings(nepochs=200, batch_size=16, learning_rate=5.0e-3,
                          validation_split=0.3, early_stopping_epochs=3)
    tl, vl = train_model!(_ft_model(), ts, input, target)
    @test length(tl) == length(vl)
    @test length(tl) < 200
end

@testset "early stopping disabled with nothing runs full nepochs" begin
    input, target = _ft_input(ntimes=80), _ft_target(ntimes=80)
    ts = TrainingSettings(nepochs=8, batch_size=16, learning_rate=1.0e-3,
                          validation_split=0.3, early_stopping_epochs=nothing)
    tl, _ = train_model!(_ft_model(), ts, input, target)
    @test length(tl) == 8
end
