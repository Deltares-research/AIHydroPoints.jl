using Test
using AIHydroPoints
using Flux

# ──────────────────────────────────────────────────────────────────────────────
# AbstractModel fallbacks
# ──────────────────────────────────────────────────────────────────────────────

@testset "AbstractModel fallbacks" begin
    struct BareModel <: AbstractModel end
    m = BareModel()

    input = Dict{String, TimeSeries}()

    @test_throws ErrorException predict(m, input)
    @test_throws ErrorException get_settings(m)
    @test_throws ErrorException save_params(m, "dummy.jld2")
    @test_throws ErrorException load_params!(m, "dummy.jld2")
    empty = Dict{String, TimeSeries}()
    @test_throws ErrorException train_model!(m, TrainingSettings(), empty, empty)
end

# ──────────────────────────────────────────────────────────────────────────────
# AbstractFluxModel / MyFluxModel
# ──────────────────────────────────────────────────────────────────────────────

@testset "MyFluxModel" begin
    chain    = Dense(4 => 2)
    settings = Dict{String, Any}("foo" => 42)
    m        = MyFluxModel(chain, settings)

    @test m isa AbstractFluxModel
    @test m isa AbstractModel
    @test get_flux_model(m) === chain
    @test get_settings(m)   === settings
    @test get_settings(m)["foo"] == 42

    # Must-implement customisation-point fallbacks should error.
    # (forward and train_model! are now provided generically by AbstractFluxModel,
    #  so they are no longer erroring fallbacks; preprocess — both the predict
    #  2-arg and the train 3-arg form — and postprocess! remain per-family.)
    input = Dict{String, TimeSeries}()
    @test_throws ErrorException preprocess(m, input)
    @test_throws ErrorException preprocess(m, input, input)   # train-form (x, y)
    @test_throws ErrorException postprocess!(input, m, zeros(Float32, 1, 1))
end
