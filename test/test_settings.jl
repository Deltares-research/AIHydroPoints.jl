using Test
using AIHydroPoints

config_dir = joinpath(@__DIR__, "..", "test_data", "config_files")

@testset "load_settings" begin

    @testset "WaveSettings" begin
        fn = joinpath(config_dir, "settings_wavemodel.toml")
        @test isfile(fn)

        ms, ts = load_settings(fn)
        @test ms isa WaveSettings
        @test ts isa TrainingSettings
        @test ms.model_name      == "wave_model_10to11_explyr3_3stations"
        @test ms.nstations       == 11
        @test ms.nwind           == 10
        @test ms.nlags           == 16
        @test ms.n_input_channels == 64
        @test ms.use_gpu         == true
        @test ms.wind_scale      ≈  0.5
        @test ms.wave_scale      ≈  3.0
        @test ms.model_pars["nchannel"]   == [64, 64, 64, 1]
        @test ms.model_pars["activation"] == "swish"
        @test ts.nepochs         == 2
        @test ts.nbatches        == 256
        @test ts.learning_rate   ≈  0.001
        @test ts.input_noise_std ≈  0.3
        @test ts.patience        == 5
    end

end
