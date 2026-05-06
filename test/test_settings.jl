using Test
using AIHydroPoints

config_dir = joinpath(@__DIR__, "..", "test_data", "config_files")

@testset "load_settings" begin

    @testset "TideSettings" begin
        fn = joinpath(config_dir, "settings_tidemodel.toml")
        @test isfile(fn)

        ms, ts = load_settings(fn)
        @test ms isa TideSettings
        @test ts isa TrainingSettings
        @test ms.model_name    == "TestTideModel"
        @test ms.nstations     == 5
        @test ms.use_gpu       == true
        @test ms.model_pars["nlayers_branch"] == 2
        @test ms.model_pars["nhidden_branch"] == 16
        @test ms.model_pars["nlayers_trunk"]  == 0
        @test ms.model_pars["nhidden_trunk"]  == 8
        @test ms.model_pars["nlayers_down"]   == 1
        @test ts.nepochs       == 2
        @test ts.nbatches      == 1024
        @test ts.learning_rate ≈  0.001
        @test ts.lr_decay_factor ≈ 0.9
        @test ts.lr_decay_rate == 50
        @test ts.patience      == 10
        @test ts.val_daterange == ["2011-01-01T00:00:00", "2011-01-15T00:00:00"]
        @test ts.checkpoints   == [40, 80, 120, 160]
    end

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
