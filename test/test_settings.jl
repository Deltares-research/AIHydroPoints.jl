using Test
using AIHydroPoints

config_dir = joinpath(@__DIR__, "..", "test_data", "config_files")

@testset "load_settings" begin

    @testset "TideSettings" begin
        fn = joinpath(config_dir, "settings_tidemodel.toml")
        @test isfile(fn)

        s = load_settings(fn)
        @test s isa TideSettings
        @test s.model_name    == "TestTideModel"
        @test s.nepochs       == 2
        @test s.nbatches      == 1024
        @test s.nstations     == 5
        @test s.learning_rate ≈  0.001
        @test s.lr_decay_factor ≈ 0.9
        @test s.lr_decay_rate == 50
        @test s.patience      == 10
        @test s.use_gpu       == true
        @test s.val_daterange == ["2011-01-01T00:00:00", "2011-01-15T00:00:00"]
        @test s.checkpoints   == [40, 80, 120, 160]
        @test s.model_pars["nlayers_branch"] == 2
        @test s.model_pars["nhidden_branch"] == 16
        @test s.model_pars["nlayers_trunk"]  == 0
        @test s.model_pars["nhidden_trunk"]  == 8
        @test s.model_pars["nlayers_down"]   == 1
    end

    @testset "SurgeSettings" begin
        fn = joinpath(config_dir, "settings_surgemodel.toml")
        @test isfile(fn)

        s = load_settings(fn)
        @test s isa SurgeSettings
        @test s.model_name    == "TestSurgeModel"
        @test s.nepochs       == 2
        @test s.nbatches      == 1024
        @test s.nstations     == 5
        @test s.nwind         == 9
        @test s.nlags         == 16
        @test s.learning_rate ≈  0.001
        @test s.lr_decay_factor ≈ 0.1
        @test s.lr_decay_rate == 400
        @test s.patience      == 5
        @test s.use_gpu       == false
        @test s.val_daterange == ["2012-01-01T00:00:00", "2012-01-15T00:00:00"]
        @test s.checkpoints   == [40, 80, 120, 160]
        @test s.model_pars["theta"]          ≈ 10000.0
        @test s.model_pars["nheads"]         == 4
        @test s.model_pars["nlayers_branch"] == 2
        @test s.model_pars["nlayers_trunk"]  == 0
        @test s.model_pars["nhidden_trunk"]  == 16
        @test s.model_pars["nembed"]         == 16
    end

    @testset "WaveSettings" begin
        fn = joinpath(config_dir, "settings_wavemodel.toml")
        @test isfile(fn)

        s = load_settings(fn)
        @test s isa WaveSettings
        @test s.model_name      == "wave_model_10to11_explyr3_3stations"
        @test s.nepochs         == 2
        @test s.nbatches        == 256
        @test s.nstations       == 11
        @test s.nwind           == 10
        @test s.nlags           == 16
        @test s.n_input_channels == 64
        @test s.learning_rate   ≈  0.001
        @test s.wind_scale      ≈  0.5
        @test s.wave_scale      ≈  3.0
        @test s.input_noise_std ≈  0.3
        @test s.patience        == 5
        @test s.use_gpu         == true
        @test s.model_pars["nchannel"]   == [64, 64, 64, 1]
        @test s.model_pars["activation"] == "swish"
    end

end
