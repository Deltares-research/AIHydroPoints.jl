# test_wind_stress.jl

@testset "wind_stress.jl" begin
    @testset "uv_to_stress_xy" begin
        # Test zero wind
        τx, τy = uv_to_stress_xy(0.0, 0.0)
        @test τx ≈ 0.0 atol=1e-10
        @test τy ≈ 0.0 atol=1e-10
        
        # Test pure zonal wind
        τx, τy = uv_to_stress_xy(10.0, 0.0)
        @test τy ≈ 0.0 atol=1e-10
        @test τx ≈ 0.21216693565375663 atol=1e-6
        # Test pure meridional wind
        τx, τy = uv_to_stress_xy(0.0, 10.0)
        @test τx ≈ 0.0 atol=1e-10
        @test τy ≈ 0.21216693565375663 atol=1e-6
        
        # Test diagonal wind
        τx, τy = uv_to_stress_xy(10.0/sqrt(2), 10.0/sqrt(2)) # 45 degree wind of 10 m/s
        @test sqrt(τx^2 + τy^2) ≈ 0.21216693565375663 atol=1e-6
        # check direction
        @test atan(τy, τx) ≈ atan(10.0/sqrt(2), 10.0/sqrt(2)) atol=1e-6
        
        # Test negative components
        τx, τy = uv_to_stress_xy(-10.0/sqrt(2), -10.0/sqrt(2)) # 45 degree wind of 10 m/s in the negative direction
        @test sqrt(τx^2 + τy^2) ≈ 0.21216693565375663 atol=1e-6
        @test atan(τy, τx) ≈ atan(-10.0/sqrt(2), -10.0/sqrt(2)) atol=1e-6
        
    end
end