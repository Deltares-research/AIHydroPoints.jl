using TOML

@testset "toml_write" begin
    path = joinpath(temp_dir, "test_settings.toml")
    d = Dict{String,Any}("name" => "test", "n" => 42, "x" => 3.14, "tags" => ["a", "b"])

    toml_write(path, d)
    @test isfile(path)
    @test TOML.parsefile(path) == d

    # error if file exists and overwrite=false (default)
    @test_throws ErrorException toml_write(path, d)

    # overwrite=true replaces the file
    d2 = Dict{String,Any}("name" => "updated")
    toml_write(path, d2; overwrite=true)
    @test TOML.parsefile(path) == d2

    # error if parent directory does not exist
    bad_path = joinpath(temp_dir, "nonexistent_dir", "settings.toml")
    @test_throws ErrorException toml_write(bad_path, d)
end
