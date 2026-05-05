ENV["GKSwstype"] = "nul"   # headless GR — no display needed

@testset "save_loss_plot" begin
    train_losses = [1.0, 0.8, 0.6]
    val_losses   = [1.1, 0.9, 0.7]
    path = joinpath(temp_dir, "losses.png")

    save_loss_plot(path, train_losses, val_losses)
    @test isfile(path)

    # error if file exists and overwrite=false (default)
    @test_throws ErrorException save_loss_plot(path, train_losses, val_losses)

    # overwrite=true replaces the file
    save_loss_plot(path, train_losses, val_losses; overwrite=true)
    @test isfile(path)

    # error if parent directory does not exist
    bad_path = joinpath(temp_dir, "nonexistent_dir", "losses.png")
    @test_throws ErrorException save_loss_plot(bad_path, train_losses)

    # works without val_losses
    path2 = joinpath(temp_dir, "losses_train_only.png")
    save_loss_plot(path2, train_losses)
    @test isfile(path2)
end
