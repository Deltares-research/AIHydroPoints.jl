using Plots

"""
    save_loss_plot(path::String, train_losses::Vector, val_losses::Vector=[];
                   overwrite::Bool=false)

Plot `train_losses` (and optionally `val_losses`) against epoch number and save
the figure as a PNG to `path`.

Throws an error if the parent directory does not exist, or if the file already
exists and `overwrite` is `false`.
"""
function save_loss_plot(path::String, train_losses::Vector, val_losses::Vector=[];
                        overwrite::Bool=false)
    isdir(dirname(path)) || error("directory does not exist: $(dirname(path))")
    !overwrite && isfile(path) && error("file already exists (use overwrite=true): $path")

    epochs = 1:length(train_losses)
    p = plot(epochs, train_losses; label="train RMSE", xlabel="epoch", ylabel="RMSE",
             title="Training losses")
    isempty(val_losses) || plot!(p, epochs, val_losses; label="val RMSE")
    savefig(p, path)
end
