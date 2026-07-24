# Training settings

Shared across all model types.  These fields live in `[train_settings]` in the TOML and are
not saved with the model — they are only needed during training.

> **Disabling an optional setting:** TOML has no null literal, so you cannot write
> `key = nothing` — it is a parse error. To disable an optional key in a TOML file,
> **omit it** (or comment it out with `#`); the default then applies. The `nothing`
> value is only valid when building `TrainingSettings` (or its settings `Dict`)
> directly from Julia.

| Field | Type | Default | Description |
|---|---|---|---|
| `nepochs` | `Int` | `100` | Number of training epochs. |
| `batch_size` | `Int` | `1024` | Minibatch size. |
| `learning_rate` | `Float64` | `1.0e-3` | Initial learning rate for the Adam optimiser. |
| `lr_decay_factor` | `Float64` | *(omit to disable)* | Multiply the learning rate by this value every `lr_decay_epochs` epochs (e.g. `0.1` reduces LR 10×). Both keys must be present to activate decay. |
| `lr_decay_epochs` | `Int` | *(omit to disable)* | Epoch interval between LR decay steps (e.g. `10` decays at epochs 10, 20, 30, …). Both keys must be present to activate decay. |
| `weight_decay` | `Float64` | `0.0` | L2 weight-decay coefficient (`0` = off). When `> 0`, Adam is wrapped in `OptimiserChain(WeightDecay, Adam)`. |
| `early_stopping_epochs` | `Int` | `5` | Stop once validation RMSE has not improved for this many consecutive epochs (requires validation data; omit to disable). |
| `checkpoints` | `Vector{Int}` | *(omit to disable)* | Epoch numbers at which to save a model snapshot. |
| `validation_split` | `Float64` | `0.0` | Fraction of the training series (from the end) held out for validation loss (0 = disabled). An explicit `split = "validation"` data file takes priority over this. |
| `input_noise_std` | `Float64` | `0.0` | Std of Gaussian noise added to every input tensor per batch during training (data augmentation; `0` = off). |

> **Renamed in format version 2:** `nbatches` → `batch_size`, `lr_decay_rate` →
> `lr_decay_epochs`, `patience` → `early_stopping_epochs`, `weight_reg` →
> `weight_decay` (default also changed `1.0e-4` → `0.0`). The `val_daterange` key
> was removed. See [settings.md](settings.md#format-versions).
