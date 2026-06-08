
# Background and motivation

NOTE: This doc is about the concepts and math, not the code structure or implementation details.

> **Draft warning.** This document is still a very rough draft and should be
> read with care. The conceptual notation and the actual tensor layouts used
> in the code are not yet fully consistent — see
> [`notes_dimensions.md`](notes_dimensions.md) for an ongoing review of those
> inconsistencies and the planned refactor. The index-notation conventions
> used below are spelled out in the [Notation](notation.md) appendix.

The AIHydroPoints.jl package is developed to explore the application of machine learning models to predict water levels, tides, waves, and more at a network of coastal stations. Although this is the intended application scope at te moment, the package is designed to be flexible and extensible, allowing for the incorporation of additional phenomena and model architectures in the future. To keep the repository organized and maintainable, the codebase is structured around a common concept for the AI models. The key assumptions and design principles are as follows:
- each model takes a set of time-series as input, and produces a set of time-series as output. Both the input and output time-series are given as a combination of points (locations) and variables (e.g. "wind_x", "surge", etc.). 
- The models are causal, meaning that the prediction at time t can only depend on the input data up to time t (or t - lag, where lag is a hyperparameter). All previous times t-lag until t are considered as the "input window" for the prediction at time t. 
- The models are not autoregressive. The models are trained to predict the output time-series at time t, given the input time-series in the input window. This implies that the models do not use any values of the output time-series (at any time) as input. This implies that the computation of the output time-series can be fully parallelized over time, which is a significant advantage for training and inference speed. Another consequence of this is that the acuracy of the model does not degrade over time due to error accumulation, which is a common issue with autoregressive models, but this comes at the cost of overlapping input windows. In addition, autoregressive models can often improve the accuracy at short lead-times.
- Initially, the output at a particular location may depend on the input at all input locations, but we aim to explore also models that have a more local receptive field, where the output at a particular location only depends on the input at nearby locations. This is a natural assumption for many physical phenomena, and it can also help to reduce the number of parameters and improve the generalization of the model, and can be used to further speed up the training and inference by exploiting the sparsity of the input-output dependencies.

## Notation

Consider a set of $N$ input stations with locations $(x_i, y_i)$ for $i = 1, \dots, N$, and a set of $\tilde{N}$ output stations with locations $(\tilde{x}_j, \tilde{y}_j)$ for $j = 1, \dots, \tilde{N}$. For each input location there are $P$ input variables (e.g. "wind_x", "wind_y", "surge", etc.), and for each output location there are $\tilde{P}$ output variables (e.g. "surge", "tide", etc.). The full input time-series thus has length $N*P$ at any time $t$, denoted as $\mathbf{x}_t \in \mathbb{R}^{N*P}$, and the full output time-series has length $\tilde{N}*\tilde{P}$ at any time $t$, denoted as $\mathbf{y}_t \in \mathbb{R}^{\tilde{N}*\tilde{P}}$. Each model is a function $f_\theta$ with parameters $\theta$, that maps the input time-series in the input window to the output at time $t$:
$$\mathbf{y}_t = f_\theta(\mathbf{x}_{t-l:t},t)$$
where $\mathbf{x}_{t-l:t}$ denotes the input time-series from time $t-l$ to time $t$, with $l$ being the length of the input window. Many models will not have an explicit dependence on time $t$, and for example tides models on the other end of the spectrum will have no other inputs than the time $t$.

For training, we have a dataset of input-output pairs $\{(\mathbf{x}_{t-l:t}, \mathbf{y}_t)\}_{t=1}^T$, where $\mathbf{y}_t$ is the target output at time $t$. The training objective is to find the parameters $\theta$ that minimize the loss function:
$$\mathbf{L}(\theta) = \frac{1}{T}\sum_{t=1}^T ||f_\theta(\mathbf{x}_{t-l:t},t) - \mathbf{y}_t||^2$$
where $||.||^2$ uses the mean squared error (MSE), but other loss functions can also be used depending on the application. The trained model can then be used for inference by applying it to new input time-series to predict the output at future times.

## Surge models

The surge models all use the same input variables (wind-stress and pressure) and the same output variable (surge), but they differ in the model architecture. 

### Linear surge model

The linear surge model is a simple linear regression on the lagged wind-stress and pressure forcing. The linear model can be denoted as:
$$
\mathbf{y}_t = W\,\mathbf{x}_{t-l:t} + \mathbf{b},\qquad
W \in \mathbb{R}^{\tilde{N}*\tilde{P} \times N*P*l},\quad \mathbf{b} \in \mathbb{R}^{\tilde{N}*\tilde{P}}
$$
with $\tilde{P}=1$ for surge. This corresponds to a single dense layer with identity
activation, applied independently for each time $t$.

### Time convolution surge model

The time convolution surge model applies 1-D convolutions over the lag dimension
of the input window. Let the lagged input be reshaped to a sequence
$X_t \in \mathbb{R}^{l \times (N*P)}$, where each lag index contains the concatenated
wind-stress and pressure features for all input stations (with $P=3$). The model
applies a stack of temporal convolutions with same padding:
$$
H^{(0)}_t = X_t,\qquad
H^{(k)}_t = \sigma\bigl(K^{(k)} * H^{(k-1)}_t + b^{(k)}\bigr),\quad k=1,\dots,K
$$
where $K^{(k)}$ is a 1-D convolution kernel over the lag axis and $\sigma$ is ReLU.
Same padding keeps the lag length $l$ fixed in every layer. The final features are
flattened and mapped to surge at output stations with a linear layer:
$$
\mathbf{y}_t = W\,\mathrm{vec}(H^{(K)}_t) + \mathbf{b},
$$
with $\mathbf{y}_t \in \mathbb{R}^{\tilde{N}}$ (since $\tilde{P}=1$ for surge).

### Attention surge model

The attention surge model combines a transformer-style branch network over the
lagged wind/pressure history with a trunk network over station metadata, then
merges them using a graph adjacency matrix. Let the wind/pressure input window
at time $t$ be reshaped to $X_t \in \mathbb{R}^{l \times (3N)}$ (lag length $l$,
three variables per wind station). Let the station encoding be
$S_t \in \mathbb{R}^{6 \times \tilde{N}}$ (cos/sin of lat, lon, and day-of-year).

The branch network $g_\theta$ applies embedding, positional encoding, and
transformer layers to produce per-wind features
$$
B_t = g_\theta(X_t) \in \mathbb{R}^{N \times (3l)}.
$$
The trunk network $h_\phi$ maps station encodings to attention weights
$$
T_t = h_\phi(S_t) \in \mathbb{R}^{\tilde{N} \times N}.
$$
With a fixed adjacency matrix $A \in \mathbb{R}^{\tilde{N} \times N}$, the
graph-weighted merge is
$$
M_t = (A \odot T_t)\,B_t \in \mathbb{R}^{\tilde{N} \times (3l)},
$$
which is then downsampled by a $1\times 1$ convolution (channel-mixing) to
produce $\tilde{N} \times l$ outputs. The prediction at time $t$ is the last lag:
$$
\mathbf{y}_t = \mathrm{last\_lag}\bigl(\mathrm{Conv1x1}(M_t)\bigr),\qquad
\mathbf{y}_t \in \mathbb{R}^{\tilde{N}}.
$$
As with the other surge models, $\tilde{P}=1$ for surge.

## Tide models

The tide models use only the time $t$ as input, and predict the tide at the output stations. Internally, the models use multiple input time-series of the form $\cos(\omega t)$ and$\sin(\omega t)$, where $\omega$ is the angular frequency of a tidal constituent. The model learns to combine these input time-series to predict the tide at each output station. The model architecture can be a simple linear combination of the input time-series, or it can be a more complex neural network that learns nonlinear interactions between the input time-series.

All tide share the same two step structure:
Step 1 (construct the forcing input). For each constituent frequency $\omega_i$,
create the time features at a single dummy input location:
$$
X_t = \bigl[\cos(\omega_1 t),\sin(\omega_1 t),\dots,\cos(\omega_F t),\sin(\omega_F t)\bigr]^\top.
$$
These are the Doodson-style astronomical forcing inputs, reused by all tide
models.
Steps 2 (model-specific). The model architecture then maps the input features to the predicted tide at the output stations. 

### DeepONet tide model

The DeepONet tide model uses a branch/trunk architecture inspired by DeepONets. The second step (branch/trunk merge, no lags). The branch network $g_\theta$ maps $X_t$ to
features $B_t \in \mathbb{R}^{r}$, and the trunk network $h_\phi$ maps station
coordinates $S \in \mathbb{R}^{2 \times \tilde{N}}$ (lat, lon) to
$T \in \mathbb{R}^{r \times \tilde{N}}$. As in the surge models, they are merged
per station via a dot-product and downsampled:
$$
\mathbf{y}_t = d_\psi\bigl(T^\top B_t\bigr) \in \mathbb{R}^{\tilde{N}}.
$$
This matches the branch/trunk/downsample DeepONet in `TideModel` and produces
one tide value per station at time $t$ (with $\tilde{P}=1$).

## Wave models

Wave models predict significant wave height ($\tilde{P}=1$) at $\tilde{N}$ output
stations from wind speed and direction at $N$ input stations over a lagged input
window of length $l$.

### Input preparation

Wind speed $u_{10}$ and meteorological direction $\phi$ are first converted to
quadratic wind-stress components via the drag-coefficient relation:
$$
(\tau_x, \tau_y) = C_d\,u_{10}\,(-\sin\phi,\,-\cos\phi),
\qquad C_d \approx \rho_a c_d u_{10},
$$
then divided by a fixed scale factor $s_w$ to bring values into a unit-friendly
range.  The lagged input at time $t$ stacks these stress components for all $N$
input stations across lags $t-l+1, \dots, t$:
$$
X_t \in \mathbb{R}^{l \times 2N}.
$$
The target wave height is similarly scaled by a factor $s_h$ during training and
unscaled in the final output.

### Station encoding

Both wave model architectures encode each output station $j$ as a one-hot vector
$\mathbf{e}_j \in \{0,1\}^{\tilde{N}}$.  This means each training sample is an
independent (station, time) pair $(\mathbf{e}_j, X_t)$, so all $\tilde{N}$ output
stations are processed by independent forward passes that share the wind-branch
weights but have station-specific parameters.

### ConvWaveModel

`ConvWaveModel` processes the wind input through a custom `WaveInputLayer` that
combines a 1-D convolutional branch with a station-modulation branch, followed by
a stack of strided convolutions that halve the lag dimension at each step.

The `WaveInputLayer` applies a point-wise convolution (kernel size 1) to the lag
sequence to produce $c$ feature channels:
$$
C_t = \sigma\bigl(\mathrm{Conv}_{1}(X_t)\bigr) \in \mathbb{R}^{l \times c}.
$$
In parallel, the one-hot station vector $\mathbf{e}_j$ is projected to a
per-station sensitivity matrix:
$$
S_j = \mathrm{Dense}(\mathbf{e}_j) \in \mathbb{R}^{l \times c}
$$
(reshaped from a flat vector).  The two branches are merged element-wise with an
exponential gate, giving each station its own channel-wise sensitivity profile:
$$
H^{(0)}_{t,j} = \exp(S_j) \odot C_t \in \mathbb{R}^{l \times c}.
$$
A stack of $K$ strided convolutions (stride 2, so $l = 2^K$) then collapses the
lag dimension:
$$
H^{(k)}_{t,j} = \sigma\bigl(\mathrm{Conv}_2(H^{(k-1)}_{t,j})\bigr),\quad k=1,\dots,K,
$$
until $H^{(K)}_{t,j} \in \mathbb{R}^{1 \times 1}$.  Flattening gives the scalar
prediction $\hat{y}_{t,j} = H^{(K)}_{t,j}$.

### DeepONetWaveModel

`DeepONetWaveModel` uses a simpler dot-product merge, closer in spirit to the
DeepONet architecture used for tides.

A branch network $g_\theta$ consisting of $K$ strided convolutions (stride 2, so
$l = 2^K$) and a flatten operation maps the wind input directly to a feature vector:
$$
B_t = g_\theta(X_t) \in \mathbb{R}^{r},
$$
where the first convolution maps from $2N$ input channels.  In parallel, the
one-hot station vector is projected by a learned linear map:
$$
S_j = W_s\,\mathbf{e}_j \in \mathbb{R}^{r},\qquad W_s \in \mathbb{R}^{r \times \tilde{N}}.
$$
The prediction is the dot product of the two feature vectors:
$$
\hat{y}_{t,j} = S_j \cdot B_t = \sum_{k=1}^r (S_j)_k\,(B_t)_k.
$$

**Comparison.**  Both architectures share the one-hot station encoding and the
strided-convolution branch for wind history.  The key difference is in the merge:
`ConvWaveModel` applies an exponential channel gate before the strided convolutions
(so the station influences how the wind features are weighted at each lag and
channel), whereas `DeepONetWaveModel` applies a linear dot product after the branch
network has already collapsed the lag dimension.  The dot-product merge is simpler
and has fewer parameters in the station branch, but it gives the station less
control over the temporal processing of the wind input.

