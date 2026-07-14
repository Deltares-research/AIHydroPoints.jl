
# Background and motivation

NOTE: This doc is about the concepts and math, not the code structure or implementation details.

> **Draft warning.** This document is still a very rough draft and should be
> read with care. The conceptual notation and the actual tensor layouts used
> in the code are not yet fully consistent — see
> [`notes_dimensions.md`](notes_dimensions.md) for an ongoing review of those
> inconsistencies and the planned refactor. The index-notation conventions
> used below are spelled out in the [Notation](notation.md) appendix.
> Everything before the "## Old background" is the new background, which is incomplete and under development.

## Introduction

This describes the mathematical and conceptual background for the AIHydroPoints.jl package. The package is designed to explore the application of machine learning models to predict water levels, tides, waves, and more at a network of coastal stations. The package is designed to be flexible and extensible, allowing for the incorporation of additional phenomena and model architectures in the future.

The notation used in this document is based on the index notation for tensors, which is a concise and expressive way to describe the structure of the data and the operations performed by the models. The notation is explained in detail in the [Notation](notation.md) appendix.

## Time-lag models

All the models in the package are time-lag models, which means that to compute the output at time $t$, the model takes as input a window of past input data from time $t-l$ to time $t$, where $l$ is the length of the input window. The models are **causal**, meaning that the prediction at time $t$ can only depend on the input data up to time $t$. The models are **not autoregressive**, which means that they do not use any values of the output time-series (at any time) as input. This allows for parallel computation of the output time-series over time, which is a significant advantage for training and inference speed. The generic form of this type of model is described at the end of the [Notation](notation.md) appendix.

In this package a single call to the core model, will compute the output at a single time $t$, given the input window from $t-l$ to $t$. To compute the output time-series over a range of times, inputs can be stacked into a batch dimension, and the model can be called in parallel over the batch dimension. This implies that the inputs will probably contain overlapping time windows, but this is not a problem as long as everything fits in memory. Still, it's good to be aware that the model preprocess function will increase the memory footprint by a factor of $l$ when preparing the input data for a range of times.

## Surge models

The surge models all use the same input variables (wind-stress and pressure) and all output one variable (surge). In more detail, the input time-series should be provided for one of:
- Wind-stress components "stress_x" and "stress_y" and "pressure" at $N_p$ input stations and $N_t$ time steps.
- Wind speed components "wind_x" and "wind_y" and "pressure" at $N_p$ input stations and $N_t$ time steps. The wind speed components are converted to wind-stress components internally in the model, using the Charnock drag coefficient relation. The output time-series is the surge at $\tilde{N}_p$ output stations and $N_t$ time steps.

Input time-series are provided as a dictionary of MultiTimeSeries, with keys "stress_x", "stress_y", and "pressure" (or "wind_x", "wind_y", and "pressure"). The output time-series is a single MultiTimeSeries with key "surge".

### Linear surge model


The linear surge model is a simple linear regression on the lagged wind-stress and pressure forcing. The linear model for a single output time can be denoted as:

$$\mathbf{y}_{PT} = (\mathbf{W}_{Ppql} \star \mathbf{x}_{pqlt} + \mathbf{b}_P)$$

where $l$ is the time lag dimension, $p$ is the input station dimension, $q$ is the input variable dimension (wind-stress x, wind-stress y, pressure), and $P$ is the output station dimension. The model can be applied independently for each output time $t$, using the lagged input window $\mathbf{x}_{pql}$ from time $t-l$ to time $t$.  

The model only imposes a linear relationship between the input and output, but there are no other constraints on the model parameters. The model could be trained using standard linear regression techniques, but here we use a neural network framework to keep the training and inference code consistent with the other models. The linear model can be implemented as a single dense layer with identity activation, using the output time as the batch dimension.

## Old background

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

