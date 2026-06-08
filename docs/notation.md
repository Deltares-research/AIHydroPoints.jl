
# Notation 

The purpose of this document is to clarify the notation used in the documentation and codebase. We aim to use an internally consistent notation, but this unfortunately means that it is not everywhere consistent with the notation used in the literature. We hope that this document will help to clarify any confusion that may arise from this.

## Small example

Let's start with a small example of a single dense layer with a ReLU activation function. The input to the layer is a vector $\mathbf{x} \in \mathbb{R}^{N_i}$, and the output is a vector $\mathbf{y} \in \mathbb{R}^{N_I}$. The weights of the layer are represented by a matrix $\mathbf{W} \in \mathbb{R}^{N_I \times N_i}$, and the bias is represented by a vector $\mathbf{b} \in \mathbb{R}^{N_I}$. The output of the layer can be computed as follows:

$$\mathbf{y} = \text{ReLU}(\mathbf{W} \mathbf{x} + \mathbf{b})$$

which is shorthand for:

$$[\mathbf{y}]_I = \text{ReLU}\left(\sum_{i=1}^{N_i} [\mathbf{W}]_{Ii} [\mathbf{x}]_i + [\mathbf{b}]_I \right) \quad \text{for } I = 1, \ldots, N_I$$

In Julia with Flux this would be implemented as:

```julia
using Flux
x = randn(N_i)
model = Dense(N_i => N_I, relu)  # y = relu.(W * x .+ bias)
y = model(x)
```

In PyTorch this would be:

```python
import torch
import torch.nn as nn
x = torch.randn(N_i)
model = nn.Sequential(nn.Linear(N_i, N_I), nn.ReLU())  # y = relu(W @ x + bias)
y = model(x)
``` 

As part of our notation, we will use the einstein summation convention, i.e. there is an implied summation over repeated indices, which allows us to write the above equation more compactly as:

$$[\mathbf{y}]_I = \text{ReLU}([\mathbf{W}]_{Ii} [\mathbf{x}]_i + [\mathbf{b}]_I)$$

We'll add a few more details to this notation to arrive at:

$$\mathbf{y}_I = \sigma(\mathbf{W}_{Ii} \star \mathbf{x}_i + \mathbf{b}_I) $$

where $\sigma$ is the activation function, and $\star$ denotes "apply the weights": the exact operation (dense contraction, convolution, pass-through, …) is determined per direction by the index pattern of $\mathbf{W}$ — see the rules in the convolutional-layer section. We also use capital letters for the output indices and lowercase letters for the input indices, and we drop the brackets around the indices for brevity. We'll use brackets when we want to make the order of the indices explicit, but otherwise we'll use the names of the indices instead of their positions in the tensor. So, eg $\mathbf{W}_{Ii}$ is the same as $\mathbf{W}_{iI}$, but $[\mathbf{W}]_{Ii}$ is the transpose of $[\mathbf{W}]_{iI}$.

## Multi-dimensional data

In the case of multi-dimensional data, we will use the same notation as above, but we will also include additional indices to represent the dimensions of the data. For example a dense layer with inputs on a grid with $x$, $y$ and $z$ dimensions would be written as:

$$\mathbf{y}_{XYZ} = \sigma(\mathbf{W}_{XYZxyz} \star \mathbf{x}_{xyz} + \mathbf{b}_{XYZ}) $$

The output indices are still represented by capital letters, and the input indices are still represented by lowercase letters. The "apply weights" operation is still denoted by $\star$, and the activation function is still denoted by $\sigma$. Since $x$, $y$ and $z$ are all repeated indices (lowercase on $\mathbf{x}$, appearing in $\mathbf{W}$), the contraction is implied over all three dimensions. We use this notation to preserve the multidimensional structure of the data, and to make it clear which indices correspond to which dimensions. This is especially important when we have multiple layers in a model, and we want to keep track of how the data flows through the layers.

To express the meaning of the indices we'll use the following convention:

| Index (input) | Index (output) | Meaning |
|---|---|---|
| $p$ | $P$ | point (spatial location) |
| $t$ | $T$ | time |
| $q$ | $Q$ | quantity (e.g. temperature, pressure) |
| $c$ | $C$ | channel (feature) |
| $b$ | $B$ | batch|
| $i$ | $I$ | generic input/output index |
| $x$ | $X$ | x-dimension |
| $y$ | $Y$ | y-dimension |
| $z$ | $Z$ | z-dimension |

In addition, we will use $b$ for the batch index and $l$ index as a superscript for the layer number when we want to make it explicit. 

Ordering of the indices is not important in our notation, but is important when implementing the model in code. Flux (column-major) and PyTorch (row-major) use opposite conventions. The convention we use in our notation is the same as Flux, but we will also provide the corresponding PyTorch notation for reference.

### Intermezzo: memory layout

Consider waterlevel $h_{xt}$ along a river at 3 locations and 2 time steps:

| | $t=1$ | $t=2$ |
|---|---|---|
| $x=1$ | 1.0 | 1.1 |
| $x=2$ | 1.5 | 1.6 |
| $x=3$ | 2.0 | 2.1 |

In memory, arrays are stored as a flat sequence of numbers. Both Flux and PyTorch generally make the same performance choices: keep the spatial/feature values for a single time step contiguous, so that processing one sample at a time is fast. This means the memory sequence is the same in both:

$$1.0,\ 1.5,\ 2.0,\ 1.1,\ 1.6,\ 2.1 \quad \text{(all locations at } t=1 \text{, then all at } t=2\text{)}$$

The difference is only in how the two languages *describe* this layout:

- **Julia (column-major):** the first index varies fastest, so spatial dimension $x$ comes first: shape `[x, t]` = `[3, 2]`.
- **PyTorch (row-major):** the last index varies fastest, so spatial dimension $x$ comes last: shape `(t, x)` = `(2, 3)`.

```julia
# Julia / Flux
h = [1.0 1.1;
     1.5 1.6;
     2.0 2.1]   # shape [x, t] = [3, 2]
h[2, 1]         # → 1.5  (x=2, t=1)
```

```python
# PyTorch
import torch
h = torch.tensor([[1.0, 1.5, 2.0],
                  [1.1, 1.6, 2.1]])  # shape (t, x) = (2, 3)
h[0, 1]                              # → 1.5  (t=1, x=2)
```

The bytes in memory are identical — **only the index notation differs**. This is why the convention table above looks like a reversal: `[X, C, B]` in Flux and `[B, C, X]` in PyTorch describe the same memory layout.

| Layer type | Flux (column-major) | PyTorch (row-major) |
|---|---|---|
| Dense | $[I, B]$ | $[B, I]$ |
| 1D Conv | $[X, C, B]$ | $[B, C, X]$ |
| 2D Conv | $[X, Y, C, B]$ | $[B, C, Y, X]$ |
    
## Dense layers for multi-dimensional data

When we have multi-dimensional data, we can still use dense layers, but we need to be careful about how we apply them. A dense layer is a linear transformation that maps an input vector to an output vector. When we have multi-dimensional data, we have just seen that this maps to a one dimensional vector in memory, but we want to preserve the multi-dimensional structure of the data in our notation. To do this, we can use the same notation as above, but we will also include additional indices to represent the dimensions of the data. For example a dense layer with inputs on a grid with $x$, $y$ and $z$ dimensions would be written as:

$$\mathbf{y}_{XYZ} = \sigma(\mathbf{W}_{XYZxyz} \star \mathbf{x}_{xyz} + \mathbf{b}_{XYZ}) $$

where in for implementation we would rehape the input and output to be one-dimensional vectors. We can make this more explicit using the bracket notation by writing:

$$[\mathbf{y}]_{(XYZ)} = \sigma([\mathbf{W}]_{(XYZ)(xyz)} \star [\mathbf{x}]_{(xyz)} + \mathbf{b}_{I}) $$

The $[]$ breackets indicate that we are making the memory layout explicit (i.e. column-major), and the $()$ parentheses indicate the logical grouping of indices for the operation. In implementation this would look like:

```julia
#one of 
x_vec = reshape(x, prod(size(x)))  # flatten to 1D vector
x_vec = flatten(x)  # from Flux
x_vec = x[:] # 
```

Python/Pytorch has similar functions for flattening tensors.

## Convolutional layers

A convolutional layer does not consider all combinations of input and output indices, but only a subset of them. For example, a 1D convolutional layer with a kernel of size $N_{\Delta i}$ only considers $N_{\Delta i}$ adjacent points in the input for each output point. For generic indices $i \to I$ this can be written explicitly as:

$$[\mathbf{y}]_I = \sigma\!\left(\sum_{\Delta i=1}^{N_{\Delta i}} [\mathbf{W}]_{\Delta i} \cdot [\mathbf{x}]_{I+\Delta i-1} + b\right) \quad \text{for } I = 1, \ldots, N_I$$

where the output size is $N_I = N_i - N_{\Delta i} + 1$ (no padding).

Note that the computation for each element of the output re-uses the same weights $\mathbf{W}_{\Delta i}$, which greatly reduces the number of parameters. On the other hand this implies the assumption that this models the data well. In implementation this would look like:

```julia
# Julia / Flux
model = Conv((N_Δi,), 1 => 1, relu)  # kernel size N_Δi, 1 input channel, 1 output channel
y = model(x)
```

To express this compactly in our index notation, we write:

$$\mathbf{y}_I = \sigma(\mathbf{W}_{\Delta i} \star \mathbf{x}_{i} + b)$$

The kernel $\mathbf{W}_{\Delta i}$ is indexed by the **kernel offset** $\Delta i$ (range $1, \ldots, N_{\Delta i}$), not by the full input position $i$ (range $1, \ldots, N_i$). The $\Delta i$ on $\mathbf{W}$ tells us that $\star$ is a convolution along $i$: for each output position $I$, we sum over the kernel offset $\Delta i$ against the shifted input values $\mathbf{x}_{I+\Delta i - 1}$.

The key observation is that the weight tensor $\mathbf{W}$ has **fewer or different indices than the corresponding dense layer would have**: a dense layer would have $\mathbf{W}_{Ii}$ (both output index $I$ and input index $i$), but the convolutional version drops $I$ (weights are shared across output positions) and replaces $i$ with the kernel offset $\Delta i$.

### Notation for missing and convolved indices

When working with multi-dimensional data it helps to read off the structure of a layer directly from the index pattern of $\mathbf{W}$. Comparing to the dense reference $\mathbf{W}_{IJij}$, each index position in $\mathbf{W}$ tells us how the layer treats that direction:

- An explicit capital $I$ → **dense** over that output dimension.
- A lowercase $i$ that also appears on $\mathbf{x}$ → **dense contraction** along $i$ (summation).
- A delta-offset $\Delta i$ → **convolution** along the $i$ direction (kernel offset).
- The index *absent* from $\mathbf{W}$ → **pass-through** (input position broadcast to its capital counterpart).

Because these rules fully determine the operation, we write $\star$ without subscripts. When discussing a specific direction in prose, we may still write e.g. $\star_{i \to I}$ for emphasis.

For a 2D convolutional layer with a kernel of size 3 in both dimensions, both directions are convolved:

$$\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{\Delta i\,\Delta j} \star \mathbf{x}_{ij} + b) $$

A convolution over the $i$ direction and a dense map over the $j$ direction keeps $J$ explicit (dense) and uses $\Delta i$ for the kernel offset:

$$\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{J\,\Delta i\,j} \star \mathbf{x}_{ij} + b_J) $$

A third example applies a 1D convolution along $j$ independently for each $i$ — the same kernel is re-used at every $i$ row, and the $i$ direction is passed through unchanged:

$$\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{\Delta j} \star \mathbf{x}_{ij} + b) $$

The kernel depends only on $\Delta j$. The $i$ index in $\mathbf{x}$ is neither contracted nor present in $\mathbf{W}$, so by the **pass-through rule** it is broadcast to its capital counterpart $I$ in $\mathbf{y}$: every input row $i$ produces output row $I = i$ with the same kernel applied.

> **Pass-through rule.** An input index in $\mathbf{x}$ that does not appear in $\mathbf{W}$ and is not contracted is broadcast to its capital counterpart in $\mathbf{y}$ ($i \to I$, $j \to J$, etc.).

These are three different layers, distinguishable at a glance from the index pattern of $\mathbf{W}$.

### Summary

| Description | Formula | # weights in $\mathbf{W}$ | Flux |
|---|---|---|---|
| 2D conv (3×3 kernel) over $i$ and $j$ | $\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{\Delta i\,\Delta j} \star \mathbf{x}_{ij} + b)$ | $N_{\Delta i} N_{\Delta j}$ | `Conv((3, 3), 1 => 1, relu)` |
| 1D conv (size 3) over $i$, dense over $j$ | $\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{J\,\Delta i\,j} \star \mathbf{x}_{ij} + b_J)$ | $N_J N_{\Delta i} N_j$ | `Conv((3,), N_j => N_J, relu)` (treating $j$ as channel) |
| 1D conv (size 3) over $j$, shared for each $i$ | $\mathbf{y}_{IJ} = \sigma(\mathbf{W}_{\Delta j} \star \mathbf{x}_{ij} + b)$ | $N_{\Delta j}$ | `Conv((1, 3), 1 => 1, relu)` |

For reference, a fully dense layer $\mathbf{W}_{IJij}$ would have $N_I N_J N_i N_j$ weights — every conv variant above is a strict subset obtained by dropping or replacing indices, which is exactly the parameter sharing that makes convolutions efficient.

In the second row, $j$ is interpreted as the channel dimension so a 1D `Conv` with $N_j$ input and $N_J$ output channels achieves the dense map along $j$. In the third row, a 2D `Conv` with a width-1 kernel along $i$ reuses the same weights at every $i$ position.

## Towards a more realistic example

In practice the input tensor is 4-dimensional with axes for point, quantity, time-lag, and batch-time — though point and quantity are merged into one dimension at the tensor level so that 1D Conv treats them as channels.




