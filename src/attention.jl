using Flux

############
# Embeddings
############

struct Embedder{T}
    layer::T
end


"""
    Embedder(npars, nembed)

Embedder to move sequence of inputs into embedding space

# Arguments

- `npars`: size of single input vector / token
- `nembed`: size of embedded vector
"""
function Embedder(npars, nembed)
    return Embedder(
        Dense(npars=>nembed, identity; bias=false, init=Flux.orthogonal)
    )
end

function (l::Embedder)(x)
    return l.layer(x)
end

@Flux.layer Embedder

struct Deembedder{T}
    layer::T
end

"""
    Deembedder(l::Embedder)

Deembder to move embedded vectors back to original data space.
It is not trainable in itself, but shares weights with Embedder used as input.

# Arguments

- `l::Embedder`: Embedder to provide deembeding for
"""
function Deembedder(l::Embedder)
    return Deembedder(
        Dense(transpose(l.layer.weight), false, identity)
    )
end

function (l::Deembedder)(x)
    return l.layer(x)
end

# Make into Flux layer for some nice utilities, switch all params to non-trainable
Flux.@layer Deembedder
Flux.trainable(m::Deembedder) = ()

struct SinCosPosEmbedder{T}
    weights::AbstractArray{T,2}
end

"""
    SinCosPosEmbedder(nembed, nlags; kwargs...)

Sin - Cos position embedding

# Arguments

- `nembed`: Size of token to embed
- `nlags`: Length of token sequence

# Keywords

- `theta`: Frequency used in position embedding
    (**Default**: `10000.`)
"""
function SinCosPosEmbedder(nembed, nlags; theta=10000.)
    thetas = theta .^ (-2 .* (1:nembed÷2)./ nembed)
    pos = 1:nlags
    even_pos = sin.(pos*thetas')
    odd_pos = cos.(pos*thetas')
    return SinCosPosEmbedder(
        Float32.(reshape(vcat((odd_pos, even_pos)...), nlags, :))'
    )
end

function (m::SinCosPosEmbedder)(x)
    return x .+ m.weights
end

# Make into Flux layer for some nice utilities, switch all params to non-trainable
Flux.@layer SinCosPosEmbedder
Flux.trainable(m::SinCosPosEmbedder) = ()

#########################
# Attention & Transformer
#########################

# From Metalhead.jl
struct MultiHeadSelfAttention{P, Q, R}
    nheads::Int
    qkv_layer::P
    attn_drop::Q
    projection::R
end

"""
    MultiHeadSelfAttention(planes::Integer, nheads::Integer = 1; kwargs...)

Multihead self attention layer

# Arguments

- `planes::Integer`: Size of input tokens
- `nheads::Integer`: Number of attention heads
    (**Default**: `1`)

# Keywords

- `qkv_bias::Bool`: Use bias in Dense layer making qkv matrices
    (**Default**: `false`)
- `attn_dropout_prob`: Dropout probability in attention mechanism
    (**Default**: `0.0`)
- `proj_dropout_prob`: Dropout probability in projection layer
    (**Default**: `0.0`)
"""
function MultiHeadSelfAttention(planes::Integer, nheads::Integer = 1;
    qkv_bias::Bool = false, attn_dropout_prob = 0.0, proj_dropout_prob = 0.0)
    @assert planes % nheads == 0 "planes should be divisible by nheads"
    qkv_layer = Dense(planes, planes*3; bias=qkv_bias)
    attn_drop = Dropout(attn_dropout_prob)
    proj = Chain(Dense(planes, planes), Dropout(proj_dropout_prob))
    return MultiHeadSelfAttention(nheads, qkv_layer, attn_drop, proj)
end

function (m::MultiHeadSelfAttention)(x::AbstractArray{<:Number,3})
    qkv = m.qkv_layer(x)
    q, k, v = Flux.chunk(qkv, 3, dims=1)
    y, _ = NNlib.dot_product_attention(q, k, v; nheads=m.nheads, fdrop=m.attn_drop)
    y = m.projection(y)
    return y
end

Flux.@layer MultiHeadSelfAttention

struct Transformer{P, Q, R}
    norm1::P
    norm2::P
    attn_layer::Q
    mlp::R
end

function Transformer(nembed, nheads)
    norm1 = LayerNorm(nembed)
    norm2 = LayerNorm(nembed)
    attn_layer = MultiHeadSelfAttention(nembed, nheads)
    mlp = Dense(nembed, nembed)
    return Transformer(norm1, norm2, attn_layer, mlp)
end

function (t::Transformer)(x)
    return Chain(
        SkipConnection(
            Chain([t.norm1, t.attn_layer]...), .+
        ),
        SkipConnection(Chain([t.norm2, t.mlp]...), .+)
    )(x)
end

Flux.@layer Transformer

