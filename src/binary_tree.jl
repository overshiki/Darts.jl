using Flux
import Flux.params 
import Flux.params!
using CUDA
import Functors.@functor

const cArray = Union{Array, CuArray}

abstract type ReduceOperation end 

struct NNContainer 
    layer::Any
end
@functor NNContainer
params(a::NNContainer) = params(a.layer)

mutable struct ProbContainer{T<:AbstractArray} 
    values::T
end
@functor ProbContainer
# params(a::ProbContainer) = Params(a.values)

mutable struct Ensemble 
    # probs::Vector{Float32}
    probs::ProbContainer
    nn_ops::Vector{NNContainer}
end
@functor Ensemble
function params(a::Ensemble)
    # p = Params()
    # iters = map(a.nn_ops) do x; return params(x); end
    # union!(p, iters...)
    # union!(p, params(a.probs))

    # params!(p, a.nn_ops...)
    p = params(a.nn_ops...)
    # params!(p, a.probs)
    return p
end

function hparams(a::Ensemble)
    p = Params()
    # p = params(a.probs)
    params!(p, a.probs)
    return p
end

function effect_ensemble(inchannel::Int)
    nn_ops = [
        BatchNorm(inchannel),
        # Dropout(0.5),
        InstanceNorm(inchannel),
    ]

    nn_ops = map(nn_ops) do x 
        return x |> NNContainer
    end

    en = Ensemble(ProbContainer(rand(length(nn_ops))), nn_ops)
    # normalize!(en)
    return en
end

function conv_ensemble(inchannel::Int, outchannel::Int)
    nn_ops = [
        Conv((3, ), inchannel=>outchannel; pad=1),
        Conv((5, ), inchannel=>outchannel; pad=2),
        Conv((7, ), inchannel=>outchannel; pad=3),
        Conv((9, ), inchannel=>outchannel; pad=4),

        ConvTranspose((3, ), inchannel=>outchannel; pad=1),
        ConvTranspose((5, ), inchannel=>outchannel; pad=2),
        ConvTranspose((7, ), inchannel=>outchannel; pad=3),
        ConvTranspose((9, ), inchannel=>outchannel; pad=4),

    ]

    nn_ops = map(nn_ops) do x 
        return x |> NNContainer
    end

    en = Ensemble(ProbContainer(rand(length(nn_ops))), nn_ops)
    # normalize!(en)
    return en
end

function activation_ensemble()
    nn_ops = [
        gelu,
        hardsigmoid,
        hardtanh,
        lisht,
        mish,
        relu
    ]

    nn_ops = map(nn_ops) do fun 
        rtfun = x->fun.(x)
        return rtfun |> NNContainer
    end

    en = Ensemble(ProbContainer(rand(length(nn_ops))), nn_ops)
    # normalize!(en)
    return en
end


function pool_ensemble(out::NTuple)
    nn_ops = [
        AdaptiveMaxPool(out),
        AdaptiveMeanPool(out),
    ]

    nn_ops = map(nn_ops) do x 
        return x |> NNContainer
    end

    en = Ensemble(ProbContainer(rand(length(nn_ops))), nn_ops)
    # normalize!(en)
    return en
end



function normalize!(en::Ensemble)
    Z = sum(en.probs.values)
    en.probs = ProbContainer(en.probs.values ./ Z) 
end

import CUDA.cu
function cu(en::Ensemble)
    return fmap(cu, en)
end




function average_infer(en::Ensemble, x::cArray)

    y = map(en.nn_ops) do nnc 
        rx = nnc.layer(x)
        return reshape(rx, (1, size(rx)...))
    end

    y = vcat(y...)
    coeffs = exp.(en.probs.values)
    coeffs = coeffs ./ sum(coeffs)

    y = deepcopy(y) .* coeffs
    y = sum(y, dims=1)
    y = reshape(y, (size(y)[2:end]...))
    return y
end




mutable struct Chain_ensemble
    ens::Vector{Ensemble}
end
@functor Chain_ensemble

function params(cen::Chain_ensemble)
    p = Params()
    iters = map(cen.ens) do x; return params(x); end
    union!(p, iters...)
    return p
end

function hparams(cen::Chain_ensemble)
    p = Params()
    iters = map(cen.ens) do x; return hparams(x); end
    union!(p, iters...)
    return p
end


const uEnsemble = Union{Chain_ensemble, Ensemble}

function average_infer(cen::Chain_ensemble, x::cArray)
    for en in cen.ens 
        x = average_infer(en, x)
    end 
    return x
end

struct sum_op <: ReduceOperation
    fun::Function
end
sum_op() = sum_op((x,y) -> x .+ y)
(f::sum_op)(x, y) = f.fun(x, y)

struct product_op <: ReduceOperation
    fun::Function
end
product_op() = product_op((x,y) -> x .* y)
(f::product_op)(x, y) = f.fun(x, y)


mutable struct Cell
    left_ensemble::uEnsemble
    right_ensemble::uEnsemble
    op::ReduceOperation
    left_id::Symbol
    right_id::Symbol
    out_id::Symbol
end
@functor Cell
function params(a::Cell)
    p = Params()
    lp = params(a.left_ensemble)
    rp = params(a.right_ensemble)
    union!(p, lp, rp)
    return p
end

function hparams(a::Cell)
    p = Params()
    lp = hparams(a.left_ensemble)
    rp = hparams(a.right_ensemble)
    union!(p, lp, rp)
    return p
end


struct Graph 
    cells::Vector{Cell}
    data::Dict
end
@functor Graph
function params(a::Graph)
    p = Params()
    iters = map(a.cells) do x; return params(x); end
    union!(p, iters...)
    return p
end

function hparams(a::Graph)
    p = Params()
    iters = map(a.cells) do x; return hparams(x); end
    union!(p, iters...)
    return p
end


function average_infer!(g::Graph)
    for cell in g.cells 
        # @show cell.left_id, cell.right_id, cell.out_id
        left_d = average_infer(cell.left_ensemble, g.data[cell.left_id])
        right_d = average_infer(cell.right_ensemble, g.data[cell.right_id])
        out = cell.op(left_d, right_d)
        g.data[cell.out_id] = out
    end
end

import Base.&
function (&)(a::Vector{T}, b::Vector{T}) where {T}
    nvec = T[]
    append!(nvec, a)
    append!(nvec, b)
    return nvec
end


function gen_conv_ensemble(inchannel, outchannel, insize, outsize)
    evec = Ensemble[]
    push!(evec, conv_ensemble(inchannel, outchannel))
    if insize!=outsize
        push!(evec, pool_ensemble((outsize, )))
    end
    return Chain_ensemble(evec)
end 

function gen_activation_ensemble(inchannel, outchannel, insize, outsize)
    evec = Ensemble[]
    push!(evec, activation_ensemble())
    if inchannel != outchannel
        push!(evec, conv_ensemble(inchannel, outchannel))
    end

    if insize!=outsize
        push!(evec, pool_ensemble((outsize, )))
    end
    return Chain_ensemble(evec)
end

function gen_pool_ensemble(inchannel, outchannel, insize, outsize)
    evec = Ensemble[]
    push!(evec, pool_ensemble((outsize, )))
    if inchannel!=outchannel
        push!(evec, conv_ensemble(inchannel, outchannel))
    end
    return Chain_ensemble(evec)
end

function gen_effect_ensemble(inchannel, outchannel, insize, outsize)
    evec = Ensemble[]
    if inchannel!=outchannel
        push!(evec, conv_ensemble(inchannel, outchannel))
    end
    push!(evec, effect_ensemble(outchannel))

    if insize!=outsize
        push!(evec, pool_ensemble((outsize, )))
    end

    return Chain_ensemble(evec)
end



# using Distributions
function random_graph(input_symbol, output_symbol, input_data; depth=5, inchannel=1, insize=10, outsize=2)
    symbol_pool = [output_symbol] & [gensym() for _ in 1:depth] 

    """prepare en_generator and ops"""
    en_generator = [
        gen_conv_ensemble,
        gen_activation_ensemble,
        gen_pool_ensemble,
        gen_effect_ensemble
    ]

    ops = [sum_op(), product_op()]

    input_channel_info = Dict(input_symbol=>inchannel)
    input_size_info = Dict(input_symbol=>insize)


    """input"""
    out_channel = rand(inchannel:inchannel*10)
    en_left = rand(en_generator)(inchannel, out_channel, insize, insize)
    en_right = rand(en_generator)(inchannel, out_channel, insize, insize)
    op = rand(ops)
    out_id = rand(symbol_pool)
    # out_id = gensym()

    input_channel_info[out_id] = out_channel
    input_size_info[out_id] = insize

    cell = Cell(en_left, en_right, op, input_symbol, input_symbol, out_id)
    cells = [cell]

    # @show input_symbol, out_id
    t_outsize = insize

    """intermediate"""
    for i in 1:depth
        sym_left = rand(keys(input_channel_info))
        sym_right = rand(keys(input_channel_info))
        out_id = rand(symbol_pool)
        # out_id = gensym()

        t_outsize = rand(outsize:t_outsize)
        l_insize = input_size_info[sym_left]
        r_insize = input_size_info[sym_right]
        
        _inchannel_left = input_channel_info[sym_left]
        _inchannel_right = input_channel_info[sym_right]

        out_channel = rand(inchannel:inchannel*10)

        # @show sym_left, sym_right, out_id, _inchannel_left, _inchannel_right, out_channel

        en_left = rand(en_generator)(_inchannel_left, out_channel, l_insize, t_outsize)
        en_right = rand(en_generator)(_inchannel_right, out_channel, r_insize, t_outsize)
        op = rand(ops)

        cell = Cell(en_left, en_right, op, sym_left, sym_right, out_id)

        push!(cells, cell)
        # push!(in_symbol_pool, out_id)
        input_channel_info[out_id] = out_channel
        input_size_info[out_id] = t_outsize
    end

    """output"""
    sym_left = rand(keys(input_channel_info))
    sym_right = rand(keys(input_channel_info))

    l_insize = input_size_info[sym_left]
    r_insize = input_size_info[sym_right]

    _inchannel_left = input_channel_info[sym_left]
    _inchannel_right = input_channel_info[sym_right]

    out_channel = 1
    en_left = rand(en_generator)(_inchannel_left, out_channel, l_insize, outsize)
    en_right = rand(en_generator)(_inchannel_right, out_channel, r_insize, outsize)
    op = rand(ops)
    out_id = output_symbol

    cell = Cell(en_left, en_right, op, sym_left, sym_right, out_id)
    push!(cells, cell)

    data = Dict(input_symbol=>input_data)
    g = Graph(cells, data)
    return g
end

"""
x = rand(Float32, 10, 1, 20) |> cu

g = random_graph(:in, :out, x; depth=15, inchannel=1, insize=10)
average_infer!(g)
@show size(g.data[:out])
"""





"""
# en1 = conv_ensemble(1, 20) |> cu
# en2 = activation_ensemble() |> cu

en1 = gen_conv_ensemble(1, 20, 10, 10) |> cu
en2 = gen_activation_ensemble(1, 20, 10, 10) |> cu

cell = Cell(en1, en2, sum_op(), :l1, :r1, :o1)

en3 = pool_ensemble((5, )) |> cu

xl1 = rand(Float32, 10, 1, 20) |> cu
xr1 = rand(Float32, 10, 1, 20) |> cu
data = Dict(:l1=>xl1, :r1=>xr1)
g = Graph([cell], data)

average_infer!(g)
@show size(g.data[:o1])

"""

# y = average_infer(en1, x)
# z = average_infer(en2, y)
# z2 = average_infer(en3, z)
# @show size(y), size(z), typeof(y), typeof(z)
# @show size(z2)

# println()