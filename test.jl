include("./src/binary_tree.jl")
include("./src/helper.jl")
import Flux.Params
import Functors.functor
using CUDA

function test_params()
    A = NNContainer(Conv((3, ), 1=>10; pad=1))
    # layer = fmap(cu, A.layer)
    # A = NNContainer(layer)
    A = fmap(cu, A)

    x = rand(Float32, (10, 1, 20)) |> cu
    A.layer(x)

    @show device(params(A))

    B = conv_ensemble(1, 10)
    # B = activation_ensemble()
    B = fmap(cu, B)
    p = params(B)
    # @show p
    for param in p.order.data 
        @show typeof(param), size(param)
    end
    @show device(p)

    C = conv_ensemble(1, 10)
    C = Chain_ensemble([B, C])
    C = fmap(cu, C)

    # C = gen_conv_ensemble(1, 10, 10, 10)
    p = params(C)
    # @show cp

    for param in p.order.data 
        @show typeof(param), size(param)
    end
    @show device(p)
end 


function test_random_graph()
    x = rand(Float32, 10, 1, 20) #|> cu

    g = random_graph(:in, :out, x; depth=15, inchannel=1, insize=10)
    average_infer!(g)
    @show size(g.data[:out])
end

function test_Cell()
    out_channel = 10
    inchannel = 1
    insize = 10
    en_left = gen_conv_ensemble(inchannel, out_channel, insize, insize)
    en_right = gen_conv_ensemble(inchannel, out_channel, insize, insize)
    op = sum_op()
    out_id = :out
    input_symbol = :in
    cell = Cell(en_left, en_right, op, input_symbol, input_symbol, out_id)

    cell = fmap(cu, cell)

    p = params(cell)
    for param in p.order.data 
        @show typeof(param), size(param)
    end
    @show device(p)
end

function test_Graph()
    g = random_graph(:in, :out, x; depth=15, inchannel=1, insize=10)
    g = fmap(cu, g)
    p = params(g)
    for param in p.order.data 
        @show typeof(param), size(param)
    end
    @show device(p)
end

using Flux.Optimise: update!
function test_train()
    x = rand(Float32, 10, 1, 20) |> cu
    g = random_graph(:in, :out, x; depth=15, inchannel=1, insize=10)
    g = fmap(cu, g)

    function loss(g)
        average_infer!(g)
        return sum(g.data[:out])
    end

    opt = Descent(0.01)
    
    for i in 1:10    
        @show loss(g)
        grads = gradient(() -> loss(g), params(g))
        for p in params(g)
            grad = grads[p]
            if !(grad isa Nothing)
                # @show size(grad)
                update!(opt, p, grad)
            end
        end
    end

end


function test_htrain()
    x = rand(Float32, 10, 1, 20) |> cu
    g = random_graph(:in, :out, x; depth=15, inchannel=1, insize=10)
    # g = fmap(cu, g)
    g = gpu(g)

    function loss(g)
        average_infer!(g)
        return sum(g.data[:out])
    end

    # @show device(hparams(g))
    # @show device(params(g))
    # @show loss(g)
    # error()

    opt = Descent(0.1)
    
    for i in 1:10    
        @show loss(g)
        grads = gradient(() -> loss(g), hparams(g))
        for p in hparams(g)
            grad = grads[p]
            if !(grad isa Nothing)
                # @show size(grad)
                update!(opt, p, grad)
            end
        end
    end

end
test_htrain()


function test_zygote()
    W = rand(Float32, (10, 10)) |> cu

    function loss(x)
        data = Dict()
        for i in 1:5
            x = W * x
            x = x ./ sum(x)
            data[gensym()] = x
        end 
        # return sum(map(keys(data)) do k; return sum(data[k]); end)

        # l = zeros(Float32, (10, )) |> cu
        l = nothing

        for k in keys(data)
            li = data[k]
            if l isa Nothing 
                l = li 
            else
                l = deepcopy(l) .+ li 
            end
        end 
        return sum(l)

    end

    @show loss(rand(Float32, (10, )) |> cu)

    x = rand(Float32, (10, )) |> cu
    grads = gradient(() -> loss(x), Params([W, ]))
end

# average_infer!(g)
# @show size(g.data[:out])


# p = params(A)
# @show fieldnames(typeof(p))

# # @show p.order
# # @show p.params

# fa = functor(A)
# @show fa

# # ps = Params([p, p])
# # @show typeof(ps)
# x = rand(Float32, (10, 1, 20))
# sum(A.layer(x))

# grads = gradient(() -> sum(A.layer(x)), p)
# @show fieldnames(typeof(grads))
# # @show grads.grads

# # en = conv_ensemble