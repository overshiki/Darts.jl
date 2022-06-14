using Darts: random_graph, average_infer!, params
using Flux
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

test_train()