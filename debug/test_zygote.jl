using Zygote

function f(w::Array{Float64,2}, x::Array{Float64,2})

    result = []

    for i = 1:size(x,2)
        h = w[:,i] .+ x[:,i]
        push!(result, h)
    end

    cat = hcat(result...)
    return sum(cat)
end

s = rand(5,100)
x = rand(5,100)

v = f(s,x)

l = Zygote.gradient((w)-> f(w, x), s)