import Flux.Params
using CUDA

function device(a::Params)
    param = a.order.data 
    is_cuda = map(param) do x 
        return x isa CuArray
    end 
    if length(is_cuda)==0
        return :empty 
    end

    if all(is_cuda) !== any(is_cuda)
        return :mixed
    end 

    if all(is_cuda)
        return :gpu
    else
        return :cpu 
    end
end