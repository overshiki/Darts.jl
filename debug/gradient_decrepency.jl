


# function average_infer(en::Ensemble, x::cArray)
#     coeffs = exp.(en.probs.values)
#     coeffs = coeffs ./ sum(coeffs)

#     y = nothing
#     for (coeff, nnc) in zip(coeffs, en.nn_ops)
#         rx = nnc.layer(x) .* coeff
#         if y isa Nothing 
#             y = rx 
#         else 
#             y = deepcopy(y) .+ rx
#         end
#     end
#     return y
# end

# function average_infer(en::Ensemble, x::cArray)
#     y = []
#     for nnc in en.nn_ops
#         rx = nnc.layer(x)
#         push!(y, reshape(rx, (1, size(rx)...)))
#     end
#     y = vcat(y...)
#     coeffs = exp.(en.probs.values)
#     coeffs = coeffs ./ sum(coeffs)

#     y = deepcopy(y) .* coeffs
#     y = sum(y, dims=1)
#     y = reshape(y, (size(y)[2:end]...))
#     return y
# end

# function average_infer(en::Ensemble, x::cArray)

#     rx = en.nn_ops[1].layer(x)
#     rx_size = size(rx)
#     rx_length = length(rx)

#     y = zeros(Float32, (length(en.nn_ops), rx_length))
#     if x isa CuArray
#         y = y |> cu
#     end 


#     for (i, nnc) in enumerate(en.nn_ops)
#         rx = nnc.layer(x)
#         # push!(y, reshape(rx, (1, size(rx)...)))
#         y[i, :] .= reshape(rx, (rx_length, ))
#     end
#     # y = vcat(y...)
#     coeffs = exp.(en.probs.values)
#     coeffs = coeffs ./ sum(coeffs)

#     y = deepcopy(y) .* coeffs
#     y = sum(y, dims=1)
#     y = reshape(y, rx_size)
#     return y
# end