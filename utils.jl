function add_dim_conv(x)
    x = reshape(x, (size(x)...,1))
    x = convert(array_type, x)
end

add_dim(x) = reshape(x, (size(x)...,1))

function clip(x, low, high)
    x[x .< low] .= low
    x[x .> high] .= high
    x
end
#     out = KnetArray{Float32}(undef,size(x)...)
#     for i in 1:size(x,1)
#         for j in 1:size(x,2)
#             if x[i,j] < low
#                 out[i,j] = low
#             elseif x[i,j] > high
#                 out[i,j] = high
#             else
#                 out[i,j] = x[i,j]
#             end
#         end
#     end
#     out
# end

function batch_inv(mat)
    invs = zeros(size(mat))
    for i in 1:size(mat)[end]
        invs[:,:,i] .= inv(mat[:,:,i])
    end
    invs
end

function myGrid(images, rows, cols)
    images = convert(Array{Float32}, value(images))
    nIms = size(images)[end]
    k = 1
    big_img = zeros(64*rows, 64*cols, 3)
    big_img = convert(Array{Float32}, value(big_img))
    for i in 0:rows-1
        for j in 0:cols-1
            big_img[(64*i)+1:(64*(i+1)), (64*j)+1:(64*(j+1)),:] .= img[:,:,:,k]
            k+=1
            k > nIms && break
        end
    end
#     big_img = (big_img .+ 1) ./ 2
    big_img = big_img .- minimum(big_img)
    big_img= big_img ./ maximum(big_img)
end