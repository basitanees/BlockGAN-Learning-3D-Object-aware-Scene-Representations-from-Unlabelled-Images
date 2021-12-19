function add_dim_conv(x)
    x = reshape(x, (size(x)...,1))
    x = convert(array_type, x)
end

add_dim(x) = reshape(x, (size(x)...,1))

add_dim_start(x) = reshape(x, (1, size(x)...))

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

# function myGrid(images, rows, cols)
#     images = convert(Array{Float32}, value(images))
#     nIms = size(images)[end]
#     k = 1
#     big_img = zeros(64*rows, 64*cols, 3)
#     big_img = convert(Array{Float32}, big_img)
#     for i in 0:rows-1
#         for j in 0:cols-1
#             big_img[(64*i)+1:(64*(i+1)), (64*j)+1:(64*(j+1)),:] .= images[:,:,:,k]
#             k+=1
#             k > nIms && break
#         end
#     end
# #     big_img = (big_img .+ 1) ./ 2
#     big_img = big_img .- minimum(big_img)
#     big_img= big_img ./ maximum(big_img)
# end

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
    big_img = (big_img .+ 1) ./ 2
#     big_img = big_img .- minimum(big_img)
#     big_img= big_img ./ maximum(big_img)
end

function viewSample_fg(bs)
    generate_random_rotation_translation(bs; elevation_low=45, elevation_high=46, azimuth_low=0, azimuth_high=359,
                                        scale_low=0.5, scale_high=0.6,
                                        transX_low=-5, transX_high=5,
                                        transY_low=0, transY_high=0,
                                        transZ_low=-5, transZ_high=5,
                                        with_translation=true, with_scale=true)
end

function viewSample_bg(bs)
    generate_random_rotation_translation(bs; elevation_low=45, elevation_high=46, azimuth_low=0, azimuth_high=1,
                                        scale_low=1.0, scale_high=1.0,
                                        transX_low=0, transX_high=0,
                                        transY_low=0, transY_high=0,
                                        transZ_low=0, transZ_high=0,
                                        with_translation=true, with_scale=true)
end
