function transform_voxel_to_match_image(voxel)
    voxel = permutedims(voxel, (1, 3, 2, 4, 5)) # 2,1,3,4,5
    idx_rev = size(voxel,3):-1:1
    voxel = voxel[:, :, idx_rev, :, :]
    return voxel
end

function tf_interpolate(voxel, x, y, z, out_size)
    """
    Trilinear interpolation for batch of voxels
    :param voxel: The whole voxel grid
    :param x,y,z: indices of voxel
    :param output_size: output size of voxel
    :return:
    """

    dims = length(size(voxel))
    shape = size(voxel)
    height = shape[1]
    width = shape[2]
    depth = shape[3]
    n_channels = shape[4]
    batch_size = shape[5]

#     x = convert(array_type, x)
#     y = convert(array_type, y)
#     z = convert(array_type, z)

    out_height = out_size[1]
    out_width = out_size[2]
    out_depth = out_size[3]
    out_channel = out_size[4]

    zero = 0
#     zero = zeros() # tf.zeros([], dtype='int32')
#     zero = convert(array_type, zero)
    max_y = height-1 # as int
    max_x = width-1
    max_z = depth-1

    # do sampling
    x0 = floor.(x) # as int
    x1 = x0 .+ 1
    y0 = floor.(y)
    y1 = y0 .+ 1
    z0 = floor.(z)
    z1 = z0 .+ 1

    x0 = clip(x0, zero, max_x)
    x1 = clip(x1, zero, max_x)
    y0 = clip(y0, zero, max_y)
    y1 = clip(y1, zero, max_y)
    z0 = clip(z0, zero, max_z)
    z1 = clip(z1, zero, max_z)
    
    

    #A 1D tensor of base indicies describe First index for each shape/map in the whole batch
    #tf.range(batch_size) * width * height * depth : Element to repeat. Each selement in the list is incremented by width*height*depth amount
    # out_height * out_width * out_depth: n of repeat. Create chunks of out_height*out_width*out_depth length with the same value created by tf.rage(batch_size) *width*height*dept
    base = tf_repeat(collect(0:batch_size-1) * width * height * depth, out_height * out_width * out_depth)
    # add +1 for julia indexing

    #Find the Z element of each index
#     base = convert(array_type, base)

    base_z0 = base .+ (z0 .* (width * height))
    base_z1 = base .+ (z1 .* (width * height))
    #Find the Y element based on Z
    base_z0_y0 = base_z0 .+ (y0 .* width)
    base_z0_y1 = base_z0 .+ (y1 .* width)
    base_z1_y0 = base_z1 .+ (y0 .* width)
    base_z1_y1 = base_z1 .+ (y1 .* width)

    # Find the X element based on Y, Z for Z=0
    idx_a = base_z0_y0 .+ x0 .+ 1  # Add 1 for julia indexing
    idx_b = base_z0_y1 .+ x0 .+ 1
    idx_c = base_z0_y0 .+ x1 .+ 1
    idx_d = base_z0_y1 .+ x1 .+ 1
    # Find the X element based on Y,Z for Z =1
    idx_e = base_z1_y0 .+ x0 .+ 1
    idx_f = base_z1_y1 .+ x0 .+ 1
    idx_g = base_z1_y0 .+ x1 .+ 1
    idx_h = base_z1_y1 .+ x1 .+ 1
    
    idx_a = convert(Array{Int32}, idx_a)
    idx_b = convert(Array{Int32}, idx_b)
    idx_c = convert(Array{Int32}, idx_c)
    idx_d = convert(Array{Int32}, idx_d)
    idx_e = convert(Array{Int32}, idx_e)
    idx_f = convert(Array{Int32}, idx_f)
    idx_g = convert(Array{Int32}, idx_g)
    idx_h = convert(Array{Int32}, idx_h)
    
#     println(maximum(idx_a))
#     println(maximum(idx_b))
#     println(maximum(idx_c))
#     println(maximum(idx_d))
#     println(maximum(idx_e))
#     println(maximum(idx_f))
#     println(maximum(idx_g))
#     println(maximum(idx_h))

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    voxel = permutedims(voxel, (4,2,1,3,5))
    voxel_flat = reshape(voxel, n_channels, :) # may need to transpose
    voxel_flat = permutedims(voxel_flat, (2,1))
    
#     voxel_flat = convert(array_type, voxel_flat)

#     Ia = view(voxel_flat, idx_a.+1, :)
#     Ib = view(voxel_flat, idx_b.+1, :)
#     Ic = view(voxel_flat, idx_c.+1, :)
#     Id = view(voxel_flat, idx_d.+1, :)
#     Ie = view(voxel_flat, idx_e.+1, :)
#     If = view(voxel_flat, idx_f.+1, :)
#     Ig = view(voxel_flat, idx_g.+1, :)
#     Ih = view(voxel_flat, idx_h.+1, :)

    # and finally calculate interpolated values
#     x0_f = convert(array_type, x0)
#     x1_f = convert(array_type, x1)
#     y0_f = convert(array_type, y0)
#     y1_f = convert(array_type, y1)
#     z0_f = convert(array_type, z0)
#     z1_f = convert(array_type, z1)
    
    x0_f = x0
    x1_f = x1
    y0_f = y0
    y1_f = y1
    z0_f = z0
    z1_f = z1

    #First slice XY along Z where z=0
    wa = add_dim_conv((x1_f - x) .* (y1_f - y) .* (z1_f-z))
    wb = add_dim_conv((x1_f - x) .* (y - y0_f) .* (z1_f-z))
    wc = add_dim_conv((x - x0_f) .* (y1_f - y) .* (z1_f-z))
    wd = add_dim_conv((x - x0_f) .* (y - y0_f) .* (z1_f-z))
    # First slice XY along Z where z=1
    we = add_dim_conv((x1_f - x) .* (y1_f - y) .* (z-z0_f))
    wf = add_dim_conv((x1_f - x) .* (y - y0_f) .* (z-z0_f))
    wg = add_dim_conv((x - x0_f) .* (y1_f - y) .* (z-z0_f))
    wh = add_dim_conv((x - x0_f) .* (y - y0_f) .* (z-z0_f))

    ca = wa .* voxel_flat[idx_a, :]
    cb = wb .* voxel_flat[idx_b, :]
    cc = wc .* voxel_flat[idx_c, :]
    cd = wd .* voxel_flat[idx_d, :]
    ce = we .* voxel_flat[idx_e, :]
    cf = wf .* voxel_flat[idx_f, :]
    cg = wg .* voxel_flat[idx_g, :]
    ch = wh .* voxel_flat[idx_h, :]
#     ca = wa .* Ia
#     cb = wb .* Ib
#     cc = wc .* Ic
#     cd = wd .* Id
#     ce = we .* Ie
#     cf = wf .* If
#     cg = wg .* Ig
#     ch = wh .* Ih
    output = ca .+ cb .+ cc .+ cd .+ ce .+ cf .+ cg .+ ch
#     output = wa .* Ia.+ wb .* Ib.+ wc .* Ic.+ wd .* Id.+  we .* Ie.+ wf .* If.+ wg .* Ig.+ wh .* Ih
    output
end

function tf_rotation_resampling_skew(voxel_array, transformation_matrix, skew_matrix, params, Scale_matrix = None, size1=64, new_size=128)
    """
    Batch resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size:
    :param new_size:
    :return:
    """

    batch_size = size(voxel_array)[end]
    n_channels = size(voxel_array)[end-1]
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = vcat([1 0 0 -size1*0.5],
             [0 1 0 -size1*0.5],
             [0 0 1 -size1*0.5],
             [0 0 0 1])
    T = reshape(T,4,4,1)
    repeatHelper = ones(4,4,batch_size)
    T = repeatHelper .* T

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = vcat([1 0 0 new_size*0.5],
                     [0 1 0 new_size*0.5],
                     [0 0 1 new_size*0.5],
                     [0 0 0 1])
    T_new_inv = reshape(T_new_inv,4,4,1)
    T_new_inv = repeatHelper .* T_new_inv

    # Add the actual shifting in x and y dimension accoding to input param
    x_shift = reshape(params[4, :], 1, 1, batch_size)
    y_shift = reshape(params[5, :], 1, 1, batch_size)
    z_shift = reshape(params[6, :], 1, 1, batch_size)
    # ========================================================
    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    oness = ones(size(x_shift))
    zeross = zeros(size(x_shift))

    a1 = cat(oness, zeross, zeross, x_shift, dims=2) # dims 2 may be swapped with 1. May be checked later
    a2 = cat(zeross, oness, zeross, y_shift, dims=2)
    a3 = cat(zeross, zeross, oness, z_shift, dims=2)
    a4 = cat(zeross, zeross, zeross, oness, dims=2)
    T_translate = cat(a1, a2, a3, a4, dims=1)
    
    skew_inv = batch_inv(skew_matrix)
    
    total_M = bmm(bmm(bmm(skew_inv, T_translate), Scale_matrix), transformation_matrix)
    total_M = bmm(bmm(T_new_inv, total_M), T)
#     try
    total_M = batch_inv(total_M)
    
    total_M = total_M[1:3, :, :] #Ignore the homogenous coordinate so the results are 3D vectors
    grid = tf_voxel_meshgrid(new_size, new_size, new_size, homogeneous=true)
    repeatHelper2 = ones(size(grid)..., batch_size)
    grid = add_dim(grid)
    grid = grid .* repeatHelper2
    grid_transform = bmm(total_M, grid)
    
#     homo_coor = add_dim(grid_transform[4, :, :])
#     homo_coor = tf.tile(homo_coor, (1, 4, 1))
#     grid_transform = tf.div(grid_transform, homo_coor)
    
#     grid_transform = convert(array_type, grid_transform)
    x_s_flat = reshape(grid_transform[1, :, :], :)
    y_s_flat = reshape(grid_transform[2, :, :], :)
    z_s_flat = reshape(grid_transform[3, :, :], :)

#     x_s_flat, y_s_flat, z_s_flat, [new_size, new_size, new_size, n_channels, batch_size]
# end

# function trilinear_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat, outSize)
    input_transformed = tf_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat, [new_size, new_size, new_size, n_channels, batch_size])
    input_transformed = permutedims(input_transformed, (2,1))
    target = reshape(input_transformed, n_channels, new_size, new_size, new_size, batch_size)
    target = permutedims(target, (3,2,4,1,5))
    return target, grid_transform
#     catch
#         return None
end

function tf_rotation_resampling(voxel_array, transformation_matrix, params, Scale_matrix = None, size1=64, new_size=128)
    """
    Batch resampling function
    :param voxel_array: batch of voxels. Shape = [batch_size, height, width, depth, features]
    :param transformation_matrix: Rotation matrix. Shape = [batch_size, height, width, depth, features]
    :param size:
    :param new_size:
    :return:
    """

    batch_size = size(voxel_array)[end]
    n_channels = size(voxel_array)[end-1]
    #Aligning the centroid of the object (voxel grid) to origin for rotation,
    #then move the centroid back to the original position of the grid centroid
    T = vcat([1 0 0 -size1*0.5],
             [0 1 0 -size1*0.5],
             [0 0 1 -size1*0.5],
             [0 0 0 1])
    T = reshape(T,4,4,1)
    repeatHelper = ones(4,4,batch_size)
    T = repeatHelper .* T

    # However, since the rotated grid might be out of bound for the original grid size,
    # move the rotated grid to a new bigger grid
    T_new_inv = vcat([1 0 0 new_size*0.5],
                     [0 1 0 new_size*0.5],
                     [0 0 1 new_size*0.5],
                     [0 0 0 1])
    T_new_inv = reshape(T_new_inv,4,4,1)
    T_new_inv = repeatHelper .* T_new_inv

    # Add the actual shifting in x and y dimension accoding to input param
    x_shift = reshape(params[4, :], 1, 1, batch_size)
    y_shift = reshape(params[5, :], 1, 1, batch_size)
    z_shift = reshape(params[6, :], 1, 1, batch_size)
    # ========================================================
    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    oness = ones(size(x_shift))
    zeross = zeros(size(x_shift))

    a1 = cat(oness, zeross, zeross, x_shift, dims=2) # dims 2 may be swapped with 1. May be checked later
    a2 = cat(zeross, oness, zeross, y_shift, dims=2)
    a3 = cat(zeross, zeross, oness, z_shift, dims=2)
    a4 = cat(zeross, zeross, zeross, oness, dims=2)
    T_translate = cat(a1, a2, a3, a4, dims=1)

    total_M = bmm(bmm(bmm(bmm(T_new_inv, T_translate), Scale_matrix), transformation_matrix), T)
#     try
    total_M = batch_inv(total_M)
    
    total_M = total_M[1:3, :, :] #Ignore the homogenous coordinate so the results are 3D vectors
    grid = tf_voxel_meshgrid(new_size, new_size, new_size, homogeneous=true)
    repeatHelper2 = ones(size(grid)..., batch_size)
    grid = add_dim(grid)
    grid = grid .* repeatHelper2
    grid_transform = bmm(total_M, grid)
#     grid_transform = convert(array_type, grid_transform)
    x_s_flat = reshape(grid_transform[1, :, :], :)
    y_s_flat = reshape(grid_transform[2, :, :], :)
    z_s_flat = reshape(grid_transform[3, :, :], :)
    input_transformed = tf_interpolate(voxel_array, x_s_flat, y_s_flat, z_s_flat,[new_size, new_size, new_size, n_channels, batch_size])
    input_transformed = permutedims(input_transformed, (2,1))
    target = reshape(input_transformed, n_channels, new_size, new_size, new_size, batch_size)
    target = permutedims(target, (3,2,4,1,5))
    return target, grid_transform
#     return
#     catch
#         return None
end

function meshgrid(xin,yin,zin)
    nx=length(xin)
    ny=length(yin)
    nz=length(zin)
    xout=zeros(nz,ny,nx)
    yout=zeros(nz,ny,nx)
    zout=zeros(nz,ny,nx)
    for kx=1:nz
        for jx=1:ny
            for ix=1:nx
                xout[kx,jx,ix]=xin[ix]
                yout[kx,jx,ix]=yin[jx]
                zout[kx,jx,ix]=zin[kx]
            end
        end
    end
    return xout, yout, zout
end

function tf_voxel_meshgrid(height, width, depth; homogeneous = false)
    #Because 'ij' ordering is used for meshgrid, z_t and x_t are swapped (Think about order in 'xy' VS 'ij'
    x_t, y_t, z_t = meshgrid(1:height, 1:width, 1:depth)
    #Reshape into a big list of slices one after another along the X,Y,Z direction
    x_t_flat = reshape(x_t, 1, :)
    y_t_flat = reshape(y_t, 1, :)
    z_t_flat = reshape(z_t, 1, :)

    #Vertical stack to create a (3,N) matrix for X,Y,Z coordinates
    grid = vcat(x_t_flat, y_t_flat, z_t_flat) # (N,3)
    grid = grid .- 1
    if homogeneous
        oness = ones(size(x_t_flat))
        grid = vcat(grid, oness)
    end
    return grid
end

function tf_3D_transform(voxel_array, view_params, skew_matrix, size, new_size; shapenet_viewer=false)
    M, S = tf_rotation_around_grid_centroid(view_params[1:3, :], shapenet_viewer=shapenet_viewer)
    target, grids = tf_rotation_resampling(voxel_array, M, view_params, S, size, new_size)
    target
end

function tf_3D_transform_skew(voxel_array, view_params, skew_matrix, size, new_size; shapenet_viewer=false)
    M, S = tf_rotation_around_grid_centroid(view_params[1:3, :], shapenet_viewer=shapenet_viewer)
    target, grids = tf_rotation_resampling_skew(voxel_array, M, skew_matrix, view_params, S, size, new_size)
    target
end

function tf_rotation_around_grid_centroid(view_params; shapenet_viewer = false)
    """
    :param view_params: batch of view parameters. Shape : [batch_size, 2]
    :param radius:
    :param useX: USe when X axis and Z axis are switched
    :return:
    """
    #This function returns a rotation matrix around a center with y-axis being the up vector.
    #It first rotates the matrix by the azimuth angle (theta) around y, then around X-axis by elevation angle (gamma)
    #return a rotation matrix in homogenous coordinate
    #The default Open GL camera is to looking towards the negative Z direction
    #This function is suitable when the silhoutte projection is done along the Z direction
    batch_size = size(view_params)[end]

    azimuth    = reshape(view_params[1, :], 1, 1, batch_size)
    elevation  = reshape(view_params[2, :], 1, 1, batch_size)

    # azimuth = azimuth
    if shapenet_viewer == false
        azimuth = azimuth .- (π * 0.5)
    end

    # Because tensorflow does not allow tensor item replacement
    # A new matrix needs to be created from scratch by concatenating different vectors into rows and stacking them up
    # Batch Rotation Y matrixes
    oness = ones(size(azimuth))
    zeross = zeros(size(azimuth))
    a1 = cat(cos.(azimuth), zeross, -sin.(azimuth), zeross, dims=2) # dims 2 may be swapped with 1. May be checked later
    a2 = cat(zeross, oness, zeross, zeross, dims=2)
    a3 = cat(sin.(azimuth), zeross, cos.(azimuth), zeross, dims=2)
    a4 = cat(zeross, zeross, zeross, oness, dims=2)
    batch_Rot_Y = cat(a1, a2, a3, a4, dims=1)

    # #Batch Rotation Z matrixes
    # batch_Rot_Z = tf.concat([
    #     tf.concat([tf.cos(elevation),  tf.sin(elevation),  zeros, zeros], axis=2),
    #     tf.concat([-tf.sin(elevation), tf.cos(elevation),  zeros, zeros], axis=2),
    #     tf.concat([zeros, zeros, ones,  zeros], axis=2),
    #     tf.concat([zeros, zeros, zeros, ones], axis=2)], axis=1)
    
    b1 = cat(oness, zeross, zeross, zeross, dims=2) # dims 2 may be swapped with 1. May be checked later
    b2 = cat(zeross, cos.(elevation), -sin.(elevation), zeross, dims=2)
    b3 = cat(zeross, sin.(elevation), cos.(elevation), zeross, dims=2)
    b4 = cat(zeross, zeross, zeross, oness, dims=2)
    batch_Rot_X = cat(b1, b2, b3, b4, dims=1)
    
    transformation_matrix = bmm(batch_Rot_X, batch_Rot_Y)
    if size(view_params)[1] == 2
        return transformation_matrix
    else
    #Batch Scale matrixes:
        scale = reshape(view_params[3, :], (1, 1, batch_size))
        c1 = cat(scale,  zeross,  zeross, zeross, dims=2)
        c2 = cat(zeross, scale,  zeross, zeross, dims=2)
        c3 = cat(zeross, zeross,  scale,  zeross, dims=2)
        c4 = cat(zeross, zeross,  zeross, oness, dims=2)
        batch_Scale= cat(c1, c2, c3, c4, dims=1)
    end
    return transformation_matrix, batch_Scale
end

function tf_repeat(x, n_repeats)
    #Repeat X for n_repeats time along 0 axis
    #Return a 1D tensor of total number of elements
    rep = ones(1, n_repeats)
    xN = reshape(x',length(x))
    x = xN * rep
    reshape(x', length(x))
#     x = convert(array_type, x)
end

function generate_random_rotation_translation(batch_size; elevation_low=45, elevation_high=46, azimuth_low=0, azimuth_high=359,
                                        scale_low=0.5, scale_high=0.6,
                                        transX_low=-5, transX_high=5,
                                        transY_low=0, transY_high=0,
                                        transZ_low=-5, transZ_high=5,
                                        with_translation=true, with_scale=true)
    params = zeros(6, batch_size)
    azimuth = rand(azimuth_low:azimuth_high, batch_size) .* π ./ 180.0
    temp = rand(elevation_low:elevation_high, batch_size)
    elevation = (90. .- temp) .* π ./ 180.0
    params[1, :] .= azimuth
    params[2, :] .= elevation

    if with_translation
        shift_x = transX_low .+ rand(batch_size) .* (transX_high - transX_low)
        shift_y = transY_low .+ rand(batch_size) .* (transY_high - transY_low)
        shift_z = transZ_low .+ rand(batch_size) .* (transZ_high - transZ_low)
        params[4, :] .= shift_x
        params[5, :] .= shift_y
        params[6, :] .= shift_z
    end

    if with_scale
        scale = scale_low .+ (rand(1) .* (scale_high - scale_low))
        params[3, :] .= scale
    else
        params[3, :] .= 1.0
    end

    return params
end

function generate_random_rotation_translation_2objs(batch_size; elevation_low=90, elevation_high=91, azimuth_low=0, azimuth_high=359, scale_low=0.8, scale_high=1.5,
                                         transX_low=-3, transX_high=3,  transY_low=-3, transY_high=3, transZ_low=-3, transZ_high=3,
                                         with_translation=false, with_scale=false, margin=nothing)
   #Sampling translation in as integers, not as floats

    params = zeros(12, batch_size)
    for i in 1:batch_size
        azimuth = rand(azimuth_low:azimuth_high, 2) .* π ./ 180.0
        temp = rand(elevation_low:elevation_high, 2)
        elevation = (90. .- temp) .* π ./ 180.0
        params[1, i] = azimuth[1]
        params[2, i] = elevation[1]
        params[7, i] = azimuth[2]
        params[8, i] = elevation[2]


        if with_translation
            if (transX_low==0) && (transX_high==0)
                shift_x = [0, 0]
            else
                shift_x = rand(transX_low:transX_high, 2)
            end

            if (transY_low == 0) && (transY_high==0)
                shift_y = [0, 0]
            else
                shift_y = rand(transY_low:transY_high, 2)
            end


            shift_z = rand(transZ_low + 1:transZ_high)

            # shift_x2 = rand(transX_low:transX_high)
            # shift_y2 = rand(transY_low:transY_high)
            shift_z2 = nothing
            try
                shift_z2 = rand(transZ_low:shift_z)
            catch
                print(shift_z)
            end

            params[4, i] = shift_x[1]
            params[5, i] = shift_y[1]
            params[6, i] = shift_z
            params[10, i] = shift_x[2]
            params[11, i] = shift_y[2]
            params[12, i] = shift_z2
            
        else
            shift_x = 0
            shift_y = 0
            shift_z = 0
            params[4, i] = shift_x
            params[5, i] = shift_y
            params[6, i] = shift_z
        end

        if with_scale
            scale = rand([scale_low, scale_high], 2)
            params[3, i] = scale[1]
            params[9, i] = scale[2]
        else
            params[3, i] = 1.0
            params[9, i] = 1.0
        end
    end
    return params
end

function compute_skew_matrix_nearFar(batch_size, size, new_size; focal_length=35, sensor_size=32, distance = 10)
    ## Corners of the "normalized device coordinates" (i.e. output grid)
    x1 = x4 = -new_size / 2 + 0.5
    x2 = x3 = new_size / 2 - 0.5
    y1 = y2 = new_size / 2 - 0.5
    y3 = y4 = -new_size / 2 + 0.5

    ## Corners of the frustum (take 3: introducing z near/far).
    #Hardcoded d for debug
    # d = np.sqrt(3) * size  # distance of camera from origin [unit: input voxels]
    # d = 0.75 * size

    z_near = 0.75 * distance  # a quarter of the way from origin towards camera [input voxels]
    z_far = 1.25 * distance  # a quarter of the way from origin away from camera [input voxels]

    # d = 20; z_near = d - 2; z_far = d + 2  # fixed test

    # print(f"d = {d} || z_near ... z_far = {z_near} ... {z_far}")
    x1p = -sensor_size * z_far / (2 * focal_length)
    x2p = sensor_size * z_far / (2 * focal_length)
    x3p = sensor_size * z_near / (2 * focal_length)
    x4p = -sensor_size * z_near / (2 * focal_length)
    y1p = distance - z_near;
    y2p = distance - z_near;
    y3p = distance - z_far;
    y4p = distance - z_far
    # print(f"(x1p, y2p) = ({x1p}, {y1p})")

    ## Solve homography from 4 point correspondences (general case)
    ## Source: https://math.stackexchange.com/a/2619023
    PMat = hcat(
        [-x1, -y1, -1, 0, 0, 0, x1 * x1p, y1 * x1p, x1p],
        [0, 0, 0, -x1, -y1, -1, x1 * y1p, y1 * y1p, y1p],
        [-x2, -y2, -1, 0, 0, 0, x2 * x2p, y2 * x2p, x2p],
        [0, 0, 0, -x2, -y2, -1, x2 * y2p, y2 * y2p, y2p],
        [-x3, -y3, -1, 0, 0, 0, x3 * x3p, y3 * x3p, x3p],
        [0, 0, 0, -x3, -y3, -1, x3 * y3p, y3 * y3p, y3p],
        [-x4, -y4, -1, 0, 0, 0, x4 * x4p, y4 * x4p, x4p],
        [0, 0, 0, -x4, -y4, -1, x4 * y4p, y4 * y4p, y4p],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
            )'
#     print(size(PMat))
    H = inv(PMat) * [0, 0, 0, 0, 0, 0, 0, 0, 1]
    H = reshape(H, 3, 3)
    H_3D = hcat([H[1, 1], 0, 0, 0],
           [0, H[1, 1], 0, 0],
           [0, 0, H[2, 2], H[2, 3]],
           [0, 0, H[3, 2], 1])
    H_3D = reshape(H_3D, 4,4,1)
    tiler = ones(4, 4, batch_size)

    skew_3D_batch = H_3D .* tiler
#     return skew_3D_batch
end

# function mul3D(x,y) # bmm used now
#     shapeX = size(x)
#     shapeY = size(y)
#     z = zeros(shapeX[1],shapeY[2],shapeX[end])
#     for i in 1:shapeX[end]
#        z[:,:,i] .= x[:,:,i] * y[:,:,i]
#     end
#     z
# end