mutable struct Backgroundx4
    learntConst
    upconv1
    upconv2
    batchSize
    imSize
    zMapper1
    zMapper2
    zMapper3
    repeatHelper
    function Backgroundx4(imSize::Int, batchSize::Int, zDim::Int)
        learntConst = parameter(4,4,4,imSize*4,1)
        upconv1 = Deconv4(3,3,3,imSize*2,imSize*4; padding = 0, stride = 2, f = identity)
        upconv2 = Deconv4(3,3,3,imSize,imSize*2; padding = 0, stride = 2, f = identity)
        zMap1 = ZMapper(zDim, imSize*4) 
        zMap2 = ZMapper(zDim, imSize*2)
        zMap3 = ZMapper(zDim, imSize*1)
        tile = ones(4, 4, 4, imSize*4, batchSize)
        tile = atype(tile)
        new(learntConst,upconv1,upconv2,batchSize,imSize,zMap1,zMap2,zMap3,tile)
    end
end

function (s::Backgroundx4)(z, view_params, skew_matrix)
    w_tile = s.learntConst .* s.repeatHelper
    s0, b0 = s.zMapper1(z)
    h0 = AdaIN(w_tile, s0, b0)
    h0 = relu.(h0)
    
    h1 = s.upconv1(h0)
    h1 = h1[1:8, 1:8, 1:8, :, :]
    s1, b1 = s.zMapper2(z)
    h1 = AdaIN(h1, s1, b1)
    h1 = relu.(h1)

    h2 = s.upconv2(h1)
    h2 = h2[1:16, 1:16, 1:16, :, :]
    s2, b2 = s.zMapper3(z)
    h2 = AdaIN(h2, s2, b2)
    h2 = relu.(h2)
    h2 = tf_3D_transform_skew(h2, view_params, skew_matrix, 16, 16; shapenet_viewer=false)
end

mutable struct Foregroundx8
    learntConst
    upconv1
    upconv2
    batchSize
    imSize
    zMapper1
    zMapper2
    zMapper3
    repeatHelper
    function Foregroundx8(imSize::Int, batchSize::Int, zDim::Int)
        learntConst = parameter(4,4,4,imSize*8,1)
        upconv1 = Deconv4(3,3,3,imSize*2,imSize*8; padding = 0, stride = 2, f = identity)
        upconv2 = Deconv4(3,3,3,imSize,imSize*2; padding = 0, stride = 2, f = identity)
        zMap1 = ZMapper(zDim, imSize*8) 
        zMap2 = ZMapper(zDim, imSize*2)
        zMap3 = ZMapper(zDim, imSize*1)
        tile = ones(4, 4, 4, imSize*8, batchSize)
        tile = atype(tile)
        new(learntConst,upconv1,upconv2,batchSize,imSize,zMap1,zMap2,zMap3,tile)
    end
end

function (s::Foregroundx8)(z, view_params, skew_matrix)
    w_tile = s.learntConst .* s.repeatHelper
    s0, b0 = s.zMapper1(z)
    h0 = AdaIN(w_tile, s0, b0)
    h0 = relu.(h0)
    
    h1 = s.upconv1(h0)
    h1 = h1[1:8, 1:8, 1:8, :, :]
    s1, b1 = s.zMapper2(z)
    h1 = AdaIN(h1, s1, b1)
    h1 = relu.(h1)

    h2 = s.upconv2(h1)
    h2 = h2[1:16, 1:16, 1:16, :, :]
    s2, b2 = s.zMapper3(z)
    h2 = AdaIN(h2, s2, b2)
    h2 = relu.(h2)
    h2 = tf_3D_transform_skew(h2, view_params, skew_matrix, 16, 16; shapenet_viewer=false)
end

mutable struct Discriminator
    layers
    Discriminator(layers...) = new(layers)
end
function (model::Discriminator)(x)
    for l in model.layers
        x = l(x)
    end
    x
end

mutable struct Generator
    foreground
    background
    upconv1
    upconv2
    upconv3
    upconv4
    imSize
    skew
    function Generator(zDim, zDim2, bs, imSize, skew_matrix)
        fg = Foregroundx8(imSize, bs, zdim)
        bg = Backgroundx4(imSize, bs, zDim2)
        upconv1 = Deconv4(1,1,imSize,16*imSize; padding = 0, stride = 1, f = identity)
        upconv2 = Deconv4(4,4,imSize,imSize; padding = 1, stride = 2, f = identity)
        upconv3 = Deconv4(4,4,imSize,imSize; padding = 1, stride = 2, f = identity)
        upconv4 = Deconv4(4,4,3,imSize; padding = 1, stride = 1, f = identity)
    new(fg,bg,upconv1,upconv2,upconv3,upconv4,imSize, skew_matrix)
    end
end

function (model::Generator)(z, z2, view_fg, view_bg)
    h2_fg = model.foreground(z, view_fg, model.skew)
#     h2_fg2 = model.foreground(Z_Sample(zdim,bs)) #Adding another object
    h2_bg = model.background(z2, view_bg, model.skew)
    h2_pool = max.(h2_bg,h2_fg)
#     h2_pool = h2_bg
    
    h2_pool1 = transform_voxel_to_match_image(h2_pool)
    h2_2d = permutedims(h2_pool, (4,3,2,1,5))
    h2_2d1 = reshape(h2_2d, 16*model.imSize,16,16,:)
    h2_2d2 = permutedims(h2_2d1, (3,2,1,4));
#     h2_2d2 = reshape(h2_pool, (16,16,16*model.imSize,:))
    
    h3 = model.upconv1(h2_2d2)
    h3 = relu.(h3)
    
    h4 = model.upconv2(h3)
    h4  = relu.(h4)
    
    h5 = model.upconv3(h4)
    h5  = relu.(h5)
    
    h6 = model.upconv4(h5)
    h6 = h6[1:model.imSize, 1:model.imSize, :, :]

    output = tanh.(h6)
end

function sigmoid_cross_entropy_with_logits(logits, labels)
    zero = zeros(size(logits)); zero = atype(zero)
    one = ones(size(logits)); one = atype(one)
    max.(logits, zero) .- (logits .* labels) .+ log.(one .+ exp.(.- abs.(logits)))
end

function gLoss(G::Generator, D::Discriminator, z, z2, viewfg, viewbg)
    logits = D(G(z, z2, viewfg, viewbg))
#     logits = permutedims(logits, (2,1))
    labels = ones(size(logits))
    labels = atype(labels)
    return mean(sigmoid_cross_entropy_with_logits(logits, labels))
end

function dLoss(D::Discriminator, realIms, fakeIms)
    labelsReal = ones(size(realIms)[end],1)
    labelsReal = atype(labelsReal)
    labelsFake = zeros(size(realIms)[end],1)
    labelsFake = atype(labelsFake)
    realLoss = mean(sigmoid_cross_entropy_with_logits(D(realIms)', labelsReal))
    fakeLoss = mean(sigmoid_cross_entropy_with_logits(D(fakeIms)', labelsFake))
    return realLoss+fakeLoss
end

function dAccuracy(G, D, z, z2, real)
    fake = G(z, z2)
    fakePred = sum((D(fake) .< 0.5) .== 1)
    truePred = sum((D(real) .> 0.5) .== 1)
    accuray = (fakePred + truePred)/(2*size(real)[end])
end

function dAccFake(G, D, z, z2, viewfg, viewbg)
    fake = G(z, z2, viewfg, viewbg)
    fakePred = sum((D(fake) .< 0) .== 1)
    accuray = (fakePred )/(size(fake)[end])
end

function dAccReal(D, real_img)
    truePred = sum((D(real_img) .> 0) .== 1)
    accuray = truePred/(size(real_img)[end])
end