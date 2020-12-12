
using Knet
# using JLD2
using FileIO
using AutoGrad
using Images
using ImageMagick
using Random
using PyPlot
using Statistics
import Base: length, size, iterate, eltype, IteratorSize, IteratorEltype, haslength, @propagate_inbounds, repeat, rand, tail
import .Iterators: cycle, Cycle, take
using IterTools: ncycle, takenth
import CUDA
import Knet.Ops20: conv4x
include("layers_BlockGAN.jl")
array_type=(CUDA.functional() ? KnetArray{Float32} : Array{Float32})

dataset = load("dataset//singleObjects64Array.jld", "data");

dataTrn = convert(array_type, dataset);

# dataTrn .-= minimum(dataTrn)
# dataTrn ./= maximum(dataTrn)*0.5
# dataTrn .-= 1;

struct ClevrData
    images
    batchsize::Int
    num_instances::Int
    function ClevrData(dataset; batchsize::Int=32) #shuffle::Bool=false)
        nFullBatches, rem = divrem(size(dataTrn)[end], batchsize)
        new(dataset[:,:,:,1:nFullBatches*batchsize], batchsize, nFullBatches*batchsize)
    end
end

function length(d::ClevrData)
    nFullBatches, rem = divrem(d.num_instances, d.batchsize)
    nFullBatches + (rem > 0)*1
end

function iterate(d::ClevrData, state=collect(1:d.num_instances))
    if length(state) > 0
        batch = d.images[:,:,:,state[1:(length(state) < d.batchsize ? end : d.batchsize)]]
        state  = state[d.batchsize+1:end]
        return (Param(batch)), state
    end
end

function dAccuracy(G, D, z, real)
    fake = G(z)
    fakePred = sum((D(fake) .< 0.5) .== 1)
    truePred = sum((D(real) .> 0.5) .== 1)
    accuray = (fakePred + truePred)/(2*size(real)[end])
end

function dAccFake(G, D, z, z2)
    fake = G(z, z2)
    fakePred = sum((D(fake) .< 0) .== 1)
    accuray = (fakePred )/(size(fake)[end])
end

function dAccReal(D, real_img)
    truePred = sum((D(real_img) .> 0) .== 1)
    accuray = truePred/(size(real_img)[end])
end

struct Backgroundx4
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
        learntConst = Param(param(4,4,4,imSize*4,1, init=gaussian) .* 2)
        upconv1 = Deconv4(3,3,3,imSize*2,imSize*4,8,batchSize)
        upconv2 = Deconv4(3,3,3,imSize,imSize*2,16,batchSize)
        zMap1 = ZMapper(zDim, imSize*4) 
        zMap2 = ZMapper(zDim, imSize*2)
        zMap3 = ZMapper(zDim, imSize*1)
        tile = ones(4, 4, 4, imSize*4, batchSize)
        tile = convert(KnetArray{Float32}, tile)
        new(learntConst,upconv1,upconv2,batchSize,imSize,zMap1,zMap2,zMap3,tile)
    end
end

function (s::Backgroundx4)(z)
    w_tile = s.learntConst .* s.repeatHelper
    s0, b0 = s.zMapper1(z)
    h0 = AdaIN(w_tile, s0, b0)
    h0 = relu.(h0)
    
    h1 = s.upconv1(h0)
    s1, b1 = s.zMapper2(z)
    h1 = AdaIN(h1, s1, b1)
    h1 = relu.(h1)

    h2 = s.upconv2(h1)
    s2, b2 = s.zMapper3(z)
    h2 = AdaIN(h2, s2, b2)
    h2 = relu.(h2)
end

struct Foregroundx8
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
        learntConst = Param(param(4,4,4,imSize*8,1, init=gaussian) .* 2)
        upconv1 = Deconv4(3,3,3,imSize*2,imSize*8,8,batchSize)
        upconv2 = Deconv4(3,3,3,imSize,imSize*2,16,batchSize)
        zMap1 = ZMapper(zDim, imSize*8) 
        zMap2 = ZMapper(zDim, imSize*2)
        zMap3 = ZMapper(zDim, imSize*1)
        tile = ones(4, 4, 4, imSize*8, batchSize)
        tile = convert(KnetArray{Float32}, tile)
        new(learntConst,upconv1,upconv2,batchSize,imSize,zMap1,zMap2,zMap3,tile)
    end
end

function (s::Foregroundx8)(z)
    w_tile = s.learntConst .* s.repeatHelper
    s0, b0 = s.zMapper1(z)
    h0 = AdaIN(w_tile, s0, b0)
    h0 = relu.(h0)
    
    h1 = s.upconv1(h0)
    s1, b1 = s.zMapper2(z)
    h1 = AdaIN(h1, s1, b1)
    h1 = relu.(h1)

    h2 = s.upconv2(h1)
    s2, b2 = s.zMapper3(z)
    h2 = AdaIN(h2, s2, b2)
    h2 = relu.(h2)
end

struct Discriminator
    layers
    Discriminator(layers...) = new(layers)
end
function (model::Discriminator)(x)
    for l in model.layers
        x = l(x)
    end
    x
end

struct Generator
    foreground
    background
    upconv1
    upconv2
    upconv3
    upconv4
    imSize
    function Generator(zDim, zDim2, bs, imSize)
        fg = Foregroundx8(imSize, bs, zdim)
        bg = Backgroundx4(imSize, bs, zDim2)
        upconv1 = Deconv(1,1,16*imSize,16*imSize)
        upconv2 = Deconv2(4,4,4*imSize,16*imSize,identity)
        upconv3 = Deconv2(4,4,imSize,4*imSize,identity)
        upconv4 = Deconv3(5,5,3,imSize,identity)
    new(fg,bg,upconv1,upconv2,upconv3,upconv4,imSize)
    end
end

function (model::Generator)(z, z2)
    h2_fg = model.foreground(z)
#     h2_fg2 = model.foreground(Z_Sample(zdim,bs)) #Adding another object
    h2_bg = model.background(z2)
    h2_pool = max.(h2_bg,h2_fg)
#     h2_pool = h2_bg
    h2_2d = reshape(h2_pool, (16,16,16*model.imSize,:))
    
    h3 = model.upconv1(h2_2d)
    
    h4 = model.upconv2(h3)
    h4  = relu.(h4)
    
    h5 = model.upconv3(h4)
    h5  = relu.(h5)
    
    h6 = model.upconv4(h5)

    output = tanh.(h6)
end

function sigmoid_cross_entropy_with_logits(logits, labels)
    zero = zeros(size(logits)); zero = convert(array_type, zero)
    one = ones(size(logits)); one = convert(array_type, one)
    max.(logits, zero) .- (logits .* labels) .+ log.(one .+ exp.(.- abs.(logits)))
end

function gLoss(G::Generator, D::Discriminator, z, z2)
    logits = D(G(z, z2))'
    labels = ones(size(logits))
    labels = convert(array_type, labels)
    return mean(sigmoid_cross_entropy_with_logits(logits, labels))
end

function dLoss(D::Discriminator, realIms, fakeIms)
    labelsReal = ones(size(realIms)[end],1)
    labelsReal = convert(array_type, labelsReal)
    labelsFake = zeros(size(realIms)[end],1)
    labelsFake = convert(array_type, labelsFake)
    realLoss = mean(sigmoid_cross_entropy_with_logits(D(realIms)', labelsReal))
    fakeLoss = mean(sigmoid_cross_entropy_with_logits(D(fakeIms)', labelsFake))
    return realLoss+fakeLoss
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
    big_img = big_img .- minimum(big_img)
    big_img= big_img ./ maximum(big_img)
end

function main()
    i = 0
    for real_image in ncycle(clevrDataset, 20)
        bs = size(real_image)[end]
        z = Z_Sample(zdim,bs)
        z2 = Z_Sample(zdim2,bs)
        if i % 25 == 0
            push!(loss_g, gLoss(G, D, z, z2))
            push!(loss_d, dLoss(D, real_image, G(z, z2)))
            print("$(i): GenLoss: ");print(loss_g[end])
            print("    DisLoss: ");print(loss_d[end])
            push!(acc_d_real, dAccReal(D, real_image))
            push!(acc_d_fake, dAccFake(G,D,z, z2))
            print("    AccReal: ");print(acc_d_real[end])
            print("    AccFake: ");println(acc_d_fake[end])
        end
        adam!(dLoss, [(D, real_image, G(z, z2))], gclip = 1, lr = 0.0001, beta1=0.5)#, params=params(D))
        adam!(gLoss, [(G, D, z, z2)], params=params(G), gclip = 1, lr = 0.0001,beta1=0.5)
        adam!(gLoss, [(G, D, z, z2)], params=params(G), gclip = 1, lr = 0.0001,beta1=0.5)
        i += 1
    end
end

######################################################################################
Random.seed!(1234)
loss_g = []
loss_d = []
acc_d_real = []
acc_d_fake = []
bs = 64
zdim = 60
zdim2 = 20
imsize = 64
clevrDataset = ClevrData(dataTrn, batchsize=bs)
G = Generator(zdim, zdim2, bs, imsize)
D = Discriminator(Conv(5,5,3,imsize), Conv_spectral(5,5,imsize,imsize*2), InstanceNorm(imsize*2), Conv_spectral(5,5,imsize*2,imsize*4), InstanceNorm(imsize*4), Conv_spectral(5,5,imsize*4,imsize*8), InstanceNorm(imsize*8), Dense(4*4*imsize*8,1,identity))

main()
z = Z_Sample(zdim,bs)
z2 = Z_Sample(zdim2,bs)
img = G(z, z2)
big_img = myGrid(img, 4, 8)
imshow(big_img)
######################################################################################

# JLD2.@save "BlockGAN_models//8Epochs.jld2" G D
# JLD2.@load "BlockGAN_models//8Epochs.jld2" G D

# FileIO.save("BlockGAN_models//3Epochs.jld2", "G", G, "D", D)

# G, D = load("BlockGAN_models//20Epochs.jld2", "G", "D")

# for par in params(D)
#     println(mean(par))
#     println(std(par))
#     println()
# end

#lossPlot = plot([loss_g, loss_d],labels=["Gen" ,"Dis"],xlabel="Iters",ylabel="Loss");

#lossPlot = plot([acc_d_real, acc_d_fake],labels=["Real" ,"Fake"],xlabel="Iters",ylabel="Loss")