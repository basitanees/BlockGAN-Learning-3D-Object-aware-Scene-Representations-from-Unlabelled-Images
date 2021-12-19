function Z(dim1, dim2)
    z = (rand(dim1, dim2).*2.0).-1.0
    convert(array_type, z)
end

function Z_Sample(dim1, dim2)
    z = atype(zeros(dim1, dim2))
    z = (rand!(z).*2.0).-1.0
end

function leakyRelu(x)
    max(0.2 .* x, x)
end

mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end
(model::Chain)(x) = (for l in model.layers; x = l(x); end; x)

parameter(dims...;mean=0.2) = Param(param(dims...,init=gaussian, atype=atype()) .* mean / 0.01)

mutable struct Dense; w; b; f; end
Dense(i::Int,o::Int,f=relu) = Dense(parameter(o,i), param0(o, atype=atype()), f)
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul

mutable struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, stride = 2, padding = 2, mode = 1) .+ c.b)
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=leakyRelu) = Conv(parameter(w1,w2,cx,cy), param0(1,1,cy,1, atype=atype()), f)

mutable struct Conv_spectral; w; b; u; f; end
(c::Conv_spectral)(x) = conv4(spectralNorm(c.w, c.u), x, stride = 2, padding = 2, mode = 1) .+ c.b
Conv_spectral(w1::Int,w2::Int,cx::Int,cy::Int,f=leakyRelu) = Conv_spectral(parameter(w1,w2,cx,cy), param0(1,1,cy,1, atype=atype()), atype(value(param(1,cy,init=gaussian))./ 0.01), f)

mutable struct Deconv4; w; b; f; p; s; end
(c::Deconv4)(x) = deconv4(c.w,x, padding = c.p, stride = c.s, mode = 1) .+ c.b
Deconv4(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int; padding::Int = 0, stride::Int = 2, f=relu) = Deconv4(parameter(w1,w2,w3,cx,cy), param0(1,1,1,cx,1, atype=atype()),f, padding, stride)
Deconv4(w1::Int,w2::Int,cx::Int,cy::Int; padding::Int = 0, stride::Int = 2, f=relu) = Deconv4(parameter(w1,w2,cx,cy),param0(1,1,cx,1, atype=atype()),f, padding, stride)

function AdaIN(input, scale, bias)
    len = length(size(input))
    mu = mean(input, dims=collect(1:len-2))
    variance = var(input, dims=collect(1:len-2))
    sigma = sqrt.(variance .+ 1e-8)
    normalized = (input .- mu) ./ sigma
    broadcastSize = [1 for i in 1:len-2]
    push!(broadcastSize, size(scale,1))
    push!(broadcastSize, size(scale,2))
    scaleBroadcast = reshape(scale, tuple(broadcastSize...))
    biasBroadcast = reshape(bias, tuple(broadcastSize...))
    normalized = normalized .* scaleBroadcast
    normalized = normalized .+ biasBroadcast
    leakyRelu.(normalized)
end

mutable struct InstanceNorm
    scale
    offset
end
function InstanceNorm(nChannels)
    scale = Param(atype((param(1,1,nChannels,1, init = gaussian) .* 20) .+ 1))
    offset = Param(atype(zeros(1,1,nChannels,1)))
    InstanceNorm(scale, offset)
end
function (normLayer::InstanceNorm)(x)
    len = length(size(x))
    mu = mean(x, dims=collect(1:len-2))
    variance = var(x, dims=collect(1:len-2))
    sigma = sqrt.(variance .+ 1e-5)
    normalized = (x .- mu) ./ sigma
    normalized = (normLayer.scale .* normalized) .+ normLayer.offset
    leakyRelu.(normalized)
end


function l2_norm(v, eps=1e-12)
    v ./ (sum(v.^2).^0.5+eps)
end

function spectralNorm(w, u, iteration=1)
    w_shape = size(w)
    w = reshape(w, :, w_shape[end])
#     u = atype(value(param(1,w_shape[end],init=gaussian))./ 0.01)
    u_hat = u
    v_hat = nothing
#     for i in 1:iteration
#         """
#         power iteration
#         Usually iteration = 1 will be enough
#         """
    v_ = u_hat * w'
    v_hat = l2_norm(v_)
    u_ = v_hat * w
    u_hat = l2_norm(u_)
#     end
    a = v_hat * w
    b = u_hat'
    sigma = a * b
    w_norm = w ./ sigma
    w_norm = reshape(w_norm, w_shape)
end

mutable struct ZMapper
    w
    b
    nChannels
    f
end
function ZMapper(zDim, outputChannels; act=relu, stddev=0.2)
    w = parameter(outputChannels*2, zDim)
    b = Param(atype(zeros(outputChannels*2)))
    ZMapper(w,b,outputChannels,act)
end
function (l::ZMapper)(z)
    out = l.f.(l.w * mat(z) .+ l.b)
    scale = out[1:l.nChannels, :]
    bias = out[l.nChannels+1:end, :]
    scale, bias
end

# mutable struct SpectralNorm
#     u
# end
# function SpectralNorm(shape)
#     u = atype(value(param(1,shape,init=gaussian))./ 0.01)
#     SpectralNorm(u)
# end
# function (normLayer::SpectralNorm)(w)
#     w_shape = atype(value(param(1,w_shape[end],init=gaussian))./ 0.01)
#     w = reshape(ww, :, w_shape[end])
#     if normLayer.u == nothing
#         normLayer.u = 
#     end
#     u_hat = normLayer.u
#     v_hat = nothing
    
#     for i = 1:100
#         v_ = u_hat * w'
#         v_hat = l2_norm(v_)
#         u_ = v_hat * w
#         u_hat = l2_norm(u_)
#     end
    
#     a = v_hat * w
#     b = u_hat'
#     sigma = a * b
#     w_norm = w ./ sigma
    
#     normLayer.u = u_hat
#     w_norm = reshape(w_norm, w_shape)
# end

# struct Deconv4_1; w; b; f; end
# Deconv4_1(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv4_1(Param(param(w1,w2,cx,cy,init=gaussian) .* 2),param0(1,1,cx,1),f)
# function (c::Deconv4_1)(x)
#     deconv = c.f.(deconv4(c.w, x, padding = 2, stride = 1, mode = 1) .+ c.b)
#     out = Knet.atype(zeros(size(deconv,1)+1,size(deconv,2)+1, size(deconv,3),size(deconv,4)))
#     out[1:size(deconv,1), 1:size(deconv,2),:,:] .= deconv
#     out
# end

# mutable struct Deconv; w; b; f; end
# (c::Deconv)(x) = c.f.(deconv4(c.w, x, padding = 0, stride = 1, mode = 1) .+ c.b)
# Deconv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv(Param(param(w1,w2,cx,cy, init=gaussian) .* 2),param0(1,1,cx,1),f)
# Deconv(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int,f=relu) = Deconv(Param(param(w1,w2,w3,cx,cy,init=gaussian) .* 2),param0(1,1,1,cx,1),f)

# mutable struct Deconv2; w; b; f; end
# (c::Deconv2)(x) = c.f.(deconv4(c.w, x, padding = 1, stride = 2, mode = 1) .+ c.b)
# Deconv2(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv2(Param(param(w1,w2,cx,cy, init=gaussian) .* 2),param0(1,1,cx,1),f)
# Deconv2(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int,f=relu) = Deconv2(Param(param(w1,w2,w3,cx,cy,init=gaussian) .* 2),param0(1,1,1,cx,1),f)

# mutable struct Deconv3; w; b; f; end
# (c::Deconv3)(x) = c.f.(deconv4(c.w, x, padding = 1, stride = 1, mode = 1) .+ c.b)
# Deconv3(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv3(Param(param(w1,w2,cx,cy,init=gaussian) .* 2),param0(1,1,cx,1),f)

# function Z2(dim1, dim2, dim3, dim4)
#     z = KnetArray{Float32}(undef,dim1, dim2, dim3, dim4)
#     z = (randn!(z)).*0.02
# end

# function Z3(dim1, dim2, dim3, dim4, dim5)
#     z = (rand(dim1, dim2, dim3, dim4, dim5).*2.0).-1.0
#     convert(array_type, z)
# end

# function Z4(dim1, dim2, dim3, dim4, dim5)
#     z = KnetArray{Float32}(undef,dim1, dim2, dim3, dim4, dim5)
#     z = (randn!(z)).*0.02
# end

# function instance_Norm(input)
#     len = length(size(input))
#     mu = mean(input, dims=collect(1:len-2))
#     variance = var(input, dims=collect(1:len-2))
#     sigma = sqrt.(variance .+ 1e-5)
#     normalized = (input .- mu) ./ sigma
# end

# function myLoss(logits)
#     labels = ones(size(logits))
#     labels = atype(labels)
#     sigmoid_cross_entropy_with_logits(logits, labels)
# end