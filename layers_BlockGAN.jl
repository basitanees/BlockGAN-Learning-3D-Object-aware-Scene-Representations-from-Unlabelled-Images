function Z(dim1, dim2)
    z = (rand(dim1, dim2).*2.0).-1.0
    convert(array_type, z)
end

function Z_Sample(dim1, dim2)
    z = KnetArray{Float32}(undef,dim1, dim2)
    z = (rand!(z).*2.0).-1.0
end

function leakyRelu(x)
    max(0.2 .* x, x)
end

function Z2(dim1, dim2, dim3, dim4)
    z = KnetArray{Float32}(undef,dim1, dim2, dim3, dim4)
    z = (randn!(z)).*0.02
end

function Z3(dim1, dim2, dim3, dim4, dim5)
    z = (rand(dim1, dim2, dim3, dim4, dim5).*2.0).-1.0
    convert(array_type, z)
end

function Z4(dim1, dim2, dim3, dim4, dim5)
    z = KnetArray{Float32}(undef,dim1, dim2, dim3, dim4, dim5)
    z = (randn!(z)).*0.02
end

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(model::Chain)(x) = (for l in model.layers; x = l(x); end; x)

struct Dense; w; b; f; end
Dense(i::Int,o::Int,f=relu) = Dense(param(o,i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x) .+ d.b) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul

struct Conv; w; b; f; end
(c::Conv)(x) = c.f.(conv4(c.w, x, stride = 2, padding = 2) .+ c.b)
Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=leakyRelu) = Conv(param(w1,w2,cx,cy,init=gaussian), param0(1,1,cy,1), f)

struct Conv_spectral; w; b; f; end
(c::Conv_spectral)(x) = c.f.(conv4(spectralNorm(c.w), x, stride = 2, padding = 2) .+ c.b)
Conv_spectral(w1::Int,w2::Int,cx::Int,cy::Int,f=leakyRelu) = Conv_spectral(param(w1,w2,cx,cy,init=gaussian), param0(1,1,cy,1), f)

struct Conv1; w; b; f; end
(c::Conv1)(x) = c.f.(conv4(c.w, x, stride = 1, padding = 0) .+ c.b)
Conv1(w1::Int,w2::Int,cx::Int,cy::Int,f=leakyRelu) = Conv1(param(w1,w2,cx,cy,init=gaussian), param0(1,1,cy,1), f)

struct Deconv; w; b; f; end
(c::Deconv)(x) = c.f.(deconv4(c.w, x, padding = 0, stride = 1) .+ c.b)
Deconv(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv(param(w1,w2,cx,cy, init=gaussian),param0(1,1,cx,1),f)
Deconv(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int,f=relu) = Deconv(param(w1,w2,w3,cx,cy,init=gaussian),param0(1,1,1,cx,1),f)

struct Deconv2; w; b; f; end
(c::Deconv2)(x) = c.f.(deconv4(c.w, x, padding = 1, stride = 2) .+ c.b)
Deconv2(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv2(param(w1,w2,cx,cy, init=gaussian),param0(1,1,cx,1),f)
Deconv2(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int,f=relu) = Deconv2(param(w1,w2,w3,cx,cy,init=gaussian),param0(1,1,1,cx,1),f)

struct Deconv3; w; b; f; end
(c::Deconv3)(x) = c.f.(deconv4(c.w, x, padding = 2, stride = 1) .+ c.b)
Deconv3(w1::Int,w2::Int,cx::Int,cy::Int,f=relu) = Deconv3(param(w1,w2,cx,cy,init=gaussian),param0(1,1,cx,1),f)

struct Deconv4; w; b; f; o; p; s; end
(c::Deconv4)(x) = c.f.(conv4x(c.w, c.o, x, padding = c.p, stride = c.s) .+ c.b)
# Deconv4(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int, outSize::Int, bSize::Int, p::Int = 1, s::Int = 2, f=relu) = Deconv4(param(w1,w2,w3,cx,cy),f, convert(array_type, similar(zeros(outSize, outSize, outSize, cx, bSize))), p, s)
Deconv4(w1::Int,w2::Int,w3::Int,cx::Int,cy::Int, outSize::Int, bSize::Int, p::Int = 1, s::Int = 2, f=relu) = Deconv4(param(w1,w2,w3,cx,cy),param0(1,1,1,cx,1),f, KnetArray{Float32}(undef,outSize, outSize, outSize, cx, bSize), p, s)

struct Deconv5; w; f; o; p; s; end
(c::Deconv5)(x) = c.f.(conv4x(c.w, c.o, x, padding = c.p, stride = c.s))
Deconv5(w1::Int,w2::Int,cx::Int,cy::Int, outSize::Int, p::Int = 0, s::Int = 1, f=relu) = Deconv5(param(w1,w2,cx,cy),f, convert(array_type, similar(zeros(outSize, outSize, cx, 1))), p, s)

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
end

struct InstanceNorm
    scale
    offset
end
function InstanceNorm(nChannels)
    scale = Param((param(1,1,nChannels,1, init = gaussian) .* 2) .+ 1)
    offset = param0(1,1,nChannels,1)
    InstanceNorm(scale, offset)
end
function (normLayer::InstanceNorm)(x)
    len = length(size(x))
    mu = mean(x, dims=collect(1:len-2))
    variance = var(x, dims=collect(1:len-2))
    sigma = sqrt.(variance .+ 1e-5)
    normalized = (x .- mu) ./ sigma
    (normLayer.scale .* normalized) .+ normLayer.offset
end


function l2_norm(v, eps=1e-12)
    v ./ (sum(v.^2).^0.5+eps)
end

function spectralNorm(w, iteration=1)
    w_shape = size(w)
    w = reshape(w, :, w_shape[end])
    u = value(param(1,w_shape[end],init=gaussian))./ 0.01
    u_hat = u
    v_hat = nothing
    for i in 1:iteration
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = u_hat * w'
        v_hat = l2_norm(v_)
        u_ = v_hat * w
        u_hat = l2_norm(u_)
    end
    a = v_hat * w
    b = u_hat'
    sigma = a * b
    w_norm = w ./ sigma
    w_norm = reshape(w_norm, w_shape)
end


struct ZMapper
    w
    b
    nChannels
    f
end
function ZMapper(zDim, outputChannels; act=relu, stddev=0.02)
    w = Param(param(outputChannels*2, zDim, init=gaussian) .* (stddev/0.01)) # 0.01 is default std for gaussian
    b = param0(outputChannels*2)
    ZMapper(w,b,outputChannels,act)
end
function (l::ZMapper)(z)
    out = l.f.(l.w * mat(z) .+ l.b)
    scale = out[1:l.nChannels, :]
    bias = out[l.nChannels+1:end, :]
    scale, bias
end



# function instance_Norm(input)
#     len = length(size(input))
#     mu = mean(input, dims=collect(1:len-2))
#     variance = var(input, dims=collect(1:len-2))
#     sigma = sqrt.(variance .+ 1e-5)
#     normalized = (input .- mu) ./ sigma
# end