using FFTW
using Flux
using Flux: unsqueeze, glorot_uniform, expand, calc_padding, convfilter, @functor

function padto(A, final_size)
	@assert ndims(A) == length(final_size)
	A_type = eltype(A)
	A_size = size(A)[1:2]
	c_size = size(A)[3:end]

	top, left = (final_size[1:2] .- A_size) .÷ 2
	bottom, right = final_size[1:2] .- (top, left) .- A_size

	[zeros(A_type, top, final_size[2], c_size...);
	zeros(A_type, A_size[1], left, c_size...) A zeros(A_type, A_size[1], right, c_size...);
	zeros(A_type, bottom, final_size[2], c_size...)]
end

struct GeneralARMAConv{N,M,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end

function GeneralARMAConv(
        w::AbstractArray{T,N}, b::Union{Flux.Zeros, AbstractVector{T}}, σ = identity;
        stride = 1, pad = 0, dilation = 1) where {T,N}

    stride = expand(Val(N-2), stride)
    dilation = expand(Val(N-2), dilation)
    pad = calc_padding(pad, size(w)[1:N-2], dilation, stride)
    return GeneralARMAConv(σ, w, b, stride, pad, dilation)
end

function GeneralARMAConv(;weight::AbstractArray{T,N}, bias::Union{Flux.Zeros, AbstractVector{T}},
          activation = identity, stride = 1, pad = 0, dilation = 1) where {T,N}
    GeneralARMAConv(weight, bias, activation, stride = stride, pad = pad, dilation = dilation)
end

function GeneralARMAConv(
        k::NTuple{N,Integer}, ch::Integer, σ = identity;
        init = glorot_uniform,  stride = 1, pad = 0, dilation = 1,
        weight = convfilter(k, ch => 1, init = init), bias = Flux.Zeros()) where N

    GeneralARMAConv(weight, bias, σ,
        stride = stride, pad = pad, dilation = dilation)
end

@functor GeneralARMAConv

function (c::GeneralARMAConv)(x::AbstractArray)
    if size(c.weight, 3) != size(x, 3)  # TODO: using 3 may be too specific here?
        throw(DimensionMismatch("Input channels must match! ($(size(c.weight, 3)) vs. $(size(x, 3))"))
    end

    real.(ifft(rfft(x, 1:2) ./ rfft(padto(c.weight, size(x)), 1:2), 1:2))
end
