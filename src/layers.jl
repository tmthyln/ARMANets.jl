using FFTW
using Flux
using Tullio

function bounded_uniform(rng::AbstractRNG, init, dims...)
	bound = -log(1 - init)
	return 2bound .* rand(rng, Float32, dims...) .- bound
end
bounded_uniform(init, dims...) = bounded_uniform(Random.GLOBAL_RNG, init, dims...)
bounded_uniform(rng::AbstractRNG, init) = (dims...) -> bounded_uniform(rng, init, dims...)

function padto(A, final_size)
	@assert ndims(A) == length(final_size)
	A_type = eltype(A)
	A_size = size(A)[1:2]
	c_size = size(A)[3:end]

	bottom, right = cld.(final_size[1:2] .- A_size, 2) .- 1
	top, left = final_size[1:2] .- (bottom, right) .- A_size

	vcat(zeros(A_type, top, final_size[2], c_size...),
	hcat(zeros(A_type, A_size[1], left, c_size...), A, zeros(A_type, A_size[1], right, c_size...)),
	zeros(A_type, bottom, final_size[2], c_size...))
end

struct ARConv{F,A,W,C}
    filters::Int
	σ::F
	alpha::A
	weight::W
    center::C
end

function ARConv(filters::Int, channels, σ=identity;
    	init=bounded_uniform(Random.GLOBAL_RNG, 0.1), dilation=1)

	alpha = init(2, 2, channels, filters ÷ 2)

    weight = zeros(2, 2dilation + 1)
    weight[1, 1] = cos(-π / 4)
    weight[2, 1] = sin(-π / 4)
    weight[1, size(weight, 2)] = -sin(-π / 4)
    weight[2, size(weight, 2)] = cos(-π / 4)

    center = zeros(1, 2dilation + 1)
    center[1, cld(size(center, 2), 2)] = 1

	ARConv(filters, σ, alpha, weight, center)
end

Flux.@functor ARConv
Flux.trainable(c::ARConv) = (c.alpha,)

function Base.show(io::IO, c::ARConv)
    components = String[]

    push!(components, string("(", c.filters, ", ", c.filters, ")"))

    push!(components, string(size(c.alpha, 3)))

    c.σ != identity && push!(components, string(c.σ))

    dilation = length(c.center) ÷ 2
    dilation != 1 && push!(components, string("dilation=", dilation))

    print(io, "ARConv(")
    print(io, join(components, ", "))
    print(io, ")")
end

function (c::ARConv)(x::AbstractArray)
	alpha = tanh.(c.alpha)
	weight = c.weight
    center = c.center

	# size: [2, 2, c, b] x [2, 3] -> [2, 3, c, b]
	@tullio Axy[i, k, c, b] := alpha[i, j, c, b] * weight[j, k]
    Axy = Axy .+ center

	# size: [2, 3, c, b] -> [3, c, b], [3, c, b]
	Ax, Ay = collect(eachslice(Axy, dims=1))

	# size: [3, c, b] x [3, c, b] -> [3, 3, c, b]
	@tullio A[i, j, c, b] := Ax[i, c, b] * Ay[j, c, b]

	Apad = fftshift(padto(A, size(x)), 1:2)

	fft_convolve(x, Apad)
end

function fft_convolve(x, a)
    X = fft(x, 1:2)
	A = prod(fft(a, 1:2), dims=4)

	@tullio Y[i1, i2, c, b] := X[i1, i2, c, b] / A[i1, i2, c, x]

	return real.(ifft(Y, 1:2))
end