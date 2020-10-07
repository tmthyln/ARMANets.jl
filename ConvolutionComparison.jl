### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f7f80360-fec5-11ea-3de1-57823b2f2f4d
using PlutoUI

# ╔═╡ 28e9e1d0-06ac-11eb-1131-75a155c58fd6
using Flux

# ╔═╡ d790d020-0659-11eb-385a-e5733cc22928
using Flux: unsqueeze, glorot_uniform, expand, calc_padding, convfilter

# ╔═╡ b38f93b0-f898-11ea-3b66-619c8136e64e
using Zygote

# ╔═╡ 90a1f1d0-fcf9-11ea-2e43-6d3cd6829cc9
using CUDA

# ╔═╡ a60809c2-f89d-11ea-0660-57740a973883
using FFTW

# ╔═╡ 0daf00d0-06a7-11eb-3852-ebca5d8bb1f0
using ARMANets

# ╔═╡ daf95700-f899-11ea-0ac9-55a7ffb61020
md"""
# Comparing the Traditional Convolution with ARMA Convolutions
"""

# ╔═╡ e579ab30-fec5-11ea-1e69-bb9e796ffb11
@bind kernel_width Slider(1:10, default=3)

# ╔═╡ 0ed8a440-fec6-11ea-0828-df0a8c784b13
@bind kernel_height Slider(1:10, default=3)

# ╔═╡ 15543000-fec6-11ea-2a32-d743c529c462
@bind matrix_width Slider(kernel_width:20, default=8)

# ╔═╡ 16d37d00-fec6-11ea-1fce-538e0b068217
@bind matrix_height Slider(kernel_height:20, default=8)

# ╔═╡ 384424a0-febf-11ea-2d08-ef06dbcd24fe
padto(ones(Int, kernel_width, kernel_height, 2), (matrix_width, matrix_height, 2))

# ╔═╡ de77151e-fec5-11ea-16b0-4fdb1600fe45
function arma_pass(X, W, A)
	pad_amount = first(size(W)) ÷ 2
	T = conv(X, W, pad=pad_amount)
	ifft(rfft(T, 1:2) ./ rfft(padto(A, size(T)), 1:2), 1:2)
end

# ╔═╡ 6434b980-fec9-11ea-30a3-c9411d1d5d7a
arma_pass(rand(15, 15, 1, 1), rand(3, 3, 1, 1), rand(3, 3, 1, 1))

# ╔═╡ b8675d90-0659-11eb-30a9-1dbf0d91f020
GeneralARMAConv((3, 3), 6)(rand(15, 15, 6, 1))

# ╔═╡ 2da36ae0-06af-11eb-19ea-535b61a511ab
convfilter((3, 3), 3 => 1) 

# ╔═╡ Cell order:
# ╟─f7f80360-fec5-11ea-3de1-57823b2f2f4d
# ╟─daf95700-f899-11ea-0ac9-55a7ffb61020
# ╠═28e9e1d0-06ac-11eb-1131-75a155c58fd6
# ╠═d790d020-0659-11eb-385a-e5733cc22928
# ╠═b38f93b0-f898-11ea-3b66-619c8136e64e
# ╠═90a1f1d0-fcf9-11ea-2e43-6d3cd6829cc9
# ╠═a60809c2-f89d-11ea-0660-57740a973883
# ╠═e579ab30-fec5-11ea-1e69-bb9e796ffb11
# ╠═0ed8a440-fec6-11ea-0828-df0a8c784b13
# ╠═15543000-fec6-11ea-2a32-d743c529c462
# ╠═16d37d00-fec6-11ea-1fce-538e0b068217
# ╠═384424a0-febf-11ea-2d08-ef06dbcd24fe
# ╠═de77151e-fec5-11ea-16b0-4fdb1600fe45
# ╠═6434b980-fec9-11ea-30a3-c9411d1d5d7a
# ╠═0daf00d0-06a7-11eb-3852-ebca5d8bb1f0
# ╠═b8675d90-0659-11eb-30a9-1dbf0d91f020
# ╠═2da36ae0-06af-11eb-19ea-535b61a511ab
