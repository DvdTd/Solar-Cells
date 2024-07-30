# # using Pkg
# # Pkg.add("cuDNN")
# # Pkg.add("HighDimPDE")
using HighDimPDE, Plots, Random, Flux

# # # Are these coefficients in Joules or eV ???
# # kbT = kb*T
# # A1 = 0.5*(sigma^2*kbT^5 + sigma^4 * kbT^3)
# # A2 = -0.5*(9*(sigma*kbT)^4 + 3*sigma^6 * kbT^2)
# # A3 = 0.5*(9*sigma^4 * kbT^5 + 18*sigma^6 * kbT^3 + 3*sigma^8 * kbT)
# # A4 = -0.5*(sigma^10 + 15*sigma^6 * kbT^4  + 10*sigma^8 * kbT^2)
# # B1 = -sqrt(2)/4 * sigma^3 * kbT^5
# # B2 = -2*sqrt(2) * (sigma*kbT)^5
# # B3 = sqrt(2)/2 * sigma^5 * kbT^3
# # B4 = -(4*sqrt(2) * sigma^5 * kbT^4 + sqrt(2) * sigma^7 * kbT^2)
# # B5 = sqrt(2)/2 * (sigma^9 * kbT + 9*sigma^7 * kbT^3)
# # C1 = 0.25*sigma^2 * kbT^5
# # C2 = 2.25*sigma^4 * kbT^5
# # D1 = -kbT^5
# # D2 = 3*sigma^2 * kbT^4
# # D3 = -3*(sigma^4 * kbT^3 + sigma^2 * kbT^5)
# # D4 = 3*sigma^4 * kbT^4 + sigma^6 * kbT^2
# # E1 = sqrt(2)/2 * sigma * kbT^5
# # E2 = sqrt(2) * sigma^3 * kbT^5
# # E3 = 2*sqrt(2) * sigma^3 * kbT^4
# # E4 = -sqrt(2) * sigma^5 * kbT^3
# # F1 = -0.5*kbT^5
# # F2 = -1.5*sigma^2 * kbT^5



# # d = 100
# # x0 = repeat([1.0f0, 0.5f0], div(d, 2))
# # tspan = (0.0f0, 1.0f0)
# # r = 0.05f0
# # sigma = 0.4f0
# # # f(X, u, σᵀ∇u, p, t) = r * (u - sum(X .* σᵀ∇u))
# # f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = r * (v_x - sum(x .* ∇v_x))
# # g(x) = sum(x .^ 2)
# # μ_f(x, p, t) = zero(x) 
# # σ_f(x, p, t) = Diagonal(sigma * x) 
# # δ = fill(5f-1, d)
# # x0_sample = UniformSampling(-δ, δ)
# # prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f)

# if 1 == 0
#     d = 10 # dimension of the problem
#     tspan = (0.0, 5.0e1) # time horizon
#     x0 = fill(0.0f0, d)  # initial point
#     g(x) = exp.(-sum(x .^ 2, dims = 1)) # initial condition
#     μ(x, p, t) = 0.0f0 # advection coefficients
#     σ(x, p, t) = 0.1f0 # diffusion coefficients
#     x0_sample = UniformSampling(fill(-5.0f-1, d), fill(5.0f-1, d))
#     f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1.0f0 .- v_y)
#     prob = PIDEProblem(μ, σ, x0, tspan, g, f;
#         x0_sample = x0_sample)

#     ## Definition of the neural network to use
#     hls = d + 50 #hidden layer size

#     nn = Flux.Chain(Dense(d, hls, tanh), Dense(hls, hls, tanh), Dense(hls, 1)) # neural network used by the scheme

#     opt = ADAM(1e-2)

#     ## Definition of the algorithm
#     alg = DeepSplitting(nn, opt = opt, mc_sample = x0_sample)

#     sol = solve(prob, alg, 0.1, verbose = true, abstol = 2e-3, maxiters = 1000, batch_size = 1000)
# end
# # println([1,2,3] - [2,2,4])
# # println([1,2,3] .- 2)
# # println( sum([1,2] .* [1,0]))
# # println(typeof([1 2; 3 4]))

# # μ_f(X, p, t) = zero(X) #Vector d x 1
# # σ_f(X, p, t) = SciMLOperators.Diagonal([5,6])
# # println(typeof(μ_f([1,2,3],2,3)))
# # println(typeof(σ_f(1,2,3)))

# # println(sin(3/2*π))
# Random.seed!(1234)
# m = 2
# n = 5
# arr = randn(2,5)

# for a in arr
#     println(a, "   ")
# end
# println("sum ", sum(arr))
# println("dim1 ", length(sum(arr, dims = 1)), " ", sum(arr, dims = 1))
# println("dim2 ", length(sum(arr, dims = 2)), " ", sum(arr, dims = 2))
# # sum, dims tells you which dim you sum along, dim(result) = other dimension

# println([1,2] * (3 + 2))
# a = [1 2 3]
# b = a[1,:]
# println(size(b), "    ", b)

δ = fill(1f0, d)
x0_sample = UniformSampling(-δ, δ)

d = 100 # number of dimensions
X0 = repeat([1.0f0, 0.5f0], div(d, 2)) # initial value of stochastic state
tspan = (0.0f0, 1.0f0)
r = 0.05f0
sigma = 0.4f0
f(X, u, σᵀ∇u, p, t) = r * (u - sum(X .* σᵀ∇u))
g(X) = sum(X .^ 2)
μ_f(X, p, t) = zero(X) #Vector d x 1
σ_f(X, p, t) = Diagonal(sigma * X) #Matrix d x d
prob = PIDEProblem(μ_f, σ_f, X0, tspan, g, f)

hls = 10 + d #hide layer size
opt = Flux.Optimise.Adam(0.001)
u0 = Flux.Chain(Dense(d, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, 1))
σᵀ∇u = Flux.Chain(Dense(d + 1, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, d))
pdealg = HighDimPDE.DeepSplitting(u0, opt = opt, mc_sample = x0_sample)

ans = solve(prob, pdealg, verbose = true, maxiters = 150, dt = 0.2, pabstol = 1.0f-6)

# hls = d + 50
# nn = Flux.Chain(Dense(d, hls, tanh), Dense(hls, hls, tanh), Dense(hls, hls, tanh), Dense(hls, 1)) 
# opt = ADAM(1e-2)
# alg = DeepSplitting(nn, opt = opt, mc_sample = x0_sample)
# sol = solve(prob, alg, dt, verbose = true, abstol = 1e-4, maxiters = 1000, batch_size = batch_size)
    