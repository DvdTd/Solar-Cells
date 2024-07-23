# using Pkg
# Pkg.add("cuDNN")
# Pkg.add("HighDimPDE")
using HighDimPDE, Plots, Random, Flux

# # Are these coefficients in Joules or eV ???
# kbT = kb*T
# A1 = 0.5*(sigma^2*kbT^5 + sigma^4 * kbT^3)
# A2 = -0.5*(9*(sigma*kbT)^4 + 3*sigma^6 * kbT^2)
# A3 = 0.5*(9*sigma^4 * kbT^5 + 18*sigma^6 * kbT^3 + 3*sigma^8 * kbT)
# A4 = -0.5*(sigma^10 + 15*sigma^6 * kbT^4  + 10*sigma^8 * kbT^2)
# B1 = -sqrt(2)/4 * sigma^3 * kbT^5
# B2 = -2*sqrt(2) * (sigma*kbT)^5
# B3 = sqrt(2)/2 * sigma^5 * kbT^3
# B4 = -(4*sqrt(2) * sigma^5 * kbT^4 + sqrt(2) * sigma^7 * kbT^2)
# B5 = sqrt(2)/2 * (sigma^9 * kbT + 9*sigma^7 * kbT^3)
# C1 = 0.25*sigma^2 * kbT^5
# C2 = 2.25*sigma^4 * kbT^5
# D1 = -kbT^5
# D2 = 3*sigma^2 * kbT^4
# D3 = -3*(sigma^4 * kbT^3 + sigma^2 * kbT^5)
# D4 = 3*sigma^4 * kbT^4 + sigma^6 * kbT^2
# E1 = sqrt(2)/2 * sigma * kbT^5
# E2 = sqrt(2) * sigma^3 * kbT^5
# E3 = 2*sqrt(2) * sigma^3 * kbT^4
# E4 = -sqrt(2) * sigma^5 * kbT^3
# F1 = -0.5*kbT^5
# F2 = -1.5*sigma^2 * kbT^5



# d = 100
# x0 = repeat([1.0f0, 0.5f0], div(d, 2))
# tspan = (0.0f0, 1.0f0)
# r = 0.05f0
# sigma = 0.4f0
# # f(X, u, σᵀ∇u, p, t) = r * (u - sum(X .* σᵀ∇u))
# f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = r * (v_x - sum(x .* ∇v_x))
# g(x) = sum(x .^ 2)
# μ_f(x, p, t) = zero(x) 
# σ_f(x, p, t) = Diagonal(sigma * x) 
# δ = fill(5f-1, d)
# x0_sample = UniformSampling(-δ, δ)
# prob = PIDEProblem(μ_f, σ_f, x0, tspan, g, f)

# hls = 10 + d
# opt = Flux.Optimise.Adam(0.001)
# u0 = Flux.Chain(Dense(d, hls, relu),
#     Dense(hls, hls, relu),
#     Dense(hls, 1))
# σᵀ∇u = Flux.Chain(Dense(d + 1, hls, relu),
#     Dense(hls, hls, relu),
#     Dense(hls, hls, relu),
#     Dense(hls, d))

# alg = DeepSplitting(u0, opt = opt, mc_sample = x0_sample)
# ans = solve(prob, alg, dt = 0.2, verbose = true, pabstol = 1.0f-6, maxiters = 150, batch_size = 500)


# nn = Flux.Chain(Dense(d, hls, tanh), Dense(hls, hls, tanh), Dense(hls, hls, tanh), Dense(hls, 1)) # neural network used by the scheme
# opt = ADAM(1e-2)
# alg = DeepSplitting(nn, opt = opt, mc_sample = x0_sample)
# sol = solve(prob, alg, dt, verbose = true, abstol = 2e-3, maxiters = 1000, batch_size = batch_size)

println([1,2,3] - [2,2,4])
println([1,2,3] .- 2)
println( sum([1,2] .* [1,0]))
println(typeof([1 2; 3 4]))

μ_f(X, p, t) = zero(X) #Vector d x 1
σ_f(X, p, t) = SciMLOperators.Diagonal([5,6])
println(typeof(μ_f([1,2,3],2,3)))
println(typeof(σ_f(1,2,3)))

println(sin(π))