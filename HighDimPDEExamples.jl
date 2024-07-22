using HighDimPDE, Random, Plots, SpecialFunctions, Flux
Random.seed!(1234) # append ! to names of functions that modify their arguments

# Constants
d = 2 # dimension
kb = 1.3806e-23
eCharge = 1.602e-19
hbar = 1.05457e-34 / eCharge # J⋅s to  eV⋅s
gamma = 0.788 # Bilayers p8 0.788
# nu0 = 1
# g1 = 1
sigma = 0.1 # Tress p51 0.05->0.15 eV
# Ec = 0 # Traps p5 -5.2eV although just offsets all energy
# Lambda = 9e-2#9e-5 # 9e-6 or 9e-5eV Alexandros
T = 300 # Tress p63 300K
# K = [1/4*gamma^(-3), 3*np.pi/8*gamma^(-4), np.pi*gamma^(-5)][clamp(d-1, 1, 3)]
# C = [gamma^(-1), np.pi/2*gamma^(-2), np.pi*gamma^(-3)][clamp(d-1, 1, 3)]
# beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
# sigmaTilde = np.sqrt(sigma^2 + 2*Lambda/beta)

## Definition of the problem
numPlots = 5
dt = 0.1
tspan = (0.0, 1.0) # time horizon
x0 = fill(0f0,d)  # initial point
batch_size = 1000
μ(x, p, t) = 0.0f0 # advection coefficients - WHAT IS P
σ(x, p, t) = 0.1f0 # diffusion coefficients
δ = fill(5f-1, d)
x0_sample = UniformSampling(-δ, δ)

initialCoeff = fill(0f0,d)
initialCoeff[1] = 1
# initialCoeff[2] = 1
function initialValues(x)
    return exp.(- sum(x.^2 .* initialCoeff, dims=1) ) # not normalised Gaussian, or standard deviation
end

g(x) = initialValues(x) .- initialValues(δ) # initial condition


# Are these coefficients in Joules or eV ???
kbT = kb*T
A1 = 0.5*(sigma^2*kbT^5 + sigma^4 * kbT^3)
A2 = -0.5*(9*(sigma*kbT)^4 + 3*sigma^6 * kbT^2)
A3 = 0.5*(9*sigma^4 * kbT^5 + 18*sigma^6 * kbT^3 + 3*sigma^8 * kbT)
A4 = -0.5*(sigma^10 + 15*sigma^6 * kbT^4  + 10*sigma^8 * kbT^2)
B1 = -sqrt(2)/4 * sigma^3 * kbT^5
B2 = -2*sqrt(2) * (sigma*kbT)^5
B3 = sqrt(2)/2 * sigma^5 * kbT^3
B4 = -(4*sqrt(2) * sigma^5 * kbT^4 + sqrt(2) * sigma^7 * kbT^2)
B5 = sqrt(2)/2 * (sigma^9 * kbT + 9*sigma^7 * kbT^3)
C1 = 0.25*sigma^2 * kbT^5
C2 = 2.25*sigma^4 * kbT^5
D1 = -kbT^5
D2 = 3*sigma^2 * kbT^4
D3 = -3*(sigma^4 * kbT^3 + sigma^2 * kbT^5)
D4 = 3*sigma^4 * kbT^4 + sigma^6 * kbT^2
E1 = sqrt(2)/2 * sigma * kbT^5
E2 = sqrt(2) * sigma^3 * kbT^5
E3 = 2*sqrt(2) * sigma^3 * kbT^4
E4 = -sqrt(2) * sigma^5 * kbT^3
F1 = -0.5*kbT^5
F2 = -1.5*sigma^2 * kbT^5



f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y) .+ (v_x .* ∇v_x)#+ ∇v_x' .* ∇v_x

prob = PIDEProblem(μ, σ, x0, tspan, g, f; x0_sample = x0_sample)

## Neural network
hls = d + 50
nn = Flux.Chain(Dense(d, hls, tanh), Dense(hls, hls, tanh), Dense(hls, hls, tanh), Dense(hls, 1)) # neural network used by the scheme
opt = ADAM(1e-2)
alg = DeepSplitting(nn, opt = opt, mc_sample = x0_sample)
sol = solve(prob, alg, dt, verbose = true, abstol = 2e-3, maxiters = 1000, batch_size = batch_size)

## Plot
xgrid1 = -δ[1]:1f-2:δ[1]
maxTimeIndex = length(sol.ufuns)
numPlots = Int64(min(numPlots, floor(maxTimeIndex)))

if numPlots == 1
    plotIndices = [maxTimeIndex]
else
    plotIndices = Int64.(round.(LinRange(1, maxTimeIndex, numPlots)))
end

for plotIndex in plotIndices
    if d == 2 # mesh plot
        local ygrid1 = -δ[1]:1f-2:δ[1]
        local grid = [reshape(vcat(x, y, fill(0f0,d-2)),:,1) for x in xgrid1 for y in ygrid1]
        local result = [sol.ufuns[plotIndex](grid[i])[1,1] for i in range(1, length(xgrid1)*length(ygrid1))]
        plot = surface(xgrid1, ygrid1, result, xlabel="Energy", ylabel="Position", zlabel="n", camera=(30, 50), color=reverse(cgrad(:RdYlBu_11)))
        title!("Time = " * string((plotIndex-1)*dt)[1:min(4, length(string((plotIndex-1)*dt)))] * "s")
        display(plot)
        
    else # line plot
        local grid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1]
        local result = [sol.ufuns[plotIndex](x)[1,1] for x in grid]
        plot = Plots.plot(xgrid1, result)
        title!("Time = " * string((plotIndex-1)*dt)[1:min(4, length(string((plotIndex-1)*dt)))] * "s")
        display(plot)

    end
end
