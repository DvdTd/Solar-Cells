using HighDimPDE, Random, Plots, SpecialFunctions, Flux
Random.seed!(1234) # append ! to names of functions that modify their arguments

# Constants - make const declaration?
d = 2 # dimension
kb = 1.3806f-23
eCharge = 1.602f-19
hbar = 1.054571817f-34 / eCharge # J⋅s to  eV⋅s
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.13 # Tress p51 0.05->0.15 eV
Ec = 0 # Traps p5 -5.2eV although just offsets all energy
E = 1.7
Lambda = 9e-2#9e-5 # 9e-6 or 9e-5eV Alexandros
T = 5772 # Tress p63 300K
K = [1/4*gamma^(-3), 3*π/8*gamma^(-4), π*gamma^(-5)][clamp(d-1, 1, 3)]
C = [gamma^(-1), π/2*gamma^(-2), π*gamma^(-3)][clamp(d-1, 1, 3)]
F = [1e3][1] # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
sigmaTilde = sqrt(sigma^2 + 2*Lambda/beta)
# Light interaction Constants
c = 299792458
mu = 1.25663706f-6
EH = -5.2
EL = -3.5
muϵϵ = 7.5 * 3.33564f-30
A = 1.5
B = 7f-3
N = 5f20
solidAngleCoeff = 1f-5

## Definition of the problem
numPlots = 5
dt = 1f-3
tspan = (0.0, 1f-1) # time horizon
x0 = fill(0f0,d)  # initial point
batch_size = 1000


gaussianTerm(x) = nu0*g1*(2*π)^(-0.5)*sigmaTilde^(-2) * exp(-beta * E) * exp(- sum((x .- Ec .- Lambda).^2 .* [0.5*sigmaTilde^(-2), 0])) # dot product keeps x, removes y component
EBar(x) = Lambda * sigmaTilde^(-2) * (2/beta * ( sum(x .* [1,0]) - Ec ) + sigma^2)

μx(x) = -gaussianTerm(x) * C * EBar(x)
μy(x) = gaussianTerm(x) * K * beta/2 * F
σxx(x) = ( gaussianTerm(x) * C * (EBar(x)^2 + 2*Lambda * sigma^2 / beta * sigmaTilde^(-2)) )^0.5
σyy(x) = ( gaussianTerm(x) * K/2 )^0.5

μ(x, p, t) = [μx(x), μy(x)]

# I don't know why this vector works
σ(x, p, t) = [σxx(x), σyy(x)]#SciMLOperators.Diagonal([σxx(x), σyy(y)]) # CARE σ'σ so might sqrt values?
δ = fill(5f-1, d)
x0_sample = UniformSampling(-δ, δ)

initialCoeff = fill(0f0,d)
initialCoeff[1] = 1
initialCoeff[2] = 1
initialValues(x) = exp.(- sum(x.^2 .* initialCoeff, dims=1) )
g(x) = initialValues(x) #.- initialValues(δ) # initial condition

function bandSign(x; threshold=-4.35)
    if (x[1] > threshold) # (sum(x .* [1,0]) > threshold)
        return -1
    else
        return 1
    end
end

f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = fill(0f0, batch_size)#v_x + v_y#v_x .* (1f0 .- v_y) .+ (v_x .* ∇v_x)#+ ∇v_x' .* ∇v_x
# f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = -muϵϵ^2*mu/(π*c*hbar^2)*solidAngleCoeff * 

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

# keep axis scale the same?
for plotIndex in plotIndices
    if d == 2 # mesh plot
        local ygrid1 = -δ[1]:1f-2:δ[1]
        local grid = [reshape(vcat(x, y, fill(0f0,d-2)),:,1) for x in xgrid1 for y in ygrid1]
        local result = [sol.ufuns[plotIndex](grid[i])[1,1] for i in range(1, length(xgrid1)*length(ygrid1))]
        plot = surface(xgrid1, ygrid1, result, xlabel="Energy", ylabel="Position", zlabel="n", camera=(30, 50), color=reverse(cgrad(:RdYlBu_11)))
        title!("Time = " * string((plotIndex-1)*dt)[1:min(6, length(string((plotIndex-1)*dt)))] * "s")
        display(plot)
        
    else # line plot
        local grid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1]
        local result = [sol.ufuns[plotIndex](x)[1,1] for x in grid]
        plot = Plots.plot(xgrid1, result)
        title!("Time = " * string((plotIndex-1)*dt)[1:min(6, length(string((plotIndex-1)*dt)))] * "s")
        display(plot)

    end
end
