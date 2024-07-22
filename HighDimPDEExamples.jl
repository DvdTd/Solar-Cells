using HighDimPDE, Random, Flux, Plots
Random.seed!(1234) # append ! to names of functions that modify their arguments

## Definition of the problem
numPlots = 5
d = 2 # dimension of the problem
dt = 0.1
tspan = (0.0, 1.0) # time horizon
x0 = fill(0f0,d)  # initial point
batch_size = 1000
μ(x, p, t) = 0.0f0 # advection coefficients
σ(x, p, t) = 0.1f0 # diffusion coefficients
δ = fill(5f-1, d)
x0_sample = UniformSampling(-δ, δ)

initialCoeff = fill(0f0,d)
initialCoeff[1] = 1
initialCoeff[2] = 1
function initialValues(x)
    return exp.(- sum(x.^2 .* initialCoeff, dims=1) ) # not normalised Gaussian, or standard deviation
end

g(x) = initialValues(x) .- initialValues(δ) # initial condition
prod

f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y) .+ (v_x .* ∇v_x)#+ ∇v_x' .* ∇v_x
# println("f type ", f)
prob = PIDEProblem(μ, σ, x0, tspan, g, f; x0_sample = x0_sample)

## Definition of the neural network to use

hls = d + 50 #hidden layer size

nn = Flux.Chain(Dense(d, hls, tanh),
        Dense(hls, hls, tanh),
        Dense(hls, 1)) # neural network used by the scheme

opt = ADAM(1e-2)

## Definition of the algorithm
alg = DeepSplitting(nn,
                    opt = opt,
                    mc_sample = x0_sample)

sol = solve(prob, 
            alg, 
            dt, 
            verbose = true, 
            abstol = 2e-3,
            maxiters = 1000,
            batch_size = batch_size)

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
        # are X Y correct way round?
        local result = [sol.ufuns[plotIndex](grid[i])[1,1] for i in range(1, length(xgrid1)*length(ygrid1))]
        plot = surface(xgrid1, ygrid1, result, xlabel="Energy", ylabel="Position", zlabel="n", camera=(30, 50), color=reverse(cgrad(:RdYlBu_11)))
        title!("Time = " * string((plotIndex-1)*dt)[1:min(4, length(string((plotIndex-1)*dt)))] * "s")
        display(plot)
        

    else # line plot
        grid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1]
        result = [sol.ufuns[plotIndex](x)[1,1] for x in grid]
        display(plot(xgrid1, result))

    end
end