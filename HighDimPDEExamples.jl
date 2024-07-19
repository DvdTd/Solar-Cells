option = 2
using HighDimPDE, Random, Flux, Plots
Random.seed!(1234) # append ! to names of functions that modify their arguments

if option == 1 

    d = 2 # dimension of the problem
    tspan = (0.0,0.5) # time horizon
    x0 = fill(0.,d)  # initial point
    g(x) = exp( -sum(x.^2) ) # initial condition
    μ(x, p, t) = 0.0 # advection coefficients
    σ(x, p, t) = 0.1 # diffusion coefficients
    mc_sample = UniformSampling(fill(-5f-1, d), fill(5f-1, d))
    f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = max(0.0, v_x) * (1 -  max(0.0, v_y)) 
    
    prob = PIDEProblem(μ, σ, x0, tspan, g, f)

    ## Definition of the algorithm
    alg = MLP(mc_sample = mc_sample ) 

    sol = solve(prob, alg, multithreading=true)

    println(sol)


#=
PIDEProblem( μ, σ, x0, tspan, g, f; p, x0_sample, neumann_bc, kw...)

g : initial condition, of the form g(x, p, t).
f : nonlinear function, of the form f(x, y, u(x, t), u(y, t), ∇u(x, t), ∇u(y, t), p, t).
μ : drift function, of the form μ(x, p, t).
σ : diffusion function σ(x, p, t).
x: point where u(x,t) is approximated. Is required even in the case where x0_sample is provided. Determines the dimensionality of the PDE.
tspan: timespan of the problem.
p: the parameter vector.
x0_sample : sampling method for x0. Can be UniformSampling(a,b), NormalSampling(σ_sampling, shifted), or NoSampling (by default). If NoSampling, only solution at the single point x is evaluated.
neumann_bc: if provided, Neumann boundary conditions on the hypercube neumann_bc[1] × neumann_bc[2].

=#
elseif  option == 2
    

    ## Definition of the problem
    d = 2 # dimension of the problem
    dt = 0.1
    tspan = (0.0, 0.5) # time horizon
    x0 = fill(0f0,d)  # initial point
    g(x) = exp.(- sum(x.^2*3, dims=1) ) # initial condition
    μ(x, p, t) = 0.0f0 # advection coefficients
    σ(x, p, t) = 0.1f0 # diffusion coefficients

    δ = fill(5f-1, d)
    # println("delta is ", δ) # delta is Float32[0.5, 0.5]
    x0_sample = UniformSampling(-δ, δ)
    f(x, y, v_x, v_y, ∇v_x, ∇v_y, p, t) = v_x .* (1f0 .- v_y) #+ ∇v_x' .* ∇v_x
    # println("f type ", f)
    prob = PIDEProblem(μ, σ, x0, tspan, g, f; x0_sample = x0_sample)

    ## Definition of the neural network to uses

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
                batch_size = 1000)

    xgrid1 = -δ[1]:1f-2:δ[1] # add points back
    
    if d == 2 # mesh plot
        ygrid1 = -δ[1]:1f-2:δ[1]
        grid = [reshape(vcat(x, y, fill(0f0,d-2)),:,1) for x in xgrid1 for y in ygrid1]
        # are X Y correct way round?
        result = [sol.ufuns[1](grid[i])[1,1] for i in range(1, length(xgrid1)*length(ygrid1))]
        # display(wireframe(xgrid1, ygrid1, result, xlabel="Energy", ylabel="Position", zlabel="n", camera=(30, 50), color=reverse(cgrad(:RdYlBu_11))))
        display(surface(xgrid1, ygrid1, result, xlabel="Energy", ylabel="Position", zlabel="n", camera=(30, 50), color=reverse(cgrad(:RdYlBu_11))))


    else # line plot
        grid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1]
        result = [sol.ufuns[end](x)[1,1] for x in grid]
        display(plot(xgrid1, result))

    end

    


elseif option == 3
    using HighDimPDE, Flux, Plots
    tspan = (0f0, 15f-2)
    dt = 5f-2
    μ(x,p,t) = 0f0
    σ(x,p,t) = 1f-1

    d=10
    δ = fill(25f-2, d)
    x0_sample = UniformSampling(-δ, δ)
    x0 = fill(0f0, d)

    ss0 = 5f-2
    g(x) = Float32((2*π)^(-d/2)) * 
      ss0^(- Float32(d) * 5f-1) * 
      exp.(-5f-1 *sum(x .^2f0 / ss0, dims = 1))

    m(x) = - 5f-1 * sum(x.^2, dims=1)

    # This line is wrong
    f(x, y, u_x, u_y, du_x, du_y, p, t) = (u_x, 0f0) .* (m(y) .- prod(2*δ) * max.(u_x, 0f0) .* m(y))

    prob = PIDEProblem(μ, σ, x0, tspan, g, f, x0_sample=x0_sample)


    hls = d + 50
    nn = Flux.Chain(Dense(d, hls, tanh),
                    Dense(hls, hls, tanh),
                    Dense(hls, 1, x->x^2))

    opt = ADAM(1e-2)
    alg = DeepSplitting(nn, opt=opt, mc_sample=x0_sample)

    sol = solve(prob, alg, dt, verbose=false, abstol=1f-3, maxiters=2000, batch_size=1000)

    println(sol)
    xgrid1 = -δ[1]:5f-3:δ[1]
    xgrid = [reshape(vcat(x, fill(0f0,d-1)),:,1) for x in xgrid1]
    display(plot(xgrid1, [sol.ufuns[end](x) for x in xgrid]))

end
    # println("xgrid \n", xgrid)
    # println("sol \n", [sol.ufuns[end](x) for x in xgrid])
    # display(plot(xgrid1, [sol.ufuns[end](x) for x in xgrid]))