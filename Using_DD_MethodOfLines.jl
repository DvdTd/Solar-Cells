"""
Using MethodOfLines with the drift-diffusion equations with Ec=0 in Ebar and the Gaussian (around line 64, line 74/75)
"""

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots, Printf, JLD

d = 2 # dimension
kb = 1.3806f-23
eCharge = 1.602f-19
hbar = 1.054571817f-34 / eCharge
gamma = 1f8 
nu0 = 1 
g1 = 1 
sigma = 0.13 
Lambda = 9e-5
T = 5772 
K = [1/4*gamma^(-3), 3*π/8*gamma^(-4), π*gamma^(-5)][clamp(d-1, 1, 3)]
C = [gamma^(-1), π/2*gamma^(-2), π*gamma^(-3)][clamp(d-1, 1, 3)]
beta = 1/(kb*T) * eCharge 
sigmaTilde = sqrt(sigma^2 + 2*Lambda/beta)
EH = -5.2
EL = -3.5
Eav = (EL+EH)/2 # = -4.35
E = 1.7
numPlots = 100
energyRange = [-1.5 + Eav, 1.5 + Eav] 
maxPos = 8.0
positionRange = [-maxPos, maxPos] 
numEnergyPoints = 40
numPositionPoints = 60

# gamma = 0.788
# F = 1f5
# dt = 1f-10# 5f-4#1f-11
# maxTime = 2f-3 #5f-6

# Small F parameters
# F = 1f-1
# dt = 1f-1
# maxTime = 10e3

# New gamma
F = 1f5/eCharge
dt = 1f-6# 5f-4#1f-11
maxTime = 2f2 #5f-6

# camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  
# cameraTup = (10, 80)
cameraTup = (70, 50)
# cameraTup = (25, 50)

@parameters t, ϵ, x
@variables n(..)

Dt = Differential(t)
Dϵ = Differential(ϵ)
Dx = Differential(x)
Dxx = Differential(x)^2
Dϵϵ = Differential(ϵ)^2

shouldCalcNew = true # Gives the option to change the plot parameters without recalculating the solution.
jldFilePath = "/Users/david/Documents/Python/Solar Cells/MethodOfLinesData.jld"

step(x, x0) =  (1+sign(x-x0))/2
piecewiseE = false
if piecewiseE
    Ec(ϵ) = EH + (EL - EH)*step(ϵ, Eav)
else
    Ec(ϵ) = EH 
end

if shouldCalcNew || !isfile(jldFilePath)
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean).^2 .* [0.5*width^(-2), 0]))
    
    # Interpolate between Gaussians with different means; G(mean1) * (1-sig) + G(mean2) * (sig)
    ECgaussian(x, mean1, mean2, width) = (exp(-(x-mean1)^2*0.5*width^(-2))*(1-step(x,(mean1+mean2)/2)) + exp(-(ϵ-mean2)^2*0.5*width^(-2))*step(x,(mean1+mean2)/2) )
    Ebar(ϵ) = Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-Ec(ϵ)) + sigma^2)

    if piecewiseE
        eq = [
            Dt(n(t, ϵ, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-0.5) * ECgaussian(ϵ, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(n(t, ϵ, x)) + K/2 * Dxx(n(t, ϵ, x)) - C*Ebar(ϵ) * Dϵ(n(t, ϵ, x)) + C*(Ebar(ϵ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * Dϵϵ(n(t, ϵ, x)) )
            ]
    else
        eq = [
            Dt(n(t, ϵ, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-0.5) * ECgaussian(ϵ, EH + Lambda, EH + Lambda, sigmaTilde) * (  K*beta/2*F * Dx(n(t, ϵ, x)) + K/2 * Dxx(n(t, ϵ, x)) - C*Ebar(ϵ) * Dϵ(n(t, ϵ, x)) + C*(Ebar(ϵ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * Dϵϵ(n(t, ϵ, x)) )
            ]
    end

    # Choice of 2 initial conditions:

    # Energy Gaussian at Eav
    (posWidth, energyWidth) = (1, 1)
    initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth))# * (  normalGaussian(ϵ, Eav, energyWidth) - min(normalGaussian(energyRange[1], Eav, energyWidth), normalGaussian(energyRange[2], Eav, energyWidth))  )
    
    # 2 Energy Gaussians at EL and EH
    # (posWidth, energyWidth) = (1, 0.4)
    # initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, EH, energyWidth) + normalGaussian(ϵ, EL, energyWidth) - min(normalGaussian(energyRange[1], EH, energyWidth) + normalGaussian(energyRange[1], EL, energyWidth), normalGaussian(energyRange[2], EH, energyWidth) + normalGaussian(energyRange[2], EL, energyWidth))  )
   
    bcs = [
           n(0.0, ϵ, x) ~ initialFunc(ϵ, x),
           Dϵ(n(t, energyRange[1], x)) ~ 0.0,
           Dϵ(n(t, energyRange[2], x)) ~ 0.0,
           n(t, ϵ, positionRange[1]) ~ 0.0, 
           n(t, ϵ, positionRange[2]) ~ 0.0,
           ] 

    # Solve the equations.
    domains = [t ∈ Interval(0.0, maxTime), ϵ ∈ Interval(energyRange[1], energyRange[2]), x ∈ Interval(positionRange[1], positionRange[2])]
    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵ, x], [n(t, ϵ, x)])
    discretization = MOLFiniteDifference([ϵ => numEnergyPoints, x => numPositionPoints], t)
    
    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)

    # From sol, get solution and t, x and ϵ values.
    soln = sol[n(t, ϵ, x)]
    lenSolt = length(sol[t])
    sole = sol[ϵ]
    solx = sol[x]


    # Write to file
    solDict = Dict("soln" => soln, "lenSolt" => lenSolt, "sole" => sole, "solx" => solx, "maxTime" => maxTime)
    save(jldFilePath, "data", solDict)

# Read from file instead of recalculating.
else 
    solDict = load(jldFilePath)["data"]
    soln = solDict["soln"]
    lenSolt = solDict["lenSolt"]
    sole = solDict["sole"]
    solx = solDict["solx"]
    maxTime = solDict["maxTime"]

end


# Plot

# Plot initial condition
# initialPlot = surface(sole, solx, Surface((sole,solx)->initialFunc(sole, solx), sole, solx), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)))
# title!("Initial")
# display(initialPlot)

# Show individual plots e.g. [1,2,3,20] or []
shownPlots = [80, 90, 99]

zmin = min(soln[:,:,:]...)
zmax = max(soln[:,:,:]...)

# Make gif.
anim = @animate for i in 1:lenSolt
    plot = surface(sole, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(zmin, zmax))
    title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")

    # Plot individual plots.
    if i in shownPlots
        display(plot)
    end
    zlims!(zmin, zmax)
end
display(gif(anim, "drift_diffusion.gif", fps = floor(numPlots/5)))

