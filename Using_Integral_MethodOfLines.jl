"""
Using MethodOfLines with the drift-diffusion equations and the integral light interaction term.
Problems: requiring Δ argument for integral didn't work physically and couldn't implement n(ϵ) vs n(ϵ+Δ).
"""

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots, Printf, JLD

d = 2 # dimension
kb = 1.3806f-23
eCharge = 1.602f-19
hbar = 1.054571817f-34 / eCharge # J⋅s to  eV⋅s
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.13 # Tress p51 0.05->0.15 eV
Lambda = 9e-5#9e-5 # 9e-6 or 9e-5eV Alexandros
T = 5772 # Tress p63 300K
K = [1/4*gamma^(-3), 3*π/8*gamma^(-4), π*gamma^(-5)][clamp(d-1, 1, 3)]
C = [gamma^(-1), π/2*gamma^(-2), π*gamma^(-3)][clamp(d-1, 1, 3)]
beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
sigmaTilde = sqrt(sigma^2 + 2*Lambda/beta)
# Light interaction Constants
c = 299792458
mu = 1.25663706f-6
EH = -5.2
EL = -3.5
Eav = (EL+EH)/2 # = -4.35
E = 1.7
muϵϵ = 7.5 * 3.33564f-30
A = 1.5
B = 7f-3
N = 5f20
solidAngleCoeff = 1f-5
numPlots = 100
energyRange = [-1.5 + Eav, 1.5 + Eav] # ±infinity but cutoff when it goes to zeros
maxPos = 60.0
positionRange = [-maxPos, maxPos] 
numEnergyPoints = 13
numPositionPoints = 13 # odd is good

deltaRange = [-5f4, 5f4] # 1e2 took 90  mins for 15x15x9
numDeltaPoints = 13

F = [1f5][1]
dt = 1f-10# 5f-4#1f-11
maxTime = 5f-4 #5f-6
cameraTup = (10, 50)#(10,-5)#(85, 60) #  

# Small F parameters
# F = [1f-1][1]
# dt = 1f-1
# maxTime = 10e3
# cameraTup = (40, 55)

@parameters t, ϵ, x, Δ
@variables n(..) lightTerm(..)
Dt = Differential(t)
Dϵ = Differential(ϵ)
Dx = Differential(x)
DΔ = Differential(Δ)
Dxx = Differential(x)^2
Dϵϵ = Differential(ϵ)^2

shouldCalcNew = true # Gives the option to change the plot parameters without recalculating the solution.
jldFilePath = "/Users/david/Documents/Python/Solar Cells/MethodOfLinesData.jld"

step(x, x0) =  (1+sign(x-x0))/2 #sign(x-x0) 1/(1 + exp(-scale*(x-x0)))
Ec(ϵ) = EH + (EL - EH)*step(ϵ, Eav)

if shouldCalcNew || !isfile(jldFilePath)
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean).^2 .* [0.5*width^(-2), 0]))
    IΔ = Integral(Δ in DomainSets.ClosedInterval(deltaRange[1], Δ))
    
    # Interpolate between Gaussians with different means; G(mean1) * (1-sig) + G(mean2) * (sig)
    ECgaussian(x, mean1, mean2, width) = (exp(-(x-mean1)^2*0.5*width^(-2))*(1-step(x,(mean1+mean2)/2)) + exp(-(ϵ-mean2)^2*0.5*width^(-2))*step(x,(mean1+mean2)/2) )
    Ebar(ϵ) = Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-Ec(ϵ)) + sigma^2)

    eq = [
        lightTerm(t, ϵ, x, Δ) ~ muϵϵ^2*mu/(π*c*hbar^2)*1f-5 * (A + B*(2*π*Δ/(c*hbar))^2) * Δ^3/hbar^2 * sign(Δ) * (2*π)^(-0.5)/sigma * ECgaussian(ϵ+Δ, EL, EH, sigma) * ( (1-2*step(ϵ, Eav)) * n(t, ϵ, x, Δ)*(1-n(t, ϵ, x, Δ)) + (n(t, ϵ, x, Δ) - n(t, ϵ, x, Δ))/(exp(Δ*beta*sign(Δ))-1)) * step((Eav-ϵ)*Δ, 0),
        Dt(n(t, ϵ, x, Δ)) ~ IΔ(lightTerm(t, ϵ, x, Δ)) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵ, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(n(t, ϵ, x, Δ)) + K/2 * Dxx(n(t, ϵ, x, Δ)) - C*Ebar(ϵ) * Dϵ(n(t, ϵ, x, Δ)) + C*(Ebar(ϵ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * Dϵϵ(n(t, ϵ, x, Δ)) )
        # Dt(n(t, ϵ, x, Δ)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-0.5) * ECgaussian(ϵ, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(n(t, ϵ, x, Δ)) + K/2 * Dxx(n(t, ϵ, x, Δ)) - C*Ebar(ϵ) * Dϵ(n(t, ϵ, x, Δ)) + C*(Ebar(ϵ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * Dϵϵ(n(t, ϵ, x, Δ)) )
        ]

    # Gaussian at Eav
    (posWidth, energyWidth) = (1, 1) # (1, 3) (1, 0.4)
    initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, Eav, energyWidth) - min(normalGaussian(energyRange[1], Eav, energyWidth), normalGaussian(energyRange[2], Eav, energyWidth))  )
    # 2 Gaussians at EL and EH
    # (posWidth, energyWidth) = (1, 0.4) # (1, 3) (1, 0.4)
    # initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, EH, energyWidth) + normalGaussian(ϵ, EL, energyWidth) - min(normalGaussian(energyRange[1], EH, energyWidth) + normalGaussian(energyRange[1], EL, energyWidth), normalGaussian(energyRange[2], EH, energyWidth) + normalGaussian(energyRange[2], EL, energyWidth))  )
   
    bcs = [
           lightTerm(0.0, ϵ, x, Δ) ~ muϵϵ^2*mu/(π*c*hbar^2)*1f-5 * (A + B*(2*π*Δ/(c*hbar))^2) * abs(Δ^3/hbar^2) * (2*π)^(-0.5)/sigma * ECgaussian(ϵ+Δ, EL, EH, sigma) * ( (1-2*step(ϵ, Eav)) * initialFunc(ϵ, x)*(1-initialFunc(ϵ+Δ, x)) + (initialFunc(ϵ+Δ, x) - initialFunc(ϵ, x))/(exp(abs(Δ*beta))-1)) * step((Eav-ϵ)*Δ, 0),
           lightTerm(t, energyRange[1], x, Δ) ~ 0.0,
           lightTerm(t, energyRange[2], x, Δ) ~ 0.0,
           lightTerm(t, ϵ, positionRange[1], Δ) ~ 0.0, # at edge, n=0 so want dn/dt=0
           lightTerm(t, ϵ, positionRange[2], Δ) ~ 0.0,
           lightTerm(t, ϵ, x, deltaRange[1]) ~ 0.0, # huge jump -> 0
           lightTerm(t, ϵ, x, deltaRange[2]) ~ 0.0,

           n(0.0, ϵ, x, Δ) ~ initialFunc(ϵ, x),
           Dϵ(n(t, energyRange[1], x, Δ)) ~ 0.0,
           Dϵ(n(t, energyRange[2], x, Δ)) ~ 0.0,
           n(t, ϵ, positionRange[1], Δ) ~ 0.0, 
           n(t, ϵ, positionRange[2], Δ) ~ 0.0,
           DΔ(n(t, ϵ, x, deltaRange[1])) ~ 0.0, # n doesn't change with respect to Δ.
           DΔ(n(t, ϵ, x, deltaRange[2])) ~ 0.0] 

    domains = [t ∈ Interval(0.0, maxTime), ϵ ∈ Interval(energyRange[1], energyRange[2]), x ∈ Interval(positionRange[1], positionRange[2]), Δ ∈ Interval(deltaRange[1], deltaRange[2])]
    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵ, x, Δ], [n(t, ϵ, x, Δ), lightTerm(t, ϵ, x, Δ)])
    order = 2
    discretization = MOLFiniteDifference([ϵ => numEnergyPoints, x => numPositionPoints, Δ => numDeltaPoints], t)
    
    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)

    soln = sol[n(t, ϵ, x, Δ)]
    lenSolt = length(sol[t])
    sole = sol[ϵ]
    solx = sol[x]
    sold = sol[Δ]

    # Write to file
    solDict = Dict("soln" => soln, "lenSolt" => lenSolt, "sole" => sole, "solx" => solx, "sold" => sold, "maxTime" => maxTime)
    save(jldFilePath, "data", solDict)

else # Read from file
    solDict = load(jldFilePath)["data"]
    soln = solDict["soln"]
    lenSolt = solDict["lenSolt"]
    sole = solDict["sole"]
    solx = solDict["solx"]
    sold = solDict["sold"]
    maxTime = solDict["maxTime"]

end

# Plot
initialPlot = surface(sole, solx, Surface((sole,solx)->initialFunc(sole, solx), sole, solx), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)))
title!("Initial")
display(initialPlot)
shownPlots = []

for j in 1:numDeltaPoints #[1,2,3,numDeltaPoints] # used to show n is independent of delta - MAYBE WANT IT LARGE SO INTEG OVER WHOLE DOMAIN (ALTHOUGH LIMITS ALREADY BIG)
local zmin = min(soln[:,:,:,j]...)
    local zmax = max(soln[:,:,:,j]...)

    local anim = @animate for i in 1:lenSolt
        # camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
        plot = surface(sole, solx, transpose(soln[i, :, :, j]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(zmin, zmax))
        title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s, Delta = " * Printf.format(Printf.Format("%e"),sold[j]))
        # title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")

        if i in shownPlots
            display(plot)
        end
        zlims!(zmin, zmax)
    end
    display(gif(anim, "drift_diffusion.gif", fps = floor(numPlots/5))) # DONT NEED TO SAVE EACH TIME
end
