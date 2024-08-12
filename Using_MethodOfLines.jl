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
numPlots = 50
maxPos = 60.0
positionRange = [-maxPos, maxPos] 
numPositionPoints = 7 # odd is good

energy_tail_length = 2
energyRangeA = [EH - energy_tail_length, EH]
energyRangeB = [EH, EL]
energyRangeC = [EL, EL + energy_tail_length]
numEnergyPointsAC = 9
numEnergyPointsB = 9

F = [1f5][1]
dt = 1f-10# 5f-4#1f-11
maxTime = 1f-3 #5f-6
cameraTup = (10, 50)#(10,-5)#(85, 60) #  

# Small F parameters
# F = [0f-1][1]
# dt = 1f-4
# maxTime = 3e3
# cameraTup = (40, 55)

@parameters t, ϵA, ϵB, ϵC, x
@variables nA(..) nB(..) nC(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

DϵA = Differential(ϵA)
DϵϵA = Differential(ϵA)^2
DϵB = Differential(ϵB)
DϵϵB = Differential(ϵB)^2
DϵC = Differential(ϵC)
DϵϵC = Differential(ϵC)^2

shouldCalcNew = true # Gives the option to change the plot parameters without recalculating the solution.
jldFilePath = "/Users/david/Documents/Python/Solar Cells/MethodOfLinesData.jld"

if shouldCalcNew || !isfile(jldFilePath)
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean).^2 .* [0.5*width^(-2), 0]))

    step(x, x0) =  (1+sign(x-x0))/2
    Ec(ϵ) = EH + (EL - EH)*step(ϵ, Eav)
    ECgaussian(x, mean1, mean2, width) = (exp(-(x-mean1)^2*0.5*width^(-2)) * (1-step(x,(mean1+mean2)/2)) + exp(-(x-mean2)^2*0.5*width^(-2)) * step(x,(mean1+mean2)/2) ) #XXX only used once so remove mean args?
    piecewiseN(ϵ, n) = n(t, EL, x) * (1-step(ϵ,Eav)) + n(t, EH, x) * (step(ϵ,Eav))
    piecewiseDN(ϵ, n, Dϵϵ) = Dϵϵ(n(t, EL, x)) * (1-step(ϵ,Eav)) + Dϵϵ(n(t, EH, x)) * (step(ϵ,Eav))

    # Interpolate between Gaussians with different means; G(mean1) * (1-sig) + G(mean2) * (sig)
    Ebar(ϵ) = Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-Ec(ϵ)) + sigma^2)

    eq = [
        Dt(nA(t, ϵA, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵA, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nA(t, ϵA, x)) + K/2 * Dxx(nA(t, ϵA, x)) - C*Ebar(ϵA) * DϵA(nA(t, ϵA, x)) + C*(Ebar(ϵA)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵA(nA(t, ϵA, x)) ),
        Dt(nB(t, ϵB, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵB, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nB(t, ϵB, x)) + K/2 * Dxx(nB(t, ϵB, x)) - C*Ebar(ϵB) * DϵB(nB(t, ϵB, x)) + C*(Ebar(ϵB)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵB(nB(t, ϵB, x)) ),
        Dt(nC(t, ϵC, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵC, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nC(t, ϵC, x)) + K/2 * Dxx(nC(t, ϵC, x)) - C*Ebar(ϵC) * DϵC(nC(t, ϵC, x)) + C*(Ebar(ϵC)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵC(nC(t, ϵC, x)) ),
        ]

    # Gaussian at Eav
    (posWidth, energyWidth) = (1, 1) # (1, 3) (1, 0.4)
    initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, Eav, energyWidth) - min(normalGaussian(energyRangeA[1], Eav, energyWidth), normalGaussian(energyRangeC[2], Eav, energyWidth))  )
    # 2 Gaussians at EL and EH
    # (posWidth, energyWidth) = (1, 0.4) # (1, 3) (1, 0.4)
    # initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, EH, energyWidth) + normalGaussian(ϵ, EL, energyWidth) - min(normalGaussian(energyRange[1], EH, energyWidth) + normalGaussian(energyRange[1], EL, energyWidth), normalGaussian(energyRange[2], EH, energyWidth) + normalGaussian(energyRange[2], EL, energyWidth))  )
   
    bcs = [
        nA(0.0, ϵA, x) ~ initialFunc(ϵA, x),
        nA(t, ϵA, positionRange[1]) ~ 0.0, 
        nA(t, ϵA, positionRange[2]) ~ 0.0,
        DϵA(nA(t, energyRangeA[1], x)) ~ 0.0,

        nB(0.0, ϵB, x) ~ initialFunc(ϵB, x),
        nB(t, ϵB, positionRange[1]) ~ 0.0, 
        nB(t, ϵB, positionRange[2]) ~ 0.0,
        
        nC(0.0, ϵC, x) ~ initialFunc(ϵC, x),
        nC(t, ϵC, positionRange[1]) ~ 0.0, 
        nC(t, ϵC, positionRange[2]) ~ 0.0,
        DϵC(nC(t, energyRangeC[2], x)) ~ 0.0, 

        # Interface
        nB(t, energyRangeB[1], x) ~ nA(t, energyRangeA[2], x), 
        DϵB(nB(t, energyRangeB[1], x)) ~ DϵA(nA(t, energyRangeA[2], x)),
        nB(t, energyRangeB[2], x) ~ nC(t, energyRangeC[1], x), 
        DϵB(nB(t, energyRangeB[2], x)) ~ DϵC(nC(t, energyRangeC[1], x)),
        ] 

    domains = [t ∈ Interval(0.0, maxTime), ϵA ∈ Interval(energyRangeA[1], energyRangeA[2]), ϵB ∈ Interval(energyRangeB[1], energyRangeB[2]), ϵC ∈ Interval(energyRangeC[1], energyRangeC[2]), x ∈ Interval(positionRange[1], positionRange[2])]
    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵA, ϵB, ϵC, x], [nA(t, ϵA, x), nB(t, ϵB, x), nC(t, ϵC, x)])
    order = 2
    discretization = MOLFiniteDifference([ϵA => numEnergyPointsAC, ϵB => numEnergyPointsB, ϵC => numEnergyPointsAC, x => numPositionPoints], t)
    
    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)

    soln = hcat(sol[nA(t, ϵA, x)], sol[nB(t, ϵB, x)], sol[nC(t, ϵC, x)])
    lenSolt = length(sol[t])
    sole = [sol[ϵA]; sol[ϵB]; sol[ϵC]]
    solx = sol[x]

    # Write to file
    solDict = Dict("soln" => soln, "lenSolt" => lenSolt, "sole" => sole, "solx" => solx, "maxTime" => maxTime)
    save(jldFilePath, "data", solDict)

else # Read from file
    solDict = load(jldFilePath)["data"]
    soln = solDict["soln"]
    lenSolt = solDict["lenSolt"]
    sole = solDict["sole"]
    solx = solDict["solx"]
    maxTime = solDict["maxTime"]

end

# Plot
# initialPlot = surface(sole, solx, Surface((sole,solx)->initialFunc(sole, solx), sole, solx), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)))
# title!("Initial")
# display(initialPlot)
shownPlots = []
zmin = min(soln[:,:,:]...)
zmax = max(soln[:,:,:]...)

anim = @animate for i in 1:lenSolt
    
    # camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
    plot = surface(sole, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(zmin, zmax))

    title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")

    if i in shownPlots
        display(plot)
    end
    zlims!(zmin, zmax)
end
display(gif(anim, "drift_diffusion.gif", fps = floor(numPlots/5)))

