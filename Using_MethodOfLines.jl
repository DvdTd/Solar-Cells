using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots, Printf, JLD

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
muϵϵ = 7.5 * 3.33564f-30
A = 1.5
B = 7f-3
N = 5f20
solidAngleCoeff = 1f-5
numPlots = 50
energyRange = [-1.0, 1.0] # ±infinity but cutoff when it goes to zeros
maxPos = 60.0
positionRange = [-maxPos, maxPos] 
numEnergyPoints = 20
numPositionPoints = 20

F = [1f5][1]
dt = 1f-6# 5f-4#1f-11
maxTime = 5f-3 #5f-6
cameraTup = (80, 50)

# Small F parameters
# F = [1f-1][1]
# dt = 1f-1
# maxTime = 1e3
# cameraTup = (40, 55)

@parameters t, ϵ, x
@variables n(..) Ebar(..)
Dt = Differential(t)
Dϵ = Differential(ϵ)
Dx = Differential(x)

shouldCalcNew = true # Option to save time if changed plot parameters.
jldFilePath = "/Users/david/Documents/Python/Solar Cells/MethodOfLinesData.jld"

if shouldCalcNew || !isfile(jldFilePath)
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean).^2 .* [0.5*width^(-2), 0]))
    Ix = Integral(x in DomainSets.ClosedInterval(positionRange[1], x)) # basically cumulative sum from 0 to x

    eq = [
        Ebar(t, ϵ, x) ~ Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-Ec) + sigma^2),
        Dt(n(t, ϵ, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-0.5) * exp(-(ϵ-Ec-Lambda)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(n(t, ϵ, x)) + K/2 * Dx(Dx(n(t, ϵ, x))) - C*Ebar(t, ϵ, x) * Dϵ(n(t, ϵ, x)) + C*(Ebar(t, ϵ, x)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * Dϵ(Dϵ(n(t, ϵ, x))) )
    ]

    initialFunc(ϵ, x) = (normalGaussian(x, 0, 1) - normalGaussian(positionRange[1], 0, 1)) * (normalGaussian(ϵ, 0, 3) - normalGaussian(energyRange[1], 0, 3))
    bcs = [n(0.0, ϵ, x) ~ initialFunc(ϵ, x),
        #   Dx(n(t, energyRange[1], x)) ~ 0.0,
        Dϵ(n(t, energyRange[2], x)) ~ 0.0,
        n(t, ϵ, positionRange[1]) ~ 0.0, 
        n(t, ϵ, positionRange[2]) ~ 0.0] 

    domains = [t ∈ Interval(0.0, maxTime), ϵ ∈ Interval(energyRange[1], energyRange[2]), x ∈ Interval(positionRange[1], positionRange[2])]

    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵ, x], [n(t, ϵ, x), Ebar(t, ϵ, x)])

    order = 2
    discretization = MOLFiniteDifference([ϵ => numEnergyPoints, x => numPositionPoints], t)

    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)
    soln = sol[n(t, ϵ, x)]
    lenSolt = length(sol[t])
    sole = sol[ϵ]
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
zmin = min(soln...)
zmax = max(soln...)

anim = @animate for i in 1:lenSolt
    # camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
    plot = surface(sole, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(zmin, zmax))
    zlims!(zmin, zmax)
    titleStr = Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime)
    
    title!("Time = " * titleStr * "s")
end
gif(anim, "drift_diffusion.gif", fps = 10)

