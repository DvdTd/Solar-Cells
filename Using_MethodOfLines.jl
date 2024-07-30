using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots

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
numPlots = 4
dt = 1e-11# 5e-4#1e-11
maxTime = 1.0#5e-6# 1e1#5e-6
energyRange = [-1.0, 1.0] # ±infinity but cutoff when it goes to zeros
positionRange = [-10.0, 10.0] 
numEnergyPoints = 100
numPositionPoints = 100 

@parameters t, ϵ, x
@variables u(..) cumuSum(..)
Dt = Differential(t)
Dϵ = Differential(ϵ)
Dx = Differential(x)

normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x .- mean).^2 .* [0.5*width^(-2), 0]))
Ix = Integral(x in DomainSets.ClosedInterval(positionRange[1], x)) # basically cumulative sum from 0 to x

eq = [
    cumuSum(t, x) ~ Ix(u(t, x)), # Note wrapping the argument to the derivative with an auxiliary variable
    Dt(u(t, x)) + 2 * u(t, x) * u(t, x) + 5 * cumuSum(t, x) ~ 1
]

# Neumann examples: Dx(u(t, xmax)) ~ 0
initialFunc(x) = normalGaussian(x, 0, 1) - normalGaussian(positionRange[1], 0, 1)
bcs = [u(0.0, x) ~ initialFunc(x), u(t, positionRange[1]) ~ 0.0, u(t, positionRange[2]) ~ 0] 

domains = [t ∈ Interval(0.0, maxTime), x ∈ Interval(positionRange[1], positionRange[2])]

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x), cumuSum(t, x)])

order = 2
discretization = MOLFiniteDifference([x => numPositionPoints], t)

prob = MethodOfLines.discretize(pde_system, discretization)
sol = solve(prob, QNDF(), saveat = maxTime/numPlots)

solu = sol[u(t, x)]

display(plot(sol[x], transpose(solu)))
