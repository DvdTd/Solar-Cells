"""
Using MethodOfLines with the drift-diffusion equations and the differential light interaction term - coupled equations n and p.
Problems: currently, E = EH if ϵ>Eav else = EL. When put into the erf term, effectively gives a step function which is hard to explain physically.
"""

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots, Printf, JLD
import SpecialFunctions: erf

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
kbT = kb*T / eCharge
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

EH = -5.2
EL = -3.5
Eav = (EL+EH)/2 # = -4.35
E = 1.7 # XXX change to ϵ + F * x + const <- might need product rule...
numPlots = 100
gifTime = 10
maxPos = 20.0
positionRange = [-maxPos, maxPos] 
numPositionPoints = 15 # odd is good

energy_tail_length = 2
energyRangeNA = [EH - energy_tail_length, EL]
energyRangeNB = [EL, EL + energy_tail_length]
energyRangePA = [EH - energy_tail_length, EH]
energyRangePB = [EH, EL + energy_tail_length]

numEnergyPoints = 11 # * 2 for A and B, * 2 for n and p

# camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
F = 1f5
dt = 1f-12# 5f-4#1f-11
maxTime = 2f-3 #5f-6
cameraTup = (10, 50)#(10, 50)#(10,-5)#(85, 60) #  (90,40)

# Small F parameters
# F = 0f-1
# dt = 1f-4
# maxTime = 3e3
# cameraTup = (40, 55)

@parameters t, ϵnA, ϵnB, ϵpA, ϵpB, x
@variables nA(..) nB(..) pA(..) pB(..) 
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

DϵnA = Differential(ϵnA)
DϵϵnA = Differential(ϵnA)^2
DϵnB = Differential(ϵnB)
DϵϵnB = Differential(ϵnB)^2
DϵpA = Differential(ϵpA)
DϵϵpA = Differential(ϵpA)^2
DϵpB = Differential(ϵpB)
DϵϵpB = Differential(ϵpB)^2

shouldCalcNew = true # Gives the option to change the plot parameters without recalculating the solution.
jldFilePath = "/Users/david/Documents/Python/Solar Cells/CoupledMethodOfLinesData.jld"

if shouldCalcNew || !isfile(jldFilePath)
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean)^2 * [0.5*width^(-2), 0]))
    Ebar(ϵ, E) = Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-E) + sigma^2)

    eq = [
        # E = ϵ - F*x * eCharge(?) + C
        Dt(nA(t, ϵnA, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnA-EL)^3 + A2*(ϵnA-EL)^2 + A3*(ϵnA-EL) + A4)  +  DϵϵnB(nB(t, EL, x)) * exp(-(ϵnA-EL)^2/(2*sigma^2)) * ( (nA(t, ϵnA, x)-2)*(B1*(ϵnA-EL)^2+B2) + B3*(ϵnA-EL)^2 + B4*(ϵnA-EL) + B5) + nA(t, ϵnA, x) * DϵϵnB(nB(t, EL, x)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnA-EL)^3 + C2*(ϵnA-EL)) + (nA(t, ϵnA, x) - nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnA-EL)^3 + D2*(ϵnA-EL)^2 + D3*(ϵnA-EL) + D4)  +  ((nA(t, ϵnA, x) * nB(t, EL, x) - 4*nA(t, ϵnA, x) + 2*nB(t, EL, x))*(E1*(ϵnA-EL)^2 + E2) + (nA(t, ϵnA, x) - nB(t, EL, x)) * (E3*(ϵnA-EL)+E4)) * exp(-(ϵnA-EL)^2/(2*sigma^2))  + nA(t, ϵnA, x) * (nB(t, EL, x)-2) * (F1*(ϵnA-EL)^3 + F2*(ϵnA-EL)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnA-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nA(t, ϵnA, x)) + K/2 * Dxx(nA(t, ϵnA, x)) - C*Ebar(ϵnA, EH) * DϵnA(nA(t, ϵnA, x)) + C*(Ebar(ϵnA, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnA(nA(t, ϵnA, x)) ),
        Dt(nB(t, ϵnB, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnB-EL)^3 + A2*(ϵnB-EL)^2 + A3*(ϵnB-EL) + A4)  +  DϵϵnB(nB(t, EL, x)) * exp(-(ϵnB-EL)^2/(2*sigma^2)) * ( (nB(t, ϵnB, x)-2)*(B1*(ϵnB-EL)^2+B2) + B3*(ϵnB-EL)^2 + B4*(ϵnB-EL) + B5) + nB(t, ϵnB, x) * DϵϵnB(nB(t, EL, x)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnB-EL)^3 + C2*(ϵnB-EL)) + (nB(t, ϵnB, x) - nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnB-EL)^3 + D2*(ϵnB-EL)^2 + D3*(ϵnB-EL) + D4)  +  ((nB(t, ϵnB, x) * nB(t, EL, x) - 4*nB(t, ϵnB, x) + 2*nB(t, EL, x))*(E1*(ϵnB-EL)^2 + E2) + (nB(t, ϵnB, x) - nB(t, EL, x)) * (E3*(ϵnB-EL)+E4)) * exp(-(ϵnB-EL)^2/(2*sigma^2))  + nB(t, ϵnB, x) * (nB(t, EL, x)-2) * (F1*(ϵnB-EL)^3 + F2*(ϵnB-EL)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnB-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nB(t, ϵnB, x)) + K/2 * Dxx(nB(t, ϵnB, x)) - C*Ebar(ϵnB, EH) * DϵnB(nB(t, ϵnB, x)) + C*(Ebar(ϵnB, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnB(nB(t, ϵnB, x)) ),
        Dt(pA(t, ϵpA, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpA-EH)^3 + A2*(ϵpA-EH)^2 + A3*(ϵpA-EH) + A4)  +  DϵϵpB(pB(t, EH, x)) * exp(-(ϵpA-EH)^2/(2*sigma^2)) * ( (pA(t, ϵpA, x)-2)*(B1*(ϵpA-EH)^2+B2) + B3*(ϵpA-EH)^2 + B4*(ϵpA-EH) + B5) + pA(t, ϵpA, x) * DϵϵpB(pB(t, EH, x)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpA-EH)^3 + C2*(ϵpA-EH)) + (pA(t, ϵpA, x) - pB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpA-EH)^3 + D2*(ϵpA-EH)^2 + D3*(ϵpA-EH) + D4)  +  ((pA(t, ϵpA, x) * pB(t, EH, x) - 4*pA(t, ϵpA, x) + 2*pB(t, EH, x))*(E1*(ϵpA-EH)^2 + E2) + (pA(t, ϵpA, x) - pB(t, EH, x)) * (E3*(ϵpA-EH)+E4)) * exp(-(ϵpA-EH)^2/(2*sigma^2))  + pA(t, ϵpA, x) * (pB(t, EH, x)-2) * (F1*(ϵpA-EH)^3 + F2*(ϵpA-EH)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpA-EL)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(pA(t, ϵpA, x)) + K/2 * Dxx(pA(t, ϵpA, x)) - C*Ebar(ϵpA, EL) * DϵpA(pA(t, ϵpA, x)) + C*(Ebar(ϵpA, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpA(pA(t, ϵpA, x)) ), 
        Dt(pB(t, ϵpB, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpB-EH)^3 + A2*(ϵpB-EH)^2 + A3*(ϵpB-EH) + A4)  +  DϵϵpB(pB(t, EH, x)) * exp(-(ϵpB-EH)^2/(2*sigma^2)) * ( (pB(t, ϵpB, x)-2)*(B1*(ϵpB-EH)^2+B2) + B3*(ϵpB-EH)^2 + B4*(ϵpB-EH) + B5) + pB(t, ϵpB, x) * DϵϵpB(pB(t, EH, x)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpB-EH)^3 + C2*(ϵpB-EH)) + (pB(t, ϵpB, x) - pB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpB-EH)^3 + D2*(ϵpB-EH)^2 + D3*(ϵpB-EH) + D4)  +  ((pB(t, ϵpB, x) * pB(t, EH, x) - 4*pB(t, ϵpB, x) + 2*pB(t, EH, x))*(E1*(ϵpB-EH)^2 + E2) + (pB(t, ϵpB, x) - pB(t, EH, x)) * (E3*(ϵpB-EH)+E4)) * exp(-(ϵpB-EH)^2/(2*sigma^2))  + pB(t, ϵpB, x) * (pB(t, EH, x)-2) * (F1*(ϵpB-EH)^3 + F2*(ϵpB-EH)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpB-EL)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(pB(t, ϵpB, x)) + K/2 * Dxx(pB(t, ϵpB, x)) - C*Ebar(ϵpB, EL) * DϵpB(pB(t, ϵpB, x)) + C*(Ebar(ϵpB, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpB(pB(t, ϵpB, x)) ), 

        # Drift-diffusion only.
        # Dt(nA(t, ϵnA, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnA-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nA(t, ϵnA, x)) + K/2 * Dxx(nA(t, ϵnA, x)) - C*Ebar(ϵnA, EH) * DϵnA(nA(t, ϵnA, x)) + C*(Ebar(ϵnA, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnA(nA(t, ϵnA, x)) ),
        # Dt(nB(t, ϵnB, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnB-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nB(t, ϵnB, x)) + K/2 * Dxx(nB(t, ϵnB, x)) - C*Ebar(ϵnB, EH) * DϵnB(nB(t, ϵnB, x)) + C*(Ebar(ϵnB, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnB(nB(t, ϵnB, x)) ),
        # Dt(pA(t, ϵpA, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpA-EL)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(pA(t, ϵpA, x)) + K/2 * Dxx(pA(t, ϵpA, x)) - C*Ebar(ϵpA, EL) * DϵpA(pA(t, ϵpA, x)) + C*(Ebar(ϵpA, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpA(pA(t, ϵpA, x)) ), 
        # Dt(pB(t, ϵpB, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpB-EL)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(pB(t, ϵpB, x)) + K/2 * Dxx(pB(t, ϵpB, x)) - C*Ebar(ϵpB, EL) * DϵpB(pB(t, ϵpB, x)) + C*(Ebar(ϵpB, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpB(pB(t, ϵpB, x)) ), 
        ]
    
    # Initial: Gaussian at mean
    (posWidth, energyWidth) = (1, 0.4) # (1, 3) (1, 0.4)
    initialFunc(ϵ, x, mean, ) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, mean, energyWidth) - min(normalGaussian(energyRangeNA[1], mean, energyWidth), normalGaussian(energyRangeNB[2], mean, energyWidth))  )
   
    bcs = [
        nA(0.0, ϵnA, x) ~ initialFunc(ϵnA, x, EH),
        nB(0.0, ϵnB, x) ~ initialFunc(ϵnB, x, EH),
        pA(0.0, ϵpA, x) ~ initialFunc(ϵpA, x, EL),
        pB(0.0, ϵpB, x) ~ initialFunc(ϵpB, x, EL),

        DϵnA(nA(t, energyRangeNA[1], x)) ~ 0.0,
        DϵnB(nB(t, energyRangeNB[2], x)) ~ 0.0, 
        DϵpA(pA(t, energyRangePA[1], x)) ~ 0.0,
        DϵpB(pB(t, energyRangePB[2], x)) ~ 0.0, 

        nA(t, ϵnA, positionRange[1]) ~ 0.0, 
        nA(t, ϵnA, positionRange[2]) ~ 0.0,
        nB(t, ϵnB, positionRange[1]) ~ 0.0, 
        nB(t, ϵnB, positionRange[2]) ~ 0.0,
        pA(t, ϵpA, positionRange[1]) ~ 0.0, 
        pA(t, ϵpA, positionRange[2]) ~ 0.0,
        pB(t, ϵpB, positionRange[1]) ~ 0.0, 
        pB(t, ϵpB, positionRange[2]) ~ 0.0,
        
        # Neumann position
        # Dx(nA(t, ϵnA, positionRange[1])) ~ 0.0, 
        # Dx(nA(t, ϵnA, positionRange[2])) ~ 0.0,
        # Dx(nB(t, ϵnB, positionRange[1])) ~ 0.0, 
        # Dx(nB(t, ϵnB, positionRange[2])) ~ 0.0,
        # Dx(pA(t, ϵpA, positionRange[1])) ~ 0.0, 
        # Dx(pA(t, ϵpA, positionRange[2])) ~ 0.0,
        # Dx(pB(t, ϵpB, positionRange[1])) ~ 0.0, 
        # Dx(pB(t, ϵpB, positionRange[2])) ~ 0.0,

        # Interface
        nB(t, energyRangeNB[1], x) ~ nA(t, energyRangeNA[2], x), 
        DϵnB(nB(t, energyRangeNB[1], x)) ~ DϵnA(nA(t, energyRangeNA[2], x)),
        pB(t, energyRangePB[1], x) ~ pA(t, energyRangePA[2], x), 
        DϵpB(pB(t, energyRangePB[1], x)) ~ DϵpA(pA(t, energyRangePA[2], x)),
        ] 

    # Solve the equations.
    domains = [t ∈ Interval(0.0, maxTime), ϵnA ∈ Interval(energyRangeNA[1], energyRangeNA[2]), ϵnB ∈ Interval(energyRangeNB[1], energyRangeNB[2]), ϵpA ∈ Interval(energyRangePA[1], energyRangePA[2]), ϵpB ∈ Interval(energyRangePB[1], energyRangePB[2]), x ∈ Interval(positionRange[1], positionRange[2])]
    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵnA, ϵnB, ϵpA, ϵpB, x], [nA(t, ϵnA, x), nB(t, ϵnB, x), pA(t, ϵpA, x), pB(t, ϵpB, x)])
    order = 2
    discretization = MOLFiniteDifference([ϵnA => numEnergyPoints, ϵnB => numEnergyPoints, ϵpA => numEnergyPoints, ϵpB => numEnergyPoints, x => numPositionPoints], t)
    
    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)

    # Extract relevant parts of the solution.
    soln = hcat(sol[nA(t, ϵnA, x)], sol[nB(t, ϵnB, x)])
    lenSolt = length(sol[t])
    solne = [sol[ϵnA]; sol[ϵnB]]
    solp = hcat(sol[pA(t, ϵpA, x)], sol[pB(t, ϵpB, x)])
    lenSolt = length(sol[t])
    solpe = [sol[ϵpA]; sol[ϵpB]]
    solx = sol[x]

    # Write to file
    solDict = Dict("soln" => soln, "solp" => solp, "lenSolt" => lenSolt, "solne" => solne, "solpe" => solpe, "solx" => solx, "maxTime" => maxTime)
    save(jldFilePath, "data", solDict)

else # Read from file
    solDict = load(jldFilePath)["data"]
    soln = solDict["soln"]
    solp = solDict["solp"]
    lenSolt = solDict["lenSolt"]
    solne = solDict["solne"]
    solpe = solDict["solpe"]
    solx = solDict["solx"]
    maxTime = solDict["maxTime"]

end

# Plot
nzmin = min(soln[:,:,:]...)
nzmax = max(soln[:,:,:]...)
pzmin = min(solp[:,:,:]...)
pzmax = max(solp[:,:,:]...)

# Show individual plots.
shownPlots = [1,2,3,4,5,6]
    
for i in shownPlots
    plotn = surface(solne, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(nzmin, nzmax), legend = :none)
    # zlims!(nzmin, nzmax)

    plotp = surface(solpe, solx, transpose(solp[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="p", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(pzmin, pzmax), legend = :none)
    # zlims!(pzmin, pzmax)

    plotnp = Plots.plot(plotn, plotp, layout = (1,2))
    title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")

    display(plotnp)
end

# Make gif.
anim = @animate for i in 1:lenSolt
    # camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
    plotn = surface(solne, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(nzmin, nzmax), legend = :none)
    # zlims!(nzmin, nzmax)

    plotp = surface(solpe, solx, transpose(solp[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="p", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(pzmin, pzmax), legend = :none)
    # zlims!(pzmin, pzmax)

    plotnp = Plots.plot(plotn, plotp, layout = (1,2))
    title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")
end
display(gif(anim, "drift_diffusion.gif", fps = floor(numPlots/gifTime)))

