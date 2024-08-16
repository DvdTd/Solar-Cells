"""
Using MethodOfLines with the drift-diffusion equations and the differential light interaction term.
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
E = 1.7 # XXX remove later
numPlots = 200
maxPos = 20.0
positionRange = [-maxPos, maxPos] 
numPositionPoints = 17 # odd is good

energy_tail_length = 2
energyRangeA = [EH - energy_tail_length, EH]
energyRangeB = [EH, EL]
energyRangeC = [EL, EL + energy_tail_length]
numEnergyPointsAC = 7
numEnergyPointsB = numEnergyPointsAC

# camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z  e.g. (80, 50)
F = 1f5
dt = 1f-12# 5f-4#1f-11
maxTime = 2f-3 #5f-6
cameraTup = (90, 40)#(10, 50)#(10,-5)#(85, 60) #  (90,40)

# Small F parameters
# F = 0f-1
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
    normalGaussian(x, mean, width) = (2*π)^(-0.5)*width^(-2) * exp(- sum((x - mean)^2 * [0.5*width^(-2), 0]))

    step(x, x0) =  (1+sign(x-x0))/2
    Ec(ϵ) = EH + (EL - EH)*step(ϵ, Eav)
    ECgaussian(x, mean1, mean2, width) = (exp(-(x-mean1)^2*0.5*width^(-2)) * (1-step(x,(mean1+mean2)/2)) + exp(-(x-mean2)^2*0.5*width^(-2)) * step(x,(mean1+mean2)/2) ) #XXX only used once so remove mean args?
    # Interpolate E from EL to EH (using step function)
    EB(ϵB) = EL + (EH - EL)*step(ϵB, Eav)
    piecewiseNB(ϵB) = nB(t, EL, x) * (1-step(ϵB,Eav)) + nB(t, EH, x) * (step(ϵB,Eav))
    piecewiseD2NB(ϵB) = DϵϵB(nB(t, EL, x)) * (1-step(ϵB,Eav)) + DϵϵB(nB(t, EH, x)) * (step(ϵB,Eav))

    # Interpolate between Gaussians with different means; G(mean1) * (1-sig) + G(mean2) * (sig)
    Ebar(ϵ) = Lambda*sigmaTilde^(-2) * (2/beta*(ϵ-Ec(ϵ)) + sigma^2)

    # lightFunc(ϵ, n, E_func, D2N, argN) = 1/(2*sqrt(π) * hbar^2 * kbT) * ( D2N(ϵ) * sqrt(π) * exp(-(2*(ϵ-E_func(ϵ))*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵ-E_func(ϵ))*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵ-E_func(ϵ))^3 + A2*(ϵ-E_func(ϵ))^2 + A3*(ϵ-E_func(ϵ)) + A4)  +  D2N(ϵ) * exp(-(ϵ-E_func(ϵ))^2/(2*sigma^2)) * ( (n(t, ϵ, x)-2)*(B1*(ϵ-E_func(ϵ))^2+B2) + B3*(ϵ-E_func(ϵ))^2 + B4*(ϵ-E_func(ϵ)) + B5) + n(t, ϵ, x) * D2N(ϵ) * sqrt(π) * (erf(-(ϵ-E_func(ϵ))/(sqrt(2)*sigma))-1) * (C1*(ϵ-E_func(ϵ))^3 + C2*(ϵ-E_func(ϵ))) + (n(t, ϵ, x) - argN(ϵ)) * sqrt(π) * exp(-(2*(ϵ-E_func(ϵ))*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵ-E_func(ϵ))*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵ-E_func(ϵ))^3 + D2*(ϵ-E_func(ϵ))^2 + D3*(ϵ-E_func(ϵ)) + D4)  +  ((n(t, ϵ, x) * argN(ϵ) - 4*n(t, ϵ, x) + 2*argN(ϵ))*(E1*(ϵ-E_func(ϵ))^2 + E2) + (n(t, ϵ, x) - argN(ϵ)) * (E3*(ϵ-E_func(ϵ))+E4)) * exp(-(ϵ-E_func(ϵ))^2/(2*sigma^2))  + n(t, ϵ, x) * (argN(ϵ)-2) * (F1*(ϵ-E_func(ϵ))^3 + F2*(ϵ-E_func(ϵ))) * sqrt(π) * (erf(-(ϵ-E_func(ϵ))/(sqrt(2)*sigma))-1)    )
    # Dt(nA(t, ϵA, x)) ~ lightFunc(ϵA, nA, ϵA -> EL, ϵA -> DϵϵB(nB(t, EL, x)), ϵA -> nB(t, EL, x)) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵA, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nA(t, ϵA, x)) + K/2 * Dxx(nA(t, ϵA, x)) - C*Ebar(ϵA) * DϵA(nA(t, ϵA, x)) + C*(Ebar(ϵA)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵA(nA(t, ϵA, x)) ),
    # Dt(nB(t, ϵB, x)) ~ lightFunc(ϵB, nB, EB, piecewiseD2NB, piecewiseNB) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵB, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nB(t, ϵB, x)) + K/2 * Dxx(nB(t, ϵB, x)) - C*Ebar(ϵB) * DϵB(nB(t, ϵB, x)) + C*(Ebar(ϵB)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵB(nB(t, ϵB, x)) ),
    # Dt(nC(t, ϵC, x)) ~ lightFunc(ϵC, nC, ϵC -> EH, ϵC -> DϵϵB(nB(t, EH, x)), ϵC -> nB(t, EH, x)) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵC, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nC(t, ϵC, x)) + K/2 * Dxx(nC(t, ϵC, x)) - C*Ebar(ϵC) * DϵC(nC(t, ϵC, x)) + C*(Ebar(ϵC)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵC(nC(t, ϵC, x)) ),

    eq = [
        # Dt(nA(t, ϵA, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵB(nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵA-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵA-EL)^3 + A2*(ϵA-EL)^2 + A3*(ϵA-EL) + A4)  +  DϵϵB(nB(t, EL, x)) * exp(-(ϵA-EL)^2/(2*sigma^2)) * ( (nA(t, ϵA, x)-2)*(B1*(ϵA-EL)^2+B2) + B3*(ϵA-EL)^2 + B4*(ϵA-EL) + B5) + nA(t, ϵA, x) * DϵϵB(nB(t, EL, x)) * sqrt(π) * (erf(-(ϵA-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵA-EL)^3 + C2*(ϵA-EL)) + (nA(t, ϵA, x) - nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵA-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵA-EL)^3 + D2*(ϵA-EL)^2 + D3*(ϵA-EL) + D4)  +  ((nA(t, ϵA, x) * nB(t, EL, x) - 4*nA(t, ϵA, x) + 2*nB(t, EL, x))*(E1*(ϵA-EL)^2 + E2) + (nA(t, ϵA, x) - nB(t, EL, x)) * (E3*(ϵA-EL)+E4)) * exp(-(ϵA-EL)^2/(2*sigma^2))  + nA(t, ϵA, x) * (nB(t, EL, x)-2) * (F1*(ϵA-EL)^3 + F2*(ϵA-EL)) * sqrt(π) * (erf(-(ϵA-EL)/(sqrt(2)*sigma))-1)    )       +     exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵA, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nA(t, ϵA, x)) + K/2 * Dxx(nA(t, ϵA, x)) - C*Ebar(ϵA) * DϵA(nA(t, ϵA, x)) + C*(Ebar(ϵA)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵA(nA(t, ϵA, x)) ),
        # Dt(nB(t, ϵB, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( piecewiseD2NB(ϵB) * sqrt(π) * exp(-(2*(ϵB-EB(ϵB))*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵB-EB(ϵB))*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵB-EB(ϵB))^3 + A2*(ϵB-EB(ϵB))^2 + A3*(ϵB-EB(ϵB)) + A4)  +  piecewiseD2NB(ϵB) * exp(-(ϵB-EB(ϵB))^2/(2*sigma^2)) * ( (nB(t, ϵB, x)-2)*(B1*(ϵB-EB(ϵB))^2+B2) + B3*(ϵB-EB(ϵB))^2 + B4*(ϵB-EB(ϵB)) + B5) + nB(t, ϵB, x) * piecewiseD2NB(ϵB) * sqrt(π) * (erf(-(ϵB-EB(ϵB))/(sqrt(2)*sigma))-1) * (C1*(ϵB-EB(ϵB))^3 + C2*(ϵB-EB(ϵB))) + (nB(t, ϵB, x) - piecewiseNB(ϵB)) * sqrt(π) * exp(-(2*(ϵB-EB(ϵB))*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵB-EB(ϵB))*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵB-EB(ϵB))^3 + D2*(ϵB-EB(ϵB))^2 + D3*(ϵB-EB(ϵB)) + D4)  +  ((nB(t, ϵB, x) * piecewiseNB(ϵB) - 4*nB(t, ϵB, x) + 2*piecewiseNB(ϵB))*(E1*(ϵB-EB(ϵB))^2 + E2) + (nB(t, ϵB, x) - piecewiseNB(ϵB)) * (E3*(ϵB-EB(ϵB))+E4)) * exp(-(ϵB-EB(ϵB))^2/(2*sigma^2))  + nB(t, ϵB, x) * (piecewiseNB(ϵB)-2) * (F1*(ϵB-EB(ϵB))^3 + F2*(ϵB-EB(ϵB))) * sqrt(π) * (erf(-(ϵB-EB(ϵB))/(sqrt(2)*sigma))-1)    )       +     exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵB, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nB(t, ϵB, x)) + K/2 * Dxx(nB(t, ϵB, x)) - C*Ebar(ϵB) * DϵB(nB(t, ϵB, x)) + C*(Ebar(ϵB)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵB(nB(t, ϵB, x)) ),
        # Dt(nC(t, ϵC, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵB(nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵC-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵC-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵC-EH)^3 + A2*(ϵC-EH)^2 + A3*(ϵC-EH) + A4)  +  DϵϵB(nB(t, EH, x)) * exp(-(ϵC-EH)^2/(2*sigma^2)) * ( (nC(t, ϵC, x)-2)*(B1*(ϵC-EH)^2+B2) + B3*(ϵC-EH)^2 + B4*(ϵC-EH) + B5) + nC(t, ϵC, x) * DϵϵB(nB(t, EH, x)) * sqrt(π) * (erf(-(ϵC-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵC-EH)^3 + C2*(ϵC-EH)) + (nC(t, ϵC, x) - nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵC-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵC-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵC-EH)^3 + D2*(ϵC-EH)^2 + D3*(ϵC-EH) + D4)  +  ((nC(t, ϵC, x) * nB(t, EH, x) - 4*nC(t, ϵC, x) + 2*nB(t, EH, x))*(E1*(ϵC-EH)^2 + E2) + (nC(t, ϵC, x) - nB(t, EH, x)) * (E3*(ϵC-EH)+E4)) * exp(-(ϵC-EH)^2/(2*sigma^2))  + nC(t, ϵC, x) * (nB(t, EH, x)-2) * (F1*(ϵC-EH)^3 + F2*(ϵC-EH)) * sqrt(π) * (erf(-(ϵC-EH)/(sqrt(2)*sigma))-1)    )       +     exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵC, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nC(t, ϵC, x)) + K/2 * Dxx(nC(t, ϵC, x)) - C*Ebar(ϵC) * DϵC(nC(t, ϵC, x)) + C*(Ebar(ϵC)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵC(nC(t, ϵC, x)) ),
        
        # E = ϵ - F*x * eCharge + C
        Dt(nA(t, ϵA, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵA, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nA(t, ϵA, x)) + K/2 * Dxx(nA(t, ϵA, x)) - C*Ebar(ϵA) * DϵA(nA(t, ϵA, x)) + C*(Ebar(ϵA)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵA(nA(t, ϵA, x)) ),
        Dt(nB(t, ϵB, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵB, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nB(t, ϵB, x)) + K/2 * Dxx(nB(t, ϵB, x)) - C*Ebar(ϵB) * DϵB(nB(t, ϵB, x)) + C*(Ebar(ϵB)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵB(nB(t, ϵB, x)) ),
        Dt(nC(t, ϵC, x)) ~ exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵC, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * Dx(nC(t, ϵC, x)) + K/2 * Dxx(nC(t, ϵC, x)) - C*Ebar(ϵC) * DϵC(nC(t, ϵC, x)) + C*(Ebar(ϵC)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵC(nC(t, ϵC, x)) ),

        ]
    # XXX check for travellig behaviour with different IC.
    # Gaussian at Eav
    (posWidth, energyWidth) = (1, 1) # (1, 3) (1, 0.4)
    initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) # * (  normalGaussian(ϵ, Eav, energyWidth) - min(normalGaussian(energyRangeA[1], Eav, energyWidth), normalGaussian(energyRangeC[2], Eav, energyWidth))  )
    # 2 Gaussians at EL and EH
    # (posWidth, energyWidth) = (1, 0.4) # (1, 3) (1, 0.4)
    # initialFunc(ϵ, x) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, EH, energyWidth) + normalGaussian(ϵ, EL, energyWidth) - min(normalGaussian(energyRangeA[1], EH, energyWidth) + normalGaussian(energyRangeA[1], EL, energyWidth), normalGaussian(energyRangeC[2], EH, energyWidth) + normalGaussian(energyRangeC[2], EL, energyWidth))  )
   
    bcs = [
        nA(0.0, ϵA, x) ~ initialFunc(ϵA, x),
        nB(0.0, ϵB, x) ~ initialFunc(ϵB, x),
        nC(0.0, ϵC, x) ~ initialFunc(ϵC, x),

        DϵA(nA(t, energyRangeA[1], x)) ~ 0.0,
        DϵC(nC(t, energyRangeC[2], x)) ~ 0.0, 

        # nA(t, ϵA, positionRange[1]) ~ 0.0, 
        # nA(t, ϵA, positionRange[2]) ~ 0.0,
        # nB(t, ϵB, positionRange[1]) ~ 0.0, 
        # nB(t, ϵB, positionRange[2]) ~ 0.0,
        # nC(t, ϵC, positionRange[1]) ~ 0.0, 
        # nC(t, ϵC, positionRange[2]) ~ 0.0,
        
        # Neumann position
        Dx(nA(t, ϵA, positionRange[1])) ~ 0.0, 
        Dx(nA(t, ϵA, positionRange[2])) ~ 0.0,
        Dx(nB(t, ϵB, positionRange[1])) ~ 0.0, 
        Dx(nB(t, ϵB, positionRange[2])) ~ 0.0,
        Dx(nC(t, ϵC, positionRange[1])) ~ 0.0, 
        Dx(nC(t, ϵC, positionRange[2])) ~ 0.0,

        # periodic at position boundaries
        # nA(t, ϵA, positionRange[1]) ~ nA(t, ϵA, positionRange[2]),
        # nC(t, ϵC, positionRange[1]) ~ nC(t, ϵC, positionRange[2]),
        # nB(t, ϵB, positionRange[1]) ~ nB(t, ϵB, positionRange[2]),
        # Dx(nA(t, ϵA, positionRange[1])) ~ Dx(nA(t, ϵA, positionRange[2])),
        # Dx(nB(t, ϵB, positionRange[1])) ~ Dx(nB(t, ϵB, positionRange[2])),
        # Dx(nC(t, ϵC, positionRange[1])) ~ Dx(nC(t, ϵC, positionRange[2])),


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
    # zlims!(zmin, zmax)
end
display(gif(anim, "drift_diffusion.gif", fps = floor(numPlots/5)))

