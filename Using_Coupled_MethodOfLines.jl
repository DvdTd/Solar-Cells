"""
Using MethodOfLines with the drift-diffusion equations and the differential light interaction term - coupled equations n and p.

Searching online says the following warning isn't an issue:
Warning: The system contains interface boundaries, which are not compatible with system transformation. The system will not be transformed. Please post an issue if you need this feature.
"""

using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, DomainSets, Plots, Printf, JLD
import SpecialFunctions: erf

d = 2 # dimension
kb = 1.3806f-23
eCharge = 1.602f-19
hbar = 1.054571817f-34 / eCharge
gamma = 1f8
nu0 = 5e19
g1 = 1e14
sigma = 0.13 
Lambda = 9e-5
T = 300 # Temperature of the solar cell.
K = [1/4*gamma^(-3), 3*π/8*gamma^(-4), π*gamma^(-5)][clamp(d-1, 1, 3)]
C = [gamma^(-1), π/2*gamma^(-2), π*gamma^(-3)][clamp(d-1, 1, 3)]
beta = 1/(kb*T) * eCharge # Only used in DD so use T.
sigmaTilde = sqrt(sigma^2 + 2*Lambda/beta)
EH = -5.2
EL = -3.5
Eav = (EL+EH)/2 # = -4.35

# Light interaction Constants
T_light = 5772 # Temperature from the sun giving black body radiation.
kbT = kb*T_light / eCharge # Only used in light term so use T_light
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


# Variables which can be adjusted:

maxPos = 10.0
positionRange = [-maxPos, maxPos] 
energy_tail_length = 1.5
energyRangePA = [EH - energy_tail_length, EL]
energyRangePB = [EL, EL + energy_tail_length]
energyRangeNA = [EH - energy_tail_length, EH]
energyRangeNB = [EH, EL + energy_tail_length]

numPositionPoints = 15 # Odd is good so central peak is calculated
numEnergyPoints = 11 # Effectively *4 for nA, nB, pA and pB.

F = 1f5/eCharge
dt = 1f-19# 5f-4#1f-11
maxTime = 1f-0 #5f-6

DD_speed_coeff = 1f-35
light_speed_coeff = 1f-27

# Plot parameters
numPlots = 100
gifTime = 5 # seconds
# camera=(azimuthal, elevation), azimuthal is left-handed rotation about +ve z
# cameraTup = (140, 60)
cameraTup = (60, 40)
# cameraTup = (25, 50)


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
        # Drift-diffusion and light interaction.
        Dt(nA(t, ϵnA, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnA-EL)^3 + A2*(ϵnA-EL)^2 + A3*(ϵnA-EL) + A4)  +  DϵϵpB(pB(t, EL, x)) * exp(-(ϵnA-EL)^2/(2*sigma^2)) * ( (nA(t, ϵnA, x)-2)*(B1*(ϵnA-EL)^2+B2) + B3*(ϵnA-EL)^2 + B4*(ϵnA-EL) + B5) + nA(t, ϵnA, x) * DϵϵpB(pB(t, EL, x)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnA-EL)^3 + C2*(ϵnA-EL)) + (nA(t, ϵnA, x) - pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnA-EL)^3 + D2*(ϵnA-EL)^2 + D3*(ϵnA-EL) + D4)  +  ((nA(t, ϵnA, x) * pB(t, EL, x) - 4*nA(t, ϵnA, x) + 2*pB(t, EL, x))*(E1*(ϵnA-EL)^2 + E2) + (nA(t, ϵnA, x) - pB(t, EL, x)) * (E3*(ϵnA-EL)+E4)) * exp(-(ϵnA-EL)^2/(2*sigma^2))  + nA(t, ϵnA, x) * (pB(t, EL, x)-2) * (F1*(ϵnA-EL)^3 + F2*(ϵnA-EL)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1)    )       +        DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnA-EH)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(nA(t, ϵnA, x)) + K/2 * Dxx(nA(t, ϵnA, x)) - C*(2*beta*Ebar(ϵnA, EH)^2 - Ebar(ϵnA, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵnA(nA(t, ϵnA, x)) + C*(Ebar(ϵnA, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnA(nA(t, ϵnA, x)) + beta*C*(beta*Ebar(ϵnA, EH)^2 - Ebar(ϵnA, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * nA(t, ϵnA, x)  ),
        Dt(nB(t, ϵnB, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnB-EL)^3 + A2*(ϵnB-EL)^2 + A3*(ϵnB-EL) + A4)  +  DϵϵpB(pB(t, EL, x)) * exp(-(ϵnB-EL)^2/(2*sigma^2)) * ( (nB(t, ϵnB, x)-2)*(B1*(ϵnB-EL)^2+B2) + B3*(ϵnB-EL)^2 + B4*(ϵnB-EL) + B5) + nB(t, ϵnB, x) * DϵϵpB(pB(t, EL, x)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnB-EL)^3 + C2*(ϵnB-EL)) + (nB(t, ϵnB, x) - pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnB-EL)^3 + D2*(ϵnB-EL)^2 + D3*(ϵnB-EL) + D4)  +  ((nB(t, ϵnB, x) * pB(t, EL, x) - 4*nB(t, ϵnB, x) + 2*pB(t, EL, x))*(E1*(ϵnB-EL)^2 + E2) + (nB(t, ϵnB, x) - pB(t, EL, x)) * (E3*(ϵnB-EL)+E4)) * exp(-(ϵnB-EL)^2/(2*sigma^2))  + nB(t, ϵnB, x) * (pB(t, EL, x)-2) * (F1*(ϵnB-EL)^3 + F2*(ϵnB-EL)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1)    )       +        DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnB-EH)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(nB(t, ϵnB, x)) + K/2 * Dxx(nB(t, ϵnB, x)) - C*(2*beta*Ebar(ϵnB, EH)^2 - Ebar(ϵnB, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵnB(nB(t, ϵnB, x)) + C*(Ebar(ϵnB, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnB(nB(t, ϵnB, x)) + beta*C*(beta*Ebar(ϵnB, EH)^2 - Ebar(ϵnB, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * nB(t, ϵnB, x)  ),
        Dt(pA(t, ϵpA, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpA-EH)^3 + A2*(ϵpA-EH)^2 + A3*(ϵpA-EH) + A4)  +  DϵϵnB(nB(t, EH, x)) * exp(-(ϵpA-EH)^2/(2*sigma^2)) * ( (pA(t, ϵpA, x)-2)*(B1*(ϵpA-EH)^2+B2) + B3*(ϵpA-EH)^2 + B4*(ϵpA-EH) + B5) + pA(t, ϵpA, x) * DϵϵnB(nB(t, EH, x)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpA-EH)^3 + C2*(ϵpA-EH)) + (pA(t, ϵpA, x) - nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpA-EH)^3 + D2*(ϵpA-EH)^2 + D3*(ϵpA-EH) + D4)  +  ((pA(t, ϵpA, x) * nB(t, EH, x) - 4*pA(t, ϵpA, x) + 2*nB(t, EH, x))*(E1*(ϵpA-EH)^2 + E2) + (pA(t, ϵpA, x) - nB(t, EH, x)) * (E3*(ϵpA-EH)+E4)) * exp(-(ϵpA-EH)^2/(2*sigma^2))  + pA(t, ϵpA, x) * (nB(t, EH, x)-2) * (F1*(ϵpA-EH)^3 + F2*(ϵpA-EH)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1)    )       +        DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpA-EL)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(pA(t, ϵpA, x)) + K/2 * Dxx(pA(t, ϵpA, x)) - C*(2*beta*Ebar(ϵpA, EL)^2 - Ebar(ϵpA, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵpA(pA(t, ϵpA, x)) + C*(Ebar(ϵpA, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpA(pA(t, ϵpA, x)) + beta*C*(beta*Ebar(ϵpA, EL)^2 - Ebar(ϵpA, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * pA(t, ϵpA, x)  ), 
        Dt(pB(t, ϵpB, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpB-EH)^3 + A2*(ϵpB-EH)^2 + A3*(ϵpB-EH) + A4)  +  DϵϵnB(nB(t, EH, x)) * exp(-(ϵpB-EH)^2/(2*sigma^2)) * ( (pB(t, ϵpB, x)-2)*(B1*(ϵpB-EH)^2+B2) + B3*(ϵpB-EH)^2 + B4*(ϵpB-EH) + B5) + pB(t, ϵpB, x) * DϵϵnB(nB(t, EH, x)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpB-EH)^3 + C2*(ϵpB-EH)) + (pB(t, ϵpB, x) - nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpB-EH)^3 + D2*(ϵpB-EH)^2 + D3*(ϵpB-EH) + D4)  +  ((pB(t, ϵpB, x) * nB(t, EH, x) - 4*pB(t, ϵpB, x) + 2*nB(t, EH, x))*(E1*(ϵpB-EH)^2 + E2) + (pB(t, ϵpB, x) - nB(t, EH, x)) * (E3*(ϵpB-EH)+E4)) * exp(-(ϵpB-EH)^2/(2*sigma^2))  + pB(t, ϵpB, x) * (nB(t, EH, x)-2) * (F1*(ϵpB-EH)^3 + F2*(ϵpB-EH)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1)    )       +        DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpB-EL)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(pB(t, ϵpB, x)) + K/2 * Dxx(pB(t, ϵpB, x)) - C*(2*beta*Ebar(ϵpB, EL)^2 - Ebar(ϵpB, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵpB(pB(t, ϵpB, x)) + C*(Ebar(ϵpB, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpB(pB(t, ϵpB, x)) + beta*C*(beta*Ebar(ϵpB, EL)^2 - Ebar(ϵpB, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * pB(t, ϵpB, x)  ), 

        # Drift-diffusion only.
        # Dt(nA(t, ϵnA, x)) ~ DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnA-EH)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(nA(t, ϵnA, x)) + K/2 * Dxx(nA(t, ϵnA, x)) - C*(2*beta*Ebar(ϵnA, EH)^2 - Ebar(ϵnA, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵnA(nA(t, ϵnA, x)) + C*(Ebar(ϵnA, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnA(nA(t, ϵnA, x)) + beta*C*(beta*Ebar(ϵnA, EH)^2 - Ebar(ϵnA, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * nA(t, ϵnA, x)  ),
        # Dt(nB(t, ϵnB, x)) ~ DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnB-EH)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(nB(t, ϵnB, x)) + K/2 * Dxx(nB(t, ϵnB, x)) - C*(2*beta*Ebar(ϵnB, EH)^2 - Ebar(ϵnB, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵnB(nB(t, ϵnB, x)) + C*(Ebar(ϵnB, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnB(nB(t, ϵnB, x)) + beta*C*(beta*Ebar(ϵnB, EH)^2 - Ebar(ϵnB, EH) + 2*Lambda*sigma^2/sigmaTilde^2) * nB(t, ϵnB, x)  ),
        # Dt(pA(t, ϵpA, x)) ~ DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpA-EL)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(pA(t, ϵpA, x)) + K/2 * Dxx(pA(t, ϵpA, x)) - C*(2*beta*Ebar(ϵpA, EL)^2 - Ebar(ϵpA, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵpA(pA(t, ϵpA, x)) + C*(Ebar(ϵpA, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpA(pA(t, ϵpA, x)) + beta*C*(beta*Ebar(ϵpA, EL)^2 - Ebar(ϵpA, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * pA(t, ϵpA, x)  ), 
        # Dt(pB(t, ϵpB, x)) ~ DD_speed_coeff * nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵpB-EL)^2*0.5*sigmaTilde^(-2)) * (  -K*beta/2*F * Dx(pB(t, ϵpB, x)) + K/2 * Dxx(pB(t, ϵpB, x)) - C*(2*beta*Ebar(ϵpB, EL)^2 - Ebar(ϵpB, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * DϵpB(pB(t, ϵpB, x)) + C*(Ebar(ϵpB, EL)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵpB(pB(t, ϵpB, x)) + beta*C*(beta*Ebar(ϵpB, EL)^2 - Ebar(ϵpB, EL) + 2*Lambda*sigma^2/sigmaTilde^2) * pB(t, ϵpB, x)  ), 
        
        # Light interaction only.
        # Dt(nA(t, ϵnA, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnA-EL)^3 + A2*(ϵnA-EL)^2 + A3*(ϵnA-EL) + A4)  +  DϵϵpB(pB(t, EL, x)) * exp(-(ϵnA-EL)^2/(2*sigma^2)) * ( (nA(t, ϵnA, x)-2)*(B1*(ϵnA-EL)^2+B2) + B3*(ϵnA-EL)^2 + B4*(ϵnA-EL) + B5) + nA(t, ϵnA, x) * DϵϵpB(pB(t, EL, x)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnA-EL)^3 + C2*(ϵnA-EL)) + (nA(t, ϵnA, x) - pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnA-EL)^3 + D2*(ϵnA-EL)^2 + D3*(ϵnA-EL) + D4)  +  ((nA(t, ϵnA, x) * pB(t, EL, x) - 4*nA(t, ϵnA, x) + 2*pB(t, EL, x))*(E1*(ϵnA-EL)^2 + E2) + (nA(t, ϵnA, x) - pB(t, EL, x)) * (E3*(ϵnA-EL)+E4)) * exp(-(ϵnA-EL)^2/(2*sigma^2))  + nA(t, ϵnA, x) * (pB(t, EL, x)-2) * (F1*(ϵnA-EL)^3 + F2*(ϵnA-EL)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1)    ) ,
        # Dt(nB(t, ϵnB, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵpB(pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnB-EL)^3 + A2*(ϵnB-EL)^2 + A3*(ϵnB-EL) + A4)  +  DϵϵpB(pB(t, EL, x)) * exp(-(ϵnB-EL)^2/(2*sigma^2)) * ( (nB(t, ϵnB, x)-2)*(B1*(ϵnB-EL)^2+B2) + B3*(ϵnB-EL)^2 + B4*(ϵnB-EL) + B5) + nB(t, ϵnB, x) * DϵϵpB(pB(t, EL, x)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnB-EL)^3 + C2*(ϵnB-EL)) + (nB(t, ϵnB, x) - pB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnB-EL)^3 + D2*(ϵnB-EL)^2 + D3*(ϵnB-EL) + D4)  +  ((nB(t, ϵnB, x) * pB(t, EL, x) - 4*nB(t, ϵnB, x) + 2*pB(t, EL, x))*(E1*(ϵnB-EL)^2 + E2) + (nB(t, ϵnB, x) - pB(t, EL, x)) * (E3*(ϵnB-EL)+E4)) * exp(-(ϵnB-EL)^2/(2*sigma^2))  + nB(t, ϵnB, x) * (pB(t, EL, x)-2) * (F1*(ϵnB-EL)^3 + F2*(ϵnB-EL)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1)    ) ,
        # Dt(pA(t, ϵpA, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpA-EH)^3 + A2*(ϵpA-EH)^2 + A3*(ϵpA-EH) + A4)  +  DϵϵnB(nB(t, EH, x)) * exp(-(ϵpA-EH)^2/(2*sigma^2)) * ( (pA(t, ϵpA, x)-2)*(B1*(ϵpA-EH)^2+B2) + B3*(ϵpA-EH)^2 + B4*(ϵpA-EH) + B5) + pA(t, ϵpA, x) * DϵϵnB(nB(t, EH, x)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpA-EH)^3 + C2*(ϵpA-EH)) + (pA(t, ϵpA, x) - nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpA-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpA-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpA-EH)^3 + D2*(ϵpA-EH)^2 + D3*(ϵpA-EH) + D4)  +  ((pA(t, ϵpA, x) * nB(t, EH, x) - 4*pA(t, ϵpA, x) + 2*nB(t, EH, x))*(E1*(ϵpA-EH)^2 + E2) + (pA(t, ϵpA, x) - nB(t, EH, x)) * (E3*(ϵpA-EH)+E4)) * exp(-(ϵpA-EH)^2/(2*sigma^2))  + pA(t, ϵpA, x) * (nB(t, EH, x)-2) * (F1*(ϵpA-EH)^3 + F2*(ϵpA-EH)) * sqrt(π) * (erf(-(ϵpA-EH)/(sqrt(2)*sigma))-1)    ) , 
        # Dt(pB(t, ϵpB, x)) ~ light_speed_coeff * 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵpB-EH)^3 + A2*(ϵpB-EH)^2 + A3*(ϵpB-EH) + A4)  +  DϵϵnB(nB(t, EH, x)) * exp(-(ϵpB-EH)^2/(2*sigma^2)) * ( (pB(t, ϵpB, x)-2)*(B1*(ϵpB-EH)^2+B2) + B3*(ϵpB-EH)^2 + B4*(ϵpB-EH) + B5) + pB(t, ϵpB, x) * DϵϵnB(nB(t, EH, x)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1) * (C1*(ϵpB-EH)^3 + C2*(ϵpB-EH)) + (pB(t, ϵpB, x) - nB(t, EH, x)) * sqrt(π) * exp(-(2*(ϵpB-EH)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵpB-EH)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵpB-EH)^3 + D2*(ϵpB-EH)^2 + D3*(ϵpB-EH) + D4)  +  ((pB(t, ϵpB, x) * nB(t, EH, x) - 4*pB(t, ϵpB, x) + 2*nB(t, EH, x))*(E1*(ϵpB-EH)^2 + E2) + (pB(t, ϵpB, x) - nB(t, EH, x)) * (E3*(ϵpB-EH)+E4)) * exp(-(ϵpB-EH)^2/(2*sigma^2))  + pB(t, ϵpB, x) * (nB(t, EH, x)-2) * (F1*(ϵpB-EH)^3 + F2*(ϵpB-EH)) * sqrt(π) * (erf(-(ϵpB-EH)/(sqrt(2)*sigma))-1)    ) , 
        ]
    

    # Choice of 3 initial conditions:
    (posWidth, energyWidth) = (1, 0.4) # (1, 3) (1, 0.4)
    T_eff = 200 # Use an effective temperature to make the transition region of the initial condition wider.

    # Initial: Gaussian energy
    # initialFunc(ϵ, x, mean) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (  normalGaussian(ϵ, mean, energyWidth) - min(normalGaussian(energyRangeNA[1], mean, energyWidth), normalGaussian(energyRangeNB[2], mean, energyWidth))  )
   
    # Initial: Fermi-Dirac energy
    initialFunc(ϵ, x, mean) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (1/(exp((ϵ-Eav)/kb/T_eff*eCharge)+1))

    # Initial: Maxwell-Boltzmann energy
    # initialFunc(ϵ, x, mean) = (normalGaussian(x, 0, posWidth) - normalGaussian(positionRange[1], 0, posWidth)) * (1/(exp((ϵ-Eav)/kb/T_eff*eCharge)))


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

        # Interface
        nB(t, energyRangeNB[1], x) ~ nA(t, energyRangeNA[2], x), 
        DϵnB(nB(t, energyRangeNB[1], x)) ~ DϵnA(nA(t, energyRangeNA[2], x)),
        pB(t, energyRangePB[1], x) ~ pA(t, energyRangePA[2], x), 
        DϵpB(pB(t, energyRangePB[1], x)) ~ DϵpA(pA(t, energyRangePA[2], x)),
        ] 

    # Solve the equations.
    domains = [t ∈ Interval(0.0, maxTime), ϵnA ∈ Interval(energyRangeNA[1], energyRangeNA[2]), ϵnB ∈ Interval(energyRangeNB[1], energyRangeNB[2]), ϵpA ∈ Interval(energyRangePA[1], energyRangePA[2]), ϵpB ∈ Interval(energyRangePB[1], energyRangePB[2]), x ∈ Interval(positionRange[1], positionRange[2])]
    @named pde_system = PDESystem(eq, bcs, domains, [t, ϵnA, ϵnB, ϵpA, ϵpB, x], [nA(t, ϵnA, x), nB(t, ϵnB, x), pA(t, ϵpA, x), pB(t, ϵpB, x)])
    discretization = MOLFiniteDifference([ϵnA => numEnergyPoints, ϵnB => numEnergyPoints, ϵpA => numEnergyPoints, ϵpB => numEnergyPoints, x => numPositionPoints], t)
    
    prob = MethodOfLines.discretize(pde_system, discretization)
    sol = solve(prob, QNDF(), saveat = maxTime/numPlots, dt=dt)

    # Get n and p solutions from sol.
    soln = hcat(sol[nA(t, ϵnA, x)], sol[nB(t, ϵnB, x)])
    solp = hcat(sol[pA(t, ϵpA, x)], sol[pB(t, ϵpB, x)])

    # Get list of t, x, ϵn and ϵp values used from sol.
    lenSolt = length(sol[t])
    solx = sol[x]
    solne = [sol[ϵnA]; sol[ϵnB]]
    solpe = [sol[ϵpA]; sol[ϵpB]]

    # Write to file.
    solDict = Dict("soln" => soln, "solp" => solp, "lenSolt" => lenSolt, "solne" => solne, "solpe" => solpe, "solx" => solx, "maxTime" => maxTime)
    save(jldFilePath, "data", solDict)

# Read from file instead of recalculating.
else
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

# Show individual plots e.g. [1,2,3,20] or []
shownPlots = []#[1, 10, 20, 30, 40, 50]
shouldUseZlims = true # use for Julia's "short-circuit evaluation"
    
for i in shownPlots
    if (i <= numPlots)
        plotn = surface(solne, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(nzmin, nzmax), legend = :none)
        shouldUseZlims && zlims!(nzmin, nzmax)

        plotp = surface(solpe, solx, transpose(solp[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="p", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(pzmin, pzmax), legend = :none)
        shouldUseZlims && zlims!(pzmin, pzmax)

        # Make two subplots.
        plotnp = Plots.plot(plotn, plotp, layout = (1,2))
        title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")
        display(plotnp)
    end
end

# Option to make gif.
makeGif = true
if makeGif
    anim = @animate for i in 1:lenSolt
        
        plotn = surface(solne, solx, transpose(soln[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="n", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(nzmin, nzmax), legend = :none)
        shouldUseZlims && zlims!(nzmin, nzmax)

        plotp = surface(solpe, solx, transpose(solp[i, :, :]), xlabel="Energy", ylabel="Position", zlabel="p", camera=cameraTup, color=reverse(cgrad(:RdYlBu_11)), clims=(pzmin, pzmax), legend = :none)
        shouldUseZlims && zlims!(pzmin, pzmax)

        # Make two subplots.
        plotnp = Plots.plot(plotn, plotp, layout = (1,2))
        title!("Time = " * Printf.format(Printf.Format("%.2e"),(i-1)/lenSolt * maxTime) * "s")
    end
    display(gif(anim, "MethodOfLines.gif", fps = floor(numPlots/gifTime)))
end
