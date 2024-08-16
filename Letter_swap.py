# Swaps letters. Used to speed up copying the different regions in Using_MethodOfLines.jl and coupled equations.

string = "Dt(nA(t, ϵnA, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnA-EL)^3 + A2*(ϵnA-EL)^2 + A3*(ϵnA-EL) + A4)  +  DϵϵnB(nB(t, EL, x)) * exp(-(ϵnA-EL)^2/(2*sigma^2)) * ( (nA(t, ϵnA, x)-2)*(B1*(ϵnA-EL)^2+B2) + B3*(ϵnA-EL)^2 + B4*(ϵnA-EL) + B5) + nA(t, ϵnA, x) * DϵϵnB(nB(t, EL, x)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnA-EL)^3 + C2*(ϵnA-EL)) + (nA(t, ϵnA, x) - nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnA-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnA-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnA-EL)^3 + D2*(ϵnA-EL)^2 + D3*(ϵnA-EL) + D4)  +  ((nA(t, ϵnA, x) * nB(t, EL, x) - 4*nA(t, ϵnA, x) + 2*nB(t, EL, x))*(E1*(ϵnA-EL)^2 + E2) + (nA(t, ϵnA, x) - nB(t, EL, x)) * (E3*(ϵnA-EL)+E4)) * exp(-(ϵnA-EL)^2/(2*sigma^2))  + nA(t, ϵnA, x) * (nB(t, EL, x)-2) * (F1*(ϵnA-EL)^3 + F2*(ϵnA-EL)) * sqrt(π) * (erf(-(ϵnA-EL)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnA-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nA(t, ϵnA, x)) + K/2 * Dxx(nA(t, ϵnA, x)) - C*Ebar(ϵnA, EH) * DϵnA(nA(t, ϵnA, x)) + C*(Ebar(ϵnA, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnA(nA(t, ϵnA, x)) ),"
string2 = "Dt(nB(t, ϵnB, x)) ~ 1/(2*sqrt(π) * hbar^2 * kbT) * ( DϵϵnB(nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT-sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (A1*(ϵnB-EL)^3 + A2*(ϵnB-EL)^2 + A3*(ϵnB-EL) + A4)  +  DϵϵnB(nB(t, EL, x)) * exp(-(ϵnB-EL)^2/(2*sigma^2)) * ( (nB(t, ϵnB, x)-2)*(B1*(ϵnB-EL)^2+B2) + B3*(ϵnB-EL)^2 + B4*(ϵnB-EL) + B5) + nB(t, ϵnB, x) * DϵϵnB(nB(t, EL, x)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1) * (C1*(ϵnB-EL)^3 + C2*(ϵnB-EL)) + (nB(t, ϵnB, x) - nB(t, EL, x)) * sqrt(π) * exp(-(2*(ϵnB-EL)*kbT+sigma^2)/(2*kbT^2)) * (1 - erf(-((ϵnB-EL)*kbT-sigma^2)/(sqrt(2)*sigma*kbT))) * (D1*(ϵnB-EL)^3 + D2*(ϵnB-EL)^2 + D3*(ϵnB-EL) + D4)  +  ((nB(t, ϵnB, x) * nB(t, EL, x) - 4*nB(t, ϵnB, x) + 2*nB(t, EL, x))*(E1*(ϵnB-EL)^2 + E2) + (nB(t, ϵnB, x) - nB(t, EL, x)) * (E3*(ϵnB-EL)+E4)) * exp(-(ϵnB-EL)^2/(2*sigma^2))  + nB(t, ϵnB, x) * (nB(t, EL, x)-2) * (F1*(ϵnB-EL)^3 + F2*(ϵnB-EL)) * sqrt(π) * (erf(-(ϵnB-EL)/(sqrt(2)*sigma))-1)    )       +    exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * exp(-(ϵnB-EH)^2*0.5*sigmaTilde^(-2)) * (  K*beta/2*F * Dx(nB(t, ϵnB, x)) + K/2 * Dxx(nB(t, ϵnB, x)) - C*Ebar(ϵnB, EH) * DϵnB(nB(t, ϵnB, x)) + C*(Ebar(ϵnB, EH)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵnB(nB(t, ϵnB, x)) ),"

for string in [string, string2]:
    string = string.replace("EL", "XXX")
    string = string.replace("EH", "EL")
    string = string.replace("XXX", "EH")

    string = string.replace("nA", "pA")
    string = string.replace("nB", "pB")

    # string = string.replace("ϵnA", "ϵpA")
    # string = string.replace("ϵnB", "ϵpB")


    print("\n\n\n", string, "\n\n\n")
