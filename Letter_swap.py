# Swaps all 'Q's. Used to speed up copying the different regions in Using_MethodOfLines.jl

string = "Dt(nQ(t, ϵQ, x)) ~ DϵϵQ(nQ(t, energyRange[1], x)) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵQ, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * DxQ(nQ(t, ϵQ, x)) + K/2 * DxxQ(nQ(t, ϵQ, x)) - C*Ebar(ϵQ) * DϵQ(nQ(t, ϵQ, x)) + C*(Ebar(ϵQ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵQ(nQ(t, ϵQ, x)) )"

print("\n\n\n", string.replace("Q", "A"), "\n\n\n")

str2 = "nC(t))[1, 1], (nC(t))[2, 1], (nC(t))[3, 1], (nC(t))[4, 1], (nC(t))[5, 1], (nC(t))[6, 1], (nC(t))[7, 1], (nC(t))[7, 2], (nC(t))[7, 3], (nC(t))[7, 4], (nC(t))[7, 5], (nC(t))[7, 6], (nC(t))[1, 7], (nC(t))[2, 7], (nC(t))[3, 7], (nC(t))[4, 7], (nC(t))[5, 7], (nC(t))[6, 7], (nC(t))[7, 7], (nA(t))[1, 1], (nA(t))[2, 1], (nA(t))[3, 1], (nA(t))[4, 1], (nA(t))[5, 1], (nA(t))[6, 1], (nA(t))[7, 1], (nA(t))[1, 2], (nA(t))[6, 2], (nA(t))[7, 2], (nA(t))[1, 3], (nA(t))[6, 3], (nA(t))[7, 3], (nA(t))[1, 4], (nA(t))[6, 4], (nA(t))[7, 4], (nA(t))[1, 5], (nA(t))[6, 5], (nA(t))[7, 5], (nA(t))[1, 6], (nA(t))[6, 6], (nA(t))[7, 6], (nA(t))[1, 7], (nA(t))[2, 7], (nA(t))[3, 7], (nA(t))[4, 7], (nA(t))[5, 7], (nA(t))[6, 7], (nA(t))[7, 7], (nB(t))[1, 1], (nB(t))[2, 1], (nB(t))[3, 1], (nB(t))[4, 1], (nB(t))[5, 1], (nB(t))[6, 1], (nB(t))[7, 1], (nB(t))[1, 2], (nB(t))[2, 2], (nB(t))[1, 3], (nB(t))[2, 3], (nB(t))[1, 4], (nB(t))[2, 4], (nB(t))[1, 5], (nB(t))[2, 5], (nB(t))[1, 6], (nB(t))[2, 6], (nB(t))[1, 7], (nB(t))[2, 7], (nB(t))[3, 7], (nB(t))[4, 7], (nB(t))[5, 7], (nB(t))[6, 7], (nB(t))[7, 7]]"
ar = str2.split(", (")
ar.sort(reverse=False)

for a in ar:
    print(a)