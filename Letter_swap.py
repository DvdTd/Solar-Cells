# Swaps all 'Q's. Used to speed up copying the different regions in Using_MethodOfLines.jl

string = "Dt(nQ(t, ϵQ, x)) ~ DϵϵQ(nQ(t, energyRange[1], x)) + exp(-beta*E)*nu0*g1*(2*pi)^(-0.5)*sigmaTilde^(-2) * ECgaussian(ϵQ, EH+Lambda, EL+Lambda, sigmaTilde) * (  K*beta/2*F * DxQ(nQ(t, ϵQ, x)) + K/2 * DxxQ(nQ(t, ϵQ, x)) - C*Ebar(ϵQ) * DϵQ(nQ(t, ϵQ, x)) + C*(Ebar(ϵQ)^2 + 2*Lambda*sigma^2/beta*sigmaTilde^(-2)) * DϵϵQ(nQ(t, ϵQ, x)) )"

print("\n\n\n", string.replace("Q", "A"), "\n\n\n")
