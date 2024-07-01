'''
import sesame
import numpy as np

L = 3e-4 # length of the system in the x-direction [cm]
x = np.concatenate((np.linspace(0,1.2e-4, 100, endpoint=False),     # depletion region
                    np.linspace(1.2e-4, L, 50)))                    # neutral region

sys = sesame.Builder(x)


material = {'Nc':8e17, 'Nv':1.8e19, 'Eg':1.5, 'affinity':3.9, 'epsilon':9.4,
        'mu_e':100, 'mu_h':100, 'Et':0, 'tau_e':10e-9, 'tau_h':10e-9, 'Et':0}

sys.add_material(material)
junction = 50e-7 # extent of the junction from the left contact [cm]

def n_region(pos):
    x = pos
    return x < junction
# Add the donors
nD = 1e17 # [cm^-3]
sys.add_donor(nD, n_region)
def p_region(pos):
    x = pos
    return x >= junction

# Add the acceptors
nA = 1e15 # [cm^-3]
sys.add_acceptor(nA, p_region)
# Define Ohmic contacts
sys.contact_type('Ohmic', 'Ohmic')

# Define the surface recombination velocities for electrons and holes [cm/s]
Sn_left, Sp_left, Sn_right, Sp_right = 1e7, 0, 0, 1e7  # cm/s
sys.contact_S(Sn_left, Sp_left, Sn_right, Sp_right)
phi = 1e17       # photon flux [1/(cm^2 s)]
alpha = 2.3e4    # absorption coefficient [1/cm]

# Define a function for the generation rate
def gfcn(x):
    return phi * alpha * np.exp(-alpha * x)
sys.generation(gfcn)
voltages = np.linspace(0, 0.95, 40)
j = sesame.IVcurve(sys, voltages, '1dhomo_V')
j = j * sys.scaling.current
result = {'v':voltages, 'j':j}

np.save('jv_values', result)
result = np.load("jv_values.npy")

# np.savetxt('jv_values.txt', (v, j))
import matplotlib.pyplot as plt
plt.plot(voltages, j, '-o')
plt.xlabel('Voltage [V]')
plt.ylabel('Current [A/cm^2]')
plt.grid()      # add grid
plt.show()      # show the plot on the screen
'''