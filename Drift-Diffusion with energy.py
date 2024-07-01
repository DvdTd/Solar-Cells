
# pypde cartesian is up to 3 dimensions, energy takes one, so effectively only two.
# Let x = epsilon so y and/or z can be used for position
# Can't use grads as they involve energy

# might need to make more things with type field

import pde
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


# Variables
dt = 0.01
maxTime = 1000
energyRange = [0, 8]
positionRange = [0, 8]
numEnergyPoints = 60
numPositionPoints = 60

dimension = 1 # accepts 1 or 2
F = np.array([1]) # CHANGE, Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
# Number of plots, setting to 0 gives all
numPlots = 3
maxGraphsPerRow = 5

# Constants
kb = 1.3806e-23
gamma = 1 
nu0 = 1
g1 = 1
sigma = 1 # Gaussian width of g2(e) - Tress p55 50meV?, p51 50-150meV, 0.1*1.6e-19 (made it hardly change in 70000sec)
sigmaBar = 1 
Ec = 0 # Gaussian centre of g2(e)
Lambda = 1e-3 # Tress p114, 1e-2 to 1e-4 (cm/V)^(1/2) CHECK AS CM
T = 300 # temperature, 300 Tress p63, why so cold? or p92 273+25 = 398 

# p104? chemical potential with a distance of 0.5 eV (T = 300 K) away from the center of the Gaussian DOS
# reorganisation energy, Tress p61 - "dissociation mechanism of this state with binding energies between 0.2 and 0.5 eV is still unclear"

K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
beta = 1/(kb*T)
sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta) # BETTER TO SQUARE?



# Main
factor = f"{nu0}*{g1}*(2*{np.pi})**(-1/2)*{sigmaTilde}**(-2) * {np.e}**(-1/2*{sigmaTilde}**(-2) * (x - {Ec} - {Lambda})**2)"
EBar = f"{Lambda}*{sigmaBar}**(-2)*(2/{beta}*(x - {Ec}) + {sigma}**2)"

dotGradTerm = f"{F[0]} * d_dy(n)" 
laplaceTerm = "d_dy(d_dy(n))"
# dotGradTerm = f"{F[0]} * d_dy(n) + {F[1]} * d_dz(n)" 
# laplaceTerm = "d_dy(d_dy(n)) + d_dz(d_dz(n))"

eq = pde.PDE({"n": f"{factor} * ({K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d_dx(d_dx(n)))"})               
# could add: , bc={"value": 0}

grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints])  # generate grid

# Initial field
# field = pde.ScalarField.from_expression(grid, "x")
# field = pde.ScalarField(grid, 1)
# field = pde.ScalarField.from_expression(grid, "sin(x)")
field = pde.ScalarField.from_expression(grid, "(2.718)**(-(x-1)**2)")

storage = pde.MemoryStorage()  # store intermediate information of the simulation
res = eq.solve(field, maxTime, dt=dt, tracker=["progress", storage.tracker(1)])  # solve the PDE
# for time, field in storage.items():
#     print(f"t={time}, field={field.magnitude}, field={field.data}\n")
# pde.plot_kymograph(storage)  # visualize the result in a space-time plot


# Surface plot of results

energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
times = [time for time, _ in storage.items()]


# Limit the number of plots to the maximum
if numPlots > len(times) or numPlots == 0:
    numPlots = len(times)

desiredTimes = np.linspace(0, maxTime, numPlots)
length = len(desiredTimes)

# Change desiredTimes so it lines up with the actual times
i=0
for j in range(len(times)): 
    if i<numPlots and desiredTimes[i] <= times[j]:
        desiredTimes[i] = times[j]
        i+=1

print(desiredTimes)

# Plot graphs
X, Y = np.meshgrid(energies, positions)
numGraph = 1
columns = min(length, maxGraphsPerRow)
rows = int(np.ceil(length/5))
fig = plt.figure(figsize=(3*columns+1,4*rows))

for time, field in storage.items():
    if time in desiredTimes:
        ax = fig.add_subplot(rows, columns, numGraph, projection='3d')
        ax.plot_surface(X, Y, field.data.T, cmap=cm.coolwarm)
        ax.set_box_aspect(aspect=None, zoom=0.9)
        ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"Time = {round(time)}s")
        numGraph += 1

plt.show()