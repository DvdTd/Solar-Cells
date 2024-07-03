
# pypde cartesian is up to 3 dimensions, energy takes one, so effectively only two.
# Let x = epsilon so y and/or z can be used for position
# Can't use grads as they involve energy

import pde
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


# Variables
dt = 1e0
maxTime = 1e4
energyRange = [-3, 3] # ±infinity but cutoff when it goes to zero
positionRange = [-3, 3] # solar cell about 10cm
numEnergyPoints = 100
numPositionPoints = 100

dimension = 1 # accepts 1 or 2
F = np.array([-1e5]) # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
numPlots = 10 # Number of plots, setting to 0 gives all
maxGraphsPerRow = 5

shouldUseBCs = True
shouldPlotMesh = True

# Select initial field
# initialField = "1"
# initialField = "x"
# initialField = f"cos(2*{np.pi}/{positionRange[1]}*y)"
# initialField = f"{np.e}**(-(x-{energyRange[1] - energyRange[0]}/2)**2/{energyRange[1] - energyRange[0]}**2)"
# initialField = "heaviside(x, 0)*(x)"
# initialField = "heaviside( -(x)*(x-2), 0)"
initialField = f"{np.e}**(-(x)**2)"
# initialField = f"{np.e}**(-(x-1)**2)"

# initialField = "x+y"
# initialField = f"cos({np.pi}/({energyRange[1]} - {energyRange[0]})*x)*cos({np.pi}/({positionRange[1]} - {positionRange[0]})*y)"
# initialField = f"{np.e}**(-(x)**2/30-(y)**2)"
# initialField = f"{np.e}**(-(y)**2)"
# initialField = f"(x-{energyRange[0]})*(x-{energyRange[1]})*(y-{positionRange[0]})*(y-{positionRange[1]})"


# Constants
kb = 1.3806e-23
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.13 # Gaussian width of g2(e) - Tress p55 50meV?, p51 50-150meV, Traps p5 0.13eV
Ec = 0 # Gaussian centre of g2(e) - Traps p5 -5.2eV although just offsets all energy
Lambda = 1e-3 # Tress p114, 1e-2 to 1e-4 (cm/V)^(1/2) CHECK AS CM
T = 300 # temperature, 300 Tress p63, why so cold? or p92 273+25 = 398, Traps p5 also 300
# p104? chemical potential with a distance of 0.5 eV (T = 300 K) away from the center of the Gaussian DOS
# reorganisation energy, Tress p61 - "dissociation mechanism of this state with binding energies between 0.2 and 0.5 eV is still unclear"
# p146 1.5e22 cm−3s−1 optical generation rate


K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
beta = 1/(kb*T)
sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)


# Main
factor = f"{nu0}*{g1}*(2*{np.pi})**(-1/2)*{sigmaTilde}**(-2) * {np.e}**(-1/2*{sigmaTilde}**(-2) * (x - {Ec} - {Lambda})**2)"
EBar = f"{Lambda}*{sigmaTilde}**(-2)*(2/{beta}*(x - {Ec}) + {sigma}**2)"

dotGradTerm = f"{F[0]} * d_dy(n)" 
laplaceTerm = "d_dy(d_dy(n))"
# dotGradTerm = f"{F[0]} * d_dy(n) + {F[1]} * d_dz(n)" 
# laplaceTerm = "d_dy(d_dy(n)) + d_dz(d_dz(n))"

if shouldUseBCs == True:
    bc_x =  "dirichlet" # what BCs show energy have? 0 density at 0 energy? dirichlet or neumann, ["dirichlet", "dirichlet"]
    bc_y =  "periodic" # [{"value": "0"}, {"value": "0"}]
    eq = pde.PDE({"n": f"{factor} * ( {K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d_dx(d_dx(n)) )"}, bc=[bc_x, bc_y])               
else:
    eq = pde.PDE({"n": f"{factor} * ( {K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d_dx(d_dx(n)) )"})               

grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, True])

# Initial field
field = pde.ScalarField.from_expression(grid, initialField + f"* heaviside( (x-{energyRange[0]})*(x-{energyRange[1]})*(y-{positionRange[0]})*(y-{positionRange[1]}), 0) ")


# Constant IC remains constant for all time as all derivatives are 0.
# field = pde.ScalarField(grid, 1)


storage = pde.MemoryStorage()  # store intermediate information of the simulation
res = eq.solve(field, maxTime, dt=dt, tracker=["progress", storage.tracker(dt)])  # solve the PDE


# Plot the results:

energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
times = [time for time, _ in storage.items()]

# Limit the number of plots to the maximum.
if numPlots > len(times) or numPlots == 0:
    numPlots = len(times)
    print("Showing all times")

# Find evenly space times then round down to the nearest actual time.
desiredTimes = np.linspace(0, maxTime, numPlots)
length = len(desiredTimes)
i=0
for j in range(len(times)): 
    if i<numPlots and desiredTimes[i] <= times[j]:
        desiredTimes[i] = times[j]
        i+=1

# Decimal precision from dt.
decimalPlaces = 0
if dt < 1:
    decimalPlaces = len(str(dt)) - 2

# Plot graphs
columns = min(length, maxGraphsPerRow)
rows = int(np.ceil(length/5))

if shouldPlotMesh == True:
    X, Y = np.meshgrid(energies, positions)
    numGraph = 1
    fig = plt.figure(figsize=(3*columns+1,4*rows))

    for time, field in storage.items():
        if time in desiredTimes:
            ax = fig.add_subplot(rows, columns, numGraph, projection='3d')
            # print(field.data)
            ax.plot_surface(X, Y, field.data.T, cmap=cm.coolwarm)
            ax.set_box_aspect(aspect=None, zoom=0.9)
            ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"Time = {round(time, decimalPlaces)}s")
            numGraph += 1

else:
    fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(4.5*columns+1,2.5*rows))
    fig.suptitle("Electron density")

    # Removes extra axes
    for removingIndex in range(columns-1 - (numPlots-1)%maxGraphsPerRow):
        print(rows-1, columns - removingIndex - 1)
        axs[rows-1, columns - removingIndex - 1].remove()

    numGraph = 0

    for time, field in storage.items():
        if time in desiredTimes:
            gridRow = numGraph//maxGraphsPerRow
            gridColumn = numGraph%maxGraphsPerRow

            # axs has different dimensions depending on the number of graphs.
            if columns == 1:
                ax = axs
            elif rows == 1:
                ax = axs[gridColumn]
            else:
                ax = axs[gridRow, gridColumn]

            plot = ax.pcolor(energies, positions, field.data.T, cmap=cm.viridis)
            ax.set(xlabel="Energy", ylabel="Position", title=f"Time = {round(time, decimalPlaces)}s")
            fig.colorbar(plot, ax=ax) #, format='%.1e'

            numGraph += 1

plt.show()