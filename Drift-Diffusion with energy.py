
# py-pde cartesian is up to 3 dimensions so energy uses x, then y,z left for position.

import pde # import from installing py-pde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Variables
dt = 1e-1
maxTime = 1e1
energyRange = [-3, 3] # ±infinity but cutoff when it goes to zero
positionRange = [-3, 3] # solar cell about 10cm
numEnergyPoints = 70
numPositionPoints = 70

dimension = 1 # accepts 1 or 2
F = np.array([1e5]) # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
numPlots = 10 # Number of plots, setting to 0 gives all
maxGraphsPerRow = 5

taskType = "timeEvo" # Options: timeEvo, gridSearch
plotType = "mesh" # Used for timeEvo, options: mesh, colour2d, final


# Select initial field
# initialField = "x"
# initialField = f"cos(2*{np.pi}/{positionRange[1]}*y)"
# initialField = f"{np.e}**(-(x-{energyRange[1] - energyRange[0]}/2)**2/{energyRange[1] - energyRange[0]}**2)"
# initialField = "heaviside(x, 0)*(x)"
# initialField = "heaviside( -(x)*(x-2), 0)"
# initialField = f"{np.e}**(-(x-1)**2)"

# initialField = f"cos({np.pi}/({energyRange[1]} - {energyRange[0]})*x)*cos({np.pi}/({positionRange[1]} - {positionRange[0]})*y)"
# initialField = f"cos(2*{np.pi}/({positionRange[1]} - {positionRange[0]})*y)"
# initialField = f"{np.e}**(-(x)**2/30-(y)**2)"
initialField = f"{np.e}**(-(y)**2)"
# initialField = f"(x-{energyRange[0]})*(x-{energyRange[1]})*(y-{positionRange[0]})*(y-{positionRange[1]})"


# Constants
kb = 1.3806e-23
eCharge = 1.602e-19
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.1 # Gaussian width of g2(e) - Tress p55 5e-2eV?, p51 0.05->0.15 eV, Traps p5 0.13eV
Ec = 0 # Gaussian centre of g2(e) - Traps p5 -5.2eV although just offsets all energy
Lambda = 9e-5 # 1e-3 # Tress p114, 1e-2 to 1e-4 (cm/V)^(1/2) CHECK AS CM, 9e-6 or 9e-5eV Alexandros
T = 300 # temperature, 300 Tress p63, why so cold? or p92 273+25 = 398, Traps p5 also 300
# p104? chemical potential with a distance of 0.5 eV (T = 300 K) away from the center of the Gaussian DOS
# reorganisation energy, Tress p61 - "dissociation mechanism of this state with binding energies between 0.2 and 0.5 eV is still unclear"
# p146 1.5e22 cm−3s−1 optical generation rate


def calculatePDE(dt=dt, F = F, sigma=sigma, Lambda=Lambda):
    K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
    C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
    beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
    sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)

    factor = f"{nu0}*{g1}*(2*{np.pi})**(-1/2)*{sigmaTilde}**(-2) * {np.e}**(-1/2*{sigmaTilde}**(-2) * (x - {Ec} - {Lambda})**2)"
    EBar = f"{Lambda}*{sigmaTilde}**(-2)*(2/{beta}*(x - {Ec}) + {sigma}**2)"

    if dimension == 1:
        dotGradTerm = f"{F[0]} * d_dy(n)" 
        laplaceTerm = "d_dy(d_dy(n))"
    else: # == 2
        dotGradTerm = f"{F[0]} * d_dy(n) + {F[1]} * d_dz(n)" 
        laplaceTerm = "d_dy(d_dy(n)) + d_dz(d_dz(n))"

    bc_x =  "dirichlet" # what BCs show energy have? 0 density at 0 energy? dirichlet or neumann, ["dirichlet", "dirichlet"]
    bc_y =  "periodic" # [{"value": "0"}, {"value": "0"}]
    eq = pde.PDE({"n": f"{factor} * ( {K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d_dx(d_dx(n)) )"}, bc=[bc_x, bc_y])               
    grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, True])

    # Initial field
    state = pde.ScalarField.from_expression(grid, initialField + f"* heaviside( -(x-{energyRange[0]})*(x-{energyRange[1]}), 0) * heaviside( -(y-{positionRange[0]})*(y-{positionRange[1]}), 0) ")

    storage = pde.MemoryStorage() # store intermediate information of the simulation
    res = eq.solve(state, t_range=maxTime, dt=dt, tracker=["progress", storage.tracker(dt)])  # solve the PDE
    return res, storage


# Main

if taskType == "timeEvo":
    res, storage = calculatePDE()
    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    times = [time for time, _ in storage.items()]

    # Plot the results:

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

    if plotType == "mesh":
        X, Y = np.meshgrid(energies, positions)
        numGraph = 1
        fig = plt.figure(figsize=(3*columns+1,4*rows))

        for time, field in storage.items():
            if time in desiredTimes:
                ax = fig.add_subplot(rows, columns, numGraph, projection='3d')
                # print(field.data)
                ax.plot_surface(X, Y, field.data.T, cmap=cm.coolwarm)
                ax.set_box_aspect(aspect=None, zoom=0.9)
                ax.set(xlabel="Energy", ylabel="Position", zlabel="", title=f"n, time = {round(time, decimalPlaces)}s")
                numGraph += 1

    elif plotType == "colour2d":
        fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(4.5*columns+1,2.5*rows))
        fig.suptitle("Electron density")

        # Removes extra axes
        for removingIndex in range(columns-1 - (numPlots-1)%maxGraphsPerRow):
            # print(rows-1, columns - removingIndex - 1)
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

    elif plotType == "final":
        X, Y = np.meshgrid(energies, positions)
        fig = plt.figure(figsize=(8,8))

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(X, Y, res.data.T, cmap=cm.coolwarm)
        ax.set_box_aspect(aspect=None, zoom=0.9)
        ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"")

    plt.show()


if taskType == "gridSearch":
    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    X, Y = np.meshgrid(energies, positions)
    
    # dt = 1e-1
    # F = np.array([-1e5]) # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
    # sigma = 0.13 # Gaussian width of g2(e) - Tress p55 50meV = 0.05?, p51 50-150meV, Traps p5 0.13eV
    # Lambda = 0.1e-3 # 1e-3 # Tress p114, 1e-2 to 1e-4 (cm/V)^(1/2) CHECK AS CM
    # resultsGrid = np.zeros((numDts, numSigmas)) # resultsGrid[dtIndex][sigmaIndex]

    
    # Two parameters to vary in a grid
    parameterData = [{"name": "dt", "range": [-1.7, 0], "numPoints": 6, "defaultValue": 1e-1, "isLog": True},
                     {"name": "F", "range": [1e5, 1e6], "numPoints": 6, "defaultValue": 1e5, "isLog": False},
                     {"name": "σ", "range": [0.02, 0.20], "numPoints": 3, "defaultValue": 0.13, "isLog": False},
                     {"name": "λ", "range": [-0, -6], "numPoints": 2, "defaultValue": 9e-5, "isLog": True}] #[-0, -6]
    
    # Set parameters:
    chosenDataIndices = [2, 3]

    numChosen = len(chosenDataIndices)
    numData = len(parameterData)
    points = [0] * numData
    numParamIterations = [1] * numData
    output = []

    for i in range(numData):
        if i in chosenDataIndices:
            numParamIterations[i] = parameterData[i]["numPoints"]
            if parameterData[i]["isLog"]:
                points[i] = np.logspace(parameterData[i]["range"][0], parameterData[i]["range"][1], parameterData[i]["numPoints"], base=10.0)
            else:
                points[i] = np.linspace(parameterData[i]["range"][0], parameterData[i]["range"][1], parameterData[i]["numPoints"])

    shouldPlotGrid = True
    if shouldPlotGrid == True and numChosen == 2:
        rows = parameterData[chosenDataIndices[0]]["numPoints"]
        columns = parameterData[chosenDataIndices[1]]["numPoints"]
        fig = plt.figure(figsize=(3*rows+1,4*columns))

    for i0 in range(numParamIterations[0]):
        if 0 in chosenDataIndices:
            val0 = points[0][i0]
        else:
            val0 = parameterData[0]["defaultValue"]

        for i1 in range(numParamIterations[1]):
            if 1 in chosenDataIndices:
                val1 = points[1][i1]
            else:
                val1 = parameterData[1]["defaultValue"]

            for i2 in range(numParamIterations[2]):
                if 2 in chosenDataIndices:
                    val2 = points[2][i2]
                else:
                    val2 = parameterData[2]["defaultValue"]

                for i3 in range(numParamIterations[3]):
                    if 3 in chosenDataIndices:
                        val3 = points[3][i3]
                    else:
                        val3 = parameterData[3]["defaultValue"]

                    res, storage = calculatePDE(dt=val0, F = np.array([val1]), sigma=val2, Lambda=val3)

                    timeBeforeNaN = maxTime
                    times = [time for time, _ in storage.items()]
                    fields = [field for _, field in storage.items()]
                    
                    for storageIndex in range(len(fields)):
                        # print(fields[storageIndex].data)
                        if max([max(x) for x in fields[storageIndex].data]) > 1e100:
                            timeBeforeNaN = times[storageIndex]
                            break

                    # dt, F, sigma, lambda, timeBeforeNaN
                    output.append([val0, val1, val2, val3, timeBeforeNaN])

                # WORK IN PROGRESS
                    if shouldPlotGrid == True and numChosen == 2:
                        indices = [i0, i1, i2, i3]
                        X, Y = np.meshgrid(energies, positions)
                        print(rows, columns, columns*indices[chosenDataIndices[0]]+indices[chosenDataIndices[1]]+1)
                        ax = fig.add_subplot(rows, columns, columns*indices[chosenDataIndices[0]]+indices[chosenDataIndices[1]]+1, projection='3d')
                        ax.plot_surface(X, Y, res.data.T, cmap=cm.coolwarm)
                        ax.set_box_aspect(aspect=None, zoom=0.9)
                        # {parameterData[chosenDataIndices[0]]["name"]}={points[chosenDataIndices[0]][indices[chosenDataIndices[0]]]}, {parameterData[chosenDataIndices[1]]["name"]}={points[chosenDataIndices[1]][indices[chosenDataIndices[1]]]}
                        ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"")

    
    output.sort(key=lambda x: x[-1],reverse=True)
    for arr in output:
        print(arr, ",")
    # print(output)


    if shouldPlotGrid == True and numChosen == 2:
        plt.show()


