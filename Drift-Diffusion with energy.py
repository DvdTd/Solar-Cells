
# py-pde cartesian is up to 3 dimensions so energy uses x, then y,z left for position.

import pde # import from installing py-pde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import json


# Constants
kb = 1.3806e-23
eCharge = 1.602e-19
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.1 # Tress p51 0.05->0.15 eV
Ec = 0 # Traps p5 -5.2eV although just offsets all energy
Lambda = 9e-5 # 9e-6 or 9e-5eV Alexandros
T = 300 # Tress p63 300K

#dt = 1e-11, maxTime = 5e-7 takes 25 sec
#dt = 1e-11, maxTime = 5e-6 takes 90 sec
#dt = 1e-11, maxTime = 5e-5 takes 850 sec = 14 min 12 sec
dt = 1e-11
maxTime = 5e-8
energyRange = [-1, 1] # ±infinity but cutoff when it goes to zero
positionRange = [-10, 10] # solar cell about 10cm
numEnergyPoints = 100
numPositionPoints = 100

dimension = 1 # accepts 1 or 2
F = [1e5] # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm
numPlots = 8 # Number of plots, minimum of 1.
maxGraphsPerRow = 5

taskType = "longEvo" # Options: timeEvo, gridSearch, longEvo
plotType = "colour2d" # Used for timeEvo and longEvo, options: mesh, colour2d
shouldForceNewFile = False

# Select initial field
# initialField = f"{np.e}**(-(x)**2/30-(y)**2)"
initialField = f"{np.e}**(-(y)**2)"
# initialField = f"cos(2*{np.pi}/({positionRange[1]} - {positionRange[0]})*y)"
# initialField = f"( {np.e}**(-(x)**2/2) - {np.e}**(-({energyRange[0]})**2/2) ) * ( (y - {positionRange[0]}) / ({positionRange[1]} - {positionRange[0]})  )"


def calculatePDE(dt=dt, F = F, sigma=sigma, Lambda=Lambda):
    K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
    C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
    beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
    sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)

    factor = f"{nu0}*{g1}*(2*{np.pi})**(-1/2)*{sigmaTilde}**(-2) * {np.e}**(-1/2*{sigmaTilde}**(-2) * (x - {Ec} - {Lambda})**2)"
    EBar = f"{Lambda}*{sigmaTilde}**(-2)*(2/{beta}*(x - {Ec}) + {sigma}**2)"

    if dimension == 1:
        dotGradTerm = f"{F[0]} * d_dy(n)" 
        laplaceTerm = "d2_dy2(n)"
    else:
        dotGradTerm = f"{F[0]} * d_dy(n) + {F[1]} * d_dz(n)" 
        laplaceTerm = "d2_dy2(n) + d2_dz2(n)"

    bc_x =  "dirichlet"
    bc_y =  "neumann" 
    eq = pde.PDE({"n": f"{factor} * ( {K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d2_dx2(n) )"}, bc=[bc_x, bc_y])               
    grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, False])

    # Use initial field, or the result of running it in the past.
    if taskType == "longEvo" and json_object["pastResult"] != None:
        state = pde.ScalarField.from_state(pde.ScalarField.from_expression(grid, initialField).attributes, np.array(json_object["pastResult"]))
    else:
        state = pde.ScalarField.from_expression(grid, initialField)

    storage = pde.MemoryStorage()
    if numPlots == 1:
        timeBetweenRecords = maxTime
    else:
        timeBetweenRecords = maxTime/(numPlots-1)
    res = eq.solve(state, t_range=maxTime, dt=dt, tracker=["progress", storage.tracker(timeBetweenRecords)])

    return res, storage


def plotGraphs(energies, positions, res, storage, numPlots, plotType, maxGraphsPerRow=5, maxTime=maxTime, taskType=taskType):
    columns = min(numPlots, maxGraphsPerRow)
    rows = int(np.ceil(numPlots/5))

    if numPlots == 1:
        result = [(maxTime, res)]
    else:
        result = storage.items()

    if plotType == "mesh":
        X, Y = np.meshgrid(energies, positions)
        numGraph = 1
        fig = plt.figure(figsize=(3*columns+1,4*rows))

        for time, field in result:
            if taskType == "longEvo":
                time += json_object["cumulativeTime"] - maxTime
            ax = fig.add_subplot(rows, columns, numGraph, projection='3d')
            ax.plot_surface(X, Y, field.data.T, cmap=cm.coolwarm)
            ax.set_box_aspect(aspect=None, zoom=0.9)
            ax.set(xlabel="Energy", ylabel="Position", zlabel="", title=f"n, time = {time:.2e}s")
            numGraph += 1 

    elif plotType == "colour2d":
        fig, axs = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True, figsize=(4.5*columns+1,2.5*rows))

        # Removes extra axes
        for removingIndex in range(columns-1 - (numPlots-1)%maxGraphsPerRow):
            axs[rows-1, columns - removingIndex - 1].remove()

        numGraph = 0

        for time, field in storage.items():
            if taskType == "longEvo":
                time += json_object["cumulativeTime"] - maxTime

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
            ax.set(xlabel="Energy", ylabel="Position", title=f"n, time = {time:.2e}s")
            fig.colorbar(plot, ax=ax)

            numGraph += 1

    plt.show()
    
    
fileName = "longEvolutionData.json"
current_dir = os.path.dirname(__file__)
filePath = os.path.join(current_dir, fileName)

# Create new JSON
if not os.path.isfile(filePath) or shouldForceNewFile:
    jsonDict = {
        "constants": {
            "kb" : 1.3806e-23,
            "eCharge" : 1.602e-19,
            "gamma" : 0.788,
            "nu0" : 1,
            "g1" : 1,
            "sigma" : 0.1,
            "Ec" : 0 ,
            "Lambda" : 9e-5 ,
            "T" : 300,
            "dt" : 1e-11,
            "maxTime" : 5e-7,
            "energyRange" : [-1, 1],
            "positionRange" : [-10, 10],
            "numEnergyPoints" : 100,
            "numPositionPoints" : 100,
            "dimension" : 1 ,
            "F" : [1e5]
        },
        "cumulativeTime" : 0,
        "pastResult" : None
    }
    json_object = json.dumps(jsonDict, indent=4)
    with open(filePath, "w") as outfile:
        outfile.write(json_object)

# read JSON and get constants
if taskType == "longEvo":
    with open(filePath, 'r') as openfile:
        json_object = json.load(openfile)
    
    kb = json_object["constants"]["kb"]
    eCharge = json_object["constants"]["eCharge"]
    gamma = json_object["constants"]["gamma"]
    nu0 = json_object["constants"]["nu0"]
    g1 = json_object["constants"]["g1"]
    sigma = json_object["constants"]["sigma"]
    Ec = json_object["constants"]["Ec"]
    Lambda = json_object["constants"]["Lambda"]
    T = json_object["constants"]["T"]
    dt = json_object["constants"]["dt"]
    maxTime = json_object["constants"]["maxTime"]
    energyRange = json_object["constants"]["energyRange"]
    positionRange = json_object["constants"]["positionRange"]
    numEnergyPoints = json_object["constants"]["numEnergyPoints"]
    numPositionPoints = json_object["constants"]["numPositionPoints"]
    dimension = json_object["constants"]["dimension"]
    F = json_object["constants"]["F"]


# Main calculation
numPlots = max(numPlots, 1)
if taskType == "timeEvo":
    res, storage = calculatePDE()

    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    times = [time for time, _ in storage.items()]

    plotGraphs(energies, positions, res, storage, numPlots, plotType)


elif taskType == "longEvo":
    res, storage = calculatePDE()

    # Update JSON with new time and results
    json_object["cumulativeTime"] += maxTime
    json_object["pastResult"] = res.data.tolist()

    with open(filePath, "w") as outfile:
        json.dump(json_object, outfile)

    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    times = [time for time, _ in storage.items()]

    plotGraphs(energies, positions, res, storage, numPlots, plotType)





elif taskType == "gridSearch": # NEED TO CHANGE dt TO CLOSER TO 1e-9
    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    X, Y = np.meshgrid(energies, positions)
    
    # Two parameters to vary in a grid
    parameterData = [{"name": "dt", "range": [-1.7, 0], "numPoints": 6, "defaultValue": 1e-1, "isLog": True},
                     {"name": "F", "range": [1e5, 1e6], "numPoints": 6, "defaultValue": 1e5, "isLog": False},
                     {"name": "sigma", "range": [0.02, 0.20], "numPoints": 3, "defaultValue": 0.13, "isLog": False},
                     {"name": "λ", "range": [-0, -6], "numPoints": 2, "defaultValue": 9e-5, "isLog": True}]
    
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

                    # graph: WORK IN PROGRESS
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