import pde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from customColours import customCmap
import os
import json


# Constants
kb = 1.3806e-23
eCharge = 1.602e-19
gamma = 1e8
nu0 = 5e19
g1 = 1e14
sigma = 0.13
Ec = 0
Lambda = 9e-5
T = 300
dimension = 1

energyRange = [-1, 1]
positionRange = [-10, 10]
numEnergyPoints = 100
numPositionPoints = 100

# Drift
dt = 2e-39
maxTime = dt * 1e4
F = [1e5/eCharge]

# Still experimenting: Diffusion with new gamma
# g1 = 1
# nu0 = 1
# F = [0]  
# dt = 1.428e8 
# maxTime = dt*100000
# Values tried:
# maxTime=dt*100: 1.5e8 < dt < 1.8e8 
# dt*100000: 1.425e8 no change, 1.428e8 spikes but stable?, 1.43e8 spikes no nan,  1.45e8 nan after 8 graphs, 1.5e8 nan
# dt*100000*4: 1.428e8 nan after 8 graphs

# Diffusion - seen with different constants.
# gamma = 0.788
# g1 = 1
# nu0 = 1
# dt = 1e-5
# maxTime = 1e0
# F = [0e-2]

numPlots = 10 # Number of plots, minimum of 1.
maxGraphsPerRow = 5

taskType = "timeEvo" # Options: timeEvo, longEvo
plotType = "mesh" # Options: mesh, colour2d
shouldForceNewFile = False


# Select initial field:
# initialField = f"{np.e}**(-(x-0.4)**2)"
# initialField = f"{np.e}**(-(x)**2/30-(y)**2)"
initialField = f"{np.e}**(-(y)**2)"
# initialField = f"{np.e}**(-(y)**2/2) * (2 * {np.pi}**(-0.5))"
# initialField = f"( {np.e}**(-(x)**2/2) - {np.e}**(-({energyRange[0]})**2/2) ) * ( (y - {positionRange[0]}) / ({positionRange[1]} - {positionRange[0]})  )"

def calculatePDE(dt=dt, maxTime=maxTime, F=F, sigma=sigma, Lambda=Lambda, energyRange=energyRange, positionRange=positionRange):
    # Set up coefficient values.
    K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
    C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
    beta = 1/(kb*T) * eCharge
    sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)

    factor = f"{nu0}*{g1}*(2*{np.pi})**(-1/2)*{sigmaTilde}**(-2) * {np.e}**(-1/2*{sigmaTilde}**(-2) * (x - {Ec} - {Lambda})**2)"
    EBar = f"{Lambda}*{sigmaTilde}**(-2)*(2/{beta}*(x - {Ec}) + {sigma}**2)"

    if dimension == 1:
        dotGradTerm = f"{F[0]} * d_dy(n)" 
        laplaceTerm = "d2_dy2(n)"
    else:
        dotGradTerm = f"{F[0]} * d_dy(n) + {F[1]} * d_dz(n)" 
        laplaceTerm = "d2_dy2(n) + d2_dz2(n)"

    # Boundary conditions and equation specification.
    bc_x =  "dirichlet"
    bc_y =  "neumann" 
    eq = pde.PDE({"n": f"{factor} * ( {K[dimension-1]}*{beta}/2*{dotGradTerm} + {K[dimension-1]}/2*{laplaceTerm} - {C[dimension-1]}*{EBar}*d_dx(n) + {C[dimension-1]}*({EBar}**2 + 2*{Lambda}*{sigma}**2/{beta}*{sigmaTilde}**(-2))*d2_dx2(n) )"}, bc=[bc_x, bc_y])               
    grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, False])

    # Use initial field, or the result from running it in the past.
    if taskType == "longEvo" and json_object["pastResult"] != None:
        state = pde.ScalarField.from_state(pde.ScalarField.from_expression(grid, initialField).attributes, np.array(json_object["pastResult"]))
    else:
        state = pde.ScalarField.from_expression(grid, initialField)

    # Solve and store result.
    storage = pde.MemoryStorage()
    if numPlots == 1:
        timeBetweenRecords = maxTime
    else:
        timeBetweenRecords = maxTime/(numPlots-1)
    res = eq.solve(state, t_range=maxTime, dt=dt, tracker=["progress", storage.tracker(timeBetweenRecords)])

    return res, storage


def plotGraphs(energies, positions, res, storage, numPlots, plotType, maxGraphsPerRow=5, maxTime=maxTime, taskType=taskType):
    columns = min(numPlots, maxGraphsPerRow)
    rows = int(np.ceil(numPlots/maxGraphsPerRow))

    # Plot only the last graph, or an even spread
    if numPlots == 1:
        result = [(maxTime, res)]
    else:
        result = storage.items()

    # Option to plot a surface mesh or a 2D colour plot.
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

            # axs has different dimensions depending on the number of graphs.
            gridRow = numGraph//maxGraphsPerRow
            gridColumn = numGraph%maxGraphsPerRow
            if columns == 1:
                ax = axs
            elif rows == 1:
                ax = axs[gridColumn]
            else:
                ax = axs[gridRow, gridColumn]

            plot = ax.pcolor(energies, positions, field.data.T, cmap=customCmap(["#3399ff","#ccff66", "#000000"], 3)) # ["#3366cc", "#800000", "#009900"]
            ax.set(xlabel="Energy", ylabel="Position", title=f"n, time = {time:.2e}s")
            fig.colorbar(plot, ax=ax)
            numGraph += 1

    plt.show()


# Set up JSON storage file.
fileName = "longEvolutionData.json"
current_dir = os.path.dirname(__file__)
filePath = os.path.join(current_dir, fileName)

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
            "dt" : 5e-12,
            "maxTime" : 2.5e-5,
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
        print("Created json")



# Main calculation
numPlots = max(numPlots, 1)
if taskType == "timeEvo":
    res, storage = calculatePDE()

    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    times = [time for time, _ in storage.items()]

    plotGraphs(energies, positions, res, storage, numPlots, plotType, maxTime=maxTime, maxGraphsPerRow=maxGraphsPerRow)


elif taskType == "longEvo":
    # Read JSON and get constants
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

    print("Read json")

    res, storage = calculatePDE(dt=dt, maxTime=maxTime, F=F, sigma=sigma, Lambda=Lambda, energyRange=energyRange, positionRange=positionRange, maxGraphsPerRow=maxGraphsPerRow)

    # Update JSON with new time and results
    json_object["cumulativeTime"] += maxTime
    json_object["pastResult"] = res.data.tolist()

    with open(filePath, "w") as outfile:
        json.dump(json_object, outfile)
        print("Written to json")

    energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
    positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
    times = [time for time, _ in storage.items()]
    plotGraphs(energies, positions, res, storage, numPlots, plotType, maxTime=maxTime)
    


