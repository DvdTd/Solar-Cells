

# # Old code used to test for what time step the solution diverged
# elif taskType == "gridSearch":
#     energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
#     positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
#     X, Y = np.meshgrid(energies, positions)
    
#     # Two parameters to vary in a grid  # NEED TO CHANGE dt TO CLOSER TO 1e-9
#     parameterData = [{"name": "dt", "range": [-1.7, 0], "numPoints": 6, "defaultValue": 1e-1, "isLog": True},
#                      {"name": "F", "range": [1e5, 1e6], "numPoints": 6, "defaultValue": 1e5, "isLog": False},
#                      {"name": "sigma", "range": [0.02, 0.20], "numPoints": 3, "defaultValue": 0.13, "isLog": False},
#                      {"name": "λ", "range": [-0, -6], "numPoints": 2, "defaultValue": 9e-5, "isLog": True}]
    
#     # Set parameters to vary
#     chosenDataIndices = [2, 3]

#     numChosen = len(chosenDataIndices)
#     numData = len(parameterData)
#     points = [0] * numData
#     numParamIterations = [1] * numData
#     output = []

#     for i in range(numData):
#         if i in chosenDataIndices:
#             numParamIterations[i] = parameterData[i]["numPoints"]
#             if parameterData[i]["isLog"]:
#                 points[i] = np.logspace(parameterData[i]["range"][0], parameterData[i]["range"][1], parameterData[i]["numPoints"], base=10.0)
#             else:
#                 points[i] = np.linspace(parameterData[i]["range"][0], parameterData[i]["range"][1], parameterData[i]["numPoints"])

#     shouldPlotGrid = True
#     if shouldPlotGrid == True and numChosen == 2:
#         rows = parameterData[chosenDataIndices[0]]["numPoints"]
#         columns = parameterData[chosenDataIndices[1]]["numPoints"]
#         fig = plt.figure(figsize=(3*rows+1,4*columns))

#     for i0 in range(numParamIterations[0]):
#         if 0 in chosenDataIndices:
#             val0 = points[0][i0]
#         else:
#             val0 = parameterData[0]["defaultValue"]

#         for i1 in range(numParamIterations[1]):
#             if 1 in chosenDataIndices:
#                 val1 = points[1][i1]
#             else:
#                 val1 = parameterData[1]["defaultValue"]

#             for i2 in range(numParamIterations[2]):
#                 if 2 in chosenDataIndices:
#                     val2 = points[2][i2]
#                 else:
#                     val2 = parameterData[2]["defaultValue"]

#                 for i3 in range(numParamIterations[3]):
#                     if 3 in chosenDataIndices:
#                         val3 = points[3][i3]
#                     else:
#                         val3 = parameterData[3]["defaultValue"]

#                     res, storage = calculatePDE(dt=val0, F = np.array([val1]), sigma=val2, Lambda=val3)

#                     timeBeforeNaN = maxTime
#                     times = [time for time, _ in storage.items()]
#                     fields = [field for _, field in storage.items()]
                    
#                     for storageIndex in range(len(fields)):
#                         # print(fields[storageIndex].data)
#                         if max([max(x) for x in fields[storageIndex].data]) > 1e100:
#                             timeBeforeNaN = times[storageIndex]
#                             break

#                     # dt, F, sigma, lambda, timeBeforeNaN
#                     output.append([val0, val1, val2, val3, timeBeforeNaN])

#                     # graph: WORK IN PROGRESS
#                     if shouldPlotGrid == True and numChosen == 2:
#                         indices = [i0, i1, i2, i3]
#                         X, Y = np.meshgrid(energies, positions)
#                         print(rows, columns, columns*indices[chosenDataIndices[0]]+indices[chosenDataIndices[1]]+1)
#                         ax = fig.add_subplot(rows, columns, columns*indices[chosenDataIndices[0]]+indices[chosenDataIndices[1]]+1, projection='3d')
#                         ax.plot_surface(X, Y, res.data.T, cmap=cm.coolwarm)
#                         ax.set_box_aspect(aspect=None, zoom=0.9)
#                         # {parameterData[chosenDataIndices[0]]["name"]}={points[chosenDataIndices[0]][indices[chosenDataIndices[0]]]}, {parameterData[chosenDataIndices[1]]["name"]}={points[chosenDataIndices[1]][indices[chosenDataIndices[1]]]}
#                         ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"")

    
#     output.sort(key=lambda x: x[-1],reverse=True)
#     for arr in output:
#         print(arr, ",")
#     # print(output)


#     if shouldPlotGrid == True and numChosen == 2:
#         plt.show()