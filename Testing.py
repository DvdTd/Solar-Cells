import pde
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# fig, axs = plt.subplots(nrows=1, ncols=15, sharex=True, sharey=True, figsize=(15,15))
# fig.subplots_adjust(hspace = .5, wspace=.001)

# axs = axs.ravel()
# print(axs)


# for i in range(7):
#     print(i,"graphs:", 5-1-(i-1)%5)

def sortSecond(val):
    return val[1] 
 
# list1 to demonstrate the use of sorting 
# using second key 
list1 = [[1,2],[2,3],[1,1]]
 
 
# sorts the array in descending according to
# second element
# list1.sort(key=lambda x: x[1],reverse=True)
# print(list1)

# print(max([1,2,3, np.nan]))


    # Plot the results:
    # Decimal precision from dt.
    # decimalPlaces = 0
    # if dt < 1:
    #     decimalPlaces = len(str(dt)) - 2

import fipy as fp
baseMesh = fp.Grid2D(dx = 1.0, dy = 1.0, nx = 2, ny = 2)
print(baseMesh.cellCenters)

translatedMesh = baseMesh + ((5,), (10,))
print(translatedMesh.cellCenters)
