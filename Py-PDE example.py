import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pde
import os

# energyRange = [-6, 6] # ±infinity but cutoff when it goes to zero
# positionRange = [-6, 6] # solar cell about 10cm
# numEnergyPoints = 100
# numPositionPoints = 100
# energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
# positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)

# fileName = "exampleData.hdf5"
# current_dir = os.path.dirname(__file__)
# filePath = os.path.join(current_dir, fileName)

# grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, False])
# # state = pde.ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition
# heaviside = ""
# print(pde.FileStorage(filePath, write_mode="read_only").data)
# field = pde.ScalarField.from_expression(grid, "2.718**(-(x/2)**2-(y/2)**2)"+heaviside)


# bc_x = "neumann"
# bc_y = "neumann"


# eq = pde.PDE({"n": f"2.718**(-(x-0.1)**2) * ( (d2_dx2(n) + d_dx(n))*(1+x/2+x**2/2) + d_dy(n) + d2_dy2(n) )"}, bc=[bc_x, bc_y])

# writer = pde.FileStorage(filePath, write_mode="truncate")
# dt = 4e-4
# res = eq.solve(field, t_range=1, dt=dt, tracker=writer.tracker(dt))
# # res = eq.solve(field, t_range=1, dt=4e-4)

# # reader = pde.FileStorage(filePath, write_mode="read_only")


# X, Y = np.meshgrid(energies, positions)
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.plot_surface(X, Y, res.data.T, cmap=cm.coolwarm)
# ax.set_box_aspect(aspect=None, zoom=0.9)
# ax.set(xlabel="x", ylabel="y", zlabel="Electron density", title=f"")
# plt.show()

from tempfile import NamedTemporaryFile

import pde

# define grid, state and pde
grid = pde.UnitGrid([32])
state = pde.FieldCollection(
    [pde.ScalarField.random_uniform(grid), pde.VectorField.random_uniform(grid)]
)
eq = pde.PDE({"s": "-0.1 * s", "v": "-v"})

# get a temporary file to write data to
fileName = "exampleData.hdf5"
current_dir = os.path.dirname(__file__)
filePath = os.path.join(current_dir, fileName)

# run a simulation and write the results
writer = pde.FileStorage(filePath, write_mode="truncate")
eq.solve(state, t_range=32, dt=0.01, tracker=writer.tracker(1))

# read the simulation back in again
reader = pde.FileStorage(filePath, write_mode="read_only")
pde.plot_kymographs(reader)
