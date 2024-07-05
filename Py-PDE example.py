# from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph

# # Expanded definition of the PDE
# diffusivity = "1.01 + tanh(x)"
# term_1 = f"({diffusivity}) * laplace(c)"
# term_2 = f"dot(gradient({diffusivity}), gradient(c))"
# eq = PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})


# grid = CartesianGrid([[-5, 5]], 64)  # generate grid
# field = ScalarField(grid, 1)  # generate initial condition

# storage = MemoryStorage()  # store intermediate information of the simulation
# res = eq.solve(field, 10, dt=1e-3, tracker=storage.tracker(1))  # solve the PDE

# times = [time for time, _ in storage.items()]
# print("Times:", times)

# plot_kymograph(storage)  # visualize the result in a space-time plot

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pde

energyRange = [-3, 3] # ±infinity but cutoff when it goes to zero
positionRange = [-3, 3] # solar cell about 10cm
numEnergyPoints = 100
numPositionPoints = 100
energies = np.linspace(energyRange[0], energyRange[1], numEnergyPoints)
positions = np.linspace(positionRange[0], positionRange[1], numPositionPoints)
small = 1e0
# grid = pde.UnitGrid([32, 32], periodic=[False, True])  # generate grid
grid = pde.CartesianGrid([energyRange, positionRange], [numEnergyPoints,numPositionPoints], periodic=[False, False])
# state = pde.ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition
heaviside = f"* heaviside( -(x-{energyRange[0]}-{small})*(x-{energyRange[1]}+{small}), 0) * heaviside( -(y-{positionRange[0]}-{small})*(y-{positionRange[1]}+{small}), 0) "
# heaviside = f"* heaviside( -(x-2)*(x+2), 0)  * heaviside( -(y-2)*(y+2), 0)"
field = pde.ScalarField.from_expression(grid, "2.718**(-(x/2)**2-(y/2)**2)"+heaviside)


# set boundary conditions `bc` for all axes

# bc_x = "dirichlet"
# bc_y = "dirichlet"
# bc_x_left = {"derivative": 0}
# bc_x_right = {"value": "sin(y / 2)"}
bc_x_left = {"value": "0"}
bc_x_right = {"value": "0"}
bc_x = [bc_x_left, bc_x_right]
bc_x = "dirichlet"
bc_y = "neumann"
# eq = DiffusionPDE(bc=[bc_x, bc_y])
eq = pde.PDE({"n": f"d_dx(d_dx(n))"}, bc=[bc_x, bc_y])               


res = eq.solve(field, t_range=0.01, dt=0.005)
# result.plot()

X, Y = np.meshgrid(energies, positions)
fig = plt.figure(figsize=(8,8))

ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, res.data.T, cmap=cm.coolwarm)
ax.set_box_aspect(aspect=None, zoom=0.9)
ax.set(xlabel="x", ylabel="y", zlabel="Electron density", title=f"")


plt.show()

    # Plot the results:
    # Decimal precision from dt.
    # decimalPlaces = 0
    # if dt < 1:
    #     decimalPlaces = len(str(dt)) - 2