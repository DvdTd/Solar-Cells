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


from pde import DiffusionPDE, ScalarField, UnitGrid

grid = UnitGrid([32, 32], periodic=[True, True])  # generate grid
state = ScalarField.random_uniform(grid, 0.2, 0.3)  # generate initial condition

field = ScalarField.from_expression(grid, "x+2*y")


# set boundary conditions `bc` for all axes

bc_x = "periodic"
bc_y = "periodic"
eq = DiffusionPDE(bc=[bc_x, bc_y])

result = eq.solve(field, t_range=10, dt=0.005)
result.plot()




    # Plot the results:
    # Decimal precision from dt.
    # decimalPlaces = 0
    # if dt < 1:
    #     decimalPlaces = len(str(dt)) - 2