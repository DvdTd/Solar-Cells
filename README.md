To model charge density in a solar cell, Python's Py-PDE and FiPy modules are used to solve the drift-diffusion (DD) equations including the energy terms for the time-dependent solution and the stationary state. Later, Julia's HighDimPDE and MethodOfLines packages are used to additionally calculate the light-interaction term which is a non-local term.

```Drift-Diffusion with energy.py``` uses Py-PDE to calculate the solution of the DD equations. ```longEvolutionData.json``` stores the result which can be used as an initial condition for the next run if taskType = "longEvo". ```customColours.py``` is used to make a colour gradient for the 2D colour plot.

```Stationary State.py``` calculates the stationary state of the DD equations using FiPy. However, the output currently changes depending on the resolution of the energy-position grid.

```Using_HighDimPDE.jl``` was the first attempt at adding in the light term but was stopped as it only allowed Neumann boundary conditions.

```Using_DD_MethodOfLines.jl``` implements the DD equations with Julia's MethodOfLines package. ```Using_Integral_MethodOfLines.jl``` adds the integral version of the light term and another variable Δ to integrate over. However, this couldn't evaluate the charge density at different arguments. ```Using_ABC_regions_MethodOfLines.jl``` instead adds the differential version of the light term. The energy domain was split as the solver allowed the charge density to be evaluated on the boundaries - in this case, at E<sub>H</sub> and E<sub>L</sub>. These programs write the output data into ```MethodOfLines.jld```.

```Using_Coupled_MethodOfLines.jl``` implements the coupled equations for the electron and hole densities including the differential light term. This program writes the output data into ```CoupledMethodOfLines.jld```. 

All MethodOfLines programs animate their output in ```drift_diffusion.gif```.

```Letter_swap.py``` was used to speed up writing the similar coupled equations. ```Testing.py``` and ```LearningJulia.jl``` were used to test code and online examples.




