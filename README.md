To find the occupation probability in a solar cell in terms of time, position and energy, Python's Py-PDE and FiPy modules are used to solve the drift-diffusion (DD) equations for the time-dependent solution and the stationary state. Julia's MethodOfLines package is used to additionally solve the light-interaction term which is non-local and non-linear. ```Using_Coupled_MethodOfLines.jl``` is the most complete implementation.


```Drift-Diffusion_with_energy.py``` uses Py-PDE to calculate the solution of the DD equations. ```longEvolutionData.json``` can store the result to be used as an initial condition for the next run if taskType = "longEvo". ```customColours.py``` is used to make a colour gradient for the 2D colour plot.

```Stationary_State.py``` calculates the stationary state of the DD equations using FiPy although it currently doesn't work for the correct gamma value.

```Using_HighDimPDE.jl``` was the first attempt at adding in the light term but was stopped as it only allowed Neumann boundary conditions.

```Using_DD_MethodOfLines.jl``` implements the DD equations with Julia's MethodOfLines package. ```Using_Integral_MethodOfLines.jl``` adds the integral version of the light term and another variable Δ to integrate over. However, this couldn't evaluate the non-local occupation probability. ```Using_ABC_regions_MethodOfLines.jl``` instead adds the differential version of the light term. The energy domain was split as the solver allowed the occupation probability to be evaluated on the boundaries - in this case, at E<sub>H</sub> and E<sub>L</sub>. These programs write the output data into ```MethodOfLinesData.jld```.

```Using_Coupled_MethodOfLines.jl``` implements the coupled equations including the differential light term. This program writes the output data into ```CoupledMethodOfLinesData.jld```. 

```Using_DD_MethodOfLines.jl``` animates its output in ```MethodOfLinesDD.gif``` and all other MethodOfLines programs animate their output in ```MethodOfLines.gif```.




