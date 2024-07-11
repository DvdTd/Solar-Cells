import pde
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import fipy as fp


# examples.convection.exponential1DSource.mesh1D from 
# https://www.ctcms.nist.gov/fipy/examples/convection/generated/examples.convection.exponential1DSource.mesh1D.html

diffCoeff = 1.
convCoeff = (10.,)
sourceCoeff = 1.
from fipy import CellVariable, Grid1D, DiffusionTerm, ExponentialConvectionTerm, DefaultAsymmetricSolver, Viewer
from fipy.tools import numerix
nx = 1000
L = 10.
mesh = Grid1D(dx=L / 1000, nx=nx)
valueLeft = 0.
valueRight = 1.
var = CellVariable(name="variable", mesh=mesh)
var.constrain(valueLeft, mesh.facesLeft)
var.constrain(valueRight, mesh.facesRight)
eq = (DiffusionTerm(coeff=diffCoeff)+ ExponentialConvectionTerm(coeff=convCoeff)+ sourceCoeff)
eq.solve(var=var,solver=DefaultAsymmetricSolver(tolerance=1.e-15, iterations=10000))
axis = 0
x = mesh.cellCenters[axis]
AA = -sourceCoeff * x / convCoeff[axis]
BB = 1. + sourceCoeff * L / convCoeff[axis]
CC = 1. - numerix.exp(-convCoeff[axis] * x / diffCoeff)
DD = 1. - numerix.exp(-convCoeff[axis] * L / diffCoeff)
analyticalArray = AA + BB * CC / DD
print(var.allclose(analyticalArray, rtol=1e-4, atol=1e-4))

# viewer = Viewer(vars=var)
# viewer.plot()

# ADDED AS VIEWER NOT WORKING
plt.plot(mesh.x, var)
plt.show()

