import fipy as fp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Constants
kb = 1.3806e-23
eCharge = 1.602e-19
gamma = 0.788 # Bilayers p8 0.788
nu0 = 1
g1 = 1
sigma = 0.1 # Gaussian width of g2(e) - Tress p55 5e-2eV?, p51 0.05->0.15 eV, Traps p5 0.13eV
Ec = 0 # Gaussian centre of g2(e) - Traps p5 -5.2eV although just offsets all energy
Lambda = 9e-5 # 1e-3 # Tress p114, 1e-2 to 1e-4 (cm/V)^(1/2) CHECK AS CM, 9e-6 or 9e-5eV Alexandros
T = 300

# Variables
energyRange = [-1.0, 1.0] # ±infinity but cutoff when it goes to zero
positionRange = [-10.0, 10.0] # solar cell about 10cm
numEnergyPoints = 50
numPositionPoints = 50

dimension = 1 # accepts 1 or 2
F = np.array([1e4]) # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm

# initialField = f"{np.e}**(-(x)**2/30-(y)**2)"

K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)
dEnergy = (energyRange[1] - energyRange[0])/numEnergyPoints
dPosition = (positionRange[1] - positionRange[0])/numPositionPoints

mesh = fp.Grid2D(dx=dEnergy, dy=dPosition, nx=numEnergyPoints, ny=numPositionPoints) + ((energyRange[0],), (positionRange[0],))
# print(mesh.cellCenters)

n = fp.CellVariable(name="n", mesh=mesh)
x = mesh.cellCenters[0]
EBar = Lambda * sigmaTilde**(-2) * (2/beta * (x - Ec) + sigma**2)
SecondOrderMatrixCoeff =  K[dimension-1]/2 * fp.Variable(value=((1, 0), (0, 0))) + C[dimension-1]*(EBar**2 + 2*Lambda*sigma**2/beta*sigmaTilde**(-2)) * fp.Variable(value=((0, 0), (0, 1)))

eq = (K[dimension-1]*beta/2*F[0] * n.grad[0] - C[dimension-1]*EBar*n.grad[1] + fp.DiffusionTerm(SecondOrderMatrixCoeff) == 0)

n.constrain(0, mesh.exteriorFaces) # Set boundaries to zero.
eq.solve(var=n)
# viewer = fp.Viewer(vars=n)
# viewer.plot()


energies = mesh.x[0:numEnergyPoints]
positions = mesh.x[0:numPositionPoints]
X, Y = np.meshgrid(energies, positions)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, np.array(n).reshape(numEnergyPoints, numPositionPoints).T, cmap=cm.coolwarm)
ax.set_box_aspect(aspect=None, zoom=0.9)
ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"")
plt.show()