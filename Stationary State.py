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
sigma = 0.1 # Tress p51 0.05->0.15 eV
Ec = 0 # Traps p5 -5.2eV although just offsets all energy
Lambda = 9e-5 # 9e-6 or 9e-5eV Alexandros
T = 300 # Tress p63 300K

energyRange = [-5.0, 5.0] # ±infinity but cutoff when it goes to zero
positionRange = [-5.0, 5.0] # solar cell about 10cm
numEnergyPoints = 700
numPositionPoints = 700

dimension = 1 # accepts 1 or 2
F = np.array([1e4]) # Tress p56, reasonably strong field is 1e5 or 1e6 V/cm

K = [1/4*gamma**(-3), 3*np.pi/8*gamma**(-4), np.pi*gamma**(-5)]
C = [gamma**(-1), np.pi/2*gamma**(-2), np.pi*gamma**(-3)]
beta = 1/(kb*T) * eCharge # multiply by charge to get eV units
sigmaTilde = np.sqrt(sigma**2 + 2*Lambda/beta)
dEnergy = (energyRange[1] - energyRange[0])/numEnergyPoints
dPosition = (positionRange[1] - positionRange[0])/numPositionPoints

def PositionBC(energy):
    width = 1
    # return 1
    return np.e**(-energy**2/2*(width)**(-2))/width*(2*np.pi)**(-1/2)


mesh = fp.Grid2D(dx=dEnergy, dy=dPosition, nx=numEnergyPoints, ny=numPositionPoints) + ((energyRange[0],), (positionRange[0],))

n = fp.CellVariable(name="n", mesh=mesh) # , hasOld=True
x = mesh.cellCenters[0]
y = mesh.cellCenters[1]
EBar = Lambda * sigmaTilde**(-2) * (2/beta * (x - Ec) + sigma**2)
SecondDerivativeMatrixCoeff =  K[dimension-1]/2 * fp.Variable(value=((1, 0), (0, 0))) + C[dimension-1]*(EBar**2 + 2*Lambda*sigma**2/beta*sigmaTilde**(-2)) * fp.Variable(value=((0, 0), (0, 1)))
eq = (K[dimension-1]*beta/2*F[0] * n.grad[0] - C[dimension-1]*EBar*n.grad[1] + fp.DiffusionTerm(SecondDerivativeMatrixCoeff) == 0)

# Position BCs
n.constrain(0, mesh.facesLeft) # Set boundaries to zero.
n.constrain(PositionBC(x[0:numEnergyPoints]) - PositionBC(x[0]), mesh.facesRight)
# n.constrain(0, mesh.facesRight)

# Energy BCs
n.constrain(0, where=mesh.facesTop)
n.constrain(0, where=mesh.facesBottom)


# n.faceGrad.constrain(0, where=mesh.facesTop) # =100000 for 700x700 gives folds
# n.faceGrad.constrain(0, where=mesh.facesBottom)
# n.faceGrad.constrain(0, where=mesh.facesLeft)
# n.faceGrad.constrain(0, where=mesh.facesRight)

solver = fp.LinearLUSolver(tolerance=1e-10, iterations=10)
# eq.solve(var=n, solver=solver)

# print(np.array(mesh.y).reshape(numEnergyPoints, numPositionPoints)[:,1])

for i in range(1):
    # n.updateOld()
    res = eq.sweep(var=n, solver=solver)
    # print("res", i, "=", res)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(np.array(mesh.x).reshape(numEnergyPoints, numPositionPoints), np.array(mesh.y).reshape(numEnergyPoints, numPositionPoints), np.array(n).reshape(numEnergyPoints, numPositionPoints).T, cmap=cm.coolwarm)
    ax.set_box_aspect(aspect=None, zoom=0.9)
    ax.set(xlabel="Energy", ylabel="Position", zlabel="Electron density", title=f"")
    
    # Show position dependence is linear.
    fig = plt.figure(figsize=(7,7))
    plt.plot(np.array(mesh.y).reshape(numEnergyPoints, numPositionPoints)[:,1], np.array(n).reshape(numEnergyPoints, numPositionPoints).T[:,350])
    plt.xlabel("Position")
    plt.ylabel("Electron density")

plt.show()


"""
Current problems with FiPy
- solution dependent on points used
- Neumann conditions done automatically as adding them does nothing? making them 10000 does affect it
- changing tolerance and iterations of solver does nothing, although =0 makes the  solution 0. Means I can’t experiment with how it gets its solution. Doesn’t change an example they gave either.
- Methods of iteration, time steps and sweeps  from https://www.ctcms.nist.gov/fipy/documentation/FAQ.html - RES increases with each sweep so something is UNSTABLE
"""