'''
from numba import njit
from numpy import zeros
import matplotlib.pyplot as plt
from pypde import pde_solver
from numpy import inner, array



# material constants
Qc = 1
cv = 2.5
Ti = 0.25
K0 = 250
gam = 1.4

@njit
def internal_energy(E, v, lam):
    return E - (v[0]**2 + v[1]**2 + v[2]**2) / 2 - Qc * (lam - 1)

def F(Q, d):

    r = Q[0]
    E = Q[1] / r
    v = Q[2:5] / r
    lam = Q[5] / r

    e = internal_energy(E, v, lam)

    # pressure
    p = (gam - 1) * r * e

    F_ = v[d] * Q
    F_[1] += p * v[d]
    F_[2 + d] += p

    return F_

@njit
def reaction_rate(E, v, lam):

    e = internal_energy(E, v, lam)
    T = e / cv

    return K0 if T > Ti else 0

def S(Q):

    S_ = zeros(6)

    r = Q[0]
    E = Q[1] / r
    v = Q[2:5] / r
    lam = Q[5] / r

    S_[5] = -r * lam * reaction_rate(E, v, lam)

    return S_



def energy(r, p, v, lam):
    return p / ((gam - 1) * r) + inner(v, v) / 2 + Qc * (lam - 1)

nx = 400
L = [1.]
tf = 0.5

rL = 1.4
pL = 1
vL = [0, 0, 0]
lamL = 0
EL = energy(rL, pL, vL, lamL)

rR = 0.887565
pR = 0.191709
vR = [-0.57735, 0, 0]
lamR = 1
ER = energy(rR, pR, vR, lamR)

QL = rL * array([1, EL] + vL + [lamL])
QR = rR * array([1, ER] + vR + [lamR])

Q0 = zeros([nx, 6])
for i in range(nx):
    if i / nx < 0.25:
        Q0[i] = QL
    else:
        Q0[i] = QR



out = pde_solver(Q0, tf, L, F=F, S=S, stiff=False, flux='roe', order=3)

plt.plot(out[-1, :, 0])
plt.show()
'''