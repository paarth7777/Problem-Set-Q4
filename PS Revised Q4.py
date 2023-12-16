import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh

# Constants
m = 1.0  # Mass of the particle
omega = 1.0  # Frequency of the oscillator
hbar = 1.0  # Reduced Planck's constant
L = 5.0  # Length of the box
N = 6  # Number of basis functions


def V(x):
    """Potential function """
    return 0.5 * m * omega**2 * x**2


def psi_n(x, n):
    """Trial wavefunction"""
    return np.sqrt(2/L) * np.sin(n * np.pi * x / L)


def H_ij(i, j):
    """Matrix elements of the Hamiltonian"""
    def integrand(x):
        kinetic = -0.5 * hbar**2 / m * psi_n(x, i) * (-(j * np.pi / L)**2 * psi_n(x, j))
        potential = V(x) * psi_n(x, i) * psi_n(x, j)
        return kinetic + potential
    return quad(integrand, 0, L)[0]


def S_ij(i, j):
    """Overlap matrix elements"""
    return quad(lambda x: psi_n(x, i) * psi_n(x, j), 0, L)[0]


# Hamiltonian and overlap matrices
H = np.array([[H_ij(i, j) for j in range(1, N+1)] for i in range(1, N+1)])
S = np.array([[S_ij(i, j) for j in range(1, N+1)] for i in range(1, N+1)])

eigenvalues, eigenvectors = eigh(H, S)

# Select the two lowest eigenvalues and eigenvectors
eigenvalues = eigenvalues[:2]
eigenvectors = eigenvectors[:,:2]

print(eigenvalues)
print(eigenvectors)
