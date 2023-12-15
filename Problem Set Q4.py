import numpy as np
from scipy.linalg import eigh

# Constants
hbar = 1.0545718e-34
m = 9.10938356e-31
L = 1e-10             # Size of the box in meters
N = 10                # Number of basis states to use
omega = 5e15          # Frequency of the harmonic oscillator in Hz


def V(x):
    """Define the potential energy function of the harmonic oscillator"""
    return 0.5 * m * omega**2 * x**2


# Define the x values within the box
x_values = np.linspace(0, L, 1000)

# Define the matrix elements for the Hamiltonian in the basis of particle in a box eigenstates
H = np.zeros((N, N))

for n in range(N):
    for m in range(N):
        # Diagonal elements (kinetic + potential energy)
        if n == m:
            kinetic = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
            potential = V(L/2)  # approximated by evaluating V at the midpoint of the box
            H[n, m] = kinetic + potential
        # Off-diagonal elements (potential energy)
        else:
            # Integral of the potential energy over the eigenfunctions
            # For simplicity, approximate this by a sum over midpoints
            integral = sum(V(x) * np.sin((n+1) * np.pi * x / L) * np.sin((m+1) * np.pi * x / L) for x in x_values)
            H[n, m] = integral * L / len(x_values)


eigenvalues, eigenvectors = eigh(H)

# Convert eigenvalues from J to atomic units (1 a.u. of energy = 4.359744650e-18 J)
eigenvalues_au = eigenvalues / 4.359744650e-18

# Return the two lowest eigenvalues and their corresponding eigenvectors
lowest_eigenvalues = eigenvalues_au[:2]
lowest_eigenvectors = eigenvectors[:, :2]
