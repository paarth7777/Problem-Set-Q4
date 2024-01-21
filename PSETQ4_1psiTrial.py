from scipy.optimize import minimize
import numpy as np
import sympy as sp

""" Constants """
BOX_SIZE = 1  # L
H_BAR = 1  # h
MASS = 1  # m
FREQUENCY = 1  # w

# Define the symbols
x, L = sp.symbols('x L')

# Define the integrand
integrand = x**2 * sp.cos(sp.pi * x * 1 / (2 * L))**2
integrand2 = x**2 * sp.sin(sp.pi * x * 1 / (L))**2

# Perform the integral
integral_result = sp.integrate(integrand, (x, -L, L))
integral_result2 = sp.integrate(integrand2, (x, -L, L))

# Convert the symbolic result into a callable function
integral_function = sp.lambdify((L), integral_result, 'numpy')
integral_function2 = sp.lambdify((L), integral_result2, 'numpy')


# Define the objective function to be minimized
def objective_function2(x: list):
    """Hamiltonian"""
    kinetic = (4*H_BAR**2 * np.pi**2) / (8 * MASS * x[0]**2)  # h^2 pi^2 / 2mL^2
    potential = (MASS * (FREQUENCY**2) * integral_function2(x[0])) / (2 * x[0])
    # <H> = <T> + <V>
    return kinetic + potential


def objective_function1(x: list):
    """Hamiltonian"""
    kinetic = (H_BAR**2 * np.pi**2) / (8 * MASS * x[0]**2)  # h^2 pi^2 / 8mL^2
    potential = (MASS * (FREQUENCY**2) * integral_function(x[0])) / (2 * x[0])
    # <H> = <T> + <V>
    return kinetic + potential


# Initial guess
def initial_guess():
    """creates initial guesses for all coefficients with equal weight"""
    return [1]


# Perform the constrained optimization
result1 = minimize(objective_function1, initial_guess())
result2 = minimize(objective_function2, initial_guess())

print(result1)
print(result2)
