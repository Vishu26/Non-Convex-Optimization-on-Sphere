import numpy as np
from scipy.optimize import LinearConstraint, minimize


def double_derivate_haversine(x):
    return -(2 / (x[0] ** 2) * np.cos(2 * np.pi / x[0]))


def derive_haversine_constant():
    linear_constraint = LinearConstraint([1], [4], [np.inf])
    c = minimize(double_derivate_haversine, x0=[10], constraints=linear_constraint)
    return c.x[0]
