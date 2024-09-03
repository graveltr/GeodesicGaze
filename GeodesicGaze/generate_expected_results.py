import numpy as np
from scipy.special import ellipkinc

def elliptic_integral_f(phi, k):
    value = ellipkinc(phi, k)
    error = 0.01
    status = 0

    return value, error, status

test_angles = [
    0.0,            # Lower boundary
    0.1,            # Small angle
    0.5,            # Mid-range angle
    1.0,            # Larger angle
    1.5,            # Near upper boundary (π/2)
    np.pi / 4      # π/4 (45 degrees)
]

test_moduli = [
    0.0,    # Lower boundary (no elliptic effect)
    0.1,    # Small modulus
    0.3,    # Small to mid-range modulus
    0.5,    # Mid-range modulus
    0.7,    # Mid to high-range modulus
    0.9    # High-range modulus
]

expected_results = [elliptic_integral_f(phi, k) for phi, k in zip(test_angles, test_moduli)]

np.savetxt('expected_results.txt', expected_results, fmt='%f')
