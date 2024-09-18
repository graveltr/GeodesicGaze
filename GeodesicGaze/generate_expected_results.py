import numpy as np
import scipy.special as sp

def elliptic_integral_f(phi, m):
    value = sp.ellipkinc(phi, m)
    error = 0.01
    status = 0

    return value, error, status

def elliptic_integral_e(phi, m):
    value = sp.ellipeinc(phi, m)
    error = 0.01
    status = 0

    return value, error, status

def elliptic_integral_p(phi, m, n):
    # value = sp.elliprf(0., 1. - k, 1.) + (phi / 3.) * sp.elliprj(0., 1. - k, 1., 1. - phi)
    term1 = np.sin(phi) * sp.elliprf(np.cos(phi) * np.cos(phi), 1. - m * np.sin(phi) * np.sin(phi), 1.)
    term2 = (n / 3.) * (np.sin(phi) ** 3) * sp.elliprj(np.cos(phi) ** 2, 1. - m * np.sin(phi) * np.sin(phi), 1., 1. - n * np.sin(phi) * np.sin(phi))
    value = term1 + term2
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

test_n = [
    0.0,
    0.5,
    0.9,
    0.0,
    0.01,
    -0.5
]

expected_ellipticf_results = [elliptic_integral_f(phi, m) for phi, m in zip(test_angles, test_moduli)]
expected_elliptice_results = [elliptic_integral_e(phi, m) for phi, m in zip(test_angles, test_moduli)]
expected_ellipticp_results = [elliptic_integral_p(phi, m, n) for phi, m, n in zip(test_angles, test_moduli, test_n)]

# remove the nan result
#expected_ellipticp_results = np.delete(expected_ellipticp_results, 3, axis=0)

np.savetxt('expected_ellipticf_results.txt', expected_ellipticf_results, fmt='%f')
np.savetxt('expected_elliptice_results.txt', expected_elliptice_results, fmt='%f')
np.savetxt('expected_ellipticp_results.txt', expected_ellipticp_results, fmt='%f')
