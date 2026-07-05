# rnc_quadrics_test.py  (P1: coordinate-free RNC test from the reply)

import numpy as np
from itertools import combinations_with_replacement


def quadratic_monomials(v):
    """
    v is a projective point vector of length n=d+1.
    Return all products v_i v_j with i <= j.
    """
    mons = []
    n = len(v)

    for i, j in combinations_with_replacement(range(n), 2):
        mons.append(v[i] * v[j])

    return np.array(mons, dtype=np.complex128)


def quadratic_relation_matrix(points):
    """
    points: array shape (num_samples, d+1).
    Each row gives all quadratic monomials at that point.
    """
    return np.vstack([quadratic_monomials(p) for p in points])


def quadratic_relation_svd(points):
    """
    Detect dimension of quadratic relations vanishing on sampled projective curve.
    """
    Q = quadratic_relation_matrix(points)

    # Normalize rows to avoid scale artifacts.
    row_norms = np.linalg.norm(Q, axis=1)
    Q = Q / np.maximum(row_norms[:, None], 1e-300)

    s = np.linalg.svd(Q, compute_uv=False)

    return s


def expected_rnc_quadric_dimension(d):
    return ((d + 2) * (d + 1)) // 2 - (2 * d + 1)


def estimate_nullity(s, rel_tol=1e-10):
    thresh = rel_tol * s[0]
    return int(np.sum(s < thresh))
