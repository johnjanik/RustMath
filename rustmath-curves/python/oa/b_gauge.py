"""Real gauge for B: from sigma_B to the xi-coordinate where sigma = conj and the
12-points sit at +-i (so W = (xi^2+1)^12, all structure real).

Steps: (1) three fixed points of sigma (rootfind from spread starts) -> the fixed circle;
(2) T0 = Moebius(3 fixed pts -> 0, 1, infinity) maps the circle to R-hat; verify
T0 o sigma o T0^{-1} = conj; (3) real Moebius T1 sending T0(r1) (upper/lower half plane)
to +i (then T1(T0(r2)) = -i automatically since sigma swaps r1, r2); T = T1 o T0.
Saves TB.npy (Moebius coeffs al, be, ga, de: T(x) = (al + be x)/(ga + de x)).
"""
import numpy as np, sys, os
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(__file__))
import b_config as BC


def moeb_np(m, x):
    al, be, ga, de = m
    return (al + be*np.asarray(x))/(ga + de*np.asarray(x))


def sigma_of(M):
    def s(x):
        al, be, ga, de = M
        xc = np.conj(np.asarray(x))
        return (al + be*xc)/(ga + de*xc)
    return s


def fixed_points(sig, n=3):
    """Fixed points of the antiholomorphic involution: minimize |sig(x)-x| from many starts
    (Newton on the 2-real system via complex secant-ish iteration: x <- (x + sig(x))/2
    converges on/near the fixed circle since sig is an isometry-reflection)."""
    rng = np.random.default_rng(2)
    pts = []
    for _ in range(400):
        x = complex(rng.uniform(-1.5, 1.5), rng.uniform(-1.2, 1.2))
        for _ in range(200):
            x = 0.5*(x + sig(x))
        if abs(sig(x) - x) < 1e-13:
            if all(abs(x - p) > 0.35 for p in pts):
                pts.append(x)
        if len(pts) >= n:
            break
    return pts


def mobius_from_3(zs, ws):
    def to01inf(p):
        z0, z1, z2 = p
        return np.array([[z1 - z2, -z0*(z1 - z2)], [z1 - z0, -z2*(z1 - z0)]], complex)
    T = np.linalg.inv(to01inf(ws)) @ to01inf(zs)
    return (T[0, 1], T[0, 0], T[1, 1], T[1, 0])


def build_gauge(M, r1, r2, verbose=True):
    sig = sigma_of(M)
    fp = fixed_points(sig)
    if len(fp) < 3:
        raise RuntimeError(f"only {len(fp)} fixed points found")
    T0 = mobius_from_3(fp, [0j, 1 + 0j, 1e9 + 0j])   # approx infinity by big number? no:
    # proper: send fp[2] -> infinity via exact formula
    z0, z1, z2 = fp
    # T0(x) = (x - z0)(z1 - z2)/((x - z2)(z1 - z0))
    al = -z0*(z1 - z2); be = (z1 - z2); ga = -z2*(z1 - z0); de = (z1 - z0)
    T0 = (al, be, ga, de)
    if verbose:
        print("sigma fixed points:", [f"{p:.8f}" for p in fp])
        # verify conj-ness
        probes = np.array([0.2 - 0.1j, -0.4 + 0.3j, 0.7 + 0.2j])
        lhs = moeb_np(T0, sig(np.conj(np.conj(probes))))
        d = np.abs(moeb_np(T0, sig(probes)) - np.conj(moeb_np(T0, probes))).max()
        print(f"T0 o sigma o T0^-1 = conj defect: {d:.2e}")
    u = complex(moeb_np(T0, r1))
    if u.imag < 0:
        u = complex(moeb_np(T0, r2))
    # real Moebius sending u (Im>0) to i:  T1(x) = (x - a)/(b x + c) real coeffs:
    # standard: T1(x) = (x - Re u)/Im u maps u -> i and keeps R real.
    a, bIm = u.real, u.imag
    T1 = (-a/bIm + 0j, 1/bIm + 0j, 1 + 0j, 0j)
    # compose T = T1 o T0
    A1 = np.array([[T1[1], T1[0]], [T1[3], T1[2]]])
    A0 = np.array([[T0[1], T0[0]], [T0[3], T0[2]]])
    C = A1 @ A0
    T = (C[0, 1], C[0, 0], C[1, 1], C[1, 0])
    if verbose:
        print(f"T(r1) = {complex(moeb_np(T, r1)):.10f}  (want +i)")
        print(f"T(r2) = {complex(moeb_np(T, r2)):.10f}  (want -i)")
        probes = np.array([0.2 - 0.1j, -0.4 + 0.3j, 0.7 + 0.2j, -0.2 - 0.4j])
        d = np.abs(moeb_np(T, sig(probes)) - np.conj(moeb_np(T, probes))).max()
        print(f"T o sigma o T^-1 = conj defect: {d:.2e}")
    return T


if __name__ == "__main__":
    M = tuple(np.load(BC.SW + "sigmaB.npy"))
    D = np.load(BC.SW + "pB_r12.npy")     # [r1, r2] saved by b_sigma12 follow-up
    T = build_gauge(M, D[0], D[1])
    np.save(BC.SW + "TB.npy", np.array(T, complex))
    print("saved TB.npy")
