"""P4.5.4 general chart gluing (single-group atlas, base-1 labeling).

All atlas charts live in ONE conjugate group (M_BASE=1), centered at points p*i on the
imaginary axis (a: p=1, first 12-point: p=0.164275..., second 12-point: p=MU=6.087...).
Every chart is an ORDINARY point there (trivial stabilizer), so the dd jet-Tikhonov bases
are identity jet-normalized (C = I exactly: b_v = 1, b_{<v} = 0) and the Veronese frames
need NO normalizer: F(z) itself is ~ nu(x_chart(z)).

The transition between charts centered at q*i (the "from" frame, defining x_q) and p*i:

    z = q*i (1+w)/(1-w)  =>  w_p(w_q) = ((q-p) + (q+p) w_q) / ((q+p) + (q-p) w_q),

an exact real Moebius. The to-chart basis forms, as series in w_q:

    g_j^(q)(w) = ((1 - w_p(w))^4 / (1 - w)^4) * sum_n b_n^(p) w_p(w)^n,

whose jets J[i,j] give F_p = J^T F_q exactly (nine jets determine the 9-dim space), so

    S = J^T  satisfies  S nu(x_q) ~ nu(x_p) = nu(mu(x_q)),   S ~ Sym^8(mu),  x_p = mu(x_q).

Usage: python3 glue_pq.py <to_basis.npz> <p> <from_basis.npz> <q> [S_out.npy]
   e.g. glue_pq.py m_glue_a_N1500_ddbasis.npz 1.0 m_order12_cycle1_N3000_ddbasis.npz MU
        (p, q as decimal strings; the token MU means the order-12 vertex height)
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from sym8_glue import fit_mobius_from_sym8, sym8_projective_residual
from glue_ab import smul, sinv, spow_i, to_c128

mp.mp.dps = 40
D = 8
K = 4

_Lam = mp.cos(mp.pi / 5) / mp.sin(mp.pi / 12)
MU = _Lam + mp.sqrt(_Lam * _Lam - 1)


def parse_height(tok):
    return MU if tok.strip().upper() == "MU" else mp.mpf(tok)


def wp_of_wq_series(p, q, n):
    """w_p(w_q) = ((q-p) + (q+p) w) / ((q+p) + (q-p) w) as a series in w."""
    num = [mp.mpc(q - p), mp.mpc(q + p)] + [mp.mpc(0)] * (n - 2)
    den = [mp.mpc(q + p), mp.mpc(q - p)] + [mp.mpc(0)] * (n - 2)
    return smul(num, sinv(den, n), n)


def toform_qjets(Bh, Bl, p, q, n=D + 1):
    """w_q-jets 0..n-1 of each to-chart (center p*i) basis form. Exact series composition."""
    dim, m = Bh.shape
    wp = wp_of_wq_series(p, q, n)
    one_minus_wp = [mp.mpc(1) - wp[0]] + [-c for c in wp[1:]]
    one_minus_wq = [mp.mpc(1), mp.mpc(-1)] + [mp.mpc(0)] * (n - 2)
    pref = smul(spow_i(one_minus_wp, K, n), sinv(spow_i(one_minus_wq, K, n), n), n)
    J = np.empty((n, m), dtype=object)
    for col in range(m):
        acc = [mp.mpc(0)] * n
        pw = [mp.mpc(1)] + [mp.mpc(0)] * (n - 1)
        for j in range(dim):
            b = mp.mpc(Bh[j, col]) + mp.mpc(Bl[j, col])
            if b != 0:
                for i in range(n):
                    acc[i] += b * pw[i]
            pw = smul(pw, wp, n)
        g = smul(pref, acc, n)
        for i in range(n):
            J[i, col] = g[i]
    return J


def fit_and_report(S128, label):
    print(f"\nSym^8 Moebius fit ({label}):")
    results = []
    for rings in ([0.05, 0.1, 0.2], [0.1, 0.2, 0.3], [0.15, 0.3, 0.45]):
        xs = [0.0]
        for r in rings:
            for kk in range(16):
                xs.append(r * np.exp(1j * 2 * np.pi * (kk + 0.173) / 16))
        mob = fit_mobius_from_sym8(S128, xs=np.array(xs, complex))
        pr = sym8_projective_residual(S128, mob)
        al, be, ga, de = mob['alpha'], mob['beta'], mob['gamma'], mob['delta']
        mu0 = al / ga
        dmu0 = (be * ga - al * de) / ga**2
        results.append((rings, pr['relative_residual'], mu0, dmu0))
        print(f"  rings {rings}: proj resid={pr['relative_residual']:.2e}  "
              f"sample med={mob['sample_residual_median']:.1e}  "
              f"mu(0)={mu0:.12f}  mu'(0)={dmu0:.8e}")
    return results


if __name__ == "__main__":
    to_npz = sys.argv[1]
    p = parse_height(sys.argv[2])
    from_npz = sys.argv[3]
    q = parse_height(sys.argv[4])
    s_out = sys.argv[5] if len(sys.argv) > 5 else None

    T = np.load(to_npz, allow_pickle=True)
    F = np.load(from_npz, allow_pickle=True)
    print(f"to-chart   (defines x_p): {os.path.basename(to_npz)}  center {mp.nstr(p, 20)}*i")
    print(f"from-chart (defines x_q): {os.path.basename(from_npz)}  center {mp.nstr(q, 20)}*i")

    J = toform_qjets(T['Bh'], T['Bl'], p, q)
    S = J.T                                     # C = I on both sides
    S128 = to_c128(S)
    if s_out:
        np.save(s_out, S128)
    fit_and_report(S128, f"x_p = mu(x_q)")
    print("\nmu(0) = x_p at the from-chart center; mu'(0) = the compression scale there")
