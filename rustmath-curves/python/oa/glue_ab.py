"""P4.5.3-4: dd transition a-frame <-> b-frame and the Sym^8 Moebius gluing.

Coefficient-level transition, no pointwise overlap fitting (the scalar overlap was dirty
at FP64 because it sits where neither chart is in its good territory). The two chart maps
are Moebius in z, so w_a is an EXACT real Moebius of w_b:

    z_b = MU*i,  w_b = (z - z_b)/(z + MU*i),  w_a = (z - i)/(z + i)
    =>  w_a(w_b) = ((MU-1) + (MU+1) w_b) / ((MU+1) + (MU-1) w_b).

Each a-basis form, as a g-series in w_b (g = f/(1-w_b)^4), is then an exact series
composition of dd data:

    g_j^(b)(w_b) = ((1 - w_a(w_b))^4 / (1 - w_b)^4) * sum_n b_n^(a) w_a(w_b)^n.

J[i,j] = w_b-jet_i of a-form j. With the b-basis jet-normalized (C_b = I by construction
of the jet-Tikhonov solve), a-form j = sum_i J[i,j] * (b-basis form i) EXACTLY (a form
with w_b-valuation >= 9 in a 9-dim space with valuations 0..8 is zero). So

    F_a(z) = J^T F_b(z),   G_a = A_a F_a,   G_b = F_b,
    S = A_a J^T   satisfies   S nu(x_b) ~ nu(x_a) = nu(mu(x_b)),   S ~ Sym^8(mu).

A_a = inv(C_a^T) where C_a = the u-jets (u = w_a^2) of the a-basis forms in the
f-trivialization — unit lower-triangular with bounded entries for the jet-normalized
basis (the 1e11 condition number belonged to the SVD basis, not this one).

Everything runs in mpmath (dps 40) on the dd coefficient pairs.

Usage: python3 glue_ab.py <a_ddbasis.npz> <b_ddbasis.npz>
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import mpmath as mp
from sym8_glue import fit_mobius_from_sym8, sym8_projective_residual, sym_power_mobius

mp.mp.dps = 40
D = 8
K = 4

_Lam = mp.cos(mp.pi / 5) / mp.sin(mp.pi / 12)
MU = _Lam + mp.sqrt(_Lam * _Lam - 1)


# ---------- mpmath series helpers (ascending coefficient lists, truncated at n terms) ----------

def smul(a, b, n):
    out = [mp.mpc(0)] * n
    for i, ai in enumerate(a[:n]):
        if ai == 0:
            continue
        for j, bj in enumerate(b[:n - i]):
            out[i + j] += ai * bj
    return out


def sinv(a, n):
    out = [mp.mpc(0)] * n
    out[0] = 1 / a[0]
    for k in range(1, n):
        s = mp.mpc(0)
        for j in range(1, k + 1):
            if j < len(a):
                s += a[j] * out[k - j]
        out[k] = -s / a[0]
    return out


def spow_i(a, k, n):
    out = [mp.mpc(0)] * n
    out[0] = mp.mpc(1)
    for _ in range(k):
        out = smul(out, a, n)
    return out


def wa_of_wb_series(n):
    """w_a(w_b) = ((MU-1) + (MU+1) w) / ((MU+1) + (MU-1) w) as a series in w."""
    num = [mp.mpc(MU - 1), mp.mpc(MU + 1)] + [mp.mpc(0)] * (n - 2)
    den = [mp.mpc(MU + 1), mp.mpc(MU - 1)] + [mp.mpc(0)] * (n - 2)
    return smul(num, sinv(den, n), n)


def aform_bjets(Bh, Bl, n=D + 1):
    """w_b-jets 0..n-1 of each a-basis form: exact series composition of the dd data."""
    dim, m = Bh.shape
    wa = wa_of_wb_series(n)
    one_minus_wa = [mp.mpc(1) - wa[0]] + [-c for c in wa[1:]]
    one_minus_wb = [mp.mpc(1), mp.mpc(-1)] + [mp.mpc(0)] * (n - 2)
    pref = smul(spow_i(one_minus_wa, K, n), sinv(spow_i(one_minus_wb, K, n), n), n)
    J = np.empty((n, m), dtype=object)
    # incremental powers wa^j, j = 0..dim-1; accumulate sum_j b_j wa^j
    for col in range(m):
        acc = [mp.mpc(0)] * n
        p = [mp.mpc(1)] + [mp.mpc(0)] * (n - 1)
        for j in range(dim):
            b = mp.mpc(Bh[j, col]) + mp.mpc(Bl[j, col])
            if b != 0:
                for i in range(n):
                    acc[i] += b * p[i]
            p = smul(p, wa, n)
        g = smul(pref, acc, n)
        for i in range(n):
            J[i, col] = g[i]
    return J


def aform_ujets(Bh, Bl, n=D + 1):
    """u-jets (u = w_a^2) of each a-basis form in the f-trivialization:
    c = (1-w)^4 * series; C_a[i,j] = c[2i]. Also returns max |odd coeff| (evenness gate)."""
    dim, m = Bh.shape
    omw = [mp.mpf(1), mp.mpf(-4), mp.mpf(6), mp.mpf(-4), mp.mpf(1)]
    Ca = np.empty((n, m), dtype=object)
    odd_max = mp.mpf(0)
    for col in range(m):
        c = [mp.mpc(0)] * (2 * n)
        for i, o in enumerate(omw):
            for j in range(2 * n - i):
                if j < dim:
                    b = mp.mpc(Bh[j, col]) + mp.mpc(Bl[j, col])
                    if b != 0:
                        c[i + j] += o * b
        for i in range(n):
            Ca[i, col] = c[2 * i]
        for i in range(min(n, (2 * n - 1) // 2)):
            odd_max = max(odd_max, abs(c[2 * i + 1]))
    return Ca, odd_max


def mp_solve_upper_unit(Ca, J):
    """S = A_a J with A_a = inv(Ca^T)... i.e. solve Ca^T S = J column-by-column.
    Ca is unit lower-triangular (Ca[j,j]=1, Ca[i<j,j]=0), so Ca^T is unit upper-triangular:
    back substitution, exact structure, no conditioning surprise."""
    n, m = J.shape
    S = np.empty((n, m), dtype=object)
    for col in range(m):
        x = [mp.mpc(0)] * n
        for i in range(n - 1, -1, -1):
            s = J[i, col]
            for k in range(i + 1, n):
                s -= Ca[k, i] * x[k]      # (Ca^T)[i,k] = Ca[k,i]
            x[i] = s
        for i in range(n):
            S[i, col] = x[i]
    return S


def to_c128(S):
    return np.array([[complex(S[i, j]) for j in range(S.shape[1])]
                     for i in range(S.shape[0])], dtype=np.complex128)


if __name__ == "__main__":
    a_npz = sys.argv[1] if len(sys.argv) > 1 else "/home/john/sweep_2_12_5/m_N2950_ddbasis.npz"
    b_npz = sys.argv[2] if len(sys.argv) > 2 else "/home/john/sweep_2_12_5/m_order12_N3000_ddbasis.npz"
    A = np.load(a_npz, allow_pickle=True)
    B = np.load(b_npz, allow_pickle=True)
    print(f"a-basis: {os.path.basename(a_npz)} vals={A['vals']} lam={float(A['lam']):.0e}")
    print(f"b-basis: {os.path.basename(b_npz)} vals={B['vals']} lam={float(B['lam']):.0e}")
    print(f"MU = {mp.nstr(MU, 30)}")

    # transition: w_b-jets of the a-forms (exact series composition of dd data)
    J = aform_bjets(A['Bh'], A['Bl'])
    # a-frame normalizer from the a-basis's own u-jets (unit triangular)
    Ca, odd_max = aform_ujets(A['Bh'], A['Bl'])
    print(f"evenness gate: max |odd u-coeff| of a-forms = {mp.nstr(odd_max, 3)}")
    offd = max(abs(Ca[i, j]) for j in range(9) for i in range(j + 1, 9))
    print(f"C_a unit-triangular, max |entry| above pivots = {mp.nstr(offd, 3)}")

    # F_a = J^T F_b, so S = A_a J^T:  S nu(x_b) ~ nu(x_a)
    S = mp_solve_upper_unit(Ca, J.T)
    S128 = to_c128(S)
    np.save("/home/john/sweep_2_12_5/S_ab.npy", S128)

    # Sym^8 fit on several sample rings for stability
    print("\nSym^8 Moebius fit (x_a = mu(x_b)):")
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
        print(f"  rings {rings}: proj resid={pr['relative_residual']:.2e}  "
              f"sample resid med={mob['sample_residual_median']:.1e}  "
              f"mu(0)={mu0:.10f}  mu'(0)={dmu0:.6e}")
    print("\nmu(0)  = X(z_b) in the a-frame coordinate")
    print("mu'(0) = dX/dx_b(z_b) -- the compression scale, FP64 estimate was ~5e-3")
