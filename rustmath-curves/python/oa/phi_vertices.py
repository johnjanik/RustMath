"""phi = t as a power series in the local uniformizer at EACH cone vertex of the (a,b,c)=(2,12,5)
triangle -- the three singular points of the hypergeometric equation:

  t=0  (order a=2)  : u_kappa,  exponent diff 1/a   -- the current chart  (phi ~ u^a)
  t=1  (order b=12) : u_12,     exponent diff 1/b   -- THE TIGHT HOLE     (phi-1 ~ u_12^b)
  t=oo (order c=5)  : u_5,      exponent diff 1/c   -- the poles          (1/phi ~ u_5^c)

Same hypergeometric reversion as phi.py, re-centered at t=1 and t=oo via the local Frobenius
solutions.  Validates that the 12^2 ramification -- invisible in the t=0 germ -- is a clean
12-fold zero of phi-1 in the order-12 chart.
"""
from fractions import Fraction as Fr

def hyp_series(A, B, C, ntab):
    c = [Fr(1)]
    for n in range(1, ntab):
        c.append(c[-1] * (A + n - 1) * (B + n - 1) / ((C + n - 1) * n))
    return c

def poly_div(num, den, L):
    r = [Fr(0)] * L
    for n in range(L):
        s = num[n] if n < len(num) else Fr(0)
        for j in range(1, n + 1):
            if j < len(den): s -= den[j] * r[n - j]
        r[n] = s / den[0]
    return r

def revert(f, L):
    g = [Fr(0)] * L; g[1] = 1 / f[1]
    for n in range(2, L):
        acc = [Fr(0)] * L; p = [Fr(1)] + [Fr(0)] * (L - 1)
        for m in range(len(f)):
            if f[m] != 0:
                for i in range(L): acc[i] += f[m] * p[i]
            newp = [Fr(0)] * L
            for i in range(L):
                if p[i] != 0:
                    for j in range(L - i):
                        if g[j] != 0: newp[i + j] += p[i] * g[j]
            p = newp
        g[n] = -acc[n] / f[1]
    return g

def _revert_root(R, e, L):
    """given R (series in s), form u = s^{1/e} R(s) (= sum R_n sigma^{1+e n}, sigma=s^{1/e}),
    revert to sigma(u), return s=sigma^e as a series in u to L terms."""
    Ls = L + e
    uk = [Fr(0)] * Ls
    for n, Rn in enumerate(R):
        idx = 1 + e * n
        if idx < Ls: uk[idx] = Rn
    sig = revert(uk, Ls)
    p = [Fr(1)] + [Fr(0)] * (Ls - 1)
    for _ in range(e):
        newp = [Fr(0)] * Ls
        for i in range(Ls):
            if p[i] != 0:
                for j in range(Ls - i):
                    if sig[j] != 0: newp[i + j] += p[i] * sig[j]
        p = newp
    return p[:L]

def phi_in_u12(a, b, c, L):
    """phi = t = 1 - s, s the order-b local variable at t=1; series in u_12 to L terms."""
    A = Fr(1, 2) * (1 + Fr(1, a) - Fr(1, b) - Fr(1, c))
    B = Fr(1, 2) * (1 + Fr(1, a) - Fr(1, b) + Fr(1, c))
    C = 1 + Fr(1, a)
    ntab = L // b + 4
    g1 = hyp_series(A, B, A + B - C + 1, ntab)          # exponent 0 at t=1
    g2 = hyp_series(C - A, C - B, C - A - B + 1, ntab)  # exponent (C-A-B)=1/b
    R = poly_div(g2, g1, ntab)                          # u_12 = s^{1/b} R(s)
    s_ser = _revert_root(R, b, L)                       # s as series in u_12
    return [(Fr(1) - s_ser[0]) if n == 0 else -s_ser[n] for n in range(L)]  # phi = 1 - s

def phi_in_u5(a, b, c, L):
    """1/phi = 1/t as a series in the order-c uniformizer u_5 at t=infinity (val = c).
    Solutions at infinity have exponents A, B (difference B-A = 1/c); u_5 = tau^{1/c} (g2/g1),
    tau = 1/t.  Returns 1/phi = tau, whose val is c (the c-fold pole ramification)."""
    A = Fr(1, 2) * (1 + Fr(1, a) - Fr(1, b) - Fr(1, c))
    B = Fr(1, 2) * (1 + Fr(1, a) - Fr(1, b) + Fr(1, c))
    C = 1 + Fr(1, a)
    ntab = L // c + 4
    g1 = hyp_series(A, A - C + 1, A - B + 1, ntab)      # exponent A at infinity
    g2 = hyp_series(B, B - C + 1, B - A + 1, ntab)      # exponent B; B-A = 1/c
    R = poly_div(g2, g1, ntab)                          # u_5 = tau^{1/c} R(tau)
    return _revert_root(R, c, L)                        # 1/phi = tau as series in u_5

if __name__ == "__main__":
    a, b, c = 2, 12, 5
    L = 30
    def show(name, ser, shift, expect_val):
        val = next((n for n in range(L) if (ser[n] - (Fr(1) if (n == 0 and shift) else Fr(0))) != 0), None)
        tag = "OK" if val == expect_val else f"MISMATCH (want {expect_val})"
        print(f"  {name}: val = {val}   {tag}")
        for n in range(min(L, 3 * expect_val + 1)):
            cn = ser[n] - (Fr(1) if (n == 0 and shift) else Fr(0))
            if cn != 0: print(f"      u^{n:2d}: {cn}   (~{float(cn):+.6g})")

    print(f"(a,b,c)=({a},{b},{c})  base map phi = t at each hypergeometric vertex")
    print("t=1 (order 12):  phi - 1  in u_12   (expect val 12, the 12^2 ramification)")
    show("phi-1", phi_in_u12(a, b, c, L), True, b)
    print("t=inf (order 5): 1/phi    in u_5    (expect val 5, the 5^4 1^4 ramification)")
    show("1/phi", phi_in_u5(a, b, c, L), False, c)
