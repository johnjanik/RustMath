#!/usr/bin/env python3
"""Stage 1 of the M12:2 Belyi specialization fan.

Polish theta (Newton, FD Jacobian) at dps 820, then for each t0 compute the
fiber resolvent F(y), CF-recognize coefficients as exact rationals, and save
/home/john/sweep_2_12_5/pM_fan_t{tag}.json as [[num_str, den_str] x 25].
"""
import json
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import mpmath as mp

DPS = 820
SWEEP = '/home/john/sweep_2_12_5'
THETA_PATH = os.path.join(SWEEP, 'pM_ladder_theta.txt')
POLISHED_PATH = os.path.join(SWEEP, 'pM_ladder_theta_polished820.txt')
STATUS_PATH = os.path.join(SWEEP, 'pM_fan_stage1_status.json')

# t0 fan (Fraction, tag). "2" is a validation replicate of the banked trophy.
T0_LIST = [
    (Fraction(2), '2'),
    (Fraction(3), '3'),
    (Fraction(4), '4'),
    (Fraction(5), '5'),
    (Fraction(-1), 'm1'),
    (Fraction(-2), 'm2'),
    (Fraction(1, 2), '1o2'),
    (Fraction(3, 2), '3o2'),
    (Fraction(-1, 2), 'm1o2'),
    (Fraction(5, 2), '5o2'),
    (Fraction(1, 3), '1o3'),
    (Fraction(-3), 'm3'),
    (Fraction(7), '7'),
]


def setup():
    mp.mp.dps = DPS


def pins():
    P1 = mp.mpc('0.109455285948', '-0.238532279131')
    P2m = mp.mpc('-0.110917611396', '0.242849225257')
    return P1, P2m


def prod_roots(roots):
    """Ascending coefficients of monic prod_i (x - r_i)."""
    cf = [mp.mpc(1)]
    for r in roots:
        new = [mp.mpc(0)] * (len(cf) + 1)
        for k in range(len(cf)):
            new[k] -= r * cf[k]
            new[k + 1] += cf[k]
        cf = new
    return cf


def conv(a, b):
    out = [mp.mpc(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def load_theta(path):
    th = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            re_s, im_s = line.split()
            th.append(mp.mpc(re_s, im_s))
    assert len(th) == 25, len(th)
    return th


def unpack(th):
    a = th[0:11]
    Br = th[11:17]
    Cr = th[17:23]
    lam = th[23]
    c = th[24]
    return a, Br, Cr, lam, c


def structure(th, R12):
    a, Br, Cr, lam, c = unpack(th)
    A = prod_roots([mp.mpc(0)] + list(a))
    P = conv(A, A)                       # degree 24, 25 coeffs
    B = prod_roots(Br)                   # degree 6
    B3 = conv(conv(B, B), B)             # degree 18
    C = prod_roots(Cr)                   # degree 6
    W = conv(B3, C)                      # degree 24
    F = [P[k] - lam * R12[k] - c * W[k] for k in range(25)]
    return F, P


def newton_polish(th, R12):
    h = mp.mpf('1e-400')
    target = mp.mpf('1e-790')
    for it in range(20):
        F0, _ = structure(th, R12)
        err = max(abs(f) for f in F0)
        print(f'[newton] iter {it}  |F| = {mp.nstr(err, 6)}', flush=True)
        if err < target:
            return th, err
        J = mp.matrix(25, 25)
        for j in range(25):
            thp = list(th)
            thp[j] = thp[j] + h
            Fp, _ = structure(thp, R12)
            for k in range(25):
                J[k, j] = (Fp[k] - F0[k]) / h
        rhs = mp.matrix([-f for f in F0])
        dx = mp.lu_solve(J, rhs)
        th = [th[j] + dx[j] for j in range(25)]
    raise RuntimeError(f'Newton did not reach 1e-790; last |F| = {mp.nstr(err, 6)}')


def cf_recognize(v, max_den=10**320):
    """Recognize real mpf v as p/q via continued fractions."""
    tol = mp.mpf('1e-760') * max(mp.mpf(1), abs(v))
    x = mp.mpf(v)
    h2, h1 = 0, 1   # h_{-2}, h_{-1}
    k2, k1 = 1, 0   # k_{-2}, k_{-1}
    for _ in range(2000):
        a = int(mp.floor(x))
        h = a * h1 + h2
        k = a * k1 + k2
        if k > max_den:
            return None
        if abs(v - mp.mpf(h) / mp.mpf(k)) < tol:
            if k < 0:
                h, k = -h, -k
            from math import gcd
            g = gcd(abs(h), k)
            return (h // g, k // g)
        frac = x - a
        if frac <= mp.mpf('1e-780'):
            return None
        x = 1 / frac
        h2, h1 = h1, h
        k2, k1 = k1, k
    return None


def do_t0(args):
    """Worker: compute fiber resolvent for one t0, CF-recognize, save JSON."""
    (t0_num, t0_den, tag, th_str, P1_str, P2m_str) = args
    setup()
    t_start = time.time()
    th = [mp.mpc(r, i) for (r, i) in th_str]
    P1 = mp.mpc(*P1_str)
    P2m = mp.mpc(*P2m_str)
    R12 = prod_roots([P1] * 12 + [P2m] * 12)
    a, Br, Cr, lam, c = unpack(th)
    A = prod_roots([mp.mpc(0)] + list(a))
    P = conv(A, A)
    t0 = mp.mpf(t0_num) / mp.mpf(t0_den)

    cf = [P[k] - t0 * lam * R12[k] for k in range(25)]  # ascending
    lead = abs(cf[24])
    if lead < mp.mpf('1e-40'):
        return dict(tag=tag, status='degree-drop', detail=f'|lead|={mp.nstr(lead,4)}')
    try:
        roots = mp.polyroots(cf[::-1], maxsteps=1200, extraprec=1200)
    except Exception as e:
        return dict(tag=tag, status='polyroots-fail', detail=str(e)[:200])

    ys = []
    for x in roots:
        Bx = mp.mpc(1)
        for b in Br:
            Bx *= (x - b)
        Cx = mp.mpc(1)
        for cc in Cr:
            Cx *= (x - cc)
        ys.append(Bx / Cx)
    s1 = mp.fsum(ys)
    if abs(s1) < mp.mpf('1e-10'):
        return dict(tag=tag, status='degenerate-s1', detail=f'|s1|={mp.nstr(abs(s1),4)}')
    yn = [y / s1 for y in ys]
    Fy = prod_roots(yn)  # 25 ascending coeffs, monic

    imax = max(abs(z.imag) for z in Fy)
    if imax > mp.mpf('1e-300'):
        return dict(tag=tag, status='imag-fail', detail=f'max|Im|={mp.nstr(imax,4)}')

    rats = []
    for k, z in enumerate(Fy):
        r = cf_recognize(z.real)
        if r is None:
            return dict(tag=tag, status='cf-fail',
                        detail=f'coeff k={k} not recognized, |v|={mp.nstr(abs(z.real),4)}',
                        max_im=mp.nstr(imax, 4))
        rats.append(r)

    out = [[str(p), str(q)] for (p, q) in rats]
    path = os.path.join(SWEEP, f'pM_fan_t{tag}.json')
    with open(path, 'w') as fh:
        json.dump(out, fh)
    hmax = max(max(len(str(abs(p))), len(str(q))) for (p, q) in rats)
    return dict(tag=tag, status='recognized', json=path,
                max_im=mp.nstr(imax, 4), max_ratio_digits=hmax,
                secs=round(time.time() - t_start, 1))


def main():
    setup()
    P1, P2m = pins()
    R12 = prod_roots([P1] * 12 + [P2m] * 12)
    th = load_theta(THETA_PATH)
    th, err = newton_polish(th, R12)
    print(f'[newton] converged |F| = {mp.nstr(err, 6)}', flush=True)
    with open(POLISHED_PATH, 'w') as fh:
        for z in th:
            fh.write(f'{mp.nstr(z.real, 820)} {mp.nstr(z.imag, 820)}\n')

    th_str = [(mp.nstr(z.real, 820), mp.nstr(z.imag, 820)) for z in th]
    P1s = (mp.nstr(P1.real, 30), mp.nstr(P1.imag, 30))
    P2s = (mp.nstr(P2m.real, 30), mp.nstr(P2m.imag, 30))
    jobs = [(str(t0.numerator), str(t0.denominator), tag, th_str, P1s, P2s)
            for (t0, tag) in T0_LIST]

    with Pool(processes=min(13, os.cpu_count() or 4)) as pool:
        results = pool.map(do_t0, jobs)

    for r in results:
        print(f"[t0 {r['tag']:>5}] {r['status']}  " +
              ' '.join(f'{k}={v}' for k, v in r.items() if k not in ('tag', 'status')),
              flush=True)
    with open(STATUS_PATH, 'w') as fh:
        json.dump(results, fh, indent=1)
    print('[stage1] done', flush=True)


if __name__ == '__main__':
    main()
