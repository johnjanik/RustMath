#!/usr/bin/env sage
"""Stage 2 of the M12:2 Belyi specialization fan.

For each recognized t0 (pM_fan_t{tag}.json): build F in QQ[y], check
irreducibility, polredbest, count real roots, nfdisc (900 s guard),
gate max coefficient digits <= 40. Writes pM_fan_stage2.json summary.
"""
import json
import os
import sys
import time

SWEEP = '/home/john/sweep_2_12_5'
OUT = os.path.join(SWEEP, 'pM_fan_stage2.json')

TAGS = ['2', '3', '4', '5', 'm1', 'm2', '1o2', '3o2', 'm1o2', '5o2', '1o3', 'm3', '7']

R = PolynomialRing(QQ, 'y')
y = R.gen()
RZ = PolynomialRing(ZZ, 'y')


def process_poly(Fi_Q):
    """Given an irreducible poly in QQ[y], return dict with polredbest data."""
    den = lcm([cc.denominator() for cc in Fi_Q.coefficients()])
    Fz = RZ(Fi_Q * den)
    info = {}
    g = pari(Fz).polredbest()
    coeffs = [Integer(cc) for cc in g.Vecrev()]  # ascending, a0 first
    info['polredbest_ascending'] = [str(cc) for cc in coeffs]
    info['degree'] = int(g.poldegree())
    info['digits'] = max(len(str(abs(cc))) for cc in coeffs if cc != 0)
    info['r'] = int(pari(g).polsturm())
    # nfdisc with timeout guard
    try:
        alarm(900)
        D = pari(g).nfdisc()
        cancel_alarm()
        D = Integer(D)
        info['nfdisc'] = str(D)
        info['nfdisc_digits'] = len(str(abs(D)))
        try:
            alarm(120)
            fac = factor(D)
            cancel_alarm()
            info['nfdisc_factored'] = str(fac)
        except (AlarmInterrupt, KeyboardInterrupt):
            cancel_alarm()
            info['nfdisc_factored'] = None
    except (AlarmInterrupt, KeyboardInterrupt):
        cancel_alarm()
        info['nfdisc'] = None
        info['nfdisc_digits'] = None
        info['nfdisc_factored'] = None
    except Exception as e:
        cancel_alarm()
        info['nfdisc'] = None
        info['nfdisc_error'] = str(e)[:200]
    return info


results = {}
for tag in TAGS:
    path = os.path.join(SWEEP, 'pM_fan_t%s.json' % tag)
    if not os.path.exists(path):
        results[tag] = {'status': 'no-stage1-json'}
        print('[%s] no stage1 json' % tag)
        sys.stdout.flush()
        continue
    t_start = time.time()
    data = json.load(open(path))
    F = sum(QQ(Integer(n)) / QQ(Integer(d)) * y**k for k, (n, d) in enumerate(data))
    entry = {'status': 'ok', 'degree_F': int(F.degree())}
    fac = F.factor()
    if len(fac) == 1 and fac[0][1] == 1:
        entry['irreducible'] = True
        try:
            entry.update(process_poly(F))
        except Exception as e:
            entry['status'] = 'polredbest-error'
            entry['error'] = str(e)[:300]
    else:
        entry['irreducible'] = False
        entry['factors'] = []
        for (fi, mult) in fac:
            finfo = {'degree': int(fi.degree()), 'mult': int(mult)}
            if fi.degree() >= 1:
                try:
                    finfo.update(process_poly(fi))
                except Exception as e:
                    finfo['error'] = str(e)[:300]
            entry['factors'].append(finfo)
    entry['secs'] = float(time.time() - t_start)
    results[tag] = entry
    print('[%s] %s' % (tag, json.dumps({k: v for k, v in entry.items()
                                        if k in ('status', 'irreducible', 'degree_F',
                                                 'digits', 'r', 'nfdisc_digits', 'secs')})))
    sys.stdout.flush()
    json.dump(results, open(OUT, 'w'), indent=1)

json.dump(results, open(OUT, 'w'), indent=1)
print('[stage2] done ->', OUT)
