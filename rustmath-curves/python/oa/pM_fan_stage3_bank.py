#!/usr/bin/env python3
"""Stage 3 of the M12:2 Belyi specialization fan: bank passing polys.

Reads pM_fan_stage2.json. For each t0 with an irreducible degree-24
polredbest poly passing the digits<=40 gate, appends
  {"coeffs": "<csv ascending>", "t": 18441, "r": <r>, "src": "belyi"}
to bank_pending.jsonl (skipping duplicates), and appends full metadata
(t0, disc factorization, digits, factors if reducible) to pM_fan.jsonl.
"""
import json
import os

SWEEP = '/home/john/sweep_2_12_5'
STAGE2 = os.path.join(SWEEP, 'pM_fan_stage2.json')
BANK = '/home/john/inverse_galois/frobenius/bank_pending.jsonl'
META = os.path.join(SWEEP, 'pM_fan.jsonl')

TAG2T0 = {'2': '2', '3': '3', '4': '4', '5': '5', 'm1': '-1', 'm2': '-2',
          '1o2': '1/2', '3o2': '3/2', 'm1o2': '-1/2', '5o2': '5/2',
          '1o3': '1/3', 'm3': '-3', '7': '7'}

results = json.load(open(STAGE2))

existing = set()
with open(BANK) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            existing.add(json.loads(line)['coeffs'])
        except Exception:
            pass

bank_lines = []
meta_lines = []
for tag, e in results.items():
    t0 = TAG2T0.get(tag, tag)
    if e.get('status') != 'ok':
        meta_lines.append({'t0': t0, 'status': e.get('status')})
        continue
    meta = {'t0': t0, 'irreducible': e.get('irreducible')}
    if e.get('irreducible'):
        coeffs = e.get('polredbest_ascending')
        meta.update({
            'polredbest_ascending': coeffs,
            'digits': e.get('digits'),
            'r': e.get('r'),
            'nfdisc': e.get('nfdisc'),
            'nfdisc_factored': e.get('nfdisc_factored'),
            'nfdisc_digits': e.get('nfdisc_digits'),
        })
        gate = (e.get('digits') is not None and e['digits'] <= 40
                and e.get('degree') == 24 and coeffs)
        meta['gate_digits_le_40'] = bool(gate)
        if gate:
            cs = ','.join(coeffs)
            if cs in existing:
                meta['banked'] = 'duplicate'
            else:
                bank_lines.append({'coeffs': cs, 't': 18441,
                                   'r': int(e['r']), 'src': 'belyi'})
                existing.add(cs)
                meta['banked'] = True
        else:
            meta['banked'] = False
    else:
        meta['factors'] = e.get('factors')
        meta['banked'] = False
    meta_lines.append(meta)

with open(BANK, 'a') as fh:
    for rec in bank_lines:
        fh.write(json.dumps(rec) + '\n')
with open(META, 'a') as fh:
    for rec in meta_lines:
        fh.write(json.dumps(rec) + '\n')

print(f'banked {len(bank_lines)} new polys; wrote {len(meta_lines)} metadata lines to {META}')
for rec in bank_lines:
    print('  banked r=%s digits=%s coeffs[:60]=%s...' %
          (rec['r'], max(len(x.lstrip("-")) for x in rec['coeffs'].split(',')),
           rec['coeffs'][:60]))
