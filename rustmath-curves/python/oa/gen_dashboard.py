#!/usr/bin/env /usr/bin/python3
"""Regenerate dashboard.html by injecting the current metrics.csv into the DATA=[...] block.
System python (pandas).  Usage: /usr/bin/python3 gen_dashboard.py [sweepdir]"""
import sys, os, re, math
import pandas as pd

SWEEP = sys.argv[1] if len(sys.argv) > 1 else '/home/john/sweep_2_12_5'
html = os.path.join(SWEEP, 'dashboard.html')
df = pd.read_csv(os.path.join(SWEEP, 'metrics.csv')).sort_values('N')

def j(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 'null'
    return f'{v:.{d}g}'

rows = []
for _, r in df.iterrows():
    sigvals = list(r.iloc[-12:])                      # the sig0..sig11 block = last 12 columns
    sig = '[' + ','.join(j(v, 4) for v in sigvals) + ']'
    glitch = 1 if (isinstance(r['o12_lo'], float) and math.isnan(r['o12_lo'])) else 0
    rows.append('{' + f"N:{int(r['N'])},rhoN:{j(r['rhoN'])},smin:{j(r['smin'])},sig9:{j(sigvals[8])},"
        f"te:{j(r['te_lo'])},rl:{j(r['reality_lo'])},rp:{j(r['reality_phys'])},"
        f"o12l:{j(r['o12_lo'])},o12p:{j(r['o12_phys'])},x1r:{j(r['x1re_lo'])},x1i:{j(r['x1im_lo'])},"
        f"glitch:{glitch},sig:{sig}" + '}')
data_js = 'const DATA = [\n ' + ',\n '.join(rows) + ',\n];'

src = open(html).read()
src = re.sub(r'const DATA = \[.*?\];', data_js, src, count=1, flags=re.S)
open(html, 'w').write(src)
print(f'injected {len(rows)} rows into {html}  (N={int(df.N.min())}..{int(df.N.max())})')
