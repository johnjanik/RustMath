#!/usr/bin/env /usr/bin/python3
"""Render the [2,12,5] N-sweep dashboard from metrics.csv (system python: matplotlib+pandas).
Six panels tracking the overfit onset as N grows.  Usage: /usr/bin/python3 sweep_plot.py [sweepdir]"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SWEEP = sys.argv[1] if len(sys.argv) > 1 else '/home/john/sweep_2_12_5'
df = pd.read_csv(os.path.join(SWEEP, 'metrics.csv')).sort_values('N')
N = df['N'].values
sig_cols = [c for c in df.columns if c.startswith('sig') and c[3:].isdigit()]

fig, ax = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('[2,12,5] KMSV null-space vs truncation N — overfit-onset dashboard', fontsize=15, weight='bold')

# 1. spectral waterfall: the 12 smallest dd sigma + rho^N
a = ax[0, 0]
for c in sig_cols:
    a.semilogy(N, df[c].values, '.-', ms=3, lw=.6, alpha=.7)
a.semilogy(N, df['rhoN'].values, 'k--', lw=2, label=r'$\rho^N$ (truncation floor)')
a.set_title('1. Spectral waterfall — 12 smallest $\\sigma$ (dd)'); a.set_xlabel('N'); a.set_ylabel('$\\sigma$')
a.legend(fontsize=8); a.grid(alpha=.3)

# 2. smin & sig9 vs rho^N
a = ax[0, 1]
a.semilogy(N, df['smin'].values, 'o-', ms=4, label='smin (smallest $\\sigma$)')
a.semilogy(N, df['sig9'].values, 's-', ms=4, label='sig9 (top of null cluster)')
a.semilogy(N, df['rhoN'].values, 'k--', lw=2, label=r'$\rho^N$')
a.set_title('2. Onset: smin vs physical floor'); a.set_xlabel('N'); a.set_ylabel('$\\sigma$')
a.legend(fontsize=8); a.grid(alpha=.3)

# 3. Hauptmodul reality
a = ax[0, 2]
a.semilogy(N, np.maximum(df['reality_lo'].values, 1e-18), 'o-', ms=4, label='smallest-9')
a.semilogy(N, np.maximum(df['reality_phys'].values, 1e-18), 's-', ms=4, label='physical band')
a.axhline(1e-3, color='g', ls=':', label='real threshold')
a.set_title('3. Hauptmodul reality  |Im/|.||'); a.set_xlabel('N'); a.set_ylabel('imag/real ratio')
a.legend(fontsize=8); a.grid(alpha=.3)

# 4. order-12 spread (the goal)
a = ax[1, 0]
a.semilogy(N, df['o12_lo'].values, 'o-', ms=4, label='smallest-9')
a.semilogy(N, df['o12_phys'].values, 's-', ms=4, label='physical band')
a.axhline(3.8, color='r', ls=':', label='FP64 baseline 3.8')
a.axhline(0.12, color='g', ls=':', label='target ~0.12')
a.set_title('4. Order-12 ramification spread (payoff)'); a.set_xlabel('N'); a.set_ylabel('rescaled spread')
a.legend(fontsize=8); a.grid(alpha=.3)

# 5. tail-energy / discriminant
a = ax[1, 1]
a.plot(N, df['te_lo'].values, 'o-', ms=4, label='smallest-9')
a.plot(N, df['te_phys'].values, 's-', ms=4, label='physical band')
if 'bgrow_lo' in df.columns:
    a2 = a.twinx()
    a2.semilogy(N, df['bgrow_lo'].values, '^-', ms=3, color='purple', alpha=.5, label='b-growth (lo)')
    a2.set_ylabel('max|b|/median|b|', color='purple')
a.set_title('5. Tail-energy / overfit fingerprint'); a.set_xlabel('N'); a.set_ylabel('tail frac')
a.legend(fontsize=8); a.grid(alpha=.3)

# 6. Hauptmodul X[1] components
a = ax[1, 2]
a.plot(N, df['x1re_lo'].values, 'o-', ms=4, label='Re X[1] (lo)')
a.plot(N, df['x1im_lo'].values, 's-', ms=4, label='Im X[1] (lo)')
if 'x1re_phys' in df.columns:
    a.plot(N, df['x1re_phys'].values, '.--', ms=6, alpha=.6, label='Re X[1] (phys)')
a.set_title('6. Hauptmodul X[1] components'); a.set_xlabel('N'); a.set_ylabel('coefficient')
a.legend(fontsize=8); a.grid(alpha=.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(SWEEP, 'dashboard.png')
plt.savefig(out, dpi=110)
print(f'wrote {out}  ({len(df)} points, N={N.min()}..{N.max()})')
