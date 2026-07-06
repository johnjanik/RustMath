"""Config for the ACHIRAL passport member B (the Q-candidate dessin).

B (0-based, original labels):
  s0 = [0,4,22,3,1,6,5,17,12,15,18,20,8,13,14,9,16,7,10,19,11,21,2,23]
  s1 = [19,6,22,2,13,3,0,18,23,14,1,5,8,21,12,20,4,15,10,9,11,17,16,7]
Atlas base: M_BASE=1 (cell 1: s0 2-cycle (1 4) -> ordinary a-center, double zero at X=0;
s1 cycle-0 -> z_b = first 12-point; s_inf 5-cycle {1,16,22,3,5} -> z_c = 5-pole #1).
Second 12-point: M_CENTER=b2 M_COSET=4 -> rep_4(z_b) = i/mu.
Mirror involution tau swaps the two 12-cycles => sigma swaps r1B <-> r2B (symmetric gauge).
"""
import numpy as np, mpmath as mp

SW = "/home/john/sweep_2_12_5/"
S0 = "0,4,22,3,1,6,5,17,12,15,18,20,8,13,14,9,16,7,10,19,11,21,2,23"
S1 = "19,6,22,2,13,3,0,18,23,14,1,5,8,21,12,20,4,15,10,9,11,17,16,7"

RHO = {
    'a':  "0.967701554720170639297024876672889879431930376",
    'b':  "0.994619167131282430377659935323357900124648442",
    'b2': "0.994619167131282430377659935323357900124648442",
    'c':  "0.980867986017748246843983162787207468413632858",   # fill from dump log if differs
}
BIN = {
    'a':  SW + "mB_a_N1100.bin",
    'b':  SW + "mB_b_N6900.bin",
    'b2': SW + "mB_b2_N6900.bin",
    'c':  SW + "mB_c_N2000.bin",
}
NPZ = {k: v.replace(".bin", "_ddspan.npz") for k, v in BIN.items()}

_Lam = mp.cos(mp.pi/5)/mp.sin(mp.pi/12)
MU = _Lam + mp.sqrt(_Lam*_Lam - 1)
Z_A = mp.mpc(0, 1)
Z_B = mp.mpc(0, 1)*MU
Z_B2 = mp.mpc(0, 1)/MU
Z_C = mp.mpc('0.793538482569228956003002774509736302229259206',
             '0.608520070894728592568281209719200629678206803')
CENTER = {'a': Z_A, 'b': Z_B, 'b2': Z_B2, 'c': Z_C}
