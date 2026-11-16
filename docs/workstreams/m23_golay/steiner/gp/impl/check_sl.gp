\\ Second reference for the values of rustmath-galois/tests/steiner_labelling.rs that the
\\ design's references did not print in full (PARI/GP 2.15.4). Uses the reviewer's rev.gp
\\ (Newton in Mod(Mod(., p^N), f), roots by composition, direct sums over the 253 heptads).
default(parisizemax, 2000000000);
read("rev.gp"); read("cc0_raw.gp"); read("cc0_oracle_values.gp");
{
a = pollead(F0); n = poldegree(F0); g = a^(n-1) * subst(F0, x, x/a);
v = both(g, 29, 250); print("CC0 p=29 N=250 equal to arith.py: ", v == O29m, "  digits ", [#Str(abs(v[1])), #Str(abs(v[2]))]);
v = both(g, 67, 216); print("CC0 p=67 N=216 equal to arith.py: ", v == O67, "  B5 ", #Str(B5(g)), " digits ", [#Str(abs(v[1])), #Str(abs(v[2]))]);
print("CC0 N_rig at 67: ", nrig(67, B5(g), 0), " N30 at 67: ", nrig(67, B5(g), 30));
f = polsubcyclo(47, 23); print("C23 = ", Vecrev(f)); print("C23 best ", fbest(f), " B5 ", B5(f));
for(i = 1, 3, my(p = [2, 3, 5][i]); print("C23 p=", p, " N=20: ", both(f, p, 20), " N=", nrig(p, B5(f), 0), ": ", both(f, p, nrig(p, B5(f), 0))));
print("C23 q=5 N=1: ", both(f, 5, 1) % 5, "  q=5 N=2: ", both(f, 5, 2) % 25, "  -98317 % 25 = ", -98317 % 25, "  -98317 % 5 = ", -98317 % 5);
print("C23 q=2 N=1: ", both(f, 2, 1) % 2);
h = x^23 - x - 1; print("S23 best ", fbest(h), " B5 ", B5(h), " N_rig(23) ", nrig(23, B5(h), 0), " N_rig(157) ", nrig(157, B5(h), 0), " N30(157) ", nrig(157, B5(h), 30), " N30(23) ", nrig(23, B5(h), 30));
print("S23 p=23 N=30: ", both(h, 23, 30));
print("S23 p=23 N30: ", both(h, 23, nrig(23, B5(h), 30)));
print("S23 p=157 N=4: ", both(h, 157, 4));
print("S23 p=157 N=18: ", both(h, 157, 18));
}
quit;
