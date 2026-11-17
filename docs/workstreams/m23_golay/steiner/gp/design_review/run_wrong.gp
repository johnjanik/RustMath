read("rev.gp");
\\ Search: S23 polynomials (Gal not M23) where exactly one T5 candidate lies within B5 at a
\\ rigorous precision, i.e. the rule would return Decided under a false hypothesis.
{
found = 0;
foreach([x^23 - x - 1, x^23 - x + 1, x^23 + x^2 - 1, x^23 - 2*x - 1, x^23 + 3*x + 1], s,
  bs = B5(s);
  forprime(p = 2, 400, if(!polisirreducible(Mod(1, p) * s), next);
    N0 = nrig(p, bs, 0); r = froots(s, p, N0 + 3); 
    for(N = N0, N0 + 3, q = p^N; v = [T5(r, Hp, p^(N0+3)), T5(r, Hm, p^(N0+3))];
      w = [centerlift(Mod(v[1], q)), centerlift(Mod(v[2], q))]; 
      ok = [abs(w[1]) <= bs, abs(w[2]) <= bs];
      if(ok[1] != ok[2], found++; print(s, " p = ", p, " N = ", N, " (N_rig ", N0, "): residues ", w, " within ", ok, " B5 = ", bs)))));
print("found ", found);
}
quit;
