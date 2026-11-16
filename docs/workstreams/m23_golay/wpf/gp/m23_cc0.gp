\\ F-9 prototype on the CC0 models G, G7, G13 (Le Lay's G, vendored in RustMath): the
\\ criterion of irred.gp at the primes of the design, then at every prime 5..131.
default(parisize, "400M");
read("irred.gp");
read("G.gp");
prim(F) = F / content(F);
G = GZ;
G7 = prim(subst(G, 'y, 7*'y + 3));
G13 = prim(simplify(subst(G, 'y, (11*'y + 130)/('y + 13)) * ('y + 13)^23));
models = [["G", G], ["G7", G7], ["G13", G13]];
for (m = 1, 3, print(models[m][1], ": bidegree (", poldegree(models[m][2], 'x), ",", poldegree(models[m][2], 'y), "), content ", content(models[m][2])));
\\ the design's primes, every model (a model degenerate at p still gets the criterion: it is
\\ about F mod p only)
{
  forprime (p = 5, 131,
    my(s = Str("p = ", p, ":"));
    for (m = 1, 3,
      my(r = crit(models[m][2], 4, 23, p));
      s = Str(s, "  ", models[m][1], " ", r));
    print(s));
}
\\ PARI's bivariate factorization over F_p as an independent check at a few primes
{
  foreach ([5, 7, 11, 13], p,
    for (m = 1, 3,
      my(f = factor(Mod(1, p) * models[m][2]));
      print("factor ", models[m][1], " mod ", p, ": ", #f~, " factor(s), x-degrees ",
            vector(#f~, i, poldegree(f[i, 1], 'x)), " y-degrees ", vector(#f~, i, poldegree(f[i, 1], 'y)),
            " multiplicities ", f[, 2]~)));
}
