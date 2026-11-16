\\ F-9 design: PARI's bivariate factorization of the CC0 models over F_p at the remaining design
\\ primes (31, 107; 5..13 are in m23_cc0.gp), as an independent check of F_p-irreducibility.
default(parisize, "400M");
read("G.gp");
prim(F) = F / content(F);
G = GZ;
G7 = prim(subst(G, 'y, 7*'y + 3));
G13 = prim(simplify(subst(G, 'y, (11*'y + 130)/('y + 13)) * ('y + 13)^23));
models = [["G", G], ["G7", G7], ["G13", G13]];
{
  foreach ([31, 107], p,
    for (m = 1, 3,
      my(f = factor(Mod(1, p) * models[m][2]));
      print("factor ", models[m][1], " mod ", p, ": ", #f~, " factor(s), bidegrees ",
            vector(#f~, i, [poldegree(f[i, 1], 'x), poldegree(f[i, 1], 'y)]), " multiplicities ", f[, 2]~)));
}
