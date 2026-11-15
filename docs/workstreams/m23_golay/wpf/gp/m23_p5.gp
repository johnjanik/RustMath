\\ F-9 prototype at p = 5, where no F_5-rational fibre is irreducible of full degree (m23_cc0.gp):
\\ the X-fibres over closed points of degree 2 and 3 of the y-line, and the list of the factor
\\ degrees of every rational fibre in both rulings.
default(parisize, "400M");
read("irred.gp");
read("G.gp");
prim(F) = F / content(F);
G = GZ;
G7 = prim(subst(G, 'y, 7*'y + 3));
G13 = prim(simplify(subst(G, 'y, (11*'y + 130)/('y + 13)) * ('y + 13)^23));
models = [["G", G], ["G7", G7], ["G13", G13]];
p = 5;
fdeg(f, p) = my(m = factormod(f, p)); vector(#m~, i, [poldegree(m[i, 1]), m[i, 2]]);
{
  for (m = 1, 3,
    my(F = models[m][2]);
    print(models[m][1], " at 5, X-fibres over y0 = 0..4, inf: ",
          vector(6, k, my(v = if (k == 6, -1, k - 1), f = fibx(F, 4, 23, p, v)); [v, poldegree(f), fdeg(f, p)]));
    print(models[m][1], " at 5, Y-fibres over x0 = 0..4, inf: ",
          vector(6, k, my(v = if (k == 6, -1, k - 1), f = fiby(F, 4, 23, p, v)); [v, poldegree(f), fdeg(f, p)]));
    for (d = 2, 3, print(models[m][1], " at 5, first X-fibre irreducible over F_5^", d, ": ", fibx_deg(F, 4, 23, p, d))));
}
