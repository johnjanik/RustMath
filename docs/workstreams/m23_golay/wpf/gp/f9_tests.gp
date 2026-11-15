\\ F-9 implementation: PARI/GP reference values for the new tests of
\\ rustmath-curves/tests/biprojective_small_curves.rs that regression.gp and unit.gp do not
\\ already give. Run as `gp -q f9_tests.gp < /dev/null` in this folder (PARI/GP 2.15.4).
\\ The orders below are those of rustmath-curves/src/nodal_biprojective/irreducible.rs: closed
\\ points of degree d >= 2 as monic g by the code sum g_i p^i ascending; the smooth-point pass
\\ chart by chart (affine, x = inf, y = inf, both), y ascending, the roots x ascending.
default(parisize, "200M");
read("irred.gp");

\\ the monic irreducible g of degree d over F_p in the Rust order
rbase(p, d) = {
  my(L = List());
  for (code = 0, p^d - 1,
    my(g = 't^d + sum(e = 0, d - 1, (code \ p^e) % p * 't^e));
    if (polisirreducible(Mod(1, p) * g), listput(L, g)));
  Vec(L);
}
\\ the x-fibre (ruling 0) or y-fibre (ruling 1) over the root of g: irreducible of full degree?
rfib(F, a, b, p, ruling, g) = {
  my(t = ffgen(Mod(1, p) * g, 't), n = if (ruling == 0, a, b), u = if (ruling == 0, 'x, 'y),
     w = if (ruling == 0, 'y, 'x), cs, f);
  cs = vector(n + 1, i, subst(polcoef(F, i - 1, u), w, t) * t^0);
  if (cs[n + 1] == 0, return (0));
  f = Pol(Vecrev(cs), 'z);
  polisirreducible(f);
}
\\ the first g of degree d (Rust order, at most 64) whose fibre in the ruling is irreducible
rfirst(F, a, b, p, ruling, d) = {
  my(B = rbase(p, d));
  for (k = 1, min(#B, 64), if (rfib(F, a, b, p, ruling, B[k]), return ([B[k], k])));
  0;
}
\\ any F_p-rational fibre irreducible of full degree, in either ruling?
\\ (the base points 0, 1, ..., p - 1, then infinity = -1, as in the Rust order)
anyrational(F, a, b, p) = {
  my(vs = concat([0 .. p - 1], [-1]));
  foreach (vs, v, if (irrfull(fibx(F, a, b, p, v), a, p), return (["x-fibre", v])));
  foreach (vs, v, if (irrfull(fiby(F, a, b, p, v), b, p), return (["y-fibre", v])));
  0;
}
\\ the first smooth F_p-point in the Rust order, [x, y] with -1 for infinity, or 0
rsmooth(F, a, b, p) = {
  my(C = charts(F, a, b), G, gx, gy, xs);
  for (c = 1, 4,
    G = Mod(1, p) * C[c]; gx = deriv(G, 'x); gy = deriv(G, 'y);
    my(ys = if (c <= 2, [0 .. p - 1], [0]));
    foreach (ys, yv,
      xs = if (c == 2 || c == 4, [0],
             my(f = subst(G, 'y, yv)); if (f == 0, [0 .. p - 1],
               vecsort([lift(r) | r <- polrootsmod(lift(f), p)])));
      foreach (xs, xv,
        if (subst(subst(G, 'x, xv), 'y, yv) == 0
            && (subst(subst(gx, 'x, xv), 'y, yv) != 0 || subst(subst(gy, 'x, xv), 'y, yv) != 0),
          return ([if (c == 2 || c == 4, -1, xv), if (c == 3 || c == 4, -1, yv)])))));
  0;
}
\\ the number of factors (with multiplicity) of F over F_(p^n)
nfact(F, p, n) = {
  my(f = if (n == 1, factor(Mod(1, p) * F), my(w = ffgen(ffinit(p, n), 'w)); factor(F * w^0)));
  [#f~, vecsum(f[, 2]~)];
}
\\ #V(F)(F_(p^k)) by brute force over P^1 x P^1 (F_(p^k)), for a form of bidegree (a, b)
brutek(F, a, b, p, k) = {
  my(t = ffgen(ffinit(p, k), 't), g = ffprimroot(t), C = charts(F, a, b), n = 0);
  my(els = concat([0 * t], [g^i | i <- [0 .. p^k - 2]]));
  foreach (els, xv, foreach (els, yv, n += (substvec(C[1], ['x, 'y], [xv, yv]) * t^0 == 0)));
  foreach (els, yv, n += (substvec(C[2], ['x, 'y], [0, yv]) * t^0 == 0));
  foreach (els, xv, n += (substvec(C[3], ['x, 'y], [xv, 0]) * t^0 == 0));
  n + (substvec(C[4], ['x, 'y], [0, 0]) * t^0 == 0);
}
mat(F, a, b, p) = matrix(a + 1, b + 1, i, j, lift(Mod(polcoef(polcoef(F, i - 1, 'x), j - 1, 'y), p)));

print("== (iv) E1 alone at 13 (regress.gp): the first smooth F_13-point in the Rust order");
E1 = (y^2 + 11*y + 5)*x^2 + (8*y^2 + 12*y + 6)*x + (11*y^2 + 9*y + 11);
print("first rational x-fibre irreducible: ", anyrational(E1, 2, 2, 13), "; first smooth point: ", rsmooth(E1, 2, 2, 13));

print("== S1: a (2,2) form at p = 3 with no irreducible F_3-rational fibre (F-9 review 1, m3)");
S1 = (y^2 + y + 1)*x^2 + (y + 1)*x + (2*y + 1);
print("coefficient matrix [i][j] of x^i y^j: ", mat(S1, 2, 2, 3));
print("factors over F_3 and F_9 (number, with multiplicity): ", nfact(S1, 3, 1), " ", nfact(S1, 3, 2));
print("an F_3-rational fibre irreducible of full degree: ", anyrational(S1, 2, 2, 3));
D1 = Mod(1, 3) * (polcoef(S1, 1, 'x)^2 - 4 * polcoef(S1, 2, 'x) * polcoef(S1, 0, 'x));
print("x-discriminant ", lift(D1), ": degree ", poldegree(D1), ", squarefree ", poldegree(gcd(D1, deriv(D1))) == 0);
print("the first x-fibre over a point of degree 2 irreducible over F_9 (Rust order): ", rfirst(S1, 2, 2, 3, 0, 2));
print("#V(S1)(F_3^k), k = 1..3, brute force: ", vector(3, k, brutek(S1, 2, 2, 3, k)));
print("hyperellcharpoly of y^2 = D: ", hyperellcharpoly(D1), "; counts ", vector(3, k, 3^k + 1 - polsym(hyperellcharpoly(D1), k)[k + 1]));

print("== S2: a (4,2) form at p = 3, an irreducible F_3-rational fibre and no smooth F_3-point (F-9 review 1, m3)");
S2 = (2*y^2 + 2)*x^4 + (2*y^2 + 2*y + 1)*x^3 + (y^2 + 2*y + 1)*x^2 + (2*y^2 + 2*y + 2)*x + (2*y^2 + 2);
print("coefficient matrix [i][j] of x^i y^j: ", mat(S2, 4, 2, 3));
print("factors over F_3 and F_9 (number, with multiplicity): ", nfact(S2, 3, 1), " ", nfact(S2, 3, 2));
print("the first F_3-rational fibre irreducible of full degree: ", anyrational(S2, 4, 2, 3), "; smooth F_3-point: ", rsmooth(S2, 4, 2, 3));
print("the first fibre over a point of degree 2 irreducible over F_9 (Rust order): x-fibre ", rfirst(S2, 4, 2, 3, 0, 2), ", y-fibre ", rfirst(S2, 4, 2, 3, 1, 2));

print("== E6: y^3 - x^4 of bidegree (4,3) at 11");
E6 = 'y^3 - 'x^4;
print("an F_11-rational fibre irreducible of full degree: ", anyrational(E6, 4, 3, 11));
print("the first x-fibre over a point of degree 2 irreducible over F_121 (Rust order): ", rfirst(E6, 4, 3, 11, 0, 2));
