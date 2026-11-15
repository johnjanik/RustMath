\\ F-9 design: an independent PARI/GP prototype of the absolute irreducibility criterion.
\\ A form F of formal bidegree (a, b) is given by its affine polynomial in x (degree <= a) and
\\ y (degree <= b) with integer coefficients, reduced mod p here. x is the fibre variable of
\\ FibresInX (degree a), y the base; FibresInY swaps them.
\\
\\ Criterion (F9.md, Section 2):
\\  (C1) cont_Y(F) and cont_X(F) trivial as binary forms (no factor in one variable alone,
\\       including the lines at infinity);
\\  (C2) a base point v0 in P^1(F_p) whose fibre is irreducible over F_p of full degree
\\       (X-fibres first, then Y-fibres): then F is irreducible over F_p;
\\  (C3) gcd(a, b) = 1, or a smooth F_p-rational point: then F is absolutely irreducible.
\\ Returns [verdict, detail]: verdict in "absolute", "line", "nofibre", "nosmooth".

cx(F, i) = polcoef(F, i, 'x);   \\ coefficient of x^i, a polynomial in y
cy(F, j) = polcoef(F, j, 'y);   \\ coefficient of y^j, a polynomial in x

\\ reduce an integer polynomial mod p, coefficients in [0, p)
red(f, p) = lift(Mod(1, p) * f);

\\ the content in y as a binary form of degree b: [gcd of the c_i mod p, 1 if y = inf divides]
conty(F, a, b, p) = {
  my(g = 0, inf = 1, c);
  for (i = 0, a, c = red(cx(F, i), p); g = gcd(g, Mod(1, p) * c); if (poldegree(c, 'y) == b, inf = 0));
  [lift(g), inf];
}
contx(F, a, b, p) = {
  my(g = 0, inf = 1, c);
  for (j = 0, b, c = red(cy(F, j), p); g = gcd(g, Mod(1, p) * c); if (poldegree(c, 'x) == a, inf = 0));
  [lift(g), inf];
}

\\ the X-fibre over y0 in F_p (y0 = -1 means infinity): an integer polynomial in x
fibx(F, a, b, p, y0) = if (y0 < 0, red(sum(i = 0, a, polcoef(cx(F, i), b, 'y) * 'x^i), p), red(subst(F, 'y, y0), p));
fiby(F, a, b, p, x0) = if (x0 < 0, red(sum(j = 0, b, polcoef(cy(F, j), a, 'x) * 'y^j), p), red(subst(F, 'x, x0), p));

irrfull(f, n, p) = poldegree(f) == n && polisirreducible(Mod(1, p) * f);

\\ the chart polynomials: F, F(1/x, y) x^a, F(x, 1/y) y^b, both
charts(F, a, b) = {
  my(Fx = simplify('x^a * subst(F, 'x, 1/'x)), Fy = simplify('y^b * subst(F, 'y, 1/'y)));
  my(Fxy = simplify('y^b * subst(Fx, 'y, 1/'y)));
  [F, Fx, Fy, Fxy];
}

\\ a smooth F_p-rational point, or 0: [x, y] with -1 for infinity
smoothpoint(F, a, b, p) = {
  my(C = charts(F, a, b), G, gx, gy, xs, ys, xv, yv);
  for (c = 1, 4,
    G = Mod(1, p) * C[c]; gx = deriv(G, 'x); gy = deriv(G, 'y);
    xs = if (c == 1 || c == 3, [0 .. p - 1], [0]);
    ys = if (c == 1 || c == 2, [0 .. p - 1], [0]);
    for (i = 1, #xs, for (j = 1, #ys,
      xv = xs[i]; yv = ys[j];
      if (subst(subst(G, 'x, xv), 'y, yv) == 0
          && (subst(subst(gx, 'x, xv), 'y, yv) != 0 || subst(subst(gy, 'x, xv), 'y, yv) != 0),
        return ([if (c == 2 || c == 4, -1, xv), if (c == 3 || c == 4, -1, yv)])))));
  0;
}

\\ the criterion; tries counts the fibres examined
crit(F, a, b, p) = {
  my(cY = conty(F, a, b, p), cX = contx(F, a, b, p), w = 0, tries = 0);
  if (poldegree(cY[1]) > 0 || cY[2], return (["line", ["horizontal", cY]]));
  if (poldegree(cX[1]) > 0 || cX[2], return (["line", ["vertical", cX]]));
  for (y0 = 0, p, my(v = if (y0 == p, -1, y0)); tries++;
    if (irrfull(fibx(F, a, b, p, v), a, p), w = ["FibresInX", v, tries]; break));
  if (w == 0,
    for (x0 = 0, p, my(v = if (x0 == p, -1, x0)); tries++;
      if (irrfull(fiby(F, a, b, p, v), b, p), w = ["FibresInY", v, tries]; break)));
  if (w == 0, return (["nofibre", tries]));
  if (gcd(a, b) == 1, return (["absolute", [w, "coprime"]]));
  my(s = smoothpoint(F, a, b, p));
  if (s == 0, return (["nosmooth", [w, gcd(a, b)]]));
  ["absolute", [w, s]];
}

\\ X-fibres over closed points of degree d > 1 (for the record): the first monic irreducible
\\ g of degree d whose fibre is irreducible over F_p[y]/(g) of degree a; [g, number tried] or 0.
fibx_deg(F, a, b, p, d) = {
  my(t = ffgen(ffinit(p, d), 't), n = 0, cs, f, seen = List());
  for (k = 0, p^d - 1,
    my(v = sum(e = 0, d - 1, (k \ p^e) % p * t^e));
    if (poldegree(minpoly(v)) != d, next);
    my(m = minpoly(v)); if (setsearch(Set(seen), m), next); listput(seen, m); n++;
    cs = vector(a + 1, i, subst(cx(F, i - 1), 'y, v) * t^0);
    if (cs[a + 1] == 0, next);
    f = Pol(Vecrev(cs), 'x);
    if (polisirreducible(f), return ([m, n])));
  0;
}
