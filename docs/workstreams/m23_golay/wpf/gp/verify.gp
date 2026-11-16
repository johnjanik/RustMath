\\ WP-F minors: independent check of the adversarial cases moved into
\\ rustmath-curves/tests/biprojective_small_curves.rs. Input: p and the ascending coefficient
\\ vectors b, c of F = X^2 + b(Y) X + c(Y). Nothing is taken from adv2.gp but these vectors.
\\ Output: D = b^2 - 4c factored mod p, the squarefree part H (normalization y^2 = H), genus,
\\ hyperellcharpoly(H), N_1..N_6, and for each multiple factor S of D (a singular place):
\\ its degree, ord_S D, and whether (D / S^ord)(y0) is a square in F_p[Y]/(S).
chk(name, p, bv, cv) = {
  my(b = Polrev(bv, 'Y) * Mod(1, p), c = Polrev(cv, 'Y) * Mod(1, p), D = b^2 - 4*c);
  my(fa = factor(D), H = pollead(D), g, cp, ps, N, sing = List());
  for(i = 1, #fa~, if(fa[i, 2] % 2 == 1, H *= fa[i, 1]);
    if(fa[i, 2] >= 2, listput(sing, [fa[i, 1], fa[i, 2]])));
  g = (poldegree(H) - 1) \ 2;
  if(g >= 1, cp = hyperellcharpoly(H); ps = polsym(cp, 6), cp = 1; ps = vector(7, i, 0));
  N = vector(6, k, p^k + 1 - if(g >= 1, ps[k + 1], 0));
  print(name, ": p = ", p, ", D = ", lift(fa));
  print("  H = ", lift(H), ", g = ", g, ", hyperellcharpoly(H) = ", cp);
  print("  N_1..N_6 = ", N);
  for(j = 1, #sing, my(S = sing[j][1], e = sing[j][2], h = D / S^e, K, sq);
    K = ffgen(lift(S) * Mod(1, p), 'w);
    sq = issquare(subst(lift(h), 'Y, K));
    print("  place ", lift(S), ": degree ", poldegree(S), ", ord_S D = ", e,
          " (A", e - 1, "), h(y0) square in F_p^", poldegree(S), ": ", sq));
}
p = 13;
chk("node_d3_g1_sq0", p, [7, 1, 10, 1, 0, 1], [6, 7, 4, 12, 2, 0]);
chk("node_d3_g1_sq1", p, [11, 8, 10, 9, 0, 1], [8, 2, 1, 12, 5, 0]);
chk("A3_d2_g1_sq0", p, [12, 3, 1, 3, 11, 8, 1], [7, 10, 12, 11, 7, 4, 0]);
chk("A3_d2_g1_sq1", p, [1, 4, 7, 4, 11, 8, 1], [12, 12, 5, 11, 2, 5, 0]);
chk("A15_d1_g0_sq1", p, [12, 2, 9, 8, 12, 9, 5, 7, 12, 1], [3, 10, 2, 12, 12, 8, 5, 12, 11, 0]);
chk("A15_d1_g0_sq0", p, [11, 12, 10, 0, 3, 8, 0, 1, 6, 1], [7, 2, 3, 11, 12, 7, 11, 0, 12, 0]);
\\ E6: F = Y^3 - X^4 (cols[0][3] = 1, cols[4][0] = p - 1), at p = 7, 11, 13. Not of the shape
\\ above. It is rational: t -> (X, Y) = (t^3, t^4) is a bijection from P^1 (inverse t = Y/X), so
\\ #X0(F_q) = q + 1. Brute force over P^1 x P^1(F_p) as a check (affine part plus the points at
\\ infinity: (inf, inf) only, since Y^3 X0^4 = X^4 Y0^3 forces X0 = 0 <=> Y0 = 0).
{
  for(i = 1, 3, my(q = [7, 11, 13][i], n = 0);
    for(x = 0, q - 1, for(y = 0, q - 1, if(Mod(y, q)^3 == Mod(x, q)^4, n++)));
    print("E6 p = ", q, ": affine points ", n, ", plus (inf, inf): ", n + 1, " = p + 1: ", n + 1 == q + 1));
}
