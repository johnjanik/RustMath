\\ Independent PARI/GP reference for the Steiner labelling (no code from arith.py).
\\ Roots: polrootspadic over the unramified extension Q_p[t]/(f(t)), converted to exact
\\ elements of (Z/p^N)[t]/(f); Frobenius order by matching each root mod p with
\\ t^(p^i) mod (p, f). Heptads: weight-7 words of the cyclic code of g_+ or g_-.
heptads(sgn) =
{
  my(g = if(sgn > 0, z^11+z^9+z^7+z^6+z^5+z+1, z^11+z^10+z^6+z^5+z^4+z^2+1), H = List(), w, s);
  for(m = 0, 2^12 - 1,
    w = Mod(1,2) * subst(Pol(binary(m)), x, z) * g;
    w = lift(lift(Mod(w, Mod(1,2)*(z^23-1))));
    s = [i | i <- [0..22], polcoef(w, i, z) != 0];
    if(#s == 7, listput(H, s)));
  Vec(H);
}
frob_roots(f, p, N) =
{
  my(T = subst(f, x, t), q = p^N, R = polrootspadic(f, [T, p], N), out = vector(23), tp, E, Tq = Mod(1,q)*T);
  if(#R != 23, error("not 23 roots"));
  E = vector(23, j, my(c = Vec(lift(R[j]))); Mod(Pol(apply(a -> Mod(truncate(a), q), c), t), Tq));
  for(i = 0, 22,
    tp = lift(lift(Mod(Mod(1,p) * t, Mod(1,p) * T)^(p^i)));
    my(hit = 0);
    for(j = 1, 23,
      my(rj = Pol(apply(c -> Mod(lift(c), p), Vec(lift(E[j]))), t));
      if(lift(Mod(rj - Mod(1,p)*tp, Mod(1,p)*T)) == 0, if(hit, error("two roots match")); hit = j));
    if(!hit, error("no root matches t^(p^i)"));
    out[i+1] = E[hit]);
  \\ exact check: each is a root of f modulo p^N
  for(i = 1, 23, if(subst(f, x, out[i]) != 0, error("not a root mod p^N")));
  out;
}
symres(e, q) = { my(c = lift(e)); if(poldegree(c, t) > 0, error("not constant")); c = lift(polcoef(c, 0, t));
  if(c > q/2, c - q, c); }
t5(roots, H, q) = symres(sum(k = 1, #H, sum(j = 1, 7, roots[H[k][j] + 1])^5), q);
esym(v, k) = { my(e = vector(k + 1, i, 0)); e[1] = 1; for(j = 1, #v, forstep(i = k, 1, -1, e[i+1] += e[i] * v[j])); e[k+1]; }
jk(roots, H, k, q) = symres(sum(b = 1, #H, esym(vector(7, j, roots[H[b][j] + 1]), k)), q);
