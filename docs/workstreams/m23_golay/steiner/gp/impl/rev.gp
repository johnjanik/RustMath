\\ Reviewer's own PARI/GP check (independent of design/SL/gp/steiner.gp): Newton lift of
\\ the Frobenius image of t in (Z/p^N)[t]/(f), roots by composition, heptads as bitmasks.
hept(sgn) = {
  my(g = if(sgn > 0, 2^11+2^9+2^7+2^6+2^5+2+1, 2^11+2^10+2^6+2^5+2^4+2^2+1), H = List(), w);
  for(m = 0, 4095, w = 0; for(i = 0, 11, if(bittest(m, i), w = bitxor(w, shift(g, i))));
    if(hammingweight(w) == 7, listput(H, [i | i <- [0..22], bittest(w, i)])));
  Vec(H);
}
\\ Fujiwara best = max_k ceil(256 |a_{n-k}|^(1/k)) (exact)
fbest(f) = { my(n = poldegree(f), b = 0, a, r); for(k = 1, n, a = abs(polcoef(f, n - k));
  if(a, r = sqrtnint(a * 256^k, k); if(r^k < a * 256^k, r++); b = max(b, r))); b; }
B5(f) = floor(253 * (7 * fbest(f) / 128)^5) + 1;
nrig(p, B, e) = { my(N = 1); while(p^N <= 2 * B * 10^e, N++); N; }
\\ Newton lift of y = phi(t): returns the vector of 23 roots as polynomials in t mod (p^N, f)
froots(f, p, N) = {
  my(T = subst(f, x, t), n = poldegree(f), y, u, q = p^N, k = 1, d = deriv(T, t), roots);
  my(T1 = Mod(1, p) * T, Y1 = Mod(Mod(1, p) * t, T1)^p);
  y = lift(lift(Y1)); u = lift(lift(1 / subst(d, t, Y1)));
  while(k < N, k = min(2 * k, N);
    my(Q = p^k, TT = Mod(1, Q) * T, Y = Mod(Mod(1, Q) * y, TT), U = Mod(Mod(1, Q) * u, TT), D);
    D = subst(d, t, Y); U = U * (2 - D * U);
    Y = Y - subst(T, t, Y) * U;
    D = subst(d, t, Y); U = U * (2 - D * U);
    y = lift(lift(Y)); u = lift(lift(U)));
  my(TT = Mod(1, q) * T, Y = Mod(Mod(1, q) * y, TT));
  if(subst(T, t, Y) != 0, error("f(y) != 0"));
  if(Mod(1, p) * (y - lift(lift(Y1))) != 0, error("y != t^p mod p"));
  roots = vector(n); roots[1] = Mod(Mod(1, q) * t, TT);
  for(i = 2, n, roots[i] = subst(lift(roots[i - 1]), t, Y));
  if(subst(lift(roots[n]), t, Y) != roots[1], error("phi^n != id"));
  roots;
}
sres(e, q) = { my(c = lift(e)); if(poldegree(c, t) > 0, error("not constant")); c = lift(polcoef(c, 0, t)); if(c > q / 2, c - q, c); }
T5(r, H, q) = sres(sum(k = 1, #H, sum(j = 1, 7, r[H[k][j] + 1])^5), q);
Hp = hept(1); Hm = hept(-1);
both(f, p, N) = { my(r = froots(f, p, N), q = p^N); [T5(r, Hp, q), T5(r, Hm, q)]; }
esym(v, k) = { my(e = vector(k + 1, i, 0)); e[1] = 1; for(j = 1, #v, forstep(i = k, 1, -1, e[i + 1] += e[i] * v[j])); e[k + 1]; }
Jk(r, H, k, q) = sres(sum(b = 1, #H, esym(vector(7, j, r[H[b][j] + 1]), k)), q);
