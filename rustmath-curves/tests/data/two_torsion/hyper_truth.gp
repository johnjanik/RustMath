\\ Exact Frobenius module Jac[2] of hyperelliptic curves y^2 = f(x) over F_p (design C57.md
\\ Section 3.5): J[2] = even subsets of the 2g+2 branch points modulo the whole set, Frobenius
\\ permutes the branch points with the cycle type of f mod p (plus a fixed point at infinity when
\\ deg f is odd). Output per curve: p g [P descending] [cycle type] order [dim ker (F-1)^j, j=1..2g].
setrand(20261120);
\\ the matrix of the permutation s (vector of images, 1-based) on J in the basis b_i = e_i + e_n, i < n-1
jmat(s) = {
  my(n = #s, m = n - 2, M = matrix(m, m, i, j, Mod(0, 2)), v, w);
  for (j = 1, m,
    \\ sigma(b_j) = e_{s[j]} + e_{s[n]} = b_{s[j]} + b_{s[n]}, with b_n = 0 and b_{n-1} = sum_{i<n-1} b_i
    v = vector(n, i, Mod(0, 2)); v[s[j]] += 1; v[s[n]] += 1;
    \\ v as a combination of the b_i: coefficient of b_i (i < n) is v[i]
    w = vector(m, i, v[i] + v[n - 1]);
    M[, j] = w~);
  M;
}
cyc(d) = { my(n = vecsum(d), s = vector(n), o = 0); for (k = 1, #d, for (i = 1, d[k], s[o + i] = o + (i % d[k]) + 1); o += d[k]); s; }
morder(M) = { my(I = M^0, A = M, t = 1); while (A != I, A = A * M; t++); t; }
{
  forprime(p = 3, 47,
    for (g = 1, 4,
      for (t = 1, 60,
        my(n = 2*g + 1 + random(2), f, fa, d = [], P, M, ks);
        f = x^n + sum(i = 0, n - 1, random(p) * x^i);
        if (issquarefree(Mod(1, p) * f) && poldegree(Mod(1, p) * f) == n,
          P = hyperellcharpoly(Mod(1, p) * f);
          fa = factormod(f, p);
          for (i = 1, #fa~, for (j = 1, fa[i, 2], d = concat(d, poldegree(fa[i, 1]))));
          if (n % 2 == 1, d = concat(d, 1));
          M = jmat(cyc(d));
          ks = vector(2*g, j, 2*g - matrank((M - M^0)^j));
          print(p, " ", g, " ", Vec(P), " ", d, " ", morder(M), " ", ks)))));
}
quit;
