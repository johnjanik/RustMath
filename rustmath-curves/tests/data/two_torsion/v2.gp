\\ v_2 of #Jac(F_{p^k}) = Res(P, x^k - 1) and of the quadratic twist Res(P, x^k + 1) for every
\\ k | N_max, and P mod 2, for every row (design C57.md Section 6). Independent of the Python
\\ prototype (PARI polresultant, factormod).
read("rows.gp");
ordx(f) = { my(d = poldegree(f), n = 2^d - 1, f2 = Mod(1, 2) * f); fordiv(n, k, if (Mod(x, f2)^k == Mod(1, f2), return(k))); };
{
for (i = 1, #ROWS,
  my(r = ROWS[i], p = r[2], a = r[4..7], P, fa, S = 1, emax = 1, N, s = "");
  P = x^8 + a[1]*x^7 + a[2]*x^6 + a[3]*x^5 + a[4]*x^4 + p*a[3]*x^3 + p^2*a[2]*x^2 + p^3*a[1]*x + p^4;
  fa = factormod(P, 2);
  for (j = 1, #fa~, S = lcm(S, ordx(lift(fa[j, 1]))); emax = max(emax, fa[j, 2]));
  N = S * 2^ceil(log(emax)/log(2) - 1e-9);
  s = Str(if (r[1], "L", "X"), " ", p, " ", r[3], " N=", N, " ");
  fordiv(N, k, s = Str(s, "k", k, ":", valuation(polresultant(P, x^k - 1), 2), "/", valuation(polresultant(P, x^k + 1), 2), " "));
  print(s));
}
quit;
