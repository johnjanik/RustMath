\\ C57 review 2, n1 (rustmath-curves/tests/m23_zeta.rs, c5_order_text_says_what_was_tried):
\\ the elliptic Weil polynomial P = T^2 - 2T + q, q = 21 * 2^128 + 1 the least prime of the
\\ form k 2^128 + 1. a = 2 is prime to q and |a| <= 2 sqrt(q), so P is the Frobenius polynomial
\\ of an ordinary elliptic curve over F_q (Deuring, Waterhouse). P mod 2 = (x + 1)^2 and
\\ #E(F_q) = q - 1 = 21 * 2^128.
\\ Run: gp -q skipped_ec.gp > skipped_ec_pari.txt
k = 1; while (!isprime(k * 2^128 + 1), k++); q = k * 2^128 + 1;
print("k = ", k, ", q = ", q);
P = x^2 - 2*x + q;
print("P mod 2 = ", lift(Mod(1, 2) * P));
for (j = 1, 2, \
  u = polresultant(P, x^j - 1); t = polresultant(P, x^j + 1); \
  print("k = ", j, ": v2 #E = ", valuation(u, 2), ", v2 #E' = ", valuation(t, 2)));
\\ the 2-adic valuations of the elementary divisors of multiplication by x^j -+ 1 on Z[x]/(P)
M(g) = matrix(2, 2, r, c, polcoef(lift(Mod(x^(c - 1) * g, P)), r - 1));
for (j = 1, 2, \
  print("k = ", j, " untwisted SNF 2-valuations: ", apply(d -> valuation(d, 2), matsnf(M(x^j - 1))), \
        "; twisted: ", apply(d -> valuation(d, 2), matsnf(M(x^j + 1)))));
