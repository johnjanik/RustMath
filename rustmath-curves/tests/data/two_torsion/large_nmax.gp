\\ Review 1 of C57, B1: a genus-12 P with the functional equation over q whose reduction mod 2 is
\\ (x+1)^2 f f*, f = x^11 + x^2 + 1 (x of order 2047 mod f), so N_max = 4094. The coefficients
\\ c_0..c_12 (descending) are the bits of the reduction, c_(24-i) = q^(12-i) c_i (the same
\\ construction as synthetic() in rustmath-curves/src/weil.rs tests). For q = 3 and 10^9 + 7:
\\ v_2(Res(P, x^k - 1)) / v_2(Res(P, x^k + 1)) for every k | 4094, and Res(P, x^k - 1) mod 10^30
\\ for k = 1, 23, 89 (the values of jacobian_order(k)).
default(parisizemax, 2^30);
f = x^11 + x^2 + 1; fr = polrecip(f);
bits = lift(Mod(1, 2) * f * fr * (x^2 + 1));
mkP(q) = { my(c = vector(25, i, polcoeff(bits, 25 - i))); for (i = 0, 11, c[25 - i] = q^(12 - i) * c[i + 1]); Pol(c); };
{
foreach ([3, 10^9 + 7], q,
  my(P = mkP(q), s = Str("q=", q, " N=4094 "));
  if (lift(Mod(1, 2) * P) != bits, error("reduction"));
  fordiv(4094, k, my(r = Mod(x, P)^k); s = Str(s, "k", k, ":", valuation(norm(r - 1), 2), "/", valuation(norm(r + 1), 2), " "));
  print(s);
  foreach ([1, 23, 89], k, print("q=", q, " k=", k, " Res(P, x^k - 1) mod 10^30 = ", norm(Mod(x, P)^k - 1) % 10^30, " Res(P, x^k + 1) mod 10^30 = ", norm(Mod(x, P)^k + 1) % 10^30)));
}
quit;
