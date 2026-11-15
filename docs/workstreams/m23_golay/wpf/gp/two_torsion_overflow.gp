\\ Fix-up: a genus-67 polynomial with the functional equation whose 2-torsion order overflows u64.
f31 = x^31 + x^3 + 1; f19 = x^19 + x^5 + x^2 + x + 1; f17 = x^17 + x^3 + 1;
L = [f31, f19, f17];
{
for(i = 1, 3,
  my(f = L[i] * Mod(1, 2), d = poldegree(L[i]));
  print("deg ", d, ": irreducible ", polisirreducible(f), ", self-reciprocal ", f == polrecip(f),
        ", 2^d - 1 prime ", isprime(2^d - 1), ", x^(2^d-1) = 1: ", Mod(x, f)^(2^d - 1) == 1));
}
F = lift(prod(i = 1, 3, L[i] * polrecip(L[i])) * Mod(1, 2));
print("deg F = ", poldegree(F), ", palindromic: ", F == polrecip(F));
print("factor(F mod 2) degrees: ", apply(poldegree, lift(factormod(F, 2)[,1]~)));
o = lcm([2^31 - 1, 2^19 - 1, 2^17 - 1]);
print("lcm = ", o, " > 2^64: ", o > 2^64, ", lcm of the first two < 2^64: ", (2^31-1)*(2^19-1) < 2^64);
c = vector(68, j, polcoef(F, 135 - j));
print("c[0..=67] (descending, coefficient of T^(134-i)) = ", c);
