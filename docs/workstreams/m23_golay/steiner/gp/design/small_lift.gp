\\ Reference values for the Frobenius-ordered roots on small cases (PARI/GP 2.15.4).
\\ Roots by polrootspadic over Q_p[t]/(f(t)); order: alpha_i = the root congruent to t^(p^i) mod p.
ordered(f, p, N) =
{
  my(T = subst(f, x, t), q = p^N, R = polrootspadic(f, [T, p], N), n = poldegree(f), out = vector(n));
  my(E = vector(n, j, Pol(apply(a -> lift(Mod(truncate(a), q)), Vec(lift(R[j]))), t)));
  for(i = 0, n - 1,
    my(tp = lift(lift(Mod(Mod(1,p) * t, Mod(1,p) * T)^(p^i))), hit = 0);
    for(j = 1, n, if(lift(Mod(Mod(1,p)*E[j] - Mod(1,p)*tp, Mod(1,p)*T)) == 0, hit = j));
    out[i+1] = Vecrev(E[hit], n));
  out;
}
f = x^3 + x + 1; print("x^3+x+1 irreducible mod 5: ", polisirreducible(Mod(1,5)*f));
print("roots mod 5^6: ", ordered(f, 5, 6));
g = x^23 - x - 1; p = 2; while(!polisirreducible(Mod(1,p)*g), p = nextprime(p+1)); print("x^23-x-1: first inert prime ", p);
r = ordered(g, p, 4); print("phi(x) mod ", p, "^4: ", r[2]);
h = x^5 + 2; print("x^5+2 mod 3 irreducible: ", polisirreducible(Mod(1,3)*h));
