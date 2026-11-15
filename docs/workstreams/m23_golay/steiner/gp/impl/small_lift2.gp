\\ An independent reference (polrootspadic ordered by t^(p^i) mod p, as design SL gp/small_lift.gp).
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
h = x^5 - x + 1; print("x^5-x+1 irreducible mod 3: ", polisirreducible(Mod(1,3)*h));
print("phi(t) mod 3^7: ", ordered(h, 3, 7)[2]);
