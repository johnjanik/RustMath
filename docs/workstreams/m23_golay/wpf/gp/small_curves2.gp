\\ WP-F implementation: further small curves for rustmath-curves/tests/biprojective_small_curves.rs.
\\ Curves V^2 + b(T) V + c(T) = 0 of bidegree (4,2); D = b^2 - 4c; normalization y^2 = D / square part.
\\ c is chosen as (b^2/4) mod M plus a free high part, so that M | D.
mkM(p, b, M, chigh) = { my(R = lift(((b^2/4 - chigh)*Mod(1,p)) % (M*Mod(1,p)))); lift((chigh + R)*Mod(1,p)); }
show(name, p, b, c) = {
  my(D = lift((b^2 - 4*c)*Mod(1,p)), fa = factormod(D, p), rest = 1);
  for(i=1, #fa[,1], my(e = fa[i,2], f = lift(fa[i,1])); rest *= f^(e%2));
  rest = rest * pollead(D);
  my(cp = hyperellcharpoly(rest*Mod(1,p)));
  print(name, ": p=", p, " b=", Vecrev(b), " c=", Vecrev(lift(c*Mod(1,p))), " (ascending in T)");
  print("    D factor ", apply(x->lift(x), fa[,1]~), " mults ", fa[,2]~);
  print("    normalization y^2 = ", lift(rest), " ; charpoly = ", cp);
}
{
p = 11;
\\ J: node of residue degree 2 (b not even: the two geometric nodes lie over conjugate V)
b = T^4 + 3*T^3 + T + 5;
show("J", p, b, mkM(p, b, (T^2-2)^2, 6*T^4));
\\ J2: b even: the two conjugate nodes lie in the fibre over one rational V (quadratic extension path)
b = T^4 + 3*T^2 + 5;
show("J2", p, b, mkM(p, b, (T^2-2)^2, 6*T^4));
\\ K1: b even, (T^2-2)^3 | D: two conjugate cusps in one rational fibre
show("K1", p, b, mkM(p, b, (T^2-2)^3, 0));
p = 13;
\\ K2: b(1) = b(4): a cusp at T = 1 and a node at T = 4 in the same fibre (split two-point fibre)
b = (T - 1)*(T - 4)*(T^2 + 2*T + 7) + 3;
show("K2", p, b, mkM(p, b, (T-1)^3*(T-4)^2, 0));
print("K2 b(1), b(4) mod 13: ", [lift(Mod(subst(b,T,1),13)), lift(Mod(subst(b,T,4),13))]);
}
\\ branch classes: node at t0 with D = (T - t0)^2 h: rational iff h(t0) is a square in the residue field
{
p = 11; b = T^4 + 3*T^3 + T + 5; c = mkM(p, b, (T^2-2)^2, 6*T^4);
my(D = (b^2 - 4*c)*Mod(1,p), h = D / (T^2-2)^2, K = ffgen((T^2-2)*Mod(1,p), 'w), t0 = K);
print("J h(sqrt 2) square in F_121: ", issquare(subst(lift(h), T, t0)));
b = T^4 + 3*T^2 + 5; c = mkM(p, b, (T^2-2)^2, 6*T^4); D = (b^2 - 4*c)*Mod(1,p); h = D / (T^2-2)^2;
print("J2 h(sqrt 2) square in F_121: ", issquare(subst(lift(h), T, t0)));
p = 13; b = (T - 1)*(T - 4)*(T^2 + 2*T + 7) + 3; c = mkM(p, b, (T-1)^3*(T-4)^2, 0);
D = (b^2 - 4*c)*Mod(1,p); h = D / (T-4)^2;
print("K2 node at 4: h(4) square mod 13: ", issquare(subst(lift(h), T, 4)*Mod(1,13)));
}
