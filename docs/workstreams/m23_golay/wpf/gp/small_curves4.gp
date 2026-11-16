\\ WP-F implementation: all clean candidates with the branch class of the degree-2 nodes.
read("small_curves2.gp");
all(name, p, b, M, e) = {
  my(K = ffgen((T^2-2)*Mod(1,p), 'w));
  for(k = 1, p - 1,
    my(c = mkM(p, b, M, k*T^4), D = lift((b^2 - 4*c)*Mod(1,p)), fa = factormod(D, p), ok = 1);
    if(poldegree(D) != 8, next);
    for(i = 1, #fa[,1], my(f = lift(fa[i,1]), m = fa[i,2]);
      if(m > 1 && ((M*Mod(1,p)) % f != 0 || m != e), ok = 0));
    if(ok, my(h = (D*Mod(1,p)) / (T^2-2)^2); print1(name, " k=", k, " square=", issquare(subst(lift(h), T, K)), "  "); show(name, p, b, c)));
}
{
all("J", 11, T^4 + 3*T^3 + T + 5, (T^2-2)^2, 2);
all("J2", 11, T^4 + 3*T^2 + 5, (T^2-2)^2, 2);
}
