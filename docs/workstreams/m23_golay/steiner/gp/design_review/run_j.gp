m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("rev.gp");
read(Str(m23dir, "/polynomial_f2.gp")); read(Str(m23dir, "/polynomial_f1.gp"));
{
foreach([[f2, "f2"], [f1, "f1"]], P, f = P[1]; b = fbest(f);
  bj5 = floor(5313 * (b / 128)^5) + 1; bj7 = floor(253 * (b / 128)^7) + 1;
  N = max(nrig(29, bj5, 0), nrig(29, bj7, 0)); r = froots(f, 29, N); q = 29^N;
  print(P[2], ": BJ5 = ", bj5, " BJ7 = ", bj7, " N = ", N, " J5- = ", Jk(r, Hm, 5, q), " J7- = ", Jk(r, Hm, 7, q)));
\\ C23 and x^23 - x - 1
c = polsubcyclo(47, 23); bc = B5(c); print("C23: best ", fbest(c), " B5 ", bc);
foreach([2, 3, 5], p, if(polisirreducible(Mod(1, p) * c), N = nrig(p, bc, 0);
  forstep(M = N, N + 20, 10, v = both(c, p, M); print("  C23 p = ", p, " N = ", M, ": ", v))));
s = x^23 - x - 1; bs = B5(s); print("S23: best ", fbest(s), " B5 ", bs, " N_rig(23) ", nrig(23, bs, 0));
v = both(s, 23, 30); print("  x^23-x-1 at 23, N = 30: ", v);
}
quit;
