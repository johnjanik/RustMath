m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("rev.gp");
read(Str(m23dir, "/polynomial_f2.gp")); read(Str(m23dir, "/polynomial_f1.gp"));
{
foreach([[f2, "f2"], [f1, "f1"]], P, f = P[1]; b = B5(f);
  print(P[2], ": best = ", fbest(f), " B5 = ", b, " N_rig = ", nrig(29, b, 0), " N30 = ", nrig(29, b, 30),
        " 29^8 > 2B5? ", 29^8 > 2*b);
  for(N = nrig(29, b, 0), nrig(29, b, 0) + 1, v = both(f, 29, N);
    print("  N = ", N, ": T5+ = ", v[1], " in? ", abs(v[1]) <= b, "; T5- = ", v[2], " in? ", abs(v[2]) <= b));
  v = both(f, 29, nrig(29, b, 30)); print("  N30: T5+ = ", v[1], " (", #Str(abs(v[1])), " digits); T5- = ", v[2]));
}
quit;
