m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("steiner.gp");
read(Str(m23dir, "/polynomial_f1.gp")); read(Str(m23dir, "/polynomial_f2.gp"));
Hp = heptads(1); Hm = heptads(-1);
one(f, p, N) = { my(q = p^N, r = frob_roots(f, p, N));
  print("  p=", p, " N=", N, " T5+=", t5(r, Hp, q), " T5-=", t5(r, Hm, q));
  print("     J5+=", jk(r, Hp, 5, q), " J5-=", jk(r, Hm, 5, q), " J7+=", jk(r, Hp, 7, q), " J7-=", jk(r, Hm, 7, q)); }
print("f1 at 73"); one(f1, 73, 16);
print("f2 at 223"); one(f2, 223, 12);
print("f2 at 137 (sign +)"); one(f2, 137, 14);
quit;
