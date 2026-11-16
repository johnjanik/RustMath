m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("steiner.gp");
read(Str(m23dir, "/polynomial_f1.gp")); f = f1;
p = 29; N = 31; q = p^N;
Hp = heptads(1); Hm = heptads(-1);
r = frob_roots(f, p, N);
print("T5+ = ", t5(r, Hp, q)); print("T5- = ", t5(r, Hm, q));
print("J5- = ", jk(r, Hm, 5, q)); print("J7- = ", jk(r, Hm, 7, q));
quit;
