m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("steiner.gp");
read(Str(m23dir, "/polynomial_f2.gp")); f = f2;
p = 29; N = 30; q = p^N;
Hp = heptads(1); Hm = heptads(-1); print(#Hp, " ", #Hm);
r = frob_roots(f, p, N);
print("T5+ = ", t5(r, Hp, q)); print("T5- = ", t5(r, Hm, q));
print("J5- = ", jk(r, Hm, 5, q)); print("J7- = ", jk(r, Hm, 7, q));
quit;
