default(parisizemax, 4000000000);
read("steiner.gp");
read("cc0_poly.gp");
a = pollead(F0); n = poldegree(F0);
f = a^(n-1) * subst(F0, x, x/a);   \\ monic: x -> x/a
if(pollead(f) != 1 || denominator(content(f)) != 1, error("not monic integral"));
print("max digits ", vecmax(apply(c -> #Str(abs(c)), Vec(f))));
Hp = heptads(1); Hm = heptads(-1);
p = 29;
N = 270; q = p^N; r = frob_roots(f, p, N);
print("T5+ = ", t5(r, Hp, q)); print("T5- = ", t5(r, Hm, q));
N = 400; q = p^N; r = frob_roots(f, p, N);
print("J5- = ", jk(r, Hm, 5, q)); print("J7- = ", jk(r, Hm, 7, q));
print("J5+ = ", jk(r, Hp, 5, q));
quit;
