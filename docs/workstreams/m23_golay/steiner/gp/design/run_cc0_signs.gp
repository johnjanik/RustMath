default(parisizemax, 4000000000);
read("steiner.gp"); read("cc0_poly.gp");
a = pollead(F0); n = poldegree(F0); f = a^(n-1) * subst(F0, x, x/a);
T5 = eval(readstr("cc0_t5.txt")[1]);
Hp = heptads(1); Hm = heptads(-1);
sgn(p) = { my(r = frob_roots(f, p, 1), u = t5(r, Hp, p) % p, v = t5(r, Hm, p) % p, t = T5 % p);
  if(u == t && v != t, 1, if(v == t && u != t, -1, 0)); }
L = List(); forprime(p = 3, 1000, if(#L < 12 && polisirreducible(Mod(1,p)*f), listput(L, [p, sgn(p)]))); print(Vec(L));
quit;
