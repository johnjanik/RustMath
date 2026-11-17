read("steiner.gp"); f = polsubcyclo(47, 23); Hp = heptads(1); Hm = heptads(-1);
for(i = 1, 2, my(p = [3, 5][i], N = 20, q = p^N, r = frob_roots(f, p, N)); print("p = ", p, " T5+ = ", t5(r, Hp, q), " T5- = ", t5(r, Hm, q)));
quit;
