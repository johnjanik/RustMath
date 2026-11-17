m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("steiner.gp"); read("signs_data.gp");
read(Str(m23dir, "/polynomial_f1.gp")); read(Str(m23dir, "/polynomial_f2.gp"));
Hp = heptads(1); Hm = heptads(-1);
sgn(f, p, T5) = { my(r = frob_roots(f, p, 1), a = t5(r, Hp, p) % p, b = t5(r, Hm, p) % p, t = T5 % p);
  if(a == t && b != t, 1, if(b == t && a != t, -1, 0)); }
\\ all primes < 4000 at which f is irreducible mod p, with the sign
check(f, ps, ss, T5) = { my(bad = 0, cnt = 0);
  forprime(p = 3, 4000, if(polisirreducible(f * Mod(1, p)), cnt++;
     my(s = sgn(f, p, T5), k = setsearch(ps, p)); if(!k, print("  missing ", p); bad++, if(ss[k] != s, print("  differ ", p, " ", s); bad++))));
  print("  inert primes < 4000: ", cnt, ", listed: ", #ps, ", disagreements: ", bad); }
print("f1:"); check(f1, f1_primes, f1_signs, f1_T5);
print("f2:"); check(f2, f2_primes, f2_signs, f2_T5);
quit;
