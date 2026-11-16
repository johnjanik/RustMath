m23dir = getenv("M23ISGALOIS"); if(type(m23dir) != "t_STR", error("set M23ISGALOIS to a checkout of shaowuz/m23isgalois"));
read("rev.gp"); read("signs_f1.gp"); read("signs_f2.gp"); read("cc0_raw.gp");
read(Str(m23dir, "/polynomial_f2.gp")); read(Str(m23dir, "/polynomial_f1.gp"));
sgn(f, p, T) = { my(v = both(f, p, 1), w = [Mod(v[1], p) == T, Mod(v[2], p) == T]); if(w == [1, 0], 1, if(w == [0, 1], -1, 0)); }
{
foreach([[f1, f1_P, f1_S, 92747040, "f1"], [f2, f2_P, f2_S, -1059840, "f2"]], L,
  bad = 0; for(i = 1, #L[2], s = sgn(L[1], L[2][i], L[4]); if(s != L[3][i], bad++; print("  mismatch ", L[2][i], " ", s)));
  allp = [p | p <- primes([3, 4000]), polisirreducible(Mod(1, p) * L[1])];
  print(L[5], ": ", #L[2], " file primes, mismatches ", bad, "; inert primes in [3,4000]: ", #allp, " not in file: ", setminus(Set(allp), Set(L[2]))));
\\ second primes: f1 at 73, f2 at 223, full labelling and J at the rigorous precisions
foreach([[f1, 73, "f1"], [f2, 223, "f2"]], L, f = L[1]; p = L[2]; b = B5(f); N = nrig(p, b, 30); v = both(f, p, N);
  bb = fbest(f); bj5 = floor(5313 * (bb / 128)^5) + 1; bj7 = floor(253 * (bb / 128)^7) + 1;
  NJ = max(nrig(p, bj5, 0), nrig(p, bj7, 0)); r = froots(f, p, NJ); q = p^NJ;
  print(L[3], " at ", p, ": N ", N, " T5 ", v, " within ", [abs(v[1]) <= b, abs(v[2]) <= b], "; NJ ", NJ, " J5-,J7- ", [Jk(r, Hm, 5, q), Jk(r, Hm, 7, q)]));
\\ CC0 monic associate signs vs f1 rows
a = pollead(F0); g = a^22 * subst(F0, x, x/a); T5c = eval(readstr("cc0_rev_t5.txt")[1]);
cc = [p | p <- primes([3, 900]), polisirreducible(Mod(1, p) * g)]; print("CC0 inert primes < 900: ", cc);
print("CC0 signs: ", vector(#cc, i, sgn(g, cc[i], T5c)));
print("f1 signs : ", vector(#cc, i, my(j = select(q -> q == cc[i], f1_P, 1)); if(#j, f1_S[j[1]], "NA")));
}
quit;
