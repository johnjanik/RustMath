default(parisizemax, 2000000000);
read("rev.gp"); read("cc0_raw.gp");
{
a = pollead(F0); n = poldegree(F0); g = a^(n-1) * subst(F0, x, x/a); 
if(pollead(g) != 1 || denominator(content(g)) != 1, error("not monic integral"));
print("max digits ", vecmax(apply(c -> #Str(abs(c)), Vec(g))));
b = B5(g); print("best = ", fbest(g)); print("B5 digits ", #Str(b), " N_rig ", nrig(29, b, 0), " N30 ", nrig(29, b, 30));
t0 = getabstime();
r = froots(g, 29, 270); q = 29^270; tp = T5(r, Hp, q); tm = T5(r, Hm, q);
print("N=270: |T5+| digits ", #Str(abs(tp)), " within ", abs(tp) <= b, "; |T5-| digits ", #Str(abs(tm)), " within ", abs(tm) <= b);
write("cc0_rev_t5.txt", tm); write("cc0_rev_t5.txt", tp);
q2 = 29^250; v = [centerlift(Mod(tp, q2)), centerlift(Mod(tm, q2))];
print("N=250: within ", [abs(v[1]) <= b, abs(v[2]) <= b], " T5- same: ", v[2] == tm);
print("time T5 ", getabstime() - t0, " ms");
bj5 = floor(5313 * (fbest(g) / 128)^5) + 1; bj7 = floor(253 * (fbest(g) / 128)^7) + 1;
N = max(nrig(29, bj5, 0), nrig(29, bj7, 0)); print("J precision ", N);
t0 = getabstime(); r = froots(g, 29, N); q = 29^N; j5 = Jk(r, Hm, 5, q); j7 = Jk(r, Hm, 7, q);
print("J5 digits ", #Str(abs(j5)), " within ", abs(j5) <= bj5, "; J7 digits ", #Str(abs(j7)), " within ", abs(j7) <= bj7, " time ", getabstime() - t0, " ms");
write("cc0_rev_t5.txt", j5); write("cc0_rev_t5.txt", j7);
}
quit;
