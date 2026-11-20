# Review 2: independent counts of sample rows in GAP.
# Survivors by IsConjugate (no Golay trace), generation by Size (no transitivity theorem),
# real classes by RepresentativeAction on the conjugated pair (no involution formula).
SetPrintFormattingStatus("*stdout*", false);
g1 := (1,11)(2,23)(3,8)(4,16)(5,21)(7,20)(15,19)(18,22);;
g2 := (1,2,3,4,10,11,12,7,19,18,8,6,9,16,17,21,22,5,14,20,13,15,23);;
g3 := (1,2,11,10,16,9,6,3,23,19,20,14,21,17,4,8,22,5,18,15,13,7,12);;
G := Group(g1, g2);;
Print("order ", Size(G), " product ", g1*g2*g3 = (), " simple ", IsomorphismTypeInfoFiniteSimpleGroup(G).name, "\n");
mx := MaximalSubgroupClassReps(G);;
Print("maximal: ", List(mx, Size), " transitive: ", List(mx, m -> IsTransitive(m, [1..23])), "\n");
ct := p -> SortedList(CycleLengths(p, [1..23]));;
# one representative per cycle type, found by a fixed random walk
Reset(GlobalMersenneTwister, 2323);;
reps := rec();;
while Length(RecNames(reps)) < 12 do
  x := PseudoRandom(G);
  k := String(ct(x));
  if not IsBound(reps.(k)) then reps.(k) := x; fi;
od;
rep := function(o)  # an element of order o (o = 1..23 as element orders of M23)
  local k;
  for k in RecNames(reps) do if Order(reps.(k)) = o then return reps.(k); fi; od;
  Error("no order ", o);
end;;
NG := Size(G);;
# conjugate pair (R, D, D^-1): b in D, a in R, survivor iff a b ~ b.
conjpair := function(ordR, b)
  local cls, surv, gen, real, a, ab, cb, t0;
  t0 := Runtime();
  cls := AsList(ConjugacyClass(G, rep(ordR)));
  cb := Size(Centralizer(G, b));
  surv := 0; gen := 0; real := 0;
  for a in cls do
    ab := a * b;
    if Order(ab) = Order(b) and IsConjugate(G, ab, b) then
      surv := surv + 1;
      if Size(Group(a, b)) = NG then
        gen := gen + 1;
        if RepresentativeAction(G, [a, b], [a^-1, a*b], OnTuples) <> fail then real := real + 1; fi;
      fi;
    fi;
  od;
  Print("conjugate_pair R=", ordR, " D=", Order(b), ": |C(b)|=", cb, " survivors=", surv,
        " n=", surv/cb, " size=", gen/cb, " real=", real/cb, " (", Runtime()-t0, " ms)\n");
end;;
# rational (A, B, C): b in B, a in A, survivor iff a b has the cycle type of C.
rational := function(ordA, ordB, ordC)
  local cls, b, surv, gen, real, a, ab, cb, t0, tc, trans_ne;
  t0 := Runtime();
  b := rep(ordB);
  cls := AsList(ConjugacyClass(G, rep(ordA)));
  cb := Size(Centralizer(G, b));
  tc := ct(rep(ordC));
  surv := 0; gen := 0; real := 0; trans_ne := 0;
  for a in cls do
    ab := a * b;
    if ct(ab) = tc then
      surv := surv + 1;
      if Size(Group(a, b)) = NG then
        gen := gen + 1;
        if RepresentativeAction(G, [a, b], [a^-1, b^-1], OnTuples) <> fail then real := real + 1; fi;
      fi;
      if (Size(Group(a, b)) = NG) <> IsTransitive(Group(a, b), [1..23]) then trans_ne := trans_ne + 1; fi;
    fi;
  od;
  Print("rational (", ordA, ",", ordB, ",", ordC, ") b in ", ordB, ": |C(b)|=", cb, " survivors=", surv,
        " n=", surv/cb, " size=", gen/cb, " real=", real/cb, " transitive<>generating: ", trans_ne,
        " (", Runtime()-t0, " ms)\n");
end;;
conjpair(2, rep(7));;
conjpair(2, g2);;
conjpair(2, rep(14));;
conjpair(2, rep(11));;
conjpair(2, rep(15));;
conjpair(3, g2);;
rational(2, 8, 8);;
rational(3, 6, 5);;
rational(3, 6, 6);;
rational(4, 5, 4);;
QUIT;
