# Review check (not part of RustMath): Nielsen class counts of sample rows by a different
# method. Classes are GAP's conjugacy classes of RustMath's representatives (reps, ATLAS order);
# generation by Size(Group(a, b)) = |G|; Nielsen classes as Centralizer(G, b)-orbits; real
# classes by RepresentativeAction on tuples:
#   conjugate pair (R, D, D^-1): (a, b) -> (a^-1, a b)        [(g1,g2,g3) -> (g1^-1,g3^-1,g2^-1)]
#   rational:                    (a, b) -> (a^-1, b^-1)
Read("reps.g");
g1 := (1,11)(2,23)(3,8)(4,16)(5,21)(7,20)(15,19)(18,22);;
g2 := (1,2,3,4,10,11,12,7,19,18,8,6,9,16,17,21,22,5,14,20,13,15,23);;
G := Group(g1, g2);; Size(G);;
N := ["1A","2A","3A","4A","5A","6A","7A","7B","8A","11A","11B","14A","14B","15A","15B","23A","23B"];;
idx := s -> Position(N, s);;
inclass := function(x, c)
  if CycleStructurePerm(x) <> CycleStructurePerm(reps[c]) then return false; fi;
  return IsConjugate(G, x, reps[c]);
end;;
row := function(kind, ia, ib, ic)
  # a over class ia, b = reps[ib] fixed, survivors: a b in class ic
  local b, C, A, surv, gen, orbs, real, o, a, target, t0;
  t0 := Runtime();
  b := reps[ib]; C := Centralizer(G, b);
  A := AsList(ConjugacyClass(G, reps[ia]));
  surv := Filtered(A, a -> inclass(a * b, ic));
  gen := Filtered(surv, a -> Size(Group(a, b)) = Size(G));
  orbs := Orbits(C, gen, OnPoints);
  if not ForAll(orbs, o -> Length(o) = Size(C)) then Error("not free"); fi;
  real := 0;
  for o in orbs do
    a := o[1];
    if kind = "conjugate_pair" then target := [a^-1, a * b]; else target := [a^-1, b^-1]; fi;
    if RepresentativeAction(G, [a, b], target, OnTuples) <> fail then real := real + 1; fi;
  od;
  Print("ROW ", kind, " a=", N[ia], " b=", N[ib], " ab=", N[ic], " |C(b)|=", Size(C),
        " survivors=", Length(surv), " generating=", Length(gen), " size=", Length(orbs),
        " real=", real, " (", Runtime() - t0, " ms)\n");
end;;
# conjugate pairs (R, D, D^-1): a in R, b = rep(D), a b in D
for v in [["2A","23A"], ["2A","14A"], ["2A","11A"], ["2A","15A"], ["2A","7A"], ["3A","23A"], ["3A","11A"]] do
  row("conjugate_pair", idx(v[1]), idx(v[2]), idx(v[2]));
od;
# rational (X, Y, Z): roles chosen differently from RustMath's: a in X, b = rep(Y), a b in Z
for v in [["2A","8A","8A"], ["3A","5A","6A"], ["3A","4A","8A"], ["4A","5A","4A"], ["3A","6A","6A"]] do
  row("rational", idx(v[1]), idx(v[2]), idx(v[3]));
od;
QUIT;
