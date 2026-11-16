# Review 1 (independent): group, maximal subgroups of the published group, class fusion to the
# ATLAS table, random elements with their ATLAS class, 57 structure constants via
# ClassMultiplicationCoefficient.
SetPrintFormattingStatus("*stdout*", false);
g1 := (1,11)(2,23)(3,8)(4,16)(5,21)(7,20)(15,19)(18,22);;
g2 := (1,2,3,4,10,11,12,7,19,18,8,6,9,16,17,21,22,5,14,20,13,15,23);;
g3 := (1,2,11,10,16,9,6,3,23,19,20,14,21,17,4,8,22,5,18,15,13,7,12);;
G := Group(g1, g2);;
Print("#order ", Size(G), " ", IsomorphismTypeInfoFiniteSimpleGroup(G).name, " prod ", g1*g2*g3, "\n");
mx := MaximalSubgroupClassReps(G);;
Print("#maxes ", List(mx, H -> [Size(H), IsTransitive(H, [1..23])]), "\n");
tM := CharacterTable("M23");;
ccl := ConjugacyClasses(G);;
tG := CharacterTable(G);;
SetConjugacyClasses(tG, ccl);
tr := TransformingPermutationsCharacterTables(tG, tM);;
# columns: tG column i corresponds to tM column i^tr.columns
Print("#autgroup ", Size(tr.group), "\n");
names := ClassNames(tM);;
cls := List(ccl, c -> names[(Position(ccl, c))^tr.columns]);;
Print("#gapclass_of_g2 ", cls[PositionProperty(ccl, c -> g2 in c)], " g3 ", cls[PositionProperty(ccl, c -> g3 in c)], "\n");
for i in [1..Length(ccl)] do
  Print("REP ", cls[i], " ", ListPerm(Representative(ccl[i]), 23) - 1, "\n");
od;
# random elements, labelled by GAP's ATLAS name under the fusion tr
for k in [1..3000] do
  x := PseudoRandom(G);
  Print("RND ", cls[PositionProperty(ccl, c -> x in c)], " ", ListPerm(x, 23) - 1, "\n");
od;
# structure constants n(i,j,k) = #{x1 x2 x3 = 1}/|G| = ClassMultiplicationCoefficient(i,j,k^-1)/|C(k)|
inv := PowerMap(tM, -1);;
for i in [2..17] do for j in [i..17] do for k in [j..17] do
  Print("N ", names[i], " ", names[j], " ", names[k], " ",
        ClassMultiplicationCoefficient(tM, i, j, inv[k]) / SizesCentralizers(tM)[k], "\n");
od; od; od;
