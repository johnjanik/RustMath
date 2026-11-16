# Reference checks for the WP-P design (GAP 4.12.1, CTblLib, TomLib not needed).
t := CharacterTable("M23");;
aut := AutomorphismsOfTable(t);;
Print("table automorphisms: order ", Size(aut), ", generators ", GeneratorsOfGroup(aut), "\n");
G := MathieuGroup(23);;
mx := MaximalSubgroupClassReps(G);;
Print("maximal subgroups (order, transitive on 23 points):\n");
for H in mx do
  Print("  ", Size(H), " ", IsTransitive(H, [1..23]), " ", StructureDescription(H : short), "\n");
od;
# The published generators of HJLPPZ (1-based cycles, right action as in GAP).
g1 := (1,11)(2,23)(3,8)(4,16)(5,21)(7,20)(15,19)(18,22);;
g2 := (1,2,3,4,10,11,12,7,19,18,8,6,9,16,17,21,22,5,14,20,13,15,23);;
g3 := (1,2,11,10,16,9,6,3,23,19,20,14,21,17,4,8,22,5,18,15,13,7,12);;
M := Group(g1, g2);;
Print("|<g1,g2>| = ", Size(M), ", g1*g2*g3 = ", g1*g2*g3, "\n");
