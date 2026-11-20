# The 14 classes of GL_4(2) = A_8 (design C57.md Section 3.4). One line per class:
#   charpoly (ascending 0/1 coefficients) | order | class size | dim ker (g - 1)^j, j = 1..4 |
#   cycle type on the 15 points of PG(3,2) | on the 15 hyperplanes | on 8 letters
# The 8-letter type is taken through an isomorphism GL(4,2) -> A_8 found by GAP; every
# automorphism of A_8 is induced by S_8, so the cycle type does not depend on the choice.
# Run: gap -q gl42_classes.g > gl42_classes.txt
OnBreak := function() Print("BREAK\n"); FORCE_QUIT_GAP(1); end;;
G := GL(4, 2);;
A := AlternatingGroup(8);;
iso := IsomorphismGroups(G, A);;
pts := NormedRowVectors(GF(2)^4);;
g := One(G);;
ctype := function(perm, n)
  local c, s, i;
  c := Collected(CycleLengths(perm, [1..n]));
  s := "";
  for i in [1..Length(c)] do
    if i > 1 then Append(s, " "); fi;
    Append(s, String(c[i][1]));
    if c[i][2] > 1 then Append(s, "^"); Append(s, String(c[i][2])); fi;
  od;
  return s;
end;;
for cl in ConjugacyClasses(G) do
  g := Representative(cl);
  onpts := Permutation(g, pts, OnRight);
  onhyp := Permutation(TransposedMat(g)^-1, pts, OnRight);
  kers := List([1..4], j -> 4 - RankMat((g - One(g))^j));
  cp := List(CoefficientsOfUnivariatePolynomial(CharacteristicPolynomial(g)), IntFFE);
  Print(JoinStringsWithSeparator(List(cp, String), ","), " | ", Order(g), " | ", Size(cl), " | ",
        JoinStringsWithSeparator(List(kers, String), ","), " | ", ctype(onpts, 15), " | ",
        ctype(onhyp, 15), " | ", ctype(Image(iso, g), 8), "\n");
od;
QUIT;
