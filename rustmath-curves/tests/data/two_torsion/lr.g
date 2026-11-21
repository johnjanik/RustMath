# Littlewood-Richardson coefficients c^lam_{mu,nu} for |mu| = a, |nu| = b, a, b >= 1, a + b <= 8,
# from the character tables of the symmetric groups (Frobenius reciprocity):
# c = sum over classes alpha of S_a, beta of S_b of
#     chi^lam(alpha u beta) chi^mu(alpha) chi^nu(beta) / (z_alpha z_beta).
# Output: the nonzero values, one per line: lam mu nu c. Design C57.md Section 3.3.
OnBreak := function() Print("BREAK\n"); FORCE_QUIT_GAP(1); end;;
chi := function(t, lam, rho)   # value of the character lam at the class rho
  local ci, ri;
  ci := Position(List(CharacterParameters(t), x -> x[2]), lam);
  ri := Position(List(ClassParameters(t), x -> x[2]), rho);
  return Irr(t)[ci][ri];
end;;
for n in [2..8] do
  tn := CharacterTable("Symmetric", n);;
  for a in [1..n-1] do
    b := n - a;
    ta := CharacterTable("Symmetric", a);; tb := CharacterTable("Symmetric", b);;
    for lam in Partitions(n) do
      for mu in Partitions(a) do
        for nu in Partitions(b) do
          c := 0;
          for i in [1..NrConjugacyClasses(ta)] do
            for j in [1..NrConjugacyClasses(tb)] do
              al := ClassParameters(ta)[i][2]; be := ClassParameters(tb)[j][2];
              ab := Reversed(SortedList(Concatenation(al, be)));
              c := c + chi(tn, lam, ab) * chi(ta, mu, al) * chi(tb, nu, be)
                       / (SizesCentralizers(ta)[i] * SizesCentralizers(tb)[j]);
            od;
          od;
          if c <> 0 then Print(lam, " ", mu, " ", nu, " ", c, "\n"); fi;
        od;
      od;
    od;
  od;
od;
QUIT;
