# Conjugacy classes of Sp(2n,2) for 2n = 2,4,6,8: representatives as 0/1 matrices, class sizes, and the form.
for n in [2,4,6,8] do
  G := Sp(n, 2);
  form := InvariantBilinearForm(G).matrix;
  cc := ConjugacyClasses(G);
  out := OutputTextFile(Concatenation("sp", String(n), "_classes.txt"), false);
  SetPrintFormattingStatus(out, false);
  PrintTo(out, "");
  AppendTo(out, "FORM ", List(form, r -> List(r, x -> IntFFE(x))), "\n");
  for c in cc do
    m := Representative(c);
    AppendTo(out, "CLASS ", Size(c), " ", Order(m), " ", List(m, r -> List(r, x -> IntFFE(x))), "\n");
  od;
  CloseStream(out);
  Print(n, " ", Size(G), " ", Length(cc), " classes, sum ", Sum(List(cc, Size)), "\n");
od;
QUIT;
