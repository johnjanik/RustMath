# The ATLAS character table of M23 from GAP's CTblLib, as JSON (RustMath WP-P).
# Source: CharacterTable("M23") of CTblLib (the ATLAS of Finite Groups, 1985, M23 p. 71).
# Each character value is written as [x, y, d]: the value x + y*b_d, where
# b_d = (-1 + Sqrt(-d))/2 is the ATLAS irrationality (d in 7, 11, 15, 23),
# and d = 1 with y = 0 for a rational value. The script checks every value
# and stops with an error if the encoding fails.
# Run from the workspace root (GAP 4.12.1, CTblLib 1.3.7 give the committed file):
#   gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null > rustmath-groups/data/m23/m23_atlas_table.json
SetPrintFormattingStatus("*stdout*", false);
t := CharacterTable("M23");;
cn := ClassNames(t);;
bval := d -> (-1 + Sqrt(-d)) / 2;;
enc := function(v)
  local d, b, y, x;
  if IsRat(v) then
    if not IsInt(v) then Error("non-integral value"); fi;
    return [v, 0, 1];
  fi;
  for d in [7, 11, 15, 23] do
    b := bval(d);
    y := (v - ComplexConjugate(v)) / (b - ComplexConjugate(b));
    x := v - y * b;
    if IsRat(y) and IsRat(x) then
      if not (IsInt(x) and IsInt(y)) then Error("non-integral coordinates"); fi;
      if x + y * b <> v then Error("reconstruction failed"); fi;
      return [x, y, d];
    fi;
  od;
  Error("value outside the four quadratic fields");
end;;
js := l -> Concatenation("[", JoinStringsWithSeparator(List(l, String), ","), "]");;
jss := l -> Concatenation("[", JoinStringsWithSeparator(List(l, s -> Concatenation("\"", s, "\"")), ","), "]");;
pm := List([2, 3, 5, 7, 11, 23, -1], k -> js(PowerMap(t, k)));;
Print("{\"group\":\"M23\",\"source\":\"GAP ", GAPInfo.Version, ", CTblLib ",
      InstalledPackageVersion("ctbllib"), ", CharacterTable(\\\"", Identifier(t), "\\\")\",\n");
Print("\"value_encoding\":\"[x, y, d] is x + y*b_d with b_d = (-1+sqrt(-d))/2; d = 1 for a rational value\",\n");
Print("\"class_names\":", jss(cn), ",\n");
Print("\"orders\":", js(OrdersClassRepresentatives(t)), ",\n");
Print("\"centralizer_orders\":", js(SizesCentralizers(t)), ",\n");
Print("\"class_sizes\":", js(SizesConjugacyClasses(t)), ",\n");
Print("\"power_maps\":{\"2\":", pm[1], ",\"3\":", pm[2], ",\"5\":", pm[3], ",\"7\":", pm[4],
      ",\"11\":", pm[5], ",\"23\":", pm[6], ",\"-1\":", pm[7], "},\n");
Print("\"permutation_character_23\":", js(PermChars(t, 23)[1]), ",\n");
Print("\"irreducibles\":[\n",
      JoinStringsWithSeparator(List(Irr(t), x -> Concatenation("[", JoinStringsWithSeparator(List(x, v -> js(enc(v))), ","), "]")), ",\n"),
      "\n]}\n");
