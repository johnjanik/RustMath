\\ F-9 design: reference values for the unit tests of irreducible.rs and for the E6 curve.
default(parisize, "100M");
read("irred.gp");
\\ the number of monic irreducible polynomials of degree d over F_p (Gauss's formula), and a
\\ direct count by polisirreducible
numirr(p, d) = sumdiv(d, e, moebius(d / e) * p^e) / d;
direct(p, d) = { my(n = 0); forvec(v = vector(d, i, [0, p - 1]), n += polisirreducible(Mod(1, p) * ('t^d + Pol(v, 't)))); n };
print("monic irreducibles over F_5 of degree 1..4 (Gauss): ", vector(4, d, numirr(5, d)), ", direct: ", vector(4, d, direct(5, d)));
print("over F_13 of degree 2: ", numirr(13, 2), ", direct ", direct(13, 2));
\\ E6: y^3 - x^4 of bidegree (4,3) at 7, 11, 13 (tests/biprojective_small_curves.rs::unclassified_e6_points)
{
  foreach ([7, 11, 13], p,
    my(F = 'y^3 - 'x^4);
    print("E6 at ", p, ": criterion ", crit(F, 4, 3, p), "; x-fibre over a degree-2 point: ", fibx_deg(F, 4, 3, p, 2),
          "; PARI factor over F_p: ", #factor(Mod(1, p) * F)~, " factor(s)"));
}
\\ the dy = 0 form x^2 + 11 at 13: two conjugate vertical lines
print("x^2 + 11 at 13: factor ", factor(Mod(1, 13) * ('x^2 + 11)));
