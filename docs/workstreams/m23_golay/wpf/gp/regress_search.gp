\\ F-9 design: search for the regression curves (run once; regress.gp then checks the chosen ones).
\\ (i)  E1 E2, two smooth (2,2) curves over F_p meeting transversally in 8 points;
\\ (ii) E E^sigma, E a smooth (2,2) curve over F_(p^2) not defined over F_p, meeting its conjugate
\\      transversally in 8 points, with an F_p-rational x-fibre of E E^sigma irreducible over F_p.
default(parisize, "200M");
read("irred.gp");
p = 13;
\\ a (2,2) form from 9 coefficients c[1 + i + 3j] of x^i y^j
form22(c) = sum(i = 0, 2, sum(j = 0, 2, c[1 + i + 3*j] * 'x^i * 'y^j));
\\ smooth of bidegree (2,2): the x-discriminant is a squarefree binary quartic of degree 4 in y,
\\ and the x-degree and y-degree are 2
smooth22(E) = {
  my(D = polcoef(E, 1, 'x)^2 - 4 * polcoef(E, 2, 'x) * polcoef(E, 0, 'x));
  poldegree(E, 'x) == 2 && poldegree(E, 'y) == 2 && poldegree(D, 'y) == 4 && poldegree(gcd(D, deriv(D, 'y))) == 0;
}
\\ transversal in 8 points with distinct y: Res_x squarefree of degree 8
transversal(E1, E2) = my(R = polresultant(E1, E2, 'x)); poldegree(R, 'y) == 8 && poldegree(gcd(R, deriv(R, 'y))) == 0;

setrand(1);
{
  found = 0;
  for (t = 1, 2000,
    my(c1 = vector(9, k, random(p)), c2 = vector(9, k, random(p)));
    c1[9] = 1; c2[9] = 1;
    my(E1 = Mod(1, p) * form22(c1), E2 = Mod(1, p) * form22(c2));
    if (smooth22(E1) && smooth22(E2) && transversal(E1, E2),
      print("(i) p = ", p, " E1 coeffs (x^i y^j, i fastest) ", c1, " E2 ", c2);
      found++; if (found == 3, break)));
}
\\ (ii) F_(p^2) = F_p(w), w^2 = n, n the least nonsquare
n = 2; while (kronecker(n, p) != -1, n++);
print("(ii) nonsquare n = ", n);
w = ffgen(Mod(1, p) * ('w^2 - n), 'w);
{
  found = 0;
  for (t = 1, 4000,
    my(c0 = vector(9, k, random(p)), c1 = vector(9, k, random(p)));
    c0[9] = 1; c1[9] = 0;
    if (c1 == vector(9), next);
    my(E = form22(c0) + w * form22(c1), Es = form22(c0) - w * form22(c1));
    my(F = lift(Mod(1, p) * (form22(c0)^2 - n * form22(c1)^2)));
    if (!smooth22(E) || !transversal(E, Es), next);
    my(r = crit(F, 4, 4, p));
    if (r[1] == "nosmooth",
      print("(ii) p = ", p, " E0 ", c0, " E1 ", c1, " criterion ", r);
      found++; if (found == 3, break)));
}
