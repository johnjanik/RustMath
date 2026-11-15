\\ F-9 design: reference values of the regression curves (PARI/GP 2.15.4). The curves were found
\\ by regress_search.gp. Coefficient vectors list the coefficient of x^i y^j at 1 + i + 3j
\\ (bidegree (2,2)); x is the fibre variable of FibresInX.
default(parisize, "200M");
read("irred.gp");
p = 13;
form22(c) = sum(i = 0, 2, sum(j = 0, 2, c[1 + i + 3*j] * 'x^i * 'y^j));
\\ x-discriminant of a form of x-degree 2, and its squarefreeness as a binary form of degree d
xdisc(E) = polcoef(E, 1, 'x)^2 - 4 * polcoef(E, 2, 'x) * polcoef(E, 0, 'x);
ydisc(E) = polcoef(E, 1, 'y)^2 - 4 * polcoef(E, 2, 'y) * polcoef(E, 0, 'y);
sqfree(D, v, d) = poldegree(D, v) == d && poldegree(gcd(D, deriv(D, v))) == 0;
\\ #C(F_(q^k)) for a smooth curve over F_q C : w^2 = D(t) with D squarefree of even degree (genus
\\ deg/2 - 1): q + 1 - (trace of Frobenius)
hcount(D, k, q) = q^k + 1 - polsym(hyperellcharpoly(D), k)[k + 1];
\\ brute-force #V(F)(F_p) in P^1 x P^1 for a form of formal bidegree (a, b)
brute(F, a, b) = {
  my(C = charts(F, a, b), n = 0);
  for (xv = 0, p - 1, for (yv = 0, p - 1, n += (Mod(subst(subst(C[1], 'x, xv), 'y, yv), p) == 0)));
  for (yv = 0, p - 1, n += (Mod(subst(subst(C[2], 'x, 0), 'y, yv), p) == 0));
  for (xv = 0, p - 1, n += (Mod(subst(subst(C[3], 'x, xv), 'y, 0), p) == 0));
  n + (Mod(subst(subst(C[4], 'x, 0), 'y, 0), p) == 0);
}
fact(F) = my(f = factor(F)); vector(#f~, i, [poldegree(f[i, 1], 'x), poldegree(f[i, 1], 'y), f[i, 2]]);
coeffs(F, a, b) = matrix(a + 1, b + 1, i, j, lift(polcoef(polcoef(F, i - 1, 'x), j - 1, 'y)));

print("== (i) E1 E2 at p = 13: two smooth (2,2) curves meeting transversally in 8 points");
c1 = [11, 6, 5, 9, 12, 11, 11, 8, 1]; c2 = [9, 5, 1, 0, 3, 7, 4, 3, 1];
E1 = Mod(1, p) * form22(c1); E2 = Mod(1, p) * form22(c2);
D1 = xdisc(E1); D2 = xdisc(E2);
print("E1 = ", lift(E1)); print("E2 = ", lift(E2));
print("x-discriminants squarefree quartics: ", [sqfree(D1, 'y, 4), sqfree(D2, 'y, 4)]);
R = polresultant(E1, E2, 'x);
print("Res_x(E1, E2) squarefree of degree 8: ", sqfree(R, 'y, 8), "; its factor degrees ", [poldegree(f) | f <- factor(R)[, 1]~]);
F1 = E1 * E2;
print("F = E1 E2: factorization over F_13 (x-degree, y-degree, multiplicity): ", fact(F1));
print("F coefficient matrix [i][j] of x^i y^j: ", coeffs(F1, 4, 4));
n1 = [hcount(D1, k, 13) | k <- [1 .. 3]]; n2 = [hcount(D2, k, 13) | k <- [1 .. 3]];
print("#E1(F_13^k), k = 1..3: ", n1, "; #E2: ", n2, "; sum (the normalization E1 + E2): ", n1 + n2);
print("charpolys: E1 ", hyperellcharpoly(D1), ", E2 ", hyperellcharpoly(D2));
print("#V(F)(F_13) by brute force: ", brute(lift(F1), 4, 4));
print("criterion: ", crit(lift(F1), 4, 4, p));
a1 = p + 1 - (n1 + n2)[1];
print("with genus input 1: a_1 = p + 1 - N_1 = ", a1, ", |a_1| <= 2 sqrt(13)? ", abs(a1) <= 2 * sqrt(13));

print("== (ii) E E^sigma at p = 13, F_169 = F_13(w), w^2 = 2");
w = ffgen(Mod(1, p) * ('w^2 - 2), 'w);
c0 = [8, 2, 10, 12, 11, 12, 1, 2, 1]; cw = [3, 3, 4, 1, 12, 2, 2, 7, 0];
E = form22(c0) + w * form22(cw); Es = form22(c0) - w * form22(cw);
F2 = Mod(1, p) * (form22(c0)^2 - 2 * form22(cw)^2);
print("E0 = ", form22(c0), ", E1 = ", form22(cw), "; E = E0 + w E1, F = E0^2 - 2 E1^2");
print("F coefficient matrix [i][j] of x^i y^j: ", coeffs(F2, 4, 4));
DE = xdisc(E);
print("x-discriminant of E squarefree quartic over F_169: ", sqfree(DE, 'y, 4));
RE = polresultant(E, Es, 'x);
print("Res_x(E, E^sigma) squarefree of degree 8: ", sqfree(RE, 'y, 8));
print("F over F_13: ", fact(F2));
print("F over F_169: ", fact(F2 * w^0));
nE = [hcount(DE, k, 169) | k <- [1 .. 2]];
print("#E(F_169^k), k = 1, 2: ", nE, "; so the normalization has N_1 = 0, N_2 = 2 #E(F_169) = ", 2 * nE[1]);
print("#V(F)(F_13) by brute force: ", brute(lift(F2), 4, 4));
print("criterion: ", crit(lift(F2), 4, 4, p));
print("smooth F_13-point: ", smoothpoint(lift(F2), 4, 4, p));
print("x-fibres over closed points of degree 2 irreducible over F_169: ", fibx_deg(lift(F2), 4, 4, p, 2));

print("== (iii) a line component: F = (x - 3) G and F = x_0 G (the line x = infinity), G smooth of bidegree (3,2)");
\\ G = a(x) y^2 + b(x) y + c(x)
G = Mod(1, p) * ((x^3 + 2*x + 5) * y^2 + (x^2 + 3*x + 1) * y + (2*x^3 + x + 7));
DG = ydisc(G);
print("G = ", lift(G), "; y-discriminant squarefree sextic: ", sqfree(DG, 'x, 6), ", genus 2");
print("G(3, y) = ", lift(subst(G, 'x, 3)), ", discriminant ", lift(subst(DG, 'x, 3)), " (nonzero: two points on x = 3)");
F3 = (x - 3) * G;
print("F = (x - 3) G: ", fact(F3), "; coefficient matrix ", coeffs(F3, 4, 2));
print("leading coefficients in x of G (the points on x = infinity): ", lift(polcoef(G, 3, 'x)), ", discriminant ", lift(poldisc(polcoef(G, 3, 'x), 'y)));
print("F = 1 * G as a (4,2) form: coefficient matrix ", coeffs(G, 4, 2));
print("criterion (x - 3) G: ", crit(lift(F3), 4, 2, p), "; G as (4,2): ", crit(lift(G), 4, 2, p));
print("#G(F_13^k), k = 1, 2 (hyperellcharpoly of the sextic): ", [hcount(DG, k, 13) | k <- [1 .. 2]]);
Fy = Mod(1, p) * ((x^4 + 2*x + 1) * y + (x^3 + 5));
print("y = infinity: F = (x^4 + 2x + 1) y + (x^3 + 5) as a (4,2) form, coefficient matrix ", coeffs(Fy, 4, 2));
print("criterion: ", crit(lift(Fy), 4, 2, p));

print("== (iv) E1 alone: a smooth (2,2) curve, gcd(2, 2) = 2");
print("criterion: ", crit(lift(E1), 2, 2, p));
print("N_k = #E1(F_13^k), k = 1..3: ", n1, "; charpoly ", hyperellcharpoly(D1));
