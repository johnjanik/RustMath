\\ #Jac(F_p^k) and the twisted orders for the rows quoted in the tests of design C57.md Section 9.
read("rows.gp");
{
for (i = 1, #ROWS,
  my(r = ROWS[i], p = r[2], a = r[4..7], P);
  if (!((r[1] == 0 && setsearch([17, 59, 107, 109], p)) || (r[1] == 1 && (p == 41 || (p == 89 && r[3] == 69)))), next);
  P = x^8 + a[1]*x^7 + a[2]*x^6 + a[3]*x^5 + a[4]*x^4 + p*a[3]*x^3 + p^2*a[2]*x^2 + p^3*a[1]*x + p^4;
  print(if (r[1], "L ", "X "), p, " ", r[3], ": P(1) = ", subst(P, x, 1), ", P(-1) = ", subst(P, x, -1),
        ", Res(P, x^4+1) = ", polresultant(P, x^4 + 1), ", Res(P, x^5-1) = ", polresultant(P, x^5 - 1)));
}
quit;
