\\ Genus-1 examples for the unit test degenerate_genus_0_and_1 (design C57.md 9.1):
\\ #E(F_p), a_p, P(-1) = p + 1 + a, and the factorization of the cubic (E[2] over F_p).
show(a4, a6, p) = { my(E = ellinit([0, 0, 0, a4, a6], p), n = ellcard(E), a = p + 1 - n);
  print("y^2 = x^3 + ", a4, " x + ", a6, " over F_", p, ": #E = ", n, ", a = ", a, ", P(-1) = ", p + 1 + a,
        ", cubic mod p: ", factormod(x^3 + a4*x + a6, p)); }
show(1, 1, 5); show(-1, 0, 5); show(3, 5, 7); show(0, 1, 5);
quit;
