\\ Reference values for the small-curve tests of WP-F (PARI/GP hyperellcharpoly, ellap).
\\ Curves a(T) V^2 + b(T) V + c(T) = 0 of bidegree (4,2) in P^1 x P^1 (fibre variable T, degree 4).
\\ Its singular points are those of y^2 = D(T), D = b^2 - 4ac (y = 2aV + b where a != 0).
\\ Constrained coefficients of c are solved mod p so that D has a root of order r at t0.
mk(p, t0, r, b, c) = {
  \\ keep the terms of c of degree >= r, choose the low part so that b^2 - 4c = 0 mod (T - t0)^r
  my(ch = sum(i = r, 4, polcoeff(c, i, 'T)*'T^i)*Mod(1,p), b2 = b*Mod(1,p));
  my(R = lift((b2^2/4 - ch) % ('T - t0)^r));
  lift(ch) + R;
}
show(name, p, a, b, c) = {
  my(D = lift((b^2 - 4*a*c)*Mod(1,p)), fa = factormod(D, p), sq = 1, rest = 1);
  for(i=1, #fa[,1], my(e = fa[i,2], f = lift(fa[i,1])); rest *= f^(e%2));
  rest = rest * pollead(D);
  my(cp = hyperellcharpoly(rest*Mod(1,p)));
  print(name, ": p=", p, " a=", a, " b=", b, " c=", lift(c*Mod(1,p)));
  print("    D=", D, "  factor degrees/mults ", apply(x->poldegree(lift(x)), fa[,1]~), fa[,2]~);
  print("    normalization y^2 = ", lift(rest), " ; charpoly = ", cp);
}
{
p = 11;
show("A smooth genus 3", p, 2*T^4 + T + 3, T^3 + 5*T^2 + 1, T^4 + 4*T^2 + 7);
b = T^4 + 3*T^3 + T + 5; c0 = 6*T^4 + 2*T^3 + 9*T^2;
show("B node t0=2", p, 1, b, mk(p, 2, 2, b, c0));
show("C node t0=4", p, 1, b, mk(p, 4, 2, b, c0));
show("C2 node t0=5", p, 1, b, mk(p, 5, 2, b, c0));
show("D cusp t0=3", p, 1, b, mk(p, 3, 3, b, c0));
show("E tacnode t0=1", p, 1, b, mk(p, 1, 4, b, c0));
show("F tacnode t0=6", p, 1, b, mk(p, 6, 4, b, c0));
show("F2 tacnode t0=7", p, 1, b, mk(p, 7, 4, b, c0));
p = 13;
b = 2*T^4 + T^3 + 7*T + 1; c0 = 3*T^4 + 5*T^3 + T^2;
show("G node t0=0", p, 1, b, mk(p, 0, 2, b, c0));
show("H node t0=1", p, 1, b, mk(p, 1, 2, b, c0));
show("I tacnode t0=2", p, 1, b, mk(p, 2, 4, b, c0));
show("I2 tacnode t0=3", p, 1, b, mk(p, 3, 4, b, c0));
\\ elliptic curves in P^1 x P^1: V^2 = T^3 + a T + b  (bidegree (3,2); A2 cusp at (inf, inf))
E1 = ellinit([0,0,0,3,5]); print("E1 V^2=T^3+3T+5: ap(7)=", ellap(E1,7), " ap(11)=", ellap(E1,11), " ap(13)=", ellap(E1,13));
\\ plane cubic V^3 = g(T) as a (3,3) form: ordinary triple point at (inf, inf)
e = ellfromeqn(y^3 - (x^3 + 2*x + 3)); E2 = ellinit(e); print("E2 V^3=T^3+2T+3: model ", e, " ap(7)=", ellap(E2,7), " ap(13)=", ellap(E2,13), " ap(19)=", ellap(E2,19), " ap(11)=", ellap(E2, 11));
}
