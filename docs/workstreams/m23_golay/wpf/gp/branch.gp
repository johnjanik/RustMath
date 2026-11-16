\\ branch rationality of the node / tacnode at t0 for y^2 = D(T) = (T - t0)^r h(T): rational iff h(t0) is a square
br(p, b, c, t0, r) = { my(D = (b^2 - 4*c)*Mod(1,p), h = D / ('T - t0)^r); [t0, r, issquare(subst(h, 'T, t0))]; }
{
p = 11; b = T^4 + 3*T^3 + T + 5;
print("B ", br(p, b, 6*T^4 + 2*T^3 + 9*T^2, 2, 2));
print("C ", br(p, b, 6*T^4 + 2*T^3 + 9*T^2 + 5*T + 7, 4, 2));
print("C2 ", br(p, b, 6*T^4 + 2*T^3 + 9*T^2 + 5*T + 8, 5, 2));
print("E ", br(p, b, 6*T^4 + 6*T^3 + 4*T^2 + 9*T, 1, 4));
print("F ", br(p, b, 6*T^4 + 7*T^2 + 5*T + 10, 6, 4));
p = 13; b = 2*T^4 + T^3 + 7*T + 1;
print("G ", br(p, b, 3*T^4 + 5*T^3 + T^2 + 10*T + 10, 0, 2));
print("H ", br(p, b, 3*T^4 + 5*T^3 + T^2 + 5*T, 1, 2));
print("I ", br(p, b, 3*T^4 + 12*T^2 + 7, 2, 4));
print("I2 ", br(p, b, 3*T^4 + 8*T^3 + 12*T^2 + 10*T, 3, 4));
}
