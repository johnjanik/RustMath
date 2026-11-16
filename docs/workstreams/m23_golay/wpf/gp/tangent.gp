\\ F = V^2 + b(T) V + c(T), b = T^3 + 2 T^4, c = 3 T^4: A3 at (0,0) whose tangent line is the fibre V = 0
p = 11; b = T^3 + 2*T^4; c = 3*T^4;
D = b^2 - 4*c; print("D = ", factor(D));
h = D / T^4; print("normalization y^2 = ", h, "  disc mod p = ", poldisc(h) % p, "  branch class h(0) square mod p: ", issquare(Mod(subst(h,T,0),p)));
e = ellfromeqn(y^2 - h); E = ellinit(e*Mod(1,p)); print("a_p = ", ellap(ellinit(e), p), "  N1 of normalization = ", p + 1 - ellap(ellinit(e), p));
