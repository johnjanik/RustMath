\\ Reference values for rustmath-finitefields/tests/galois_ring_frobenius.rs (PARI/GP 2.15.4).
print("x^3-x-1 mod 5: ", factormod(x^3-x-1, 5));
print("x^2+1 mod 2: ", factormod(x^2+1, 2));
P = nextprime(3037000500); while(P % 4 != 3, P = nextprime(P+1)); print("large prime 3 mod 4: ", P);
print("x^2+1 irreducible mod P: ", polisirreducible(Mod(1,P)*(x^2+1)));
print("x^3-x-1 mod P: ", factormod(x^3-x-1, P));
Q = nextprime(3037000500); while(polisirreducible(Mod(1,Q)*(x^3-x-1))==0, Q = nextprime(Q+1)); print("first prime > 3037000500 with x^3-x-1 inert: ", Q);
\\ the Frobenius image of t for x^3-x-1 at Q^2: root of f congruent to t^Q mod Q
T = t^3-t-1; N = 2; q = Q^N;
y = lift(lift(Mod(Mod(1,Q)*t, Mod(1,Q)*T)^Q));
\\ Newton in Mod(Mod(.,q),T)
Y = Mod(Mod(1,q)*y, Mod(1,q)*T); for(k=1,3, Y = Y - subst(T,t,Y)/subst(T',t,Y));
print("phi(t) for x^3-x-1 mod Q^2: ", Vecrev(apply(c->lift(c), lift(Y)), 3));
\\ x^5 + 2 at 3 (irreducible), and phi(t) mod 3^7
T = t^5+2; q = 3^7; y = lift(lift(Mod(Mod(1,3)*t, Mod(1,3)*T)^3));
Y = Mod(Mod(1,q)*y, Mod(1,q)*T); for(k=1,4, Y = Y - subst(T,t,Y)/subst(T',t,Y));
print("phi(t) for x^5+2 mod 3^7: ", Vecrev(apply(c->lift(c), lift(Y)), 5));
print("inert primes of x^23-x-1 below 400: ", select(p->polisirreducible(Mod(1,p)*(x^23-x-1)), primes(78)));
print("polsym(x^3+x+1, 2): ", polsym(x^3+x+1, 2));
