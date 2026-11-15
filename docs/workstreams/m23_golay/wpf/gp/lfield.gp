\\ The sextic field L of HJLPPZ 3.7 (the polynomial is in RustMath's tests/common/m23.rs): the
\\ index of Z[s] in O_L and the discriminant, for the remark on degree-one primes.
nf = nfinit(s^6 - s^5 - 3*s^4 - 3*s^3 + s^2 + 5*s + 4);
print("index [O_L : Z[s]] = ", nf.index, ", disc(L) = ", factor(nf.disc));
print("poldisc = ", factor(poldisc(s^6 - s^5 - 3*s^4 - 3*s^3 + s^2 + 5*s + 4)));
