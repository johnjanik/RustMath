\\ Reference fixtures for rustmath-integers/tests/ch18_residues.rs
\\ (MAGMA Handbook ch. 18 §18.7 KroneckerSymbol, §18.12.1 Modexp/Modinv/Modsqrt, Binomial).
\\ Generated with PARI/GP 2.15.4. Run from this directory:  gp -q < gen_csv.gp
\\ random() streams can differ between PARI versions; the committed CSVs are the 2.15.4 output.
\\ Columns:
\\   kron_grid.csv, kron_big.csv : a, n, kronecker(a, n)
\\   modpow_big.csv              : b, e, m, lift(Mod(b, m)^e)   ("E" = impossible inverse)
\\   modinv_grid.csv             : a, m, lift(Mod(a, m)^-1)     ("N" = no inverse)
\\   sqrt_big.csv                : a, p, issquare(Mod(a, p))
\\   binom_grid.csv              : n, k, p, binomial(n, k) % p
\\ kronecker grid
f=fileopen("kron_grid.csv","w"); for(a=-30,30, for(n=-30,30, filewrite(f, Str(a, ",", n, ",", kronecker(a,n))))); fileclose(f);
\\ kronecker large / random
setrand(12345); f=fileopen("kron_big.csv","w");
for(i=1,300, a=random(2^300)-2^299; n=random(2^200)-2^199; if(i%3==0, n=n*2^random(40)); if(i%5==0, n=nextprime(abs(n))*(if(random(2),1,-1))); filewrite(f, Str(a, ",", n, ",", kronecker(a,n)))); fileclose(f);
\\ modpow large / random (negative exponents where invertible, "E" when not)
f=fileopen("modpow_big.csv","w");
for(i=1,200, b=random(2^400)-2^399; e=random(2^300); if(i%4==0,e=-e); m=random(2^256)+2; if(i%2==0, m=m*2^random(20)); r=iferr(lift(Mod(b,m)^e), E, "E"); filewrite(f, Str(b, ",", e, ",", m, ",", r))); fileclose(f);
\\ sqrt mod p for large primes: a, p, issquare flag
f=fileopen("sqrt_big.csv","w");
P=[2^127-1, 10000000000041470962696193, 1000000000000000000000000000469, nextprime(2^255), nextprime(10^40)];
for(j=1,#P, p=P[j]; for(i=1,40, a=random(2^300)-2^299; filewrite(f, Str(a, ",", p, ",", issquare(Mod(a,p)))))); fileclose(f);
\\ binomial mod p
f=fileopen("binom_grid.csv","w");
forprime(p=2,13, for(n=-25,25, for(k=-3,25, filewrite(f, Str(n, ",", k, ",", p, ",", binomial(n,k)%p))))); fileclose(f);
\\ mod inverse
f=fileopen("modinv_grid.csv","w");
for(m=1,30, for(a=-40,40, r=iferr(lift(Mod(a,m)^-1), E, "N"); filewrite(f, Str(a, ",", m, ",", r)))); fileclose(f);
quit;
