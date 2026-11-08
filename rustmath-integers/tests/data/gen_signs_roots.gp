\\ Reference fixtures for rustmath-integers/tests/ch18_signs_roots.rs
\\ (MAGMA Handbook ch. 18: §18.4.5 IsPower, §18.4.8 Iroot, §18.6 Gcd/Lcm/Xgcd, §18.8 Binomial).
\\ Generated with PARI/GP 2.15.4. write() APPENDS, so run it in an EMPTY directory and copy the
\\ CSVs here:  mkdir /tmp/g && cd /tmp/g && gp -q < .../gen_signs_roots.gp
\\ random() streams can differ between PARI versions; the committed CSVs are the 2.15.4 output.
\\ Rows are comma-separated; "E" marks an input the function must refuse. Columns:
\\   binom_int_grid.csv              : n, k, binomial(n, k)
\\   gcd_grid.csv                    : a, b, gcd, lcm, d, u, v   (gcdext(a, b) = [u, v, d])
\\   xgcd_big.csv                    : a, b, d, u, v, lcm
\\   iroot_grid.csv, iroot_big.csv   : a, n, floor(a^(1/n))   (fl below; "E" = even n, a < 0)
\\   ispower_k_grid.csv              : a, k, ispower(a, k)
\\   perfect_power_grid.csv, perfect_power_big.csv : a, k, r with k = ispower(a, , &r)
\\                                     (k = 0 and r = 0 when a is not a perfect power)
setrand(20261108);
fl(a,n)=if(a>=0, sqrtnint(a,n), my(t=sqrtnint(-a,n)); if(t^n==-a, -t, -t-1));
\\ binomial(n,k), n in -30..30, k in -5..30
for(n=-30,30, for(k=-5,30, write("binom_int_grid.csv", n, ",", k, ",", binomial(n,k))));
\\ gcd, lcm, gcdext [u,v,d] for a,b in -30..30
for(a=-30,30, for(b=-30,30, my(e=gcdext(a,b)); write("gcd_grid.csv", a, ",", b, ",", gcd(a,b), ",", lcm(a,b), ",", e[3], ",", e[1], ",", e[2])));
\\ random big pairs (some sharing a factor, every 10th with b = 0): a, b, d, u, v, lcm(a,b)
for(i=1,300, my(a=random(2^130)-2^129, b=random(2^100)-2^99, c=random(2^40)+1); if(i%3==0, a*=c; b*=c); if(i%10==0, b=0); my(e=gcdext(a,b)); write("xgcd_big.csv", a, ",", b, ",", e[3], ",", e[1], ",", e[2], ",", lcm(a,b)));
\\ floor n-th root, a in -400..400, n in 1..9 (E: even n, a < 0)
for(a=-400,400, for(n=1,9, write("iroot_grid.csv", a, ",", n, ",", if(a<0 && n%2==0, "E", fl(a,n)))));
\\ random big roots: a, n, floor root
for(i=1,200, my(a=random(2^256)-2^255, n=2+random(15)); if(i%4==0, my(b=random(2^40)-2^39); a=b^n); if(a<0 && n%2==0, n++); write("iroot_big.csv", a, ",", n, ",", fl(a,n)));
\\ ispower(a,k) for a in -400..400, k in 1..8
for(a=-400,400, for(k=1,8, write("ispower_k_grid.csv", a, ",", k, ",", ispower(a,k))));
\\ ispower(a): largest k and the root (0,0 when a is not a perfect power), a in -3000..3000
for(a=-3000,3000, my(r, k=ispower(a,,&r)); write("perfect_power_grid.csv", a, ",", k, ",", if(k, r, 0)));
\\ big perfect powers: b^k with random b, k, both signs, plus products that are not powers
for(i=1,120, my(b=2+random(2^20), k=2+random(70), s=if(random(2), -1, 1), a=s*b^k); if(i%5==0, a=a*(2*random(2^30)+1)+1); my(r, kk=ispower(a,,&r)); write("perfect_power_big.csv", a, ",", kk, ",", if(kk, r, 0)));
