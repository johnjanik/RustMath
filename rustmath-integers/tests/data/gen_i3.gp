\\ Reference fixtures for rustmath-integers/tests/ch18_signs_roots.rs, second set
\\ (MAGMA Handbook ch. 18: §18.4.5 IsPower(n, k) with the root, §18.4.8 Isqrt, §18.6 Xgcd at
\\ Lehmer sizes, §18.8 Binomial at sizes that need the product tree).
\\ Generated with PARI/GP 2.15.4. write() APPENDS, so run it in an EMPTY directory and copy the
\\ CSVs here:  mkdir /tmp/g && cd /tmp/g && gp -q < .../gen_i3.gp
\\ random() streams can differ between PARI versions; the committed CSVs are the 2.15.4 output.
\\ Rows are comma-separated. Columns:
\\   isqrt_big.csv     : a, sqrtint(a)
\\   ispower_root.csv  : a, k, ispower(a, k), r   (r = 0 when ispower(a, k) = 0)
\\   xgcd_lehmer.csv   : a, b, d, u, v           (gcdext(a, b) = [u, v, d])
\\   binom_big.csv     : n, k, binomial(n, k) % (2^127 - 1), bit length of |binomial(n, k)|
default(parisizemax, 2000000000);
setrand(20261112);
M127 = 2^127 - 1;
\\ isqrt: random sizes, squares and their neighbours
{
for(i = 1, 90,
  my(bits = [65, 100, 128, 200, 500, 1000, 2000, 3000][1 + (i % 8)], a);
  a = random(2^bits) + 2^(bits - 1);
  if(i % 3 == 1, my(b = random(2^(bits \ 2)) + 2^(bits \ 2 - 1)); a = b^2 + (i % 9 == 1) - (i % 9 == 4));
  write("isqrt_big.csv", a, ",", sqrtint(a)));
}
\\ ispower(a, k, &r): small grid, k in 1..6
{
for(a = -200, 200, for(k = 1, 6,
  my(r = 0, f = ispower(a, k, &r)); write("ispower_root.csv", a, ",", k, ",", f, ",", if(f, r, 0))));
}
\\ ispower: large exact powers, neighbours, composite k
{
for(i = 1, 80,
  my(b = random(2^40) + 2, k = [2, 3, 4, 5, 6, 7, 9, 12, 15, 30, 31, 64][1 + (i % 12)], s, a, r = 0, f);
  s = if(i % 4 == 0 && k % 2 == 1, -1, 1);
  a = s * b^k;
  if(i % 5 == 0, a += 2);
  if(i % 7 == 0, a = s * 2^(k * (i % 11 + 1)) * b^k);
  f = ispower(a, k, &r);
  write("ispower_root.csv", a, ",", k, ",", f, ",", if(f, r, 0)));
}
\\ xgcd at Lehmer sizes (65..3000 bits): random, common factors, Fibonacci, b = a, b | a, unbalanced, zeros
{
for(i = 1, 24,
  my(ba = [65, 128, 300, 1000, 3000][1 + (i % 5)], bb, a, b, c, e);
  bb = max(65, ba - random(40));
  a = (random(2^ba) + 2^(ba - 1)) * if(random(2), 1, -1);
  b = (random(2^bb) + 2^(bb - 1)) * if(random(2), 1, -1);
  if(i % 3 == 0, c = random(2^200) + 1; a *= c; b *= c);
  if(i % 8 == 1, a = fibonacci(1000 + 300 * i); b = -fibonacci(999 + 300 * i));
  if(i % 8 == 2, b = -a);
  if(i % 8 == 3, b = 7 * a);
  if(i % 8 == 4, b = random(2^70) + 1);
  if(i % 8 == 5, a = 0);
  if(i % 8 == 6, b = 2^(ba + 5) * 3);
  e = gcdext(a, b);
  write("xgcd_lehmer.csv", a, ",", b, ",", e[3], ",", e[1], ",", e[2]));
}
\\ binomial: residues mod 2^127-1 and bit lengths
{
my(cases = List());
for(i = 1, 40, my(n = random(10^6) + 2, k = random(n + 1)); listput(cases, [n, k]));
for(i = 1, 10, my(n = random(10^4) + 2); listput(cases, [n, n \ 2]));
for(i = 1, 10, listput(cases, [10^30 + 7 + i, random(2000) + 2]));
for(i = 1, 6, listput(cases, [2^64 - 59 + i, random(5000) + 2]));
for(i = 1, 8, my(n = -(random(10^6) + 1), k = random(3000) + 2); listput(cases, [n, k]));
listput(cases, [524288, 262144]); listput(cases, [524288, 262143]); listput(cases, [10^6, 500000]);
listput(cases, [5, -3]); listput(cases, [-5, -3]); listput(cases, [100, 101]);
for(j = 1, #cases, my(n = cases[j][1], k = cases[j][2], c = binomial(n, k));
  write("binom_big.csv", n, ",", k, ",", c % M127, ",", if(c, #binary(abs(c)), 0)));
}
