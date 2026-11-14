\\ C1 fixture rustmath-combinatorics/tests/data/numbpart.csv: n,residue,bits with residue = numbpart(n) mod (2^127-1)
M = 2^127 - 1;
row(n) = my(p = numbpart(n)); print(n, ",", lift(Mod(p, M)), ",", logint(p, 2) + 1);
print("n,residue,bits");
foreach([0, 1, 2, 3, 4, 5, 100, 127, 128, 416, 417, 1000, 1001, 2000, 2001, 3000, 10000, 100000, 1000000, 10000000, 100000000], n, row(n));
setrand(20261112);
for (i = 1, 40, row(2 + random(10^6 - 1)));
for (i = 1, 10, row(10^6 + random(99*10^6)));
quit
