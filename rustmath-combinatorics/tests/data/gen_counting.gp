\\ C1 fixture rustmath-combinatorics/tests/data/ch144_counting.csv
\\ columns: fn,a,b,sign,residue,bits  with residue = value mod (2^127-1) in [0, M), bits = bit length of |value|
default(parisizemax, 2^31)
M = 2^127 - 1;
row(f, a, b, v) = print(f, ",", a, ",", b, ",", sign(v), ",", lift(Mod(v, M)), ",", if(v == 0, 0, logint(abs(v), 2) + 1));
eul(n, k) = if(k >= n, (n == 0 && k == 0), sum(j = 0, k, (-1)^j * binomial(n+1, j) * (k+1-j)^n));
mot(n) = sum(k = 0, n\2, binomial(n, 2*k) * binomial(2*k, k) / (k+1));
sch(n) = sum(k = 0, n, binomial(n+k, n-k) * binomial(2*k, k) / (k+1));
der(n) = my(d = 1); for(i = 1, n, d = i*d + (-1)^i); d;
ff(x, k) = prod(i = 0, k-1, x - i);
rf(x, k) = prod(i = 0, k-1, x + i);
print("fn,a,b,sign,residue,bits");
foreach([0, 1, 5, 30, 1000, 31622, 100000, 316228], n, row("catalan", n, 0, binomial(2*n, n)/(n+1)));
foreach([0, 1, 2, 93, 94, -1, -2, -94, 1000, 100000, 1000000, -1000000, -999999, 1300001], n, row("fibonacci", n, 0, fibonacci(n)));
foreach([0, 1, 2, 5, -1, -2, -5, 1000, 100000, -100001, 1300001], n, row("lucas", n, 0, fibonacci(n-1) + fibonacci(n+1)));
foreach([0, 1, 10, 100, 600, 1000], n, row("bell", n, 0, sum(k = 0, n, stirling(n, k, 2))));
foreach([[0,0],[5,0],[1000,999],[1000,500],[1000,1],[2000,3],[500,250],[7,9]], p, row("stirling1", p[1], p[2], abs(stirling(p[1], p[2], 1))));
foreach([[0,0],[5,0],[1000,999],[1000,500],[1000,1],[2000,3],[500,250],[7,9]], p, row("stirling2", p[1], p[2], stirling(p[1], p[2], 2)));
foreach([[0,0],[1,0],[1000,999],[1000,998],[1000,500],[1000,1],[2000,3],[1201,600],[7,9]], p, row("eulerian", p[1], p[2], eul(p[1], p[2])));
foreach([0, 1, 2, 100, 4500, 10000], n, row("motzkin", n, 0, mot(n)));
foreach([0, 1, 2, 100, 10000, 20000], n, row("schroder_large", n, 0, sch(n)));
foreach([1, 2, 100, 10000], n, row("schroder_small", n, 0, sch(n)/2));
foreach([0, 1, 2, 10, 1400, 10000, 30000], n, row("derangements", n, 0, der(n)));
foreach([0, 1, 20, 21, 10000, 30000], n, row("factorial", n, 0, n!));
foreach([[10,3],[10,10],[5,6],[5,0],[4294967295,2],[4294967295,20000],[-5,3],[-4294967296,3],[2^64+5,3],[-(2^64+5),2]], p, row("falling_factorial", p[1], p[2], ff(p[1], p[2])));
foreach([[10,3],[5,0],[4294967295,2],[4294967295,20000],[-5,3],[-5,5],[-5,6],[2^64-1,3],[-(2^64),2]], p, row("rising_factorial", p[1], p[2], rf(p[1], p[2])));
foreach([[200000,100000],[4294967295,2],[4294967295,4294967293],[5,6]], p, row("binomial", p[1], p[2], binomial(p[1], p[2])));
quit
