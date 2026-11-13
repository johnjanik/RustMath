\\ Generator of ecpp_discriminants.txt, the discriminant list of the ECPP prover
\\ (rustmath-integers/src/ecpp/hilbert.rs; MAGMA Handbook ch. 18 §18.9.1 IsPrime and
\\ PrimalityCertificate, [AM93] A. O. L. Atkin, F. Morain, Elliptic curves and primality proving,
\\ Math. Comp. 61 (1993) 29-68).
\\
\\ Output: line h (h = 1..100) is "h |D_1| |D_2| ...", the absolute values of all fundamental
\\ discriminants D < 0 with class number h(D) = h, ascending. 42,272 discriminants, 273,422 bytes,
\\ largest |D| = 2,383,747, sha256 33c5463dc8c737aa15a2bbb3db5ba3d9d0c4dd8f9f606b223077d7854132ee3b.
\\
\\ Completeness: M. Watkins, Class numbers of imaginary quadratic fields, Math. Comp. 73 (2004)
\\ 907-938, lists all imaginary quadratic fields with h <= 100 (42,272 fields, the largest |D| is
\\ 2,383,747), so the scan to |D| <= 2,400,000 finds all of them. The script prints the counts per h;
\\ their total must be Watkins' number 42,272.
\\
\\ PARI/GP 2.15.4: gp -q ecpp_discriminants.gp (about 83 s). Run it in an empty directory: write()
\\ appends to ecpp_discriminants.txt.
default(parisizemax, 2^30);
H = 100; X = 2400000;
L = vector(H, h, List());
for(n = 3, X, D = -n; if(isfundamental(D), h = qfbclassno(D); if(h <= H, listput(L[h], n))));
c = vector(H, h, #L[h]);
print("counts ", c);
print("total ", vecsum(c));
f = "ecpp_discriminants.txt"; for(h = 1, H, write(f, h, " ", strjoin(apply(x -> Str(x), Vec(L[h])), " ")));
quit
