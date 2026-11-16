"""F-9 prototype on the covers over L: reads HJLPPZ's polynomial_calF.m from $M23ISGALOIS at run
time (no licence: nothing is written to disk), reduces it at every degree-one prime (p, s - s0),
5 <= p <= 127, and pipes the reduced forms to gp, which runs the criterion of irred.gp.
Usage: python3 -I l_covers.py /path/to/m23isgalois | gp -q"""
import re, sys
d = sys.argv[1]
text = open(d + "/polynomial_calF.m").read()
body = text[text.index("calF :=") + 7: text.index("assert")]
body = re.sub(r"\s+", "", body)
terms = {}
for m in re.finditer(r"\(((?:\([-0-9]+\)\*s\^\d\+?)+)\)\*T\^(\d+)\*V\^(\d+)", body):
    cs = [0] * 6
    for c, e in re.findall(r"\(([-0-9]+)\)\*s\^(\d)", m.group(1)):
        cs[int(e)] = int(c)
    terms[(int(m.group(2)), int(m.group(3)))] = cs
assert len(terms) == 120, len(terms)
L = [4, 5, 1, -3, -3, -1, 1]
def primes(lo, hi):
    return [n for n in range(lo, hi + 1) if n > 1 and all(n % q for q in range(2, int(n ** 0.5) + 1))]
out = ['default(parisize, "400M");', 'read("irred.gp");']
for p in primes(5, 127):
    for s0 in range(p):
        if sum(c * s0 ** e for e, c in enumerate(L)) % p:
            continue
        poly = "+".join("%d*x^%d*y^%d" % (sum(c * pow(s0, e, p) for e, c in enumerate(cs)) % p, i, j)
                        for (i, j), cs in sorted(terms.items()))
        out.append("F = %s;" % poly)
        # print the verdict only: for a line component the content polynomial (derived from the
        # unlicensed data) is not printed, only the ruling
        out.append('r = crit(F, 4, 23, %d); print("L (%d, %d): ", if (r[1] == "line", ["line", r[2][1]], r));' % (p, p, s0))
print("\n".join(out))
