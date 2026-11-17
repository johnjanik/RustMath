# Builds cc0_t2_steiner_reference.json: the owner's arith.py (M23 repository d05475b, ported
# algorithm reference) run with python-flint 0.9.0 on the monic associate of Le Lay's CC0
# polynomial data/specialization_t2_Z.json (commit 3f6ab4b). Usage:
#   venv/bin/python -I make_cc0_reference.py <specialization_t2_Z.json> <out.json> <M23 checkout>
import sys, json, time, hashlib
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "../M23")  # a checkout of johnjanik/M23 at d05475b
sys.set_int_max_str_digits(1000000)
import flint
from m23golay import arith

src = sys.argv[1]
raw = open(src, 'rb').read()
d = json.loads(raw)
c = [int(x) for x in d['coefficients_low_to_high']]
a = c[-1]; n = len(c) - 1
g = [c[j] * a**(n - 1 - j) for j in range(n)] + [1]      # a^(n-1) f(x/a)
M = arith.root_bound(g)
best = M * 128
assert best.denominator == 1
best = best.numerator
B5 = int(253 * (7 * M)**5) + 1
out = {
    "source": {"file": "lelay_specialization_t2_Z.json", "sha256": hashlib.sha256(raw).hexdigest()},
    "monic_associate": "g(x) = a^22 f(x/a), a = leading coefficient of f; coefficient of x^i is f_i a^(22-i)",
    "leading_coefficient": str(a),
    "fujiwara_best": str(best),
    "t5_bound": str(B5),
    "j5_bound": str(int(5313 * M**5) + 1),
    "j7_bound": str(int(253 * M**7) + 1),
    "reference": "arith.py (johnjanik/M23 d05475b) with python-flint %s, Python %s" % (flint.__version__, sys.version.split()[0]),
    "second_reference": "PARI/GP 2.15.4: T5 at 29 (N = 270, 250), J5, J7 and the signs by polrootspadic (design SL gp/steiner.gp, run_cc0.gp, run_cc0_signs.gp) and by Newton in Mod(Mod(., p^N), f) (review 1 of design SL, rev.gp); T5 at 67 (N = 216) by rev.gp (implementation report SL, gp/check_sl.gp). All equal.",
}
t0 = time.time()
lab = arith.steiner_labelling(g, 29)
J = arith.golay_invariants(g, lab)
lab0 = arith.steiner_labelling(g, 29, extra_digits=0)
lab67 = arith.steiner_labelling(g, 67)
J67 = arith.golay_invariants(g, lab67)
ps = arith.inert_primes(g, 3, 12)
T5 = lab.t5[lab.sign]
signs = {str(p): arith.frobenius_sign(g, p, T5) for p in ps}
out["labelling"] = {
    "29": {"N": lab.N, "sign": lab.sign, "T5_plus": str(lab.t5[1]), "T5_minus": str(lab.t5[-1])},
    "29_minimal": {"N": lab0.N, "sign": lab0.sign, "T5_plus": str(lab0.t5[1]), "T5_minus": str(lab0.t5[-1])},
    "67": {"N": lab67.N, "sign": lab67.sign, "T5_plus": str(lab67.t5[1]), "T5_minus": str(lab67.t5[-1])},
}
out["T5"] = str(T5)
out["J5"] = str(J["J5"]); out["J7"] = str(J["J7"])
out["J_at_67_equal"] = (J67["J5"] == J["J5"] and J67["J7"] == J["J7"])
out["inert_primes"] = ps
out["frobenius_signs"] = signs
print("oracle time %.1f s" % (time.time() - t0))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print({k: (v if len(str(v)) < 80 else '<%d chars>' % len(str(v))) for k, v in out.items()})
