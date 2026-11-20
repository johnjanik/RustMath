# Generator of tests/data/s1_fuzz.tsv: a fixed sample of review 9's random harness with gp references.
# Run from rustmath-symbolic/tests/data (the prototype binary is in the session scratchpad):
#     python3 -I gen_s1_fuzz.py > s1_fuzz.tsv 2> s1_fuzz.log
# env: BIN (default EXP/rev7rt4/bin/rt2_rel, the section 2.12 parser probe linked against the
#      design's prototype proto13), PER_SEED (default 60 inputs per seed).
#
# The inputs are review 9's (EXP/review9/fuzz.py with seeds 1 to 4, fuzz2.py with seeds 11 to 14,
# EXP = fu2/design/S1-symbolic in the session scratchpad): the generator below is a verbatim copy of
# their `leaf` and `expr` and of their main loop's random calls, so seed s gives the same sequence
# of (input, point, order) as `fuzz.py s N out` (checked against the first column of
# EXP/review9/fz_<s>.out and fy_<s>.out at start-up). fuzz2.py differs from fuzz.py only in FUN (it
# adds the inverse functions) and in tagging suspect refusals.
#
# For each input: gp's series (s1_gp.py, realprecision 300, the input translated with the grammar of
# design section 2.12) and the prototype's outcome (`rt2_rel both`, taylor_with_budget with the
# REPL budget 1e8, and ENTRY=lim, the limit path). The class of a row comes from these two outcomes;
# the generator does not compare the prototype's value with gp's (the tests do that, against the
# port; on the prototype, every V row passed and agreed with gp, tests_report.md). The kinds of the
# R and U rows are the prototype's kinds, each reviewed by hand against the design and the owner's
# rules (tests_report.md); the hand review moved the rows of OVERRIDES below.
#   V  gp gives a series and the prototype gives a value: `value` with gp's coefficients (the test
#      compares the engine with gp to 60 digits).
#   R  gp gives an error and the prototype refuses: `err Kind` with the prototype's kind.
#   U  gp gives a series (it decides with floating point) but the prototype's exact arithmetic
#      cannot decide: `err Kind` for UndecidedZero, PrecisionExhausted, Unsupported, BudgetExceeded,
#      LimitExceeded (the design's honest refusals; tagged U in the id).
#      An UndecidedZero row also accepts gp's value (round 3, testcheck n7): the undecided quantity
#      is a nonzero constant that the frozen design's constant field cannot decide, so a port that
#      decides more gives gp's value, which is the true one (written `err UndecidedZero | value ...`,
#      with the limit of each outcome; the values were checked against gp's derivnum of the scalar
#      function, tests_report.md).
#   N  gp gives a series but the prototype says NotLaurent or DivisionByZero: hand review only
#      (gp's series of sqrt(x^2), abs and sign conventions, formal one-sided branches).
#   E  gp gives an error but the prototype gives a value: left out (no reference), unless reviewed by
#      hand in OVERRIDES with gp's series of the simplified input (class E; round 3, testcheck n6).
# R rows listed in ALSO_ACCEPTED (round 3, testcheck n7 and n8) accept a second refusal kind: the true
# outcome where the prototype's kind comes from the frozen design's constant field, or the kind of a
# second refusal cause that another evaluation order meets first.
#   B  a candidate of review 9's blockers, whatever gp says: a DivisionByZero "0 to a negative power"
#      of an input that contains asin, acos, acosh or asinh (B1), "arctanh at 1 or -1" (B2), or an
#      UndecidedZero of an integrand factor (1 + c^2, 1 - c^2, c - 1, c + 1, printed by the
#      prototype as "(1 + ...)" or "(-1 + ...)") in an input with an atan-family function (B1 when
#      the operand is an exact constant: the owner's rule forms no integrand there). Every B row
#      must be reviewed by hand: either in OVERRIDES, or (an UndecidedZero whose operand is not
#      constant, so the integrand is needed and its zero test is honestly undecided) in
#      INTEGRAND_NOT_CONSTANT, which keeps its automatic class. An unreviewed B row stops the
#      generator.
# The hand-reviewed rows of OVERRIDES are tagged B1, B2, N, or O: a row whose outcome depends on a
# fold that the scope of the owner's rule leaves optional (tests/s1_common/mod.rs), written with
# several accepted outcomes, `value ... | err Kind`, and one limit expectation per outcome.
# The limit column: Finite(q) when the prototype's limit path gives Finite(q) and gp's constant term
# is q; for a prototype `Unknown`: `Unknown` when gp's constant term is not a rational, and
# `Finite(q),Unknown` (either) when gp's constant term equals a rational q to the 70 digits it prints (the
# design's constant field keeps it as a kernel such as exp(4 log 2), and a port that renders q
# decides the true limit); `-` (not checked) otherwise.
import os
import random
import re
import subprocess
import sys
from fractions import Fraction

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_gp as g

EXP = '/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/design/S1-symbolic'
BIN = os.environ.get('BIN', EXP + '/rev7rt4/bin/rt2_rel')
PER_SEED = int(os.environ.get('PER_SEED', '60'))
SEEDS = [(1, 'fuzz'), (2, 'fuzz'), (3, 'fuzz'), (4, 'fuzz'), (11, 'fuzz2'), (12, 'fuzz2'), (13, 'fuzz2'), (14, 'fuzz2')]

# ---- verbatim from EXP/review9/fuzz.py (FUN of fuzz2.py below) ----
POINTS = ['0', '1', '-1/2', '2', '1/3', '-1']
FUN_FUZZ = ['sin', 'cos', 'exp', 'atan', 'sqrt', 'log', 'tanh', 'sinh', 'cosh', 'asinh', 'asin', 'acos', 'abs', 'sign', 'atanh', 'tan']
FUN_FUZZ2 = ['asin', 'acos', 'acosh', 'asinh', 'atanh', 'atan', 'sin', 'cos', 'exp', 'atan', 'sqrt', 'log', 'tanh', 'sinh', 'cosh', 'asinh', 'asin', 'acos', 'abs', 'sign', 'atanh', 'tan']
FUN = FUN_FUZZ


def leaf(pt):
    r = random.random()
    if r < 0.40: return 'x'
    if r < 0.55: return random.choice(['2', '3', '1/2', '(1/3)', '5', '(-2)', '(-1/2)', 'pi'])
    if r < 0.65: return random.choice(['(x-x)', '((x+1)^2-x^2-2*x-1)', '(0*((1)/(x-%s)))' % pt, '((x-%s)/(x-%s))' % (pt, pt)])
    if r < 0.72: return random.choice(['sign(x^2+1)', 'sign((x-%s)^2)' % pt, '(sin(1)^2+cos(1)^2)', '(x-%s)' % pt])
    if r < 0.80: return '(x-%s)' % pt
    return random.choice(['(x+1)', '(x-1)', '(2*x+1)', '(x^2+1)', '(1-x)'])


def expr(d, pt):
    if d <= 0: return leaf(pt)
    r = random.random()
    if r < 0.30:
        return random.choice(FUN) + '(' + expr(d - 1, pt) + ')'
    if r < 0.45:
        a, b = expr(d - 1, pt), expr(d - 1, pt)
        return '(' + a + ')' + random.choice(['+', '-']) + '(' + b + ')'
    if r < 0.60:
        a, b = expr(d - 1, pt), expr(d - 1, pt)
        return '(' + a + ')*(' + b + ')'
    if r < 0.70:
        a, b = expr(d - 1, pt), expr(d - 1, pt)
        return '(' + a + ')/(' + b + ')'
    if r < 0.85:
        e = random.choice(['2', '3', '(-1)', '(-2)', '(1/2)', '(1/3)', '(2/3)', '(-1/2)', 'x', '(x+1)', '(2+x)', '(x-%s+2)' % pt, '(sin(x)+1)', '(2*sign(x^2+1))', '((x-x)+2)'])
        return '(' + expr(d - 1, pt) + ')^' + e
    if r < 0.93:
        return '(' + random.choice(['2', '3', '1/2', 'pi', '(1+sin(1))']) + ')^(' + expr(d - 1, pt) + ')'
    return '(' + expr(d - 1, pt) + ')+(' + leaf(pt) + ')'
# ---- end of the verbatim copy ----


def inputs_of(seed, which, count):
    global FUN
    FUN = FUN_FUZZ if which == 'fuzz' else FUN_FUZZ2
    random.seed(seed)
    out = []
    i = 0
    while len(out) < count and i < 10 * count:
        i += 1
        pt = random.choice(POINTS)
        f = expr(random.choice([2, 3, 3, 4]), pt)
        if len(f) > 400:
            continue
        n = random.choice([0, 1, 2, 3])
        out.append((f, pt, n))
    return out


def check_against_review9(seed, which, rows):
    name = ('fz_%d.out' if which == 'fuzz' else 'fy_%d.out') % seed
    path = EXP + '/review9/' + name
    if not os.path.exists(path):
        print('# no %s to check the replication against' % path, file=sys.stderr)
        return
    rec = []
    for l in open(path):
        p = l.rstrip('\n').split('\t')
        if len(p) >= 4 and not l.startswith('STATS'):
            rec.append((p[1], p[2], int(p[3])))
    for a, b in zip(rows, rec):
        if a != b:
            raise SystemExit('replication of %s differs: %r vs %r' % (name, a, b))
    print('# seed %d (%s): %d inputs equal to the first ones of %s' % (seed, which, len(rows), name), file=sys.stderr)


KIND = {
    'NotLaurent': 'NotLaurent', 'Unsupported': 'Unsupported', 'Undecided': 'UndecidedZero',
    'DivisionByZero': 'DivisionByZero', 'BadPoint': 'InvalidPoint', 'PrecisionExhausted': 'PrecisionExhausted',
    'Budget': 'BudgetExceeded',
}


def proto(f, pt, n, entry):
    fn = os.environ.get('TMPIN', '/tmp/claude-0/-home-user-RustMath/501faa10-c1ae-5db8-88c8-be37c311b5d8/scratchpad/fu2/s1impl/fuzz_cur.txt')
    open(fn, 'w').write(f + '\n')
    env = dict(os.environ)
    env['ENTRY'] = entry
    env['PRINT'] = '1'
    r = subprocess.run([BIN, 'both', fn, pt, str(n), '8388608'], capture_output=True, text=True, env=env, timeout=120)
    last = (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']
    return r.returncode, last[0]


def proto_kind(line):
    m = re.match(r'^ERR (\w+)', line)
    if not m:
        return None, ''
    k = m.group(1)
    if k == 'Limit':
        what = re.search(r'Limit\("([^"]*)"', line)
        w = what.group(1) if what else ''
        lim = {'factors of one monomial': 'MonomialFactors', 'terms of one constant': 'ConstantTerms',
               "depth of a kernel's expression": 'KernelDepth', "nodes of the result's tree": 'TreeSize'}.get(w, 'Kernels')
        return 'LimitExceeded(%s)' % lim, line
    return KIND.get(k, k), line


def proto_msg(msg):
    """The prototype's error without its timing and counters (they vary from run to run)."""
    return re.sub(r' [0-9.]+(ns|µs|ms|s) units.*$', '', msg[4:])[:120]


def rational_or_none(s):
    if re.fullmatch(r'-?\d+(/\d+)?', s):
        return Fraction(s)
    return None


def close_to(s, q):
    try:
        v = float(s)
    except ValueError:
        return False
    return abs(v - float(q)) <= 1e-30 * max(1.0, abs(float(q)))


def numerically_rational(c0):
    """The rational q (denominator at most 10^6) that gp's decimal constant term c0 = (k, re, im)
    equals, or None. gp computes at 300 digits and prints 70 significant digits (s1_gp.CO), so the
    test is on those: |re - q| <= 10^-68 max(1, |q|), and an imaginary part that is 0 or gp's
    rounding noise, |im| <= 10^-250 (`exp(2 log(-1))` has gp's imaginary part -2.9e-309)."""
    if c0 is None:
        return None
    try:
        r = Fraction(c0[1])
        i = Fraction(c0[2])
    except ValueError:
        return None
    if abs(i) > Fraction(1, 10 ** 250):
        return None
    q = r.limit_denominator(10 ** 6)
    if abs(r - q) <= Fraction(1, 10 ** 68) * max(1, abs(q)):
        return q
    return None


def value_limit(v, coefs):
    """The limit path's expectation for an accepted value outcome with gp's coefficients (valuation
    v): `Unknown` for a pole; for a rational constant term q (exact, or to gp's 70 digits), or q = 0
    when v > 0, `Finite(q),Unknown` (a port that renders q decides it, the design's constant field
    may keep a kernel); `Unknown` otherwise."""
    if v < 0:
        return 'Unknown'
    if v > 0:
        return 'Finite(0),Unknown'
    c0 = next((c for c in coefs if c[0] == 0), None)
    q = rational_or_none(c0[1]) if c0 is not None and c0[2] == '0' else None
    if q is None:
        q = numerically_rational(c0)
    return 'Finite(%s),Unknown' % q if q is not None else 'Unknown'


def main():
    all_rows = []
    for seed, which in SEEDS:
        rows = inputs_of(seed, which, PER_SEED)
        check_against_review9(seed, which, rows)
        for j, (f, pt, n) in enumerate(rows):
            all_rows.append(('%s%d.%d' % ('z' if which == 'fuzz' else 'y', seed, j), f, pt, n))
    gpres = g.series_batch([(f, pt, n) for (_, f, pt, n) in all_rows])
    stats = {}
    out = []
    review = []
    for (rid, f, pt, n), gr in zip(all_rows, gpres):
        rc, tw_line = proto(f, pt, n, 'tw')
        rc2, lim_line = proto(f, pt, n, 'lim')
        if rc != 0 or rc2 != 0:
            review.append('CRASH %s\t%s\t%s\t%d\t%s' % (rid, f, pt, n, tw_line))
            stats['crash'] = stats.get('crash', 0) + 1
            continue
        lim_finite = re.search(r'Finite\((.*)\) VmHWM', lim_line)
        lim_proto = ('Finite', lim_finite.group(1).strip()) if lim_finite else ('Unknown', None)
        kind, msg = proto_kind(tw_line)
        gp_ok = gr[0] == 'ok'
        if kind is None and gp_ok:
            cls = 'V'
        elif kind is not None and not gp_ok:
            cls = 'R'
        elif kind is not None and gp_ok:
            cls = 'U' if kind in ('UndecidedZero', 'PrecisionExhausted', 'Unsupported', 'BudgetExceeded') or kind.startswith('LimitExceeded') else 'N'
        else:
            cls = 'E'
        b1 = kind == 'DivisionByZero' and '0 to a negative power' in msg and re.search(r'\ba(sin|cos|cosh|sinh)\(', f)
        b2 = kind == 'DivisionByZero' and 'arctanh at 1 or -1' in msg
        b1_integrand = (kind == 'UndecidedZero' and re.match(r'^ERR Undecided\(\(-?1 \+ ', msg)
                        and re.search(r'\ba(sin|cos|tan|sinh|cosh|tanh)\(', f))
        if b1_integrand and rid in INTEGRAND_NOT_CONSTANT:
            review.append('B1-CANDIDATE-KEPT %s\t%s\t%s\t%d\t%s' % (rid, f, pt, n, INTEGRAND_NOT_CONSTANT[rid]))
            b1_integrand = False
        if b1 or b2 or b1_integrand:
            cls = 'B'
            if rid not in OVERRIDES:
                raise SystemExit('unreviewed candidate of review 9 B1/B2: %s %r at %s, order %d: %s' % (rid, f, pt, n, msg))
        stats[cls] = stats.get(cls, 0) + 1
        if rid in EXCLUDED:
            review.append('EXCLUDED %s\t%s\t%s\t%d\t%s' % (rid, f, pt, n, EXCLUDED[rid]))
            stats['excluded'] = stats.get('excluded', 0) + 1
            continue
        if cls in ('N', 'B', 'E') or rid in OVERRIDES:
            review.append('%s %s\t%s\t%s\t%d\tproto: %s\tgp: %s' % (cls, rid, f, pt, n, tw_line[:160], gr[1] if not gp_ok else 'series v=%d c0=%s' % (gr[1], gr[2][0][1:])))
            if rid not in OVERRIDES:
                continue
            want_o, lim_o, why, simplified = OVERRIDES[rid]
            several = isinstance(want_o, list)
            if several:
                tag = 'O'
            elif cls == 'E':
                tag = 'E'
            else:
                tag = 'B2' if 'arctanh at 1 or -1' in msg or why.startswith('B2') else ('N' if cls == 'N' else 'B1')
            wants = []
            for w in (want_o if several else [want_o]):
                if w == 'value':
                    ref = gr
                    if simplified is not None:
                        ref = g.series(simplified, pt, n)
                    if ref[0] != 'ok':
                        raise SystemExit('override %s wants gp\'s value, but gp failed: %s' % (rid, ref[1]))
                    wants.append('value %d %s' % (ref[1], ' '.join('%s:%s' % (re_, im_) for (_, re_, im_) in ref[2])))
                else:
                    wants.append(w)
            want = ' | '.join(wants)
            if several:
                lim_o = ' | '.join(lim_o)
            gp_note = ('series of the simplified input %s' % simplified) if simplified is not None else ('series' if gp_ok else gr[1][:80])
            note = 'hand-reviewed: %s; gp: %s; prototype: %s' % (why, gp_note, proto_msg(msg) if kind else 'value')
            out.append('\t'.join(['%s:%s' % (tag, rid), f, pt, str(n), want, lim_o, note]))
            stats['hand'] = stats.get('hand', 0) + 1
            continue
        # the limit column
        if cls == 'V':
            v, coefs = gr[1], gr[2]
            c0 = next((c for c in coefs if c[0] == 0), None)
            if lim_proto[0] == 'Finite':
                q = rational_or_none(lim_proto[1].replace(' ', '').strip('()'))
                if q is not None and c0 is not None and c0[2] == '0' and (rational_or_none(c0[1]) == q or close_to(c0[1], q)):
                    lim = 'Finite(%s)' % q
                else:
                    review.append('LIMIT %s\t%s\t%s\t%d\tproto lim %s, gp c0 %s' % (rid, f, pt, n, lim_proto, c0))
                    lim = '-'
            else:
                exact_c0 = c0 is not None and c0[2] == '0' and rational_or_none(c0[1]) is not None
                q = numerically_rational(c0) if (v == 0 and not exact_c0) else None
                if q is not None:
                    lim = 'Finite(%s),Unknown' % q
                else:
                    lim = 'Unknown' if (v < 0 or not exact_c0) else '-'
            want = 'value %d %s' % (v, ' '.join('%s:%s' % (re_, im_) for (_, re_, im_) in coefs))
        else:
            lim = 'Unknown' if lim_proto[0] == 'Unknown' else '-'
            want = 'err ' + kind
            if cls == 'U' and kind == 'UndecidedZero' and rid in C1_NOT_A_VALUE:
                want += ' | ' + C1_NOT_A_VALUE[rid][0]
                lim = '%s | Unknown' % lim
            elif cls == 'U' and kind == 'UndecidedZero':
                v, coefs = gr[1], gr[2]
                want += ' | value %d %s' % (v, ' '.join('%s:%s' % (re_, im_) for (_, re_, im_) in coefs))
                lim = '%s | %s' % (lim, value_limit(v, coefs))
            if rid in ALSO_ACCEPTED:
                if cls != 'R':
                    raise SystemExit('ALSO_ACCEPTED lists %s of class %s' % (rid, cls))
                want += ' | ' + ALSO_ACCEPTED[rid][0]
        note = 'gp: ' + ('series' if gp_ok else gr[1][:100]) + '; prototype: ' + (proto_msg(msg) if kind else 'value')
        if cls == 'U' and kind == 'UndecidedZero' and rid in C1_NOT_A_VALUE:
            note += '; also accepted: %s (%s)' % C1_NOT_A_VALUE[rid]
        elif cls == 'U' and kind == 'UndecidedZero':
            note += '; also accepted: gp\'s value (the undecided constant is nonzero)'
        if rid in ALSO_ACCEPTED:
            note += '; also accepted: %s (%s)' % (ALSO_ACCEPTED[rid][0], ALSO_ACCEPTED[rid][1])
        out.append('\t'.join(['%s:%s' % (cls, rid), f, pt, str(n), want, lim, note]))
    print('# S1 random inputs of review 9 (fuzz.py seeds 1-4, fuzz2.py seeds 11-14; the first %d of each)' % PER_SEED)
    print('# with gp references. Generated by gen_s1_fuzz.py (python3 -I gen_s1_fuzz.py > s1_fuzz.tsv in')
    print('# rustmath-symbolic/tests/data; gp 2.15.4 at realprecision 300 through s1_gp.py; the classes and')
    print('# the hand-reviewed rows are described in the generator). Columns: id (class:seed.index), input,')
    print('# point, order, expectation, limit path, note. Class counts: %s' % ', '.join('%s %d' % kv for kv in sorted(stats.items())))
    for l in out:
        print(l)
    for r in review:
        print(r, file=sys.stderr)
    print('# stats %s' % stats, file=sys.stderr)


# Rows reviewed by hand (s1_fuzz.log lists every candidate): id -> (the expectation the owner's
# rules and the design give, the limit path, the reason, the simplified input for gp or None).
# `value` takes gp's series of the input, or of the simplified input when gp cannot expand the
# exact-zero shape (an exact zero raised to a series, or a series O(h^k) without terms). The E rows
# (gp fails, the prototype gives a value) are left out, except the seven reviewed below (round 3):
# each of those is an exact zero that gp cannot raise to a series or a series without terms, and
# its value is gp's series of the simplified input. The one left out, z1.9 (log(log(K)) with K =
# sin(1)^2+cos(1)^2), is a kernel at an undecided pole (design section 7 item 10).
B1_VALUE = 'B1 (review 9): the argument of the atan-family function is an exact constant (%s), so the value is f(c)'
B2_NL = 'B2 (review 9): arctanh of a non-constant argument with constant term %s, a logarithmic branch point'
B1_EXACT = ('the owner\'s rule at a regular point: %s is an exact constant, so f(c) is the exact constant and '
            'the result is exact (the prototype builds the integrand and gets a truncated O(h^P), then refuses)')
OVERRIDES = {
    'z4.2': ('value', 'Unknown', B1_VALUE % 'sign((x-1)^2) at 1 is 1', None),
    'y11.26': ('value', 'Unknown', B1_VALUE % 'x/x at 0 is 1', None),
    'y11.45': ('value', 'Unknown', B1_VALUE % '3^(0*(1/(x+1/2))) is 1; acos(acosh(1)) = acos(0) = pi/2; acos(pi/2) on the cut', None),
    'y12.8': ('value', 'Unknown', B1_VALUE % 'sign((x+1)^2) at -1 is 1', None),
    'y13.36': ('value', 'Finite(-2/3)', B1_VALUE % 'sign((x+1/2)^2) at -1/2 is 1; sign(asin(1)) = 1', None),
    'y14.0': ('value', 'Finite(1)', B1_VALUE % 'x/x at 0 is 1; acosh(1) = 0 exactly', None),
    'y14.48': ('value', 'Unknown', B1_VALUE % 'sign(x^2+1) is 1', None),
    'z2.29': ('err NotLaurent', 'Unknown', B2_NL % '1 (2(x+1) at -1/2 is 1 + 2h)', None),
    'z2.40': ('err NotLaurent', 'Unknown', B2_NL % '1 (atanh(x) at 1)', None),
    'y12.32': ('err NotLaurent', 'Unknown', B2_NL % '-1 (atanh(x) at -1)', None),
    'y13.20': ('err NotLaurent', 'Unknown', B2_NL % '1 ((x-1) + sign((x-1)^2) at 1 is 1 + h)', None),
    'y14.50': ('err NotLaurent', 'Unknown', B2_NL % '-1 (1 - x at 2 is -1 - h)', None),
    'y13.29': ('err DivisionByZero', 'Unknown', 'B2 control: atanh(sign((x+1)^2)) at -1 is atanh of the exact constant 1, a pole of the constant (gp: domain error in atanh: x = 1)', None),
    'z2.5': ('err NotLaurent', 'Unknown', 'asin(cosh(h)^2) = asin(1 + h^2 + ...): 1 - b^2 = -2h^2 + ..., the square-root branch point of valuation 2; the principal function is pi/2 - I*sqrt(2)*|h| + ..., not Laurent (gp gives a formal one-sided series)', None),
    'y12.39': ('err NotLaurent', 'Unknown', 'acosh(1 + x^2) at 0 has no Laurent expansion (sqrt(2)*|x| + ...); the engine expands each subexpression on its own, so the product with an exact zero factor is still NotLaurent (design section 2.9; gp gives the formal 0)', None),
    'z3.7': ('value', 'Finite(0)', B1_EXACT % 'the argument of asin, (sin(1)^2+cos(1)^2)*(0*(1/(x-2))),', None),
    'y14.59': ('value', 'Finite(0)', B1_EXACT % 'the argument of asinh, x - x,', None),
    'z1.43': ('value', 'Finite(0)', B1_EXACT % 'x - x (atanh(0) = 0, abs(0) = 0)', '0'),
    'z2.37': ('value', 'Finite(0)', B1_EXACT % '(x-x)*(1-x) (atan(0) = 0, abs(0) = 0)', '0'),
    'z3.20': ('value', 'Unknown', B1_EXACT % 'x - x (asin(0) = 0 and 0^(2+x) = 0 at 0, Re r(0) = 2 > 0)', '(1/2)^(atan((1-x)))'),
    'y11.17': ('value', 'Finite(0)', B1_EXACT % '0*(1/x) (asinh(0) = 0, so the factor 0^(2+x) is 0 at 0)', '0'),
    # an UndecidedZero of the integrand of an exact-constant operand (testcheck T1): the owner's rule
    # forms no integrand, so the value is the kernel
    'y14.33': ('value', 'Unknown', B1_VALUE % 'sign((x+1/2)^2) at -1/2 is 1, so atanh gets the exact 3 and atan the exact constant arctanh(3); the prototype tested 1 + arctanh(3)^2 for zero in the integrand', None),
    'y13.19': ('value', 'Unknown', B1_VALUE % '(1/2)^(sign(x^2+1)) at -1 is the exact 1/2 by the exponent rule of design section 2.5, so asin gets the exact constant 2*arcsin(pi); the prototype tested 1 - 4*arcsin(pi)^2 for zero in the integrand', None),
    # rows whose outcome depends on a fold the scope of the owner's rule leaves optional (testcheck
    # T2; tests/s1_common/mod.rs): every honest outcome is accepted
    'y13.36': (['value', 'err Unsupported'], ['Finite(-2/3),Unknown', 'Unknown'],
               'optional fold: asin gets the exact 1 (sign((x+1/2)^2) at -1/2), so asin(1) is evaluated (B1, mandatory: never DivisionByZero); sign of the exact non-rational constant arcsin(1) is Unsupported (design section 2.4) or, if the port folds sign of an exact constant, 1 (the value -2/3, and the limit path Finite(-2/3), or Unknown when the constant term stays the kernel (-2/3)/sign(arcsin(1)))', None),
    'z2.10': (['value', 'err Unsupported'], ['Unknown', 'Unknown'],
              'optional fold: sign of the exact non-rational constant 1/(sin(1)^2+cos(1)^2) is Unsupported (design section 2.4) or, if the port folds sign of an exact constant, 1 (the value (2*5)^(-1/2) = 1/sqrt(10))', None),
    'y11.18': (['value', 'err PrecisionExhausted'], ['Unknown', 'Unknown'],
               'optional fold: sign((x+1)^2) at -1 is the exact 1, and 1^x = exp(x*log(1)) is exp of the exact 0, the exact 1 only if exp of an exact constant is folded; then asin(1) = pi/2 (B1) and the value asinh(pi/2), else asin(1 + O(h^P)) is PrecisionExhausted (gp segfaults on the input; the value is gp on asinh(asin(1)))', 'asinh(asin(1))'),
    # E rows (gp fails on the exact-zero shape, the prototype gives a value; round 3, testcheck n6):
    # the value is gp's series of the simplified input
    'z3.22': ('value', 'Finite(0)', 'E: (x + Z)^2 * (Z*(x-1))^((x-x)+2) with Z = (x+1)^2-x^2-2*x-1 the exact 0: '
              'the base Z*(x-1) is the exact 0 and the exponent the exact 2, so the power is 0, the product '
              'the exact 0 and its square root 0 (gp: irrational exponent of a zero series)', '0'),
    'z4.37': ('value', 'Finite(0)', 'E: the base 0*(1/(x-1)) is the exact 0 and Re(1 + sin(1)) > 0, so 0^(sin(x)+1) '
              'is the exact 0 (design section 2.5 zero_pow_sign), the quotient by 2x - 1 the exact 0 and '
              'asin of it 0 (gp: gpow(0, t_SER))', '0'),
    'z4.58': ('value', 'Finite(0),Unknown', 'E: (x-x)^(sin(x)+1) at 2 is the exact 0 (Re(1 + sin(2)) > 0) and sin '
              'of it 0; the limit path is Finite(0), or Unknown if the port keeps sin(0) as a kernel (an '
              'optional fold) (gp: irrational exponent of a zero series)', '0'),
    'z4.41': ('value', 'Finite(2)', 'E: (x-2)^((x-x)+2) is (x-2)^2 by the exact exponent 2, so the first term is '
              'sqrt((x-2)^2 + 2x) = 2 + h/2 + ...; the second is the exact 0 (the factor x*0 + (x-x)) times '
              'sqrt(Z) - asinh(x), with sqrt of the exact 0 equal to 0 (gp: irrational exponent of a zero '
              'series)', 'sqrt((x-2)^2 + 2*x)'),
    'z4.53': ('value', 'Unknown', 'E: (2x - 3/2)/(3x + 1) - acos(abs(-1/2*(x-x))) at -1/2; abs of the exact 0 is 0 and '
              'acos(0) = pi/2, so the value is 5 - pi/2 + 26h (gp: polcoef of an empty series)',
              '(2*x - 3/2)/(3*x + 1) - acos(0)'),
    'y11.24': ('value', 'Unknown', 'E: (0*(1/(x-1/3)))^((x-x)+2) is 0^2 = 0 and asin(x-x) = asin(0) = 0, so the '
               'value is (1+sin(1))^pi - x*(1-x) (gp: gpow(0, t_SER))', '(1+sin(1))^(pi) - x*(1-x)'),
    'y14.58': ('value', 'Finite(1),Unknown', 'E: sign of the exact 0 is 0 and cosh(0) = 1; the limit path is '
               'Finite(1), or Unknown if the port keeps cosh(0) as a kernel (an optional fold) (gp: polcoef '
               'of an empty series)', '1'),
    'y13.0': (['err DivisionByZero', 'err PrecisionExhausted'], ['Unknown', 'Unknown'],
              'optional fold: 1/(atan(x)*tan(x-x)) at -1/2; tan of the exact 0 is the exact 0 only if tan of an exact constant is folded (then the input divides by the exact 0, DivisionByZero), else tan(0 + ...) is a truncated O(h^P) and the inverse is PrecisionExhausted', None),
}

# Candidates of B1 by an UndecidedZero of an integrand factor whose operand is not an exact constant
# (reviewed by hand: the integrand is needed, and its factor at the complex or kernel constant term
# is honestly undecided): id -> the reason. They keep their automatic class.
INTEGRAND_NOT_CONSTANT = {
    'y11.4': 'acos(asin(atanh(2x))) at 2: atanh gets 4 + 2h, so the operand of asin is atanh(4) + 2h/(1-16) + ..., not constant',
    'y11.25': 'acos(pi^x * x * pi) at -1: the operand is -pi*exp(-log(pi)) + ... h, not constant (its constant term is -1, which the constant field cannot decide, so the branch point is not seen)',
    'y11.42': 'acos(log(-1/2) - x*(x/x)) at 0: x/x at 0 is the exact 1, the operand log(-1/2) - h is not constant',
    'y12.1': 'acos((x(x^2+1))^(2+x)) at -1: base and exponent vary, the operand is not constant',
    'y13.27': 'atanh(acos((1/2)^x)) at -1/2: (1/2)^x varies, so both operands vary',
    'y14.10': 'asinh(x*(x-0) + log(-1/2)) at 0: the operand log(-1/2) + h^2 is not constant',
    'y14.52': 'asin((1+sin(1))^(acos(2x+1) - asin(x))) at 2: the exponent varies, the operand is not constant',
    'z4.5': 'acos((1/2)^(x^(1/3)) / sqrt((1-x)^(2/3))) at -1/2: the operand varies',
}

# R rows with a second honest refusal kind (round 3, testcheck n7 and n8): id -> (the second
# outcome, the reason). The prototype's kind stays accepted.
ALSO_ACCEPTED = {
    'y11.25': ('err NotLaurent', 'the operand of acos is -1 + (1 - log(pi)) h + ..., not constant, '
               'with constant term -pi*exp(-log(pi)) = -1: a branch point of a non-constant argument; '
               'the design\'s constant field cannot simplify exp(-log(pi)) to 1/pi, a port that can '
               'gives NotLaurent'),
    'y11.42': ('err NotLaurent', 'two causes: the left factor\'s integrand factor 1 - log(-1/2)^2 is '
               'a nonzero constant that the design\'s real-only enclosures cannot decide '
               '(UndecidedZero), and the right factor ((2x)^(-1))^x = exp(x log(1/(2x))) at 0 has a '
               'h log h term (NotLaurent); a port that decides the first, or meets the second first, '
               'gives NotLaurent'),
    'y13.13': ('err NotLaurent', 'two causes: the numerator is sign of a series with non-rational '
               'coefficients (Unsupported, design section 2.4) and the denominator holds acos(x) at 1, '
               'a branch point of a non-constant argument (NotLaurent); the kind follows the '
               'evaluation order'),
    'z2.56': ('err DivisionByZero', 'acos of the exact constant sin(1)^2+cos(1)^2, which is 1: '
              'acos(1) = 0, so the input raises the exact 0 to the power -1; the design\'s constant '
              'field cannot decide sin(1)^2+cos(1)^2 = 1 (UndecidedZero), a port that can gives '
              'DivisionByZero'),
}

# U rows whose UndecidedZero no longer accepts gp's value (S1 request C-1, a real variable): the
# two-sided oracle (python3 -I s1_gp.py --two-sided s1_fuzz.tsv) shows that gp's formal series is
# not the function on one side of the point, so the input has no Laurent expansion there; the
# second accepted outcome is NotLaurent: id -> (the outcome, the reason with the gp evidence).
C1_NOT_A_VALUE = {
    'y12.1': ('err NotLaurent', 'S1 request C-1: the operand of acos is (x(x^2+1))^(2+x) = '
              '(-2 + 4h + ...)^(1 + h) at -1, with constant term -2 on the cut of acos and the '
              'coefficient of h, 4 - 2 log 2 - 2 pi I (gp), not real, so it leaves the cut; gp, two-sided at '
              'realprecision 300: the formal series is the function at -1 - 1e-16 (error 1.4e-14) and '
              'not at -1 + 1e-16 (error 73.4); the engine cannot decide that exp(log(-2)) is real, '
              'so its message says that it cannot decide'),
}

# Rows left out, with the reason (none since the multi-outcome rows: y11.18 and y13.0, left out
# before because two outcomes are honest, are now O rows of OVERRIDES).
EXCLUDED = {}

if __name__ == '__main__':
    main()
