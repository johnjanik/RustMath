# Generator of tests/data/s1_cuts.tsv (S1 request C-1 of docs/workstreams/m23_golay/REQUESTS.md,
# section 2: an argument that leaves a branch cut). Run from rustmath-symbolic/tests/data:
#     python3 -I gen_s1_cuts.py > s1_cuts.tsv
# It calls gp (PARI/GP 2.15.4, /usr/bin/gp) through s1_gp.py at realprecision 300; coefficients
# are written with 70 significant digits and compared to 60 by tests/s1_cuts.rs. The rows are then
# checked by the two-sided oracle of s1_gp.py (python3 -I s1_gp.py --two-sided s1_cuts.tsv), whose
# per-row result is in the note column.
#
# The rule (series.rs, module conventions): the variable is real. At an application of a branch
# function (log, a non-integer power, arcsin, arccos, arctan, arcsinh, arccosh, arctanh) whose
# argument has its constant term c0 on an open branch cut, the engine expands only when it decides
# that every other coefficient of the argument is a real multiple of the cut's direction (real for
# the cuts on the real axis, purely imaginary for those of arctan and arcsinh on the imaginary
# axis). Then the argument stays on the cut and the expansion through the principal value at c0 is
# the function. A decided coefficient off that direction is NotLaurent ("leaves a branch cut");
# an undecided real constant is UndecidedZero. The open cuts: (-oo, 0) for log and the powers;
# (-oo, -1) and (1, oo) for arcsin, arccos and arctanh; (-oo, -1) and (-1, 1) for arccosh;
# (-I oo, -I) and (I, I oo) for arctan and arcsinh.
#
# Rows, in this order:
#   c1:1 to c1:5    the five inputs of the C-1 table (NotLaurent), then
#   c1m:1 to c1m:5  their mirror images I -> -I (NotLaurent). Four of the mirrors (log, sqrt,
#                   arctanh, arcsin) leave the cut to the side that the principal value at c0 is
#                   continuous with, so gp's formal series is the function on both sides (the
#                   oracle's note says so): the rule refuses them anyway (a finer rule may expand
#                   them later); log(-1 - I*x) crosses the cut.
#   ctl:*           real-coefficient controls on each open cut of each function (a value, checked
#                   against gp's series and by the two-sided oracle), the four of the request first;
#   off:*           complex arguments whose constant term is off every cut (a value);
#   lv:*            arguments that leave a cut (NotLaurent), one or more per function and cut;
#   und:*           an undecided real constant (UndecidedZero);
#   pg:*, cls:*     (C-1 minor items) real arguments whose class is decided by the structure or
#                   the enclosure of a constant: a variable exponent of a positive base, a
#                   truncated argument scaled by a constant real by its enclosure (a value).
#   enc:*           (C-1 minors check, m1) products scaled by a constant real or imaginary only by
#                   its enclosure, some levels below the branch function whose cut check reads
#                   their class (ten values), and two refusals (NotLaurent).
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_gp as g

NL = 'err NotLaurent'
UZ = 'err UndecidedZero'

ROWS = [
    # the C-1 table and its mirror images
    ('c1:1', 'log(-1 - I*x^2)', '0', 3, NL),
    ('c1:2', 'sqrt(-1 - I*x^2)', '0', 3, NL),
    ('c1:3', 'log(-1 + I*x)', '0', 3, NL),
    ('c1:4', 'arctanh(2 + I*x^2)', '0', 3, NL),
    ('c1:5', 'arcsin(3/2 + I*x^2)', '0', 3, NL),
    ('c1m:1', 'log(-1 + I*x^2)', '0', 3, NL),
    ('c1m:2', 'sqrt(-1 + I*x^2)', '0', 3, NL),
    ('c1m:3', 'log(-1 - I*x)', '0', 3, NL),
    ('c1m:4', 'arctanh(2 - I*x^2)', '0', 3, NL),
    ('c1m:5', 'arcsin(3/2 - I*x^2)', '0', 3, NL),
    # the controls of the request
    ('ctl:log:1', 'log(-1 - x^2)', '0', 3, 'value'),
    ('ctl:sqrt:1', 'sqrt(-4 + x)', '0', 3, 'value'),
    ('ctl:arcsin:1', 'arcsin(2 + x)', '0', 3, 'value'),
    ('ctl:arctanh:1', 'arctanh(2 - x^2)', '0', 3, 'value'),
    # more controls: each function, each open cut, exact and truncated arguments
    ('ctl:log:2', 'log(-2 + x)', '0', 3, 'value'),
    ('ctl:log:3', 'log(-1/2 + sin(x))', '0', 3, 'value'),
    ('ctl:log:4', 'log(-pi + x^2)', '0', 3, 'value'),
    ('ctl:log:5', 'log(x)', '-1', 3, 'value'),
    ('ctl:log:6', 'log(-1 - x)', '0', 3, 'value'),
    ('ctl:log:7', 'log(-2 + x*I*log(-1))', '0', 3, 'value'),
    ('ctl:sqrt:2', 'sqrt(-2 - x^3)', '0', 3, 'value'),
    ('ctl:sqrt:3', 'sqrt(-3 + exp(x))', '0', 3, 'value'),
    ('ctl:pow:1', '(-8 + x)^(1/3)', '0', 3, 'value'),
    ('ctl:pow:2', '(-1 - x^2)^(2/3)', '0', 3, 'value'),
    ('ctl:pow:3', '(-2 + x)^(-1/2)', '0', 3, 'value'),
    ('ctl:pow:4', '(-3 + x)^pi', '0', 3, 'value'),
    ('ctl:pow:5', '(-1 + x^2)^I', '0', 3, 'value'),
    ('ctl:pow:6', '(-2 + x)^(1/2 + I)', '0', 3, 'value'),
    ('ctl:powvar:1', '(-2 + x)^(1 + x)', '0', 3, 'value'),
    ('ctl:powvar:2', '(-1 - x^2)^(2 + sin(x))', '0', 3, 'value'),
    ('ctl:arcsin:2', 'arcsin(-3/2 - x^2)', '0', 3, 'value'),
    ('ctl:arcsin:3', 'arcsin(3/2 + sin(x))', '0', 3, 'value'),
    ('ctl:arccos:1', 'arccos(2 + x)', '0', 3, 'value'),
    ('ctl:arccos:2', 'arccos(-2 + x^3)', '0', 3, 'value'),
    ('ctl:arctanh:2', 'arctanh(-3 + x)', '0', 3, 'value'),
    ('ctl:arctanh:3', 'arctanh(-2 + sin(x))', '0', 3, 'value'),
    ('ctl:arccosh:1', 'arccosh(-2 + x)', '0', 3, 'value'),
    ('ctl:arccosh:2', 'arccosh(1/2 + x)', '0', 3, 'value'),
    ('ctl:arccosh:3', 'arccosh(-1/2 - x^2)', '0', 3, 'value'),
    ('ctl:arccosh:4', 'arccosh(sinh(x))', '0', 3, 'value'),
    ('ctl:arctan:1', 'arctan(2*I + I*x)', '0', 3, 'value'),
    ('ctl:arctan:2', 'arctan(2*I + I*sin(x))', '0', 3, 'value'),
    ('ctl:arctan:3', 'arctan(-3*I + I*x^2)', '0', 3, 'value'),
    ('ctl:arctan:4', 'arctan(2*I*sqrt(1 + x))', '0', 3, 'value'),
    ('ctl:arcsinh:1', 'arcsinh(2*I + I*x)', '0', 3, 'value'),
    ('ctl:arcsinh:2', 'arcsinh(-2*I - I*x^3)', '0', 3, 'value'),
    ('ctl:arcsinh:3', 'arcsinh(-2*I + I*tan(x))', '0', 3, 'value'),
    # complex arguments off every cut
    ('off:1', 'log(1 + I*x)', '0', 3, 'value'),
    ('off:2', 'log(-1 + I + x)', '0', 3, 'value'),
    ('off:3', 'arcsin(1/2 + I*x)', '0', 3, 'value'),
    ('off:4', 'arctan(1/2*I + x)', '0', 3, 'value'),
    ('off:5', 'sqrt(x)', 'I', 3, 'value'),
    ('off:6', 'arctanh(2*I + x)', '0', 3, 'value'),
    ('off:7', 'arccosh(2 + I*x)', '0', 3, 'value'),
    ('off:8', 'arcsinh(1/2*I + x)', '0', 3, 'value'),
    ('off:9', 'log(sqrt(-4) + I*x)', '0', 3, 'value'),
    ('off:10', 'log(log(-2) + I*x)', '0', 3, 'value'),
    # arguments that leave a cut
    ('lv:log:1', 'log(-2 + I*x^3)', '0', 3, NL),
    ('lv:log:2', 'log(-pi + I*x)', '0', 3, NL),
    ('lv:log:3', 'log(I*x)', 'I', 3, NL),
    ('lv:log:4', 'log(-1/2 + I*sin(x))', '0', 3, NL),
    ('lv:log:5', 'log(-1 + x + I*x^2)', '0', 3, NL),
    ('lv:log:6', '(log(-1 - I*x^2) - log(-1 + I*x^2))/(I*pi)', '0', 3, NL),
    ('lv:sqrt:1', 'sqrt(-4 + I*x)', '0', 3, NL),
    ('lv:pow:1', '(-8 + I*x)^(1/3)', '0', 3, NL),
    ('lv:pow:2', '(-1 + I*x^2)^I', '0', 3, NL),
    ('lv:pow:3', '(-2 - I*x)^pi', '0', 3, NL),
    ('lv:powvar:1', '(-2 + I*x)^(1 + x)', '0', 3, NL),
    ('lv:arcsin:1', 'arcsin(-2 + I*x)', '0', 3, NL),
    ('lv:arccos:1', 'arccos(3/2 + I*x^3)', '0', 3, NL),
    ('lv:arctanh:1', 'arctanh(-2 - I*x)', '0', 3, NL),
    ('lv:arccosh:1', 'arccosh(-2 + I*x)', '0', 3, NL),
    ('lv:arccosh:2', 'arccosh(1/2 + I*x^2)', '0', 3, NL),
    ('lv:arctan:1', 'arctan(2*I + x)', '0', 3, NL),
    ('lv:arctan:2', 'arctan(-2*I + x^2)', '0', 3, NL),
    ('lv:arctan:3', 'arctan(2*I + sin(x))', '0', 3, NL),
    ('lv:arctan:4', 'arctan(sqrt(-4) + x)', '0', 3, NL),
    ('lv:arcsinh:1', 'arcsinh(2*I + x^2)', '0', 3, NL),
    ('lv:arcsinh:2', 'arcsinh(-3*I + x)', '0', 3, NL),
    # undecided real constants
    ('und:1', 'log(-1 + I*(sin(1)^2 + cos(1)^2 - 1)*x)', '0', 3, UZ),
    ('und:2', 'arctan(2*I + (sin(1)^2 + cos(1)^2 - 1)*x)', '0', 3, UZ),
    ('und:3', 'log(-1 + I*(sin(1)^2 + cos(1)^2 - 1) + I*x)', '0', 3, UZ),
    # S1 C-1 minor items (appended, so the rows above keep their order)
    # regressions review m3: c0 = sqrt(-1) is purely imaginary by its class, but not written as
    # t*I; its distance from the ends +-I of the cuts is the real constant 1 + c0^2 (0 here, which
    # the constant field does not decide): UndecidedZero, as before C-1 (the integrand's 1 + b^2)
    ('und:4', 'arctan(sqrt(-1) + x)', '0', 3, UZ),
    # regressions review m2: f^g with a variable exponent is real for a real f whose constant term
    # is positive by its structure (pi, sqrt(2), e) or by its enclosure (1 + sin(1)), and a real g
    ('pg:1', 'log(-5 + (pi + x)^x)', '0', 3, 'value'),
    ('pg:2', 'sqrt(-4 + (pi + x)^x)', '0', 3, 'value'),
    ('pg:3', 'arctanh(x^(2*(1+x)/(1+x)))', 'sqrt(2)', 3, 'value'),
    ('pg:4', 'arcsin(x^(1 + (x - e)))', 'e', 3, 'value'),
    ('pg:5', 'log(-3 + (1 + sin(1) + x)^x)', '0', 3, 'value'),
    # semantics review m3: a truncated argument scaled by a constant that is real only by its
    # enclosure (log(-1)^2 = -pi^2) keeps the real class
    ('cls:1', 'arccos(-3 + log(-1)^2*sinh(x))', '0', 3, 'value'),
    ('cls:2', 'log(-1 + log(-1)^2*sin(x))', '0', 3, 'value'),
    # S1 C-1 minors check, m1 (appended): a product scaled by a constant real or imaginary only by
    # its enclosure (log(-1)^2 = -pi^2, I*log(-1)^2 = -pi^2 I), read by a cut check of a branch
    # function some levels above it: under exp, a power, the exponent of a constant base, another
    # product, a negation, cos, erf; on the imaginary axis's cuts; and two refusals (a constant
    # complex by its enclosure, log(1 + I); a real class on an imaginary-axis cut)
    ('enc:1', 'log(-1 + exp(log(-1)^2*sin(x)) - 1)', '0', 3, 'value'),
    ('enc:2', '(-1 + log(-1)^2*sin(x))^(1/3)', '0', 3, 'value'),
    ('enc:3', '(-1 + log(-1)^2*sin(x))^(1/2 + x)', '0', 3, 'value'),
    ('enc:4', 'log(-3 + 2^(log(-1)^2*sin(x)))', '0', 3, 'value'),
    ('enc:5', 'log(-1 + sin(x)*(log(-1)^2*sin(x) + x))', '0', 3, 'value'),
    ('enc:6', 'log(-2 + cos(-log(-1)^2*x))', '0', 3, 'value'),
    ('enc:7', 'arctan(2*I + I*log(-1)^2*sin(x))', '0', 3, 'value'),
    ('enc:8', 'arcsinh(-3*I + I*log(-1)^2*x^2)', '0', 3, 'value'),
    ('enc:9', 'log(-1 + erf(log(-1)^2*x))', '0', 3, 'value'),
    ('enc:10', 'arccosh(1/2 + log(-1)^2*sin(x)^2)', '0', 3, 'value'),
    ('enc:11', 'log(-1 + log(1 + I)*sin(x))', '0', 3, NL),
    ('enc:12', 'arctan(2*I + log(-1)^2*sin(x))', '0', 3, NL),
]

# the limit path where gp's floating constant term hides a rational one (the engine's constant
# term (-2)^1 and (-1)^2 is exact: Finite only for a rational constant term, design 2.13)
LIMITS = {
    'ctl:powvar:1': 'Finite(-2)',
    'ctl:powvar:2': 'Finite(1)',
}

NOTES = {
    'c1:1': 'C-1 table: the series is off by 2 pi I on both sides',
    'c1:2': 'C-1 table: the series has the wrong sign on both sides',
    'c1:3': 'C-1 table: the series is the function for x0 > 0 only (2 pi I at -x0)',
    'c1:4': 'C-1 table: the series is off by pi I on both sides',
    'c1:5': 'C-1 table: the series is off by 1.92 I on both sides',
    'c1m:1': 'mirror of c1:1',
    'c1m:2': 'mirror of c1:2',
    'c1m:3': 'mirror of c1:3',
    'c1m:4': 'mirror of c1:4',
    'c1m:5': 'mirror of c1:5',
    'lv:log:6': 'the limit path read Finite(0) before C-1; the true limit is -2',
    'und:4': 'C-1 minor items: NotLaurent (cannot decide) before',
    'pg:1': 'C-1 minor items: NotLaurent (cannot decide) before',
    'pg:2': 'C-1 minor items: NotLaurent (cannot decide) before',
    'pg:3': 'C-1 minor items: NotLaurent (cannot decide) before',
    'pg:4': 'C-1 minor items: NotLaurent (cannot decide) before',
    'pg:5': 'C-1 minor items: NotLaurent (cannot decide) before',
    'cls:1': 'C-1 minor items: NotLaurent (cannot decide) before',
    'cls:2': 'C-1 minor items: NotLaurent (cannot decide) before',
}


def limit_of(v, coefs):
    """The limit path's answer for a value: Finite(q) for a rational constant term (valuation
    >= 0), Finite(0) for a positive valuation, else Unknown (series.rs, design 2.13)."""
    if v > 0:
        return 'Finite(0)'
    for (k, re_, im_) in coefs:
        if k == 0:
            if im_ == '0' and '.' not in re_ and 'e' not in re_.lower():
                return 'Finite(%s)' % re_
            return 'Unknown'
    return 'Unknown'


def main():
    res = g.series_batch([(t, p, n) for (_, t, p, n, _) in ROWS])
    jobs = []
    for (rid, t, p, n, want), r in zip(ROWS, res):
        jobs.append((t, p, n, None))
    two = g.two_sided_batch(jobs)
    print('# S1 request C-1 (docs/workstreams/m23_golay/REQUESTS.md, section 2): arguments on a branch')
    print('# cut. Generated by gen_s1_cuts.py (python3 -I gen_s1_cuts.py > s1_cuts.tsv, in')
    print('# rustmath-symbolic/tests/data; gp 2.15.4 at realprecision 300 through s1_gp.py). Columns: id,')
    print('# input, point, order, expectation (value v k:re:im ... from exponent v up to the order, or')
    print('# err Kind), limit path, note (gp\'s own outcome, and the two-sided oracle on gp\'s formal series).')
    bad = 0
    for (rid, t, p, n, want), r, tw in zip(ROWS, res, two):
        if tw['err'] is not None or len(tw['sides']) != 2:
            side = 'two-sided: gp error ' + (tw['err'] or 'incomplete')[:80]
        else:
            wrong = [s for s in tw['sides'] if s[4] > s[3]]
            if wrong:
                side = 'two-sided: the formal series is not the function at %s' % ' and '.join(
                    '%sx0 (error %.2e relative to max(1, |f|))' % ('' if s[0] > 0 else '-', s[4]) for s in wrong)
            else:
                side = 'two-sided: the formal series is the function at x0 and -x0'
        note = NOTES.get(rid, '')
        if want == 'value':
            if r[0] != 'ok':
                print('# GP FAILED for %s: %s' % (rid, r[1]), file=sys.stderr)
                bad += 1
                continue
            _, v, coefs = r
            w = 'value %d %s' % (v, ' '.join('%s:%s' % (re_, im_) for (_, re_, im_) in coefs))
            lim = LIMITS.get(rid, limit_of(v, coefs))
            note = '; '.join(x for x in [note, 'gp: series', side] if x)
        else:
            w = want
            lim = 'Unknown'
            gpo = 'gp: a formal series (valuation %d)' % r[1] if r[0] == 'ok' else 'gp: ' + r[1][:90]
            note = '; '.join(x for x in [note, gpo, side] if x)
        print('\t'.join([rid, t, p, str(n), w, lim, note]))
    if bad:
        print('%d value rows without a gp reference' % bad, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
