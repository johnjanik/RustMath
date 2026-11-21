# Generator of tests/data/s1_singular_points.tsv (the singular-point matrix of the S1 owner decision,
# 2026-11-15). Run from rustmath-symbolic/tests/data:
#     python3 -I gen_s1_singular.py > s1_singular_points.tsv
# It calls gp (PARI/GP 2.15.4, /usr/bin/gp) through s1_gp.py, at realprecision 300; coefficients are
# written with 70 significant digits and compared to 60 by tests/s1_singular_points.rs.
#
# For every inverse function of the design (section 2.4, the atan family: arcsin, arccos, arctan,
# arcsinh, arccosh, arctanh) and the other branch functions the design treats (log, sqrt, the
# non-integer powers u^(1/3), u^(2/3), u^(-1/2), u^pi; section 2.4 "Non-integer powers" and the
# Pow rules of section 2.5), and every finite branch point c of each:
#   arcsin, arccos, arccosh at c = +-1 (square-root branch points);
#   arctan at c = +-I and arctanh at c = +-1 (logarithmic: arctan z = (1/(2I)) log((1+Iz)/(1-Iz)),
#     arctanh z = (1/2) log((1+z)/(1-z)); f(c) is a pole of the constant);
#   arcsinh at c = +-I (square-root: arcsinh z = log(z + sqrt(1+z^2)));
#   log, sqrt and the powers at c = 0;
#   the power with a variable exponent, l^r = exp(r log l) (design section 2.5: a base whose series
#     is the exact zero takes the rule 0^r, any other base the rule exp(r log l)), at l = 0, with
#     the exponents r = x + 1 and r = 2 + sin(x) (Re r > 0 at every point of the matrix, so 0^r = 0);
#     families pow(x+1) and pow(2+sin(x));
#   the powers with a complex constant exponent, u^I, u^(1/2+I), u^(-1/2+I), at u = 0 (families
#     powI, pow(1/2+I), pow(-1/2+I); round 3, testcheck n2). For the exact zero, design section 2.5
#     `zero_pow_sign` decides 0^r from the sign of Re r: Re r > 0 gives the exact 0, Re r < 0
#     DivisionByZero ("0 to a power with negative real part"), and Re r = 0 with a decided imaginary
#     part NotLaurent (0^I has no value and no Laurent expansion: |h^I| = 1 and its argument turns
#     without limit). So the (a) rows of powI are the third outcome of an exact-constant argument,
#     NotLaurent, the only (a) family with it. A base of positive valuation is NotLaurent (design
#     section 2.4, "non-integer powers": `x^(1/2 + I)` at 0 is NotLaurent).
# the matrix has
#   (a) arguments whose series is the exact constant c, in several shapes (c itself, c*x^0 at 0 and
#       at 2, c*(x/x) at 0, c*((x-2)/(x-2)) at 2, c*sign(x^2+1) at 0 and at -1/2, c + 0*x, c plus the
#       exact zero (x+1)^2-x^2-2*x-1, and the sums x + f(c*x^0) and x + 0*f(c*x^0); round 3,
#       testcheck n3, three more producers of an exact constant at 0: abs of a rational series,
#       c*abs(-x^0) (design section 2.4: abs of a series with rational coefficients and even
#       valuation is sign(c0) b, so abs(-1) is the exact 1); a power with an exact-constant
#       exponent, c*x^(x-x) (section 2.5: the exponent x - x is the exact 0, so the power takes the
#       constant-exponent rule, and section 2.4 pow_int rule 1 gives the exact 1, although the base
#       x has valuation 1 at 0); and a nested inverse function,
#       c + arcsin(x-x) (the owner's rule: arcsin of the exact 0 is the exact 0); for c = 0 the
#       shapes abs(x^0 - 1), x^(x-x) - 1 and arcsin(x-x)). The owner's rule:
#       the value f(c), no expansion and no integrand. Expected: the value (gp: f(c) evaluated on the
#       simplified constant, since gp's own series of these shapes is unreliable: acosh(-X^0) with
#       X^0 a t_POL gives O(h^8), and asin((X-1)/(X-1)) is a gp error), or, where f(c) is a pole of
#       the constant (arctan(+-I), arctanh(+-1), log(0), 0^(-1/2)), the design's typed answer for a
#       pole of a constant, DivisionByZero (design section 2.2 `apply`; review 9 B1 required change 1
#       and m2 for arctan(+-I); the input divides by an exact zero there).
#   (b) arguments with constant term c that are not constant (c + x, c - x^2, c + x^3, c + sin(x),
#       c*exp(x) at 0; x at the point c; c*x^2 at 1; x + 0*f(c + x) at 0). Expected: NotLaurent (the
#       owner's rule, never DivisionByZero), the limit path Unknown. Each is a genuine non-Laurent
#       point: the integrand Q of section 2.4 has a non-integer power of a series of positive
#       valuation (square-root type) or a term h^-1 (logarithmic type), and log, sqrt and the
#       non-integer powers see an argument of positive valuation, and the powers with a variable
#       exponent take log of a base of positive valuation (x^(x+1) = exp((x+1) log x) at 0).
#   (c) regular points next to each branch point, c + 1/2 and c - 1/2: f(x) there, order 3, against
#       gp's series (points on a branch cut included: arcsin and arccos at +-3/2, arccosh at 1/2,
#       -1/2, -3/2, arctanh at +-3/2, log, sqrt and the powers at -1/2; the engine expands the branch
#       whose value at the point is principal, section 2.4 "Branch at the point").
#   (d) two more shapes of a branch point (S1 implementation review 1, semantics lens, m5), after
#       all other rows so that their ids and order do not change: the branch point at infinity,
#       f(1/x) at 0 (an argument with a pole), once per function; and non-constant arguments whose
#       constant term is c only through an exact kernel value (sin(pi/2) = 1, sin(pi/6) = 1/2,
#       cos(pi) = -1, cosh(0) = 1, sin(pi) = 0, cos(pi/2) = 0): sin(pi/2 + x), 2*sin(pi/6 + x),
#       -sin(pi/2 + x), cos(pi + x), c*sin(pi/2 + x) and c*cosh(x) for c = +-I, at 0, and sin(x) at
#       pi, cos(x) at pi/2 for c = 0. Expected: NotLaurent (the owner's rule), the limit path
#       Unknown.
#   (e) three guard rows (S1 implementation review 2, semantics lens, minor 6), after the (d) rows:
#       arcsin of 1^(1+x), (-1)^(1+x) and sin(1)^2+cos(1)^2+x at 0, each an honest typed refusal
#       (PrecisionExhausted, NotLaurent, UndecidedZero), never DivisionByZero.
#   (f) S1 fix-up round 4 (review 4, semantics lens, m1), after the (e) rows: 1/(3 + f(c x^0)),
#       1/(x + 3 + f(c)), log(x + 3 + f(c)) and sqrt(x + 3 + f(c)) at 0 for every pair with a
#       finite f(c): the value (the exact f(c) is decided nonzero in the sum).
#   (g) S1 fix-up round 5 (review 5, semantics lens, m1, and the owner's first rule), after the (f)
#       rows: one pair per inverse function whose argument reaches a branch point only through a
#       value folded in round 4 (NotLaurent for c + x, the value or DivisionByZero for c), and the
#       exact values round 5 adds (arcsin(+-1/2), arccos(+-1/2), arccosh(+-1/2), arcsinh(+-i/2),
#       arctanh(+-i)) inverted at 0, and divided into x minus the value (DivisionByZero).
#   (h) S1 finish under the narrowed bar (review 6, semantics lens, m2), after the (g) rows: two
#       truncated-constant arguments per pair, c*(1+x)/(1+x) and c + (sin(x)^2+cos(x)^2-1) at 0
#       (for c = 0, (1+x)/(1+x) - 1 and sin(x)^2+cos(x)^2-1): PrecisionExhausted, never a value,
#       NotLaurent or DivisionByZero.
import sys
import os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s1_gp as g

# f(c) has no Laurent expansion: the (a) rows expect NotLaurent (only 0^I, design section 2.5)
NOT_LAURENT = 'NotLaurent'

# (function template, list of (branch point, exact value of f(c) as gp text, None for a pole, or
#  NOT_LAURENT, whether f(c) is rational))
FUNCS = [
    ('arcsin(%s)', [('1', 'asin(1)', False), ('-1', 'asin(-1)', False)]),
    ('arccos(%s)', [('1', 'acos(1)', True), ('-1', 'acos(-1)', False)]),
    ('arctan(%s)', [('I', None, False), ('-I', None, False)]),
    ('arcsinh(%s)', [('I', 'asinh(I)', False), ('-I', 'asinh(-I)', False)]),
    ('arccosh(%s)', [('1', 'acosh(1)', True), ('-1', 'acosh(-1)', False)]),
    ('arctanh(%s)', [('1', None, False), ('-1', None, False)]),
    ('log(%s)', [('0', None, False)]),
    ('sqrt(%s)', [('0', '0', True)]),
    ('(%s)^(1/3)', [('0', '0', True)]),
    ('(%s)^(2/3)', [('0', '0', True)]),
    ('(%s)^(-1/2)', [('0', None, False)]),
    ('(%s)^pi', [('0', '0', True)]),
    ('(%s)^(x+1)', [('0', '0', True)]),
    ('(%s)^(2+sin(x))', [('0', '0', True)]),
    # complex constant exponents (round 3, testcheck n2): 0^I is NotLaurent (Re r = 0, Im r decided
    # nonzero), 0^(1/2+I) the exact 0, 0^(-1/2+I) DivisionByZero (design section 2.5 zero_pow_sign)
    ('(%s)^I', [('0', NOT_LAURENT, False)]),
    ('(%s)^(1/2+I)', [('0', '0', True)]),
    ('(%s)^(-1/2+I)', [('0', None, False)]),
]


def scaled(c, u):
    """c times the exact one u (a text), as the engine would read it."""
    if c == '1':
        return u
    if c == '-1':
        return '-' + u
    return '%s*%s' % (c, u)


def exact_shapes(c):
    """(argument, point) pairs whose series is the exact constant c."""
    # the last three shapes of each list (round 3, testcheck n3) are abs of a rational series, a power
    # with an exact-constant exponent and a nested inverse function; they come last so that the ids
    # of the earlier rows do not change
    if c == '0':
        return [('0', '0'), ('x^0 - 1', '0'), ('x^0 - 1', '2'), ('x/x - 1', '0'), ('(x-2)/(x-2) - 1', '2'),
                ('sign(x^2+1) - 1', '0'), ('sign(x^2+1) - 1', '-1/2'), ('0*x', '0'), ('x - x', '0'),
                ('(x+1)^2 - x^2 - 2*x - 1', '0'),
                ('abs(x^0 - 1)', '0'), ('x^(x-x) - 1', '0'), ('arcsin(x-x)', '0')]
    return [(c, '0'), (scaled(c, 'x^0'), '0'), (scaled(c, 'x^0'), '2'), (scaled(c, '(x/x)'), '0'),
            (scaled(c, '((x-2)/(x-2))'), '2'), (scaled(c, 'sign(x^2+1)'), '0'),
            (scaled(c, 'sign(x^2+1)'), '-1/2'), ('%s + 0*x' % c, '0'),
            ('%s + ((x+1)^2 - x^2 - 2*x - 1)' % c, '0'),
            (scaled(c, 'abs(-x^0)'), '0'), (scaled(c, 'x^(x-x)'), '0'), ('%s + arcsin(x-x)' % c, '0')]


def nonconstant_shapes(c):
    """(argument, point) pairs with constant term c that are not constant."""
    if c == '0':
        return [('x', '0'), ('-x^2', '0'), ('x^3', '0'), ('sin(x)', '0'), ('x*exp(x)', '0'),
                ('x - 2', '2'), ('x^2 - 1', '1')]
    return [('%s + x' % c, '0'), ('%s - x^2' % c, '0'), ('%s + x^3' % c, '0'), ('%s + sin(x)' % c, '0'),
            (scaled(c, 'exp(x)'), '0'), ('x', c), (scaled(c, 'x^2'), '1')]


def kernel_identity_shapes(c):
    """(argument, point) pairs that are not constant and whose constant term is c only through an
    exact kernel value (class (d))."""
    if c == '0':
        return [('sin(x)', 'pi'), ('cos(x)', 'pi/2')]
    if c == '1':
        return [('sin(pi/2 + x)', '0'), ('2*sin(pi/6 + x)', '0')]
    if c == '-1':
        return [('-sin(pi/2 + x)', '0'), ('cos(pi + x)', '0')]
    return [(scaled(c, 'sin(pi/2 + x)'), '0'), (scaled(c, 'cosh(x)'), '0')]


def neighbours(c):
    if c == '0':
        return ['1/2', '-1/2']
    if c in ('1', '-1'):
        q = int(c)
        return ['%d/2' % (2 * q + 1), '%d/2' % (2 * q - 1)]
    return ['%s + 1/2' % c, '%s - 1/2' % c]


def fname(tmpl):
    return tmpl.split('(')[0] if not tmpl.startswith('(') else 'pow' + tmpl.split('^')[1]


def main():
    rows = []      # (id, input, point, order, kind, gp_text_or_None, lim, note)
    gp_jobs = []   # (row index, text, point, order)
    for tmpl, points in FUNCS:
        fn = fname(tmpl)
        for (c, val, rational) in points:
            cid = '%s@%s' % (fn, c)
            for j, (u, p) in enumerate(exact_shapes(c)):
                for wrap in ('%s', 'x + %s', 'x + 0*%s'):
                    if wrap != '%s' and j not in (1, 7):
                        continue
                    f = wrap % (tmpl % u)
                    rid = 'a:%s:%d%s' % (cid, j, {'%s': '', 'x + %s': '+x', 'x + 0*%s': '+0x'}[wrap])
                    if val is None:
                        rows.append([rid, f, p, 2, 'err DivisionByZero', None, 'Unknown',
                                     'f(c) is a pole of the constant: DivisionByZero (gp: %s is an error)'
                                     % (tmpl % c).replace('arc', 'a')])
                        continue
                    if val == NOT_LAURENT:
                        rows.append([rid, f, p, 2, 'err NotLaurent', None, 'Unknown',
                                     'f(c) = %s has no value and no Laurent expansion (Re r = 0): NotLaurent '
                                     '(design section 2.5 zero_pow_sign; gp: 0^I is an error)' % (tmpl % c)])
                        continue
                    # the value: f(c), plus x for the sums (gp on the simplified constant)
                    if wrap == '%s':
                        ref = '(%s)' % val
                    elif wrap == 'x + %s':
                        ref = '(%s) + x' % val
                    else:
                        ref = 'x'
                    # the limit path (section 2.13): Finite(c0) only for a rational c0; every
                    # rational f(c) of the matrix is 0, and x + 0*f(..) has c0 = 0
                    lim = 'Finite(0)' if (rational or wrap == 'x + 0*%s') else 'Unknown'
                    rows.append([rid, f, p, 2, 'value', ref, lim, 'gp: %s at %s' % (ref, p)])
                    gp_jobs.append((len(rows) - 1, ref, p, 2))
            for j, (u, p) in enumerate(nonconstant_shapes(c)):
                f = tmpl % u
                rows.append(['b:%s:%d' % (cid, j), f, p, 2, 'err NotLaurent', None, 'Unknown',
                             'a branch point of a non-constant argument: NotLaurent, never DivisionByZero'])
                gp_jobs.append((len(rows) - 1, f, p, 2))
            f = 'x + 0*' + (tmpl % ('%s + x' % c if c != '0' else 'x'))
            rows.append(['b:%s:sub' % cid, f, '0', 2, 'err NotLaurent', None, 'Unknown',
                         'a subexpression without a Laurent expansion (design section 2.9 NotLaurent)'])
            gp_jobs.append((len(rows) - 1, f, '0', 2))
            for k, p in enumerate(neighbours(c)):
                f = tmpl % 'x'
                rows.append(['c:%s:%d' % (cid, k), f, p, 3, 'value', f, 'Unknown', 'gp: %s at %s' % (f, p)])
                gp_jobs.append((len(rows) - 1, f, p, 3))
    for tmpl, points in FUNCS:
        fn = fname(tmpl)
        f = tmpl % '1/x'
        rows.append(['d:%s@inf:0' % fn, f, '0', 2, 'err NotLaurent', None, 'Unknown',
                     'the branch point at infinity (an argument with a pole): NotLaurent'])
        gp_jobs.append((len(rows) - 1, f, '0', 2))
        for (c, val, rational) in points:
            cid = '%s@%s' % (fn, c)
            for j, (u, p) in enumerate(kernel_identity_shapes(c)):
                f = tmpl % u
                rows.append(['d:%s:kid%d' % (cid, j), f, p, 2, 'err NotLaurent', None, 'Unknown',
                             'a non-constant argument whose constant term is the branch point '
                             'through an exact kernel value: NotLaurent, never DivisionByZero (gp '
                             'evaluates pi as a float, so its constant term is only near c)'])
                gp_jobs.append((len(rows) - 1, f, p, 2))
    # (e) guard rows (S1 implementation review 2, semantics lens, minor 6), after every other row so
    # that no id changes: shapes of an arcsin argument whose constant term the engine cannot read as
    # the branch point exactly. Each is an honest refusal, never DivisionByZero.
    #   1^(1+x) = exp((1+x) log 1) is the constant 1, but its series through exp of a truncated zero
    #     is truncated, so the value asin(1) needs a decided valuation: PrecisionExhausted (a bounded
    #     search; gp loses the precision too: "serprec 0").
    #   (-1)^(1+x) = -exp(i pi x) is not constant and has the constant term exp(log(-1)) = -1, exact
    #     since fix-up round 2 folds exp(n log q) = q^n: NotLaurent (the owner's rule; review 2 found
    #     UndecidedZero before the fold).
    #   sin(1)^2 + cos(1)^2 + x has the constant term sin(1)^2 + cos(1)^2, which is 1, but the
    #     constant field cannot decide it (design section 2.2): UndecidedZero.
    for j, (f, kind, why) in enumerate([
        ('arcsin(1^(1+x))', 'err PrecisionExhausted',
         'a constant argument whose series is truncated (exp of a truncated 0): a bounded search, '
         'never DivisionByZero'),
        ('arcsin((-1)^(1+x))', 'err NotLaurent',
         'a non-constant argument whose constant term exp(log(-1)) is folded to the branch point '
         '-1: NotLaurent, never DivisionByZero'),
        ('arcsin(sin(1)^2+cos(1)^2+x)', 'err UndecidedZero',
         'a non-constant argument whose constant term is 1 only through an identity the constant '
         'field cannot decide: UndecidedZero, never DivisionByZero'),
    ]):
        rows.append(['e:arcsin:guard%d' % j, f, '0', 2, kind, None, 'Unknown', why])
        gp_jobs.append((len(rows) - 1, f, '0', 2))
    # (f) S1 fix-up round 4 (review 4, semantics lens, m1), after every other row so that no id
    # changes: the value f(c) of an exact-constant argument c*x^0 at a branch point, inverted,
    # logged and rooted at 0, for every pair whose f(c) is finite. The owner's first rule asks for a
    # value, and each input is regular at 0 (3 + f(c) is not 0). The engine folds the exact values
    # (arcsin(+-1) = +-pi/2, arccos(-1) = pi, arccosh(-1) = i pi, arcsinh(+-i) = +-i pi/2), so the
    # zero test decides 3 + f(c); as kernels without a ball enclosure they were UndecidedZero. gp on
    # the simplified constant. The limit path: Finite(1/3) where f(c) = 0 and the constant term is
    # 1/3, else Unknown (a constant term that is not rational, design section 2.13).
    for tmpl, points in FUNCS:
        fn = fname(tmpl)
        for (c, val, rational) in points:
            if val is None or val == NOT_LAURENT:
                continue
            u = scaled(c, 'x^0') if c != '0' else 'x^0 - 1'
            fc = tmpl % u
            for j, (shape, ref) in enumerate([('1/(3 + %s)', '1/(3 + (%s))'),
                                              ('1/(x + 3 + %s)', '1/(x + 3 + (%s))'),
                                              ('log(x + 3 + %s)', 'log(x + 3 + (%s))'),
                                              ('sqrt(x + 3 + %s)', 'sqrt(x + 3 + (%s))')]):
                f = shape % fc
                r = ref % val
                lim = 'Finite(1/3)' if (rational and j in (0, 1)) else 'Unknown'
                rows.append(['f:%s@%s:%d' % (fn, c, j), f, '0', 2, 'value', r, lim,
                             'gp: %s at 0' % r])
                gp_jobs.append((len(rows) - 1, r, '0', 2))
    # (g) S1 fix-up round 5 (review 5, semantics lens, m1, and the owner's first rule), after every
    # other row so that no id changes.
    # (g1) one pair per inverse function whose argument reaches a branch point c only through an
    #   exact value folded in round 4 (2 arcsin(1)/pi = 1, arccos(-1)/pi = 1, 2 arcsinh(i)/pi = i,
    #   2 arccosh(0)/pi = i, arccosh(-1)/(i pi) = 1, 4 arctan(1)/pi = 1): the non-constant argument
    #   c + x (or c + x^2) is NotLaurent (the owner's second rule); the exact-constant argument is
    #   the value f(c) (gp on the simplified constant), or DivisionByZero at a pole of the constant.
    #   Before round 4 the non-constant rows were UndecidedZero (the folded value was a kernel).
    # (g2) the exact values that round 5 adds (Abramowitz and Stegun 4.4 and 4.6: arcsin(+-1/2) =
    #   +-pi/6, arccos(1/2) = pi/3, arccos(-1/2) = 2 pi/3, arccosh(+-1/2) = i arccos(+-1/2),
    #   arcsinh(+-i/2) = +-i pi/6, arctanh(+-i) = +-i pi/4), each inverted at 0 (the value; as
    #   kernels without an enclosure the sums were UndecidedZero), and the exact zero
    #   x/(f(c) - value) (DivisionByZero: the input divides by an exact zero).
    g1 = [
        ('arcsin', 'arcsin(2*arcsin(x^0)/pi + x)', 'arcsin(2*arcsin(x^0)/pi)', 'asin(1)'),
        ('arccos', 'arccos(-arccos(-x^0)/pi + x)', 'arccos(-arccos(-x^0)/pi)', 'acos(-1)'),
        ('arctan', 'arctan(2*arcsinh(I*x^0)/pi + x)', 'arctan(2*arcsinh(I*x^0)/pi)', None),
        ('arcsinh', 'arcsinh(2*arccosh(0*x)/pi + x)', 'arcsinh(2*arccosh(0*x)/pi)', 'asinh(I)'),
        ('arccosh', 'arccosh(-arccosh(-x^0)/(I*pi) + x)', 'arccosh(-arccosh(-x^0)/(I*pi))',
         'acosh(-1)'),
        ('arctanh', 'arctanh(4*arctan(x^0)/pi + x^2)', 'arctanh(4*arctan(x^0)/pi)', None),
    ]
    for fn, nonconst, exact, ref in g1:
        rows.append(['g:%s:nonconst' % fn, nonconst, '0', 2, 'err NotLaurent', None, 'Unknown',
                     'a non-constant argument whose constant term is a branch point through a '
                     'folded exact value: NotLaurent, never DivisionByZero'])
        gp_jobs.append((len(rows) - 1, nonconst.replace('arc', 'a'), '0', 2))
        if ref is None:
            rows.append(['g:%s:exact' % fn, exact, '0', 2, 'err DivisionByZero', None, 'Unknown',
                         'f(c) is a pole of the constant: DivisionByZero'])
        else:
            r = '(%s)' % ref
            rows.append(['g:%s:exact' % fn, exact, '0', 2, 'value', r, 'Unknown',
                         'gp: %s at 0' % r])
            gp_jobs.append((len(rows) - 1, r, '0', 2))
    g2 = [
        ('arcsin@1/2', 'arcsin(x^0/2)', 'asin(1/2)', 'pi/6'),
        ('arcsin@-1/2', 'arcsin(-x^0/2)', 'asin(-1/2)', '-pi/6'),
        ('arccos@1/2', 'arccos(x^0/2)', 'acos(1/2)', 'pi/3'),
        ('arccos@-1/2', 'arccos(-x^0/2)', 'acos(-1/2)', '2*pi/3'),
        ('arccosh@1/2', 'arccosh(x^0/2)', 'acosh(1/2)', 'I*pi/3'),
        ('arccosh@-1/2', 'arccosh(-x^0/2)', 'acosh(-1/2)', '2*I*pi/3'),
        ('arcsinh@I/2', 'arcsinh(I*x^0/2)', 'asinh(I/2)', 'I*pi/6'),
        ('arcsinh@-I/2', 'arcsinh(-I*x^0/2)', 'asinh(-I/2)', '-I*pi/6'),
        ('arctanh@I', 'arctanh(I*x^0)', 'atanh(I)', 'I*pi/4'),
        ('arctanh@-I', 'arctanh(-I*x^0)', 'atanh(-I)', '-I*pi/4'),
    ]
    for cid, fc, ref, value in g2:
        f = '1/(3 + %s)' % fc
        r = '1/(3 + (%s))' % ref
        rows.append(['g:%s:inv' % cid, f, '0', 2, 'value', r, 'Unknown', 'gp: %s at 0' % r])
        gp_jobs.append((len(rows) - 1, r, '0', 2))
        f = 'x/(%s - (%s))' % (fc, value)
        rows.append(['g:%s:zero' % cid, f, '0', 2, 'err DivisionByZero', None, 'Unknown',
                     'the divisor is the exact zero f(c) - %s: DivisionByZero' % value])
    # (h) S1 finish under the narrowed bar (review 6, semantics lens, m2), after every other row so
    # that no id changes: the third case a branch point meets, a truncated constant. For every
    # pair, the argument c*(1+x)/(1+x) (for c = 0, (1+x)/(1+x) - 1) and c + (sin(x)^2+cos(x)^2-1)
    # at 0. Each is the constant c, but its series is c + O(h^P), not exact (the inverse of 1 + x
    # is truncated), so whether the argument is the branch point is undecided: the integrand (or
    # log, the root, the power of the base) asks for precision and the driver never decides,
    # PrecisionExhausted (a bounded search, section 2.1), never a value, NotLaurent or
    # DivisionByZero. gp's outcome is in the note: it reads (1+x)/(1+x) as the rational function
    # 1 and expands f(c), or fails on an O-term ("impossible inverse", "domain error").
    for tmpl, points in FUNCS:
        fn = fname(tmpl)
        for (c, val, rational) in points:
            if c == '0':
                args = ['(1+x)/(1+x) - 1', 'sin(x)^2+cos(x)^2-1']
            else:
                args = [scaled(c, '((1+x)/(1+x))'), '%s + (sin(x)^2+cos(x)^2-1)' % c]
            for j, u in enumerate(args):
                f = tmpl % u
                rows.append(['h:%s@%s:trunc%d' % (fn, c, j), f, '0', 2, 'err PrecisionExhausted',
                             None, 'Unknown',
                             'a truncated constant argument at the branch point: a bounded '
                             'search (PrecisionExhausted), never a value, NotLaurent or '
                             'DivisionByZero'])
                gp_jobs.append((len(rows) - 1, f, '0', 2))
    res = g.series_batch([(t, p, n) for (_, t, p, n) in gp_jobs])
    gp_of_row = {}
    for (i, t, p, n), r in zip(gp_jobs, res):
        gp_of_row[i] = r
    print('# S1 singular-point matrix (owner decision 2026-11-15). Generated by gen_s1_singular.py')
    print('# (python3 -I gen_s1_singular.py > s1_singular_points.tsv, in rustmath-symbolic/tests/data;')
    print('# gp 2.15.4 at realprecision 300 through s1_gp.py). Columns: id, input, point, order,')
    print('# expectation (value v k:re:im ... from exponent v up to the order, or err Kind), limit path,')
    print('# note (the gp reference, or the reason for the refusal; for a refusal, gp\'s own outcome).')
    bad = 0
    for i, r in enumerate(rows):
        rid, f, p, n, kind, ref, lim, note = r
        if kind == 'value':
            gr = gp_of_row[i]
            if gr[0] != 'ok':
                print('# GP FAILED for %s: %s' % (rid, gr[1]), file=sys.stderr)
                bad += 1
                continue
            _, v, coefs = gr
            want = 'value %d %s' % (v, ' '.join('%s:%s' % (re_, im_) for (_, re_, im_) in coefs))
        else:
            want = kind
            if i in gp_of_row:
                gr = gp_of_row[i]
                note += '; gp: ' + ('a formal series (valuation %d)' % gr[1] if gr[0] == 'ok' else gr[1][:90])
        print('\t'.join([rid, f, p, str(n), want, lim, note]))
    if bad:
        print('%d value rows without a gp reference' % bad, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
