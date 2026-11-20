# S1 (rustmath-symbolic series engine): gp references for the test data of
# tests/s1_singular_points.rs and tests/s1_fuzz_data.rs.
#
# Design: S1-symbolic design (frozen at revision 7rt4 + 8), sections 2.4, 2.5, 2.9, 2.12, 4.1.
# MAGMA Handbook ch. 49 (rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md).
# Reference values: PARI/GP 2.15.4 (/usr/bin/gp) at realprecision 300.
#
# to_gp(text) reads an input with the grammar of design section 2.12 (the port's parser: runs of
# signs by a loop, `a/b` an operator with the literal fold, exponent := signs power, `^` right
# associative) and writes a fully parenthesized gp expression in the variable x:
#   arcsin/asin -> asin, ..., arccosh/acosh -> ACOSH (on a series, the principal
#   log(u + sqrt(u + 1) sqrt(u - 1)), the formula of design section 2.4; gp's own acosh of a series
#   integrates (u^2 - 1)^(-1/2) and takes the other sign left of -1: acosh(-3/2 + h) has gp's
#   h coefficient +0.894..., the principal function -0.894...), pi -> Pi, e -> exp(1), I/i -> I, sign -> SG, abs -> AB (the
#   real-variable conventions of design section 2.4 on series: the sign of the leading coefficient,
#   an error at a zero of odd order; on a scalar, SG is the sign of the real part and AB is gp's
#   abs, the modulus, as the engine's abs of a constant is: S1 C-1 minor items, semantics review
#   m5; before, AB of a scalar was SG(s)*s, which differs from the modulus for a non-real s, and
#   no data row has abs of a non-real constant), and every power `b^a` -> PW(b, a), which takes the principal
#   value exp(a log b) when a is a non-integer rational and b is a series with a negative real
#   constant term (review 9 m1: gp's own t_SER^t_FRAC takes the real root of a negative rational
#   constant term, the engine takes the principal value).
# series(text, point, order) runs gp on F = f(P + h + O(h^(order+31))) (30 spare terms against gp's
# own loss of precision; an error when F still has no term of exponent order) and returns
# ('ok', valuation, [(k, re, im), ...]) or ('gperr', message).
import math
import re
import subprocess

FUNCS = {
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh',
    'arcsin': 'asin', 'asin': 'asin', 'arccos': 'acos', 'acos': 'acos', 'arctan': 'atan', 'atan': 'atan',
    'arcsinh': 'asinh', 'asinh': 'asinh', 'arccosh': 'ACOSH', 'acosh': 'ACOSH',
    'arctanh': 'atanh', 'atanh': 'atanh', 'exp': 'exp', 'log': 'log', 'ln': 'log', 'sqrt': 'sqrt',
    'abs': 'AB', 'sign': 'SG', 'erf': 'ERF', 'gamma': 'gamma', 'zeta': 'zeta',
}

GPHEAD = r'''default(realprecision, 300);
SG(s) = if(type(s) == "t_SER", my(v = valuation(s, h)); if(v % 2, error("sign or abs at a zero of odd order")); sign(real(polcoef(s, v, h))), sign(real(s)));
AB(s) = if(type(s) == "t_SER", SG(s) * s, abs(s));
SNR(z) = if(type(z) == "t_COMPLEX" && abs(imag(z)) < 1e-250*max(1, abs(z)), real(z), z);
SNI(z) = if(type(z) == "t_COMPLEX" && abs(real(z)) < 1e-250*max(1, abs(z)), I*imag(z), z);
ACOSH(z) = if(type(z) == "t_SER", log(z + sqrt(z + 1)*sqrt(z - 1)), acosh(z));
ERF(z) = if(type(z) == "t_SER", (1 - erfc(polcoef(z, 0, h))) + 2/sqrt(Pi)*intformal(exp(-z^2)*deriv(z, h), h), 1 - erfc(z));
PW(b, a) = if(a == 0, 1, type(a) == "t_FRAC" && type(b) == "t_SER" && valuation(b, h) == 0 && imag(polcoef(b, 0, h)) == 0 && real(polcoef(b, 0, h)) < 0, exp(a * log(b)), b^a);
NS(s) = strjoin(strsplit(s, " "), "");
CO(c) = my(r = real(c), i = imag(c)); NS(Str(if(r == 0, "0", if(type(r) == "t_INT" || type(r) == "t_FRAC", r, Strprintf("%.70g", r))), ":", if(i == 0, "0", if(type(i) == "t_INT" || type(i) == "t_FRAC", i, Strprintf("%.70g", i)))));
'''


class P:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self):
        self.ws()
        return self.s[self.i] if self.i < len(self.s) else None

    def expression(self):
        e = self.multiplicative()
        while self.peek() in ('+', '-'):
            op = self.s[self.i]
            self.i += 1
            r = self.multiplicative()
            e = ('bin', op, e, r)
        return e

    @staticmethod
    def signed_int(e):
        if e[0] == 'int':
            return e[1]
        if e[0] == 'neg' and e[1][0] == 'int':
            return -e[1][1]
        return None

    def multiplicative(self):
        e = self.unary()
        while self.peek() in ('*', '/'):
            op = self.s[self.i]
            self.i += 1
            r = self.unary()
            if op == '/':
                a, b = self.signed_int(e), self.signed_int(r)
                if a is not None and b is not None:
                    if b == 0:
                        raise ValueError('division by zero in a literal')
                    from fractions import Fraction
                    q = Fraction(a, b)
                    e = ('int', q.numerator) if q.denominator == 1 else ('rat', q)
                    continue
            e = ('bin', op, e, r)
        return e

    def signs(self):
        neg = False
        while self.peek() in ('-', '+'):
            if self.s[self.i] == '-':
                neg = not neg
            self.i += 1
        return neg

    def unary(self):
        neg = self.signs()
        p = self.power()
        return ('neg', p) if neg else p

    def power(self):
        base = self.atom()
        if self.peek() == '^':
            self.i += 1
            neg = self.signs()
            ex = self.power()
            if neg:
                ex = ('int', -ex[1]) if ex[0] == 'int' else ('neg', ex)
            return ('bin', '^', base, ex)
        return base

    def atom(self):
        c = self.peek()
        if c is None:
            raise ValueError('unexpected end')
        if c.isdigit():
            st = self.i
            while self.i < len(self.s) and self.s[self.i].isdigit():
                self.i += 1
            if self.i + 1 < len(self.s) and self.s[self.i] == '.' and self.s[self.i + 1].isdigit():
                self.i += 1
                while self.i < len(self.s) and self.s[self.i].isdigit():
                    self.i += 1
                return ('real', self.s[st:self.i])
            return ('int', int(self.s[st:self.i]))
        if c == '(':
            self.i += 1
            e = self.expression()
            if self.peek() != ')':
                raise ValueError('expected ) at %d' % self.i)
            self.i += 1
            return e
        if c.isalpha():
            st = self.i
            while self.i < len(self.s) and (self.s[self.i].isalnum() or self.s[self.i] == '_'):
                self.i += 1
            name = self.s[st:self.i]
            if self.peek() == '(':
                self.i += 1
                args = []
                if self.peek() == ')':
                    self.i += 1
                else:
                    while True:
                        args.append(self.expression())
                        if self.peek() == ',':
                            self.i += 1
                        elif self.peek() == ')':
                            self.i += 1
                            break
                        else:
                            raise ValueError('expected , or )')
                return ('call', name, args)
            return ('sym', {'i': 'I'}.get(name, name))
        raise ValueError('unexpected %r at %d' % (c, self.i))


def parse(text):
    p = P(text)
    e = p.expression()
    p.ws()
    if p.i != len(text):
        raise ValueError('unexpected input remaining at %d in %r' % (p.i, text))
    return e


# S1 C-1 minor items (semantics review m4): with snap=True (the scalar F of the two-sided oracle),
# the argument of each branch function is snapped to the axis of its cut when its other part is
# below 1e-250 relative to max(1, |z|): SNR (log, sqrt, the bases of the powers, asin, acos,
# atanh, acosh, whose cuts lie on the real axis) drops an imaginary part, SNI (atan, asinh, on the
# imaginary axis) a real part. That part is then gp's rounding residue of an exact cancellation on
# a cut (log(-1 + x) - log(-1) is exactly log(1 - x), and gp leaves an imaginary part near 1e-300
# of either sign), which would put an outer branch function on either side of its cut. A genuine
# part at the smallest x0 the oracle uses (1e-40) is above 1e-250 for terms up to x^6.
SNAP_REAL = {'log', 'sqrt', 'asin', 'acos', 'atanh', 'ACOSH'}
SNAP_IMAG = {'atan', 'asinh'}


def gp_of(e, xname='x', snap=False):
    t = e[0]
    if t == 'int':
        return '(%d)' % e[1]
    if t == 'rat':
        return '(%d/%d)' % (e[1].numerator, e[1].denominator)
    if t == 'real':
        return '(%s)' % e[1]
    if t == 'sym':
        n = e[1]
        return {'pi': 'Pi', 'e': 'exp(1)', 'I': 'I', 'x': xname}.get(n, n)
    if t == 'neg':
        return '(-%s)' % gp_of(e[1], xname, snap)
    if t == 'bin':
        op, l, r = e[1], gp_of(e[2], xname, snap), gp_of(e[3], xname, snap)
        if op == '^':
            return 'PW(%s,%s)' % ('SNR(%s)' % l if snap else l, r)
        return '(%s%s%s)' % (l, op, r)
    if t == 'call':
        name, args = e[1], e[2]
        if len(args) == 1 and name in FUNCS:
            f, a = FUNCS[name], gp_of(args[0], xname, snap)
            if snap and f in SNAP_REAL:
                a = 'SNR(%s)' % a
            elif snap and f in SNAP_IMAG:
                a = 'SNI(%s)' % a
            return '%s(%s)' % (f, a)
        raise ValueError('no gp form for the function %s' % name)
    raise ValueError(t)


def to_gp(text, xname='x'):
    return gp_of(parse(text), xname)


def series_prog(text, point, order, tag=''):
    """gp lines that print 'RES<tag> v k:re:im ...' or 'GPERR<tag> message'."""
    fx = to_gp(text, '(X)')
    pt = to_gp(point)
    if 'I' in pt:
        # gp computes the sqrt of a series with an exact complex constant term at 19 digits
        # only (gp 2.15.4: sqrt(1/4 + I + h + O(h^3)) has a constant term of 19 correct
        # digits); a complex point is therefore taken as a t_REAL complex at realprecision
        pt = '(%s)*1.' % pt
    n = int(order)
    return ('iferr(P = %s; X = P + h + O(h^%d); F = %s; '
            'if(type(F) != "t_SER", F = F + O(h^%d)); v = min(valuation(F, h), 0); '
            'if(v < -60, error("valuation below -60")); '
            'if(serprec(F, h) <= %d, error("gp lost the precision: serprec ", serprec(F, h))); '
            'print("RES%s ", v, " ", strjoin(vector(%d - v + 1, j, CO(polcoef(F, v + j - 1, h))), " ")), '
            'E, print("GPERR%s ", errname(E), " ", strjoin(strsplit(Str(E), "\\n"), " ")));\n'
            % (pt, n + 31, fx, n + 31, n, tag, n, tag))


def run_gp(prog, timeout=600):
    p = subprocess.run(['gp', '-q', '-f', '-s', '1000000000'], input=GPHEAD + prog,
                       capture_output=True, text=True, timeout=timeout)
    return p.stdout, p.stderr


def parse_res(line):
    """'RES v c c ...' -> ('ok', v, [(k, re, im)]) with exponents v, v+1, ..."""
    parts = line.split(' ')
    v = int(parts[1])
    out = []
    for j, c in enumerate(parts[2:]):
        re_, im_ = c.split(':')
        out.append((v + j, re_, im_))
    return ('ok', v, out)


def series(text, point, order):
    out, err = run_gp(series_prog(text, point, order))
    for l in out.splitlines():
        if l.startswith('RES '):
            return parse_res(l)
        if l.startswith('GPERR '):
            return ('gperr', l[6:])
    return ('gperr', 'no output: ' + err.strip()[:200])


def series_batch(rows, timeout=3000):
    """rows: [(text, point, order)] -> one result per row, as series() gives, in one gp run."""
    prog = ''
    for i, (t, p, n) in enumerate(rows):
        prog += series_prog(t, p, n, tag='#%d' % i)
    out, err = run_gp(prog, timeout=timeout)
    res = [('gperr', 'no output')] * len(rows)
    for l in out.splitlines():
        m = re.match(r'^(RES|GPERR)#(\d+) (.*)$', l)
        if not m:
            continue
        i = int(m.group(2))
        if m.group(1) == 'RES':
            try:
                res[i] = parse_res('RES ' + m.group(3))
            except ValueError:
                res[i] = ('gperr', 'unreadable gp output: ' + m.group(3)[:200])
        else:
            res[i] = ('gperr', m.group(3))
    return res


# ---------------------------------------------------------------------------------------------
# Two-sided numeric oracle (S1 request C-1, docs/workstreams/m23_golay/REQUESTS.md section 2).
#
# gp's formal series of f at P (series_prog above) is the series of the branch that is analytic
# through the principal value at P. When an argument of a branch function has its constant term on
# an open cut and a coefficient that takes it off the cut, that series is the function of a real
# variable on one side of P, or on neither. No comparison of formal series can see it, since gp and
# the engine compute the same one. This oracle compares a truncated series with the scalar function
# itself, which gp evaluates with its principal branches at the real points P + x0 and P - x0:
#
#   S(t)  = sum_{k=v}^{n} c_k t^k   (the row's coefficients: the value the engine must return),
#   F(t)  = f(P + t)                (gp's scalar function),
#   T(t)  = sum_{k=n+1}^{m} |g_k| |t|^k (the rest of gp's own formal series G = sum g_k t^k, in
#           absolute values, m = serprec(G) - 1, so that terms of the rest cannot cancel),
#
# for t = x0 and t = -x0. x0 starts at 1/1000 and is divided by 10 until T is at most
# 1e-15 max(1, |S|) (at most 37 times), so x0 lies well inside the radius of convergence. The row
# passes on a side when |S(t) - F(t)| <= 100 T + 1e-60 sum |c_k| |t|^k + 1e-100 max(1, |F(t)|):
# the truncation error of a series that represents f there, plus the rounding of the data's
# coefficients (70 significant digits). The errors and tolerances are printed relative to
# max(1, |F(t)|). Where gp has no formal series of the text (a gp error on
# some exact-constant shapes), x0 = 1e-30 and the tolerance gets 1e-20 max(1, sum |c_k| |t|^k).
# A branch error
# is of order 1 (2 pi I, a sign, pi I), far above that tolerance. The same check of gp's formal
# series truncated at n (S replaced by G up to n) says whether the formal series represents f on
# each side, which documents a refusal row.
#
# The scalar conventions are those of the engine: principal log, sqrt and powers exp(a log b);
# asin, acos, atan, asinh, atanh of gp (principal); acosh(z) = log(z + sqrt(z + 1) sqrt(z - 1))
# (gp's scalar acosh is that function: checked on a grid of step 1/4 in [-3, 3]^2 and on the real
# axis, largest difference 2.5e-76 at 60 digits); sign as the sign of the real part and abs as
# the modulus (AB, at GPHEAD). The branch arguments of F are snapped to the axis of their cut when
# their other part is a rounding residue (gp_of with snap=True); those of G are not.
#
# A row is "selected" when the argument of a branch function in it (log, sqrt, the inverse
# functions, a power whose exponent is not an integer literal) has a coefficient of exponent != 0
# with a nonzero imaginary part in gp's series at P (|Im| > 1e-200). The oracle runs on every row
# and reports the selected ones apart.
#
# Command (from rustmath-symbolic/tests/data):
#     python3 -I s1_gp.py --two-sided s1_cuts.tsv s1_singular_points.tsv s1_fuzz.tsv
# It prints one line per row and outcome and a summary per file.

BRANCH_CALLS = {'log', 'ln', 'sqrt', 'arcsin', 'asin', 'arccos', 'acos', 'arctan', 'atan',
                'arcsinh', 'asinh', 'arccosh', 'acosh', 'arctanh', 'atanh'}


def branch_args(e, out):
    """The arguments of the branch functions in the parse tree e (bases of non-integer powers)."""
    t = e[0]
    if t == 'neg':
        branch_args(e[1], out)
    elif t == 'bin':
        branch_args(e[2], out)
        branch_args(e[3], out)
        if e[1] == '^' and e[3][0] != 'int':
            out.append(e[2])
    elif t == 'call':
        for a in e[2]:
            branch_args(a, out)
        if e[1] in BRANCH_CALLS and len(e[2]) == 1:
            out.append(e[2][0])
    return out


def gp_num(re_, im_):
    return '((%s)+(%s)*I)' % (re_, im_)


def two_sided_prog(text, point, order, coefs, tag):
    """gp lines for one row: 'NR<tag> 0|1', then per side 'TWO<tag> side x0exp err tol ferr', or
    'TWOERR<tag> message'. coefs: [(k, re, im)] for the exponents v..order, or None (a refusal row:
    only gp's formal series is checked)."""
    e = parse(text)
    fx = gp_of(e, '(X)')
    # the scalar function, its branch arguments snapped (semantics review m4, at gp_of); G, gp's
    # formal series, is not snapped, so a constant term of G that cancels exactly on a cut can
    # still land on either side of it with gp's rounding (U:y12.1 of s1_fuzz.tsv changes side
    # between realprecision 80 and 300): the "formal" verdict of a refusal row documents it only
    fs = gp_of(e, '(X)', snap=True)
    pt = to_gp(point)
    if 'I' in pt:
        pt = '(%s)*1.' % pt
    n = int(order)
    m = n + 31
    if coefs:
        v = coefs[0][0]
        cv = '[' + ','.join(gp_num(r, i) for (_, r, i) in coefs) + ']'
    else:
        v, cv = 0, '0'
    args = branch_args(e, [])
    nr = '0'
    if args:
        nr = '(' + ' || '.join('NRF(%s, %d)' % (gp_of(a, '(X)'), n + 5) for a in args) + ')'
    # G: gp's formal series; when gp cannot compute it (gp's own series of some exact-constant
    # shapes fails, s1_singular_points.tsv class (a)), Gok = 0: the value row is still checked,
    # at x0 = 1e-30 with the tolerance 1e-20 (a truncation error |c_(n+1)| x0^(n+1) for
    # coefficients below 1e10), and the line says so (km = -1)
    return (
        'iferr(P = %s; Gok = iferr(X = P + h + O(h^%d); G = %s; '
        'if(type(G) != "t_SER", G = G + O(h^%d)); serprec(G, h); 1, E3, 0); '
        'NRv = if(Gok, iferr(%s, E2, 0), 0); print("NR%s ", if(NRv, 1, 0)); '
        'gv = if(Gok, min(valuation(G, h), 0), 0); km = if(Gok, serprec(G, h) - 1, -1); '
        'Cv = %s; V = %d; '
        'if(!Gok && type(Cv) != "t_VEC", error("no formal series")); '
        'for(s = 0, 1, sg = if(s, -1, 1); x0 = 1/1000; ex = 3; '
        'if(!Gok, x0 = 1/10^30; ex = 30); '
        'for(it = 1, if(Gok, 37, 1), '
        'T = if(Gok, sum(k = %d, km, abs(polcoef(G, k, h))*x0^k, 0.), 0.); '
        'Gn = if(Gok, sum(k = gv, min(%d, km), polcoef(G, k, h)*(sg*x0)^k, 0.), 0.); '
        'Sv = if(type(Cv) == "t_VEC", sum(j = 1, #Cv, Cv[j]*(sg*x0)^(V + j - 1)), Gn); '
        'Sa = if(type(Cv) == "t_VEC", sum(j = 1, #Cv, abs(Cv[j])*x0^(V + j - 1)), abs(Gn)); '
        'if(!Gok || abs(T) <= 1e-15*max(1, abs(Sv)), break); x0 = x0/10; ex++); '
        'X = P + sg*x0; F = %s; '
        # the data's coefficients have 70 significant digits (compared to 60 by the tests)
        'tol = 100*abs(T) + 1e-60*Sa + 1e-100*max(1, abs(F)) + if(Gok, 0, 1e-20*max(1, Sa)); '
        # printed relative to max(1, |F|), so that a value beyond the doubles (3^(10^4)) prints
        'Fs = max(1, abs(F)); '
        'print("TWO%s ", sg, " ", ex, " ", Strprintf("%%.6e", abs(Sv - F)/Fs), " ", '
        'Strprintf("%%.6e", tol/Fs), " ", Strprintf("%%.6e", if(Gok, abs(Gn - F)/Fs, 0)), " ", '
        'km))'
        ', E, print("TWOERR%s ", errname(E), " ", strjoin(strsplit(Str(E), "\\n"), " ")));\n'
        % (pt, m, fx, m, nr, tag, cv, v, n + 1, n, fs, tag, tag))


# NRF(A, n): whether the series A has a coefficient of exponent != 0 (up to n) with a nonzero
# imaginary part
TWOHEAD = r'''NRF(A, n) = if(type(A) != "t_SER", 0, my(mx = 0, va = valuation(A, h)); for(k = min(va, 0), min(n, serprec(A, h) - 1), if(k != 0, mx = max(mx, abs(imag(polcoef(A, k, h)))))); mx > 1e-200);
'''


def two_sided_batch(jobs, timeout=6000):
    """jobs: [(text, point, order, coefs or None)] -> per job a dict with 'nr' (selected),
    'sides' [(side, x0exp, err, tol, ferr, km)] or 'err' (gp's message)."""
    prog = TWOHEAD
    for i, (t, p, n, c) in enumerate(jobs):
        prog += two_sided_prog(t, p, n, c, '#%d' % i)
    out, err = run_gp(prog, timeout=timeout)
    res = [{'nr': None, 'sides': [], 'err': None} for _ in jobs]
    for l in out.splitlines():
        mm = re.match(r'^(NR|TWO|TWOERR)#(\d+) (.*)$', l)
        if not mm:
            continue
        i, body = int(mm.group(2)), mm.group(3)
        if mm.group(1) == 'NR':
            res[i]['nr'] = body.strip() == '1'
        elif mm.group(1) == 'TWO':
            # gp's %e prints a space before the exponent ('6.283185 e0')
            sg, ex, e_, tol, fe, km = body.replace(' e', 'e').split(' ')
            res[i]['sides'].append((int(sg), int(ex), float(e_), float(tol), float(fe), int(km)))
        else:
            res[i]['err'] = body
    for r in res:
        if r['nr'] is None and r['err'] is None:
            r['err'] = 'no output'
    return res


def two_sided_rows(path):
    """(id, input, point, order, [value alternatives as coefficient lists], refusal kinds)."""
    rows = []
    for line in open(path):
        if not line.strip() or line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        vals, refs = [], []
        for w in f[4].split(' | '):
            if w.startswith('value '):
                parts = w[6:].split(' ')
                v = int(parts[0])
                vals.append([(v + j, c.split(':')[0], c.split(':')[1])
                             for j, c in enumerate(parts[1:])])
            else:
                refs.append(w[4:])
        rows.append((f[0], f[1], f[2], int(f[3]), vals, refs))
    return rows


def two_sided_report(path):
    rows = two_sided_rows(path)
    jobs, owner = [], []
    for ri, (rid, t, p, n, vals, refs) in enumerate(rows):
        for vi, c in enumerate(vals):
            jobs.append((t, p, n, c))
            owner.append((ri, 'value%d' % vi))
        if refs:
            jobs.append((t, p, n, None))
            owner.append((ri, 'formal'))
    res = two_sided_batch(jobs)
    stats = {'value': 0, 'value_sel': 0, 'value_fail': [], 'value_gperr': [],
             'formal': 0, 'formal_wrong': []}
    print('== %s' % path)
    for (ri, what), r in zip(owner, res):
        rid, t, p, n, vals, refs = rows[ri]
        sel = 'sel' if r['nr'] else '-'
        if r['err'] is not None or len(r['sides']) != 2:
            msg = r['err'] or 'incomplete output'
            print('%s\t%s\t%s\t%s\tGPERR %s' % (rid, what, sel, t, msg[:120]))
            if what != 'formal':
                stats['value'] += 1
                stats['value_gperr'].append(rid)
            continue
        cells = []
        ok = True
        fok = True
        # a tolerance that is not finite (gp's rest of the series did not shrink at any x0 tried:
        # a floating constant term next to a pole) decides nothing
        undetermined = any(not math.isfinite(sd[3]) for sd in r['sides'])
        for (sg, ex, e_, tol, fe, km) in r['sides']:
            if km < 0:
                cells.append('%+d x0=1e-%d err=%.2e tol=%.2e (no gp series)' % (sg, ex, e_, tol))
            else:
                cells.append('%+d x0=1e-%d err=%.2e tol=%.2e formal=%.2e' % (sg, ex, e_, tol, fe))
            ok = ok and e_ <= tol
            fok = fok and fe <= tol and km >= 0
        if undetermined:
            ok = fok = False
        if what == 'formal':
            stats['formal'] += 1
            if undetermined:
                stats.setdefault('formal_undetermined', []).append(rid)
                verdict = 'no verdict (the rest of the formal series does not shrink)'
            elif fok:
                verdict = 'formal series represents f on both sides'
            else:
                stats['formal_wrong'].append(rid)
                verdict = 'formal series WRONG on side(s) %s' % ','.join(
                    '%+d' % sd[0] for sd in r['sides'] if sd[4] > sd[3])
        else:
            stats['value'] += 1
            if r['nr']:
                stats['value_sel'] += 1
            if not ok:
                stats['value_fail'].append(rid)
            verdict = 'ok' if ok else 'FAIL'
        print('%s\t%s\t%s\t%s at %s\t%s\t%s' % (rid, what, sel, t, p, verdict, ' | '.join(cells)))
    und = stats.get('formal_undetermined', [])
    print('summary %s: %d value outcomes (%d selected: a branch argument with a non-real '
          'coefficient), %d fail %s, %d gp errors %s; %d refusal rows, formal series wrong on a '
          'side for %d %s, no verdict for %d %s' % (
              path, stats['value'], stats['value_sel'], len(stats['value_fail']),
              stats['value_fail'], len(stats['value_gperr']), stats['value_gperr'],
              stats['formal'], len(stats['formal_wrong']), stats['formal_wrong'], len(und), und))
    return stats


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--two-sided':
        bad = 0
        for path in sys.argv[2:]:
            st = two_sided_report(path)
            bad += len(st['value_fail'])
        sys.exit(1 if bad else 0)
    for t in sys.argv[1:]:
        print(t, '->', to_gp(t))
