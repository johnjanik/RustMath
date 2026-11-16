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
#   an error at a zero of odd order), and every power `b^a` -> PW(b, a), which takes the principal
#   value exp(a log b) when a is a non-integer rational and b is a series with a negative real
#   constant term (review 9 m1: gp's own t_SER^t_FRAC takes the real root of a negative rational
#   constant term, the engine takes the principal value).
# series(text, point, order) runs gp on F = f(P + h + O(h^(order+31))) (30 spare terms against gp's
# own loss of precision; an error when F still has no term of exponent order) and returns
# ('ok', valuation, [(k, re, im), ...]) or ('gperr', message).
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
AB(s) = SG(s) * s;
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


def gp_of(e, xname='x'):
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
        return '(-%s)' % gp_of(e[1], xname)
    if t == 'bin':
        op, l, r = e[1], gp_of(e[2], xname), gp_of(e[3], xname)
        if op == '^':
            return 'PW(%s,%s)' % (l, r)
        return '(%s%s%s)' % (l, op, r)
    if t == 'call':
        name, args = e[1], e[2]
        if len(args) == 1 and name in FUNCS:
            return '%s(%s)' % (FUNCS[name], gp_of(args[0], xname))
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


if __name__ == '__main__':
    import sys
    for t in sys.argv[1:]:
        print(t, '->', to_gp(t))


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
