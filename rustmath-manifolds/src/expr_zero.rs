//! An exact three-valued zero test for symbolic expressions, and the scans that the exterior
//! algebra of [`crate::diff_form`] needs.
//!
//! # References
//!
//! * MAGMA Handbook, Chapter 24 (Multivariate Polynomial Rings), §24.4.2 "Equality and
//!   Membership" and §24.4.7 "Derivative, Integral"; Chapter 41 (Rational Function Fields).
//!   Digests: `rustmath-lava/docs/Chapter_24_Multivariate_Polynomial_Rings.md`,
//!   `rustmath-lava/docs/Chapter_41_Rational_Function_Fields.md`.
//! * B. Buchberger, R. Loos, "Algebraic Simplification", in *Computer Algebra: Symbolic and
//!   Algebraic Computation*, Springer 1982, pp. 11-43: the canonical simplifier for rational
//!   functions (numerator and denominator in a polynomial ring with integer coefficients).
//! * D. Richardson, "Some undecidable problems involving elementary functions of a real
//!   variable", J. Symbolic Logic 33 (1968) 514-520: zero equivalence of elementary
//!   expressions is undecidable, so an answer `Unresolved` cannot be avoided.
//! * F. Lindemann, "Über die Zahl π", Math. Ann. 20 (1882) 213-225, and C. Hermite, "Sur la
//!   fonction exponentielle", C. R. Acad. Sci. 77 (1873): π and e are transcendental.
//! * The reserved symbol names follow `rustmath-symbolic/src/parser.rs:216-237`.
//! * The node kinds that [`unsupported_derivative`] refuses follow
//!   `rustmath-symbolic/src/differentiate.rs` (`:90` Mod, `:184-188` Gamma, `:191` Factorial,
//!   `:194-198` Erf, `:204` Zeta, `:215-221` a `root_sum` whose polynomial contains the
//!   variable, `:230-234` every other `Function`), tracker row "Derivative of a named
//!   function" (`PROJECT_TRACKER.md:2117`).
//!
//! # Semantics
//!
//! An expression denotes a function of independent real indeterminates. Every [`Symbol`] is
//! such an indeterminate, except the reserved names of the parser:
//!
//! * `I` (the parser maps `I` and `i` to it) is an opaque atom. A `Zero` answer holds for any
//!   value of it. `NonZero` is never claimed when it occurs. It is not reduced with
//!   I^2 = -1, because a chart may have a coordinate named `I`.
//! * `pi` and `e` are named constants. `NonZero` is claimed only when at most one of them
//!   occurs, by the transcendence of π and of e. Both together give `Unresolved`, because
//!   their algebraic independence is open.
//! * `oo` (the parser maps `oo`, `inf` and `infinity` to it) gives `Unresolved` wherever it
//!   occurs, also inside an opaque subterm. A chart with a coordinate named `oo` cannot be used
//!   with forms: every expression that contains it counts as not finite.
//!
//! A finite `Real(f)` is its exact binary value, a dyadic rational (every finite IEEE 754
//! double is one). So `0.5*x - x/2` is `Zero` and `0.1*x - x/10` is `NonZero`: 0.1 is
//! 3602879701896397/2^55. A NaN or infinite `Real` anywhere gives `Unresolved`.
//!
//! Precondition: every opaque subterm (function applications, `Unary` nodes other than `Neg`,
//! `Mod`, powers whose exponent is not an `Integer` of absolute value at most 2^20, the symbol
//! `I`) is finite on a dense open subset of the domain. `log(0)` and
//! `1 / (sin(x)^2 + cos(x)^2 - 1)` are outside this contract. A divisor that converts to the
//! zero polynomial is caught (`Unresolved`).
//!
//! # Algorithm
//!
//! A prescan (iterative, one work unit a node) refuses infinities and trees deeper than
//! `MAX_TREE_DEPTH`. Then the expression is converted to a fraction N/D of polynomials with
//! integer coefficients over atoms: the indeterminates, the named constants and the opaque
//! subterms (compared structurally). Sum and product chains are flattened along their left
//! spine. N = 0 is `Zero`. Otherwise the answer is `NonZero` when no opaque atom and at most
//! one named constant remains in N or D, else `Unresolved`.
//!
//! Every step is charged to one work budget per call ([`ZERO_TEST_WORK`] units) before it
//! runs: nodes, term products, term insertions, negations, clones, comparisons. Coefficient
//! sizes, total degrees, nesting depth and the number of opaque atoms are bounded too. An
//! exhausted budget or limit gives `Unresolved`, never a wrong decision.

use crate::errors::{ManifoldError, Result};
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::MultivariatePolynomial;
use rustmath_rationals::Rational;
use rustmath_symbolic::expression::{BinaryOp, UnaryOp};
use rustmath_symbolic::{Expr, Symbol};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The answer of the zero test.
///
/// `Zero` and `NonZero` are proofs under the precondition of the module documentation.
/// `Unresolved` is not a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroTest {
    /// The expression is the zero function.
    Zero,
    /// The expression is not identically zero on any open set.
    NonZero,
    /// Not decided (opaque atoms, infinities, or an exhausted budget).
    Unresolved,
}

/// The symbol names that rustmath-symbolic's parser gives a fixed meaning
/// (`rustmath-symbolic/src/parser.rs:216-237`).
pub mod reserved {
    /// The imaginary unit. The parser maps `I` and `i` to it.
    pub const IMAGINARY_UNIT: &str = "I";
    /// π and Euler's number. The parser maps `pi`, `π` and `e` to them.
    pub const NAMED_CONSTANTS: [&str; 2] = ["pi", "e"];
    /// Infinity. The parser maps `oo`, `inf` and `infinity` to it.
    pub const INFINITY: &str = "oo";
}

/// Work units one call of [`zero_test`] may spend.
pub const ZERO_TEST_WORK: u64 = 100_000;

/// Upper bound of the bit length of every coefficient of N and D.
const COEFF_BITS: u64 = 4096;
/// Upper bound of the total degree of N and D (`Monomial` adds `u32` exponents unchecked).
const MAX_DEGREE: usize = 1 << 20;
/// Depth bound of the prescan; it keeps the recursive `==` of `Expr` safe on small stacks.
const MAX_TREE_DEPTH: usize = 4096;
/// Recursion bound of the conversion, after flattening sum and product chains.
const MAX_NESTING: usize = 200;
/// Number of distinct opaque atoms one call may hold.
const MAX_OPAQUE: usize = 64;
/// Nodes a scan visits before it starts to memoize shared nodes, so that small scans do not
/// allocate a set.
const MEMO_AFTER: usize = 64;

type Poly = MultivariatePolynomial<Integer>;

/// A polynomial with an upper bound of the bit length of its coefficients.
#[derive(Clone)]
struct BoundedPoly {
    p: Poly,
    bits: u64,
}

/// A fraction of two bounded polynomials.
struct Fraction {
    num: BoundedPoly,
    den: BoundedPoly,
}

/// Why a conversion stopped. Both give `Unresolved`.
enum Fail {
    /// A division by the zero polynomial, or a value that is not finite.
    Undefined,
    /// The work budget or a limit is exhausted.
    Budget,
}

/// The kind of an atom of the conversion.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Indeterminate,
    Named,
    Opaque,
}

/// State of one conversion: the work left, the atoms seen so far.
struct Conv {
    left: u64,
    syms: HashMap<usize, usize>,
    opaque: Vec<(Expr, usize)>,
    kinds: Vec<Kind>,
}

fn one() -> BoundedPoly {
    BoundedPoly { p: Poly::constant(Integer::one()), bits: 1 }
}

fn is_one(p: &BoundedPoly) -> bool {
    p.p.num_terms() == 1 && p.p.is_constant() && p.p.terms().all(|(_, c)| c.is_one())
}

/// Σ over the terms of (1 + number of variables of the monomial). It covers the cost of a
/// monomial product (a `BTreeMap` merge) and of the degree scan.
fn weight(p: &Poly) -> u64 {
    p.terms().map(|(m, _)| 1 + m.exponents().count() as u64).sum()
}

fn max_bits(p: &Poly) -> u64 {
    p.terms().map(|(_, c)| c.bit_length()).max().unwrap_or(0)
}

fn bit_len(x: u64) -> u64 {
    64 - x.leading_zeros() as u64
}

impl Conv {
    fn charge(&mut self, units: u64) -> std::result::Result<(), Fail> {
        if units > self.left {
            self.left = 0;
            Err(Fail::Budget)
        } else {
            self.left -= units;
            Ok(())
        }
    }

    fn new_atom(&mut self, kind: Kind) -> usize {
        self.kinds.push(kind);
        self.kinds.len() - 1
    }

    fn symbol(&mut self, s: &Symbol) -> usize {
        if let Some(&v) = self.syms.get(&s.id()) {
            return v;
        }
        let kind = if s.name() == reserved::IMAGINARY_UNIT {
            Kind::Opaque
        } else if reserved::NAMED_CONSTANTS.contains(&s.name()) {
            Kind::Named
        } else {
            Kind::Indeterminate
        };
        let v = self.new_atom(kind);
        self.syms.insert(s.id(), v);
        v
    }

    /// The atom of an opaque subterm. Charged its node count times one plus the number of
    /// opaque atoms seen so far: the count and the `==` comparisons.
    fn opaque(&mut self, e: &Expr) -> std::result::Result<usize, Fail> {
        let size = node_count(e, self.left)?;
        self.charge(size.saturating_mul(1 + self.opaque.len() as u64))?;
        if let Some((_, v)) = self.opaque.iter().find(|(a, _)| a == e) {
            return Ok(*v);
        }
        if self.opaque.len() >= MAX_OPAQUE {
            return Err(Fail::Budget);
        }
        let v = self.new_atom(Kind::Opaque);
        self.opaque.push((e.clone(), v));
        Ok(v)
    }

    fn atom(&self, v: usize) -> Fraction {
        Fraction { num: BoundedPoly { p: Poly::variable(v), bits: 1 }, den: one() }
    }

    fn constant(&self, r: &Rational) -> std::result::Result<Fraction, Fail> {
        let (nb, db) = (r.numerator().bit_length(), r.denominator().bit_length());
        if nb.max(db) > COEFF_BITS {
            return Err(Fail::Budget);
        }
        Ok(Fraction {
            num: BoundedPoly { p: Poly::constant(r.numerator().clone()), bits: nb },
            den: BoundedPoly { p: Poly::constant(r.denominator().clone()), bits: db },
        })
    }

    /// Negation. `MultivariatePolynomial::neg` rebuilds the map with one `add_term` per term,
    /// so it is charged the weight.
    fn negate(&mut self, a: BoundedPoly) -> std::result::Result<BoundedPoly, Fail> {
        self.charge(weight(&a.p) + 1)?;
        Ok(BoundedPoly { p: -a.p, bits: a.bits })
    }

    /// Replaces a cached coefficient bound by the exact maximum (one unit a term).
    fn tighten(&mut self, a: &mut BoundedPoly) -> std::result::Result<(), Fail> {
        self.charge(a.p.num_terms() as u64)?;
        a.bits = max_bits(&a.p);
        Ok(())
    }

    /// Sum: the smaller operand is inserted into the larger one.
    fn add(&mut self, a: BoundedPoly, b: BoundedPoly) -> std::result::Result<BoundedPoly, Fail> {
        let (mut big, mut small) = if a.p.num_terms() >= b.p.num_terms() { (a, b) } else { (b, a) };
        self.charge(weight(&small.p) + 1)?;
        if big.bits.max(small.bits) + 1 > COEFF_BITS {
            self.tighten(&mut big)?;
            self.tighten(&mut small)?;
            if big.bits.max(small.bits) + 1 > COEFF_BITS {
                return Err(Fail::Budget);
            }
        }
        let bits = big.bits.max(small.bits) + 1;
        Ok(BoundedPoly { p: big.p + small.p, bits })
    }

    /// Product. A coefficient of the product is a sum of at most min(s, t) products, so its
    /// bit length is at most b_a + b_b + bitlen(min(s, t)); this bound is checked first.
    fn mul(&mut self, mut a: BoundedPoly, mut b: BoundedPoly) -> std::result::Result<BoundedPoly, Fail> {
        if is_one(&a) {
            return Ok(b);
        }
        if is_one(&b) {
            return Ok(a);
        }
        if a.p.is_zero() || b.p.is_zero() {
            return Ok(BoundedPoly { p: Poly::zero(), bits: 0 });
        }
        let (s, t) = (a.p.num_terms() as u64, b.p.num_terms() as u64);
        if a.bits + b.bits + bit_len(s.min(t)) > COEFF_BITS {
            self.tighten(&mut a)?;
            self.tighten(&mut b)?;
            if a.bits + b.bits + bit_len(s.min(t)) > COEFF_BITS {
                return Err(Fail::Budget);
            }
        }
        let cost = s
            .saturating_mul(t)
            .saturating_add(s.saturating_mul(weight(&b.p)))
            .saturating_add(t.saturating_mul(weight(&a.p)));
        self.charge(cost)?;
        if a.p.total_degree() + b.p.total_degree() > MAX_DEGREE {
            return Err(Fail::Budget);
        }
        let bits = a.bits + b.bits + bit_len(s.min(t));
        Ok(BoundedPoly { p: a.p * b.p, bits })
    }

    fn frac_add(&mut self, a: Fraction, b: Fraction, subtract: bool) -> std::result::Result<Fraction, Fail> {
        let bn = if subtract { self.negate(b.num)? } else { b.num };
        self.charge(a.den.p.num_terms().min(b.den.p.num_terms()) as u64)?;
        if a.den.p == b.den.p {
            let num = self.add(a.num, bn)?;
            return Ok(Fraction { num, den: a.den });
        }
        // the two clones below
        self.charge((a.den.p.num_terms() + b.den.p.num_terms()) as u64)?;
        let l = self.mul(a.num, b.den.clone())?;
        let r = self.mul(bn, a.den.clone())?;
        let num = self.add(l, r)?;
        let den = self.mul(a.den, b.den)?;
        Ok(Fraction { num, den })
    }

    fn frac_mul(&mut self, a: Fraction, b: Fraction) -> std::result::Result<Fraction, Fail> {
        Ok(Fraction { num: self.mul(a.num, b.num)?, den: self.mul(a.den, b.den)? })
    }

    /// Binary powering; every clone of an operand is charged its weight.
    fn frac_pow(&mut self, b: Fraction, k: u64) -> std::result::Result<Fraction, Fail> {
        let mut acc = Fraction { num: one(), den: one() };
        let mut base = Some(b);
        let mut k = k;
        while k > 0 {
            let bb = match base.take() {
                Some(bb) => bb,
                None => return Err(Fail::Budget),
            };
            if k & 1 == 1 {
                self.charge(weight(&bb.num.p) + weight(&bb.den.p))?;
                let c = Fraction { num: bb.num.clone(), den: bb.den.clone() };
                acc = self.frac_mul(acc, c)?;
            }
            k >>= 1;
            if k > 0 {
                self.charge(weight(&bb.num.p) + weight(&bb.den.p))?;
                let c = Fraction { num: bb.num.clone(), den: bb.den.clone() };
                base = Some(self.frac_mul(bb, c)?);
            }
        }
        Ok(acc)
    }

    fn convert(&mut self, e: &Expr, depth: usize) -> std::result::Result<Fraction, Fail> {
        if depth > MAX_NESTING {
            return Err(Fail::Budget);
        }
        match e {
            Expr::Integer(n) => {
                let r = Rational::new(n.clone(), Integer::one()).map_err(|_| Fail::Undefined)?;
                self.constant(&r)
            }
            Expr::Rational(r) => self.constant(r),
            Expr::Real(f) => {
                let r = exact_real(*f).ok_or(Fail::Undefined)?;
                self.constant(&r)
            }
            Expr::Symbol(s) => {
                let v = self.symbol(s);
                Ok(self.atom(v))
            }
            Expr::Unary(UnaryOp::Neg, a) => {
                let r = self.convert(a, depth + 1)?;
                Ok(Fraction { num: self.negate(r.num)?, den: r.den })
            }
            Expr::Binary(BinaryOp::Add | BinaryOp::Sub, _, _) => {
                // flatten the left spine of the chain iteratively
                let mut parts: Vec<(&Expr, bool)> = Vec::new();
                let mut cur = e;
                loop {
                    match cur {
                        Expr::Binary(BinaryOp::Add, l, r) => {
                            parts.push((r.as_ref(), false));
                            cur = l.as_ref();
                        }
                        Expr::Binary(BinaryOp::Sub, l, r) => {
                            parts.push((r.as_ref(), true));
                            cur = l.as_ref();
                        }
                        _ => {
                            parts.push((cur, false));
                            break;
                        }
                    }
                }
                let mut acc: Option<Fraction> = None;
                for (x, negated) in parts.into_iter().rev() {
                    let t = self.convert(x, depth + 1)?;
                    acc = Some(match acc {
                        None if negated => Fraction { num: self.negate(t.num)?, den: t.den },
                        None => t,
                        Some(a) => self.frac_add(a, t, negated)?,
                    });
                }
                acc.ok_or(Fail::Budget)
            }
            Expr::Binary(BinaryOp::Mul, _, _) => {
                let mut parts: Vec<&Expr> = Vec::new();
                let mut cur = e;
                while let Expr::Binary(BinaryOp::Mul, l, r) = cur {
                    parts.push(r.as_ref());
                    cur = l.as_ref();
                }
                parts.push(cur);
                let mut acc: Option<Fraction> = None;
                for x in parts.into_iter().rev() {
                    let t = self.convert(x, depth + 1)?;
                    acc = Some(match acc {
                        None => t,
                        Some(a) => self.frac_mul(a, t)?,
                    });
                }
                acc.ok_or(Fail::Budget)
            }
            Expr::Binary(BinaryOp::Div, l, r) => {
                let a = self.convert(l, depth + 1)?;
                let b = self.convert(r, depth + 1)?;
                if b.num.p.is_zero() {
                    return Err(Fail::Undefined);
                }
                Ok(Fraction { num: self.mul(a.num, b.den)?, den: self.mul(a.den, b.num)? })
            }
            Expr::Binary(BinaryOp::Pow, b, k) => {
                if let Expr::Integer(k) = k.as_ref() {
                    if let Some(k) = NumericConversion::to_i64(k) {
                        if k.unsigned_abs() <= MAX_DEGREE as u64 {
                            let base = self.convert(b, depth + 1)?;
                            let r = self.frac_pow(base, k.unsigned_abs())?;
                            if k < 0 {
                                if r.num.p.is_zero() {
                                    return Err(Fail::Undefined);
                                }
                                return Ok(Fraction { num: r.den, den: r.num });
                            }
                            return Ok(r);
                        }
                    }
                }
                let v = self.opaque(e)?;
                Ok(self.atom(v))
            }
            Expr::Function(_, _) | Expr::Unary(_, _) | Expr::Binary(BinaryOp::Mod, _, _) => {
                let v = self.opaque(e)?;
                Ok(self.atom(v))
            }
        }
    }
}

/// The exact value of a finite f64, a dyadic rational m·2^e decoded from its bits. `None` for
/// NaN and infinities. (`Rational::from_f64` is not used: it returns an approximation.)
fn exact_real(f: f64) -> Option<Rational> {
    if !f.is_finite() {
        return None;
    }
    if f == 0.0 {
        return Some(Rational::zero());
    }
    let bits = f.to_bits();
    let negative = bits >> 63 == 1;
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1u64 << 52) - 1);
    let (m, e) = if biased == 0 {
        (fraction, -1074i64)
    } else {
        (fraction | (1u64 << 52), biased - 1075)
    };
    let mut num = Integer::from(m);
    if negative {
        num = -num;
    }
    let two = Integer::from(2i64);
    // |e| <= 1074, so the exponents fit in u32
    if e >= 0 {
        Rational::new(num * two.pow(e as u32), Integer::one()).ok()
    } else {
        Rational::new(num, two.pow((-e) as u32)).ok()
    }
}

/// Number of nodes of `e` (iterative). Err when it exceeds `limit`.
fn node_count(e: &Expr, limit: u64) -> std::result::Result<u64, Fail> {
    let mut stack = vec![e];
    let mut n = 0u64;
    while let Some(x) = stack.pop() {
        n += 1;
        if n > limit {
            return Err(Fail::Budget);
        }
        match x {
            Expr::Binary(_, l, r) => {
                stack.push(l);
                stack.push(r);
            }
            Expr::Unary(_, a) => stack.push(a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| a.as_ref())),
            _ => {}
        }
    }
    Ok(n)
}

/// The prescan (iterative, one unit a node). `Undefined` when the symbol `oo` or a NaN or
/// infinite `Real` occurs anywhere; `Budget` when the tree has more than `limit` nodes or is
/// deeper than `MAX_TREE_DEPTH`. Otherwise the node count and whether the tree has a division
/// (a `Div`, a negative `Integer` power or a power with another exponent).
fn prescan(e: &Expr, limit: u64) -> std::result::Result<(u64, bool), Fail> {
    let mut stack = vec![(e, 1usize)];
    let mut n = 0u64;
    let mut division = false;
    while let Some((x, d)) = stack.pop() {
        n += 1;
        if n > limit || d > MAX_TREE_DEPTH {
            return Err(Fail::Budget);
        }
        match x {
            Expr::Real(f) if !f.is_finite() => return Err(Fail::Undefined),
            Expr::Symbol(s) if s.name() == reserved::INFINITY => return Err(Fail::Undefined),
            Expr::Binary(op, l, r) => {
                if *op == BinaryOp::Div {
                    division = true;
                }
                if *op == BinaryOp::Pow {
                    match r.as_ref() {
                        Expr::Integer(k) if k.signum() >= 0 => {}
                        _ => division = true,
                    }
                }
                stack.push((l, d + 1));
                stack.push((r, d + 1));
            }
            Expr::Unary(_, a) => stack.push((a, d + 1)),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| (a.as_ref(), d + 1))),
            _ => {}
        }
    }
    Ok((n, division))
}

/// Decides whether `e` is the zero function (see the module documentation for the semantics
/// and the precondition). Equals `zero_test_with_budget(e, ZERO_TEST_WORK)`.
///
/// ```
/// use rustmath_manifolds::{zero_test, ZeroTest};
/// use rustmath_symbolic::Expr;
///
/// let x = Expr::symbol("x");
/// let y = Expr::symbol("y");
/// let e = (x.clone() + y.clone()).pow(Expr::from(2))
///     - x.clone() * x.clone() - Expr::from(2) * x.clone() * y.clone() - y.clone() * y;
/// assert_eq!(zero_test(&e), ZeroTest::Zero);
/// assert_eq!(zero_test(&(x.clone().sin() * x.clone().sin() + x.clone().cos() * x.cos()
///     - Expr::from(1))), ZeroTest::Unresolved);
/// ```
pub fn zero_test(e: &Expr) -> ZeroTest {
    zero_test_with_budget(e, ZERO_TEST_WORK)
}

/// [`zero_test`] with another work budget. A smaller budget can only turn answers into
/// `Unresolved`.
pub fn zero_test_with_budget(e: &Expr, work: u64) -> ZeroTest {
    let (nodes, division) = match prescan(e, work) {
        Ok(x) => x,
        Err(_) => return ZeroTest::Unresolved,
    };
    match e {
        Expr::Integer(n) => {
            return if n.is_zero() { ZeroTest::Zero } else { ZeroTest::NonZero };
        }
        Expr::Rational(r) => {
            return if r.is_zero() { ZeroTest::Zero } else { ZeroTest::NonZero };
        }
        // finite: the prescan refused the others
        Expr::Real(f) => {
            return if *f == 0.0 { ZeroTest::Zero } else { ZeroTest::NonZero };
        }
        // a - a is zero wherever a is defined; without a division a is defined everywhere
        Expr::Binary(BinaryOp::Sub, l, r) if !division && l == r => return ZeroTest::Zero,
        _ => {}
    }
    let mut c = Conv { left: work - nodes, syms: HashMap::new(), opaque: Vec::new(), kinds: Vec::new() };
    match c.convert(e, 0) {
        Ok(f) if f.num.p.is_zero() => ZeroTest::Zero,
        Ok(f) => {
            // atoms that cancelled do not count
            let mut vars = f.num.p.variables();
            vars.extend(f.den.p.variables());
            vars.sort_unstable();
            vars.dedup();
            let opaque = vars.iter().any(|&v| c.kinds[v] == Kind::Opaque);
            let named = vars.iter().filter(|&&v| c.kinds[v] == Kind::Named).count();
            if !opaque && named <= 1 {
                ZeroTest::NonZero
            } else {
                ZeroTest::Unresolved
            }
        }
        Err(_) => ZeroTest::Unresolved,
    }
}

/// False when the symbol `oo` or a NaN or infinite `Real` occurs anywhere in `e`.
///
/// Iterative. After `MEMO_AFTER` visited nodes, a child held by an `Arc` with a strong count
/// above 1 is visited once, keyed by its address. A child with strong count 1 is reachable only
/// through its single parent, so the scan is linear in the number of distinct nodes.
pub fn is_finite_expr(e: &Expr) -> bool {
    fn push<'a>(stack: &mut Vec<&'a Expr>, seen: &mut Option<HashSet<usize>>, c: &'a Arc<Expr>) {
        if let Some(seen) = seen.as_mut() {
            if Arc::strong_count(c) > 1 && !seen.insert(Arc::as_ptr(c) as usize) {
                return;
            }
        }
        stack.push(c.as_ref());
    }

    match e {
        Expr::Real(f) => return f.is_finite(),
        Expr::Symbol(s) => return s.name() != reserved::INFINITY,
        Expr::Integer(_) | Expr::Rational(_) => return true,
        _ => {}
    }
    let mut seen: Option<HashSet<usize>> = None;
    let mut visited = 0usize;
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(x) = stack.pop() {
        visited += 1;
        if visited == MEMO_AFTER {
            seen = Some(HashSet::new());
        }
        match x {
            Expr::Real(f) if !f.is_finite() => return false,
            Expr::Symbol(s) if s.name() == reserved::INFINITY => return false,
            Expr::Binary(_, l, r) => {
                push(&mut stack, &mut seen, l);
                push(&mut stack, &mut seen, r);
            }
            Expr::Unary(_, a) => push(&mut stack, &mut seen, a),
            Expr::Function(_, args) => {
                for a in args {
                    push(&mut stack, &mut seen, a);
                }
            }
            _ => {}
        }
    }
    true
}

/// `Err(ValidationError)` when `e` is not finite ([`is_finite_expr`]); this check comes first,
/// because `simplify` folds `0 * oo` to 0. Otherwise `e.simplify()`, replaced by the literal 0
/// when [`zero_test`] decides it is zero.
pub(crate) fn normalize(e: Expr) -> Result<Expr> {
    if !is_finite_expr(&e) {
        return Err(ManifoldError::ValidationError(format!("the expression {} is not finite", e)));
    }
    let s = e.simplify();
    if matches!(s, Expr::Integer(_) | Expr::Rational(_)) {
        return Ok(s);
    }
    if zero_test(&s) == ZeroTest::Zero {
        Ok(Expr::from(0))
    } else {
        Ok(s)
    }
}

/// The balanced sum of the signed terms (`true` means negated). The tree has depth
/// ceil(log2 m) + 1 and only `Add` and `Sub` nodes (one `Neg` at the root when every term is
/// negated), so no consumer recurses once per term. `None` for no terms.
pub(crate) fn sum_signed(mut terms: Vec<(Expr, bool)>) -> Option<Expr> {
    if terms.is_empty() {
        return None;
    }
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut it = terms.into_iter();
        while let Some((a, na)) = it.next() {
            match it.next() {
                Some((b, nb)) => next.push(match (na, nb) {
                    (false, false) => (a + b, false),
                    (true, true) => (a + b, true),
                    (false, true) => (a - b, false),
                    (true, false) => (b - a, false),
                }),
                None => next.push((a, na)),
            }
        }
        terms = next;
    }
    terms.pop().map(|(e, negated)| if negated { -e } else { e })
}

/// Whether rustmath-symbolic's `differentiate` is not the derivative on this node
/// (`differentiate.rs:90` Mod, `:184-188` Gamma, `:191` Factorial, `:194-198` Erf, whose
/// coefficient 500/443 approximates 2/sqrt(π), `:204` Zeta, `:230-234` every `Function` but
/// `root_sum` with two arguments). The two-argument `root_sum` is handled by its first
/// argument in [`unsupported_derivative`].
fn derivative_unsupported_kind(e: &Expr) -> bool {
    match e {
        Expr::Function(name, args) => !(name == "root_sum" && args.len() == 2),
        Expr::Unary(UnaryOp::Gamma | UnaryOp::Factorial | UnaryOp::Zeta | UnaryOp::Erf, _) => true,
        Expr::Binary(BinaryOp::Mod, _, _) => true,
        _ => false,
    }
}

/// The outermost subterm of `e` that contains `var` and is a `Function` other than `root_sum`
/// with two arguments, a `Unary` Gamma, Factorial, Zeta or Erf, or a `Binary` Mod; or a
/// two-argument `root_sum(f, g)` whose f contains `var`. `None` when there is none.
///
/// `differentiate.rs:215-221` returns 0 for `root_sum(f, g)` when f contains `var`, trusting
/// that `var` is then the bound variable. The roots of f may move with `var`
/// (`root_sum(t^2 - x, t^2)` is 2x), so `var` anywhere in f makes the `root_sum` itself the
/// offending subterm. This also refuses a `var` that is the bound symbol, where 0 is right: the
/// scan cannot tell which symbol of f is bound. The Risch integrator binds `_t` (or `_t0`),
/// which no chart in the workspace uses.
///
/// Iterative pre-order scan; each stack entry carries its outermost unsupported ancestor. After
/// `MEMO_AFTER` visited nodes, a child held by an `Arc` with strong count above 1 is pushed at
/// most once per (address, below an ancestor) pair, and not again below no ancestor once it was
/// visited below one (a visit below an ancestor finds every occurrence of `var`). So the scan
/// is linear in the number of distinct nodes.
pub(crate) fn unsupported_derivative<'a>(e: &'a Expr, var: &Symbol) -> Option<&'a Expr> {
    type Stack<'a> = Vec<(&'a Expr, Option<&'a Expr>)>;
    fn push<'a>(
        stack: &mut Stack<'a>,
        seen: &mut Option<HashSet<(usize, bool)>>,
        c: &'a Arc<Expr>,
        ancestor: Option<&'a Expr>,
    ) {
        if let Some(seen) = seen.as_mut() {
            if Arc::strong_count(c) > 1 {
                let key = Arc::as_ptr(c) as usize;
                if seen.contains(&(key, true)) || !seen.insert((key, ancestor.is_some())) {
                    return;
                }
            }
        }
        stack.push((c.as_ref(), ancestor));
    }

    // a leaf has no subterm of an unsupported kind: no allocation
    if matches!(e, Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_)) {
        return None;
    }
    let mut seen: Option<HashSet<(usize, bool)>> = None;
    let mut visited = 0usize;
    let mut stack: Stack<'a> = vec![(e, None)];
    while let Some((x, ancestor)) = stack.pop() {
        visited += 1;
        if visited == MEMO_AFTER {
            seen = Some(HashSet::new());
        }
        let ancestor = if ancestor.is_none() && derivative_unsupported_kind(x) { Some(x) } else { ancestor };
        match x {
            Expr::Symbol(s) => {
                if s == var {
                    if let Some(a) = ancestor {
                        return Some(a);
                    }
                }
            }
            Expr::Binary(_, l, r) => {
                push(&mut stack, &mut seen, l, ancestor);
                push(&mut stack, &mut seen, r, ancestor);
            }
            Expr::Unary(_, a) => push(&mut stack, &mut seen, a, ancestor),
            Expr::Function(name, args) if name == "root_sum" && args.len() == 2 => {
                push(&mut stack, &mut seen, &args[0], ancestor.or(Some(x)));
                push(&mut stack, &mut seen, &args[1], ancestor);
            }
            Expr::Function(_, args) => {
                for a in args {
                    push(&mut stack, &mut seen, a, ancestor);
                }
            }
            _ => {}
        }
    }
    None
}

/// `Ok(())` when [`unsupported_derivative`] finds nothing. Otherwise
/// `Err(NotImplemented)` naming the offending subterm and the variable: rustmath-symbolic's
/// `differentiate` would return a fabricated value for it (see the module references).
pub(crate) fn check_derivative(e: &Expr, var: &Symbol) -> Result<()> {
    match unsupported_derivative(e, var) {
        None => Ok(()),
        Some(u) => Err(ManifoldError::NotImplemented(format!(
            "the derivative of {} with respect to {} is not available: rustmath-symbolic's differentiate does not implement it",
            u,
            var.name()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_symbolic::parse;
    use std::time::{Duration, Instant};

    fn sym(s: &str) -> Expr {
        Expr::symbol(s)
    }

    fn int(n: i64) -> Expr {
        Expr::from(n)
    }

    fn rat(n: i64, d: i64) -> Expr {
        Expr::Rational(Rational::new(n, d).unwrap())
    }

    fn sum_of(terms: Vec<Expr>) -> Expr {
        sum_signed(terms.into_iter().map(|t| (t, false)).collect()).unwrap()
    }

    /// Runs `f` in a thread with a 2 MB stack, as `cargo test` does.
    fn small_stack<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new().stack_size(2 << 20).spawn(f).unwrap().join().unwrap();
    }

    #[test]
    fn test_zero_test_cases() {
        let (x, y) = (sym("x"), sym("y"));
        let (pi, e, i) = (sym("pi"), sym("e"), sym("I"));
        let zero = vec![
            (x.clone() + y.clone()).pow(int(2)) - x.clone().pow(int(2)) - int(2) * x.clone() * y.clone()
                - y.clone().pow(int(2)),
            int(2) * x.clone() - x.clone() * int(2),
            x.clone() / y.clone() - x.clone() * y.clone().pow(int(-1)),
            x.clone().sin() - x.clone().sin(),
            x.clone().pow(int(1000)) - x.clone().pow(int(1000)),
            x.clone().pow(Expr::Integer(Integer::from(10).pow(30)))
                - x.clone().pow(Expr::Integer(Integer::from(10).pow(30))),
            Expr::Real(0.5) - Expr::Real(0.5),
            x.clone().pow(int(0)) - int(1),
            i.clone() * x.clone() - x.clone() * i.clone(),
            pi.clone() * x.clone() - x.clone() * pi.clone(),
            pi.clone() * e.clone() - e.clone() * pi.clone(),
            Expr::Real(0.5) * x.clone() - x.clone() / int(2),
            Expr::Real(0.0),
        ];
        for z in &zero {
            assert_eq!(zero_test(z), ZeroTest::Zero, "{}", z);
        }
        let nonzero = vec![
            x.clone() - y.clone(),
            pi.clone() * x.clone(),
            pi.clone() - int(3),
            e.clone() * x.clone() + int(1),
            (pi.clone() - pi.clone()) + e.clone() * x.clone(),
            x.clone().sin() - x.clone().sin() + y.clone(),
            Expr::Real(0.1) * x.clone() - x.clone() / int(10),
            Expr::Real(0.1 + 0.2 - 0.3),
            Expr::Real(0.1) * x.clone(),
        ];
        for z in &nonzero {
            assert_eq!(zero_test(z), ZeroTest::NonZero, "{}", z);
        }
        let i_i = parse("I*I + 1").unwrap();
        let sum64: Vec<Expr> = (0..65).map(|k| (x.clone() + int(k)).sin()).collect();
        let s65 = sum_of(sum64);
        let unresolved = vec![
            x.clone().sin().pow(int(2)) + x.clone().cos().pow(int(2)) - int(1),
            x.clone().sin(),
            x.clone() / (y.clone() - y.clone()),
            (int(0) * x.clone()).pow(int(-1)),
            x.clone().pow(rat(1, 2)) * x.clone().pow(rat(1, 2)) - x.clone(),
            i_i.clone(),
            i_i.simplify(),
            parse("i^2 + 1").unwrap(),
            parse("I").unwrap(),
            sym("oo") - sym("oo"),
            int(0) * parse("oo").unwrap(),
            Expr::Real(f64::INFINITY) * x.clone() - Expr::Real(f64::INFINITY) * x.clone(),
            Expr::Real(f64::NAN) * x.clone() - Expr::Real(f64::NAN) * x.clone(),
            sym("oo").sin() - sym("oo").sin(),
            parse("infinity").unwrap() - parse("inf").unwrap(),
            pi.clone() - e.clone(),
            int(1) / (y.clone() - y.clone()) - int(1) / (y.clone() - y.clone()),
            s65.clone() - s65 * int(1),
        ];
        for z in &unresolved {
            assert_eq!(zero_test(z), ZeroTest::Unresolved, "{}", z);
        }
        // the same sum with 64 terms is decided (MAX_OPAQUE = 64)
        let s64 = sum_of((0..64).map(|k| (x.clone() + int(k)).sin()).collect());
        assert_eq!(zero_test(&(s64.clone() - s64 * int(1))), ZeroTest::Zero);
    }

    #[test]
    fn test_zero_test_budget() {
        let x = sym("x");
        // s = 1 + x + ... + x^49; s^64 - (s*1)^64 (53 s on the first prototype)
        let s = sum_of((0..50).map(|k| x.clone().pow(int(k))).collect());
        let t = Instant::now();
        let e = s.clone().pow(int(64)) - (s.clone() * int(1)).pow(int(64));
        assert_eq!(zero_test(&e), ZeroTest::Unresolved);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());

        let s10 = sum_of((1..=10).map(|k| sym(&format!("x{}", k))).collect());
        let e10 = s10.clone().pow(int(10)) - (s10.clone() * int(1)).pow(int(10));
        assert_eq!(zero_test(&e10), ZeroTest::Unresolved);
        let e4 = s10.clone().pow(int(4)) - (s10.clone() * int(1)).pow(int(4));
        assert_eq!(zero_test(&e4), ZeroTest::Zero);

        let y = sym("y");
        let binom = (x.clone() + y.clone()).pow(int(2)) - x.clone().pow(int(2))
            - int(2) * x.clone() * y.clone() - y.clone().pow(int(2));
        assert_eq!(zero_test_with_budget(&binom, 10), ZeroTest::Unresolved);
        assert_eq!(zero_test(&binom), ZeroTest::Zero);

        // 30000 distinct symbols, built balanced (a left-deep sum this long aborts the test
        // binary when it is dropped in a 2 MB thread)
        small_stack(|| {
            let big = sum_of((0..30000).map(|k| sym(&format!("a{}", k))).collect());
            let t = Instant::now();
            assert_eq!(zero_test(&(big.clone() - big * int(1))), ZeroTest::Unresolved);
            assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
        });

        // sin applied 4000 and 5000 times (MAX_TREE_DEPTH = 4096), in a 2 MB thread
        small_stack(|| {
            let x = sym("x");
            let mut c = x.clone();
            for _ in 0..4000 {
                c = c.sin();
            }
            assert_eq!(zero_test(&(c.clone() - c.clone() * int(1))), ZeroTest::Zero);
            for _ in 0..1000 {
                c = c.sin();
            }
            assert_eq!(zero_test(&(c.clone() - c * int(1))), ZeroTest::Unresolved);
        });

        // rational coefficients with 1000-bit denominators
        let big = Integer::from(2).pow(999);
        let terms: Vec<Expr> = (0..60i64)
            .map(|i| {
                let c = Rational::new(Integer::from(2 * i + 1), big.clone() + Integer::from(2 * i + 3)).unwrap();
                Expr::Rational(c) * x.clone().pow(int(i))
            })
            .collect();
        let r = sum_of(terms);
        let t = Instant::now();
        let e = r.clone().pow(int(2)) - (r * int(1)).pow(int(2));
        assert_eq!(zero_test(&e), ZeroTest::Unresolved);
        assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());

        // negation is charged (review 2): P has 10^4 terms
        let a = sum_of((0..100).map(|k| sym(&format!("a{}", k))).collect());
        let b = sum_of((0..100).map(|k| sym(&format!("b{}", k))).collect());
        let p = a * b;
        let mut nested = p.clone();
        for _ in 0..197 {
            nested = -nested;
        }
        let t = Instant::now();
        assert_eq!(zero_test(&nested), ZeroTest::Unresolved);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        let mut right = p.clone();
        for _ in 0..197 {
            right = int(0) - right;
        }
        let t = Instant::now();
        assert_eq!(zero_test(&right), ZeroTest::Unresolved);
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert_eq!(zero_test(&(-p)), ZeroTest::NonZero);
        let mut small = x.clone() - y.clone();
        for _ in 0..197 {
            small = -small;
        }
        assert_eq!(zero_test(&small), ZeroTest::NonZero);
        assert_eq!(zero_test(&(small + (x - y))), ZeroTest::Zero);
    }

    /// 40 doublings of a leaf L, alternating `e + e` and `e * e`: the tree has 2^40 copies of
    /// L, the DAG fewer than 100 nodes.
    fn doubled(leaf: &Expr, shared: bool) -> Expr {
        let mut e = leaf.clone();
        for k in 0..40 {
            let op = if k % 2 == 0 { BinaryOp::Add } else { BinaryOp::Mul };
            e = if shared {
                let a = Arc::new(e);
                Expr::Binary(op, a.clone(), a)
            } else {
                Expr::Binary(op, Arc::new(e.clone()), Arc::new(e))
            };
        }
        e
    }

    #[test]
    fn test_scans_on_shared_subtrees() {
        let (x, y) = (sym("x"), sym("y"));
        let (sx, sy) = (Symbol::new("x"), Symbol::new("y"));
        let t2 = sym("t").pow(int(2));
        let root_sum = Expr::Function("root_sum".into(), vec![Arc::new(t2 - x.clone()), Arc::new(sym("t"))]);
        let gamma_x = x.clone().gamma();
        for shared in [false, true] {
            let t = Instant::now();
            let e = doubled(&(x.clone() + y.clone()), shared);
            assert!(is_finite_expr(&e));
            assert!(unsupported_derivative(&e, &sx).is_none());
            let e = doubled(&(y.clone().gamma() + x.clone()), shared);
            assert!(unsupported_derivative(&e, &sx).is_none());
            let e = doubled(&(gamma_x.clone() + y.clone()), shared);
            assert_eq!(unsupported_derivative(&e, &sx), Some(&gamma_x));
            let e = doubled(&(x.clone() + sym("oo")), shared);
            assert!(!is_finite_expr(&e));
            let e = doubled(&(root_sum.clone() + y.clone()), shared);
            assert_eq!(unsupported_derivative(&e, &sx), Some(&root_sum));
            assert!(unsupported_derivative(&e, &sy).is_none());
            assert!(t.elapsed() < Duration::from_secs(1), "{:?}", t.elapsed());
        }
    }

    #[test]
    fn test_normalize_and_finiteness() {
        let (x, y) = (sym("x"), sym("y"));
        assert!(!is_finite_expr(&sym("oo")));
        assert!(!is_finite_expr(&sym("oo").sin()));
        assert!(!is_finite_expr(&(Expr::Real(f64::NAN) * x.clone())));
        assert!(!is_finite_expr(&Expr::Real(f64::NEG_INFINITY)));
        assert!(is_finite_expr(&(x.clone() + sym("pi") + sym("I"))));
        // simplify alone folds 0 * oo to 0
        assert_eq!((int(0) * sym("oo")).simplify(), int(0));
        assert!(matches!(normalize(int(0) * sym("oo")), Err(ManifoldError::ValidationError(_))));
        assert_eq!(normalize(int(2) * x.clone() - x.clone() * int(2)).unwrap(), int(0));
        let d = x.clone() - y.clone();
        assert_eq!(normalize(d.clone()).unwrap(), d.simplify());
        assert_eq!(normalize(d.clone()).unwrap(), d);
    }

    #[test]
    fn test_sum_signed_shape() {
        let terms: Vec<(Expr, bool)> = (0..5).map(|k| (sym(&format!("a{}", k)), k % 2 == 1)).collect();
        let s = sum_signed(terms).unwrap();
        // a0 - a1 + a2 - a3 + a4
        let reference = sym("a0") - sym("a1") + sym("a2") - sym("a3") + sym("a4");
        assert_eq!(zero_test(&(s - reference)), ZeroTest::Zero);
        let all_neg = sum_signed(vec![(sym("a"), true), (sym("b"), true)]).unwrap();
        assert_eq!(zero_test(&(all_neg + sym("a") + sym("b"))), ZeroTest::Zero);
        assert!(sum_signed(Vec::new()).is_none());
    }

    #[test]
    fn test_exact_real() {
        // 0.1 = 3602879701896397 / 2^55 (gp: bestappr is not used; exact binary value)
        let r = exact_real(0.1).unwrap();
        assert_eq!(r, Rational::new(Integer::from(3602879701896397i64), Integer::from(2).pow(55)).unwrap());
        assert_eq!(exact_real(-2.0).unwrap(), Rational::new(Integer::from(-2), Integer::from(1)).unwrap());
        assert_eq!(exact_real(5e-324).unwrap(), Rational::new(Integer::from(1), Integer::from(2).pow(1074)).unwrap());
        assert!(exact_real(f64::NAN).is_none());
    }
}
