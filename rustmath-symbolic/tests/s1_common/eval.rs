//! Numeric evaluation of the engine's coefficients, for comparison with PARI/GP (test code only).
//!
//! The coefficients of an expansion are exact constants rendered as `Expr` trees: rationals, `pi`,
//! `e`, `I`, and kernels such as `arcsin(1)`, `exp(2)`, `(1 + sin(1))^500`, `(-2)^(1/3)`. `eval`
//! walks the tree once per distinct node (memoized by address, with an explicit stack, so a DAG
//! of shared kernels costs its DAG) and evaluates it with MPC at `PREC` bits through
//! `rustmath_complex::ComplexMPFR`.
//!
//! Branches: the convention these tests fix. Every function takes its principal value, as the
//! engine does (design §2.2 `apply`, §2.4 "Branch at the point"). The frozen design does not say
//! which side of a branch cut a kernel on the cut denotes (report `tests_report.md`, finding 4;
//! the port must state it). These tests take gp 2.15's scalar functions, which are the
//! logarithmic formulas with the principal logarithm and the principal square root (argument in
//! (-pi, pi], no signed zeros):
//!
//! - arcsin z = -I log(I z + sqrt(1 - z^2)), arccos z = pi/2 - arcsin z;
//! - arctanh z = (log(1 + z) - log(1 - z))/2 (not (1/2) log((1 + z)/(1 - z)), which differs on
//!   the cut: at z = 3/2 it is 0.805 + 1.571*I);
//! - arccosh z = log(z + sqrt(z + 1) sqrt(z - 1)) (design §2.4), arcsinh z = log(z + sqrt(1 + z^2)),
//!   log and the powers z^a = exp(a log z) with the principal log.
//!
//! gp (`default(realprecision, 40)`): `asin(3/2)` = 1.5707963267948966192 - 0.96242365011920689500*I,
//! `acos(3/2)` = 0.96242365011920689500*I, `atanh(3/2)` = 0.80471895621705018730 -
//! 1.5707963267948966192*I, `asin(-3/2)` = -1.5707963267948966192 + 0.96242365011920689500*I,
//! `acos(-3/2)` = 3.1415926535897932385 - 0.96242365011920689500*I, `atanh(-3/2)` =
//! -0.80471895621705018730 + 1.5707963267948966192*I; `(log(1+3/2)-log(1-3/2))/2` gives the
//! `atanh(3/2)` value, `1/2*log(-5)` the other side.
//!
//! For arcsin and arccos at a real c with |c| > 1 this is the only side that continues with the
//! engine's next coefficient, the principal power of the integrand at the constant term: the
//! engine's c1 of `arcsin(x)` at 3/2 is `(-5/4)^(-1/2)` = -0.89442719099991587856*I, the derivative
//! of pi/2 - I*acosh(x) (the limit from below). For arctanh the integrand 1/(1 - z^2) has no cut,
//! so the side is the convention above. A port that renders such a constant in another form must
//! give the same value, or the rows on a cut (`c:arcsin@1:0`, `c:arccos@1:0`, `c:arctanh@1:0`,
//! `c:arctanh@-1:1` of the matrix, and fuzz rows such as `B1:y14.33`, atan(atanh(3))) change, with
//! their gp references.
//!
//! In MPC: for a real argument above 1, arcsin, arccos and arctanh with a -0 imaginary part give
//! these values (MPC's default for a real argument, a +0 imaginary part, is the limit from above);
//! every other real argument on a cut (log and sqrt of c < 0, arccosh of c < 1, arcsin, arccos and
//! arctanh of c < -1) is the limit from above in gp and in MPC. Every computed value has its zero
//! parts normalized to +0 (`canon`), so the side of a cut never depends on the sign of a rounding
//! zero.
//!
//! `eval_scaled` evaluates the same trees as m * exp(t) (|m| = 1, t real), for constants beyond
//! MPFR's default exponent range: `F_3` of red team 3 structure H2 at 1 is about exp(2.04e12)
//! (`rt_structure_h2_small_k_values` compares log(c0) and c1/c0 with gp).
use rustmath_complex::ComplexMPFR;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_symbolic::{BinaryOp, Expr, UnaryOp};
use std::collections::HashMap;

/// Working precision of the evaluation, in bits (308 decimal digits; gp's references are
/// computed at 300 digits and compared to 60).
pub const PREC: u32 = 1024;

pub type Num = ComplexMPFR;

pub fn zero() -> Num {
    ComplexMPFR::zero_with_prec(PREC)
}

pub fn from_rational(q: &Rational) -> Num {
    ComplexMPFR::with_val_rationals(PREC, q, &Rational::from(0))
}

pub fn from_parts(re: &Rational, im: &Rational) -> Num {
    ComplexMPFR::with_val_rationals(PREC, re, im)
}

fn from_integer(n: &Integer) -> Num {
    ComplexMPFR::with_val_integers(PREC, n, &Integer::from(0))
}

fn is_real(z: &Num) -> bool {
    z.imag_part() == zero().real_part()
}

fn is_zero(z: &Num) -> bool {
    is_real(z) && z.real_part() == zero().real_part()
}

/// z with a -0 imaginary part when z is real and above 1 (the side of the cut gp takes, see the
/// module comment), else z.
fn below_cut_above_one(z: &Num) -> Num {
    let one = from_rational(&Rational::from(1));
    if is_real(z) && z.real_part() > one.real_part() {
        let neg_zero = ComplexMPFR::with_val(PREC, (0.0, -0.0)).imag_part();
        ComplexMPFR::with_val_reals(z.real_part(), neg_zero)
    } else {
        z.clone()
    }
}

/// z with every zero part made +0: arithmetic can leave a -0 (the negation of 1 is -1 - 0*I),
/// and MPC reads the sign of a zero imaginary part as the side of a branch cut, so a real
/// argument would otherwise take the side of its last operation.
fn canon(z: Num) -> Num {
    let zr = zero().real_part();
    let re_zero = z.real_part() == zr;
    let im_zero = z.imag_part() == zr;
    if !re_zero && !im_zero {
        return z;
    }
    let re = if re_zero {
        zero().real_part()
    } else {
        z.real_part()
    };
    let im = if im_zero {
        zero().real_part()
    } else {
        z.imag_part()
    };
    ComplexMPFR::with_val_reals(re, im)
}

fn finite(z: Num, what: &str) -> Result<Num, String> {
    if z.is_finite() {
        Ok(z)
    } else {
        Err(format!("{what} is not finite"))
    }
}

/// The value of a leaf (an integer, a rational, a float, `pi`, `e` or `I`).
fn leaf(n: &Expr) -> Result<Num, String> {
    Ok(match n {
        Expr::Integer(k) => from_integer(k),
        Expr::Rational(q) => from_rational(q),
        Expr::Real(f) => ComplexMPFR::with_val(PREC, (*f, 0.0)),
        Expr::Symbol(s) => match s.name() {
            "pi" => ComplexMPFR::with_val(PREC, (-1.0, 0.0)).acos(),
            "e" => ComplexMPFR::one_with_prec(PREC).exp(),
            "I" => ComplexMPFR::i_with_prec(PREC),
            other => return Err(format!("free symbol {other}")),
        },
        _ => return Err("not a leaf".into()),
    })
}

/// The value of a constant `Expr` (no free symbol other than `pi`, `e`, `I`).
pub fn eval(root: &Expr) -> Result<Num, String> {
    let mut memo: HashMap<*const Expr, Num> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    while let Some((n, children_done)) = stack.pop() {
        let key = n as *const Expr;
        if memo.contains_key(&key) {
            continue;
        }
        if !children_done {
            stack.push((n, true));
            match n {
                Expr::Binary(_, l, r) => {
                    stack.push((&**r, false));
                    stack.push((&**l, false));
                }
                Expr::Unary(_, a) => stack.push((&**a, false)),
                Expr::Function(_, args) => {
                    for a in args {
                        stack.push((&**a, false));
                    }
                }
                _ => {}
            }
            continue;
        }
        let get = |c: &Expr| {
            memo.get(&(c as *const Expr))
                .cloned()
                .expect("child evaluated")
        };
        let v = match n {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => leaf(n)?,
            Expr::Binary(op, l, r) => {
                let (a, b) = (get(l), get(r));
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => {
                        if is_zero(&b) {
                            return Err("division by 0".into());
                        }
                        a / b
                    }
                    BinaryOp::Pow => pow(&a, r, &b)?,
                    _ => return Err(format!("operator {op:?}")),
                }
            }
            Expr::Unary(op, arg) => unary(op, get(arg))?,
            Expr::Function(name, _) => return Err(format!("no evaluation of the function {name}")),
        };
        memo.insert(key, canon(v));
    }
    Ok(memo.remove(&(root as *const Expr)).expect("root evaluated"))
}

/// S1 request C-1: z = t*I with t < -1 (a zero real part, `canon` makes it +0)
fn on_lower_imaginary_cut(a: &Num) -> bool {
    let z = zero().real_part();
    let m1 = from_rational(&Rational::from(-1)).real_part();
    a.real_part() == z && a.imag_part() < m1
}

/// A unary operation on a plain value.
fn unary(op: &UnaryOp, a: Num) -> Result<Num, String> {
    let v = match op {
        UnaryOp::Neg => -a,
        UnaryOp::Sin => a.sin(),
        UnaryOp::Cos => a.cos(),
        UnaryOp::Tan => a.tan(),
        UnaryOp::Sinh => a.sinh(),
        UnaryOp::Cosh => a.cosh(),
        UnaryOp::Tanh => a.tanh(),
        UnaryOp::Exp => a.exp(),
        UnaryOp::Log => {
            if is_zero(&a) {
                return Err("log(0)".into());
            }
            a.ln()
        }
        UnaryOp::Sqrt => a.sqrt(),
        UnaryOp::Arcsin => below_cut_above_one(&a).asin(),
        UnaryOp::Arccos => below_cut_above_one(&a).acos(),
        // S1 request C-1: on the lower cut (an imaginary argument below -I) gp's principal
        // formulas are the limit from Re < 0, MPC's with a +0 real part the limit from Re > 0:
        // the odd extension -f(-z) of the upper cut gives gp's value (gp 2.15.4,
        // realprecision 40: `atan(-3*I)` = -1.5707963267948966192 - 0.34657359027997265471*I,
        // `asinh(-2*I)` = -1.3169578969248167086 - 1.5707963267948966192*I; MPC gives
        // 1.5707963267948966192 - 0.34657359027997265471*I and 1.3169578969248167086 -
        // 1.5707963267948966192*I). On the upper cut and off the cuts both agree.
        UnaryOp::Arctan if on_lower_imaginary_cut(&a) => -(canon(-a).atan()),
        UnaryOp::Arcsinh if on_lower_imaginary_cut(&a) => -(canon(-a).asinh()),
        UnaryOp::Arctan => a.atan(),
        UnaryOp::Arcsinh => a.asinh(),
        UnaryOp::Arccosh => a.acosh(),
        UnaryOp::Arctanh => below_cut_above_one(&a).atanh(),
        UnaryOp::Abs => ComplexMPFR::with_val_reals(a.abs(), zero().real_part()),
        UnaryOp::Sign => {
            if !is_real(&a) {
                return Err("sign of a non-real value".into());
            }
            let z = zero().real_part();
            let s = if a.real_part() > z {
                1
            } else if a.real_part() < z {
                -1
            } else {
                0
            };
            from_rational(&Rational::from(s))
        }
        other => return Err(format!("no evaluation of {other:?}")),
    };
    finite(v, "a function value")
}

fn pow(a: &Num, exponent: &Expr, b: &Num) -> Result<Num, String> {
    if let Expr::Integer(n) = exponent {
        if let Some(k) = n.to_i64() {
            if is_zero(a) {
                return match k {
                    0 => Ok(ComplexMPFR::one_with_prec(PREC)),
                    k if k > 0 => Ok(zero()),
                    _ => Err("0 to a negative power".into()),
                };
            }
            if let Ok(k32) = i32::try_from(k) {
                return finite(a.powi(k32), "an integer power");
            }
        }
    }
    if is_zero(a) {
        let z = zero().real_part();
        if b.real_part() > z {
            return Ok(zero());
        }
        return Err("0 to a power with real part <= 0".into());
    }
    finite(a.pow(b), "a power")
}

/// A value m * exp(t) with |m| = 1 and t real (a `Num` with a zero imaginary part), or the exact
/// 0: the representation of `eval_scaled`.
#[derive(Clone)]
pub struct Scaled {
    pub m: Num,
    pub t: Num,
    pub is_zero: bool,
}

/// |t| up to this is converted to a plain number (exp(2^24) is far inside MPFR's default exponent
/// range of 2^30 bits).
const PLAIN_LOG_LIMIT: f64 = 16_777_216.0;

fn real_num(t: &Num) -> Num {
    ComplexMPFR::with_val_reals(t.real_part(), zero().real_part())
}

impl Scaled {
    fn zero() -> Scaled {
        Scaled {
            m: zero(),
            t: zero(),
            is_zero: true,
        }
    }

    /// m * exp(t) for any nonzero m, renormalized.
    fn new(m: Num, t: Num) -> Scaled {
        if is_zero(&m) {
            return Scaled::zero();
        }
        let a = real_num(&ComplexMPFR::with_val_reals(m.abs(), zero().real_part()));
        let unit = m / a.clone();
        Scaled {
            m: canon(unit),
            t: real_num(&(t + a.ln())),
            is_zero: false,
        }
    }

    pub fn from_plain(z: Num) -> Scaled {
        Scaled::new(z, zero())
    }

    /// The plain value, when |t| is small enough.
    pub fn plain(&self) -> Result<Num, String> {
        if self.is_zero {
            return Ok(zero());
        }
        if self.t.real().abs() > PLAIN_LOG_LIMIT {
            return Err(format!(
                "a value of modulus exp({}) used where a plain number is needed",
                self.t.real()
            ));
        }
        Ok(canon(self.m.clone() * self.t.exp()))
    }

    /// log of the value, principal: t + I arg(m).
    pub fn ln(&self) -> Result<Num, String> {
        if self.is_zero {
            return Err("log(0)".into());
        }
        Ok(canon(self.t.clone() + self.m.ln()))
    }

    /// exp(w) for a plain w: modulus exp(Re w), phase exp(I Im w).
    fn exp_of(w: &Num) -> Scaled {
        let phase = ComplexMPFR::with_val_reals(zero().real_part(), w.imag_part()).exp();
        Scaled::new(phase, real_num(w))
    }

    fn add(&self, b: &Scaled) -> Scaled {
        if self.is_zero {
            return b.clone();
        }
        if b.is_zero {
            return self.clone();
        }
        let (big, small) = if self.t.real_part() >= b.t.real_part() {
            (self, b)
        } else {
            (b, self)
        };
        let d = small.t.clone() - big.t.clone();
        let sum = if d.real() < -PLAIN_LOG_LIMIT {
            big.m.clone()
        } else {
            big.m.clone() + small.m.clone() * d.exp()
        };
        Scaled::new(sum, big.t.clone())
    }

    fn neg(&self) -> Scaled {
        if self.is_zero {
            return self.clone();
        }
        Scaled {
            m: canon(-self.m.clone()),
            t: self.t.clone(),
            is_zero: false,
        }
    }

    fn mul(&self, b: &Scaled) -> Scaled {
        if self.is_zero || b.is_zero {
            return Scaled::zero();
        }
        Scaled::new(self.m.clone() * b.m.clone(), self.t.clone() + b.t.clone())
    }

    fn div(&self, b: &Scaled) -> Result<Scaled, String> {
        if b.is_zero {
            return Err("division by 0".into());
        }
        if self.is_zero {
            return Ok(Scaled::zero());
        }
        Ok(Scaled::new(
            self.m.clone() / b.m.clone(),
            self.t.clone() - b.t.clone(),
        ))
    }

    fn pow(&self, exponent: &Expr, b: &Scaled) -> Result<Scaled, String> {
        if let Expr::Integer(n) = exponent {
            if let Some(k) = n.to_i64() {
                if self.is_zero {
                    return match k {
                        0 => Ok(Scaled::from_plain(ComplexMPFR::one_with_prec(PREC))),
                        k if k > 0 => Ok(Scaled::zero()),
                        _ => Err("0 to a negative power".into()),
                    };
                }
                if let Ok(k32) = i32::try_from(k) {
                    let kk = from_integer(n);
                    return Ok(Scaled::new(self.m.powi(k32), self.t.clone() * kk));
                }
            }
        }
        let bp = b.plain()?;
        if self.is_zero {
            if bp.real_part() > zero().real_part() {
                return Ok(Scaled::zero());
            }
            return Err("0 to a power with real part <= 0".into());
        }
        Ok(Scaled::exp_of(&(bp * self.ln()?)))
    }
}

/// The value of a constant `Expr` as m * exp(t) (see `Scaled`): the same tree walk and branches
/// as `eval`, but exp, products, quotients and integer powers keep the modulus in t, so a value
/// such as exp(2.04e12) is represented. Every other function needs a plain argument.
pub fn eval_scaled(root: &Expr) -> Result<Scaled, String> {
    let mut memo: HashMap<*const Expr, Scaled> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    while let Some((n, children_done)) = stack.pop() {
        let key = n as *const Expr;
        if memo.contains_key(&key) {
            continue;
        }
        if !children_done {
            stack.push((n, true));
            match n {
                Expr::Binary(_, l, r) => {
                    stack.push((&**r, false));
                    stack.push((&**l, false));
                }
                Expr::Unary(_, a) => stack.push((&**a, false)),
                Expr::Function(_, args) => {
                    for a in args {
                        stack.push((&**a, false));
                    }
                }
                _ => {}
            }
            continue;
        }
        let get = |c: &Expr| {
            memo.get(&(c as *const Expr))
                .cloned()
                .expect("child evaluated")
        };
        let v = match n {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => {
                Scaled::from_plain(leaf(n)?)
            }
            Expr::Binary(op, l, r) => {
                let (a, b) = (get(l), get(r));
                match op {
                    BinaryOp::Add => a.add(&b),
                    BinaryOp::Sub => a.add(&b.neg()),
                    BinaryOp::Mul => a.mul(&b),
                    BinaryOp::Div => a.div(&b)?,
                    BinaryOp::Pow => a.pow(r, &b)?,
                    _ => return Err(format!("operator {op:?}")),
                }
            }
            Expr::Unary(op, arg) => {
                let a = get(arg);
                match op {
                    UnaryOp::Neg => a.neg(),
                    UnaryOp::Exp => Scaled::exp_of(&a.plain()?),
                    UnaryOp::Log => Scaled::from_plain(a.ln()?),
                    other => Scaled::from_plain(unary(other, a.plain()?)?),
                }
            }
            Expr::Function(name, _) => return Err(format!("no evaluation of the function {name}")),
        };
        memo.insert(key, v);
    }
    Ok(memo.remove(&(root as *const Expr)).expect("root evaluated"))
}

/// The exact rational of a decimal string printed by gp: "3", "-4/5", "1.25", "-0.5e-7", "2E+3".
pub fn parse_decimal(s: &str) -> Rational {
    let s = s.trim();
    if let Some((n, d)) = s.split_once('/') {
        let n = Integer::from_decimal_str(n.trim_start_matches('+'))
            .unwrap_or_else(|| panic!("bad {s}"));
        let d = Integer::from_decimal_str(d).unwrap_or_else(|| panic!("bad {s}"));
        return Rational::new(n, d).expect("nonzero denominator");
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.trim_start_matches('+')),
    };
    let (mant, exp) = match body.find(['e', 'E']) {
        Some(i) => (
            &body[..i],
            body[i + 1..]
                .parse::<i64>()
                .unwrap_or_else(|_| panic!("bad {s}")),
        ),
        None => (body, 0),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((mant, ""));
    let digits = format!("{int_part}{frac_part}");
    let digits = if digits.is_empty() {
        "0".to_string()
    } else {
        digits
    };
    let mut num = Integer::from_decimal_str(&digits).unwrap_or_else(|| panic!("bad {s}"));
    let shift = exp - frac_part.len() as i64;
    let ten = Integer::from(10);
    let mut den = Integer::from(1);
    if shift >= 0 {
        for _ in 0..shift {
            num = num * ten.clone();
        }
    } else {
        for _ in 0..(-shift) {
            den = den * ten.clone();
        }
    }
    if neg {
        num = -num;
    }
    Rational::new(num, den).expect("nonzero denominator")
}

/// gp's coefficient "re:im" (each part as `parse_decimal` reads it).
pub fn parse_coefficient(re: &str, im: &str) -> Num {
    from_parts(&parse_decimal(re), &parse_decimal(im))
}

/// |got - want| <= 10^(-digits) * max(1, |want|).
pub fn close(got: &Num, want: &Num, digits: u32) -> bool {
    let diff = (got.clone() - want.clone()).abs();
    let scale = {
        let w = want.abs();
        let one = from_rational(&Rational::from(1)).real_part();
        if w > one {
            w
        } else {
            one
        }
    };
    let mut tol_den = Integer::from(1);
    for _ in 0..digits {
        tol_den = tol_den * Integer::from(10);
    }
    let tol = from_rational(&Rational::new(Integer::from(1), tol_den).expect("nonzero"));
    let bound = ComplexMPFR::with_val_reals(scale, zero().real_part()) * tol;
    diff <= bound.real_part()
}

/// A short decimal rendering for failure messages.
pub fn show(z: &Num) -> String {
    format!(
        "{} + {}*I",
        z.real_part().to_decimal_string(30),
        z.imag_part().to_decimal_string(30)
    )
}
