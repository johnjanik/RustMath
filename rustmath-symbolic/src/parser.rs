//! Expression parser for symbolic mathematics
//!
//! Parses mathematical expressions from strings into `Expr` trees.
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::parse;
//!
//! let expr = parse("x^2 + 3*x + 2").unwrap();
//! let expr2 = parse("sin(x) * exp(-x)").unwrap();
//! let expr3 = parse("(x + y) * z").unwrap();
//! ```
//!
//! # Future Enhancements (TODO)
//!
//! ## Unicode Mathematical Notation Support
//! - **Unicode operators**: ×, ÷, ⋅ (multiplication), √ (sqrt), ∫ (integral)
//! - **Greek letters**: Direct input like `α`, `β`, `γ` instead of "alpha", "beta"
//! - **Superscripts/subscripts**: Parse x² as x^2, x₁ as x_1
//! - **Special symbols**: ∑ (sum), ∏ (product), ∂ (partial derivative)
//! - **Combining characters**: Handle composed Unicode properly
//!
//! ## Implicit Multiplication
//! - `2x` → `2*x` (number followed by variable)
//! - `(x+1)(x-1)` → `(x+1)*(x-1)` (adjacent parentheses)
//! - `sin x` → `sin(x)` (function followed by expression)
//! - `xy` → `x*y` (adjacent variables, careful with multi-char symbols!)
//!
//! ## Enhanced Error Messages
//! - **Position tracking**: Show exactly where the error occurred
//! - **Suggestions**: "Did you mean 'sin(x)' instead of 'sine(x)'?"
//! - **Context**: Show the input with error highlighted
//! - **Recovery**: Try to continue parsing after errors
//!
//! ## Extended Mathematical Notation
//! - **Scientific notation**: `1.23e-4`, `6.02e23`
//! - **Complex numbers**: `3+4i`, `2∠45°` (polar form)
//! - **Matrices**: `[[1,2],[3,4]]` or `⌈1 2⌉` notation
//! - **Intervals**: `[0, 1]`, `(0, ∞)`, `[a, b)`
//! - **Sets**: `{1, 2, 3}`, `{x | x > 0}`
//! - **Vectors**: `⟨1, 2, 3⟩` or `<1, 2, 3>`
//!
//! ## Functions and Operators
//! - **Derivatives**: `d/dx(x^2)`, `∂f/∂x`
//! - **Integrals**: `∫ x dx`, `∫₀¹ x² dx`
//! - **Limits**: `lim(x→0) sin(x)/x`
//! - **Summations**: `∑(i=1 to n) i²`, `∑ᵢ₌₁ⁿ i²`
//! - **Products**: `∏(i=1 to n) i`
//!
//! ## Performance & Safety
//! - **Input size limits**: Prevent OOM with very long expressions
//! - **Recursion depth limits**: Prevent stack overflow
//! - **Timeout mechanism**: Prevent infinite parsing loops
//! - **Fuzzing**: Add comprehensive fuzz testing
//!
//! ## Internationalization
//! - **Decimal separators**: Support both `.` and `,` (European notation)
//! - **Function names**: Support localized function names (sin/sen/sinus)
//! - **Number formats**: Different digit grouping (1,000 vs 1.000 vs 1'000)
//!
//! ## Compatibility Modes
//! - **SymPy mode**: Match SymPy's parsing behavior exactly
//! - **Mathematica mode**: Support Mathematica-style notation
//! - **LaTeX mode**: Parse LaTeX math expressions
//! - **Strict mode**: Only allow explicit operators (no implicit multiplication)
//!
//! ## Implementation Notes
//! - Consider switching to a more powerful parser generator if needed
//! - Maintain backward compatibility with existing parse() API
//! - Add feature flags for optional parsing modes
//! - Comprehensive test suite for each new feature (1000+ test cases)

use crate::expression::{BinaryOp, Expr, UnaryOp};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0},
    combinator::{map, opt, recognize},
    error::{Error, ErrorKind},
    multi::many0,
    sequence::{delimited, pair, tuple},
    IResult,
};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::sync::Arc;

/// Parse error type
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse a mathematical expression from a string
///
/// # Examples
///
/// ```
/// use rustmath_symbolic::parse;
///
/// let expr = parse("x^2 + 3*x + 2").unwrap();
/// let expr2 = parse("sin(x) * exp(-x)").unwrap();
/// ```
pub fn parse(input: &str) -> Result<Expr, ParseError> {
    // TODO: Add input validation and safety checks
    // - Check input length (prevent OOM): if input.len() > MAX_INPUT_LENGTH
    // - Validate UTF-8 (should be guaranteed by &str, but check for edge cases)
    // - Normalize Unicode (NFC/NFD) for consistent parsing
    // - Check for null bytes or other problematic characters
    //
    // TODO: Add better error reporting
    // - Track line and column numbers
    // - Provide suggestions for common mistakes
    // - Show context around the error
    // - Support error recovery for interactive use

    depth::reset();
    let r = expression(input);
    let too_deep = depth::refused();
    depth::reset();
    match r {
        Ok(("", expr)) => Ok(expr),
        // S1 finish under the narrowed bar: a level past the limit of a text in the S1 syntax
        // (`depth`); the rule that met it failed, and so did the parse
        _ if too_deep => Err(ParseError {
            message: format!(
                "the expression nests too deeply for its syntax: after a sign run, a signed \
                 exponent such as x^-y, an integer literal outside i64, or an integer literal \
                 divided at once by something other than an integer literal (2/x), at most {} \
                 levels of parentheses and calls (or {} of ^) are read",
                depth::LIMIT / depth::NEST,
                depth::LIMIT / depth::TOWER
            ),
        }),
        Ok((remaining, _)) => Err(ParseError {
            message: format!("Unexpected input remaining: '{}'", remaining),
            // TODO: Add position information: message: format!("at position {}", input.len() - remaining.len())
        }),
        // S1 finish, minor items of the reviews fin2 (resources minor 2): the literal fold of
        // `expression` refuses a zero divisor with `Verify` (its only use of that kind); the
        // message is design §2.12's
        Err(nom::Err::Failure(e)) if e.code == ErrorKind::Verify => Err(ParseError {
            message: "division by zero in a literal".to_string(),
        }),
        // S1 fix-up round 3: the refusal of a float literal that underflows names the literal
        Err(nom::Err::Failure(e)) if e.code == ErrorKind::Float => {
            // `float` puts the literal's own text in the error
            let lit: String = e.input.chars().take(64).collect();
            Err(ParseError {
                message: format!(
                    "the float literal {lit} underflows to 0 (its f64 value is 0 but a digit is \
                     not): write 0.0, or an exact rational such as 10^-400"
                ),
            })
        }
        Err(e) => Err(ParseError {
            message: format!("Failed to parse expression: {}", e),
            // TODO: Convert nom error to more helpful message
        }),
    }
}

/// S1 finish under the narrowed bar (review 6, resources lens, minor 2): the depth of the parse.
///
/// The parser before S1 (HEAD 8a0b310) refused four constructs at once with a `ParseError`:
/// - a sign after `^` other than a literal's own (`x^-2` and `x^ -2.5` it read, `x^-y`, `x^+2`
///   and `x^--2` it refused);
/// - more than one sign before an operand other than a literal's (`--5` and `- -5` it read as
///   -(-5), `--x` and `- - -x` it refused);
/// - an integer literal outside i64: above i64::MAX, or below i64::MIN for the negative literal
///   of the two rules above (`x^-9223372036854775808` it read);
/// - a literal rational (an integer literal, then '/' at once) whose denominator is not an
///   integer literal at once: `2/3` and `2/-3` it read, `2/x`, `1/(x+1)`, `2/ 3`, `2/--3`,
///   `2/3.5` and `x^2/x` it refused (`head_denominator`).
///
/// The S1 parser reads them (design §2.12). Its recursive rules then went on into the rest of the
/// text, so `x^-x^-...^-x` of 1 500 levels, or any of these constructs before 300 nested
/// parentheses, aborted a 2 MB debug thread, where HEAD returned a `ParseError` (S1_BAR (b)).
/// Once the text has used one of these constructs (`relax`), the parse refuses to nest deeper
/// than `LIMIT` units, a level of atoms (a parenthesis or call) `NEST` units and a `^` level
/// `TOWER` unit: at most 128 levels of parentheses and calls or 1 024 levels of `^` in all (the
/// depth counts from the start of the text, so a construct met deep inside the text refuses any
/// deeper level at once). The counter is only checked after `relax`, and `relax` runs only for
/// these constructs, so a text that HEAD reads is never limited (S1 finish fix-up round 1:
/// HEAD's negative literal i64::MIN and its literal rational relaxed the limit wrongly before,
/// `Lit`). The state is per thread and reset by `parse`.
///
/// S1 finish, fix-up round 1 (fin1 resources B4): the parser no longer recurses (`expression`
/// runs on a work stack on the heap), so no text overflows its thread's stack, the re-parse of a
/// printed result included. The limit keeps HEAD's outcome, a `ParseError`, for deep texts in the
/// syntax HEAD refused, so their trees, deeper than any HEAD built from such a text, do not reach
/// the consumers that recurse (`differentiate`, `substitute`, `==`).
///
/// The exception for printed results (S1 finish, minor items of the reviews fin2: semantics m1,
/// resources minor 1). The limit also applies when a printed result is re-parsed, and a result
/// prints relaxing constructs of its own: an integer above i64::MAX (a factorial from 21! on, as
/// in the coefficient 1/21!), a `Rational` with such a denominator, or `^ -(`, which is how
/// `Display` writes the negated exponent this parser builds for HEAD's `x^-2^-2`. So a printed
/// result does not re-parse when its text holds such a construct and then nests more than
/// `LIMIT` units counted from the start of the text (more than 128 levels of parentheses and
/// calls, or 1 024 of `^`): the parse stops there with the refusal of `parse`, at once, with no
/// time or memory beyond the text read so far, on any thread. Measured (2 MB debug and 8 MB
/// release threads, `tests/s1_finish1.rs`, `fin2_a_printed_relaxing_construct_limits_the_reparse`;
/// the reviewers' examples), as the result of `taylor_with_budget` with the REPL's budget:
/// - `x` followed by 110 copies of `^-2` (332 bytes, HEAD's syntax) at 1, order 3 (100 copies
///   re-parse), and followed by 33 333 copies (100 KB) at 1, order 3;
/// - `exp(x) + sin^150(1)/(1-x)` (767 bytes, 150 nested calls of `sin` around 1) at 0 from
///   order 21, once 1/21! is printed (order 20 re-parses);
/// - `exp(x)*sin^130(1)` (659 bytes, HEAD's syntax) at 0, order 25 (order 20 re-parses);
/// - `sin(A1+...+A200)/(1-x)` with 30-digit integers Ak at 0, order 3 and up (measured at orders
///   3 and 5 177): `Display` writes the kernel's source, a left-deep chain, with a parenthesis per
///   `+`, so the first literal comes after 200 of them (100 terms re-parse).
///
/// HEAD's parser refused each of these constructs, so it read none of these printed texts either.
/// A follow-up may exempt the parser's own printed forms from the limit.
mod depth {
    use std::cell::Cell;

    /// the units a text in the S1 syntax may nest
    pub const LIMIT: u32 = 1024;
    /// the units of a level of nested atoms, a parenthesis or a call argument (`atom`)
    pub const NEST: u32 = 8;
    /// the units of one level of `^` (`power`)
    pub const TOWER: u32 = 1;

    /// S1 finish, fix-up round 1 (fin1 semantics B2, resources B1 and B2): what HEAD's `integer`
    /// rule read at the next integer literal of the text
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Lit {
        /// an atom after the signs that HEAD's `unary` took: an integer up to i64::MAX
        Atom,
        /// an atom whose '-' HEAD's `integer` took (after `^`, or the second sign of an
        /// operand: `x^-5`, `--5`): its magnitude up to 2^63 (i64::MIN)
        NegAtom,
        /// the denominator of HEAD's literal rational (`2/3`, `2/-3`), which its numerator
        /// checked (`super::head_denominator`)
        Denom,
    }

    thread_local! {
        static DEPTH: Cell<u32> = const { Cell::new(0) };
        static RELAXED: Cell<bool> = const { Cell::new(false) };
        static REFUSED: Cell<bool> = const { Cell::new(false) };
        static LIT: Cell<Lit> = const { Cell::new(Lit::Atom) };
    }

    pub fn reset() {
        DEPTH.with(|d| d.set(0));
        RELAXED.with(|r| r.set(false));
        REFUSED.with(|r| r.set(false));
        LIT.with(|l| l.set(Lit::Atom));
    }

    /// what HEAD read at the next integer literal (`number` takes it at every atom)
    #[inline(never)]
    pub fn set_lit(l: Lit) {
        LIT.with(|c| c.set(l));
    }

    /// the state `set_lit` left for this atom, and `Lit::Atom` for the next one
    #[inline(never)]
    pub fn take_lit() -> Lit {
        LIT.with(|c| c.replace(Lit::Atom))
    }

    /// whether `enter` refused a level during this parse
    pub fn refused() -> bool {
        REFUSED.with(|r| r.get())
    }

    /// the text used a construct the parser before S1 refused
    #[inline(never)]
    pub fn relax() {
        RELAXED.with(|r| r.set(true));
    }

    /// enter a level of `units`: false (and no change but `refused`) when the text has used such
    /// a construct and the level would pass `LIMIT`
    #[inline(never)]
    pub fn enter(units: u32) -> bool {
        let n = DEPTH.with(|d| d.get()).saturating_add(units);
        if n > LIMIT && RELAXED.with(|r| r.get()) {
            REFUSED.with(|r| r.set(true));
            return false;
        }
        DEPTH.with(|d| d.set(n));
        true
    }

    /// leave a level of `units` that `enter` accepted
    #[inline(never)]
    pub fn exit(units: u32) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(units)));
    }
}

/// The refusal of a level past `depth::LIMIT` (a Failure: no other rule reads the text; `parse`
/// names it by `depth::refused`). Out of line, written into the caller's result, so the
/// recursive frames hold no error value.
#[inline(never)]
fn too_deep<O>(input: &str) -> IResult<&str, O> {
    Err(nom::Err::Failure(Error::new(input, ErrorKind::TooLarge)))
}

// Whitespace handling
fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

// Unsigned integer literal of any size (S1 design §2.12; the sign is the unary rule's). The
// literal `a/b` is not a token: `multiplicative` folds a quotient of two signed integer literals.
// S1 finish: the constructs of HEAD's `integer` and `rational` rules (`depth`). `lit` says what
// HEAD read here (`depth::take_lit`): an atom up to i64::MAX, a negative literal down to
// i64::MIN (fix-up round 1, fin1 semantics B2 and resources B1: 2^63 relaxed the limit there
// too, where HEAD read -9223372036854775808), or the denominator of a literal rational, which
// its numerator checked. An atom directly followed by '/' was HEAD's literal rational, which
// needs HEAD's integer right after the '/' (fix-up round 1, fin1 resources B2: HEAD refused
// `2/x`, `1/(x+1)`, `2/ 3`, `2/3.5` and `2/--3` at once).
fn integer_literal(input: &str, lit: depth::Lit) -> IResult<&str, Integer> {
    let (rest, digits) = digit1(input)?;
    if lit != depth::Lit::Denom {
        let max = if lit == depth::Lit::NegAtom {
            "9223372036854775808"
        } else {
            "9223372036854775807"
        };
        let sig = digits.trim_start_matches('0');
        if sig.len() > 19 || (sig.len() == 19 && sig > max) {
            depth::relax();
        }
        if let Some(den) = rest.strip_prefix('/') {
            if head_denominator(den) {
                depth::set_lit(depth::Lit::Denom);
            } else {
                depth::relax();
            }
        }
    }
    match Integer::from_decimal_str(digits) {
        Some(n) => Ok((rest, n)),
        None => Err(nom::Err::Error(Error::new(input, ErrorKind::Digit))),
    }
}

// S1 finish, fix-up round 1 (fin1 resources B2): whether the text right after the '/' of a
// literal rational is HEAD's denominator (`integer`): an optional '-' and digits at once, whose
// value with the sign is in i64 and is not 0 (HEAD's `Rational::new` refused 0), and no '.'
// after the digits (HEAD's `integer` stops there, and no rule of HEAD's grammar reads a '.', so
// HEAD refused `2/3.5`). Then the port reads the same text: a sign run of one '-' and the digits.
#[inline(never)]
fn head_denominator(t: &str) -> bool {
    let (neg, t) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t),
    };
    let n = t.bytes().take_while(u8::is_ascii_digit).count();
    if n == 0 || t[n..].starts_with('.') {
        return false;
    }
    let sig = t[..n].trim_start_matches('0');
    let max = if neg {
        "9223372036854775808"
    } else {
        "9223372036854775807"
    };
    !sig.is_empty() && (sig.len() < 19 || (sig.len() == 19 && sig <= max))
}

// Parse a floating-point number (must have decimal point, no sign: the unary rule reads it)
// Examples: 3.14, 0.7, 1.5e-3, 2.0E10
fn float(input: &str) -> IResult<&str, f64> {
    // Recognize the full float pattern
    let (input, num_str) = recognize(tuple((
        digit1,
        char('.'),
        digit1,
        opt(tuple((
            alt((char('e'), char('E'))),
            opt(alt((char('+'), char('-')))),
            digit1,
        ))),
    )))(input)?;

    let num = num_str.parse::<f64>().map_err(|_| {
        nom::Err::Error(Error::new(input, ErrorKind::Float))
    })?;
    // S1 fix-up round 3 (review 3, semantics m2): a literal with a nonzero digit whose f64 value is
    // 0 (it underflows, as 1.0e-400 does) is refused, not read as the exact 0 (`x/1.0e-400` was a
    // division by zero, `1/(x + 1.0e-400)` was 1/x). The series engine refuses an overflow to inf
    // the same way. A Failure, so no other rule reads the literal's digits.
    if num == 0.0 {
        let mantissa = num_str.split(['e', 'E']).next().unwrap_or("");
        if mantissa.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
            return Err(nom::Err::Failure(Error::new(num_str, ErrorKind::Float)));
        }
    }
    Ok((input, num))
}

// Parse a variable/symbol name
fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alpha1,
        many0(alt((alphanumeric1, tag("_"))))
    ))(input)
}

// Parse a variable or mathematical constant
fn variable(input: &str) -> IResult<&str, Expr> {
    // TODO: Unicode Greek letter support
    // Currently we parse "alpha" as a string, but we should also accept "α" directly
    // This requires handling multi-byte UTF-8 characters in the identifier parser
    //
    // TODO: Subscript support
    // Parse x₁, x₂ as x_1, x_2 (requires Unicode normalization)

    map(identifier, |name: &str| {
        // Handle mathematical constants
        match name {
            "pi" | "π" => {
                // For now, use a symbol named "pi"
                // TODO: Add Expr::Pi variant for exact symbolic representation
                // This would enable exact arithmetic: pi * 2 stays as Pi(2) not Symbol("pi") * 2
                Expr::symbol("pi")
            }
            "e" => {
                // Euler's number
                // TODO: Add Expr::E variant for exact symbolic representation
                Expr::symbol("e")
            }
            "I" | "i" => {
                // Imaginary unit (note: lowercase 'i' might conflict with variables)
                // Using uppercase 'I' as primary
                // TODO: Add Expr::Complex variant for proper complex number support
                Expr::symbol("I")
            }
            "oo" | "inf" | "infinity" => {
                // Infinity
                // TODO: Add Expr::Infinity variant with sign (PositiveInfinity, NegativeInfinity)
                Expr::symbol("oo")
            }
            _ => Expr::symbol(name),
        }
    })(input)
}

// The unary operation of a one-argument call by its name.
fn unary_op_by_name(name: &str) -> Option<UnaryOp> {
    Some(match name {
        "sin" => UnaryOp::Sin,
        "cos" => UnaryOp::Cos,
        "tan" => UnaryOp::Tan,
        "sinh" => UnaryOp::Sinh,
        "cosh" => UnaryOp::Cosh,
        "tanh" => UnaryOp::Tanh,
        "arcsin" | "asin" => UnaryOp::Arcsin,
        "arccos" | "acos" => UnaryOp::Arccos,
        "arctan" | "atan" => UnaryOp::Arctan,
        "arcsinh" | "asinh" => UnaryOp::Arcsinh,
        "arccosh" | "acosh" => UnaryOp::Arccosh,
        "arctanh" | "atanh" => UnaryOp::Arctanh,
        "exp" => UnaryOp::Exp,
        "log" | "ln" => UnaryOp::Log,
        "sqrt" => UnaryOp::Sqrt,
        "abs" => UnaryOp::Abs,
        "sign" => UnaryOp::Sign,
        "gamma" => UnaryOp::Gamma,
        "factorial" => UnaryOp::Factorial,
        "erf" => UnaryOp::Erf,
        "zeta" => UnaryOp::Zeta,
        _ => return None,
    })
}

// A call: a unary operation for a known one-argument name, else a `Function` node (multi-argument
// or zero-argument calls, and unknown names).
#[inline(never)]
fn make_call(name: &str, mut args: Vec<Arc<Expr>>) -> Expr {
    if args.len() == 1 {
        if let Some(op) = unary_op_by_name(name) {
            return Expr::Unary(op, args.pop().expect("one argument"));
        }
    }
    Expr::Function(name.to_string(), args)
}

// Parse a number: a float, or an unsigned integer literal of any size. The first rule of every
// atom, so it takes what HEAD read at this atom (`depth::take_lit`, S1 finish fix-up round 1).
// Never inlined: its locals stay out of the recursive frame of `atom`.
#[inline(never)]
fn number(input: &str) -> IResult<&str, Expr> {
    let lit = depth::take_lit();
    // Try to parse as float first (with decimal point); a literal that underflows is a Failure
    // (S1 fix-up round 3), so its digits are not read again as an integer
    match float(input) {
        Ok((remaining, f)) => return Ok((remaining, Expr::Real(f))),
        Err(e @ nom::Err::Failure(_)) => return Err(e),
        Err(_) => {}
    }
    let (rest, n) = integer_literal(input, lit)?;
    Ok((rest, Expr::Integer(n)))
}

// A run of signs ('-' | '+')*, read by a loop (no recursion per sign, so 100 KB of signs costs
// no stack): whether the run is negative (an odd number of '-'), the number of signs (at most 3
// is told apart), and whether the last sign is a '-' followed at once by a digit (a negative
// literal for the parser before S1). The white space between signs is the grammar's
// (`multispace0`: space, tab, CR, LF; S1 finish: before, any Unicode white space).
#[inline(never)]
fn sign_run(input: &str) -> (&str, bool, u8, bool) {
    let mut neg = false;
    let mut n: u8 = 0;
    let mut minus_last = false;
    let mut rest = input;
    loop {
        let t = rest.trim_start_matches([' ', '\t', '\r', '\n']);
        if let Some(r) = t.strip_prefix('-') {
            neg = !neg;
            rest = r;
            minus_last = true;
        } else if let Some(r) = t.strip_prefix('+') {
            rest = r;
            minus_last = false;
        } else {
            let literal = minus_last && rest.starts_with(|c: char| c.is_ascii_digit());
            return (rest, neg, n, literal);
        }
        n = n.saturating_add(1).min(3);
    }
}

// The signs of an operand (`unary`). HEAD read one sign, or a sign and a negative literal
// (`--5`, `- -5`, `+-5`); any other run relaxes the depth limit (`depth`). The negative literal
// is HEAD's down to i64::MIN (`depth::Lit::NegAtom`, S1 finish fix-up round 1).
#[inline(never)]
fn operand_signs(input: &str) -> (&str, bool) {
    let (rest, neg, n, literal) = sign_run(input);
    if n >= 3 || (n == 2 && !literal) {
        depth::relax();
    } else if n == 2 {
        depth::set_lit(depth::Lit::NegAtom);
    }
    (rest, neg)
}

// The signs of an exponent, after `^`, and the entry into the exponent's level of the tower,
// which `make_power` leaves. HEAD read no sign, or a negative literal (`x^-2`); any other sign
// relaxes the depth limit. `None` past the limit (`depth::refused` names the reason).
fn exponent_signs(input: &str) -> Option<(&str, bool)> {
    let (rest, neg, n, literal) = sign_run(input);
    if n >= 2 || (n == 1 && !literal) {
        depth::relax();
    } else if n == 1 {
        depth::set_lit(depth::Lit::NegAtom);
    }
    if depth::enter(depth::TOWER) {
        Some((rest, neg))
    } else {
        None
    }
}

// A signed integer literal as the parser builds it: `Integer(n)` or `Neg(Integer(n))`.
fn signed_integer_literal(e: &Expr) -> Option<Integer> {
    match e {
        Expr::Integer(n) => Some(n.clone()),
        Expr::Unary(UnaryOp::Neg, i) => match &**i {
            Expr::Integer(n) => Some(-n.clone()),
            _ => None,
        },
        _ => None,
    }
}

// The white space of `multispace0` (space, tab, CR, LF) at the start of the text, skipped.
fn skip_space(input: &str) -> &str {
    input.trim_start_matches([' ', '\t', '\r', '\n'])
}

// The character `c` with the white space around it: the rest of the text after it
fn token(input: &str, c: char) -> Option<&str> {
    ws(char::<&str, Error<&str>>(c))(input)
        .ok()
        .map(|(rest, _)| rest)
}

// `^` with the white space around it: the rest of the text after it
fn caret(input: &str) -> Option<&str> {
    token(input, '^')
}

// The node of `base ^ (sign) exp`: a negated integer literal folds to a negative literal
// (`x^-2` is Pow(x, Integer(-2)); `2^-x` is Pow(2, Neg(x))). It leaves the level of the tower
// that `exponent_signs` entered (S1 finish). On an error the level is not left: an error fails
// the whole parse (no rule of this grammar recovers from one), and `parse` resets the depth.
fn make_power(base: Expr, neg: bool, exp: Expr) -> Expr {
    depth::exit(depth::TOWER);
    let exp = match (neg, &exp) {
        (false, _) => exp,
        (true, Expr::Integer(n)) => Expr::Integer(-n.clone()),
        (true, _) => Expr::Unary(UnaryOp::Neg, Arc::new(exp)),
    };
    Expr::Binary(BinaryOp::Pow, Arc::new(base), Arc::new(exp))
}

fn negate_if(neg: bool, e: Expr) -> Expr {
    if neg {
        Expr::Unary(UnaryOp::Neg, Arc::new(e))
    } else {
        e
    }
}

// One step of a `*` `/` chain. A quotient of two signed integer literals is the exact rational
// (an Integer when it divides): value-preserving, so `1/3` is still Rational(1/3) and `-2/3`
// Rational(-2/3). A zero divisor is an error (`None`).
fn product_step(op: char, l: Expr, r: Expr) -> Option<Expr> {
    if op == '/' {
        if let (Some(a), Some(b)) = (signed_integer_literal(&l), signed_integer_literal(&r)) {
            let q = Rational::new(a, b).ok()?;
            return Some(if q.denominator().is_one() {
                Expr::Integer(q.numerator().clone())
            } else {
                Expr::Rational(q)
            });
        }
    }
    let bin_op = if op == '*' { BinaryOp::Mul } else { BinaryOp::Div };
    Some(Expr::Binary(bin_op, Arc::new(l), Arc::new(r)))
}

// One open level of the text in `expression`'s work stack.
enum Frame<'a> {
    /// a sum (`additive`): the terms so far and the operator ('+' or '-') of the pending one
    Sum(Option<(Expr, char)>),
    /// a product (`multiplicative`): the factors so far and the operator ('*' or '/') of the
    /// pending one
    Product(Option<(Expr, char)>),
    /// the signs of an operand (`unary`): whether its run is negative
    Signs(bool),
    /// the base of `^` and the sign of its pending exponent (`power`)
    Power(Expr, bool),
    /// a parenthesis (`parens`)
    Paren,
    /// a call (`function_call`): its name and the arguments read so far
    Call(&'a str, Vec<Arc<Expr>>),
}

// What `expression` reads next.
enum Next<'a> {
    /// an operand (`unary`): its signs, then a `power`
    Operand(&'a str),
    /// an atom (the first one of a `power`, or of an exponent)
    Atom(&'a str),
    /// an atom was read (its level of `depth` is left here): `^`, or the end of a power, of a
    /// product and of a sum
    AtomDone(&'a str, Expr),
    /// an expression was read: what its frame reads after it
    ExprDone(&'a str, Expr),
}

/// Parse a full expression, with the white space around it:
///
/// ```text
/// expression := additive
/// additive   := multiplicative (('+' | '-') multiplicative)*      left to right
/// multiplicative := unary (('*' | '/') unary)*                    left to right
/// unary      := ('-' | '+')* power             an odd run of '-' gives one Neg
/// power      := atom ('^' ('-' | '+')* power)?  right to left: 2^3^2 = 2^(3^2)
/// atom       := number | name '(' (expression (',' expression)*)? ')' | '(' expression ')' | name
/// ```
///
/// Power binds tighter than the signs, so -x^2 = -(x^2), not (-x)^2. A quotient of two signed
/// integer literals folds to the exact rational (`product_step`), and a negated integer exponent
/// to a negative literal (`make_power`).
///
/// S1 finish, fix-up round 1 (fin1 resources B4): the rules run on an explicit work stack on
/// the heap (`Frame`), not by recursion, so the parser's use of the thread's stack does not grow
/// with the nesting of the text. A result of the series engine prints deeper than its input
/// (the sum of its terms, the product with h^k, and `Display`'s parentheses around every binary
/// node of a kernel's source), so its re-parse overflowed the thread on which its input parsed
/// (`sin^195(1)/(1-x)` at order 1000 on a 2 MB debug thread, `sin^4320(1)/(1-x)` at order 100 on
/// an 8 MB release thread, `sin(1+2+...+300)/(1-x)` in debug). Every text is read as the
/// recursive rules of the S1 parser read it (`fu2/s1impl/fix1f/pdiff/`: the same trees and
/// refusals on the fuzz texts of the reviews); where those rules backtracked from a call whose
/// arguments fail (to the name alone, whose '(' no rule then reads), this one fails at once, and
/// the text is a `ParseError` either way. The limit of `depth` still refuses deep texts in the
/// syntax HEAD refused.
fn expression(input: &str) -> IResult<&str, Expr> {
    let mut stack: Vec<Frame<'_>> = vec![Frame::Sum(None), Frame::Product(None)];
    let mut next = Next::Operand(input);
    loop {
        next = match next {
            Next::Operand(t) => {
                let (rest, neg) = operand_signs(t);
                stack.push(Frame::Signs(neg));
                Next::Atom(rest)
            }
            Next::Atom(t) => {
                // S1 finish: one level of `depth` per atom (every parenthesis and call argument is
                // read through an atom, so the levels of nested atoms are the nesting of the text)
                if !depth::enter(depth::NEST) {
                    return too_deep(t);
                }
                let s = skip_space(t);
                // `number` first, as in every atom (it takes `depth::take_lit`)
                match number(s) {
                    Ok((rest, e)) => Next::AtomDone(rest, e),
                    Err(e @ nom::Err::Failure(_)) => return Err(e),
                    Err(_) => {
                        if let Ok((after, name)) = identifier(s) {
                            match token(after, '(') {
                                Some(rest) => {
                                    // `name()`: the first argument fails at once (the recursive
                                    // rule entered its atom's level there)
                                    if !depth::enter(depth::NEST) {
                                        return too_deep(rest);
                                    }
                                    depth::exit(depth::NEST);
                                    match token(rest, ')') {
                                        Some(r) => Next::AtomDone(r, make_call(name, Vec::new())),
                                        None => {
                                            stack.push(Frame::Call(name, Vec::new()));
                                            stack.push(Frame::Sum(None));
                                            stack.push(Frame::Product(None));
                                            Next::Operand(rest)
                                        }
                                    }
                                }
                                None => {
                                    let (rest, v) = variable(s)?;
                                    Next::AtomDone(rest, v)
                                }
                            }
                        } else if let Some(rest) = token(s, '(') {
                            stack.push(Frame::Paren);
                            stack.push(Frame::Sum(None));
                            stack.push(Frame::Product(None));
                            Next::Operand(rest)
                        } else {
                            return Err(nom::Err::Error(Error::new(s, ErrorKind::Alt)));
                        }
                    }
                }
            }
            Next::AtomDone(t, e) => {
                depth::exit(depth::NEST);
                let t = skip_space(t);
                if let Some(rest) = caret(t) {
                    match exponent_signs(rest) {
                        Some((rest, neg)) => {
                            stack.push(Frame::Power(e, neg));
                            Next::Atom(rest)
                        }
                        None => return too_deep(rest),
                    }
                } else {
                    // the end of a power: the bases waiting for it (right to left), then the
                    // signs of its operand
                    let mut v = e;
                    loop {
                        match stack.pop() {
                            Some(Frame::Power(base, neg)) => v = make_power(base, neg, v),
                            Some(Frame::Signs(neg)) => {
                                v = negate_if(neg, v);
                                break;
                            }
                            _ => unreachable!("parser: a power belongs to an operand"),
                        }
                    }
                    // the operand is a factor of the product below it
                    let product = match stack.pop() {
                        Some(Frame::Product(None)) => v,
                        Some(Frame::Product(Some((l, op)))) => match product_step(op, l, v) {
                            Some(p) => p,
                            // "division by zero in a literal" (`parse` names it by the kind)
                            None => {
                                return Err(nom::Err::Failure(Error::new(t, ErrorKind::Verify)))
                            }
                        },
                        _ => unreachable!("parser: an operand belongs to a product"),
                    };
                    if let Ok((rest, op)) = ws(alt((char::<&str, Error<&str>>('*'), char('/'))))(t)
                    {
                        stack.push(Frame::Product(Some((product, op))));
                        Next::Operand(rest)
                    } else {
                        // the product is a term of the sum below it
                        let sum = match stack.pop() {
                            Some(Frame::Sum(None)) => product,
                            Some(Frame::Sum(Some((l, op)))) => {
                                let bin_op = if op == '+' {
                                    BinaryOp::Add
                                } else {
                                    BinaryOp::Sub
                                };
                                Expr::Binary(bin_op, Arc::new(l), Arc::new(product))
                            }
                            _ => unreachable!("parser: a product belongs to a sum"),
                        };
                        if let Ok((rest, op)) =
                            ws(alt((char::<&str, Error<&str>>('+'), char('-'))))(t)
                        {
                            stack.push(Frame::Sum(Some((sum, op))));
                            stack.push(Frame::Product(None));
                            Next::Operand(rest)
                        } else {
                            Next::ExprDone(skip_space(t), sum)
                        }
                    }
                }
            }
            Next::ExprDone(t, v) => match stack.pop() {
                None => return Ok((t, v)),
                Some(Frame::Paren) => match token(t, ')') {
                    Some(rest) => Next::AtomDone(rest, v),
                    None => return Err(nom::Err::Error(Error::new(t, ErrorKind::Char))),
                },
                Some(Frame::Call(name, mut args)) => {
                    args.push(Arc::new(v));
                    if let Some(rest) = token(t, ',') {
                        stack.push(Frame::Call(name, args));
                        stack.push(Frame::Sum(None));
                        stack.push(Frame::Product(None));
                        Next::Operand(rest)
                    } else {
                        match token(t, ')') {
                            Some(rest) => Next::AtomDone(rest, make_call(name, args)),
                            None => return Err(nom::Err::Error(Error::new(t, ErrorKind::Char))),
                        }
                    }
                }
                _ => unreachable!("parser: an expression belongs to a parenthesis or a call"),
            },
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer() {
        let expr = parse("42").unwrap();
        assert_eq!(expr, Expr::Integer(Integer::from(42)));
    }

    #[test]
    fn test_parse_negative_integer() {
        let expr = parse("-17").unwrap();
        assert_eq!(
            expr,
            Expr::Unary(UnaryOp::Neg, Arc::new(Expr::Integer(Integer::from(17))))
        );
    }

    #[test]
    fn test_parse_rational() {
        let expr = parse("1/3").unwrap();
        assert_eq!(
            expr,
            Expr::Rational(Rational::new(Integer::from(1), Integer::from(3)).unwrap())
        );
    }

    #[test]
    fn test_parse_float() {
        let expr = parse("3.14159").unwrap();
        if let Expr::Real(f) = expr {
            assert!((f - 3.14159).abs() < 1e-10);
        } else {
            panic!("Expected Real, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_float_small() {
        let expr = parse("0.7").unwrap();
        if let Expr::Real(f) = expr {
            assert!((f - 0.7).abs() < 1e-10);
        } else {
            panic!("Expected Real, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_float_negative() {
        let expr = parse("-1.5").unwrap();
        // Negative floats are parsed as Unary(Neg, Real(1.5))
        if let Expr::Unary(UnaryOp::Neg, inner) = &expr {
            if let Expr::Real(f) = inner.as_ref() {
                assert!((f - 1.5).abs() < 1e-10);
            } else {
                panic!("Expected inner Real, got {:?}", inner);
            }
        } else {
            panic!("Expected Unary Neg, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_float_scientific() {
        let expr = parse("1.5e-3").unwrap();
        if let Expr::Real(f) = expr {
            assert!((f - 0.0015).abs() < 1e-10);
        } else {
            panic!("Expected Real, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_float_scientific_positive_exp() {
        let expr = parse("2.0e10").unwrap();
        if let Expr::Real(f) = expr {
            assert!((f - 2.0e10).abs() < 1e5);
        } else {
            panic!("Expected Real, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_float_in_expression() {
        let expr = parse("0.7 * x").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                if let Expr::Real(f) = left.as_ref() {
                    assert!((f - 0.7).abs() < 1e-10);
                } else {
                    panic!("Expected left to be Real, got {:?}", left);
                }
                if let Expr::Symbol(s) = right.as_ref() {
                    assert_eq!(s.name(), "x");
                } else {
                    panic!("Expected right to be Symbol");
                }
            }
            _ => panic!("Expected Mul at top level"),
        }
    }

    #[test]
    fn test_parse_float_addition() {
        let expr = parse("3.14 + 2.71").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Add, left, right) => {
                if let Expr::Real(f) = left.as_ref() {
                    assert!((f - 3.14).abs() < 1e-10);
                } else {
                    panic!("Expected left to be Real");
                }
                if let Expr::Real(f) = right.as_ref() {
                    assert!((f - 2.71).abs() < 1e-10);
                } else {
                    panic!("Expected right to be Real");
                }
            }
            _ => panic!("Expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse("x").unwrap();
        if let Expr::Symbol(sym) = &expr {
            assert_eq!(sym.name(), "x");
        } else {
            panic!("Expected Symbol");
        }
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse("x + y").unwrap();
        match expr {
            Expr::Binary(BinaryOp::Add, _, _) => (),
            _ => panic!("Expected Add binary operation"),
        }
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse("x - y").unwrap();
        match expr {
            Expr::Binary(BinaryOp::Sub, _, _) => (),
            _ => panic!("Expected Sub binary operation"),
        }
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse("x * y").unwrap();
        match expr {
            Expr::Binary(BinaryOp::Mul, _, _) => (),
            _ => panic!("Expected Mul binary operation"),
        }
    }

    #[test]
    fn test_parse_division() {
        let expr = parse("x / y").unwrap();
        match expr {
            Expr::Binary(BinaryOp::Div, _, _) => (),
            _ => panic!("Expected Div binary operation"),
        }
    }

    #[test]
    fn test_parse_power() {
        let expr = parse("x ^ 2").unwrap();
        match expr {
            Expr::Binary(BinaryOp::Pow, _, _) => (),
            _ => panic!("Expected Pow binary operation"),
        }
    }

    #[test]
    fn test_parse_precedence() {
        // x + y * z should be x + (y * z)
        let expr = parse("x + y * z").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Add, left, right) => {
                // Left should be x
                if let Expr::Symbol(_) = left.as_ref() {
                } else {
                    panic!("Expected left to be Symbol");
                }
                // Right should be y * z
                if let Expr::Binary(BinaryOp::Mul, _, _) = right.as_ref() {
                } else {
                    panic!("Expected right to be Mul");
                }
            }
            _ => panic!("Expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_power_precedence() {
        // x * y ^ 2 should be x * (y ^ 2)
        let expr = parse("x * y ^ 2").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                // Left should be x
                if let Expr::Symbol(_) = left.as_ref() {
                } else {
                    panic!("Expected left to be Symbol");
                }
                // Right should be y ^ 2
                if let Expr::Binary(BinaryOp::Pow, _, _) = right.as_ref() {
                } else {
                    panic!("Expected right to be Pow");
                }
            }
            _ => panic!("Expected Mul at top level"),
        }
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2 ^ 3 ^ 2 should be 2 ^ (3 ^ 2)
        let expr = parse("2 ^ 3 ^ 2").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Pow, left, right) => {
                // Left should be 2
                if let Expr::Integer(_) = left.as_ref() {
                } else {
                    panic!("Expected left to be Integer");
                }
                // Right should be 3 ^ 2
                if let Expr::Binary(BinaryOp::Pow, _, _) = right.as_ref() {
                } else {
                    panic!("Expected right to be Pow");
                }
            }
            _ => panic!("Expected Pow at top level"),
        }
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse("(x + y) * z").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                // Left should be x + y
                if let Expr::Binary(BinaryOp::Add, _, _) = left.as_ref() {
                } else {
                    panic!("Expected left to be Add");
                }
                // Right should be z
                if let Expr::Symbol(_) = right.as_ref() {
                } else {
                    panic!("Expected right to be Symbol");
                }
            }
            _ => panic!("Expected Mul at top level"),
        }
    }

    #[test]
    fn test_parse_function_sin() {
        let expr = parse("sin(x)").unwrap();
        match expr {
            Expr::Unary(UnaryOp::Sin, _) => (),
            _ => panic!("Expected Sin unary operation"),
        }
    }

    #[test]
    fn test_parse_function_cos() {
        let expr = parse("cos(x)").unwrap();
        match expr {
            Expr::Unary(UnaryOp::Cos, _) => (),
            _ => panic!("Expected Cos unary operation"),
        }
    }

    #[test]
    fn test_parse_function_exp() {
        let expr = parse("exp(x)").unwrap();
        match expr {
            Expr::Unary(UnaryOp::Exp, _) => (),
            _ => panic!("Expected Exp unary operation"),
        }
    }

    #[test]
    fn test_parse_function_log() {
        let expr = parse("log(x)").unwrap();
        match expr {
            Expr::Unary(UnaryOp::Log, _) => (),
            _ => panic!("Expected Log unary operation"),
        }
    }

    #[test]
    fn test_parse_function_sqrt() {
        let expr = parse("sqrt(x)").unwrap();
        match expr {
            Expr::Unary(UnaryOp::Sqrt, _) => (),
            _ => panic!("Expected Sqrt unary operation"),
        }
    }

    #[test]
    fn test_parse_nested_functions() {
        let expr = parse("sin(exp(x))").unwrap();
        match &expr {
            Expr::Unary(UnaryOp::Sin, inner) => {
                if let Expr::Unary(UnaryOp::Exp, _) = inner.as_ref() {
                } else {
                    panic!("Expected inner to be Exp");
                }
            }
            _ => panic!("Expected Sin unary operation"),
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse("x^2 + 3*x + 2").unwrap();
        // Just verify it parses without error
        match expr {
            Expr::Binary(BinaryOp::Add, _, _) => (),
            _ => panic!("Expected Add at top level"),
        }
    }

    #[test]
    fn test_parse_complex_expression_2() {
        let expr = parse("sin(x) * exp(-x)").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                // Left should be sin(x)
                if let Expr::Unary(UnaryOp::Sin, _) = left.as_ref() {
                } else {
                    panic!("Expected left to be Sin");
                }
                // Right should be exp(-x)
                if let Expr::Unary(UnaryOp::Exp, _) = right.as_ref() {
                } else {
                    panic!("Expected right to be Exp");
                }
            }
            _ => panic!("Expected Mul at top level"),
        }
    }

    #[test]
    fn test_parse_whitespace() {
        let expr1 = parse("x+y").unwrap();
        let expr2 = parse("x + y").unwrap();
        let expr3 = parse("  x  +  y  ").unwrap();
        // All should parse to the same structure
        match (&expr1, &expr2, &expr3) {
            (
                Expr::Binary(BinaryOp::Add, _, _),
                Expr::Binary(BinaryOp::Add, _, _),
                Expr::Binary(BinaryOp::Add, _, _),
            ) => (),
            _ => panic!("All should be Add operations"),
        }
    }

    #[test]
    fn test_parse_multi_arg_function() {
        let expr = parse("bessel_j(n, x)").unwrap();
        match &expr {
            Expr::Function(name, args) => {
                assert_eq!(name, "bessel_j");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Function variant"),
        }
    }

    #[test]
    fn test_parse_unary_minus_precedence() {
        // -x^2 should be -(x^2), not (-x)^2
        let expr = parse("-x^2").unwrap();
        match &expr {
            Expr::Unary(UnaryOp::Neg, inner) => {
                if let Expr::Binary(BinaryOp::Pow, _, _) = inner.as_ref() {
                } else {
                    panic!("Expected inner to be Pow");
                }
            }
            _ => panic!("Expected Neg unary operation"),
        }
    }

    #[test]
    fn test_parse_alternative_function_names() {
        // Test alternative function names
        let expr1 = parse("ln(x)").unwrap();
        let expr2 = parse("log(x)").unwrap();
        match (&expr1, &expr2) {
            (Expr::Unary(UnaryOp::Log, _), Expr::Unary(UnaryOp::Log, _)) => (),
            _ => panic!("Both ln and log should parse to Log"),
        }

        let expr3 = parse("asin(x)").unwrap();
        let expr4 = parse("arcsin(x)").unwrap();
        match (&expr3, &expr4) {
            (Expr::Unary(UnaryOp::Arcsin, _), Expr::Unary(UnaryOp::Arcsin, _)) => (),
            _ => panic!("Both asin and arcsin should parse to Arcsin"),
        }
    }

    #[test]
    fn test_parse_all_trig_functions() {
        let funcs = vec![
            ("sin", UnaryOp::Sin),
            ("cos", UnaryOp::Cos),
            ("tan", UnaryOp::Tan),
            ("sinh", UnaryOp::Sinh),
            ("cosh", UnaryOp::Cosh),
            ("tanh", UnaryOp::Tanh),
            ("arcsin", UnaryOp::Arcsin),
            ("arccos", UnaryOp::Arccos),
            ("arctan", UnaryOp::Arctan),
        ];

        for (func_name, expected_op) in funcs {
            let expr = parse(&format!("{}(x)", func_name)).unwrap();
            match expr {
                Expr::Unary(op, _) => assert_eq!(op, expected_op),
                _ => panic!("Expected Unary operation for {}", func_name),
            }
        }
    }

    #[test]
    fn test_parse_all_special_functions() {
        let funcs = vec![
            ("abs", UnaryOp::Abs),
            ("sign", UnaryOp::Sign),
            ("gamma", UnaryOp::Gamma),
            ("factorial", UnaryOp::Factorial),
            ("erf", UnaryOp::Erf),
            ("zeta", UnaryOp::Zeta),
        ];

        for (func_name, expected_op) in funcs {
            let expr = parse(&format!("{}(x)", func_name)).unwrap();
            match expr {
                Expr::Unary(op, _) => assert_eq!(op, expected_op),
                _ => panic!("Expected Unary operation for {}", func_name),
            }
        }
    }

    #[test]
    fn test_parse_error_handling() {
        // Test various invalid inputs
        assert!(parse("x +").is_err());
        assert!(parse("* x").is_err());
        assert!(parse("(x + y").is_err());
        assert!(parse("x + y)").is_err());
        assert!(parse("sin(").is_err());
        assert!(parse("sin)").is_err());
    }

    #[test]
    fn test_parse_constant_pi() {
        let expr = parse("pi").unwrap();
        if let Expr::Symbol(sym) = &expr {
            assert_eq!(sym.name(), "pi");
        } else {
            panic!("Expected Symbol for pi");
        }
    }

    #[test]
    fn test_parse_constant_e() {
        let expr = parse("e").unwrap();
        if let Expr::Symbol(sym) = &expr {
            assert_eq!(sym.name(), "e");
        } else {
            panic!("Expected Symbol for e");
        }
    }

    #[test]
    fn test_parse_constant_I() {
        let expr = parse("I").unwrap();
        if let Expr::Symbol(sym) = &expr {
            assert_eq!(sym.name(), "I");
        } else {
            panic!("Expected Symbol for I");
        }
    }

    #[test]
    fn test_parse_constant_infinity() {
        let expr = parse("oo").unwrap();
        if let Expr::Symbol(sym) = &expr {
            assert_eq!(sym.name(), "oo");
        } else {
            panic!("Expected Symbol for oo");
        }

        // Test alternative names
        let expr2 = parse("infinity").unwrap();
        if let Expr::Symbol(sym) = &expr2 {
            assert_eq!(sym.name(), "oo");
        } else {
            panic!("Expected Symbol for infinity");
        }
    }

    #[test]
    fn test_parse_expression_with_pi() {
        // 2*pi
        let expr = parse("2 * pi").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Mul, left, right) => {
                assert!(matches!(left.as_ref(), Expr::Integer(_)));
                if let Expr::Symbol(sym) = right.as_ref() {
                    assert_eq!(sym.name(), "pi");
                } else {
                    panic!("Expected pi symbol");
                }
            }
            _ => panic!("Expected Mul operation"),
        }
    }

    #[test]
    fn test_parse_expression_with_e() {
        // e^x
        let expr = parse("e ^ x").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Pow, left, right) => {
                if let Expr::Symbol(sym) = left.as_ref() {
                    assert_eq!(sym.name(), "e");
                } else {
                    panic!("Expected e symbol");
                }
                assert!(matches!(right.as_ref(), Expr::Symbol(_)));
            }
            _ => panic!("Expected Pow operation"),
        }
    }

    #[test]
    fn test_parse_complex_expression_with_I() {
        // 3 + 2*I
        let expr = parse("3 + 2 * I").unwrap();
        match &expr {
            Expr::Binary(BinaryOp::Add, _, right) => {
                if let Expr::Binary(BinaryOp::Mul, _, i_part) = right.as_ref() {
                    if let Expr::Symbol(sym) = i_part.as_ref() {
                        assert_eq!(sym.name(), "I");
                    } else {
                        panic!("Expected I symbol");
                    }
                } else {
                    panic!("Expected multiplication with I");
                }
            }
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_parse_trig_with_pi() {
        // sin(pi/2)
        let expr = parse("sin(pi / 2)").unwrap();
        match &expr {
            Expr::Unary(UnaryOp::Sin, arg) => {
                if let Expr::Binary(BinaryOp::Div, _, _) = arg.as_ref() {
                } else {
                    panic!("Expected division inside sin");
                }
            }
            _ => panic!("Expected Sin operation"),
        }
    }

    // ---------------------------------------------------------------------------------------
    // S1 design §2.12 and §4.3: bignum literals, `a/b` as an operator (with the value-preserving
    // fold of two signed integer literals), signed exponents, sign runs, inverse hyperbolic names
    // ---------------------------------------------------------------------------------------

    fn int(s: &str) -> Expr {
        Expr::Integer(Integer::from_decimal_str(s).unwrap())
    }

    fn rat(n: i64, d: i64) -> Expr {
        Expr::Rational(Rational::new(Integer::from(n), Integer::from(d)).unwrap())
    }

    fn on_2mb<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("the thread panicked or overflowed its stack")
    }

    #[test]
    fn test_parse_big_integer() {
        match &parse("x + 18446744073709551616").unwrap() {
            Expr::Binary(BinaryOp::Add, _, r) => assert_eq!(**r, int("18446744073709551616")),
            other => panic!("expected a sum, got {other}"),
        }
        assert_eq!(
            parse("-9223372036854775808").unwrap(),
            Expr::Unary(UnaryOp::Neg, Arc::new(int("9223372036854775808")))
        );
        assert_eq!(
            parse("123456789012345678901234567890").unwrap(),
            int("123456789012345678901234567890")
        );
    }

    #[test]
    fn test_parse_division_by_non_literal() {
        for s in ["1/x", "1/sin(x)", "exp(1/x)", "1/(y+x)"] {
            let e = parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
            let div = match &e {
                Expr::Unary(UnaryOp::Exp, a) => (**a).clone(),
                other => other.clone(),
            };
            match &div {
                Expr::Binary(BinaryOp::Div, l, _) => assert_eq!(**l, Expr::from(1), "{s}"),
                other => panic!("{s}: expected a quotient, got {other}"),
            }
        }
    }

    #[test]
    fn test_parse_rational_precedence() {
        // gp: 2^3/4 = 2, 2/3^2 = 2/9, 3/4^2 = 3/16, 2^-1/3 = 1/6
        let x = Expr::symbol("x");
        assert_eq!(
            parse("x^2/4").unwrap(),
            Expr::Binary(
                BinaryOp::Div,
                Arc::new(Expr::Binary(BinaryOp::Pow, Arc::new(x.clone()), Arc::new(Expr::from(2)))),
                Arc::new(Expr::from(4))
            )
        );
        assert_eq!(parse("2^3/4").unwrap().simplify(), Expr::from(2));
        assert_eq!(parse("2/3^2").unwrap().simplify(), rat(2, 9));
        assert_eq!(parse("3/4^2").unwrap().simplify(), rat(3, 16));
        assert_eq!(parse("2^-1/3").unwrap().simplify(), rat(1, 6));
        let at = Expr::Rational(Rational::new(Integer::from(5), Integer::from(7)).unwrap());
        let xs = crate::symbol::Symbol::new("x");
        let val = |s: &str| parse(s).unwrap().substitute(&xs, &at).eval_rational().unwrap();
        // x*2/3^2 = 2x/9 and x/2/3 = x/6 at x = 5/7
        assert_eq!(val("x*2/3^2"), Rational::new(Integer::from(10), Integer::from(63)).unwrap());
        assert_eq!(val("x/2/3"), Rational::new(Integer::from(5), Integer::from(42)).unwrap());
    }

    #[test]
    fn test_parse_signed_exponent() {
        let pow = |a: Expr, b: Expr| Expr::Binary(BinaryOp::Pow, Arc::new(a), Arc::new(b));
        let x = Expr::symbol("x");
        assert_eq!(parse("2^-1").unwrap(), pow(Expr::from(2), Expr::from(-1)));
        assert_eq!(parse("x^-2").unwrap(), pow(x.clone(), Expr::from(-2)));
        assert_eq!(
            parse("2^-x").unwrap(),
            pow(Expr::from(2), Expr::Unary(UnaryOp::Neg, Arc::new(x)))
        );
    }

    #[test]
    fn test_parse_literal_fold() {
        assert_eq!(parse("1/3").unwrap(), rat(1, 3));
        assert_eq!(parse("6/4").unwrap(), rat(3, 2));
        assert_eq!(parse("1/-3").unwrap(), rat(-1, 3));
        assert_eq!(parse("-2/3").unwrap(), rat(-2, 3));
        assert_eq!(parse("4/2").unwrap(), Expr::from(2));
        assert!(parse("1/0").is_err());
    }

    #[test]
    fn test_parse_inverse_hyperbolic() {
        let x = || Arc::new(Expr::symbol("x"));
        assert_eq!(parse("asinh(x)").unwrap(), Expr::Unary(UnaryOp::Arcsinh, x()));
        assert_eq!(parse("arcsinh(x)").unwrap(), Expr::Unary(UnaryOp::Arcsinh, x()));
        assert_eq!(parse("arccosh(x)").unwrap(), Expr::Unary(UnaryOp::Arccosh, x()));
        assert_eq!(parse("acosh(x)").unwrap(), Expr::Unary(UnaryOp::Arccosh, x()));
        assert_eq!(parse("atanh(x)").unwrap(), Expr::Unary(UnaryOp::Arctanh, x()));
        assert_eq!(parse("arctanh(x)").unwrap(), Expr::Unary(UnaryOp::Arctanh, x()));
    }

    #[test]
    fn test_parse_sign_runs() {
        let x = Expr::symbol("x");
        assert_eq!(parse("--5").unwrap(), Expr::from(5));
        assert_eq!(parse("---x").unwrap(), Expr::Unary(UnaryOp::Neg, Arc::new(x.clone())));
        assert_eq!(
            parse("x^--2").unwrap(),
            Expr::Binary(BinaryOp::Pow, Arc::new(x.clone()), Arc::new(Expr::from(2)))
        );
        assert_eq!(parse("+-+x").unwrap(), Expr::Unary(UnaryOp::Neg, Arc::new(x)));
        // 100 000 signs on a 2 MB thread: an even run, so x
        on_2mb(|| {
            let s = format!("{}x", "-".repeat(100_000));
            assert_eq!(parse(&s).unwrap(), Expr::symbol("x"));
        });
    }

    #[test]
    fn test_parse_depth_not_below_head() {
        on_2mb(|| {
            let s = format!("{}x{}", "sin(".repeat(160), ")".repeat(160));
            assert!(parse(&s).is_ok());
            let s = format!("{}x{}", "(".repeat(200), ")".repeat(200));
            assert!(parse(&s).is_ok());
        });
    }

    #[test]
    fn test_display_parse_round_trip() {
        // S1 design §4.3: the Display of §2.11 reads back as the same tree, up to the parser's
        // own folds (simplify folds Neg(Integer(2)) and Real(-1.5), which the parser rebuilds as
        // negations)
        let x = || Arc::new(Expr::symbol("x"));
        let pow = |a: Expr, b: Expr| Expr::Binary(BinaryOp::Pow, Arc::new(a), Arc::new(b));
        let cases = vec![
            pow(Expr::symbol("x"), rat(1, 2)),
            pow(rat(3, 4), rat(-1, 2)),
            Expr::Binary(BinaryOp::Div, x(), Arc::new(rat(1, 2))),
            pow(Expr::from(-2), Expr::from(2)),
            pow(Expr::Unary(UnaryOp::Neg, Arc::new(Expr::symbol("y"))), rat(1, 2)),
            pow(Expr::Real(-1.5), Expr::from(2)),
            Expr::Unary(UnaryOp::Neg, Arc::new(Expr::Unary(UnaryOp::Neg, x()))),
            pow(Expr::symbol("pi"), Expr::Unary(UnaryOp::Neg, Arc::new(rat(2, 3)))),
            Expr::Binary(BinaryOp::Div, Arc::new(Expr::from(3)), Arc::new(Expr::Unary(UnaryOp::Neg, Arc::new(rat(1, 2))))),
            Expr::Real(1e23),
            Expr::Unary(UnaryOp::Arctanh, x()),
        ];
        for e in cases {
            let text = format!("{e}");
            let back = parse(&text).unwrap_or_else(|err| panic!("{text}: {err}"));
            if matches!(text.as_str(), "((-1.5) ^ 2)") {
                // simplify keeps Neg(Real(1.5)) and Real(-1.5) apart: compare the values
                assert_eq!(back.eval_float(), e.eval_float(), "{text}");
            } else {
                assert_eq!(back.simplify(), e.simplify(), "{text}");
            }
        }
    }
}
