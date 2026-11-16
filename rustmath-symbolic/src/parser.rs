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
    multi::{many0, separated_list0},
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

    match expression(input) {
        Ok(("", expr)) => Ok(expr),
        Ok((remaining, _)) => Err(ParseError {
            message: format!("Unexpected input remaining: '{}'", remaining),
            // TODO: Add position information: message: format!("at position {}", input.len() - remaining.len())
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

// Whitespace handling
fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O>
where
    F: FnMut(&'a str) -> IResult<&'a str, O>,
{
    delimited(multispace0, inner, multispace0)
}

// Unsigned integer literal of any size (S1 design §2.12; the sign is the unary rule's). The
// literal `a/b` is not a token: `multiplicative` folds a quotient of two signed integer literals.
fn integer_literal(input: &str) -> IResult<&str, Integer> {
    let (rest, digits) = digit1(input)?;
    match Integer::from_decimal_str(digits) {
        Some(n) => Ok((rest, n)),
        None => Err(nom::Err::Error(Error::new(input, ErrorKind::Digit))),
    }
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

// Parse a function name and determine the appropriate UnaryOp or Function variant
fn function_call(input: &str) -> IResult<&str, Expr> {
    let (input, func_name) = identifier(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, args) = separated_list0(ws(char(',')), expression)(input)?;
    let (input, _) = ws(char(')'))(input)?;

    // Convert args to Vec<Arc<Expr>>
    let arc_args: Vec<Arc<Expr>> = args.into_iter().map(Arc::new).collect();

    // Map known function names to UnaryOp (one construction, out of the recursive frame)
    let expr = make_call(func_name, arc_args);

    Ok((input, expr))
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

// Parse a number: a float, or an unsigned integer literal of any size
fn number(input: &str) -> IResult<&str, Expr> {
    // Try to parse as float first (with decimal point); a literal that underflows is a Failure
    // (S1 fix-up round 3), so its digits are not read again as an integer
    match float(input) {
        Ok((remaining, f)) => return Ok((remaining, Expr::Real(f))),
        Err(e @ nom::Err::Failure(_)) => return Err(e),
        Err(_) => {}
    }
    map(integer_literal, Expr::Integer)(input)
}

// A run of signs ('-' | '+')*, read by a loop (no recursion per sign, so 100 KB of signs costs
// no stack): whether the run is negative (an odd number of '-').
#[inline(never)]
fn sign_run(input: &str) -> (&str, bool) {
    let mut neg = false;
    let mut rest = input;
    loop {
        let t = rest.trim_start();
        if let Some(r) = t.strip_prefix('-') {
            neg = !neg;
            rest = r;
        } else if let Some(r) = t.strip_prefix('+') {
            rest = r;
        } else {
            return (rest, neg);
        }
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

// Parse a parenthesized expression
fn parens(input: &str) -> IResult<&str, Expr> {
    delimited(
        ws(char('(')),
        expression,
        ws(char(')')),
    )(input)
}

// Parse an atom: number, variable, function call, or parenthesized expression.
// Never inlined: in release `atom` was inlined into the recursive `power`, whose frame then held
// its locals across the recursion (552 bytes per `^` level against HEAD's 312), so a caret tower
// that HEAD parses overflowed (S1 review 1, B1; `tests/s1_depth.rs`).
#[inline(never)]
fn atom(input: &str) -> IResult<&str, Expr> {
    alt((
        number,
        function_call,
        parens,
        variable,
    ))(input)
}

// Parse power expression (right-associative)
// Example: 2^3^2 = 2^(3^2) = 2^9 = 512
// Note: Power has HIGHER precedence than unary minus
// So -x^2 = -(x^2), not (-x)^2
// The exponent is signed: exponent := ('-' | '+')* power, and a negated integer literal folds to
// a negative literal (`x^-2` is Pow(x, Integer(-2)); `2^-x` is Pow(2, Neg(x))).
fn power(input: &str) -> IResult<&str, Expr> {
    let (input, base) = ws(atom)(input)?;

    if let Ok((input, _)) = ws(char::<&str, Error<&str>>('^'))(input) {
        let (input, neg) = sign_run(input);
        let (input, exp) = power(input)?; // Right-associative recursion
        Ok((input, make_power(base, neg, exp)))
    } else {
        Ok((input, base))
    }
}

// The node of `base ^ (sign) exp`. Out of line (and in `unary` and `multiplicative` likewise),
// so that the frames of the recursive rules stay as small as HEAD's: the parser's recursion over
// parentheses and calls is the only one over a text (S1 design §2.12, §2.16).
#[inline(never)]
fn make_power(base: Expr, neg: bool, exp: Expr) -> Expr {
    let exp = match (neg, &exp) {
        (false, _) => exp,
        (true, Expr::Integer(n)) => Expr::Integer(-n.clone()),
        (true, _) => Expr::Unary(UnaryOp::Neg, Arc::new(exp)),
    };
    Expr::Binary(BinaryOp::Pow, Arc::new(base), Arc::new(exp))
}

#[inline(never)]
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
#[inline(never)]
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

// Parse a unary expression: a run of signs (an odd number of '-' gives one Neg, an even number
// none: "--5" is 5, "---x" is Neg(x)), then a power.
// This comes AFTER power to ensure -x^2 = -(x^2)
fn unary(input: &str) -> IResult<&str, Expr> {
    let (input, neg) = sign_run(input);
    let (input, expr) = power(input)?;
    Ok((input, negate_if(neg, expr)))
}

// Parse multiplicative expression (*, /)
fn multiplicative(input: &str) -> IResult<&str, Expr> {
    // TODO: Implicit multiplication support
    // Detect and insert implicit * in cases like:
    // - "2x" → "2*x" (number followed by identifier)
    // - "2(x+1)" → "2*(x+1)" (number followed by parenthesis)
    // - "(x+1)(x-1)" → "(x+1)*(x-1)" (adjacent parentheses)
    // - "x y" → "x*y" (adjacent identifiers, only if no whitespace in strict mode)
    //
    // Challenges:
    // - Must not break multi-character symbols like "sin", "alpha"
    // - Need lookahead to distinguish "2e5" (scientific) from "2e" (2*e)
    // - Unicode operators (×, ⋅) should also be recognized

    let (input, mut expr) = unary(input)?;

    let mut input = input;
    loop {
        if let Ok((new_input, op)) = ws(alt((char::<&str, Error<&str>>('*'), char('/'))))(input) {
            let (new_input, rhs) = unary(new_input)?;
            expr = match product_step(op, expr, rhs) {
                Some(e) => e,
                // "division by zero in a literal"
                None => return Err(nom::Err::Failure(Error::new(new_input, ErrorKind::Verify))),
            };
            input = new_input;
        } else {
            break;
        }
    }

    Ok((input, expr))
}

// Parse additive expression (+, -)
fn additive(input: &str) -> IResult<&str, Expr> {
    let (input, mut expr) = multiplicative(input)?;

    let mut input = input;
    loop {
        if let Ok((new_input, op)) = ws(alt((char::<&str, Error<&str>>('+'), char('-'))))(input) {
            let (new_input, rhs) = multiplicative(new_input)?;
            let bin_op = if op == '+' { BinaryOp::Add } else { BinaryOp::Sub };
            expr = Expr::Binary(bin_op, Arc::new(expr), Arc::new(rhs));
            input = new_input;
        } else {
            break;
        }
    }

    Ok((input, expr))
}

// Parse a full expression
fn expression(input: &str) -> IResult<&str, Expr> {
    ws(additive)(input)
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
