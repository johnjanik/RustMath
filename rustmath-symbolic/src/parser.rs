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

use crate::expression::{BinaryOp, Expr, UnaryOp};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, digit1, multispace0},
    combinator::{map, opt, recognize},
    error::{Error, ErrorKind},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, tuple},
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
    match expression(input) {
        Ok(("", expr)) => Ok(expr),
        Ok((remaining, _)) => Err(ParseError {
            message: format!("Unexpected input remaining: '{}'", remaining),
        }),
        Err(e) => Err(ParseError {
            message: format!("Failed to parse expression: {}", e),
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

// Parse an integer
fn integer(input: &str) -> IResult<&str, Integer> {
    let (input, num_str) = recognize(pair(opt(char('-')), digit1))(input)?;
    let num = num_str.parse::<i64>().map_err(|_| {
        nom::Err::Error(Error::new(input, ErrorKind::Digit))
    })?;
    Ok((input, Integer::from(num)))
}

// Parse a rational number (e.g., "1/3" or just "5")
fn rational(input: &str) -> IResult<&str, Rational> {
    let (input, numer) = integer(input)?;

    // Try to parse "/denominator"
    if let Ok((input, _)) = char::<&str, Error<&str>>('/')(input) {
        let (input, denom) = integer(input)?;
        // Rational::new returns Result, but division by zero would be caught
        // We'll handle the error by converting to nom error
        match Rational::new(numer, denom) {
            Ok(r) => Ok((input, r)),
            Err(_) => Err(nom::Err::Error(Error::new(input, ErrorKind::Verify))),
        }
    } else {
        Ok((input, Rational::from_integer(numer)))
    }
}

// Parse a variable/symbol name
fn identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alpha1,
        many0(alt((alphanumeric1, tag("_"))))
    ))(input)
}

// Parse a variable
fn variable(input: &str) -> IResult<&str, Expr> {
    map(identifier, |name: &str| Expr::symbol(name))(input)
}

// Parse a function name and determine the appropriate UnaryOp or Function variant
fn function_call(input: &str) -> IResult<&str, Expr> {
    let (input, func_name) = identifier(input)?;
    let (input, _) = ws(char('('))(input)?;
    let (input, args) = separated_list0(ws(char(',')), expression)(input)?;
    let (input, _) = ws(char(')'))(input)?;

    // Convert args to Vec<Arc<Expr>>
    let arc_args: Vec<Arc<Expr>> = args.into_iter().map(Arc::new).collect();

    // Map known function names to UnaryOp
    let expr = if arc_args.len() == 1 {
        let arg = arc_args[0].as_ref().clone();
        match func_name {
            "sin" => Expr::Unary(UnaryOp::Sin, Arc::new(arg)),
            "cos" => Expr::Unary(UnaryOp::Cos, Arc::new(arg)),
            "tan" => Expr::Unary(UnaryOp::Tan, Arc::new(arg)),
            "sinh" => Expr::Unary(UnaryOp::Sinh, Arc::new(arg)),
            "cosh" => Expr::Unary(UnaryOp::Cosh, Arc::new(arg)),
            "tanh" => Expr::Unary(UnaryOp::Tanh, Arc::new(arg)),
            "arcsin" | "asin" => Expr::Unary(UnaryOp::Arcsin, Arc::new(arg)),
            "arccos" | "acos" => Expr::Unary(UnaryOp::Arccos, Arc::new(arg)),
            "arctan" | "atan" => Expr::Unary(UnaryOp::Arctan, Arc::new(arg)),
            "exp" => Expr::Unary(UnaryOp::Exp, Arc::new(arg)),
            "log" | "ln" => Expr::Unary(UnaryOp::Log, Arc::new(arg)),
            "sqrt" => Expr::Unary(UnaryOp::Sqrt, Arc::new(arg)),
            "abs" => Expr::Unary(UnaryOp::Abs, Arc::new(arg)),
            "sign" => Expr::Unary(UnaryOp::Sign, Arc::new(arg)),
            "gamma" => Expr::Unary(UnaryOp::Gamma, Arc::new(arg)),
            "factorial" => Expr::Unary(UnaryOp::Factorial, Arc::new(arg)),
            "erf" => Expr::Unary(UnaryOp::Erf, Arc::new(arg)),
            "zeta" => Expr::Unary(UnaryOp::Zeta, Arc::new(arg)),
            _ => Expr::Function(func_name.to_string(), arc_args),
        }
    } else {
        // Multi-argument function or zero-argument (use generic Function variant)
        Expr::Function(func_name.to_string(), arc_args)
    };

    Ok((input, expr))
}

// Parse a number (integer or rational)
fn number(input: &str) -> IResult<&str, Expr> {
    map(rational, |r| {
        if r.denominator() == &Integer::from(1) {
            Expr::Integer(r.numerator().clone())
        } else {
            Expr::Rational(r)
        }
    })(input)
}

// Parse a parenthesized expression
fn parens(input: &str) -> IResult<&str, Expr> {
    delimited(
        ws(char('(')),
        expression,
        ws(char(')')),
    )(input)
}

// Parse an atom: number, variable, function call, or parenthesized expression
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
fn power(input: &str) -> IResult<&str, Expr> {
    let (input, base) = ws(atom)(input)?;

    if let Ok((input, _)) = ws(char::<&str, Error<&str>>('^'))(input) {
        let (input, exp) = power(input)?; // Right-associative recursion
        Ok((input, Expr::Binary(BinaryOp::Pow, Arc::new(base), Arc::new(exp))))
    } else {
        Ok((input, base))
    }
}

// Parse a unary expression (handles unary minus and plus)
// This comes AFTER power to ensure -x^2 = -(x^2)
fn unary(input: &str) -> IResult<&str, Expr> {
    let (input, op) = opt(ws(alt((char('-'), char('+')))))(input)?;
    let (input, expr) = power(input)?;

    match op {
        Some('-') => Ok((input, Expr::Unary(UnaryOp::Neg, Arc::new(expr)))),
        _ => Ok((input, expr)),
    }
}

// Parse multiplicative expression (*, /)
fn multiplicative(input: &str) -> IResult<&str, Expr> {
    let (input, mut expr) = unary(input)?;

    let mut input = input;
    loop {
        if let Ok((new_input, op)) = ws(alt((char::<&str, Error<&str>>('*'), char('/'))))(input) {
            let (new_input, rhs) = unary(new_input)?;
            let bin_op = if op == '*' { BinaryOp::Mul } else { BinaryOp::Div };
            expr = Expr::Binary(bin_op, Arc::new(expr), Arc::new(rhs));
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
    fn test_parse_variable() {
        let expr = parse("x").unwrap();
        if let Expr::Symbol(sym) = expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
        match expr {
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
}
