//! RustMath REPL evaluation engine
//!
//! Parses and evaluates RustMath expressions in an interactive context.

use std::collections::HashMap;
use std::str::FromStr;
use num_bigint::BigInt;
use rustmath_integers::Integer;
use rustmath_integers::SageInteger; // For factorial, factor
use rustmath_rationals::Rational;
use rustmath_complex::Complex;
use rustmath_symbolic::Symbol;

/// Result of evaluating an expression
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Plain text representation
    pub text: String,
    /// LaTeX representation (if available)
    pub latex: Option<String>,
    /// HTML representation (if available)
    pub html: Option<String>,
    /// SVG representation (for plots)
    pub svg: Option<String>,
    /// Whether this produced output
    pub has_output: bool,
}

impl EvalResult {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            latex: None,
            html: None,
            svg: None,
            has_output: false,
        }
    }

    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            latex: None,
            html: None,
            svg: None,
            has_output: true,
        }
    }

    pub fn with_latex(mut self, latex: impl Into<String>) -> Self {
        self.latex = Some(latex.into());
        self
    }

    pub fn with_html(mut self, html: impl Into<String>) -> Self {
        self.html = Some(html.into());
        self
    }

    /// Convert to mime-type data map for Jupyter
    pub fn to_data(&self) -> HashMap<String, String> {
        let mut data = HashMap::new();

        if !self.text.is_empty() {
            data.insert("text/plain".to_string(), self.text.clone());
        }

        if let Some(latex) = &self.latex {
            data.insert("text/latex".to_string(), latex.clone());
        }

        if let Some(html) = &self.html {
            data.insert("text/html".to_string(), html.clone());
        }

        if let Some(svg) = &self.svg {
            data.insert("image/svg+xml".to_string(), svg.clone());
        }

        data
    }
}

/// Error from evaluation
#[derive(Debug, Clone)]
pub struct EvalError {
    pub name: String,
    pub message: String,
    pub traceback: Vec<String>,
}

impl EvalError {
    pub fn new(name: impl Into<String>, message: impl Into<String>) -> Self {
        let msg = message.into();
        Self {
            name: name.into(),
            traceback: vec![msg.clone()],
            message: msg,
        }
    }
}

/// Variable stored in the REPL context
#[derive(Debug, Clone)]
pub enum RustMathValue {
    Integer(Integer),
    Rational(Rational),
    Complex(Complex),
    Symbol(Symbol),
    String(String),
    Bool(bool),
    None,
}

impl RustMathValue {
    pub fn to_display(&self) -> EvalResult {
        match self {
            RustMathValue::Integer(n) => {
                let text = n.to_string();
                EvalResult::text(&text)
            }
            RustMathValue::Rational(r) => {
                let text = r.to_string();
                let latex = format!("$\\frac{{{}}}{{{}}}$", r.numerator(), r.denominator());
                EvalResult::text(&text).with_latex(latex)
            }
            RustMathValue::Complex(c) => {
                let text = format!("{} + {}i", c.real(), c.imag());
                EvalResult::text(&text)
            }
            RustMathValue::Symbol(s) => {
                let text = s.name().to_string();
                let latex = format!("${}$", s.name());
                EvalResult::text(&text).with_latex(latex)
            }
            RustMathValue::String(s) => EvalResult::text(format!("\"{}\"", s)),
            RustMathValue::Bool(b) => EvalResult::text(if *b { "True" } else { "False" }),
            RustMathValue::None => EvalResult::empty(),
        }
    }
}

/// The RustMath REPL context
pub struct ReplContext {
    /// Named variables
    variables: HashMap<String, RustMathValue>,
    /// Execution counter
    execution_count: u64,
    /// Last result (stored as _)
    last_result: Option<RustMathValue>,
    /// Captured stdout
    stdout: String,
    /// Captured stderr
    stderr: String,
}

impl ReplContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            execution_count: 0,
            last_result: None,
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    pub fn increment_count(&mut self) {
        self.execution_count += 1;
    }

    pub fn stdout(&self) -> &str {
        &self.stdout
    }

    pub fn stderr(&self) -> &str {
        &self.stderr
    }

    pub fn clear_output(&mut self) {
        self.stdout.clear();
        self.stderr.clear();
    }

    fn print(&mut self, text: &str) {
        self.stdout.push_str(text);
        self.stdout.push('\n');
    }

    /// Evaluate code and return result
    pub fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError> {
        self.clear_output();

        let code = code.trim();
        if code.is_empty() {
            return Ok(EvalResult::empty());
        }

        // Handle multiple lines
        let lines: Vec<&str> = code.lines().collect();
        let mut last_result = EvalResult::empty();

        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            last_result = self.eval_line(line)?;
        }

        Ok(last_result)
    }

    fn eval_line(&mut self, line: &str) -> Result<EvalResult, EvalError> {
        // Check for assignment: var = expr
        if let Some(eq_pos) = line.find('=') {
            let before_eq = line[..eq_pos].trim();
            // Make sure it's not ==, !=, <=, >=
            if !line[eq_pos..].starts_with("==")
                && !before_eq.ends_with('!')
                && !before_eq.ends_with('<')
                && !before_eq.ends_with('>')
                && is_valid_identifier(before_eq)
            {
                let var_name = before_eq.to_string();
                let expr = line[eq_pos + 1..].trim();
                let value = self.eval_expr(expr)?;
                self.variables.insert(var_name, value.clone());
                self.last_result = Some(value.clone());
                return Ok(value.to_display());
            }
        }

        // Check for built-in commands
        if line.starts_with("print(") && line.ends_with(')') {
            let inner = &line[6..line.len() - 1];
            let value = self.eval_expr(inner)?;
            let display = value.to_display();
            self.print(&display.text);
            return Ok(EvalResult::empty());
        }

        if line == "help" || line == "help()" {
            return Ok(self.show_help());
        }

        if line == "vars" || line == "vars()" {
            return Ok(self.show_vars());
        }

        // Evaluate as expression
        let value = self.eval_expr(line)?;
        self.last_result = Some(value.clone());
        Ok(value.to_display())
    }

    fn eval_expr(&mut self, expr: &str) -> Result<RustMathValue, EvalError> {
        let expr = expr.trim();

        // Check for variable reference
        if is_valid_identifier(expr) {
            if let Some(value) = self.variables.get(expr) {
                return Ok(value.clone());
            }
            if expr == "_" {
                if let Some(value) = &self.last_result {
                    return Ok(value.clone());
                }
            }
            // Special constants
            if expr == "True" || expr == "true" {
                return Ok(RustMathValue::Bool(true));
            }
            if expr == "False" || expr == "false" {
                return Ok(RustMathValue::Bool(false));
            }
        }

        // Try to parse as integer (supports arbitrary precision)
        if expr.chars().all(|c| c.is_ascii_digit() || c == '-') && !expr.is_empty() {
            if let Ok(n) = BigInt::from_str(expr) {
                return Ok(RustMathValue::Integer(Integer::new(n)));
            }
        }

        // Try to parse as rational (a/b) - simple form only
        if let Some(slash_pos) = expr.find('/') {
            let num_str = expr[..slash_pos].trim();
            let den_str = expr[slash_pos + 1..].trim();
            if let (Ok(num), Ok(den)) = (num_str.parse::<i64>(), den_str.parse::<i64>()) {
                if den != 0 {
                    match Rational::new(Integer::from(num), Integer::from(den)) {
                        Ok(r) => return Ok(RustMathValue::Rational(r)),
                        Err(_) => {}
                    }
                }
            }
        }

        // Function calls
        if expr.contains('(') && expr.ends_with(')') {
            return self.eval_function_call(expr);
        }

        // Binary operations
        if let Some(result) = self.try_eval_binary_op(expr)? {
            return Ok(result);
        }

        // String literal
        if (expr.starts_with('"') && expr.ends_with('"'))
            || (expr.starts_with('\'') && expr.ends_with('\''))
        {
            return Ok(RustMathValue::String(expr[1..expr.len() - 1].to_string()));
        }

        // Unknown expression - treat as symbol
        if is_valid_identifier(expr) {
            return Ok(RustMathValue::Symbol(Symbol::new(expr)));
        }

        Err(EvalError::new("SyntaxError", format!("Cannot parse: {}", expr)))
    }

    fn eval_function_call(&mut self, expr: &str) -> Result<RustMathValue, EvalError> {
        let paren_pos = expr.find('(').unwrap();
        let func_name = expr[..paren_pos].trim();
        let args_str = &expr[paren_pos + 1..expr.len() - 1];

        match func_name {
            // Integer constructor (supports arbitrary precision)
            "Integer" => {
                let n = BigInt::from_str(args_str.trim())
                    .map_err(|_| EvalError::new("ValueError", "Invalid integer"))?;
                Ok(RustMathValue::Integer(Integer::new(n)))
            }

            // Factorial
            "factorial" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let n_val = n.to_i64();
                        if n_val < 0 {
                            return Err(EvalError::new("ValueError", "Factorial requires non-negative integer"));
                        }
                        if n_val > 10000 {
                            return Err(EvalError::new("ValueError", "Factorial argument too large"));
                        }
                        let result = Integer::factorial(n_val as u32);
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "factorial requires an integer")),
                }
            }

            // GCD
            "gcd" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "gcd requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(b)) => {
                        Ok(RustMathValue::Integer(a.gcd(b)))
                    }
                    _ => Err(EvalError::new("TypeError", "gcd requires integers")),
                }
            }

            // LCM
            "lcm" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "lcm requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(b)) => {
                        Ok(RustMathValue::Integer(a.lcm(b)))
                    }
                    _ => Err(EvalError::new("TypeError", "lcm requires integers")),
                }
            }

            // Primality test
            "is_prime" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let result = n.is_prime();
                        Ok(RustMathValue::Bool(result))
                    }
                    _ => Err(EvalError::new("TypeError", "is_prime requires an integer")),
                }
            }

            // Factorization
            "factor" | "factorint" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let factors = n.factor();
                        let result: Vec<String> = factors.iter()
                            .map(|(p, e)| if *e == 1 { p.to_string() } else { format!("{}^{}", p, e) })
                            .collect();
                        Ok(RustMathValue::String(result.join(" * ")))
                    }
                    _ => Err(EvalError::new("TypeError", "factor requires an integer")),
                }
            }

            // Rational constructor
            "Rational" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "Rational requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(d)) => {
                        Rational::new(n.clone(), d.clone())
                            .map(RustMathValue::Rational)
                            .map_err(|e| EvalError::new("MathError", format!("{:?}", e)))
                    }
                    _ => Err(EvalError::new("TypeError", "Rational requires integers")),
                }
            }

            // Symbol constructor
            "Symbol" | "var" => {
                let name = args_str.trim().trim_matches(|c| c == '"' || c == '\'');
                Ok(RustMathValue::Symbol(Symbol::new(name)))
            }

            // Complex constructor
            "Complex" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "Complex requires 2 arguments"));
                }
                let re = match &args[0] {
                    RustMathValue::Integer(n) => n.to_i64() as f64,
                    _ => return Err(EvalError::new("TypeError", "Complex requires numeric arguments")),
                };
                let im = match &args[1] {
                    RustMathValue::Integer(n) => n.to_i64() as f64,
                    _ => return Err(EvalError::new("TypeError", "Complex requires numeric arguments")),
                };
                Ok(RustMathValue::Complex(Complex::new(re, im)))
            }

            // Absolute value
            "abs" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => Ok(RustMathValue::Integer(n.abs())),
                    RustMathValue::Complex(c) => {
                        Ok(RustMathValue::String(format!("{}", c.abs())))
                    }
                    _ => Err(EvalError::new("TypeError", "abs requires a number")),
                }
            }

            // Power
            "pow" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "pow requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(base), RustMathValue::Integer(exp)) => {
                        let exp_val = exp.to_i64();
                        if exp_val < 0 {
                            return Err(EvalError::new("ValueError", "Negative exponent not supported for integers"));
                        }
                        if exp_val > 1000 {
                            return Err(EvalError::new("ValueError", "Exponent too large"));
                        }
                        Ok(RustMathValue::Integer(base.pow(exp_val as u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "pow requires integers")),
                }
            }

            // Unknown function
            _ => Err(EvalError::new("NameError", format!("Unknown function: {}", func_name))),
        }
    }

    fn parse_args(&mut self, args_str: &str) -> Result<Vec<RustMathValue>, EvalError> {
        if args_str.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut args = Vec::new();
        let mut depth = 0;
        let mut current = String::new();

        for ch in args_str.chars() {
            match ch {
                '(' | '[' | '{' => {
                    depth += 1;
                    current.push(ch);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    args.push(self.eval_expr(&current)?);
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        if !current.is_empty() {
            args.push(self.eval_expr(&current)?);
        }

        Ok(args)
    }

    fn try_eval_binary_op(&mut self, expr: &str) -> Result<Option<RustMathValue>, EvalError> {
        // Check for power operators first (right-to-left)
        if let Some(pos) = expr.rfind("**") {
            if pos > 0 {
                let left = expr[..pos].trim();
                let right = expr[pos + 2..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, "**", &right_val)?));
                }
            }
        }

        if let Some(pos) = expr.rfind('^') {
            if pos > 0 {
                let left = expr[..pos].trim();
                let right = expr[pos + 1..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, "^", &right_val)?));
                }
            }
        }

        // Addition/subtraction (left-to-right, find rightmost)
        for op in &["+", "-"] {
            if let Some(pos) = expr.rfind(op) {
                if pos > 0 {
                    let left = expr[..pos].trim();
                    let right = expr[pos + 1..].trim();
                    if !left.is_empty() && !right.is_empty() {
                        let left_val = self.eval_expr(left)?;
                        let right_val = self.eval_expr(right)?;
                        return Ok(Some(self.apply_binary_op(&left_val, op, &right_val)?));
                    }
                }
            }
        }

        // Multiplication
        if let Some(pos) = expr.rfind('*') {
            // Make sure it's not **
            if pos > 0 && !expr[..pos].ends_with('*') {
                let left = expr[..pos].trim();
                let right = expr[pos + 1..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, "*", &right_val)?));
                }
            }
        }

        Ok(None)
    }

    fn apply_binary_op(&self, left: &RustMathValue, op: &str, right: &RustMathValue) -> Result<RustMathValue, EvalError> {
        use RustMathValue::*;

        match (left, op, right) {
            // Integer operations
            (Integer(a), "+", Integer(b)) => Ok(Integer(a.clone() + b.clone())),
            (Integer(a), "-", Integer(b)) => Ok(Integer(a.clone() - b.clone())),
            (Integer(a), "*", Integer(b)) => Ok(Integer(a.clone() * b.clone())),
            (Integer(a), "**", Integer(b)) | (Integer(a), "^", Integer(b)) => {
                let exp = b.to_i64();
                if exp < 0 {
                    return Err(EvalError::new("ValueError", "Negative exponent"));
                }
                if exp > 10000 {
                    return Err(EvalError::new("ValueError", "Exponent too large"));
                }
                Ok(Integer(a.pow(exp as u32)))
            }

            // Rational operations
            (Rational(a), "+", Rational(b)) => Ok(Rational(a.clone() + b.clone())),
            (Rational(a), "-", Rational(b)) => Ok(Rational(a.clone() - b.clone())),
            (Rational(a), "*", Rational(b)) => Ok(Rational(a.clone() * b.clone())),
            (Rational(a), "/", Rational(b)) => Ok(Rational(a.clone() / b.clone())),

            // Complex operations
            (Complex(a), "+", Complex(b)) => Ok(Complex(a.clone() + b.clone())),
            (Complex(a), "-", Complex(b)) => Ok(Complex(a.clone() - b.clone())),
            (Complex(a), "*", Complex(b)) => Ok(Complex(a.clone() * b.clone())),

            _ => Err(EvalError::new("TypeError", format!("Cannot apply {} to {:?} and {:?}", op, left, right))),
        }
    }

    fn show_help(&self) -> EvalResult {
        let help_text = r#"RustMath Jupyter Kernel - Help

Available Types:
  Integer(n)         - Arbitrary precision integer
  Rational(n, d)     - Exact rational number
  Complex(re, im)    - Complex number
  Symbol("x")        - Symbolic variable

Integer Functions:
  factorial(n)       - Compute n!
  gcd(a, b)          - Greatest common divisor
  lcm(a, b)          - Least common multiple
  is_prime(n)        - Test primality
  factor(n)          - Prime factorization
  abs(n)             - Absolute value
  pow(base, exp)     - Exponentiation

Operators:
  +, -, *            - Arithmetic
  ** or ^            - Exponentiation
  /                  - Division (for rationals)

Commands:
  help               - Show this help
  vars               - Show defined variables

Examples:
  x = Integer(12345678901234567890)
  factorial(100)
  gcd(48, 18)
  factor(1234567)
  Rational(3, 4) + Rational(1, 2)
  2 ** 100
"#;
        EvalResult::text(help_text)
    }

    fn show_vars(&self) -> EvalResult {
        if self.variables.is_empty() {
            return EvalResult::text("No variables defined");
        }

        let mut lines = Vec::new();
        for (name, value) in &self.variables {
            let display = value.to_display();
            lines.push(format!("{} = {}", name, display.text));
        }

        EvalResult::text(lines.join("\n"))
    }
}

impl Default for ReplContext {
    fn default() -> Self {
        Self::new()
    }
}

fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_eval() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("42").unwrap();
        assert_eq!(result.text, "42");
    }

    #[test]
    fn test_factorial() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("factorial(5)").unwrap();
        assert_eq!(result.text, "120");
    }

    #[test]
    fn test_assignment() {
        let mut ctx = ReplContext::new();
        ctx.eval("x = 42").unwrap();
        let result = ctx.eval("x").unwrap();
        assert_eq!(result.text, "42");
    }

    #[test]
    fn test_addition() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("2+2").unwrap();
        assert_eq!(result.text, "4");
    }

    #[test]
    fn test_subtraction() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("10-3").unwrap();
        assert_eq!(result.text, "7");
    }

    #[test]
    fn test_multiplication() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("3*4").unwrap();
        assert_eq!(result.text, "12");
    }

    #[test]
    fn test_power() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("2**10").unwrap();
        assert_eq!(result.text, "1024");
    }

    #[test]
    fn test_large_integer() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("12345678901234567890").unwrap();
        assert_eq!(result.text, "12345678901234567890");
    }

    #[test]
    fn test_integer_constructor_large() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("Integer(99999999999999999999999999999999)").unwrap();
        assert_eq!(result.text, "99999999999999999999999999999999");
    }
}
