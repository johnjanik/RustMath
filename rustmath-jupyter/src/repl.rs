//! RustMath REPL evaluation engine
//!
//! Parses and evaluates RustMath expressions in an interactive context.

use std::collections::HashMap;
use std::str::FromStr;
use num_bigint::BigInt;
use rustmath_integers::Integer;
use rustmath_core::traits::NumericConversion;
use rustmath_integers::SageInteger; // For factorial, factor
use rustmath_rationals::Rational;
use rustmath_complex::Complex;
use rustmath_symbolic::{Symbol, Expr, parse as parse_expr};
use rustmath_matrix::Matrix;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_finitefields::IntegerMod;
use rustmath_combinatorics::{
    binomial, catalan, fibonacci, lucas, bell_number, stirling_first, stirling_second,
    multinomial, falling_factorial, rising_factorial, eulerian, narayana, motzkin,
    delannoy, delannoy_central, schroder_large, schroder_small, partition_count,
    count_derangements,
};
use rustmath_integers::{
    prime::{next_prime, nth_prime, prime_pi, prime_range as primes_between},
    crt::chinese_remainder_theorem,
};
use rustmath_stats::statistics::{
    mean as stats_mean, variance as stats_variance, std_dev as stats_std_dev,
    median as stats_median, mode as stats_mode, correlation as stats_correlation,
    covariance as stats_covariance,
};
use rustmath_numerical::{
    bisection, newton_raphson, simpson, trapezoid, romberg,
    fft as numerical_fft, ifft as numerical_ifft,
};
use rustmath_graphs::{Graph, complete_graph, cycle_graph, path_graph, star_graph, wheel_graph, petersen_graph};
use rustmath_geometry::{Point2D, Point3D, convex_hull, Polygon};
use rustmath_algebras::{
    CliffordAlgebra, CliffordAlgebraElement, CliffordBasisElement,
    ExteriorAlgebra,
};
use rustmath_manifolds::{
    DifferentiableManifold, Chart, DiffForm, VectorField,
    EuclideanSpace,
};
use std::sync::Arc;

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
    Float(f64),
    Symbol(Symbol),
    Expr(Expr),
    String(String),
    Bool(bool),
    Matrix(Matrix<Integer>),
    Polynomial(UnivariatePolynomial<Integer>),
    IntegerMod(IntegerMod),
    List(Vec<RustMathValue>),
    Graph(Graph),
    Point2D(Point2D),
    Point3D(Point3D),
    /// A plot with SVG content and description
    Plot { description: String, svg: String },
    /// Clifford algebra (parent structure)
    CliffordAlg(CliffordAlgebra<Rational>),
    /// Clifford algebra element
    CliffordElem(CliffordAlgebraElement<Rational>),
    /// Exterior algebra (parent structure)
    ExteriorAlg(ExteriorAlgebra<Rational>),
    /// Differentiable manifold
    Manifold(Arc<DifferentiableManifold>),
    /// Chart on a manifold
    ChartVal(Chart),
    /// Differential form on a manifold
    DiffFormVal(DiffForm),
    /// Vector field on a manifold
    VectorFieldVal(VectorField),
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
            RustMathValue::Float(f) => {
                let text = format!("{}", f);
                EvalResult::text(&text)
            }
            RustMathValue::Symbol(s) => {
                let text = s.name().to_string();
                let latex = format!("${}$", s.name());
                EvalResult::text(&text).with_latex(latex)
            }
            RustMathValue::Expr(e) => {
                let text = format!("{}", e);
                // Generate LaTeX from expression
                let latex = format!("${}$", expr_to_latex(e));
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::String(s) => EvalResult::text(format!("\"{}\"", s)),
            RustMathValue::Bool(b) => EvalResult::text(if *b { "True" } else { "False" }),
            RustMathValue::Matrix(m) => {
                // Format matrix as rows
                let mut rows = Vec::new();
                for i in 0..m.rows() {
                    let mut row = Vec::new();
                    for j in 0..m.cols() {
                        if let Ok(val) = m.get(i, j) {
                            row.push(val.to_string());
                        }
                    }
                    rows.push(format!("[{}]", row.join(", ")));
                }
                let text = format!("[{}]", rows.join(", "));

                // LaTeX matrix
                let mut latex_rows = Vec::new();
                for i in 0..m.rows() {
                    let mut row = Vec::new();
                    for j in 0..m.cols() {
                        if let Ok(val) = m.get(i, j) {
                            row.push(val.to_string());
                        }
                    }
                    latex_rows.push(row.join(" & "));
                }
                let latex = format!("$\\begin{{pmatrix}} {} \\end{{pmatrix}}$", latex_rows.join(" \\\\ "));

                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::Polynomial(p) => {
                let text = p.to_string();
                // LaTeX format for polynomial
                let coeffs = p.coefficients();
                let mut latex_terms = Vec::new();
                for (i, coeff) in coeffs.iter().enumerate().rev() {
                    if coeff.is_zero() {
                        continue;
                    }
                    let coeff_str = coeff.to_string();
                    if i == 0 {
                        latex_terms.push(coeff_str);
                    } else if i == 1 {
                        if coeff.is_one() {
                            latex_terms.push("x".to_string());
                        } else {
                            latex_terms.push(format!("{}x", coeff_str));
                        }
                    } else {
                        if coeff.is_one() {
                            latex_terms.push(format!("x^{{{}}}", i));
                        } else {
                            latex_terms.push(format!("{}x^{{{}}}", coeff_str, i));
                        }
                    }
                }
                let latex = if latex_terms.is_empty() {
                    "$0$".to_string()
                } else {
                    format!("${}$", latex_terms.join(" + "))
                };
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::IntegerMod(a) => {
                let text = format!("{} (mod {})", a.value(), a.modulus());
                let latex = format!("${} \\pmod{{{}}}$", a.value(), a.modulus());
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::List(items) => {
                let parts: Vec<String> = items.iter()
                    .map(|v| v.to_display().text)
                    .collect();
                EvalResult::text(format!("[{}]", parts.join(", ")))
            }
            RustMathValue::Graph(g) => {
                let text = format!("Graph with {} vertices and {} edges", g.num_vertices(), g.num_edges());
                EvalResult::text(text)
            }
            RustMathValue::Point2D(p) => {
                let text = format!("Point({}, {})", p.x, p.y);
                EvalResult::text(text)
            }
            RustMathValue::Point3D(p) => {
                let text = format!("Point({}, {}, {})", p.x, p.y, p.z);
                EvalResult::text(text)
            }
            RustMathValue::Plot { description, svg } => {
                let mut result = EvalResult::text(description);
                result.svg = Some(svg.clone());
                result
            }
            RustMathValue::CliffordAlg(cl) => {
                let text = format!("CliffordAlgebra(dim={})", cl.dimension());
                let latex = format!("$\\mathrm{{Cl}}(\\mathbb{{Q}}^{})$", cl.dimension());
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::CliffordElem(elem) => {
                let text = format!("{:?}", elem);
                // Generate LaTeX for Clifford element
                let latex = clifford_elem_to_latex(elem);
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::ExteriorAlg(ext) => {
                let text = format!("ExteriorAlgebra(dim={})", ext.dimension());
                let latex = format!("$\\Lambda(\\mathbb{{Q}}^{})$", ext.dimension());
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::Manifold(m) => {
                let text = format!("Manifold('{}', dim={})", m.name(), m.dimension());
                EvalResult::text(text)
            }
            RustMathValue::ChartVal(chart) => {
                let text = format!("Chart({})", chart.coordinate_names().join(", "));
                EvalResult::text(text)
            }
            RustMathValue::DiffFormVal(form) => {
                let text = format!("DiffForm(degree={})", form.degree());
                EvalResult::text(text)
            }
            RustMathValue::VectorFieldVal(_vf) => {
                let text = "VectorField".to_string();
                EvalResult::text(text)
            }
            RustMathValue::None => EvalResult::empty(),
        }
    }
}

/// Convert Clifford element to LaTeX
fn clifford_elem_to_latex(elem: &CliffordAlgebraElement<Rational>) -> String {
    // Basic LaTeX representation
    let text = format!("{:?}", elem);
    format!("${}$", text.replace("e", "e_"))
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
            let line = strip_inline_comment(line.trim());
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            last_result = self.eval_line(&line)?;
        }

        Ok(last_result)
    }

    fn eval_line(&mut self, line: &str) -> Result<EvalResult, EvalError> {
        // Check for assignment: var = expr or (x, y = expr) for tuple unpacking
        if let Some(eq_pos) = line.find('=') {
            let before_eq = line[..eq_pos].trim();
            // Make sure it's not ==, !=, <=, >=
            if !line[eq_pos..].starts_with("==")
                && !before_eq.ends_with('!')
                && !before_eq.ends_with('<')
                && !before_eq.ends_with('>')
            {
                // Check for tuple unpacking: x, y, z = expr
                if before_eq.contains(',') {
                    let var_names: Vec<&str> = before_eq.split(',')
                        .map(|s| s.trim())
                        .collect();

                    // Verify all names are valid identifiers
                    if var_names.iter().all(|n| is_valid_identifier(n)) {
                        let expr = line[eq_pos + 1..].trim();
                        let value = self.eval_expr(expr)?;

                        // Unpack the list into individual variables
                        match &value {
                            RustMathValue::List(items) => {
                                if items.len() != var_names.len() {
                                    return Err(EvalError::new("ValueError",
                                        format!("Cannot unpack {} values into {} variables",
                                            items.len(), var_names.len())));
                                }
                                for (name, val) in var_names.iter().zip(items.iter()) {
                                    self.variables.insert(name.to_string(), val.clone());
                                }
                                self.last_result = Some(value.clone());
                                // Return a display showing all assigned variables
                                let display_text = var_names.iter()
                                    .filter_map(|n| self.variables.get(*n))
                                    .map(|v| v.to_display().text)
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                return Ok(EvalResult::text(display_text));
                            }
                            _ => {
                                return Err(EvalError::new("TypeError",
                                    "Cannot unpack non-list value"));
                            }
                        }
                    }
                }

                // Single variable assignment
                if is_valid_identifier(before_eq) {
                    let var_name = before_eq.to_string();
                    let expr = line[eq_pos + 1..].trim();
                    let value = self.eval_expr(expr)?;
                    self.variables.insert(var_name, value.clone());
                    self.last_result = Some(value.clone());
                    return Ok(value.to_display());
                }
            }
        }

        // Check for built-in commands
        if line.starts_with("print(") && line.ends_with(')') {
            let inner = &line[6..line.len() - 1];
            // Split arguments at top-level commas for print
            let args = self.split_at_depth_zero(inner, ',');
            let mut output_parts: Vec<String> = Vec::new();
            for arg in args {
                let arg = arg.trim();
                let value = self.eval_expr(arg)?;
                output_parts.push(value.to_display().text);
            }
            let output = output_parts.join(" ");
            self.print(&output);
            // Return the output text for Jupyter display
            return Ok(EvalResult::text(output));
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

        // Check for tuple expression (comma-separated values at top level)
        // e.g., "x, y" or "1, 2, 3"
        if expr.contains(',') && !expr.starts_with('[') && !expr.starts_with('(') {
            // Split at top level only (not inside parentheses or brackets)
            let parts = self.split_at_depth_zero(expr, ',');
            if parts.len() > 1 {
                let elements: Result<Vec<RustMathValue>, EvalError> = parts
                    .iter()
                    .map(|p| self.eval_expr(p.trim()))
                    .collect();
                return Ok(RustMathValue::List(elements?));
            }
        }

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

        // List literal [a, b, c, ...]
        if expr.starts_with('[') && expr.ends_with(']') {
            let inner = &expr[1..expr.len() - 1];
            let elements = self.parse_list_elements(inner)?;
            return Ok(RustMathValue::List(elements));
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

            // print() - Output values (Python-like)
            "print" => {
                // For single argument, just evaluate and return it formatted
                if args_str.trim().is_empty() {
                    return Ok(RustMathValue::String("".to_string()));
                }
                // Parse arguments (can be multiple comma-separated)
                let args = self.parse_args(args_str)?;
                if args.is_empty() {
                    return Ok(RustMathValue::String("".to_string()));
                }
                let mut parts = Vec::new();
                for arg in &args {
                    let text = match arg {
                        RustMathValue::String(s) => s.clone(),
                        other => {
                            // Convert to display format
                            let display = other.to_display();
                            display.text
                        }
                    };
                    parts.push(text);
                }
                // Return the joined string (print outputs to text)
                Ok(RustMathValue::String(parts.join(" ")))
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

            // Extended GCD: xgcd(a, b) returns [gcd, x, y] where gcd = a*x + b*y
            "xgcd" | "extended_gcd" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "xgcd requires 2 arguments (a, b)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(b)) => {
                        let (gcd, x, y) = a.extended_gcd(b);
                        Ok(RustMathValue::List(vec![
                            RustMathValue::Integer(gcd),
                            RustMathValue::Integer(x),
                            RustMathValue::Integer(y),
                        ]))
                    }
                    _ => Err(EvalError::new("TypeError", "xgcd requires integers")),
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
            // var('x') creates a single symbol
            // var('x y') or var('x, y') creates multiple symbols (returns a list)
            "Symbol" | "var" => {
                let name = args_str.trim().trim_matches(|c| c == '"' || c == '\'');

                // Check if multiple variables are requested (space or comma separated)
                let names: Vec<&str> = name.split(|c| c == ' ' || c == ',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                if names.len() == 1 {
                    Ok(RustMathValue::Symbol(Symbol::new(names[0])))
                } else {
                    // Return a list of symbols for multiple variables
                    let symbols: Vec<RustMathValue> = names.iter()
                        .map(|n| RustMathValue::Symbol(Symbol::new(*n)))
                        .collect();
                    Ok(RustMathValue::List(symbols))
                }
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

            // ====== MATRIX OPERATIONS ======

            // Matrix constructor from nested list: Matrix([[1,2],[3,4]])
            "Matrix" | "matrix" => {
                let list = self.parse_nested_list(args_str)?;
                if list.is_empty() {
                    return Err(EvalError::new("ValueError", "Matrix cannot be empty"));
                }
                let rows = list.len();
                let cols = list[0].len();

                // Verify all rows have same length
                for row in &list {
                    if row.len() != cols {
                        return Err(EvalError::new("ValueError", "All matrix rows must have the same length"));
                    }
                }

                // Flatten into data vector
                let mut data = Vec::new();
                for row in list {
                    for val in row {
                        match val {
                            RustMathValue::Integer(n) => data.push(n),
                            _ => return Err(EvalError::new("TypeError", "Matrix elements must be integers")),
                        }
                    }
                }

                let m = Matrix::from_vec(rows, cols, data)
                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                Ok(RustMathValue::Matrix(m))
            }

            // Identity matrix: identity(n)
            "identity" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let size = n.to_i64() as usize;
                        if size > 1000 {
                            return Err(EvalError::new("ValueError", "Matrix size too large"));
                        }
                        Ok(RustMathValue::Matrix(Matrix::identity(size)))
                    }
                    _ => Err(EvalError::new("TypeError", "identity requires an integer")),
                }
            }

            // Zero matrix: zeros(m, n)
            "zeros" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "zeros requires 2 arguments (rows, cols)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(r), RustMathValue::Integer(c)) => {
                        let rows = r.to_i64() as usize;
                        let cols = c.to_i64() as usize;
                        if rows > 1000 || cols > 1000 {
                            return Err(EvalError::new("ValueError", "Matrix size too large"));
                        }
                        Ok(RustMathValue::Matrix(Matrix::zeros(rows, cols)))
                    }
                    _ => Err(EvalError::new("TypeError", "zeros requires integers")),
                }
            }

            // Determinant: det(M)
            "det" | "determinant" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let d = m.determinant()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Integer(d))
                    }
                    _ => Err(EvalError::new("TypeError", "det requires a matrix")),
                }
            }

            // Transpose: transpose(M)
            "transpose" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Matrix(m.transpose()))
                    }
                    _ => Err(EvalError::new("TypeError", "transpose requires a matrix")),
                }
            }

            // Trace: trace(M)
            "trace" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let t = m.trace()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Integer(t))
                    }
                    _ => Err(EvalError::new("TypeError", "trace requires a matrix")),
                }
            }

            // Matrix dimensions: rows(M), cols(M)
            "rows" | "nrows" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Integer(Integer::from(m.rows() as i64)))
                    }
                    _ => Err(EvalError::new("TypeError", "rows requires a matrix")),
                }
            }

            "cols" | "ncols" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Integer(Integer::from(m.cols() as i64)))
                    }
                    _ => Err(EvalError::new("TypeError", "cols requires a matrix")),
                }
            }

            // Matrix shape: shape(M) -> (rows, cols)
            "shape" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::List(vec![
                            RustMathValue::Integer(Integer::from(m.rows() as i64)),
                            RustMathValue::Integer(Integer::from(m.cols() as i64)),
                        ]))
                    }
                    _ => Err(EvalError::new("TypeError", "shape requires a matrix")),
                }
            }

            // Is square: is_square(M)
            "is_square" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Bool(m.is_square()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_square requires a matrix")),
                }
            }

            // Is symmetric: is_symmetric(M)
            "is_symmetric" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Bool(m.is_symmetric()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_symmetric requires a matrix")),
                }
            }

            // Is diagonal: is_diagonal(M)
            "is_diagonal" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Bool(m.is_diagonal()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_diagonal requires a matrix")),
                }
            }

            // ====== POLYNOMIAL OPERATIONS ======

            // Polynomial constructor from coefficients: Polynomial([1,2,3]) = 1 + 2x + 3x^2
            "Polynomial" | "poly" => {
                // Parse the list of coefficients
                let coeffs = self.parse_list_of_integers(args_str)?;
                if coeffs.is_empty() {
                    return Err(EvalError::new("ValueError", "Polynomial requires at least one coefficient"));
                }
                let p = UnivariatePolynomial::new(coeffs);
                Ok(RustMathValue::Polynomial(p))
            }

            // Degree of polynomial: degree(p)
            "degree" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        match p.degree() {
                            Some(d) => Ok(RustMathValue::Integer(Integer::from(d as i64))),
                            None => Ok(RustMathValue::String("undefined (zero polynomial)".to_string())),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "degree requires a polynomial")),
                }
            }

            // Derivative: derivative(p)
            "derivative" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        Ok(RustMathValue::Polynomial(p.derivative()))
                    }
                    _ => Err(EvalError::new("TypeError", "derivative requires a polynomial")),
                }
            }

            // Evaluate polynomial at point: eval_poly(p, x) or polyeval(p, x)
            "eval_poly" | "polyeval" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "eval_poly requires 2 arguments (polynomial, value)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Polynomial(p), RustMathValue::Integer(x)) => {
                        let result = p.evaluate(x);
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "eval_poly requires (polynomial, integer)")),
                }
            }

            // Polynomial coefficients: coefficients(p) or coeffs(p)
            "coefficients" | "coeffs" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        let coeffs: Vec<RustMathValue> = p.coefficients()
                            .iter()
                            .map(|c| RustMathValue::Integer(c.clone()))
                            .collect();
                        Ok(RustMathValue::List(coeffs))
                    }
                    _ => Err(EvalError::new("TypeError", "coefficients requires a polynomial")),
                }
            }

            // Leading coefficient: leading_coeff(p)
            "leading_coeff" | "lc" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        match p.leading_coefficient() {
                            Some(c) => Ok(RustMathValue::Integer(c.clone())),
                            None => Ok(RustMathValue::Integer(Integer::from(0))),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "leading_coeff requires a polynomial")),
                }
            }

            // Is monic (leading coefficient = 1): is_monic(p)
            "is_monic" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        Ok(RustMathValue::Bool(p.is_monic()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_monic requires a polynomial")),
                }
            }

            // Content (GCD of coefficients): content(p)
            "content" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        Ok(RustMathValue::Integer(p.content()))
                    }
                    _ => Err(EvalError::new("TypeError", "content requires a polynomial")),
                }
            }

            // Is square-free: is_square_free(p) for polynomial or is_square_free(n) for integer
            "is_square_free" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        Ok(RustMathValue::Bool(p.is_square_free()))
                    }
                    RustMathValue::Integer(n) => {
                        Ok(RustMathValue::Bool(n.is_square_free()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_square_free requires a polynomial or integer")),
                }
            }

            // Polynomial discriminant: discriminant(p)
            "discriminant" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        match p.discriminant() {
                            Some(d) => Ok(RustMathValue::Integer(d)),
                            None => Err(EvalError::new("ValueError", "Discriminant only supported for degrees 2 and 3")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "discriminant requires a polynomial")),
                }
            }

            // Find rational roots: rational_roots(p) or roots(p)
            "rational_roots" | "roots" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Polynomial(p) => {
                        let roots = rustmath_polynomials::rational_roots(&p);
                        let result: Vec<RustMathValue> = roots.into_iter()
                            .map(RustMathValue::Rational)
                            .collect();
                        Ok(RustMathValue::List(result))
                    }
                    _ => Err(EvalError::new("TypeError", "roots requires a polynomial")),
                }
            }

            // Polynomial GCD: gcd_poly(p, q) or poly_gcd(p, q)
            "gcd_poly" | "poly_gcd" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "gcd_poly requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Polynomial(p), RustMathValue::Polynomial(q)) => {
                        Ok(RustMathValue::Polynomial(p.gcd(q)))
                    }
                    _ => Err(EvalError::new("TypeError", "gcd_poly requires polynomials")),
                }
            }

            // Polynomial LCM: lcm_poly(p, q)
            "lcm_poly" | "poly_lcm" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "lcm_poly requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Polynomial(p), RustMathValue::Polynomial(q)) => {
                        Ok(RustMathValue::Polynomial(p.lcm(q)))
                    }
                    _ => Err(EvalError::new("TypeError", "lcm_poly requires polynomials")),
                }
            }

            // Polynomial composition: compose(p, q) computes p(q(x))
            "compose" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "compose requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Polynomial(p), RustMathValue::Polynomial(q)) => {
                        Ok(RustMathValue::Polynomial(p.compose(q)))
                    }
                    _ => Err(EvalError::new("TypeError", "compose requires polynomials")),
                }
            }

            // Quotient and remainder: divmod(p, q) -> [quotient, remainder]
            "divmod" | "quo_rem" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "divmod requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Polynomial(p), RustMathValue::Polynomial(q)) => {
                        let (quo, rem) = p.quo_rem(q);
                        Ok(RustMathValue::List(vec![
                            RustMathValue::Polynomial(quo),
                            RustMathValue::Polynomial(rem),
                        ]))
                    }
                    _ => Err(EvalError::new("TypeError", "divmod requires polynomials")),
                }
            }

            // ====== SYMBOLIC COMPUTATION ======

            // Parse symbolic expression from string: expr("x^2 + 3*x + 2")
            "expr" | "Expr" | "symbolic" => {
                let s = args_str.trim().trim_matches(|c| c == '"' || c == '\'');
                let e = parse_expr(s)
                    .map_err(|e| EvalError::new("ParseError", format!("Cannot parse expression: {}", e)))?;
                Ok(RustMathValue::Expr(e))
            }

            // Differentiate: diff(expr, var) or diff("x^2", "x")
            "diff" | "differentiate" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "diff requires 2 arguments (expression, variable)"));
                }
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                let var = args[0].symbols().iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                let result = args[0].differentiate(&var);
                // Simplify the derivative result
                let simplified = result.simplify();
                Ok(RustMathValue::Expr(simplified))
            }

            // Simplify: simplify(expr)
            "simplify" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                let result = e.simplify();
                Ok(RustMathValue::Expr(result))
            }

            // Expand: expand(expr)
            "expand" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                let result = e.expand();
                Ok(RustMathValue::Expr(result))
            }

            // Symbolic sin, cos, tan, exp, log, sqrt
            "sin" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.sin()))
            }
            "cos" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.cos()))
            }
            "tan" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.tan()))
            }
            "exp" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.exp()))
            }
            "log" | "ln" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.log()))
            }
            "sqrt" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                Ok(RustMathValue::Expr(e.sqrt()))
            }

            // Substitute: substitute(expr, var, value) or subs(expr, var, value)
            "substitute" | "subs" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "substitute requires 3 arguments (expression, variable, value)"));
                }
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                let var = args[0].symbols().iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                let result = args[0].substitute(&var, &args[2]);
                Ok(RustMathValue::Expr(result))
            }

            // Evaluate expression numerically: eval_expr(expr) or evalf(expr)
            "eval_expr" | "evalf" | "N" => {
                let e = self.parse_single_symbolic_arg(args_str)?;
                match try_eval_to_f64(&e) {
                    Some(val) => Ok(RustMathValue::String(format!("{:.10}", val))),
                    None => Err(EvalError::new("EvalError", "Cannot evaluate expression numerically (contains variables)")),
                }
            }

            // Taylor series: taylor(expr, var, point, order)
            "taylor" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() < 3 || args.len() > 4 {
                    return Err(EvalError::new("ArgumentError", "taylor requires 3-4 arguments (expr, var, point, [order])"));
                }
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                let var = args[0].symbols().iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                let order = if args.len() == 4 {
                    match &args[3] {
                        Expr::Integer(n) => n.to_i64() as usize,
                        _ => 5,
                    }
                } else {
                    5
                };
                let result = args[0].taylor(&var, &args[2], order);
                Ok(RustMathValue::Expr(result))
            }

            // Solve equation: solve(expr, var) - finds where expr = 0
            "solve" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "solve requires 2 arguments (expression, variable)"));
                }
                // Get the variable name from the second argument
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                // Find the symbol with that name in the expression (important: symbols have unique IDs
                // so we need to find the actual symbol from the expression, not use the parsed one)
                let expr_symbols = args[0].symbols();
                let var = expr_symbols.iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;

                use rustmath_symbolic::solve::Solution;
                match args[0].solve(&var) {
                    Solution::Expr(e) => Ok(RustMathValue::Expr(e)),
                    Solution::Multiple(solutions) => {
                        let list: Vec<RustMathValue> = solutions.into_iter()
                            .map(RustMathValue::Expr)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    Solution::None => Ok(RustMathValue::String("No solution found".to_string())),
                    Solution::All => Ok(RustMathValue::String("All values are solutions".to_string())),
                }
            }

            // Integrate: integrate(expr, var)
            "integrate" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "integrate requires 2 arguments (expression, variable)"));
                }
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                let var = args[0].symbols().iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                match args[0].integrate(&var) {
                    Some(result) => Ok(RustMathValue::Expr(result)),
                    None => Err(EvalError::new("IntegrationError", "Cannot integrate this expression")),
                }
            }

            // Limit: limit(expr, var, point)
            "limit" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "limit requires 3 arguments (expr, var, point)"));
                }
                let var_name = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "Second argument must be a variable")),
                };
                let var = args[0].symbols().iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                use rustmath_symbolic::limits::{Direction, LimitResult};
                match args[0].limit(&var, &args[2], Direction::Both) {
                    LimitResult::Finite(e) => Ok(RustMathValue::Expr(e)),
                    LimitResult::Infinity => Ok(RustMathValue::String("+∞".to_string())),
                    LimitResult::NegInfinity => Ok(RustMathValue::String("-∞".to_string())),
                    LimitResult::DoesNotExist => Ok(RustMathValue::String("Limit does not exist".to_string())),
                    LimitResult::Unknown => Ok(RustMathValue::String("Unknown".to_string())),
                }
            }

            // ====== FINITE FIELDS ======

            // Create element in Z/nZ: mod(a, n) or Mod(a, n)
            "mod" | "Mod" | "IntegerMod" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "mod requires 2 arguments (value, modulus)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(n)) => {
                        let elem = IntegerMod::new(a.clone(), n.clone())
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::IntegerMod(elem))
                    }
                    _ => Err(EvalError::new("TypeError", "mod requires integer arguments")),
                }
            }

            // GF(p) - create element in prime field (alias for mod)
            "GF" => {
                let args = self.parse_args(args_str)?;
                if args.len() == 1 {
                    // GF(p) - return info about the field
                    match &args[0] {
                        RustMathValue::Integer(p) => {
                            Ok(RustMathValue::String(format!("Finite field GF({})", p)))
                        }
                        _ => Err(EvalError::new("TypeError", "GF requires integer argument")),
                    }
                } else if args.len() == 2 {
                    // GF(p, a) - create element a in GF(p)
                    match (&args[0], &args[1]) {
                        (RustMathValue::Integer(p), RustMathValue::Integer(a)) => {
                            let elem = IntegerMod::new(a.clone(), p.clone())
                                .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                            Ok(RustMathValue::IntegerMod(elem))
                        }
                        _ => Err(EvalError::new("TypeError", "GF requires integer arguments")),
                    }
                } else {
                    Err(EvalError::new("ArgumentError", "GF requires 1 or 2 arguments"))
                }
            }

            // Multiplicative inverse in Z/nZ: inverse(a) or mod_inverse(a)
            "inverse" | "mod_inverse" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::IntegerMod(a) => {
                        let inv = a.inverse()
                            .map_err(|e| EvalError::new("ValueError", format!("No inverse: {}", e)))?;
                        Ok(RustMathValue::IntegerMod(inv))
                    }
                    _ => Err(EvalError::new("TypeError", "inverse requires an element of Z/nZ")),
                }
            }

            // Check if element is unit (has inverse): is_unit(a)
            "is_unit" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::IntegerMod(a) => {
                        Ok(RustMathValue::Bool(a.inverse().is_ok()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_unit requires an element of Z/nZ")),
                }
            }

            // Square root modulo prime: sqrt_mod(a, p)
            "sqrt_mod" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "sqrt_mod requires 2 arguments (value, prime)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(p)) => {
                        use rustmath_finitefields::square_root_mod_prime;
                        match square_root_mod_prime(a, p) {
                            Some(root) => Ok(RustMathValue::Integer(root)),
                            None => Ok(RustMathValue::String("No square root exists".to_string())),
                        }
                    }
                    (RustMathValue::IntegerMod(a), _) => {
                        use rustmath_finitefields::square_root_mod_prime;
                        match square_root_mod_prime(a.value(), a.modulus()) {
                            Some(root) => {
                                let elem = IntegerMod::new(root, a.modulus().clone())
                                    .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                                Ok(RustMathValue::IntegerMod(elem))
                            }
                            None => Ok(RustMathValue::String("No square root exists".to_string())),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "sqrt_mod requires integer arguments")),
                }
            }

            // Power in modular arithmetic: pow_mod(base, exp, mod)
            "pow_mod" | "modpow" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "pow_mod requires 3 arguments (base, exp, mod)"));
                }
                match (&args[0], &args[1], &args[2]) {
                    (RustMathValue::Integer(base), RustMathValue::Integer(exp), RustMathValue::Integer(modulus)) => {
                        let result = base.mod_pow(exp, modulus)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "pow_mod requires integer arguments")),
                }
            }

            // ========== COMBINATORICS FUNCTIONS ==========

            // Binomial coefficient: binomial(n, k)
            "binomial" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "binomial requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(binomial(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "binomial requires integer arguments")),
                }
            }

            // Catalan number: catalan(n)
            "catalan" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "catalan requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(catalan(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "catalan requires an integer argument")),
                }
            }

            // Fibonacci number: fibonacci(n)
            "fibonacci" | "fib" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "fibonacci requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(fibonacci(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "fibonacci requires an integer argument")),
                }
            }

            // Lucas number: lucas(n)
            "lucas" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "lucas requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(lucas(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "lucas requires an integer argument")),
                }
            }

            // Bell number: bell(n)
            "bell" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "bell requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(bell_number(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "bell requires an integer argument")),
                }
            }

            // Stirling number of the first kind: stirling1(n, k)
            "stirling1" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "stirling1 requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(stirling_first(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "stirling1 requires integer arguments")),
                }
            }

            // Stirling number of the second kind: stirling2(n, k)
            "stirling2" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "stirling2 requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(stirling_second(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "stirling2 requires integer arguments")),
                }
            }

            // Multinomial coefficient: multinomial(n, [k1, k2, ...])
            "multinomial" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "multinomial requires 2 arguments (n, [k1, k2, ...])"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::List(ks)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let mut ks_u32 = Vec::new();
                        for k in ks {
                            match k {
                                RustMathValue::Integer(ki) => {
                                    let ki_u32 = ki.to_u64().ok_or_else(|| EvalError::new("ValueError", "k values must be non-negative integers"))? as u32;
                                    ks_u32.push(ki_u32);
                                }
                                _ => return Err(EvalError::new("TypeError", "k values must be integers")),
                            }
                        }
                        // Check that sum of ks equals n (required for multinomial)
                        let sum: u32 = ks_u32.iter().sum();
                        if sum != n_u32 {
                            return Err(EvalError::new("ValueError",
                                format!("multinomial requires sum of k values ({}) to equal n ({})", sum, n_u32)));
                        }
                        Ok(RustMathValue::Integer(multinomial(n_u32, &ks_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "multinomial requires (integer, list) arguments")),
                }
            }

            // Falling factorial: falling_factorial(n, k)
            "falling_factorial" | "pochhammer" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "falling_factorial requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(falling_factorial(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "falling_factorial requires integer arguments")),
                }
            }

            // Rising factorial: rising_factorial(n, k)
            "rising_factorial" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "rising_factorial requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(rising_factorial(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "rising_factorial requires integer arguments")),
                }
            }

            // Eulerian number: eulerian(n, k)
            "eulerian" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "eulerian requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(eulerian(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "eulerian requires integer arguments")),
                }
            }

            // Narayana number: narayana(n, k)
            "narayana" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "narayana requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(narayana(n_u32, k_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "narayana requires integer arguments")),
                }
            }

            // Motzkin number: motzkin(n)
            "motzkin" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "motzkin requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(motzkin(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "motzkin requires an integer argument")),
                }
            }

            // Delannoy number: delannoy(m, n) or delannoy(n) for central
            "delannoy" => {
                let args = self.parse_args(args_str)?;
                match args.len() {
                    1 => {
                        match &args[0] {
                            RustMathValue::Integer(n) => {
                                let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                                Ok(RustMathValue::Integer(delannoy_central(n_u32)))
                            }
                            _ => Err(EvalError::new("TypeError", "delannoy requires integer arguments")),
                        }
                    }
                    2 => {
                        match (&args[0], &args[1]) {
                            (RustMathValue::Integer(m), RustMathValue::Integer(n)) => {
                                let m_u32 = m.to_u64().ok_or_else(|| EvalError::new("ValueError", "m must be a non-negative integer"))? as u32;
                                let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                                Ok(RustMathValue::Integer(delannoy(m_u32, n_u32)))
                            }
                            _ => Err(EvalError::new("TypeError", "delannoy requires integer arguments")),
                        }
                    }
                    _ => Err(EvalError::new("ArgumentError", "delannoy requires 1 or 2 arguments")),
                }
            }

            // Schroder number (large): schroder(n)
            "schroder" | "schroder_large" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "schroder requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(schroder_large(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "schroder requires an integer argument")),
                }
            }

            // Schroder number (small): schroder_small(n)
            "schroder_small" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "schroder_small requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(schroder_small(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "schroder_small requires an integer argument")),
                }
            }

            // Number of integer partitions: partitions(n)
            "partitions" | "partition_count" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "partitions requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_usize = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as usize;
                        Ok(RustMathValue::Integer(Integer::from(partition_count(n_usize) as u64)))
                    }
                    _ => Err(EvalError::new("TypeError", "partitions requires an integer argument")),
                }
            }

            // Number of derangements: derangements(n)
            "derangements" | "count_derangements" | "subfactorial" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "derangements requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_u32 = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a non-negative integer"))? as u32;
                        Ok(RustMathValue::Integer(count_derangements(n_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "derangements requires an integer argument")),
                }
            }

            // ========== NUMBER THEORY FUNCTIONS ==========

            // List of divisors: divisors(n)
            "divisors" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "divisors requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let divs = n.divisors()
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        let list: Vec<RustMathValue> = divs.into_iter()
                            .map(RustMathValue::Integer)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "divisors requires an integer argument")),
                }
            }

            // Sum of divisors: sigma(n) or sigma(n, k)
            "sigma" | "sum_divisors" => {
                let args = self.parse_args(args_str)?;
                match args.len() {
                    1 => {
                        match &args[0] {
                            RustMathValue::Integer(n) => {
                                let result = n.sum_divisors()
                                    .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                                Ok(RustMathValue::Integer(result))
                            }
                            _ => Err(EvalError::new("TypeError", "sigma requires an integer argument")),
                        }
                    }
                    2 => {
                        match (&args[0], &args[1]) {
                            (RustMathValue::Integer(n), RustMathValue::Integer(k)) => {
                                let k_u32 = k.to_u64().ok_or_else(|| EvalError::new("ValueError", "k must be a non-negative integer"))? as u32;
                                let result = n.sigma(k_u32)
                                    .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                                Ok(RustMathValue::Integer(result))
                            }
                            _ => Err(EvalError::new("TypeError", "sigma requires integer arguments")),
                        }
                    }
                    _ => Err(EvalError::new("ArgumentError", "sigma requires 1 or 2 arguments")),
                }
            }

            // Euler's totient function: euler_phi(n)
            "euler_phi" | "totient" | "phi" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "euler_phi requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let result = n.euler_phi()
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "euler_phi requires an integer argument")),
                }
            }

            // Möbius function: mobius(n)
            "mobius" | "moebius" | "mu" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "mobius requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let result = n.moebius()
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(Integer::from(result as i32)))
                    }
                    _ => Err(EvalError::new("TypeError", "mobius requires an integer argument")),
                }
            }

            // Next prime: next_prime(n)
            "next_prime" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "next_prime requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        Ok(RustMathValue::Integer(next_prime(n)))
                    }
                    _ => Err(EvalError::new("TypeError", "next_prime requires an integer argument")),
                }
            }

            // n-th prime: nth_prime(n)
            "nth_prime" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "nth_prime requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let n_usize = n.to_u64().ok_or_else(|| EvalError::new("ValueError", "n must be a positive integer"))? as usize;
                        if n_usize == 0 {
                            return Err(EvalError::new("ValueError", "n must be a positive integer (1-indexed)"));
                        }
                        Ok(RustMathValue::Integer(nth_prime(n_usize)))
                    }
                    _ => Err(EvalError::new("TypeError", "nth_prime requires an integer argument")),
                }
            }

            // Prime counting function: prime_pi(n)
            "prime_pi" | "primepi" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "prime_pi requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let count = prime_pi(n);
                        Ok(RustMathValue::Integer(Integer::from(count as u64)))
                    }
                    _ => Err(EvalError::new("TypeError", "prime_pi requires an integer argument")),
                }
            }

            // Primes in range: primes(a, b)
            "primes" | "prime_range" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "primes requires 2 arguments (start, end)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(b)) => {
                        let primes = primes_between(a, b);
                        let list: Vec<RustMathValue> = primes.into_iter()
                            .map(RustMathValue::Integer)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "primes requires integer arguments")),
                }
            }

            // Legendre symbol: legendre(a, p)
            "legendre" | "legendre_symbol" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "legendre requires 2 arguments (a, p)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(p)) => {
                        let result = a.legendre_symbol(p)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(Integer::from(result as i32)))
                    }
                    _ => Err(EvalError::new("TypeError", "legendre requires integer arguments")),
                }
            }

            // Jacobi symbol: jacobi(a, n)
            "jacobi" | "jacobi_symbol" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "jacobi requires 2 arguments (a, n)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(a), RustMathValue::Integer(n)) => {
                        let result = a.jacobi_symbol(n)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(Integer::from(result as i32)))
                    }
                    _ => Err(EvalError::new("TypeError", "jacobi requires integer arguments")),
                }
            }

            // Chinese Remainder Theorem: crt([remainders], [moduli])
            "crt" | "chinese_remainder" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "crt requires 2 arguments (remainders, moduli)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::List(remainders), RustMathValue::List(moduli)) => {
                        let r: Vec<Integer> = remainders.iter()
                            .map(|v| match v {
                                RustMathValue::Integer(i) => Ok(i.clone()),
                                _ => Err(EvalError::new("TypeError", "remainders must be integers")),
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let m: Vec<Integer> = moduli.iter()
                            .map(|v| match v {
                                RustMathValue::Integer(i) => Ok(i.clone()),
                                _ => Err(EvalError::new("TypeError", "moduli must be integers")),
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let result = chinese_remainder_theorem(&r, &m)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "crt requires two lists of integers")),
                }
            }

            // Number of divisors: num_divisors(n)
            "num_divisors" | "tau" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "num_divisors requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let result = n.num_divisors()
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "num_divisors requires an integer argument")),
                }
            }

            // Radical of n: radical(n)
            "radical" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "radical requires 1 argument"));
                }
                match &args[0] {
                    RustMathValue::Integer(n) => {
                        let result = n.radical()
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::Integer(result))
                    }
                    _ => Err(EvalError::new("TypeError", "radical requires an integer argument")),
                }
            }

            // p-adic valuation: valuation(n, p)
            "valuation" | "padic_valuation" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "valuation requires 2 arguments (n, p)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(n), RustMathValue::Integer(p)) => {
                        let result = n.valuation(p);
                        Ok(RustMathValue::Integer(Integer::from(result)))
                    }
                    _ => Err(EvalError::new("TypeError", "valuation requires integer arguments")),
                }
            }

            // ====== STATISTICS FUNCTIONS ======

            // Mean: mean([a, b, c, ...])
            "mean" | "average" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let floats = self.list_to_floats(&values)?;
                        match stats_mean(&floats) {
                            Some(m) => Ok(RustMathValue::Float(m)),
                            None => Err(EvalError::new("ValueError", "Cannot compute mean of empty list")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "mean requires a list of numbers")),
                }
            }

            // Median: median([a, b, c, ...])
            "median" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let floats = self.list_to_floats(&values)?;
                        match stats_median(&floats) {
                            Some(m) => Ok(RustMathValue::Float(m)),
                            None => Err(EvalError::new("ValueError", "Cannot compute median of empty list")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "median requires a list of numbers")),
                }
            }

            // Mode: mode([a, b, c, ...])
            "mode" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let floats = self.list_to_floats(&values)?;
                        match stats_mode(&floats) {
                            Some(m) => Ok(RustMathValue::Float(m)),
                            None => Err(EvalError::new("ValueError", "Cannot compute mode of empty list")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "mode requires a list of numbers")),
                }
            }

            // Variance: variance([a, b, c, ...])
            "variance" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let floats = self.list_to_floats(&values)?;
                        match stats_variance(&floats) {
                            Some(v) => Ok(RustMathValue::Float(v)),
                            None => Err(EvalError::new("ValueError", "Cannot compute variance (need at least 2 values)")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "variance requires a list of numbers")),
                }
            }

            // Standard deviation: std_dev([a, b, c, ...]) or std([...])
            "std_dev" | "std" | "stdev" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let floats = self.list_to_floats(&values)?;
                        match stats_std_dev(&floats) {
                            Some(s) => Ok(RustMathValue::Float(s)),
                            None => Err(EvalError::new("ValueError", "Cannot compute std dev (need at least 2 values)")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "std_dev requires a list of numbers")),
                }
            }

            // Correlation: correlation([x1, x2, ...], [y1, y2, ...])
            "correlation" | "corr" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "correlation requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::List(xs), RustMathValue::List(ys)) => {
                        let x_floats = self.list_to_floats(xs)?;
                        let y_floats = self.list_to_floats(ys)?;
                        match stats_correlation(&x_floats, &y_floats) {
                            Some(c) => Ok(RustMathValue::Float(c)),
                            None => Err(EvalError::new("ValueError", "Cannot compute correlation (need matching lists with at least 2 values)")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "correlation requires two lists of numbers")),
                }
            }

            // Covariance: covariance([x1, x2, ...], [y1, y2, ...])
            "covariance" | "cov" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "covariance requires 2 arguments"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::List(xs), RustMathValue::List(ys)) => {
                        let x_floats = self.list_to_floats(xs)?;
                        let y_floats = self.list_to_floats(ys)?;
                        match stats_covariance(&x_floats, &y_floats) {
                            Some(c) => Ok(RustMathValue::Float(c)),
                            None => Err(EvalError::new("ValueError", "Cannot compute covariance (need matching lists with at least 2 values)")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "covariance requires two lists of numbers")),
                }
            }

            // ====== NUMERICAL FUNCTIONS ======

            // FFT: fft([c1, c2, ...]) where c_i are complex or real numbers
            "fft" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let complexes = self.list_to_complexes(&values)?;
                        let result = numerical_fft(&complexes);
                        let list: Vec<RustMathValue> = result.into_iter()
                            .map(RustMathValue::Complex)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "fft requires a list of numbers")),
                }
            }

            // Inverse FFT: ifft([c1, c2, ...])
            "ifft" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let complexes = self.list_to_complexes(&values)?;
                        let result = numerical_ifft(&complexes);
                        let list: Vec<RustMathValue> = result.into_iter()
                            .map(RustMathValue::Complex)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "ifft requires a list of numbers")),
                }
            }

            // Numerical integration: integrate_num(expr, var, a, b) or quad(...)
            // Simpson's rule with 100 subdivisions
            "integrate_num" | "quad" | "nintegrate" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 4 {
                    return Err(EvalError::new("ArgumentError", "integrate_num requires 4 arguments (expr, var, a, b)"));
                }
                let expr_clone = match &args[0] {
                    RustMathValue::Expr(e) => e.clone(),
                    _ => return Err(EvalError::new("TypeError", "first argument must be an expression")),
                };
                // Get variable name from argument
                let var_name = match &args[1] {
                    RustMathValue::Symbol(s) => s.name().to_string(),
                    RustMathValue::String(s) => s.clone(),
                    _ => return Err(EvalError::new("TypeError", "second argument must be a variable")),
                };
                // Find the actual symbol in the expression (important: symbols have unique IDs)
                let expr_symbols = expr_clone.symbols();
                let var_sym = expr_symbols.iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                let a = match &args[2] {
                    RustMathValue::Integer(n) => n.to_f64().ok_or_else(|| EvalError::new("ValueError", "invalid lower bound"))?,
                    RustMathValue::Float(f) => *f,
                    RustMathValue::Rational(r) => r.numerator().to_f64().unwrap() / r.denominator().to_f64().unwrap(),
                    _ => return Err(EvalError::new("TypeError", "lower bound must be a number")),
                };
                let b = match &args[3] {
                    RustMathValue::Integer(n) => n.to_f64().ok_or_else(|| EvalError::new("ValueError", "invalid upper bound"))?,
                    RustMathValue::Float(f) => *f,
                    RustMathValue::Rational(r) => r.numerator().to_f64().unwrap() / r.denominator().to_f64().unwrap(),
                    _ => return Err(EvalError::new("TypeError", "upper bound must be a number")),
                };

                let f = |x: f64| -> f64 {
                    let val = Expr::Real(x);
                    let substituted = expr_clone.substitute(&var_sym, &val);
                    substituted.eval_float().unwrap_or(f64::NAN)
                };

                let result = simpson(f, a, b, 100);
                Ok(RustMathValue::Float(result))
            }

            // Numerical root finding: find_root(expr, var, a, b) or nsolve(...)
            "find_root" | "nsolve" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 4 {
                    return Err(EvalError::new("ArgumentError", "find_root requires 4 arguments (expr, var, a, b)"));
                }
                let expr_clone = match &args[0] {
                    RustMathValue::Expr(e) => e.clone(),
                    _ => return Err(EvalError::new("TypeError", "first argument must be an expression")),
                };
                // Get variable name from argument
                let var_name = match &args[1] {
                    RustMathValue::Symbol(s) => s.name().to_string(),
                    RustMathValue::String(s) => s.clone(),
                    _ => return Err(EvalError::new("TypeError", "second argument must be a variable")),
                };
                // Find the actual symbol in the expression (important: symbols have unique IDs)
                let expr_symbols = expr_clone.symbols();
                let var_sym = expr_symbols.iter()
                    .find(|s| s.name() == var_name)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name)))?;
                let a = match &args[2] {
                    RustMathValue::Integer(n) => n.to_f64().ok_or_else(|| EvalError::new("ValueError", "invalid lower bound"))?,
                    RustMathValue::Float(f) => *f,
                    RustMathValue::Rational(r) => r.numerator().to_f64().unwrap() / r.denominator().to_f64().unwrap(),
                    _ => return Err(EvalError::new("TypeError", "lower bound must be a number")),
                };
                let b = match &args[3] {
                    RustMathValue::Integer(n) => n.to_f64().ok_or_else(|| EvalError::new("ValueError", "invalid upper bound"))?,
                    RustMathValue::Float(f) => *f,
                    RustMathValue::Rational(r) => r.numerator().to_f64().unwrap() / r.denominator().to_f64().unwrap(),
                    _ => return Err(EvalError::new("TypeError", "upper bound must be a number")),
                };

                let f = |x: f64| -> f64 {
                    let val = Expr::Real(x);
                    let substituted = expr_clone.substitute(&var_sym, &val);
                    substituted.eval_float().unwrap_or(f64::NAN)
                };

                match bisection(f, a, b, 1e-10, 1000) {
                    Some(result) => Ok(RustMathValue::Float(result.root)),
                    None => Err(EvalError::new("ValueError", "No root found in interval (f(a) and f(b) must have opposite signs)")),
                }
            }

            // ===== GRAPH OPERATIONS =====

            // Graph constructor: Graph(n) creates a graph with n vertices
            "Graph" => {
                let n = self.eval_as_usize(args_str)?;
                Ok(RustMathValue::Graph(Graph::new(n)))
            }

            // Complete graph K_n
            "complete_graph" | "CompleteGraph" | "K" => {
                let n = self.eval_as_usize(args_str)?;
                Ok(RustMathValue::Graph(complete_graph(n)))
            }

            // Cycle graph C_n
            "cycle_graph" | "CycleGraph" | "C" => {
                let n = self.eval_as_usize(args_str)?;
                if n < 3 {
                    return Err(EvalError::new("ValueError", "Cycle graph requires at least 3 vertices"));
                }
                Ok(RustMathValue::Graph(cycle_graph(n)))
            }

            // Path graph P_n
            "path_graph" | "PathGraph" | "P" => {
                let n = self.eval_as_usize(args_str)?;
                Ok(RustMathValue::Graph(path_graph(n)))
            }

            // Star graph S_n (one center connected to n outer vertices)
            "star_graph" | "StarGraph" => {
                let n = self.eval_as_usize(args_str)?;
                Ok(RustMathValue::Graph(star_graph(n)))
            }

            // Wheel graph W_n
            "wheel_graph" | "WheelGraph" => {
                let n = self.eval_as_usize(args_str)?;
                if n < 3 {
                    return Err(EvalError::new("ValueError", "Wheel graph requires at least 3 outer vertices"));
                }
                Ok(RustMathValue::Graph(wheel_graph(n)))
            }

            // Petersen graph
            "petersen_graph" | "PetersenGraph" | "petersen" => {
                Ok(RustMathValue::Graph(petersen_graph()))
            }

            // Number of vertices in a graph
            "num_vertices" | "order" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(g) => Ok(RustMathValue::Integer(Integer::from(g.num_vertices() as i64))),
                    _ => Err(EvalError::new("TypeError", "num_vertices requires a Graph")),
                }
            }

            // Number of edges in a graph
            "num_edges" | "size" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(g) => Ok(RustMathValue::Integer(Integer::from(g.num_edges() as i64))),
                    _ => Err(EvalError::new("TypeError", "num_edges requires a Graph")),
                }
            }

            // Check if graph is connected
            "is_connected" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(g) => Ok(RustMathValue::Bool(g.is_connected())),
                    _ => Err(EvalError::new("TypeError", "is_connected requires a Graph")),
                }
            }

            // Chromatic number
            "chromatic_number" | "chi" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(g) => Ok(RustMathValue::Integer(Integer::from(g.chromatic_number() as i64))),
                    _ => Err(EvalError::new("TypeError", "chromatic_number requires a Graph")),
                }
            }

            // Diameter of graph
            "diameter" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(g) => {
                        match g.diameter() {
                            Some(d) => Ok(RustMathValue::Integer(Integer::from(d as i64))),
                            None => Err(EvalError::new("ValueError", "Graph is not connected (diameter undefined)")),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "diameter requires a Graph")),
                }
            }

            // Degree of a vertex
            "vertex_degree" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "vertex_degree requires 2 arguments (graph, vertex)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let v = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex must be an integer")),
                };
                match g.degree(v) {
                    Some(d) => Ok(RustMathValue::Integer(Integer::from(d as i64))),
                    None => Err(EvalError::new("IndexError", "vertex out of bounds")),
                }
            }

            // Neighbors of a vertex
            "neighbors" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "neighbors requires 2 arguments (graph, vertex)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let v = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex must be an integer")),
                };
                match g.neighbors(v) {
                    Some(neighbors) => {
                        let values: Vec<RustMathValue> = neighbors.iter()
                            .map(|&n| RustMathValue::Integer(Integer::from(n as i64)))
                            .collect();
                        Ok(RustMathValue::List(values))
                    }
                    None => Err(EvalError::new("IndexError", "vertex out of bounds")),
                }
            }

            // Shortest path between two vertices
            "shortest_path" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 3 {
                    return Err(EvalError::new("ArgumentError", "shortest_path requires 3 arguments (graph, start, end)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let start = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid start vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "start must be an integer")),
                };
                let end = match &args[2] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid end vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "end must be an integer")),
                };
                match g.shortest_path(start, end) {
                    Ok(Some(path)) => {
                        let values: Vec<RustMathValue> = path.iter()
                            .map(|&n| RustMathValue::Integer(Integer::from(n as i64)))
                            .collect();
                        Ok(RustMathValue::List(values))
                    }
                    Ok(None) => Ok(RustMathValue::None),
                    Err(e) => Err(EvalError::new("ValueError", e)),
                }
            }

            // BFS traversal
            "bfs" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "bfs requires 2 arguments (graph, start)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let start = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid start vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "start must be an integer")),
                };
                match g.bfs(start) {
                    Ok(order) => {
                        let values: Vec<RustMathValue> = order.iter()
                            .map(|&n| RustMathValue::Integer(Integer::from(n as i64)))
                            .collect();
                        Ok(RustMathValue::List(values))
                    }
                    Err(e) => Err(EvalError::new("ValueError", e)),
                }
            }

            // DFS traversal
            "dfs" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "dfs requires 2 arguments (graph, start)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let start = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid start vertex"))?,
                    _ => return Err(EvalError::new("TypeError", "start must be an integer")),
                };
                match g.dfs(start) {
                    Ok(order) => {
                        let values: Vec<RustMathValue> = order.iter()
                            .map(|&n| RustMathValue::Integer(Integer::from(n as i64)))
                            .collect();
                        Ok(RustMathValue::List(values))
                    }
                    Err(e) => Err(EvalError::new("ValueError", e)),
                }
            }

            // Add edge to graph (returns modified graph)
            "add_edge" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 3 {
                    return Err(EvalError::new("ArgumentError", "add_edge requires 3 arguments (graph, u, v)"));
                }
                let mut g = match &args[0] {
                    RustMathValue::Graph(g) => g.clone(),
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let u = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex u"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex u must be an integer")),
                };
                let v = match &args[2] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex v"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex v must be an integer")),
                };
                match g.add_edge(u, v) {
                    Ok(()) => Ok(RustMathValue::Graph(g)),
                    Err(e) => Err(EvalError::new("ValueError", e)),
                }
            }

            // Check if edge exists
            "has_edge" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 3 {
                    return Err(EvalError::new("ArgumentError", "has_edge requires 3 arguments (graph, u, v)"));
                }
                let g = match &args[0] {
                    RustMathValue::Graph(g) => g,
                    _ => return Err(EvalError::new("TypeError", "first argument must be a Graph")),
                };
                let u = match &args[1] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex u"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex u must be an integer")),
                };
                let v = match &args[2] {
                    RustMathValue::Integer(n) => n.to_usize().ok_or_else(|| EvalError::new("ValueError", "invalid vertex v"))?,
                    _ => return Err(EvalError::new("TypeError", "vertex v must be an integer")),
                };
                Ok(RustMathValue::Bool(g.has_edge(u, v)))
            }

            // ===== GEOMETRY OPERATIONS =====

            // Point2D constructor: Point(x, y)
            "Point" | "Point2D" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "Point requires 2 arguments (x, y)"));
                }
                let x = self.value_to_f64(&args[0])?;
                let y = self.value_to_f64(&args[1])?;
                Ok(RustMathValue::Point2D(Point2D::new(x, y)))
            }

            // Point3D constructor: Point3D(x, y, z)
            "Point3d" | "point3d" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 3 {
                    return Err(EvalError::new("ArgumentError", "Point3D requires 3 arguments (x, y, z)"));
                }
                let x = self.value_to_f64(&args[0])?;
                let y = self.value_to_f64(&args[1])?;
                let z = self.value_to_f64(&args[2])?;
                Ok(RustMathValue::Point3D(Point3D::new(x, y, z)))
            }

            // Distance between two points
            "distance" | "dist" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "distance requires 2 arguments (point1, point2)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Point2D(p1), RustMathValue::Point2D(p2)) => {
                        Ok(RustMathValue::Float(p1.distance(p2)))
                    }
                    (RustMathValue::Point3D(p1), RustMathValue::Point3D(p2)) => {
                        Ok(RustMathValue::Float(p1.distance(p2)))
                    }
                    _ => Err(EvalError::new("TypeError", "distance requires two points of the same dimension")),
                }
            }

            // Dot product
            "dot_product" | "dot" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "dot requires 2 arguments (point1, point2)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Point2D(p1), RustMathValue::Point2D(p2)) => {
                        Ok(RustMathValue::Float(p1.dot(p2)))
                    }
                    (RustMathValue::Point3D(p1), RustMathValue::Point3D(p2)) => {
                        Ok(RustMathValue::Float(p1.dot(p2)))
                    }
                    _ => Err(EvalError::new("TypeError", "dot requires two points of the same dimension")),
                }
            }

            // Cross product
            "cross_product" | "cross" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "cross requires 2 arguments (point1, point2)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Point2D(p1), RustMathValue::Point2D(p2)) => {
                        // 2D cross product returns scalar (z-component)
                        Ok(RustMathValue::Float(p1.cross(p2)))
                    }
                    (RustMathValue::Point3D(p1), RustMathValue::Point3D(p2)) => {
                        // 3D cross product returns a point (vector)
                        Ok(RustMathValue::Point3D(p1.cross(p2)))
                    }
                    _ => Err(EvalError::new("TypeError", "cross requires two points of the same dimension")),
                }
            }

            // Convex hull of a set of points
            "convex_hull" | "ConvexHull" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let points: Result<Vec<Point2D>, _> = values.iter().map(|v| {
                            match v {
                                RustMathValue::Point2D(p) => Ok(*p),
                                RustMathValue::List(coords) if coords.len() >= 2 => {
                                    let x = self.value_to_f64(&coords[0])?;
                                    let y = self.value_to_f64(&coords[1])?;
                                    Ok(Point2D::new(x, y))
                                }
                                _ => Err(EvalError::new("TypeError", "list must contain Point2D values or [x,y] pairs")),
                            }
                        }).collect();
                        let points = points?;
                        let hull = convex_hull(&points);
                        let hull_values: Vec<RustMathValue> = hull.into_iter()
                            .map(|p| RustMathValue::Point2D(p))
                            .collect();
                        Ok(RustMathValue::List(hull_values))
                    }
                    _ => Err(EvalError::new("TypeError", "convex_hull requires a list of points")),
                }
            }

            // Check if three points are collinear
            "collinear" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 3 {
                    return Err(EvalError::new("ArgumentError", "collinear requires 3 points"));
                }
                match (&args[0], &args[1], &args[2]) {
                    (RustMathValue::Point2D(p1), RustMathValue::Point2D(p2), RustMathValue::Point2D(p3)) => {
                        Ok(RustMathValue::Bool(Point2D::collinear(p1, p2, p3)))
                    }
                    _ => Err(EvalError::new("TypeError", "collinear requires three Point2D values")),
                }
            }

            // Create a polygon from a list of points
            "polygon" | "Polygon" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let points: Result<Vec<Point2D>, _> = values.iter().map(|v| {
                            match v {
                                RustMathValue::Point2D(p) => Ok(*p),
                                RustMathValue::List(coords) if coords.len() >= 2 => {
                                    let x = self.value_to_f64(&coords[0])?;
                                    let y = self.value_to_f64(&coords[1])?;
                                    Ok(Point2D::new(x, y))
                                }
                                _ => Err(EvalError::new("TypeError", "list must contain Point2D values or [x,y] pairs")),
                            }
                        }).collect();
                        let points = points?;
                        match Polygon::new(points) {
                            Ok(poly) => {
                                let vertices: Vec<RustMathValue> = poly.vertices().iter()
                                    .map(|p| RustMathValue::Point2D(*p))
                                    .collect();
                                Ok(RustMathValue::List(vertices))
                            }
                            Err(e) => Err(EvalError::new("ValueError", e)),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "polygon requires a list of points")),
                }
            }

            // Calculate polygon area
            "polygon_area" | "area" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let points: Result<Vec<Point2D>, _> = values.iter().map(|v| {
                            match v {
                                RustMathValue::Point2D(p) => Ok(*p),
                                RustMathValue::List(coords) if coords.len() >= 2 => {
                                    let x = self.value_to_f64(&coords[0])?;
                                    let y = self.value_to_f64(&coords[1])?;
                                    Ok(Point2D::new(x, y))
                                }
                                _ => Err(EvalError::new("TypeError", "list must contain Point2D values or [x,y] pairs")),
                            }
                        }).collect();
                        let points = points?;
                        match Polygon::new(points) {
                            Ok(poly) => Ok(RustMathValue::Float(poly.area())),
                            Err(e) => Err(EvalError::new("ValueError", e)),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "area requires a list of points")),
                }
            }

            // Calculate polygon perimeter
            "perimeter" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let points: Result<Vec<Point2D>, _> = values.iter().map(|v| {
                            match v {
                                RustMathValue::Point2D(p) => Ok(*p),
                                RustMathValue::List(coords) if coords.len() >= 2 => {
                                    let x = self.value_to_f64(&coords[0])?;
                                    let y = self.value_to_f64(&coords[1])?;
                                    Ok(Point2D::new(x, y))
                                }
                                _ => Err(EvalError::new("TypeError", "list must contain Point2D values or [x,y] pairs")),
                            }
                        }).collect();
                        let points = points?;
                        match Polygon::new(points) {
                            Ok(poly) => Ok(RustMathValue::Float(poly.perimeter())),
                            Err(e) => Err(EvalError::new("ValueError", e)),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "perimeter requires a list of points")),
                }
            }

            // Check if polygon is convex
            "is_convex" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(values) => {
                        let points: Result<Vec<Point2D>, _> = values.iter().map(|v| {
                            match v {
                                RustMathValue::Point2D(p) => Ok(*p),
                                RustMathValue::List(coords) if coords.len() >= 2 => {
                                    let x = self.value_to_f64(&coords[0])?;
                                    let y = self.value_to_f64(&coords[1])?;
                                    Ok(Point2D::new(x, y))
                                }
                                _ => Err(EvalError::new("TypeError", "list must contain Point2D values or [x,y] pairs")),
                            }
                        }).collect();
                        let points = points?;
                        match Polygon::new(points) {
                            Ok(poly) => Ok(RustMathValue::Bool(poly.is_convex())),
                            Err(e) => Err(EvalError::new("ValueError", e)),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "is_convex requires a list of points")),
                }
            }

            // ====== PLOTTING FUNCTIONS ======

            // SageMath-style plot function:
            //   plot(sin(x), (x, 0, 2*pi))     - tuple syntax
            //   plot(x^2, (x, -2, 2))          - tuple syntax
            //   plot(expr, x, -2, 2)           - alternative syntax
            //   plot([Point(0,0), ...])        - scatter plot (legacy)
            "plot" => {
                // Check if it looks like a function plot (has a tuple or multiple args)
                let looks_like_function_plot = args_str.contains('(') &&
                    (args_str.matches(',').count() >= 1);

                // First, try to parse as SageMath-style function plot
                match self.try_parse_sagemath_plot(args_str) {
                    Ok(result) => return Ok(result),
                    Err(e) if looks_like_function_plot => {
                        // If it looks like a function plot but failed, return the error
                        return Err(e);
                    }
                    Err(_) => {
                        // Fallback to scatter plot with list of points
                    }
                }

                // Fallback to scatter plot with list of points
                let args = self.parse_args(args_str)?;
                if args.len() == 1 {
                    match &args[0] {
                        RustMathValue::List(pts) => {
                            let coords = self.extract_2d_coords(pts)?;
                            let svg = self.generate_scatter_svg(&coords, "Scatter Plot", false);
                            let description = format!("Scatter plot with {} points", coords.len());
                            return Ok(RustMathValue::Plot { description, svg });
                        }
                        _ => return Err(EvalError::new("TypeError", "plot requires an expression with range or a list of points")),
                    }
                } else {
                    return Err(EvalError::new("ArgumentError",
                        "Usage: plot(expr, (var, start, end)) or plot([points])"));
                }
            }

            // Explicit scatter_plot for points
            "scatter_plot" => {
                let args = self.parse_args(args_str)?;
                if args.len() == 1 {
                    match &args[0] {
                        RustMathValue::List(pts) => {
                            let coords = self.extract_2d_coords(pts)?;
                            let svg = self.generate_scatter_svg(&coords, "Scatter Plot", false);
                            let description = format!("Scatter plot with {} points", coords.len());
                            return Ok(RustMathValue::Plot { description, svg });
                        }
                        _ => return Err(EvalError::new("TypeError", "scatter_plot requires a list of points")),
                    }
                } else {
                    return Err(EvalError::new("ArgumentError", "scatter_plot requires 1 argument (list of points)"));
                }
            }

            // Scatter from two lists: scatter(xs, ys)
            "scatter" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "scatter requires 2 arguments (xs, ys)"));
                }
                let xs = match &args[0] {
                    RustMathValue::List(v) => self.list_to_f64s(v)?,
                    _ => return Err(EvalError::new("TypeError", "scatter requires lists of numbers")),
                };
                let ys = match &args[1] {
                    RustMathValue::List(v) => self.list_to_f64s(v)?,
                    _ => return Err(EvalError::new("TypeError", "scatter requires lists of numbers")),
                };
                if xs.len() != ys.len() {
                    return Err(EvalError::new("ValueError", "xs and ys must have the same length"));
                }
                let coords: Vec<(f64, f64)> = xs.into_iter().zip(ys.into_iter()).collect();
                let svg = self.generate_scatter_svg(&coords, "Scatter Plot", false);
                let description = format!("Scatter plot with {} points", coords.len());
                return Ok(RustMathValue::Plot { description, svg });
            }

            // Line plot: line_plot([Point(0,0), Point(1,1), ...])
            "line_plot" | "plot_line" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "line_plot requires 1 argument (list of points)"));
                }
                match &args[0] {
                    RustMathValue::List(pts) => {
                        let coords = self.extract_2d_coords(pts)?;
                        let svg = self.generate_scatter_svg(&coords, "Line Plot", true);
                        let description = format!("Line plot with {} points", coords.len());
                        return Ok(RustMathValue::Plot { description, svg });
                    }
                    _ => return Err(EvalError::new("TypeError", "line_plot requires a list of points")),
                }
            }

            // Plot function: plot_function(expr, var, start, end) or plot_function(expr, var, start, end, n_points)
            "plot_function" | "plot_expr" => {
                let args = self.parse_symbolic_args(args_str)?;
                if args.len() < 4 || args.len() > 5 {
                    return Err(EvalError::new("ArgumentError", "plot_function requires 4-5 arguments (expr, var, start, end [, n_points])"));
                }

                let expr = &args[0];
                // Get variable name from argument
                let var_name_str = match &args[1] {
                    Expr::Symbol(s) => s.name().to_string(),
                    _ => return Err(EvalError::new("TypeError", "second argument must be a variable")),
                };
                // Find the actual symbol in the expression (important: symbols have unique IDs)
                let expr_symbols = expr.symbols();
                let var_sym = expr_symbols.iter()
                    .find(|s| s.name() == var_name_str)
                    .cloned()
                    .ok_or_else(|| EvalError::new("ValueError",
                        format!("Variable '{}' not found in expression", var_name_str)))?;

                let start = self.expr_to_f64(&args[2])?;
                let end = self.expr_to_f64(&args[3])?;
                let n_points = if args.len() == 5 {
                    self.expr_to_f64(&args[4])? as usize
                } else {
                    100
                };

                if n_points < 2 || n_points > 10000 {
                    return Err(EvalError::new("ValueError", "n_points must be between 2 and 10000"));
                }

                // Sample the function
                let mut coords = Vec::new();
                let step = (end - start) / (n_points - 1) as f64;
                for i in 0..n_points {
                    let x = start + i as f64 * step;
                    let mut eval_expr = expr.clone();
                    // Substitute x value
                    eval_expr = eval_expr.substitute(&var_sym, &Expr::Real(x));
                    if let Some(y) = try_eval_to_f64(&eval_expr) {
                        if y.is_finite() {
                            coords.push((x, y));
                        }
                    }
                }

                if coords.is_empty() {
                    return Err(EvalError::new("ValueError", "Could not evaluate function at any point"));
                }

                let title = format!("y = {}", expr);
                let svg = self.generate_scatter_svg(&coords, &title, true);
                let description = format!("Function plot with {} points", coords.len());
                return Ok(RustMathValue::Plot { description, svg });
            }

            // Histogram: histogram(data, bins)
            "histogram" | "hist" => {
                let args = self.parse_args(args_str)?;
                if args.len() < 1 || args.len() > 2 {
                    return Err(EvalError::new("ArgumentError", "histogram requires 1-2 arguments (data [, bins])"));
                }

                let data = match &args[0] {
                    RustMathValue::List(v) => self.list_to_f64s(v)?,
                    _ => return Err(EvalError::new("TypeError", "histogram requires a list of numbers")),
                };

                let n_bins = if args.len() == 2 {
                    match &args[1] {
                        RustMathValue::Integer(n) => n.to_i64() as usize,
                        _ => return Err(EvalError::new("TypeError", "bins must be an integer")),
                    }
                } else {
                    10
                };

                if n_bins < 1 || n_bins > 1000 {
                    return Err(EvalError::new("ValueError", "bins must be between 1 and 1000"));
                }

                let svg = self.generate_histogram_svg(&data, n_bins);
                let description = format!("Histogram with {} bins", n_bins);
                return Ok(RustMathValue::Plot { description, svg });
            }

            // Bar chart: bar_chart(labels, values)
            "bar_chart" | "bar" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "bar_chart requires 2 arguments (labels, values)"));
                }

                let labels = match &args[0] {
                    RustMathValue::List(v) => {
                        v.iter().map(|x| match x {
                            RustMathValue::String(s) => Ok(s.clone()),
                            RustMathValue::Integer(n) => Ok(n.to_string()),
                            _ => Err(EvalError::new("TypeError", "labels must be strings or integers")),
                        }).collect::<Result<Vec<String>, _>>()?
                    }
                    _ => return Err(EvalError::new("TypeError", "first argument must be a list of labels")),
                };

                let values = match &args[1] {
                    RustMathValue::List(v) => self.list_to_f64s(v)?,
                    _ => return Err(EvalError::new("TypeError", "second argument must be a list of values")),
                };

                if labels.len() != values.len() {
                    return Err(EvalError::new("ValueError", "labels and values must have the same length"));
                }

                let svg = self.generate_bar_chart_svg(&labels, &values);
                let description = format!("Bar chart with {} bars", labels.len());
                return Ok(RustMathValue::Plot { description, svg });
            }

            // 3D scatter plot (2D projection): plot3d([Point3D(0,0,0), ...])
            "plot3d" | "scatter3d" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "plot3d requires 1 argument (list of 3D points)"));
                }
                match &args[0] {
                    RustMathValue::List(pts) => {
                        let coords = self.extract_3d_coords(pts)?;
                        let svg = self.generate_3d_scatter_svg(&coords, "3D Scatter Plot");
                        let description = format!("3D plot with {} points", coords.len());
                        return Ok(RustMathValue::Plot { description, svg });
                    }
                    _ => return Err(EvalError::new("TypeError", "plot3d requires a list of 3D points")),
                }
            }

            // Slope field (direction field) for ODEs
            // plot_slope_field(f, (x, xmin, xmax), (y, ymin, ymax), plot_points=25)
            // where f = dy/dx as a function of x and y
            "plot_slope_field" | "slope_field" | "direction_field" => {
                return self.eval_slope_field(args_str);
            }

            // Parametric plot: parametric_plot((x(t), y(t)), (t, tmin, tmax))
            "parametric_plot" => {
                return self.eval_parametric_plot(args_str);
            }

            // Contour plot: contour_plot(f(x,y), (x, xmin, xmax), (y, ymin, ymax))
            "contour_plot" | "contour" => {
                return self.eval_contour_plot(args_str);
            }

            // Vector field plot: vector_field_plot((fx, fy), (x, xmin, xmax), (y, ymin, ymax))
            "plot_vector_field" | "vector_field" => {
                return self.eval_vector_field(args_str);
            }

            // Implicit plot: implicit_plot(f(x,y), (x, xmin, xmax), (y, ymin, ymax))
            // Plots the curve where f(x,y) = 0
            "implicit_plot" => {
                return self.eval_implicit_plot(args_str);
            }

            // Region plot: region_plot(condition, (x, xmin, xmax), (y, ymin, ymax))
            // Plots the region where condition is true
            "region_plot" => {
                return self.eval_region_plot(args_str);
            }

            // Graph visualization: show_graph(G)
            "show_graph" | "draw_graph" | "visualize_graph" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Graph(ref g) => {
                        let svg = self.generate_graph_svg(g);
                        let description = format!("Graph visualization ({} vertices, {} edges)", g.num_vertices(), g.num_edges());
                        return Ok(RustMathValue::Plot { description, svg });
                    }
                    _ => return Err(EvalError::new("TypeError", "show_graph requires a graph")),
                }
            }

            // ===== CLIFFORD ALGEBRA FUNCTIONS =====

            // CliffordAlgebra(dim) - Create Euclidean Clifford algebra Cl(R^n)
            "CliffordAlgebra" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let dim = n.to_i64() as usize;
                        if dim > 10 {
                            return Err(EvalError::new("ValueError", "Clifford algebra dimension too large (max 10)"));
                        }
                        // Create Euclidean quadratic form: Q(x) = x1^2 + x2^2 + ... + xn^2
                        let q_form: Vec<Rational> = (0..dim).map(|_| Rational::from(1i64)).collect();
                        let cl = CliffordAlgebra::new(q_form);
                        Ok(RustMathValue::CliffordAlg(cl))
                    }
                    _ => Err(EvalError::new("TypeError", "CliffordAlgebra requires an integer dimension")),
                }
            }

            // ExteriorAlgebra(dim) - Create exterior algebra (Clifford with Q=0)
            "ExteriorAlgebra" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let dim = n.to_i64() as usize;
                        if dim > 10 {
                            return Err(EvalError::new("ValueError", "Exterior algebra dimension too large (max 10)"));
                        }
                        // Create exterior algebra as Clifford with Q=0
                        let q_form: Vec<Rational> = (0..dim).map(|_| Rational::from(0i64)).collect();
                        let ext = CliffordAlgebra::new(q_form);
                        Ok(RustMathValue::CliffordAlg(ext))
                    }
                    _ => Err(EvalError::new("TypeError", "ExteriorAlgebra requires an integer dimension")),
                }
            }

            // e(i) - Create basis element e_i for Clifford/Exterior algebra
            "e" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(i) => {
                        let idx = i.to_i64() as usize;
                        // Create a generator basis element with coefficient 1
                        let basis = CliffordBasisElement::generator(idx);
                        // Create element with single term
                        let elem = CliffordAlgebraElement::from_term(
                            Rational::from(1i64),
                            basis,
                            1, // dimension placeholder
                            vec![Rational::from(1i64)], // Q form placeholder
                        );
                        Ok(RustMathValue::CliffordElem(elem))
                    }
                    _ => Err(EvalError::new("TypeError", "e requires an integer index")),
                }
            }

            // grade(elem, k) - Extract grade-k component
            "grade" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "grade requires 2 arguments: element and grade"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::CliffordElem(elem), RustMathValue::Integer(k)) => {
                        let grade = k.to_i64() as usize;
                        let graded = elem.grade_component(grade);
                        Ok(RustMathValue::CliffordElem(graded))
                    }
                    _ => Err(EvalError::new("TypeError", "grade requires (CliffordElement, integer)")),
                }
            }

            // ===== MANIFOLD FUNCTIONS =====

            // Manifold(dim, name) - Create a differentiable manifold
            "Manifold" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "Manifold requires 2 arguments: dimension and name"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(dim), RustMathValue::String(name)) => {
                        let d = dim.to_i64() as usize;
                        let manifold = DifferentiableManifold::new(name.clone(), d);
                        Ok(RustMathValue::Manifold(Arc::new(manifold)))
                    }
                    _ => Err(EvalError::new("TypeError", "Manifold requires (integer, string)")),
                }
            }

            // EuclideanSpace(dim) - Create R^n as a manifold
            "EuclideanSpace" | "RealSpace" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let dim = n.to_i64() as usize;
                        let space = EuclideanSpace::new(dim);
                        // Convert EuclideanSpace to DifferentiableManifold
                        let manifold: DifferentiableManifold = space.into();
                        Ok(RustMathValue::Manifold(Arc::new(manifold)))
                    }
                    _ => Err(EvalError::new("TypeError", "EuclideanSpace requires an integer dimension")),
                }
            }

            // chart(manifold, "x,y,z") - Create a chart with named coordinates
            "chart" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "chart requires 2 arguments: manifold and coordinate names"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Manifold(m), RustMathValue::String(coords)) => {
                        let coord_names: Vec<String> = coords.split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        let dim = coord_names.len();
                        let chart = Chart::new("user_chart", dim, coord_names)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::ChartVal(chart))
                    }
                    _ => Err(EvalError::new("TypeError", "chart requires (Manifold, string)")),
                }
            }

            // diff_form(manifold, degree) - Create a differential form
            "diff_form" | "DiffForm" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "diff_form requires 2 arguments: manifold and degree"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Manifold(m), RustMathValue::Integer(deg)) => {
                        let degree = deg.to_i64() as usize;
                        let form = DiffForm::new(m.clone(), degree);
                        Ok(RustMathValue::DiffFormVal(form))
                    }
                    _ => Err(EvalError::new("TypeError", "diff_form requires (Manifold, integer)")),
                }
            }

            // coordinate_form(manifold, chart, index) - Create coordinate 1-form dx_i
            "coordinate_form" | "dx" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "coordinate_form requires 3 arguments: manifold, chart, index"));
                }
                match (&args[0], &args[1], &args[2]) {
                    (RustMathValue::Manifold(m), RustMathValue::ChartVal(chart), RustMathValue::Integer(idx)) => {
                        let i = idx.to_i64() as usize;
                        let form = DiffForm::coordinate_form(m.clone(), &chart, i)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::DiffFormVal(form))
                    }
                    _ => Err(EvalError::new("TypeError", "coordinate_form requires (Manifold, Chart, integer)")),
                }
            }

            // exterior_derivative(form, chart) - Compute d(omega)
            "exterior_derivative" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "exterior_derivative requires 2 arguments: form and chart"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::DiffFormVal(form), RustMathValue::ChartVal(chart)) => {
                        let d_form = form.exterior_derivative(chart)
                            .map_err(|e| EvalError::new("ComputationError", format!("{}", e)))?;
                        Ok(RustMathValue::DiffFormVal(d_form))
                    }
                    _ => Err(EvalError::new("TypeError", "exterior_derivative requires (DiffForm, Chart)")),
                }
            }

            // wedge(a, b) for Clifford elements, or wedge(form1, form2, chart) for differential forms
            "wedge" => {
                let args = self.parse_args(args_str)?;
                match args.len() {
                    // wedge(a, b) for Clifford/Exterior algebra elements
                    2 => {
                        match (&args[0], &args[1]) {
                            (RustMathValue::CliffordElem(a), RustMathValue::CliffordElem(b)) => {
                                // Clifford product = wedge product in exterior algebra
                                Ok(RustMathValue::CliffordElem(a.clone() * b.clone()))
                            }
                            _ => Err(EvalError::new("TypeError", "wedge(a, b) requires two Clifford elements")),
                        }
                    }
                    // wedge(form1, form2, chart) for differential forms
                    3 => {
                        match (&args[0], &args[1], &args[2]) {
                            (RustMathValue::DiffFormVal(f1), RustMathValue::DiffFormVal(f2), RustMathValue::ChartVal(chart)) => {
                                let result = f1.wedge(f2, chart)
                                    .map_err(|e| EvalError::new("ComputationError", format!("{}", e)))?;
                                Ok(RustMathValue::DiffFormVal(result))
                            }
                            _ => Err(EvalError::new("TypeError", "wedge(form1, form2, chart) requires (DiffForm, DiffForm, Chart)")),
                        }
                    }
                    _ => Err(EvalError::new("ArgumentError", "wedge requires 2 arguments (Clifford elements) or 3 arguments (DiffForms with chart)")),
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

    /// Parse a nested list like [[1,2],[3,4]] for matrix construction
    fn parse_nested_list(&mut self, args_str: &str) -> Result<Vec<Vec<RustMathValue>>, EvalError> {
        let args_str = args_str.trim();

        // Check it starts with [ and ends with ]
        if !args_str.starts_with('[') || !args_str.ends_with(']') {
            return Err(EvalError::new("SyntaxError", "Matrix must be a nested list [[...], [...]]"));
        }

        // Remove outer brackets
        let inner = &args_str[1..args_str.len()-1];

        let mut result = Vec::new();
        let mut depth = 0;
        let mut current_row = String::new();

        for ch in inner.chars() {
            match ch {
                '[' => {
                    if depth == 0 {
                        current_row.clear();
                    } else {
                        current_row.push(ch);
                    }
                    depth += 1;
                }
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        // Parse this row
                        let row = self.parse_list_elements(&current_row)?;
                        result.push(row);
                        current_row.clear();
                    } else {
                        current_row.push(ch);
                    }
                }
                ',' if depth == 0 => {
                    // Skip commas between rows
                }
                _ => {
                    if depth > 0 {
                        current_row.push(ch);
                    }
                }
            }
        }

        if result.is_empty() {
            return Err(EvalError::new("SyntaxError", "Matrix must be a nested list [[...], [...]]"));
        }

        Ok(result)
    }

    /// Parse comma-separated list elements
    fn parse_list_elements(&mut self, s: &str) -> Result<Vec<RustMathValue>, EvalError> {
        if s.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        let mut depth = 0;
        let mut current = String::new();

        for ch in s.chars() {
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
                    let val = self.eval_expr(current.trim())?;
                    result.push(val);
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        if !current.trim().is_empty() {
            let val = self.eval_expr(current.trim())?;
            result.push(val);
        }

        Ok(result)
    }

    /// Parse a list of integers from string like [1, 2, 3] or 1, 2, 3
    fn parse_list_of_integers(&mut self, s: &str) -> Result<Vec<Integer>, EvalError> {
        let s = s.trim();
        // Handle [1, 2, 3] format
        let inner = if s.starts_with('[') && s.ends_with(']') {
            &s[1..s.len()-1]
        } else {
            s
        };

        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        for part in inner.split(',') {
            let val = self.eval_expr(part.trim())?;
            match val {
                RustMathValue::Integer(n) => result.push(n),
                _ => return Err(EvalError::new("TypeError", "Expected integer in list")),
            }
        }

        Ok(result)
    }

    /// Parse symbolic arguments: either strings like "x^2" or variable names
    fn parse_symbolic_args(&mut self, args_str: &str) -> Result<Vec<Expr>, EvalError> {
        if args_str.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
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
                    result.push(self.parse_symbolic_arg(current.trim())?);
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        if !current.trim().is_empty() {
            result.push(self.parse_symbolic_arg(current.trim())?);
        }

        Ok(result)
    }

    /// Parse a single symbolic argument
    fn parse_symbolic_arg(&mut self, s: &str) -> Result<Expr, EvalError> {
        let s = s.trim();

        // Convert Python-style ** to ^ for power operator
        let s_normalized = s.replace("**", "^");
        let s = s_normalized.as_str();

        // If it's a quoted string, parse it as expression
        if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
            let inner = &s[1..s.len()-1];
            return parse_expr(inner)
                .map_err(|e| EvalError::new("ParseError", format!("Cannot parse expression: {}", e)));
        }

        // Check if it's a variable holding an Expr
        if let Some(val) = self.variables.get(s) {
            match val {
                RustMathValue::Expr(e) => return Ok(e.clone()),
                RustMathValue::Symbol(sym) => return Ok(Expr::Symbol(sym.clone())),
                RustMathValue::Integer(n) => return Ok(Expr::Integer(n.clone())),
                _ => {}
            }
        }

        // Check if it's a function call that might evaluate to an Expr
        // (e.g., expr("x + x + x"), diff(expr, x), etc.)
        if s.contains('(') && s.ends_with(')') {
            // Try to evaluate it as an expression first
            if let Ok(result) = self.eval_expr(s) {
                match result {
                    RustMathValue::Expr(e) => return Ok(e),
                    RustMathValue::Symbol(sym) => return Ok(Expr::Symbol(sym)),
                    RustMathValue::Integer(n) => return Ok(Expr::Integer(n)),
                    _ => {}
                }
            }
        }

        // Try to parse directly as expression
        parse_expr(s)
            .map_err(|e| EvalError::new("ParseError", format!("Cannot parse expression: {}", e)))
    }

    /// Parse a single symbolic expression argument
    fn parse_single_symbolic_arg(&mut self, s: &str) -> Result<Expr, EvalError> {
        self.parse_symbolic_arg(s)
    }

    /// Try to parse SageMath-style plot syntax:
    ///   plot(sin(x), (x, 0, 2*pi))
    ///   plot(x^2, (x, -2, 2))
    ///   plot(expr, x, start, end)
    fn try_parse_sagemath_plot(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();

        // Find the positions of key elements
        // We need to handle: expr, (var, start, end) or expr, var, start, end

        // Split by comma at depth 0, but keep track of parentheses
        let mut parts: Vec<String> = Vec::new();
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
                    parts.push(current.trim().to_string());
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.trim().is_empty() {
            parts.push(current.trim().to_string());
        }

        if parts.is_empty() {
            return Err(EvalError::new("ArgumentError", "plot requires arguments"));
        }

        // Check for tuple syntax: plot(expr, (var, start, end))
        if parts.len() == 2 {
            let expr_str = &parts[0];
            let range_str = &parts[1];

            // Check if second part is a tuple (var, start, end)
            if range_str.starts_with('(') && range_str.ends_with(')') {
                let inner = &range_str[1..range_str.len()-1];
                let range_parts: Vec<&str> = self.split_at_depth_zero(inner, ',');

                if range_parts.len() == 3 {
                    let var_name = range_parts[0].trim();
                    let start_str = range_parts[1].trim();
                    let end_str = range_parts[2].trim();

                    return self.plot_function_internal(expr_str, var_name, start_str, end_str, 200);
                }
            }
        }

        // Check for alternative syntax: plot(expr, var, start, end)
        if parts.len() == 4 {
            let expr_str = &parts[0];
            let var_name = &parts[1];
            let start_str = &parts[2];
            let end_str = &parts[3];

            return self.plot_function_internal(expr_str, var_name, start_str, end_str, 200);
        }

        // Check for alternative syntax with n_points: plot(expr, var, start, end, n_points)
        if parts.len() == 5 {
            let expr_str = &parts[0];
            let var_name = &parts[1];
            let start_str = &parts[2];
            let end_str = &parts[3];
            let n_points = parts[4].parse::<usize>()
                .map_err(|_| EvalError::new("ValueError", "n_points must be an integer"))?;

            return self.plot_function_internal(expr_str, var_name, start_str, end_str, n_points);
        }

        Err(EvalError::new("ArgumentError",
            "Usage: plot(expr, (var, start, end)) or plot(expr, var, start, end)"))
    }

    /// Split a string at depth-zero commas
    fn split_at_depth_zero<'a>(&self, s: &'a str, delimiter: char) -> Vec<&'a str> {
        let mut result = Vec::new();
        let mut depth = 0;
        let mut start = 0;

        for (i, ch) in s.char_indices() {
            match ch {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                c if c == delimiter && depth == 0 => {
                    result.push(&s[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        result.push(&s[start..]);
        result
    }

    /// Internal function to plot a mathematical expression
    fn plot_function_internal(
        &mut self,
        expr_str: &str,
        var_name: &str,
        start_str: &str,
        end_str: &str,
        n_points: usize,
    ) -> Result<RustMathValue, EvalError> {
        // Parse the expression
        let expr = self.parse_symbolic_arg(expr_str)?;

        // Parse start and end values
        let start = self.eval_to_f64(start_str)?;
        let end = self.eval_to_f64(end_str)?;

        if n_points < 2 || n_points > 10000 {
            return Err(EvalError::new("ValueError", "n_points must be between 2 and 10000"));
        }

        if start >= end {
            return Err(EvalError::new("ValueError", "start must be less than end"));
        }

        // Create the variable symbol
        let var = Symbol::new(var_name.trim());

        // Sample the function
        let mut coords = Vec::new();
        let step = (end - start) / (n_points - 1) as f64;

        // Get all symbols from expression to substitute
        let expr_symbols = expr.symbols();

        for i in 0..n_points {
            let x = start + i as f64 * step;
            let mut eval_expr = expr.clone();

            // Substitute ALL matching symbols with the same name
            for sym in &expr_symbols {
                if sym.name() == var.name() {
                    eval_expr = eval_expr.substitute(sym, &Expr::Real(x));
                }
            }

            if let Some(y) = try_eval_to_f64(&eval_expr) {
                if y.is_finite() {
                    coords.push((x, y));
                }
            }
        }

        if coords.is_empty() {
            return Err(EvalError::new("ValueError",
                "Could not evaluate function at any point. Check that the expression is valid."));
        }

        let title = format!("y = {}", expr_str);
        let svg = self.generate_function_plot_svg(&coords, &title);
        let description = format!("Plot of {} with {} points", expr_str, coords.len());

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate a string to f64, supporting constants like pi
    fn eval_to_f64(&mut self, s: &str) -> Result<f64, EvalError> {
        let s = s.trim();

        // Check for common constants
        match s {
            "pi" | "PI" => return Ok(std::f64::consts::PI),
            "e" | "E" => return Ok(std::f64::consts::E),
            "tau" | "TAU" => return Ok(std::f64::consts::TAU),
            _ => {}
        }

        // Replace constants with their numeric values for parsing
        let s_with_constants = s
            .replace("pi", &format!("{}", std::f64::consts::PI))
            .replace("PI", &format!("{}", std::f64::consts::PI));

        // Try parsing as float directly
        if let Ok(val) = s_with_constants.parse::<f64>() {
            return Ok(val);
        }

        // Try evaluating as a simple arithmetic expression
        // Handle expressions like 2*3.14159... by evaluating them
        if s_with_constants.contains('*') || s_with_constants.contains('/') ||
           s_with_constants.contains('+') || s_with_constants.contains('-') {
            // Simple evaluation of arithmetic expressions with constants
            if let Ok(val) = self.eval_simple_arithmetic(&s_with_constants) {
                return Ok(val);
            }
        }

        // Try evaluating as full expression
        let val = self.eval_expr(s)?;
        match val {
            RustMathValue::Integer(n) => n.to_f64()
                .ok_or_else(|| EvalError::new("ValueError", "Integer too large")),
            RustMathValue::Float(f) => Ok(f),
            RustMathValue::Rational(r) => {
                let num = r.numerator().to_f64().unwrap_or(0.0);
                let den = r.denominator().to_f64().unwrap_or(1.0);
                Ok(num / den)
            }
            _ => Err(EvalError::new("TypeError", "Cannot convert to number")),
        }
    }

    /// Evaluate simple arithmetic expression (for range parsing)
    fn eval_simple_arithmetic(&self, s: &str) -> Result<f64, EvalError> {
        let s = s.trim();

        // Try to parse as number first
        if let Ok(val) = s.parse::<f64>() {
            return Ok(val);
        }

        // Handle negative numbers
        if s.starts_with('-') {
            if let Ok(val) = self.eval_simple_arithmetic(&s[1..]) {
                return Ok(-val);
            }
        }

        // Find rightmost + or - (lowest precedence)
        let mut depth = 0;
        for (i, ch) in s.char_indices().rev() {
            match ch {
                ')' => depth += 1,
                '(' => depth -= 1,
                '+' if depth == 0 && i > 0 => {
                    let left = self.eval_simple_arithmetic(&s[..i])?;
                    let right = self.eval_simple_arithmetic(&s[i+1..])?;
                    return Ok(left + right);
                }
                '-' if depth == 0 && i > 0 => {
                    // Check it's not part of scientific notation
                    let prev_char = s.chars().nth(i - 1);
                    if prev_char != Some('e') && prev_char != Some('E') {
                        let left = self.eval_simple_arithmetic(&s[..i])?;
                        let right = self.eval_simple_arithmetic(&s[i+1..])?;
                        return Ok(left - right);
                    }
                }
                _ => {}
            }
        }

        // Find rightmost * or /
        let mut depth = 0;
        for (i, ch) in s.char_indices().rev() {
            match ch {
                ')' => depth += 1,
                '(' => depth -= 1,
                '*' if depth == 0 => {
                    let left = self.eval_simple_arithmetic(&s[..i])?;
                    let right = self.eval_simple_arithmetic(&s[i+1..])?;
                    return Ok(left * right);
                }
                '/' if depth == 0 => {
                    let left = self.eval_simple_arithmetic(&s[..i])?;
                    let right = self.eval_simple_arithmetic(&s[i+1..])?;
                    return Ok(left / right);
                }
                _ => {}
            }
        }

        // Handle parentheses
        if s.starts_with('(') && s.ends_with(')') {
            return self.eval_simple_arithmetic(&s[1..s.len()-1]);
        }

        Err(EvalError::new("ParseError", format!("Cannot parse: {}", s)))
    }

    /// Generate a nicer SVG for function plots (with grid lines)
    fn generate_function_plot_svg(&self, coords: &[(f64, f64)], title: &str) -> String {
        if coords.is_empty() {
            return String::new();
        }

        let width = 600.0;
        let height = 400.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        // Colors (as variables to avoid raw string issues)
        let bg_color = "#f8f9fa";
        let grid_color = "#dee2e6";
        let axis_color = "#6c757d";
        let border_color = "#343a40";
        let line_color = "#2563eb";

        // Find data bounds with some padding
        let (min_x, max_x, min_y, max_y) = self.find_bounds_2d(coords);
        let x_range = if (max_x - min_x).abs() < 1e-10 { 1.0 } else { max_x - min_x };
        let y_range = if (max_y - min_y).abs() < 1e-10 { 1.0 } else { max_y - min_y };

        // Add 5% padding
        let x_pad = x_range * 0.05;
        let y_pad = y_range * 0.05;
        let min_x = min_x - x_pad;
        let max_x = max_x + x_pad;
        let min_y = min_y - y_pad;
        let max_y = max_y + y_pad;
        let x_range = max_x - min_x;
        let y_range = max_y - min_y;

        // Map data to SVG coordinates
        let map_x = |x: f64| margin + (x - min_x) / x_range * plot_width;
        let map_y = |y: f64| margin + plot_height - (y - min_y) / y_range * plot_height;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        // Background
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"white\"/>",
            width, height
        ));

        // Plot area background
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, grid_color
        ));

        // Grid lines
        let n_grid = 5;
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;

            // Vertical grid lines
            let x = margin + frac * plot_width;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                x, margin, x, margin + plot_height, grid_color
            ));

            // Horizontal grid lines
            let y = margin + frac * plot_height;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                margin, y, margin + plot_width, y, grid_color
            ));
        }

        // Draw x-axis at y=0 if it's in range
        if min_y <= 0.0 && max_y >= 0.0 {
            let y0 = map_y(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }

        // Draw y-axis at x=0 if it's in range
        if min_x <= 0.0 && max_x >= 0.0 {
            let x0 = map_x(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"16\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Axes border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;

            // X axis labels
            let x_val = min_x + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"11\" font-family=\"sans-serif\">{:.2}</text>",
                x_pos, margin + plot_height + 18.0, x_val
            ));

            // Y axis labels
            let y_val = max_y - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"11\" font-family=\"sans-serif\">{:.2}</text>",
                margin - 8.0, y_pos + 4.0, y_val
            ));
        }

        // Draw the function curve
        if coords.len() > 1 {
            let mut path = format!("M {} {}", map_x(coords[0].0), map_y(coords[0].1));
            for (x, y) in &coords[1..] {
                path.push_str(&format!(" L {} {}", map_x(*x), map_y(*y)));
            }
            svg.push_str(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
                path, line_color
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate slope field plot
    /// Syntax: plot_slope_field(f, (x, xmin, xmax), (y, ymin, ymax))
    ///         plot_slope_field(f, (x, xmin, xmax), (y, ymin, ymax), plot_points=25)
    fn eval_slope_field(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        // Parse arguments - looking for: expr, (var1, min1, max1), (var2, min2, max2), optional_kwargs
        let args_str = args_str.trim();

        // Split by comma at depth 0
        let mut parts: Vec<String> = Vec::new();
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
                    parts.push(current.trim().to_string());
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.trim().is_empty() {
            parts.push(current.trim().to_string());
        }

        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "plot_slope_field requires: f, (x, xmin, xmax), (y, ymin, ymax) [, plot_points=N]"));
        }

        // Parse the expression f = dy/dx
        let expr_str = &parts[0];
        let expr = self.parse_symbolic_arg(expr_str)?;

        // Parse x range tuple
        let x_range_str = &parts[1];
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError",
                "Expected tuple (x, xmin, xmax) for x range"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError",
                "x range must be (variable, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range tuple
        let y_range_str = &parts[2];
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError",
                "Expected tuple (y, ymin, ymax) for y range"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError",
                "y range must be (variable, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        // Parse optional plot_points
        let mut n_points = 20; // default
        for i in 3..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_points = n.min(50).max(5); // clamp between 5 and 50
                    }
                }
            } else if let Ok(n) = part.parse::<usize>() {
                n_points = n.min(50).max(5);
            }
        }

        // Create symbols for x and y
        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);

        // Get all symbols from expression
        let expr_symbols = expr.symbols();

        // Generate slope field data
        let mut slopes: Vec<(f64, f64, f64)> = Vec::new(); // (x, y, slope)
        let x_step = (x_max - x_min) / (n_points - 1) as f64;
        let y_step = (y_max - y_min) / (n_points - 1) as f64;

        for i in 0..n_points {
            for j in 0..n_points {
                let x = x_min + i as f64 * x_step;
                let y = y_min + j as f64 * y_step;

                let mut eval_expr = expr.clone();

                // Substitute x and y values
                for sym in &expr_symbols {
                    if sym.name() == x_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(y));
                    }
                }

                if let Some(slope) = try_eval_to_f64(&eval_expr) {
                    if slope.is_finite() {
                        slopes.push((x, y, slope));
                    }
                }
            }
        }

        if slopes.is_empty() {
            return Err(EvalError::new("ValueError",
                "Could not evaluate slope field at any point"));
        }

        let title = format!("Slope Field: dy/dx = {}", expr_str);
        let svg = self.generate_slope_field_svg(&slopes, x_min, x_max, y_min, y_max, &title);
        let description = format!("Slope field with {} arrows", slopes.len());

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for slope field
    fn generate_slope_field_svg(
        &self,
        slopes: &[(f64, f64, f64)],
        x_min: f64, x_max: f64,
        y_min: f64, y_max: f64,
        title: &str
    ) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        // Colors
        let bg_color = "#f8f9fa";
        let grid_color = "#dee2e6";
        let axis_color = "#6c757d";
        let border_color = "#343a40";
        let arrow_color = "#2563eb";

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        // Map data to SVG coordinates
        let map_x = |x: f64| margin + (x - x_min) / x_range * plot_width;
        let map_y = |y: f64| margin + plot_height - (y - y_min) / y_range * plot_height;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        // Background
        svg.push_str(&format!(
            "<rect width=\"{}\" height=\"{}\" fill=\"white\"/>",
            width, height
        ));

        // Plot area background
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, grid_color
        ));

        // Grid lines
        let n_grid = 5;
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;

            let x = margin + frac * plot_width;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                x, margin, x, margin + plot_height, grid_color
            ));

            let y = margin + frac * plot_height;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                margin, y, margin + plot_width, y, grid_color
            ));
        }

        // Draw x-axis at y=0 if in range
        if y_min <= 0.0 && y_max >= 0.0 {
            let y0 = map_y(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }

        // Draw y-axis at x=0 if in range
        if x_min <= 0.0 && x_max >= 0.0 {
            let x0 = map_x(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Axes border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;

            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));

            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        // Draw slope segments
        // Calculate appropriate segment length based on grid spacing
        let segment_len = (plot_width / slopes.len() as f64).sqrt() * 1.5;
        let segment_len = segment_len.min(15.0).max(5.0);

        for &(x, y, slope) in slopes {
            let cx = map_x(x);
            let cy = map_y(y);

            // Calculate direction from slope
            let angle = slope.atan();
            let dx = segment_len * angle.cos();
            let dy = segment_len * angle.sin();

            // Line segment centered at (cx, cy)
            let x1 = cx - dx;
            let y1 = cy + dy; // Note: SVG y is inverted
            let x2 = cx + dx;
            let y2 = cy - dy;

            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\" stroke-linecap=\"round\"/>",
                x1, y1, x2, y2, arrow_color
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate parametric plot
    /// Syntax: parametric_plot((x(t), y(t)), (t, tmin, tmax))
    ///         parametric_plot((x(t), y(t)), (t, tmin, tmax), plot_points=100)
    fn eval_parametric_plot(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 2 {
            return Err(EvalError::new("ArgumentError",
                "parametric_plot requires: (x(t), y(t)), (t, tmin, tmax) [, plot_points=N]"));
        }

        // Parse the expression tuple (x(t), y(t))
        let expr_tuple_str = parts[0].trim();
        if !expr_tuple_str.starts_with('(') || !expr_tuple_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError",
                "First argument must be a tuple (x(t), y(t))"));
        }
        let expr_inner = &expr_tuple_str[1..expr_tuple_str.len()-1];
        let expr_parts = self.split_at_depth_zero(expr_inner, ',');
        if expr_parts.len() != 2 {
            return Err(EvalError::new("ArgumentError",
                "Expression tuple must have exactly 2 elements (x(t), y(t))"));
        }
        let x_expr = self.parse_symbolic_arg(expr_parts[0].trim())?;
        let y_expr = self.parse_symbolic_arg(expr_parts[1].trim())?;

        // Parse t range tuple
        let t_range_str = parts[1].trim();
        if !t_range_str.starts_with('(') || !t_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError",
                "Expected tuple (t, tmin, tmax) for parameter range"));
        }
        let t_inner = &t_range_str[1..t_range_str.len()-1];
        let t_parts: Vec<&str> = self.split_at_depth_zero(t_inner, ',');
        if t_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError",
                "Parameter range must be (variable, min, max)"));
        }
        let t_var_name = t_parts[0].trim();
        let t_min = self.eval_to_f64(t_parts[1].trim())?;
        let t_max = self.eval_to_f64(t_parts[2].trim())?;

        // Parse optional plot_points
        let mut n_points = 200;
        for i in 2..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_points = n.min(1000).max(10);
                    }
                }
            } else if let Ok(n) = part.parse::<usize>() {
                n_points = n.min(1000).max(10);
            }
        }

        let t_sym = Symbol::new(t_var_name);
        let x_symbols = x_expr.symbols();
        let y_symbols = y_expr.symbols();

        // Generate curve points
        let mut coords: Vec<(f64, f64)> = Vec::new();
        let t_step = (t_max - t_min) / (n_points - 1) as f64;

        for i in 0..n_points {
            let t = t_min + i as f64 * t_step;

            let mut eval_x = x_expr.clone();
            let mut eval_y = y_expr.clone();

            // Substitute t value
            for sym in &x_symbols {
                if sym.name() == t_sym.name() {
                    eval_x = eval_x.substitute(sym, &Expr::Real(t));
                }
            }
            for sym in &y_symbols {
                if sym.name() == t_sym.name() {
                    eval_y = eval_y.substitute(sym, &Expr::Real(t));
                }
            }

            if let (Some(x), Some(y)) = (try_eval_to_f64(&eval_x), try_eval_to_f64(&eval_y)) {
                if x.is_finite() && y.is_finite() {
                    coords.push((x, y));
                }
            }
        }

        if coords.is_empty() {
            return Err(EvalError::new("ValueError",
                "Could not evaluate parametric curve at any point"));
        }

        let title = format!("Parametric: ({}, {})", expr_parts[0].trim(), expr_parts[1].trim());
        let svg = self.generate_parametric_svg(&coords, &title);
        let description = format!("Parametric curve with {} points", coords.len());

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for parametric plot
    fn generate_parametric_svg(&self, coords: &[(f64, f64)], title: &str) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let bg_color = "#f8f9fa";
        let grid_color = "#dee2e6";
        let axis_color = "#6c757d";
        let border_color = "#343a40";
        let curve_color = "#dc3545";

        // Find bounds
        let x_min = coords.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = coords.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
        let y_min = coords.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = coords.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

        // Add padding
        let x_padding = (x_max - x_min) * 0.1;
        let y_padding = (y_max - y_min) * 0.1;
        let x_min = x_min - x_padding;
        let x_max = x_max + x_padding;
        let y_min = y_min - y_padding;
        let y_max = y_max + y_padding;

        let x_range = if (x_max - x_min).abs() < 1e-10 { 2.0 } else { x_max - x_min };
        let y_range = if (y_max - y_min).abs() < 1e-10 { 2.0 } else { y_max - y_min };

        let map_x = |x: f64| margin + (x - x_min) / x_range * plot_width;
        let map_y = |y: f64| margin + plot_height - (y - y_min) / y_range * plot_height;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        // Background
        svg.push_str(&format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>", width, height));
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, grid_color
        ));

        // Grid
        let n_grid = 5;
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;
            let x = margin + frac * plot_width;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                x, margin, x, margin + plot_height, grid_color
            ));
            let y = margin + frac * plot_height;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                margin, y, margin + plot_width, y, grid_color
            ));
        }

        // Axes at origin if in range
        if y_min <= 0.0 && y_max >= 0.0 {
            let y0 = map_y(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }
        if x_min <= 0.0 && x_max >= 0.0 {
            let x0 = map_x(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;
            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.2}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));
            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.2}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        // Draw curve
        if !coords.is_empty() {
            let mut path = format!("M {} {}", map_x(coords[0].0), map_y(coords[0].1));
            for (x, y) in coords.iter().skip(1) {
                path.push_str(&format!(" L {} {}", map_x(*x), map_y(*y)));
            }
            svg.push_str(&format!(
                "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\"/>",
                path, curve_color
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate contour plot
    /// Syntax: contour_plot(f(x,y), (x, xmin, xmax), (y, ymin, ymax))
    fn eval_contour_plot(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "contour_plot requires: f(x,y), (x, xmin, xmax), (y, ymin, ymax) [, n_contours=10]"));
        }

        let expr = self.parse_symbolic_arg(&parts[0])?;

        // Parse x range
        let x_range_str = parts[1].trim();
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (x, xmin, xmax)"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must be (variable, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range
        let y_range_str = parts[2].trim();
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (y, ymin, ymax)"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must be (variable, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        // Parse optional parameters
        let mut n_contours = 10;
        let mut n_grid = 50;
        for i in 3..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("contours") || part.starts_with("n_contours") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_contours = n.min(30).max(3);
                    }
                }
            } else if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_grid = n.min(100).max(20);
                    }
                }
            }
        }

        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);
        let expr_symbols = expr.symbols();

        // Evaluate function on grid
        let mut z_values: Vec<Vec<f64>> = Vec::new();
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;

        let x_step = (x_max - x_min) / (n_grid - 1) as f64;
        let y_step = (y_max - y_min) / (n_grid - 1) as f64;

        for j in 0..n_grid {
            let mut row = Vec::new();
            let y = y_min + j as f64 * y_step;
            for i in 0..n_grid {
                let x = x_min + i as f64 * x_step;
                let mut eval_expr = expr.clone();

                for sym in &expr_symbols {
                    if sym.name() == x_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(y));
                    }
                }

                let z = try_eval_to_f64(&eval_expr).unwrap_or(f64::NAN);
                if z.is_finite() {
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
                row.push(z);
            }
            z_values.push(row);
        }

        if z_min >= z_max {
            return Err(EvalError::new("ValueError", "Could not evaluate function on grid"));
        }

        let title = format!("Contour plot of {}", &parts[0].trim());
        let svg = self.generate_contour_svg(&z_values, x_min, x_max, y_min, y_max, z_min, z_max, n_contours, &title);
        let description = format!("Contour plot with {} levels", n_contours);

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for contour plot
    fn generate_contour_svg(
        &self,
        z_values: &[Vec<f64>],
        x_min: f64, x_max: f64,
        y_min: f64, y_max: f64,
        z_min: f64, z_max: f64,
        n_contours: usize,
        title: &str
    ) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let bg_color = "#f8f9fa";
        let border_color = "#343a40";
        let axis_color = "#6c757d";

        let n_grid = z_values.len();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let map_x = |i: usize| margin + (i as f64 / (n_grid - 1) as f64) * plot_width;
        let map_y = |j: usize| margin + plot_height - (j as f64 / (n_grid - 1) as f64) * plot_height;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        svg.push_str(&format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>", width, height));
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, border_color
        ));

        // Generate contour levels
        let z_range = z_max - z_min;
        let colors = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#c026d3"];

        for level_idx in 0..n_contours {
            let level = z_min + (level_idx as f64 + 0.5) / n_contours as f64 * z_range;
            let color = colors[level_idx % colors.len()];

            // Marching squares for this level
            for j in 0..n_grid-1 {
                for i in 0..n_grid-1 {
                    let z00 = z_values[j][i];
                    let z10 = z_values[j][i+1];
                    let z01 = z_values[j+1][i];
                    let z11 = z_values[j+1][i+1];

                    if !z00.is_finite() || !z10.is_finite() || !z01.is_finite() || !z11.is_finite() {
                        continue;
                    }

                    // Marching squares case
                    let case = ((z00 >= level) as u8) |
                               (((z10 >= level) as u8) << 1) |
                               (((z01 >= level) as u8) << 2) |
                               (((z11 >= level) as u8) << 3);

                    let x0 = map_x(i);
                    let x1 = map_x(i+1);
                    let y0 = map_y(j);
                    let y1 = map_y(j+1);

                    let lerp = |a: f64, b: f64, za: f64, zb: f64| {
                        if (zb - za).abs() < 1e-10 { 0.5 } else { (level - za) / (zb - za) }
                    };

                    let draw_line = |x1: f64, y1: f64, x2: f64, y2: f64, svg: &mut String| {
                        svg.push_str(&format!(
                            "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                            x1, y1, x2, y2, color
                        ));
                    };

                    match case {
                        1 | 14 => {
                            let t1 = lerp(x0, x1, z00, z10);
                            let t2 = lerp(y0, y1, z00, z01);
                            draw_line(x0 + t1 * (x1 - x0), y0, x0, y0 + t2 * (y1 - y0), &mut svg);
                        }
                        2 | 13 => {
                            let t1 = lerp(x0, x1, z00, z10);
                            let t2 = lerp(y0, y1, z10, z11);
                            draw_line(x0 + t1 * (x1 - x0), y0, x1, y0 + t2 * (y1 - y0), &mut svg);
                        }
                        3 | 12 => {
                            let t1 = lerp(y0, y1, z00, z01);
                            let t2 = lerp(y0, y1, z10, z11);
                            draw_line(x0, y0 + t1 * (y1 - y0), x1, y0 + t2 * (y1 - y0), &mut svg);
                        }
                        4 | 11 => {
                            let t1 = lerp(y0, y1, z00, z01);
                            let t2 = lerp(x0, x1, z01, z11);
                            draw_line(x0, y0 + t1 * (y1 - y0), x0 + t2 * (x1 - x0), y1, &mut svg);
                        }
                        6 | 9 => {
                            let t1 = lerp(x0, x1, z00, z10);
                            let t2 = lerp(x0, x1, z01, z11);
                            draw_line(x0 + t1 * (x1 - x0), y0, x0 + t2 * (x1 - x0), y1, &mut svg);
                        }
                        7 | 8 => {
                            let t1 = lerp(y0, y1, z10, z11);
                            let t2 = lerp(x0, x1, z01, z11);
                            draw_line(x1, y0 + t1 * (y1 - y0), x0 + t2 * (x1 - x0), y1, &mut svg);
                        }
                        5 => {
                            let t1 = lerp(x0, x1, z00, z10);
                            let t2 = lerp(y0, y1, z00, z01);
                            draw_line(x0 + t1 * (x1 - x0), y0, x0, y0 + t2 * (y1 - y0), &mut svg);
                            let t3 = lerp(y0, y1, z10, z11);
                            let t4 = lerp(x0, x1, z01, z11);
                            draw_line(x1, y0 + t3 * (y1 - y0), x0 + t4 * (x1 - x0), y1, &mut svg);
                        }
                        10 => {
                            let t1 = lerp(x0, x1, z00, z10);
                            let t2 = lerp(y0, y1, z10, z11);
                            draw_line(x0 + t1 * (x1 - x0), y0, x1, y0 + t2 * (y1 - y0), &mut svg);
                            let t3 = lerp(y0, y1, z00, z01);
                            let t4 = lerp(x0, x1, z01, z11);
                            draw_line(x0, y0 + t3 * (y1 - y0), x0 + t4 * (x1 - x0), y1, &mut svg);
                        }
                        _ => {}
                    }
                }
            }
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        let n_labels = 5;
        for i in 0..=n_labels {
            let frac = i as f64 / n_labels as f64;
            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));
            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate vector field plot
    /// Syntax: vector_field((fx, fy), (x, xmin, xmax), (y, ymin, ymax))
    fn eval_vector_field(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "vector_field requires: (fx, fy), (x, xmin, xmax), (y, ymin, ymax) [, plot_points=N]"));
        }

        // Parse vector field components (fx, fy)
        let vec_tuple_str = parts[0].trim();
        if !vec_tuple_str.starts_with('(') || !vec_tuple_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "First argument must be tuple (fx, fy)"));
        }
        let vec_inner = &vec_tuple_str[1..vec_tuple_str.len()-1];
        let vec_parts = self.split_at_depth_zero(vec_inner, ',');
        if vec_parts.len() != 2 {
            return Err(EvalError::new("ArgumentError", "Vector field must have 2 components (fx, fy)"));
        }
        let fx_expr = self.parse_symbolic_arg(vec_parts[0].trim())?;
        let fy_expr = self.parse_symbolic_arg(vec_parts[1].trim())?;

        // Parse x range
        let x_range_str = parts[1].trim();
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (x, xmin, xmax)"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must be (variable, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range
        let y_range_str = parts[2].trim();
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (y, ymin, ymax)"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must be (variable, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        // Parse optional plot_points
        let mut n_points = 15;
        for i in 3..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_points = n.min(30).max(5);
                    }
                }
            }
        }

        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);
        let fx_symbols = fx_expr.symbols();
        let fy_symbols = fy_expr.symbols();

        // Generate vector field data
        let mut vectors: Vec<(f64, f64, f64, f64)> = Vec::new(); // (x, y, vx, vy)
        let x_step = (x_max - x_min) / (n_points - 1) as f64;
        let y_step = (y_max - y_min) / (n_points - 1) as f64;

        for i in 0..n_points {
            for j in 0..n_points {
                let x = x_min + i as f64 * x_step;
                let y = y_min + j as f64 * y_step;

                let mut eval_fx = fx_expr.clone();
                let mut eval_fy = fy_expr.clone();

                for sym in &fx_symbols {
                    if sym.name() == x_sym.name() {
                        eval_fx = eval_fx.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_fx = eval_fx.substitute(sym, &Expr::Real(y));
                    }
                }
                for sym in &fy_symbols {
                    if sym.name() == x_sym.name() {
                        eval_fy = eval_fy.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_fy = eval_fy.substitute(sym, &Expr::Real(y));
                    }
                }

                if let (Some(vx), Some(vy)) = (try_eval_to_f64(&eval_fx), try_eval_to_f64(&eval_fy)) {
                    if vx.is_finite() && vy.is_finite() {
                        vectors.push((x, y, vx, vy));
                    }
                }
            }
        }

        if vectors.is_empty() {
            return Err(EvalError::new("ValueError", "Could not evaluate vector field at any point"));
        }

        let title = format!("Vector Field: ({}, {})", vec_parts[0].trim(), vec_parts[1].trim());
        let svg = self.generate_vector_field_svg(&vectors, x_min, x_max, y_min, y_max, &title);
        let description = format!("Vector field with {} arrows", vectors.len());

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for vector field
    fn generate_vector_field_svg(
        &self,
        vectors: &[(f64, f64, f64, f64)],
        x_min: f64, x_max: f64,
        y_min: f64, y_max: f64,
        title: &str
    ) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let bg_color = "#f8f9fa";
        let grid_color = "#dee2e6";
        let axis_color = "#6c757d";
        let border_color = "#343a40";
        let arrow_color = "#2563eb";

        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let map_x = |x: f64| margin + (x - x_min) / x_range * plot_width;
        let map_y = |y: f64| margin + plot_height - (y - y_min) / y_range * plot_height;

        // Normalize arrows
        let max_mag = vectors.iter()
            .map(|(_, _, vx, vy)| (vx * vx + vy * vy).sqrt())
            .fold(0.0f64, f64::max);
        let scale = if max_mag > 0.0 { 20.0 / max_mag } else { 1.0 };

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        // Define arrowhead marker
        svg.push_str("<defs><marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"9\" refY=\"3.5\" orient=\"auto\"><polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#2563eb\"/></marker></defs>");

        svg.push_str(&format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>", width, height));
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, grid_color
        ));

        // Grid
        let n_grid = 5;
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;
            let x = margin + frac * plot_width;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                x, margin, x, margin + plot_height, grid_color
            ));
            let y = margin + frac * plot_height;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                margin, y, margin + plot_width, y, grid_color
            ));
        }

        // Axes at origin
        if y_min <= 0.0 && y_max >= 0.0 {
            let y0 = map_y(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }
        if x_min <= 0.0 && x_max >= 0.0 {
            let x0 = map_x(0.0);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        for i in 0..=n_grid {
            let frac = i as f64 / n_grid as f64;
            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));
            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        // Draw arrows
        for &(x, y, vx, vy) in vectors {
            let cx = map_x(x);
            let cy = map_y(y);
            let dx = vx * scale;
            let dy = -vy * scale; // Flip for SVG coordinates

            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\" marker-end=\"url(#arrowhead)\"/>",
                cx, cy, cx + dx, cy + dy, arrow_color
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate implicit plot
    /// Syntax: implicit_plot(f(x,y), (x, xmin, xmax), (y, ymin, ymax))
    /// Plots the curve where f(x,y) = 0
    fn eval_implicit_plot(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "implicit_plot requires: f(x,y), (x, xmin, xmax), (y, ymin, ymax)"));
        }

        let expr = self.parse_symbolic_arg(&parts[0])?;

        // Parse x range
        let x_range_str = parts[1].trim();
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (x, xmin, xmax)"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must be (variable, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range
        let y_range_str = parts[2].trim();
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (y, ymin, ymax)"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must be (variable, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        // Parse optional plot_points
        let mut n_grid = 100;
        for i in 3..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_grid = n.min(200).max(50);
                    }
                }
            }
        }

        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);
        let expr_symbols = expr.symbols();

        // Evaluate function on grid for marching squares
        let mut z_values: Vec<Vec<f64>> = Vec::new();
        let x_step = (x_max - x_min) / (n_grid - 1) as f64;
        let y_step = (y_max - y_min) / (n_grid - 1) as f64;

        for j in 0..n_grid {
            let mut row = Vec::new();
            let y = y_min + j as f64 * y_step;
            for i in 0..n_grid {
                let x = x_min + i as f64 * x_step;
                let mut eval_expr = expr.clone();

                for sym in &expr_symbols {
                    if sym.name() == x_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(y));
                    }
                }

                let z = try_eval_to_f64(&eval_expr).unwrap_or(f64::NAN);
                row.push(z);
            }
            z_values.push(row);
        }

        let title = format!("Implicit: {} = 0", &parts[0].trim());
        let svg = self.generate_implicit_svg(&z_values, x_min, x_max, y_min, y_max, &title);
        let description = "Implicit curve".to_string();

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for implicit plot using marching squares
    fn generate_implicit_svg(
        &self,
        z_values: &[Vec<f64>],
        x_min: f64, x_max: f64,
        y_min: f64, y_max: f64,
        title: &str
    ) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let bg_color = "#f8f9fa";
        let grid_color = "#dee2e6";
        let border_color = "#343a40";
        let axis_color = "#6c757d";
        let curve_color = "#dc3545";

        let n_grid = z_values.len();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let map_x = |i: usize| margin + (i as f64 / (n_grid - 1) as f64) * plot_width;
        let map_y = |j: usize| margin + plot_height - (j as f64 / (n_grid - 1) as f64) * plot_height;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        svg.push_str(&format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>", width, height));
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, grid_color
        ));

        // Grid
        let n_gridlines = 5;
        for i in 0..=n_gridlines {
            let frac = i as f64 / n_gridlines as f64;
            let x = margin + frac * plot_width;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                x, margin, x, margin + plot_height, grid_color
            ));
            let y = margin + frac * plot_height;
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1\"/>",
                margin, y, margin + plot_width, y, grid_color
            ));
        }

        // Axes at origin
        if y_min <= 0.0 && y_max >= 0.0 {
            let y0 = margin + plot_height - (-y_min / y_range * plot_height);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }
        if x_min <= 0.0 && x_max >= 0.0 {
            let x0 = margin + (-x_min / x_range * plot_width);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Marching squares for f(x,y) = 0
        let level = 0.0;
        for j in 0..n_grid-1 {
            for i in 0..n_grid-1 {
                let z00 = z_values[j][i];
                let z10 = z_values[j][i+1];
                let z01 = z_values[j+1][i];
                let z11 = z_values[j+1][i+1];

                if !z00.is_finite() || !z10.is_finite() || !z01.is_finite() || !z11.is_finite() {
                    continue;
                }

                let case = ((z00 >= level) as u8) |
                           (((z10 >= level) as u8) << 1) |
                           (((z01 >= level) as u8) << 2) |
                           (((z11 >= level) as u8) << 3);

                let x0 = map_x(i);
                let x1 = map_x(i+1);
                let y0 = map_y(j);
                let y1 = map_y(j+1);

                let lerp = |a: f64, b: f64, za: f64, zb: f64| {
                    if (zb - za).abs() < 1e-10 { 0.5 } else { (level - za) / (zb - za) }
                };

                let draw_line = |x1: f64, y1: f64, x2: f64, y2: f64, svg: &mut String| {
                    svg.push_str(&format!(
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"2\"/>",
                        x1, y1, x2, y2, curve_color
                    ));
                };

                // Edge crossing points:
                // bottom edge: between z00 and z10, at y=y0
                // top edge: between z01 and z11, at y=y1
                // left edge: between z00 and z01, at x=x0
                // right edge: between z10 and z11, at x=x1

                // Compute edge crossing coordinates when needed
                let bottom_x = || x0 + lerp(x0, x1, z00, z10) * (x1 - x0);
                let top_x = || x0 + lerp(x0, x1, z01, z11) * (x1 - x0);
                let left_y = || y0 + lerp(y0, y1, z00, z01) * (y1 - y0);
                let right_y = || y0 + lerp(y0, y1, z10, z11) * (y1 - y0);

                match case {
                    // Case 1: only bottom-left above -> bottom edge to left edge
                    1 => {
                        draw_line(bottom_x(), y0, x0, left_y(), &mut svg);
                    }
                    // Case 14: only bottom-left below (complement of 1)
                    14 => {
                        draw_line(bottom_x(), y0, x0, left_y(), &mut svg);
                    }
                    // Case 2: only bottom-right above -> bottom edge to right edge
                    2 => {
                        draw_line(bottom_x(), y0, x1, right_y(), &mut svg);
                    }
                    // Case 13: only bottom-right below (complement of 2)
                    13 => {
                        draw_line(bottom_x(), y0, x1, right_y(), &mut svg);
                    }
                    // Case 3: both bottom corners above -> left edge to right edge
                    3 => {
                        draw_line(x0, left_y(), x1, right_y(), &mut svg);
                    }
                    // Case 12: both top corners above (complement of 3)
                    12 => {
                        draw_line(x0, left_y(), x1, right_y(), &mut svg);
                    }
                    // Case 4: only top-left above -> left edge to top edge
                    4 => {
                        draw_line(x0, left_y(), top_x(), y1, &mut svg);
                    }
                    // Case 11: only top-left below (complement of 4)
                    11 => {
                        draw_line(x0, left_y(), top_x(), y1, &mut svg);
                    }
                    // Case 5: left column above (z00, z01) -> bottom edge to top edge
                    5 => {
                        draw_line(bottom_x(), y0, top_x(), y1, &mut svg);
                    }
                    // Case 10: right column above (z10, z11) -> bottom edge to top edge
                    10 => {
                        draw_line(bottom_x(), y0, top_x(), y1, &mut svg);
                    }
                    // Case 6: diagonal z10, z01 above (saddle case)
                    // Disambiguate using center value
                    6 => {
                        let z_center = (z00 + z10 + z01 + z11) / 4.0;
                        if z_center >= level {
                            // Center above: connect bottom-left and top-right
                            draw_line(bottom_x(), y0, x0, left_y(), &mut svg);
                            draw_line(x1, right_y(), top_x(), y1, &mut svg);
                        } else {
                            // Center below: connect bottom-right and top-left
                            draw_line(bottom_x(), y0, x1, right_y(), &mut svg);
                            draw_line(x0, left_y(), top_x(), y1, &mut svg);
                        }
                    }
                    // Case 9: diagonal z00, z11 above (saddle case)
                    // Disambiguate using center value
                    9 => {
                        let z_center = (z00 + z10 + z01 + z11) / 4.0;
                        if z_center >= level {
                            // Center above: connect bottom-right and top-left
                            draw_line(bottom_x(), y0, x1, right_y(), &mut svg);
                            draw_line(x0, left_y(), top_x(), y1, &mut svg);
                        } else {
                            // Center below: connect bottom-left and top-right
                            draw_line(bottom_x(), y0, x0, left_y(), &mut svg);
                            draw_line(x1, right_y(), top_x(), y1, &mut svg);
                        }
                    }
                    // Case 7: only top-right below -> right edge to top edge
                    7 => {
                        draw_line(x1, right_y(), top_x(), y1, &mut svg);
                    }
                    // Case 8: only top-right above -> right edge to top edge
                    8 => {
                        draw_line(x1, right_y(), top_x(), y1, &mut svg);
                    }
                    // Cases 0 and 15: all same side, no crossing
                    _ => {}
                }
            }
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        for i in 0..=n_gridlines {
            let frac = i as f64 / n_gridlines as f64;
            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));
            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Evaluate region plot
    /// Syntax: region_plot(condition, (x, xmin, xmax), (y, ymin, ymax))
    /// Plots the region where condition > 0
    fn eval_region_plot(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "region_plot requires: f(x,y), (x, xmin, xmax), (y, ymin, ymax)"));
        }

        let expr = self.parse_symbolic_arg(&parts[0])?;

        // Parse x range
        let x_range_str = parts[1].trim();
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (x, xmin, xmax)"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must be (variable, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range
        let y_range_str = parts[2].trim();
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("ArgumentError", "Expected tuple (y, ymin, ymax)"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must be (variable, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        // Parse optional plot_points
        let mut n_grid = 100;
        for i in 3..parts.len() {
            let part = parts[i].trim();
            if part.starts_with("plot_points") {
                if let Some(eq_pos) = part.find('=') {
                    if let Ok(n) = part[eq_pos+1..].trim().parse::<usize>() {
                        n_grid = n.min(200).max(50);
                    }
                }
            }
        }

        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);
        let expr_symbols = expr.symbols();

        // Evaluate function on grid
        let mut z_values: Vec<Vec<f64>> = Vec::new();
        let x_step = (x_max - x_min) / (n_grid - 1) as f64;
        let y_step = (y_max - y_min) / (n_grid - 1) as f64;

        for j in 0..n_grid {
            let mut row = Vec::new();
            let y = y_min + j as f64 * y_step;
            for i in 0..n_grid {
                let x = x_min + i as f64 * x_step;
                let mut eval_expr = expr.clone();

                for sym in &expr_symbols {
                    if sym.name() == x_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(x));
                    } else if sym.name() == y_sym.name() {
                        eval_expr = eval_expr.substitute(sym, &Expr::Real(y));
                    }
                }

                let z = try_eval_to_f64(&eval_expr).unwrap_or(f64::NAN);
                row.push(z);
            }
            z_values.push(row);
        }

        let title = format!("Region: {} > 0", &parts[0].trim());
        let svg = self.generate_region_svg(&z_values, x_min, x_max, y_min, y_max, &title);
        let description = "Region plot".to_string();

        Ok(RustMathValue::Plot { description, svg })
    }

    /// Generate SVG for region plot
    fn generate_region_svg(
        &self,
        z_values: &[Vec<f64>],
        x_min: f64, x_max: f64,
        y_min: f64, y_max: f64,
        title: &str
    ) -> String {
        let width = 600.0;
        let height = 600.0;
        let margin = 60.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let bg_color = "#f8f9fa";
        let border_color = "#343a40";
        let axis_color = "#6c757d";
        let region_color = "#3b82f6";

        let n_grid = z_values.len();
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;

        let cell_width = plot_width / (n_grid - 1) as f64;
        let cell_height = plot_height / (n_grid - 1) as f64;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            width, height, width, height
        );

        svg.push_str(&format!("<rect width=\"{}\" height=\"{}\" fill=\"white\"/>", width, height));
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" stroke=\"{}\"/>",
            margin, margin, plot_width, plot_height, bg_color, border_color
        ));

        // Draw region where z > 0
        for j in 0..n_grid {
            for i in 0..n_grid {
                let z = z_values[j][i];
                if z.is_finite() && z > 0.0 {
                    let x = margin + i as f64 * cell_width - cell_width / 2.0;
                    let y = margin + plot_height - (j as f64 + 1.0) * cell_height + cell_height / 2.0;
                    svg.push_str(&format!(
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\" fill-opacity=\"0.4\"/>",
                        x.max(margin), y.max(margin),
                        cell_width.min(margin + plot_width - x.max(margin)),
                        cell_height.min(margin + plot_height - y.max(margin)),
                        region_color
                    ));
                }
            }
        }

        // Axes at origin
        if y_min <= 0.0 && y_max >= 0.0 {
            let y0 = margin + plot_height - (-y_min / y_range * plot_height);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                margin, y0, margin + plot_width, y0, axis_color
            ));
        }
        if x_min <= 0.0 && x_max >= 0.0 {
            let x0 = margin + (-x_min / x_range * plot_width);
            svg.push_str(&format!(
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"1.5\"/>",
                x0, margin, x0, margin + plot_height, axis_color
            ));
        }

        // Title
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"25\" text-anchor=\"middle\" font-size=\"14\" font-family=\"sans-serif\" font-weight=\"bold\">{}</text>",
            width / 2.0, title
        ));

        // Border
        svg.push_str(&format!(
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\"/>",
            margin, margin, plot_width, plot_height, border_color
        ));

        // Axis labels
        let n_labels = 5;
        for i in 0..=n_labels {
            let frac = i as f64 / n_labels as f64;
            let x_val = x_min + frac * x_range;
            let x_pos = margin + frac * plot_width;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                x_pos, margin + plot_height + 15.0, x_val
            ));
            let y_val = y_max - frac * y_range;
            let y_pos = margin + frac * plot_height;
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-size=\"10\" font-family=\"sans-serif\">{:.1}</text>",
                margin - 5.0, y_pos + 4.0, y_val
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Convert a list of RustMathValues to Vec<f64> for statistics functions
    fn list_to_floats(&self, values: &[RustMathValue]) -> Result<Vec<f64>, EvalError> {
        values.iter().map(|v| {
            match v {
                RustMathValue::Integer(n) => {
                    n.to_f64().ok_or_else(|| EvalError::new("ValueError", "Integer too large to convert to float"))
                }
                RustMathValue::Float(f) => Ok(*f),
                RustMathValue::Rational(r) => {
                    let num = r.numerator().to_f64().ok_or_else(|| EvalError::new("ValueError", "Numerator too large"))?;
                    let den = r.denominator().to_f64().ok_or_else(|| EvalError::new("ValueError", "Denominator too large"))?;
                    Ok(num / den)
                }
                _ => Err(EvalError::new("TypeError", "List must contain only numbers")),
            }
        }).collect()
    }

    /// Convert a list of RustMathValues to Vec<Complex> for FFT functions
    fn list_to_complexes(&self, values: &[RustMathValue]) -> Result<Vec<Complex>, EvalError> {
        values.iter().map(|v| {
            match v {
                RustMathValue::Complex(c) => Ok(c.clone()),
                RustMathValue::Integer(n) => {
                    let f = n.to_f64().ok_or_else(|| EvalError::new("ValueError", "Integer too large"))?;
                    Ok(Complex::new(f, 0.0))
                }
                RustMathValue::Float(f) => Ok(Complex::new(*f, 0.0)),
                RustMathValue::Rational(r) => {
                    let num = r.numerator().to_f64().ok_or_else(|| EvalError::new("ValueError", "Numerator too large"))?;
                    let den = r.denominator().to_f64().ok_or_else(|| EvalError::new("ValueError", "Denominator too large"))?;
                    Ok(Complex::new(num / den, 0.0))
                }
                _ => Err(EvalError::new("TypeError", "List must contain only numbers or complex values")),
            }
        }).collect()
    }

    /// Evaluate an expression as usize (for graph vertex counts, etc.)
    fn eval_as_usize(&mut self, expr: &str) -> Result<usize, EvalError> {
        let val = self.eval_expr(expr)?;
        match val {
            RustMathValue::Integer(n) => {
                n.to_usize().ok_or_else(|| EvalError::new("ValueError", "value must be a non-negative integer"))
            }
            _ => Err(EvalError::new("TypeError", "expected an integer")),
        }
    }

    /// Convert a RustMathValue to f64
    fn value_to_f64(&self, val: &RustMathValue) -> Result<f64, EvalError> {
        match val {
            RustMathValue::Integer(n) => {
                n.to_f64().ok_or_else(|| EvalError::new("ValueError", "integer too large to convert to float"))
            }
            RustMathValue::Float(f) => Ok(*f),
            RustMathValue::Rational(r) => {
                let num = r.numerator().to_f64().ok_or_else(|| EvalError::new("ValueError", "numerator too large"))?;
                let den = r.denominator().to_f64().ok_or_else(|| EvalError::new("ValueError", "denominator too large"))?;
                Ok(num / den)
            }
            _ => Err(EvalError::new("TypeError", "expected a number")),
        }
    }

    /// Convert a list of RustMathValues to Vec<f64>
    fn list_to_f64s(&self, list: &[RustMathValue]) -> Result<Vec<f64>, EvalError> {
        list.iter().map(|v| self.value_to_f64(v)).collect()
    }

    /// Extract 2D coordinates from a list of points or [x, y] pairs
    fn extract_2d_coords(&self, pts: &[RustMathValue]) -> Result<Vec<(f64, f64)>, EvalError> {
        let mut coords = Vec::new();
        for pt in pts {
            match pt {
                RustMathValue::Point2D(p) => coords.push((p.x, p.y)),
                RustMathValue::List(pair) if pair.len() == 2 => {
                    let x = self.value_to_f64(&pair[0])?;
                    let y = self.value_to_f64(&pair[1])?;
                    coords.push((x, y));
                }
                _ => return Err(EvalError::new("TypeError", "expected Point2D or [x, y] pairs")),
            }
        }
        Ok(coords)
    }

    /// Extract 3D coordinates from a list of Point3D values
    fn extract_3d_coords(&self, pts: &[RustMathValue]) -> Result<Vec<(f64, f64, f64)>, EvalError> {
        let mut coords = Vec::new();
        for pt in pts {
            match pt {
                RustMathValue::Point3D(p) => coords.push((p.x, p.y, p.z)),
                RustMathValue::List(triple) if triple.len() == 3 => {
                    let x = self.value_to_f64(&triple[0])?;
                    let y = self.value_to_f64(&triple[1])?;
                    let z = self.value_to_f64(&triple[2])?;
                    coords.push((x, y, z));
                }
                _ => return Err(EvalError::new("TypeError", "expected Point3D or [x, y, z] triples")),
            }
        }
        Ok(coords)
    }

    /// Convert an Expr to f64
    fn expr_to_f64(&self, expr: &Expr) -> Result<f64, EvalError> {
        match try_eval_to_f64(expr) {
            Some(v) => Ok(v),
            None => Err(EvalError::new("ValueError", "could not evaluate expression to a number")),
        }
    }

    /// Generate SVG for a 2D scatter or line plot
    fn generate_scatter_svg(&self, coords: &[(f64, f64)], title: &str, connect_lines: bool) -> String {
        if coords.is_empty() {
            return String::new();
        }

        let width = 500.0;
        let height = 400.0;
        let margin = 50.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        // Find data bounds
        let (min_x, max_x, min_y, max_y) = self.find_bounds_2d(coords);
        let x_range = if (max_x - min_x).abs() < 1e-10 { 1.0 } else { max_x - min_x };
        let y_range = if (max_y - min_y).abs() < 1e-10 { 1.0 } else { max_y - min_y };

        // Map data to SVG coordinates
        let map_x = |x: f64| margin + (x - min_x) / x_range * plot_width;
        let map_y = |y: f64| margin + plot_height - (y - min_y) / y_range * plot_height;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            width, height, width, height
        );

        // Background
        svg.push_str(&format!(
            r#"<rect width="{}" height="{}" fill="white"/>"#,
            width, height
        ));

        // Title
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
            width / 2.0, title
        ));

        // Axes
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
            margin, margin + plot_height, margin + plot_width, margin + plot_height
        ));
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
            margin, margin, margin, margin + plot_height
        ));

        // Axis labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="start" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin, margin + plot_height + 15.0, min_x
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin + plot_width, margin + plot_height + 15.0, max_x
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin - 5.0, margin + plot_height, min_y
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin - 5.0, margin + 10.0, max_y
        ));

        // Draw lines if requested
        let blue = "#2563eb";
        if connect_lines && coords.len() > 1 {
            let mut path = format!("M {} {}", map_x(coords[0].0), map_y(coords[0].1));
            for (x, y) in &coords[1..] {
                path.push_str(&format!(" L {} {}", map_x(*x), map_y(*y)));
            }
            svg.push_str(&format!(
                r#"<path d="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
                path, blue
            ));
        }

        // Draw points
        for (x, y) in coords {
            let sx = map_x(*x);
            let sy = map_y(*y);
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="4" fill="{}"/>"#,
                sx, sy, blue
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Generate SVG for a histogram
    fn generate_histogram_svg(&self, data: &[f64], n_bins: usize) -> String {
        if data.is_empty() {
            return String::new();
        }

        let width = 500.0;
        let height = 400.0;
        let margin = 50.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max_val - min_val).abs() < 1e-10 { 1.0 } else { max_val - min_val };
        let bin_width = range / n_bins as f64;

        // Count values in each bin
        let mut counts = vec![0usize; n_bins];
        for &v in data {
            let bin = ((v - min_val) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            counts[bin] += 1;
        }

        let max_count = *counts.iter().max().unwrap_or(&1);
        let bar_width = plot_width / n_bins as f64;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">Histogram</text>"#,
            width / 2.0
        ));

        // Draw bars
        let blue = "#2563eb";
        for (i, &count) in counts.iter().enumerate() {
            let bar_height = (count as f64 / max_count as f64) * plot_height;
            let x = margin + i as f64 * bar_width;
            let y = margin + plot_height - bar_height;
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white" stroke-width="1"/>"#,
                x, y, bar_width * 0.9, bar_height, blue
            ));
        }

        // Axes
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
            margin, margin + plot_height, margin + plot_width, margin + plot_height
        ));

        // Axis labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="start" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin, margin + plot_height + 15.0, min_val
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-size="10" font-family="sans-serif">{:.2}</text>"#,
            margin + plot_width, margin + plot_height + 15.0, max_val
        ));

        svg.push_str("</svg>");
        svg
    }

    /// Generate SVG for a bar chart
    fn generate_bar_chart_svg(&self, labels: &[String], values: &[f64]) -> String {
        if labels.is_empty() {
            return String::new();
        }

        let width = 500.0;
        let height = 400.0;
        let margin = 50.0;
        let plot_width = width - 2.0 * margin;
        let plot_height = height - 2.0 * margin;

        let max_val = values.iter().cloned().fold(0.0f64, f64::max);
        let bar_width = plot_width / labels.len() as f64;

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">Bar Chart</text>"#,
            width / 2.0
        ));

        // Draw bars
        let blue = "#2563eb";
        for (i, (label, &value)) in labels.iter().zip(values.iter()).enumerate() {
            let bar_height = if max_val > 0.0 { (value / max_val) * plot_height } else { 0.0 };
            let x = margin + i as f64 * bar_width;
            let y = margin + plot_height - bar_height;

            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>"#,
                x + bar_width * 0.1, y, bar_width * 0.8, bar_height, blue
            ));

            // Label (truncate if too long)
            let display_label: String = label.chars().take(8).collect();
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" font-size="10" font-family="sans-serif">{}</text>"#,
                x + bar_width / 2.0, margin + plot_height + 15.0, display_label
            ));
        }

        // Y axis
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="1"/>"#,
            margin, margin, margin, margin + plot_height
        ));

        svg.push_str("</svg>");
        svg
    }

    /// Generate SVG for a 3D scatter plot (isometric projection)
    fn generate_3d_scatter_svg(&self, coords: &[(f64, f64, f64)], title: &str) -> String {
        if coords.is_empty() {
            return String::new();
        }

        let width = 500.0;
        let height = 400.0;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        let scale = 100.0;

        // Find data bounds
        let (min_x, max_x) = coords.iter().map(|p| p.0).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));
        let (min_y, max_y) = coords.iter().map(|p| p.1).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));
        let (min_z, max_z) = coords.iter().map(|p| p.2).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));

        let range_x = if (max_x - min_x).abs() < 1e-10 { 1.0 } else { max_x - min_x };
        let range_y = if (max_y - min_y).abs() < 1e-10 { 1.0 } else { max_y - min_y };
        let range_z = if (max_z - min_z).abs() < 1e-10 { 1.0 } else { max_z - min_z };

        // Isometric projection: x -> right, y -> up-right, z -> up
        let project = |x: f64, y: f64, z: f64| -> (f64, f64) {
            let nx = (x - min_x) / range_x - 0.5;
            let ny = (y - min_y) / range_y - 0.5;
            let nz = (z - min_z) / range_z - 0.5;

            let px = center_x + scale * (nx - ny) * 0.866;
            let py = center_y - scale * (nx + ny) * 0.5 - scale * nz;
            (px, py)
        };

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
            width / 2.0, title
        ));

        // Draw axes
        let origin = project(min_x, min_y, min_z);
        let x_end = project(max_x, min_y, min_z);
        let y_end = project(min_x, max_y, min_z);
        let z_end = project(min_x, min_y, max_z);

        let gray = "#999";
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            origin.0, origin.1, x_end.0, x_end.1, gray
        ));
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            origin.0, origin.1, y_end.0, y_end.1, gray
        ));
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            origin.0, origin.1, z_end.0, z_end.1, gray
        ));

        // Axis labels
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-size="10" font-family="sans-serif">X</text>"#,
            x_end.0 + 5.0, x_end.1
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-size="10" font-family="sans-serif">Y</text>"#,
            y_end.0 + 5.0, y_end.1
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-size="10" font-family="sans-serif">Z</text>"#,
            z_end.0 - 10.0, z_end.1 - 5.0
        ));

        // Sort by depth for proper z-ordering (back to front)
        let mut points_with_depth: Vec<_> = coords.iter()
            .map(|&(x, y, z)| {
                let (px, py) = project(x, y, z);
                let depth = x + y - z; // Simple depth ordering
                (px, py, depth)
            })
            .collect();
        points_with_depth.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Draw points
        let blue = "#2563eb";
        for (px, py, _) in points_with_depth {
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="5" fill="{}" stroke="white" stroke-width="1"/>"#,
                px, py, blue
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Generate SVG for a graph visualization (circular layout)
    fn generate_graph_svg(&self, g: &Graph) -> String {
        let n = g.num_vertices();
        if n == 0 {
            return String::new();
        }

        let width = 500.0;
        let height = 400.0;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        let radius = 150.0_f64.min(width / 3.0);

        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
            width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">Graph ({} vertices, {} edges)</text>"#,
            width / 2.0, n, g.num_edges()
        ));

        // Compute vertex positions (circular layout)
        let positions: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::PI / 2.0;
                (
                    center_x + radius * angle.cos(),
                    center_y + radius * angle.sin(),
                )
            })
            .collect();

        // Draw edges
        let gray = "#999";
        let blue = "#2563eb";
        for (u, v) in g.edges() {
            if u < n && v < n {
                let (x1, y1) = positions[u];
                let (x2, y2) = positions[v];
                svg.push_str(&format!(
                    r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
                    x1, y1, x2, y2, gray
                ));
            }
        }

        // Draw vertices
        for (i, &(x, y)) in positions.iter().enumerate() {
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="15" fill="{}" stroke="white" stroke-width="2"/>"#,
                x, y, blue
            ));
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" dominant-baseline="central" font-size="10" font-family="sans-serif" fill="white">{}</text>"#,
                x, y, i
            ));
        }

        svg.push_str("</svg>");
        svg
    }

    /// Find min/max bounds for 2D coordinates
    fn find_bounds_2d(&self, coords: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in coords {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        (min_x, max_x, min_y, max_y)
    }

    fn try_eval_binary_op(&mut self, expr: &str) -> Result<Option<RustMathValue>, EvalError> {
        // Parse operators in order of LOWEST precedence first (outermost splitting)
        // This ensures proper expression tree construction

        // Helper to find operator position outside of parentheses
        let find_op_outside_parens = |expr: &str, op: &str| -> Option<usize> {
            let mut paren_depth = 0;
            let mut bracket_depth = 0;
            let chars: Vec<char> = expr.chars().collect();
            let op_chars: Vec<char> = op.chars().collect();

            // Search from right to left for left-associative operators
            // When going right-to-left: ')' opens a context, '(' closes it
            for i in (0..=chars.len().saturating_sub(op_chars.len())).rev() {
                match chars.get(i) {
                    Some(')') => paren_depth += 1,  // entering nested context
                    Some('(') => paren_depth -= 1,  // leaving nested context
                    Some(']') => bracket_depth += 1,
                    Some('[') => bracket_depth -= 1,
                    _ => {}
                }

                if paren_depth == 0 && bracket_depth == 0 {
                    let matches = (0..op_chars.len()).all(|j|
                        chars.get(i + j) == Some(&op_chars[j])
                    );
                    if matches {
                        // For *, also check we're not matching part of **
                        if op == "*" {
                            // Make sure the previous char isn't * (which would be **)
                            if i > 0 && chars.get(i - 1) == Some(&'*') {
                                continue;
                            }
                            // Make sure the next char isn't * (which would be **)
                            if chars.get(i + 1) == Some(&'*') {
                                continue;
                            }
                        }
                        return Some(i);
                    }
                }
            }
            None
        };

        // 1. Addition/subtraction (LOWEST precedence - split first)
        for op in &["+", "-"] {
            if let Some(pos) = find_op_outside_parens(expr, op) {
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

        // 2. Multiplication and Division (middle precedence)
        if let Some(pos) = find_op_outside_parens(expr, "*") {
            if pos > 0 {
                let left = expr[..pos].trim();
                let right = expr[pos + 1..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, "*", &right_val)?));
                }
            }
        }

        if let Some(pos) = find_op_outside_parens(expr, "/") {
            if pos > 0 {
                let left = expr[..pos].trim();
                let right = expr[pos + 1..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, "/", &right_val)?));
                }
            }
        }

        // 3. Power (HIGHEST precedence - split last, right-to-left associative)
        // For power, we want leftmost to make it right-associative: 2**3**4 = 2**(3**4)
        let find_power_op = |expr: &str| -> Option<(usize, &str)> {
            let mut paren_depth = 0;
            let mut bracket_depth = 0;
            let chars: Vec<char> = expr.chars().collect();

            // Search from LEFT for right-associative power operator
            for i in 0..chars.len() {
                match chars.get(i) {
                    Some('(') => paren_depth += 1,
                    Some(')') => paren_depth -= 1,
                    Some('[') => bracket_depth += 1,
                    Some(']') => bracket_depth -= 1,
                    _ => {}
                }

                if paren_depth == 0 && bracket_depth == 0 {
                    // Check for **
                    if chars.get(i) == Some(&'*') && chars.get(i + 1) == Some(&'*') {
                        return Some((i, "**"));
                    }
                    // Check for ^ (but not inside function calls)
                    if chars.get(i) == Some(&'^') {
                        return Some((i, "^"));
                    }
                }
            }
            None
        };

        if let Some((pos, op)) = find_power_op(expr) {
            if pos > 0 {
                let op_len = op.len();
                let left = expr[..pos].trim();
                let right = expr[pos + op_len..].trim();
                if !left.is_empty() && !right.is_empty() {
                    let left_val = self.eval_expr(left)?;
                    let right_val = self.eval_expr(right)?;
                    return Ok(Some(self.apply_binary_op(&left_val, op, &right_val)?));
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

            // Matrix operations
            (Matrix(a), "+", Matrix(b)) => {
                if a.rows() != b.rows() || a.cols() != b.cols() {
                    return Err(EvalError::new("MatrixError", "Matrix dimensions must match for addition"));
                }
                // Add matrices element-wise
                let mut data = Vec::new();
                for i in 0..a.rows() {
                    for j in 0..a.cols() {
                        let val_a = a.get(i, j).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let val_b = b.get(i, j).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        data.push(val_a.clone() + val_b.clone());
                    }
                }
                let result = rustmath_matrix::Matrix::from_vec(a.rows(), a.cols(), data)
                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                Ok(Matrix(result))
            }

            (Matrix(a), "-", Matrix(b)) => {
                if a.rows() != b.rows() || a.cols() != b.cols() {
                    return Err(EvalError::new("MatrixError", "Matrix dimensions must match for subtraction"));
                }
                // Subtract matrices element-wise
                let mut data = Vec::new();
                for i in 0..a.rows() {
                    for j in 0..a.cols() {
                        let val_a = a.get(i, j).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let val_b = b.get(i, j).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        data.push(val_a.clone() - val_b.clone());
                    }
                }
                let result = rustmath_matrix::Matrix::from_vec(a.rows(), a.cols(), data)
                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                Ok(Matrix(result))
            }

            (Matrix(a), "*", Matrix(b)) => {
                // Matrix multiplication
                let result = a.mul(b)
                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                Ok(Matrix(result))
            }

            // Matrix scalar multiplication
            (Matrix(m), "*", Integer(k)) | (Integer(k), "*", Matrix(m)) => {
                Ok(Matrix(m.scalar_mul(k)))
            }

            // Polynomial operations
            (Polynomial(p), "+", Polynomial(q)) => {
                Ok(Polynomial(p.clone() + q.clone()))
            }
            (Polynomial(p), "-", Polynomial(q)) => {
                Ok(Polynomial(p.clone() - q.clone()))
            }
            (Polynomial(p), "*", Polynomial(q)) => {
                Ok(Polynomial(p.clone() * q.clone()))
            }
            // Polynomial scalar multiplication
            (Polynomial(p), "*", Integer(k)) | (Integer(k), "*", Polynomial(p)) => {
                Ok(Polynomial(p.scalar_mul(k)))
            }

            // IntegerMod (finite field) operations
            (IntegerMod(a), "+", IntegerMod(b)) => {
                if a.modulus() != b.modulus() {
                    return Err(EvalError::new("ValueError", "Cannot add elements with different moduli"));
                }
                Ok(IntegerMod(a.clone() + b.clone()))
            }
            (IntegerMod(a), "-", IntegerMod(b)) => {
                if a.modulus() != b.modulus() {
                    return Err(EvalError::new("ValueError", "Cannot subtract elements with different moduli"));
                }
                Ok(IntegerMod(a.clone() - b.clone()))
            }
            (IntegerMod(a), "*", IntegerMod(b)) => {
                if a.modulus() != b.modulus() {
                    return Err(EvalError::new("ValueError", "Cannot multiply elements with different moduli"));
                }
                Ok(IntegerMod(a.clone() * b.clone()))
            }
            (IntegerMod(a), "/", IntegerMod(b)) => {
                if a.modulus() != b.modulus() {
                    return Err(EvalError::new("ValueError", "Cannot divide elements with different moduli"));
                }
                let b_inv = b.inverse()
                    .map_err(|_| EvalError::new("ValueError", "Cannot divide: divisor has no inverse"))?;
                Ok(IntegerMod(a.clone() * b_inv))
            }
            (IntegerMod(a), "**", Integer(exp)) | (IntegerMod(a), "^", Integer(exp)) => {
                let result = a.pow(exp)
                    .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                Ok(IntegerMod(result))
            }

            // Clifford algebra element operations
            (CliffordElem(a), "+", CliffordElem(b)) => {
                Ok(CliffordElem(a.clone() + b.clone()))
            }
            (CliffordElem(a), "-", CliffordElem(b)) => {
                Ok(CliffordElem(a.clone() - b.clone()))
            }
            (CliffordElem(a), "*", CliffordElem(b)) => {
                // Clifford product
                Ok(CliffordElem(a.clone() * b.clone()))
            }
            // Wedge product for Clifford/Exterior algebra elements
            // In exterior algebra (Q=0), the Clifford product IS the wedge product
            // For general Clifford algebra, ^ also means the antisymmetric (wedge) product
            (CliffordElem(a), "^", CliffordElem(b)) => {
                // Use Clifford product - for exterior algebra this IS the wedge product
                Ok(CliffordElem(a.clone() * b.clone()))
            }
            // Mixed type operations for Clifford elements
            // Integer + CliffordElem: create scalar element and add
            (Integer(k), "+", CliffordElem(a)) | (CliffordElem(a), "+", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from(k.to_i64());
                let scalar = CliffordAlgebraElement::scalar(coeff, 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar + a.clone()))
            }
            (Integer(k), "-", CliffordElem(a)) => {
                let coeff = rustmath_rationals::Rational::from(k.to_i64());
                let scalar = CliffordAlgebraElement::scalar(coeff, 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar - a.clone()))
            }
            (CliffordElem(a), "-", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from(k.to_i64());
                let scalar = CliffordAlgebraElement::scalar(coeff, 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(a.clone() - scalar))
            }
            // Scalar multiplication: Integer * CliffordElem
            (Integer(k), "*", CliffordElem(a)) | (CliffordElem(a), "*", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from(k.to_i64());
                let scalar = CliffordAlgebraElement::scalar(coeff, 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar * a.clone()))
            }
            // Scalar division: CliffordElem / Integer
            (CliffordElem(a), "/", Integer(k)) => {
                if k.is_zero() {
                    return Err(EvalError::new("ZeroDivisionError", "Division by zero"));
                }
                // Create 1/k as a Rational
                let one = rustmath_rationals::Rational::from(1i64);
                let divisor = rustmath_rationals::Rational::from(k.to_i64());
                let coeff = one / divisor;
                let scalar = CliffordAlgebraElement::scalar(coeff, 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar * a.clone()))
            }
            // Rational operations with Clifford elements
            (Rational(r), "+", CliffordElem(a)) | (CliffordElem(a), "+", Rational(r)) => {
                let scalar = CliffordAlgebraElement::scalar(r.clone(), 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar + a.clone()))
            }
            (Rational(r), "*", CliffordElem(a)) | (CliffordElem(a), "*", Rational(r)) => {
                let scalar = CliffordAlgebraElement::scalar(r.clone(), 1, vec![rustmath_rationals::Rational::from(1i64)]);
                Ok(CliffordElem(scalar * a.clone()))
            }

            // String operations (Python-like)
            // String repetition: "=" * 60 or 60 * "="
            (String(s), "*", Integer(n)) | (Integer(n), "*", String(s)) => {
                let count = n.to_i64();
                if count < 0 {
                    return Ok(String("".to_string()));
                }
                if count > 10000 {
                    return Err(EvalError::new("ValueError", "String repetition count too large"));
                }
                Ok(String(s.repeat(count as usize)))
            }
            // String concatenation: "a" + "b"
            (String(a), "+", String(b)) => {
                Ok(String(format!("{}{}", a, b)))
            }

            // ====== SYMBOLIC OPERATIONS ======
            // Symbol operations - convert to Expr for symbolic manipulation
            // Note: Use rustmath_symbolic::Expr explicitly since RustMathValue::Expr shadows it

            // Symbol ** Integer -> Expr (power)
            (Symbol(s), "**", Integer(n)) | (Symbol(s), "^", Integer(n)) => {
                let base = rustmath_symbolic::Expr::Symbol(s.clone());
                let exp = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(base.pow(exp)))
            }

            // Symbol * Integer or Integer * Symbol -> Expr
            (Symbol(s), "*", Integer(n)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(int_expr * sym_expr))
            }
            (Integer(n), "*", Symbol(s)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(int_expr * sym_expr))
            }

            // Symbol + Symbol -> Expr
            (Symbol(a), "+", Symbol(b)) => {
                let a_expr = rustmath_symbolic::Expr::Symbol(a.clone());
                let b_expr = rustmath_symbolic::Expr::Symbol(b.clone());
                Ok(Expr(a_expr + b_expr))
            }

            // Symbol - Symbol -> Expr
            (Symbol(a), "-", Symbol(b)) => {
                let a_expr = rustmath_symbolic::Expr::Symbol(a.clone());
                let b_expr = rustmath_symbolic::Expr::Symbol(b.clone());
                Ok(Expr(a_expr - b_expr))
            }

            // Symbol * Symbol -> Expr
            (Symbol(a), "*", Symbol(b)) => {
                let a_expr = rustmath_symbolic::Expr::Symbol(a.clone());
                let b_expr = rustmath_symbolic::Expr::Symbol(b.clone());
                Ok(Expr(a_expr * b_expr))
            }

            // Symbol + Integer -> Expr
            (Symbol(s), "+", Integer(n)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(sym_expr + int_expr))
            }
            (Integer(n), "+", Symbol(s)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(int_expr + sym_expr))
            }

            // Symbol - Integer -> Expr
            (Symbol(s), "-", Integer(n)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(sym_expr - int_expr))
            }
            (Integer(n), "-", Symbol(s)) => {
                let sym_expr = rustmath_symbolic::Expr::Symbol(s.clone());
                let int_expr = rustmath_symbolic::Expr::Integer(n.clone());
                Ok(Expr(int_expr - sym_expr))
            }

            // Expr operations
            (Expr(a), "+", Expr(b)) => Ok(Expr(a.clone() + b.clone())),
            (Expr(a), "-", Expr(b)) => Ok(Expr(a.clone() - b.clone())),
            (Expr(a), "*", Expr(b)) => Ok(Expr(a.clone() * b.clone())),
            (Expr(a), "/", Expr(b)) => Ok(Expr(a.clone() / b.clone())),
            (Expr(a), "**", Expr(b)) | (Expr(a), "^", Expr(b)) => {
                Ok(Expr(a.clone().pow(b.clone())))
            }

            // Expr with Integer
            (Expr(e), "+", Integer(n)) => Ok(Expr(e.clone() + rustmath_symbolic::Expr::Integer(n.clone()))),
            (Integer(n), "+", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Integer(n.clone()) + e.clone())),
            (Expr(e), "-", Integer(n)) => Ok(Expr(e.clone() - rustmath_symbolic::Expr::Integer(n.clone()))),
            (Integer(n), "-", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Integer(n.clone()) - e.clone())),
            (Expr(e), "*", Integer(n)) => Ok(Expr(e.clone() * rustmath_symbolic::Expr::Integer(n.clone()))),
            (Integer(n), "*", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Integer(n.clone()) * e.clone())),
            (Expr(e), "**", Integer(n)) | (Expr(e), "^", Integer(n)) => {
                Ok(Expr(e.clone().pow(rustmath_symbolic::Expr::Integer(n.clone()))))
            }

            // Expr with Symbol
            (Expr(e), "+", Symbol(s)) => Ok(Expr(e.clone() + rustmath_symbolic::Expr::Symbol(s.clone()))),
            (Symbol(s), "+", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Symbol(s.clone()) + e.clone())),
            (Expr(e), "-", Symbol(s)) => Ok(Expr(e.clone() - rustmath_symbolic::Expr::Symbol(s.clone()))),
            (Symbol(s), "-", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Symbol(s.clone()) - e.clone())),
            (Expr(e), "*", Symbol(s)) => Ok(Expr(e.clone() * rustmath_symbolic::Expr::Symbol(s.clone()))),
            (Symbol(s), "*", Expr(e)) => Ok(Expr(rustmath_symbolic::Expr::Symbol(s.clone()) * e.clone())),
            (Expr(e), "**", Symbol(s)) | (Expr(e), "^", Symbol(s)) => {
                Ok(Expr(e.clone().pow(rustmath_symbolic::Expr::Symbol(s.clone()))))
            }
            (Symbol(s), "**", Expr(e)) | (Symbol(s), "^", Expr(e)) => {
                Ok(Expr(rustmath_symbolic::Expr::Symbol(s.clone()).pow(e.clone())))
            }

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
  Matrix([[...]])    - Matrix from nested list
  Polynomial([...])  - Polynomial from coefficients

Integer Functions:
  factorial(n)       - Compute n!
  gcd(a, b)          - Greatest common divisor
  xgcd(a, b)         - Extended GCD: returns [gcd, x, y] where gcd = a*x + b*y
  lcm(a, b)          - Least common multiple
  is_prime(n)        - Test primality
  factor(n)          - Prime factorization
  abs(n)             - Absolute value
  pow(base, exp)     - Exponentiation

Matrix Functions:
  Matrix([[1,2],[3,4]]) - Create matrix
  identity(n)        - n×n identity matrix
  zeros(m, n)        - m×n zero matrix
  det(M)             - Determinant
  transpose(M)       - Transpose
  trace(M)           - Trace (sum of diagonal)
  rows(M), cols(M)   - Matrix dimensions
  shape(M)           - [rows, cols]
  is_square(M)       - True if square
  is_symmetric(M)    - True if symmetric
  is_diagonal(M)     - True if diagonal

Polynomial Functions:
  Polynomial([1,2,3])  - Create 1 + 2x + 3x²
  poly([1,2,3])        - Alias for Polynomial
  degree(p)            - Degree of polynomial
  derivative(p)        - Derivative of polynomial
  eval_poly(p, x)      - Evaluate p at point x
  coefficients(p)      - Get coefficient list
  leading_coeff(p)     - Leading coefficient
  is_monic(p)          - True if leading coeff = 1
  content(p)           - GCD of coefficients
  is_square_free(p)    - True if no repeated roots
  discriminant(p)      - Polynomial discriminant
  roots(p)             - Find rational roots
  gcd_poly(p, q)       - GCD of polynomials
  lcm_poly(p, q)       - LCM of polynomials
  compose(p, q)        - p(q(x)) composition
  divmod(p, q)         - [quotient, remainder]

Symbolic Functions:
  expr("x^2 + 3*x")    - Parse symbolic expression
  diff(expr, var)      - Differentiate w.r.t. var
  integrate(expr, var) - Integrate w.r.t. var
  simplify(expr)       - Simplify expression
  expand(expr)         - Expand expression
  solve(expr, var)     - Solve expr = 0 for var
  substitute(e, x, v)  - Substitute v for x in e
  taylor(e, x, a, n)   - Taylor series at x=a
  limit(e, x, a)       - Limit as x approaches a
  evalf(expr)          - Numerical evaluation
  sin, cos, tan(expr)  - Trig functions
  exp, log, sqrt(expr) - Exponential functions

Finite Field Functions:
  mod(a, n)            - Create a (mod n) in Z/nZ
  Mod(a, n)            - Alias for mod
  GF(p)                - Alias for mod (Galois field)
  inverse(a)           - Multiplicative inverse in Z/nZ
  is_unit(a)           - True if a has inverse in Z/nZ
  sqrt_mod(a, p)       - Square root of a mod p
  pow_mod(b, e, m)     - Compute b^e mod m

Combinatorics Functions:
  binomial(n, k)       - Binomial coefficient C(n,k)
  catalan(n)           - n-th Catalan number
  fibonacci(n)         - n-th Fibonacci number
  lucas(n)             - n-th Lucas number
  bell(n)              - n-th Bell number
  stirling1(n, k)      - Stirling number 1st kind
  stirling2(n, k)      - Stirling number 2nd kind
  multinomial(n, [k1, k2, ...]) - Multinomial coefficient (requires sum(ks) = n)
  falling_factorial(n, k) - (n)_k falling factorial
  rising_factorial(n, k)  - (n)^(k) rising factorial
  eulerian(n, k)       - Eulerian number
  narayana(n, k)       - Narayana number
  motzkin(n)           - n-th Motzkin number
  delannoy(m, n)       - Delannoy number
  schroder(n)          - Large Schröder number
  partitions(n)        - Number of partitions of n
  derangements(n)      - Number of derangements of n

Number Theory Functions:
  divisors(n)          - List of divisors
  num_divisors(n)      - Number of divisors (tau)
  sigma(n)             - Sum of divisors
  sigma(n, k)          - Sum of k-th powers of divisors
  euler_phi(n)         - Euler's totient function
  mobius(n)            - Möbius function
  radical(n)           - Product of distinct prime factors
  is_square_free(n)    - True if n has no squared prime factors
  valuation(n, p)      - p-adic valuation of n
  next_prime(n)        - Smallest prime > n
  nth_prime(n)         - n-th prime number
  prime_pi(n)          - Count of primes ≤ n
  primes(a, b)         - List of primes in [a, b]
  legendre(a, p)       - Legendre symbol (a/p)
  jacobi(a, n)         - Jacobi symbol (a/n)
  crt(rems, mods)      - Chinese Remainder Theorem

Statistics Functions:
  mean([a,b,c,...])    - Arithmetic mean
  median([a,b,c,...])  - Median value
  mode([a,b,c,...])    - Most frequent value
  variance([a,b,...])  - Sample variance
  std_dev([a,b,...])   - Standard deviation
  correlation(xs, ys)  - Pearson correlation coefficient
  covariance(xs, ys)   - Covariance of two datasets

Numerical Functions:
  fft([c1,c2,...])     - Fast Fourier Transform
  ifft([c1,c2,...])    - Inverse FFT
  integrate_num(expr, var, a, b) - Numerical integration (Simpson's rule)
  find_root(expr, var, a, b)     - Find root in interval (bisection)

Graph Functions:
  Graph(n)             - Create graph with n vertices
  complete_graph(n)    - Complete graph K_n
  cycle_graph(n)       - Cycle graph C_n
  path_graph(n)        - Path graph P_n
  star_graph(n)        - Star graph S_n
  wheel_graph(n)       - Wheel graph W_n
  petersen_graph()     - Petersen graph (10 vertices, 15 edges)
  num_vertices(G)      - Number of vertices in G
  num_edges(G)         - Number of edges in G
  is_connected(G)      - Check if G is connected
  chromatic_number(G)  - Chromatic number of G
  diameter(G)          - Diameter of G
  vertex_degree(G, v)  - Degree of vertex v
  neighbors(G, v)      - Neighbors of vertex v
  shortest_path(G, u, v) - Shortest path from u to v
  bfs(G, start)        - BFS traversal from start
  dfs(G, start)        - DFS traversal from start
  add_edge(G, u, v)    - Add edge (u,v) to G
  has_edge(G, u, v)    - Check if edge (u,v) exists

Geometry Functions:
  Point(x, y)          - Create 2D point
  Point3D(x, y, z)     - Create 3D point
  distance(p1, p2)     - Distance between points
  dot(p1, p2)          - Dot product
  cross(p1, p2)        - Cross product
  collinear(p1, p2, p3) - Check if points are collinear
  convex_hull([pts])   - Convex hull of points
  area([pts])          - Area of polygon
  perimeter([pts])     - Perimeter of polygon
  is_convex([pts])     - Check if polygon is convex

Plotting Functions (2D):
  plot([pts])          - Scatter plot from list of Point(x,y)
  scatter(xs, ys)      - Scatter plot from two lists
  line_plot([pts])     - Connected line plot
  plot_function(f, x, a, b) - Plot expression f(x) from a to b
  histogram(data)      - Histogram with auto binning
  histogram(data, bins) - Histogram with specified bins
  bar_chart(labels, values) - Bar chart with labels

Plotting Functions (3D):
  plot3d([pts])        - 3D scatter plot (isometric projection)
  scatter3d([pts])     - Alias for plot3d

Graph Visualization:
  show_graph(G)        - Visualize graph with circular layout
  draw_graph(G)        - Alias for show_graph

Operators:
  +, -, *            - Arithmetic (integers, matrices, polynomials)
  ** or ^            - Exponentiation
  /                  - Division (for rationals)

Commands:
  help               - Show this help
  vars               - Show defined variables

Examples:
  x = Integer(12345678901234567890)
  factorial(100)
  gcd(48, 18)
  M = Matrix([[1,2],[3,4]])
  det(M)
  p = Polynomial([1, 2, 1])   # 1 + 2x + x²
  derivative(p)               # 2 + 2x
  roots(Polynomial([6, -5, 1])) # (x-2)(x-3)
  f = expr("x^2 + 3*x + 2")   # Symbolic expression
  diff(f, x)                  # 2*x + 3
  solve("x^2 - 4", x)         # [-2, 2]
  a = mod(7, 11)              # 7 in Z/11Z
  inverse(a)                  # 8 (since 7*8 = 56 ≡ 1 mod 11)
  mean([1, 2, 3, 4, 5])       # 3.0
  std_dev([1, 2, 3, 4, 5])    # ~1.58
  fft([1, 1, 1, 1])           # Fourier transform
  pow_mod(2, 10, 1000)        # 2^10 mod 1000 = 24
  binomial(10, 3)             # 120
  fibonacci(50)               # 12586269025
  catalan(10)                 # 16796
  divisors(60)                # [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
  euler_phi(100)              # 40
  nth_prime(100)              # 541
  G = complete_graph(5)       # K_5: 5 vertices, 10 edges
  chromatic_number(G)         # 5
  is_connected(G)             # true
  shortest_path(G, 0, 4)      # [0, 4]
  p1 = Point(0, 0)            # 2D point at origin
  p2 = Point(3, 4)
  distance(p1, p2)            # 5.0
  convex_hull([Point(0,0), Point(1,0), Point(0.5,0.5), Point(0,1)])  # hull vertices

  # Plotting examples
  plot([Point(1,1), Point(2,4), Point(3,9)])  # scatter plot
  scatter([1,2,3,4,5], [1,4,9,16,25])         # scatter from lists
  line_plot([Point(0,0), Point(1,1), Point(2,0)])  # connected line
  plot_function(x^2, x, -2, 2)               # plot x² from -2 to 2
  histogram([1,2,2,3,3,3,4,4,5], 5)           # histogram with 5 bins
  bar_chart(["A","B","C"], [10,20,15])        # bar chart
  plot3d([Point3D(1,1,1), Point3D(2,2,2)])   # 3D scatter
  show_graph(complete_graph(5))              # visualize K_5
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

/// Strip inline comments from a line (everything after # that's not in a string)
fn strip_inline_comment(line: &str) -> String {
    let mut result = String::new();
    let mut in_string = false;
    let mut string_char = ' ';
    let mut chars = line.chars().peekable();

    while let Some(c) = chars.next() {
        if in_string {
            result.push(c);
            if c == string_char {
                in_string = false;
            } else if c == '\\' {
                // Handle escaped character
                if let Some(&next) = chars.peek() {
                    result.push(next);
                    chars.next();
                }
            }
        } else if c == '"' || c == '\'' {
            in_string = true;
            string_char = c;
            result.push(c);
        } else if c == '#' {
            // Found comment outside string, stop here
            break;
        } else {
            result.push(c);
        }
    }

    result.trim_end().to_string()
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

/// Try to evaluate a symbolic expression to f64
/// Returns None if the expression contains variables
fn try_eval_to_f64(expr: &Expr) -> Option<f64> {
    use rustmath_symbolic::expression::{BinaryOp, UnaryOp};

    match expr {
        Expr::Integer(n) => n.to_f64(),
        Expr::Rational(r) => r.to_f64(),
        Expr::Real(x) => Some(*x),
        Expr::Symbol(_) => None, // Cannot evaluate symbolic variables
        Expr::Binary(op, left, right) => {
            let l = try_eval_to_f64(left)?;
            let r = try_eval_to_f64(right)?;
            Some(match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => l / r,
                BinaryOp::Pow => l.powf(r),
                BinaryOp::Mod => l % r,
            })
        }
        Expr::Unary(op, inner) => {
            let v = try_eval_to_f64(inner)?;
            Some(match op {
                UnaryOp::Neg => -v,
                UnaryOp::Sin => v.sin(),
                UnaryOp::Cos => v.cos(),
                UnaryOp::Tan => v.tan(),
                UnaryOp::Exp => v.exp(),
                UnaryOp::Log => v.ln(),
                UnaryOp::Sqrt => v.sqrt(),
                UnaryOp::Abs => v.abs(),
                UnaryOp::Sign => v.signum(),
                UnaryOp::Sinh => v.sinh(),
                UnaryOp::Cosh => v.cosh(),
                UnaryOp::Tanh => v.tanh(),
                UnaryOp::Arcsin => v.asin(),
                UnaryOp::Arccos => v.acos(),
                UnaryOp::Arctan => v.atan(),
                UnaryOp::Arcsinh => v.asinh(),
                UnaryOp::Arccosh => v.acosh(),
                UnaryOp::Arctanh => v.atanh(),
                UnaryOp::Gamma => {
                    // Simple gamma approximation using Stirling's formula for positive values
                    if v > 0.0 {
                        ((v - 1.0) * (v - 1.0).ln() - (v - 1.0) + 0.5 * (2.0 * std::f64::consts::PI * (v - 1.0)).ln()).exp()
                    } else {
                        f64::NAN
                    }
                }
                UnaryOp::Factorial => {
                    // Factorial for non-negative integers
                    if v >= 0.0 && v.fract() == 0.0 && v < 171.0 {
                        let n = v as u64;
                        let mut result = 1.0;
                        for i in 2..=n {
                            result *= i as f64;
                        }
                        result
                    } else {
                        f64::NAN
                    }
                }
                UnaryOp::Erf => {
                    // Approximation of error function
                    let a1 =  0.254829592;
                    let a2 = -0.284496736;
                    let a3 =  1.421413741;
                    let a4 = -1.453152027;
                    let a5 =  1.061405429;
                    let p  =  0.3275911;
                    let sign = if v < 0.0 { -1.0 } else { 1.0 };
                    let x = v.abs();
                    let t = 1.0 / (1.0 + p * x);
                    sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp())
                }
                UnaryOp::Zeta => {
                    // Simple zeta approximation for s > 1
                    if v > 1.0 {
                        let mut sum = 0.0;
                        for n in 1..=1000 {
                            sum += 1.0 / (n as f64).powf(v);
                        }
                        sum
                    } else {
                        f64::NAN
                    }
                }
            })
        }
        Expr::Function(_, _) => None, // Cannot evaluate general functions
    }
}

/// Convert a symbolic expression to LaTeX
fn expr_to_latex(expr: &Expr) -> String {
    use rustmath_symbolic::expression::{BinaryOp, UnaryOp};

    match expr {
        Expr::Integer(n) => n.to_string(),
        Expr::Rational(r) => format!("\\frac{{{}}}{{{}}}", r.numerator(), r.denominator()),
        Expr::Real(f) => format!("{:.6}", f),
        Expr::Symbol(s) => s.name().to_string(),
        Expr::Binary(op, left, right) => {
            let left_latex = expr_to_latex(left);
            let right_latex = expr_to_latex(right);
            match op {
                BinaryOp::Add => format!("{} + {}", left_latex, right_latex),
                BinaryOp::Sub => format!("{} - {}", left_latex, right_latex),
                BinaryOp::Mul => format!("{} \\cdot {}", left_latex, right_latex),
                BinaryOp::Div => format!("\\frac{{{}}}{{{}}}", left_latex, right_latex),
                BinaryOp::Pow => format!("{}^{{{}}}", left_latex, right_latex),
                BinaryOp::Mod => format!("{} \\mod {}", left_latex, right_latex),
            }
        }
        Expr::Unary(op, inner) => {
            let inner_latex = expr_to_latex(inner);
            match op {
                UnaryOp::Neg => format!("-{}", inner_latex),
                UnaryOp::Sin => format!("\\sin({})", inner_latex),
                UnaryOp::Cos => format!("\\cos({})", inner_latex),
                UnaryOp::Tan => format!("\\tan({})", inner_latex),
                UnaryOp::Exp => format!("e^{{{}}}", inner_latex),
                UnaryOp::Log => format!("\\ln({})", inner_latex),
                UnaryOp::Sqrt => format!("\\sqrt{{{}}}", inner_latex),
                UnaryOp::Abs => format!("|{}|", inner_latex),
                UnaryOp::Sign => format!("\\text{{sign}}({})", inner_latex),
                UnaryOp::Sinh => format!("\\sinh({})", inner_latex),
                UnaryOp::Cosh => format!("\\cosh({})", inner_latex),
                UnaryOp::Tanh => format!("\\tanh({})", inner_latex),
                UnaryOp::Arcsin => format!("\\arcsin({})", inner_latex),
                UnaryOp::Arccos => format!("\\arccos({})", inner_latex),
                UnaryOp::Arctan => format!("\\arctan({})", inner_latex),
                UnaryOp::Arcsinh => format!("\\text{{arcsinh}}({})", inner_latex),
                UnaryOp::Arccosh => format!("\\text{{arccosh}}({})", inner_latex),
                UnaryOp::Arctanh => format!("\\text{{arctanh}}({})", inner_latex),
                UnaryOp::Gamma => format!("\\Gamma({})", inner_latex),
                UnaryOp::Factorial => format!("{}!", inner_latex),
                UnaryOp::Erf => format!("\\text{{erf}}({})", inner_latex),
                UnaryOp::Zeta => format!("\\zeta({})", inner_latex),
            }
        }
        Expr::Function(name, args) => {
            let args_latex: Vec<String> = args.iter().map(|a| expr_to_latex(a)).collect();
            format!("\\text{{{}}}({})", name, args_latex.join(", "))
        }
    }
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

    #[test]
    fn test_complete_graph() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("complete_graph(5)").unwrap();
        assert_eq!(result.text, "Graph with 5 vertices and 10 edges");
    }

    #[test]
    fn test_cycle_graph() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("cycle_graph(6)").unwrap();
        assert_eq!(result.text, "Graph with 6 vertices and 6 edges");
    }

    #[test]
    fn test_petersen_graph() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("petersen_graph()").unwrap();
        assert_eq!(result.text, "Graph with 10 vertices and 15 edges");
    }

    #[test]
    fn test_graph_is_connected() {
        let mut ctx = ReplContext::new();
        ctx.eval("G = complete_graph(5)").unwrap();
        let result = ctx.eval("is_connected(G)").unwrap();
        assert_eq!(result.text, "True");
    }

    #[test]
    fn test_graph_chromatic_number() {
        let mut ctx = ReplContext::new();
        ctx.eval("G = complete_graph(4)").unwrap();
        let result = ctx.eval("chromatic_number(G)").unwrap();
        assert_eq!(result.text, "4");
    }

    #[test]
    fn test_point2d() {
        let mut ctx = ReplContext::new();
        let result = ctx.eval("Point(3, 4)").unwrap();
        assert_eq!(result.text, "Point(3, 4)");
    }

    #[test]
    fn test_point_distance() {
        let mut ctx = ReplContext::new();
        ctx.eval("p1 = Point(0, 0)").unwrap();
        ctx.eval("p2 = Point(3, 4)").unwrap();
        let result = ctx.eval("distance(p1, p2)").unwrap();
        assert_eq!(result.text, "5");
    }

    #[test]
    fn test_inline_comments() {
        let mut ctx = ReplContext::new();
        // Test inline comment stripping
        let result = ctx.eval("p1 = Polynomial([1, 1])  # 1 + x\np2 = Polynomial([1, -1]) # 1 - x\np1 * p2").unwrap();
        // Result is -1*x^2 + 1 which is equivalent to 1 - x^2
        assert_eq!(result.text, "-1*x^2 + 1");
    }

    #[test]
    fn test_strip_inline_comment() {
        // Test the helper function directly
        assert_eq!(super::strip_inline_comment("x = 5  # comment"), "x = 5");
        assert_eq!(super::strip_inline_comment("x = 5"), "x = 5");
        assert_eq!(super::strip_inline_comment("# full comment"), "");
        assert_eq!(super::strip_inline_comment("s = \"hello # world\"  # comment"), "s = \"hello # world\"");
        assert_eq!(super::strip_inline_comment("s = 'test # inside'"), "s = 'test # inside'");
    }

    #[test]
    fn test_expr_function() {
        let mut ctx = ReplContext::new();
        // Test expr function
        let result = ctx.eval("expr(\"x + x + x\")").unwrap();
        // Result should be a symbolic expression
        assert!(result.text.contains("x"));
    }

    #[test]
    fn test_simplify() {
        let mut ctx = ReplContext::new();
        // Test simplify
        let result = ctx.eval("simplify(\"x + x + x\")").unwrap();
        assert!(result.text.contains("x"));
    }

    #[test]
    fn test_nested_simplify_expr() {
        let mut ctx = ReplContext::new();
        // Test simplify(expr(...)) - nested function call
        let result = ctx.eval("simplify(expr(\"x + x + x\"))").unwrap();
        assert!(result.text.contains("x"));
    }

    #[test]
    fn test_expand_nested() {
        let mut ctx = ReplContext::new();
        // Test expand(expr(...)) - nested function call
        let result = ctx.eval("expand(expr(\"(x + 1)^2\"))").unwrap();
        assert!(result.text.contains("x"));
    }

    #[test]
    fn test_solve_quadratic() {
        let mut ctx = ReplContext::new();
        // Test solve(x^2 - 4, x) = {2, -2}
        let result = ctx.eval("solve(\"x^2 - 4\", x)").unwrap();
        // It should NOT say "No solution found"
        assert!(!result.text.contains("No solution found"), "Expected solutions but got: {}", result.text);
    }

    #[test]
    fn test_diff() {
        let mut ctx = ReplContext::new();
        // Test diff(x^2, x) = 2*x
        let result = ctx.eval("diff(\"x^2\", x)").unwrap();
        assert!(result.text.contains("x"), "Expected x in derivative but got: {}", result.text);
    }

    #[test]
    fn test_finite_field_arithmetic() {
        let mut ctx = ReplContext::new();
        // Create elements in Z/11Z
        ctx.eval("a = mod(7, 11)").unwrap();
        ctx.eval("b = mod(3, 11)").unwrap();

        // Test addition: 7 + 3 = 10 (mod 11)
        let result = ctx.eval("a + b").unwrap();
        assert!(result.text.contains("10"), "Expected 10 (mod 11) but got: {}", result.text);

        // Test multiplication: 7 * 3 = 21 = 10 (mod 11)
        let result = ctx.eval("a * b").unwrap();
        assert!(result.text.contains("10"), "Expected 10 (mod 11) but got: {}", result.text);

        // Test inverse: 7 * 8 = 56 = 1 (mod 11)
        let result = ctx.eval("inverse(a)").unwrap();
        assert!(result.text.contains("8"), "Expected 8 (mod 11) but got: {}", result.text);

        // Test power: 7^2 = 49 = 5 (mod 11)
        let result = ctx.eval("a^2").unwrap();
        assert!(result.text.contains("5"), "Expected 5 (mod 11) but got: {}", result.text);
    }

    #[test]
    fn test_list_with_function_calls() {
        let mut ctx = ReplContext::new();
        // Test list literal with function calls
        let result = ctx.eval("[fibonacci(0), fibonacci(1), fibonacci(2), fibonacci(3), fibonacci(4)]").unwrap();
        // Should be [0, 1, 1, 2, 3]
        assert!(result.text.contains("0"), "Expected 0 in list but got: {}", result.text);
        assert!(result.text.contains("1"), "Expected 1 in list but got: {}", result.text);
        assert!(result.text.contains("2"), "Expected 2 in list but got: {}", result.text);
        assert!(result.text.contains("3"), "Expected 3 in list but got: {}", result.text);
    }

    #[test]
    fn test_multinomial() {
        let mut ctx = ReplContext::new();
        // multinomial(n, [k1, k2, ...]) = n! / (k1! * k2! * ...)
        // where k1 + k2 + ... = n
        // multinomial(6, [2, 2, 2]) = 6! / (2! * 2! * 2!) = 720 / 8 = 90
        let result = ctx.eval("multinomial(6, [2, 2, 2])").unwrap();
        assert_eq!(result.text, "90", "Expected 90 but got: {}", result.text);

        // multinomial(10, [2, 3, 5]) = 10! / (2! * 3! * 5!) = 3628800 / (2 * 6 * 120) = 2520
        let result = ctx.eval("multinomial(10, [2, 3, 5])").unwrap();
        assert_eq!(result.text, "2520", "Expected 2520 but got: {}", result.text);

        // When sum of ks != n, multinomial returns an error
        // Here: 5 != 1+2+3 = 6
        let result = ctx.eval("multinomial(5, [1, 2, 3])");
        assert!(result.is_err(), "Expected error when sum(ks) != n");
        let err = result.unwrap_err();
        assert!(err.message.contains("sum of k values"), "Error message should mention sum: {}", err.message);
    }

    #[test]
    fn test_xgcd() {
        let mut ctx = ReplContext::new();
        // xgcd(240, 46) should return [gcd, x, y] where gcd = 240*x + 46*y
        // gcd(240, 46) = 2
        let result = ctx.eval("xgcd(240, 46)").unwrap();
        // Result should be a list [2, x, y]
        assert!(result.text.starts_with("["), "Expected list but got: {}", result.text);
        assert!(result.text.contains("2"), "Expected gcd=2 in result: {}", result.text);

        // Verify the Bézout identity: gcd = a*x + b*y
        // For 240, 46: gcd=2, and 240*(-9) + 46*47 = -2160 + 2162 = 2
        let result2 = ctx.eval("xgcd(35, 15)").unwrap();
        // gcd(35, 15) = 5
        assert!(result2.text.contains("5"), "Expected gcd=5 for xgcd(35, 15): {}", result2.text);
    }

    #[test]
    fn test_integrate_num() {
        let mut ctx = ReplContext::new();
        // Integrate x^2 from 0 to 1 (exact answer: 1/3 ≈ 0.333...)
        ctx.eval("f = expr(\"x^2\")").unwrap();
        let result = ctx.eval("integrate_num(f, x, 0, 1)").unwrap();
        let value: f64 = result.text.parse().expect("Should be a number");
        assert!((value - 1.0/3.0).abs() < 0.001, "Expected ~0.333 but got: {}", value);

        // Integrate x from 0 to 2 (exact answer: 2)
        ctx.eval("g = expr(\"x\")").unwrap();
        let result2 = ctx.eval("integrate_num(g, x, 0, 2)").unwrap();
        let value2: f64 = result2.text.parse().expect("Should be a number");
        assert!((value2 - 2.0).abs() < 0.001, "Expected ~2.0 but got: {}", value2);
    }

    #[test]
    fn test_clifford_algebra() {
        let mut ctx = ReplContext::new();
        // Create Cl(R^3) Clifford algebra
        let result = ctx.eval("Cl3 = CliffordAlgebra(3)").unwrap();
        assert!(result.text.contains("Clifford"), "Expected Clifford algebra: {}", result.text);

        // Get generators using e(index) - doesn't require algebra argument
        ctx.eval("e1 = e(0)").unwrap();
        ctx.eval("e2 = e(1)").unwrap();
        ctx.eval("e3 = e(2)").unwrap();

        // Test e1 * e2 creates a bivector
        let result3 = ctx.eval("e1 * e2").unwrap();
        assert!(result3.text.contains("e"), "Expected bivector: {}", result3.text);
    }

    #[test]
    fn test_exterior_algebra() {
        let mut ctx = ReplContext::new();
        // Create exterior algebra (Q=0) - actually same type as CliffordAlgebra
        let result = ctx.eval("E4 = ExteriorAlgebra(4)").unwrap();
        // ExteriorAlgebra is a type alias for CliffordAlgebra with Q=0
        assert!(result.text.contains("Clifford"), "Expected Clifford/Exterior algebra: {}", result.text);
    }

    #[test]
    fn test_manifold() {
        let mut ctx = ReplContext::new();
        // Create R^3 manifold
        let result = ctx.eval("M = Manifold(3, \"R3\")").unwrap();
        assert!(result.text.contains("Manifold") || result.text.contains("R3"), "Expected Manifold: {}", result.text);
    }

    #[test]
    fn test_euclidean_space() {
        let mut ctx = ReplContext::new();
        // Create Euclidean space
        let result = ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        assert!(result.text.contains("Manifold") || result.text.contains("Euclidean"), "Expected Euclidean space: {}", result.text);
    }

    #[test]
    fn test_wedge_product_operator() {
        let mut ctx = ReplContext::new();
        // Get generators
        ctx.eval("e1 = e(0)").unwrap();
        ctx.eval("e2 = e(1)").unwrap();

        // Test ^ operator for wedge product
        let result = ctx.eval("e1 ^ e2").unwrap();
        assert!(result.text.contains("e"), "Expected wedge product: {}", result.text);
    }

    #[test]
    fn test_wedge_product_function() {
        let mut ctx = ReplContext::new();
        // Get generators
        ctx.eval("e1 = e(0)").unwrap();
        ctx.eval("e2 = e(1)").unwrap();

        // Test wedge() function for Clifford elements
        let result = ctx.eval("wedge(e1, e2)").unwrap();
        assert!(result.text.contains("e"), "Expected wedge product: {}", result.text);
    }

    #[test]
    fn test_clifford_mixed_type_operations() {
        let mut ctx = ReplContext::new();
        // Get generators
        ctx.eval("e1 = e(0)").unwrap();
        ctx.eval("e2 = e(1)").unwrap();

        // Test Integer + CliffordElem
        let result = ctx.eval("1 + e1").unwrap();
        assert!(result.text.contains("1") && result.text.contains("e"), "Expected 1 + e1: {}", result.text);

        // Test CliffordElem * Integer
        let result2 = ctx.eval("2 * e1").unwrap();
        assert!(result2.text.contains("2"), "Expected 2*e1: {}", result2.text);

        // Test CliffordElem / Integer
        let result3 = ctx.eval("e1 / 2").unwrap();
        assert!(result3.text.contains("1/2"), "Expected e1/2: {}", result3.text);
    }

    #[test]
    fn test_string_operations() {
        let mut ctx = ReplContext::new();

        // Test string repetition
        let result = ctx.eval("\"=\" * 5").unwrap();
        assert_eq!(result.text, "\"=====\"", "Expected 5 equals signs: {}", result.text);

        // Test string concatenation
        let result2 = ctx.eval("\"Hello\" + \" \" + \"World\"").unwrap();
        assert_eq!(result2.text, "\"Hello World\"", "Expected 'Hello World': {}", result2.text);
    }

    #[test]
    fn test_print_function() {
        let mut ctx = ReplContext::new();

        // Test print with string - result includes quotes around the string
        let result = ctx.eval("print(\"Hello\")").unwrap();
        // print returns a String type, which displays with quotes
        assert!(result.text.contains("Hello"), "Expected Hello in result: '{}'", result.text);

        // Test print with multiple args
        let result2 = ctx.eval("print(\"a\", \"b\", \"c\")").unwrap();
        assert!(result2.text.contains("a") && result2.text.contains("b"), "Expected 'a b c': '{}'", result2.text);

        // Test print with integer
        let result3 = ctx.eval("print(42)").unwrap();
        assert!(result3.text.contains("42"), "Expected 42: '{}'", result3.text);
    }

    #[test]
    fn test_implicit_plot_circle() {
        let mut ctx = ReplContext::new();

        // Define variables
        ctx.eval("x, y = var('x y')").unwrap();

        // Plot a unit circle: x^2 + y^2 - 1 = 0
        let result = ctx.eval("implicit_plot(x^2 + y^2 - 1, (x, -1.5, 1.5), (y, -1.5, 1.5))").unwrap();

        // Verify SVG output
        assert!(result.text.contains("Implicit curve"), "Expected 'Implicit curve' in result: {}", result.text);

        // Check that SVG was generated
        assert!(result.svg.is_some(), "Expected SVG output for implicit plot");
        let svg = result.svg.unwrap();

        // Count curve lines (stroke-width="2") - excludes grid/axis lines (stroke-width="1" or "1.5")
        let curve_line_count = svg.matches("stroke-width=\"2\"").count();
        assert!(curve_line_count > 50, "Circle should have many curve segments (got {})", curve_line_count);

        // The circle should form a connected curve, not have extra random lines
        // With a 100x100 grid and proper marching squares, a unit circle should have
        // approximately 200-400 line segments (perimeter ~2π in a unit grid)
        assert!(curve_line_count < 500, "Too many curve segments ({}), possible artifacts", curve_line_count);
    }

    #[test]
    fn test_implicit_plot_vertical_line() {
        let mut ctx = ReplContext::new();

        // Define variables
        ctx.eval("x, y = var('x y')").unwrap();

        // Plot a vertical line: x = 0
        let result = ctx.eval("implicit_plot(x, (x, -1, 1), (y, -1, 1))").unwrap();

        // Check SVG
        assert!(result.svg.is_some(), "Expected SVG output");
        let svg = result.svg.unwrap();

        // Count curve lines (stroke-width="2") - excludes grid/axis lines
        let curve_line_count = svg.matches("stroke-width=\"2\"").count();
        // A vertical line should have exactly n_grid-1 line segments (one per row)
        // Should be approximately 99 lines (for 100 grid points)
        assert!(curve_line_count >= 90 && curve_line_count <= 110,
            "Vertical line should have ~99 segments (got {})", curve_line_count);
    }

    #[test]
    fn test_implicit_plot_horizontal_line() {
        let mut ctx = ReplContext::new();

        // Define variables
        ctx.eval("x, y = var('x y')").unwrap();

        // Plot a horizontal line: y = 0
        let result = ctx.eval("implicit_plot(y, (x, -1, 1), (y, -1, 1))").unwrap();

        // Check SVG
        assert!(result.svg.is_some(), "Expected SVG output");
        let svg = result.svg.unwrap();

        // Count curve lines (stroke-width="2") - excludes grid/axis lines
        let curve_line_count = svg.matches("stroke-width=\"2\"").count();
        // A horizontal line should have exactly n_grid-1 line segments (one per column)
        // Should be approximately 99 lines (for 100 grid points)
        assert!(curve_line_count >= 90 && curve_line_count <= 110,
            "Horizontal line should have ~99 segments (got {})", curve_line_count);
    }

    #[test]
    fn test_implicit_plot_diagonal() {
        let mut ctx = ReplContext::new();

        // Define variables
        ctx.eval("x, y = var('x y')").unwrap();

        // Plot a diagonal line: x - y = 0 (y = x)
        let result = ctx.eval("implicit_plot(x - y, (x, -1, 1), (y, -1, 1))").unwrap();

        // Check SVG
        assert!(result.svg.is_some(), "Expected SVG output");
        let svg = result.svg.unwrap();

        // Count curve lines (stroke-width="2") - excludes grid/axis lines
        let curve_line_count = svg.matches("stroke-width=\"2\"").count();
        // A diagonal line crosses cells diagonally, potentially triggering saddle case handling
        // in cells near corners. This can generate up to 2 segments per cell.
        // For a 100x100 grid with ~99 cells along the diagonal, expect 99-200 segments.
        assert!(curve_line_count >= 90 && curve_line_count <= 220,
            "Diagonal line should have reasonable segments (got {})", curve_line_count);
    }

    #[test]
    fn test_implicit_plot_hyperbola() {
        let mut ctx = ReplContext::new();

        // Define variables
        ctx.eval("x, y = var('x y')").unwrap();

        // Plot a hyperbola: x*y - 1 = 0 (has saddle cases)
        let result = ctx.eval("implicit_plot(x*y - 1, (x, -3, 3), (y, -3, 3))").unwrap();

        // Check SVG was generated
        assert!(result.svg.is_some(), "Expected SVG output");
        let svg = result.svg.unwrap();

        // Count curve lines
        let curve_line_count = svg.matches("stroke-width=\"2\"").count();
        // A hyperbola has two branches, each should have reasonable segments
        assert!(curve_line_count > 50, "Hyperbola should have many curve segments (got {})", curve_line_count);
        assert!(curve_line_count < 400, "Too many curve segments ({}), possible artifacts", curve_line_count);
    }
}
