//! RustMath REPL evaluation engine
//!
//! Parses and evaluates RustMath expressions in an interactive context.

use std::collections::HashMap;
use std::ops::RangeInclusive;
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
    catalan, fibonacci_signed, lucas_signed, bell_number, stirling_first, stirling_second,
    multinomial, falling_factorial, rising_factorial, eulerian, narayana, motzkin,
    delannoy, delannoy_central, schroder_large, schroder_small, partition_count,
    count_derangements,
};
use rustmath_integers::{
    prime::{
        is_probable_prime, nth_prime, prime_pi, try_is_prime, try_next_prime,
        try_prime_range as try_primes_between,
    },
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
    JordanAlgebraSymmetricBilinear, SymmetricBilinearElement,
    SpecialJordanAlgebra, SpecialJordanElement,
    ExceptionalJordanAlgebra, AlbertElement,
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
    /// Jordan algebra from symmetric bilinear form
    JordanSymBilinear(JordanAlgebraSymmetricBilinear<Rational>),
    /// Element of Jordan algebra from symmetric bilinear form
    JordanSymBilinearElem(SymmetricBilinearElement<Rational>),
    /// Special Jordan algebra (from matrices)
    SpecialJordan(SpecialJordanAlgebra<Rational>),
    /// Element of special Jordan algebra
    SpecialJordanElem(SpecialJordanElement<Rational>),
    /// Exceptional Jordan algebra (Albert algebra)
    ExceptionalJordan(ExceptionalJordanAlgebra<Rational>),
    /// Element of exceptional Jordan algebra
    AlbertElem(AlbertElement<Rational>),
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

/// Options for 3D plotting - matches SageMath defaults
#[derive(Clone, Debug)]
pub struct Plot3DOptions {
    /// Surface color (default: SageMath blue #6666ff)
    pub color: String,
    /// Surface opacity (default: 1.0)
    pub opacity: f64,
    /// Show mesh grid lines (default: false, like SageMath)
    pub mesh: bool,
    /// Mesh line color
    pub mesh_color: String,
    /// Number of plot points per axis
    pub plot_points: usize,
    /// Enable interactive mode with zoom/rotate
    pub interactive: bool,
    /// Background color
    pub background: String,
    /// Show axes
    pub axes: bool,
    /// Enable lighting/shading
    pub shading: bool,
}

impl Default for Plot3DOptions {
    fn default() -> Self {
        Self {
            color: "#6666ff".to_string(),  // SageMath default blue
            opacity: 1.0,
            mesh: false,  // SageMath default: no mesh
            mesh_color: "#333333".to_string(),
            plot_points: 40,
            interactive: true,  // Enable zoom/rotate by default
            background: "#ffffff".to_string(),
            axes: true,
            shading: true,  // Enable realistic shading
        }
    }
}

impl Plot3DOptions {
    /// Parse options from keyword arguments like "color='red', mesh=True, opacity=0.8"
    pub fn parse_from_args(args: &[&str]) -> Self {
        let mut opts = Self::default();

        for arg in args {
            let arg = arg.trim();
            if let Some(eq_pos) = arg.find('=') {
                let key = arg[..eq_pos].trim().to_lowercase();
                let value = arg[eq_pos + 1..].trim();

                match key.as_str() {
                    "color" | "rgbcolor" => {
                        // Remove quotes if present
                        let color = value.trim_matches(|c| c == '\'' || c == '"');
                        opts.color = Self::parse_color(color);
                    }
                    "opacity" => {
                        if let Ok(o) = value.parse::<f64>() {
                            opts.opacity = o.clamp(0.0, 1.0);
                        }
                    }
                    "mesh" => {
                        opts.mesh = value == "True" || value == "true" || value == "1";
                    }
                    "plot_points" => {
                        if let Ok(n) = value.parse::<usize>() {
                            opts.plot_points = n.clamp(10, 100);
                        }
                    }
                    "interactive" => {
                        opts.interactive = value == "True" || value == "true" || value == "1";
                    }
                    "axes" => {
                        opts.axes = value == "True" || value == "true" || value == "1";
                    }
                    "shading" => {
                        opts.shading = value == "True" || value == "true" || value == "1";
                    }
                    _ => {}
                }
            }
        }

        opts
    }

    /// Parse color names to hex codes
    fn parse_color(color: &str) -> String {
        match color.to_lowercase().as_str() {
            "blue" => "#6666ff".to_string(),
            "red" => "#ff6666".to_string(),
            "green" => "#66ff66".to_string(),
            "yellow" => "#ffff66".to_string(),
            "orange" => "#ff9933".to_string(),
            "purple" => "#9966ff".to_string(),
            "cyan" => "#66ffff".to_string(),
            "magenta" => "#ff66ff".to_string(),
            "white" => "#ffffff".to_string(),
            "black" => "#333333".to_string(),
            "gray" | "grey" => "#999999".to_string(),
            _ if color.starts_with('#') => color.to_string(),
            _ if color.starts_with("0x") => format!("#{}", &color[2..]),
            _ => "#6666ff".to_string(), // Default to SageMath blue
        }
    }
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
                let mut result = EvalResult::text(description.clone());
                // Check if SVG contains script (interactive) - send as HTML to preserve JS
                if svg.contains("<script") {
                    // Wrap in a container with unique ID for reliable JS targeting
                    use std::time::{SystemTime, UNIX_EPOCH};
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0);
                    let container_id = format!("rustmath-plot-{}", timestamp);

                    // Replace document.currentScript.parentElement with container-based lookup
                    // Match with 4-space indentation as used in SVG templates
                    let modified_svg = svg
                        .replace(
                            "    var svg = document.currentScript.parentElement;",
                            &format!(
                                "    var container = document.getElementById('{}'); var svg = container ? container.querySelector('svg') : null; if (!svg) return;",
                                container_id
                            )
                        );

                    let html = format!(
                        r#"<div id="{}">{}</div>"#,
                        container_id, modified_svg
                    );
                    result.html = Some(html);
                } else {
                    result.svg = Some(svg.clone());
                }
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
            RustMathValue::JordanSymBilinear(j) => {
                let text = format!("JordanAlgebra(dim={}, type=SymmetricBilinear)", j.dimension());
                let latex = format!("$J(\\mathbb{{Q}}^{})$", j.dimension());
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::JordanSymBilinearElem(elem) => {
                let text = format!("{}", elem);
                EvalResult::text(text)
            }
            RustMathValue::SpecialJordan(j) => {
                let text = format!("SpecialJordanAlgebra(matrix_size={})", j.matrix_size());
                let latex = format!("$J(M_{}(\\mathbb{{Q}}))$", j.matrix_size());
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::SpecialJordanElem(elem) => {
                let text = format!("{}", elem);
                EvalResult::text(text)
            }
            RustMathValue::ExceptionalJordan(_j) => {
                let text = "AlbertAlgebra(dim=27)".to_string();
                let latex = "$\\mathfrak{A}$ (Albert algebra)".to_string();
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::AlbertElem(elem) => {
                let text = format!("{}", elem);
                EvalResult::text(text)
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

        // Try to parse as float (e.g., 0.7, 3.14, 1.5e-3)
        if expr.contains('.') || expr.contains('e') || expr.contains('E') {
            if let Ok(f) = expr.parse::<f64>() {
                return Ok(RustMathValue::Float(f));
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

        // Binary operations - check BEFORE function calls to handle e(0)*e(1) correctly
        if let Some(result) = self.try_eval_binary_op(expr)? {
            return Ok(result);
        }

        // Function calls - only if the closing ) matches the opening ( after the function name
        if expr.contains('(') && expr.ends_with(')') {
            // Find the first '(' and verify the last ')' matches it (not a nested one)
            if let Some(open_pos) = expr.find('(') {
                let func_name = &expr[..open_pos];
                if is_valid_identifier(func_name.trim()) {
                    // Count parentheses to verify the last ')' matches the first '('
                    let args_part = &expr[open_pos..];
                    let mut depth = 0;
                    let mut valid = true;
                    for (i, ch) in args_part.chars().enumerate() {
                        match ch {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                // If depth goes to 0 before the end, it's not a simple function call
                                if depth == 0 && i != args_part.len() - 1 {
                                    valid = false;
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                    if valid && depth == 0 {
                        return self.eval_function_call(expr);
                    }
                }
            }
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
                        let n_val = int_arg::<u32>("factorial", "n", &n, 0..=repl_limits::FACTORIAL_N, "REPL limit")?;
                        let result = Integer::factorial(n_val);
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
                        let result = checked_is_prime("is_prime", &n)?;
                        Ok(RustMathValue::Bool(result))
                    }
                    _ => Err(EvalError::new("TypeError", "is_prime requires an integer")),
                }
            }

            // Probable-prime test (Baillie-PSW; MAGMA IsProbablePrime): exact below psi_12
            "is_probable_prime" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => Ok(RustMathValue::Bool(is_probable_prime(&n))),
                    _ => Err(EvalError::new("TypeError", "is_probable_prime requires an integer")),
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
                // `Integer::to_f64` rounds correctly (num-bigint) and gives ±inf above 2^1024;
                // up to 2^63 it is the same value as the old conversion through i64.
                let part = |v: &RustMathValue, name: &str| -> Result<f64, EvalError> {
                    match v {
                        RustMathValue::Integer(n) => match n.to_f64() {
                            Some(x) if x.is_finite() => Ok(x),
                            _ => Err(EvalError::new("ValueError", format!(
                                "Complex: {} = {} does not fit in a double", name, show_int(n)))),
                        },
                        _ => Err(EvalError::new("TypeError", "Complex requires numeric arguments")),
                    }
                };
                let re = part(&args[0], "re")?;
                let im = part(&args[1], "im")?;
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
                        let exp_val = int_arg::<u32>("pow", "exponent", exp, 0..=repl_limits::POW_EXP, "REPL limit")?;
                        Ok(RustMathValue::Integer(base.pow(exp_val)))
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
                        let size = int_arg::<usize>("identity", "n", &n, 0..=repl_limits::MATRIX_DIM, "REPL limit")?;
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
                        let rows = int_arg::<usize>("zeros", "rows", r, 0..=repl_limits::MATRIX_DIM, "REPL limit")?;
                        let cols = int_arg::<usize>("zeros", "cols", c, 0..=repl_limits::MATRIX_DIM, "REPL limit")?;
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

            // Matrix rank: rank(M)
            "rank" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        // Convert to Rational for rank computation
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let r = m_rat.rank()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Integer(Integer::from(r as i64)))
                    }
                    _ => Err(EvalError::new("TypeError", "rank requires a matrix")),
                }
            }

            // Matrix inverse: inverse(M)
            "matrix_inverse" | "inv" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        match m_rat.inverse() {
                            Ok(Some(inv)) => {
                                // Format as string showing rational entries
                                let mut result_rows = Vec::new();
                                for i in 0..inv.rows() {
                                    let mut row = Vec::new();
                                    for j in 0..inv.cols() {
                                        if let Ok(val) = inv.get(i, j) {
                                            row.push(format!("{}", val));
                                        }
                                    }
                                    result_rows.push(format!("[{}]", row.join(", ")));
                                }
                                Ok(RustMathValue::String(format!("[{}]", result_rows.join(", "))))
                            }
                            Ok(None) => Err(EvalError::new("ValueError", "Matrix is singular (not invertible)")),
                            Err(e) => Err(EvalError::new("MatrixError", format!("{:?}", e))),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "inverse requires a matrix")),
                }
            }

            // Row echelon form: rref(M) or echelon(M)
            "rref" | "echelon" | "reduced_row_echelon" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let rref = m_rat.reduced_row_echelon_form()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        // Format the matrix
                        let mut result_rows = Vec::new();
                        for i in 0..rref.matrix.rows() {
                            let mut row = Vec::new();
                            for j in 0..rref.matrix.cols() {
                                if let Ok(val) = rref.matrix.get(i, j) {
                                    row.push(format!("{}", val));
                                }
                            }
                            result_rows.push(format!("[{}]", row.join(", ")));
                        }
                        Ok(RustMathValue::String(format!("[{}]", result_rows.join(", "))))
                    }
                    _ => Err(EvalError::new("TypeError", "rref requires a matrix")),
                }
            }

            // Kernel (null space): kernel(M) or nullspace(M)
            "kernel" | "nullspace" | "null_space" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let kern = m_rat.kernel()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        // Format as list of vectors
                        let vectors: Vec<String> = kern.iter()
                            .map(|v| format!("[{}]", v.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(", ")))
                            .collect();
                        Ok(RustMathValue::String(format!("[{}]", vectors.join(", "))))
                    }
                    _ => Err(EvalError::new("TypeError", "kernel requires a matrix")),
                }
            }

            // Image (column space): image(M) or columnspace(M)
            "image" | "columnspace" | "column_space" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let img = m_rat.image()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let vectors: Vec<String> = img.iter()
                            .map(|v| format!("[{}]", v.iter().map(|x| format!("{}", x)).collect::<Vec<_>>().join(", ")))
                            .collect();
                        Ok(RustMathValue::String(format!("[{}]", vectors.join(", "))))
                    }
                    _ => Err(EvalError::new("TypeError", "image requires a matrix")),
                }
            }

            // LU decomposition: lu(M)
            "lu" | "lu_decomposition" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let lu = m_rat.lu_decomposition()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::String(format!("L:\n{}\nU:\n{}",
                            self.format_rational_matrix(&lu.l),
                            self.format_rational_matrix(&lu.u))))
                    }
                    _ => Err(EvalError::new("TypeError", "lu requires a matrix")),
                }
            }

            // PLU decomposition: plu(M)
            "plu" | "plu_decomposition" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let plu = m_rat.plu_decomposition()
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::String(format!("P:\n{}\nL:\n{}\nU:\n{}",
                            self.format_rational_matrix(&plu.p),
                            self.format_rational_matrix(&plu.l),
                            self.format_rational_matrix(&plu.u))))
                    }
                    _ => Err(EvalError::new("TypeError", "plu requires a matrix")),
                }
            }

            // Solve linear system: solve_linear(A, b)
            "solve_linear" | "linsolve" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "solve_linear requires 2 arguments (matrix, vector)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Matrix(m), RustMathValue::List(b_list)) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;

                        let b: Vec<Rational> = b_list.iter()
                            .map(|v| match v {
                                RustMathValue::Integer(n) => Ok(Rational::from_integer(n.clone())),
                                RustMathValue::Rational(r) => Ok(r.clone()),
                                _ => Err(EvalError::new("TypeError", "vector elements must be numbers")),
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        match m_rat.solve(&b) {
                            Ok(Some(solution)) => {
                                let sol_str: Vec<String> = solution.iter()
                                    .map(|x| format!("{}", x))
                                    .collect();
                                let sol = format!("[{}]", sol_str.join(", "));
                                // `solve` returns one particular solution; it is unique iff
                                // rank A = cols.
                                let rank = m_rat.rank()
                                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                                if rank < cols {
                                    Ok(RustMathValue::String(format!(
                                        "{} (particular solution, not unique: add any kernel vector; kernel dimension {})",
                                        sol,
                                        cols - rank
                                    )))
                                } else {
                                    Ok(RustMathValue::String(sol))
                                }
                            }
                            Ok(None) => Err(EvalError::new("ValueError", "No solution exists")),
                            Err(e) => Err(EvalError::new("MatrixError", format!("{:?}", e))),
                        }
                    }
                    _ => Err(EvalError::new("TypeError", "solve_linear requires (matrix, list)")),
                }
            }

            // Characteristic polynomial: charpoly(M)
            "charpoly" | "characteristic_polynomial" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let rows = m.rows();
                        let cols = m.cols();
                        let mut data: Vec<Rational> = Vec::with_capacity(rows * cols);
                        for i in 0..rows {
                            for j in 0..cols {
                                if let Ok(val) = m.get(i, j) {
                                    data.push(Rational::from_integer(val.clone()));
                                }
                            }
                        }
                        let m_rat: Matrix<Rational> = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        use rustmath_matrix::characteristic_polynomial;
                        let poly = characteristic_polynomial(&m_rat)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::String(format!("{}", poly)))
                    }
                    _ => Err(EvalError::new("TypeError", "charpoly requires a matrix")),
                }
            }

            // Diagonal matrix: diagonal([a, b, c, ...])
            "diagonal" | "diag" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::List(entries) => {
                        let n = entries.len();
                        let mut data = vec![Integer::from(0); n * n];
                        for (i, entry) in entries.iter().enumerate() {
                            match entry {
                                RustMathValue::Integer(v) => data[i * n + i] = v.clone(),
                                _ => return Err(EvalError::new("TypeError", "diagonal entries must be integers")),
                            }
                        }
                        let m = Matrix::from_vec(n, n, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Matrix(m))
                    }
                    _ => Err(EvalError::new("TypeError", "diagonal requires a list")),
                }
            }

            // Ones matrix: ones(m, n)
            "ones" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "ones requires 2 arguments (rows, cols)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(r), RustMathValue::Integer(c)) => {
                        let rows = int_arg::<usize>("ones", "rows", r, 0..=repl_limits::ONES_DIM, "REPL limit")?;
                        let cols = int_arg::<usize>("ones", "cols", c, 0..=repl_limits::ONES_DIM, "REPL limit")?;
                        let data = vec![Integer::from(1); rows * cols];
                        let m = Matrix::from_vec(rows, cols, data)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Matrix(m))
                    }
                    _ => Err(EvalError::new("TypeError", "ones requires integer dimensions")),
                }
            }

            // Matrix power: matrix_power(M, n)
            "matrix_power" | "mpow" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "matrix_power requires 2 arguments (matrix, exponent)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Matrix(m), RustMathValue::Integer(n)) => {
                        // The domain of Matrix::pow(u32). No REPL limit: the cost depends on
                        // how fast the entries grow, which the exponent alone cannot bound.
                        let exp = int_arg::<u32>("matrix_power", "exponent", n, 0..=u32::MAX, "")?;
                        let result = m.pow(exp)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        Ok(RustMathValue::Matrix(result))
                    }
                    _ => Err(EvalError::new("TypeError", "matrix_power requires (matrix, integer)")),
                }
            }

            // Frobenius norm: norm(M) or frobenius_norm(M)
            "norm" | "frobenius_norm" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        let n = m.frobenius_norm();
                        Ok(RustMathValue::Float(n))
                    }
                    _ => Err(EvalError::new("TypeError", "norm requires a matrix")),
                }
            }

            // Check if upper triangular: is_upper_triangular(M)
            "is_upper_triangular" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Bool(m.is_upper_triangular()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_upper_triangular requires a matrix")),
                }
            }

            // Check if lower triangular: is_lower_triangular(M)
            "is_lower_triangular" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Matrix(m) => {
                        Ok(RustMathValue::Bool(m.is_lower_triangular()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_lower_triangular requires a matrix")),
                }
            }

            // Get row: row(M, i)
            "row" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "row requires 2 arguments (matrix, index)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Matrix(m), RustMathValue::Integer(i)) => {
                        let idx = index_arg("row", "i", i, m.rows(), "rows")?;
                        let r = m.row(idx)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let list: Vec<RustMathValue> = r.into_iter()
                            .map(RustMathValue::Integer)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "row requires (matrix, integer)")),
                }
            }

            // Get column: col(M, j)
            "col" | "column" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "col requires 2 arguments (matrix, index)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Matrix(m), RustMathValue::Integer(j)) => {
                        let idx = index_arg("col", "j", j, m.cols(), "columns")?;
                        let c = m.col(idx)
                            .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                        let list: Vec<RustMathValue> = c.into_iter()
                            .map(RustMathValue::Integer)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "col requires (matrix, integer)")),
                }
            }

            // Scalar multiplication: scalar_mul(M, k)
            "scalar_mul" | "scale" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "scalar_mul requires 2 arguments (matrix, scalar)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Matrix(m), RustMathValue::Integer(k)) => {
                        let result = m.scalar_mul(k);
                        Ok(RustMathValue::Matrix(result))
                    }
                    _ => Err(EvalError::new("TypeError", "scalar_mul requires (matrix, integer)")),
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
                // `simplify` folds the parser's Neg(Integer(1)) to Integer(-1), so a negative
                // literal order reaches the range check instead of silently becoming 5.
                let order = if args.len() == 4 {
                    match args[3].simplify() {
                        Expr::Integer(n) => int_arg::<usize>("taylor", "order", &n, 0..=repl_limits::TAYLOR_ORDER, "REPL limit")?,
                        other => return Err(EvalError::new("TypeError", format!(
                            "taylor: order must be an integer, got {}", other))),
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

            // Square roots modulo any n >= 1 (MAGMA `Modsqrt(n, m)`, ch. 18 §18.12.1; `IsSquare`,
            // `Sqrt`, `AllSqrts` in Z/mZ, ch. 19 §19.5.6): sqrt_mod(a, n), sqrt_mod(Mod(a, n)),
            // sqrt_mod(Mod(a, n), n). See `sqrt_mod_report` for the output.
            "sqrt_mod" => {
                let args = self.parse_args(args_str)?;
                let (a, n) = match args.as_slice() {
                    [RustMathValue::Integer(a), RustMathValue::Integer(n)] => (a.clone(), n.clone()),
                    [RustMathValue::IntegerMod(x)] => (x.value().clone(), x.modulus().clone()),
                    [RustMathValue::IntegerMod(x), RustMathValue::Integer(n)] => {
                        if n != x.modulus() {
                            return Err(EvalError::new(
                                "ValueError",
                                format!(
                                    "sqrt_mod: {} is an element of Z/{}Z, but the modulus {} was given",
                                    x.value(),
                                    x.modulus(),
                                    n
                                ),
                            ));
                        }
                        (x.value().clone(), x.modulus().clone())
                    }
                    [_] => {
                        return Err(EvalError::new(
                            "TypeError",
                            "sqrt_mod with one argument needs an element Mod(a, n) of Z/nZ; use sqrt_mod(a, n) for integers",
                        ))
                    }
                    [_, _] => {
                        return Err(EvalError::new(
                            "TypeError",
                            "sqrt_mod requires integer arguments sqrt_mod(a, n), or sqrt_mod(Mod(a, n), n)",
                        ))
                    }
                    _ => {
                        return Err(EvalError::new(
                            "ArgumentError",
                            "sqrt_mod requires 1 or 2 arguments: sqrt_mod(a, n) or sqrt_mod(Mod(a, n))",
                        ))
                    }
                };
                sqrt_mod_report(&a, &n).map(RustMathValue::String)
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
                        Ok(RustMathValue::Integer(checked_binomial(n, k)?))
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
                        // MAGMA Catalan: small n >= 0. The bound also keeps the callee's
                        // `2 * n` inside u32.
                        let n_u32 = int_arg::<u32>("catalan", "n", n, 0..=repl_limits::CATALAN_N, "REPL limit")?;
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
                        // MAGMA ch. 144: Fibonacci(n) for all integers, F_{-n} = (-1)^{n+1} F_n.
                        let lim = repl_limits::FIBONACCI_N;
                        let n_i64 = int_arg::<i64>("fibonacci", "n", n, -lim..=lim, "REPL limit")?;
                        Ok(RustMathValue::Integer(fibonacci_signed(n_i64)))
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
                        // MAGMA ch. 144: Lucas(n) for all integers, L_{-n} = (-1)^n L_n.
                        let lim = repl_limits::FIBONACCI_N;
                        let n_i64 = int_arg::<i64>("lucas", "n", n, -lim..=lim, "REPL limit")?;
                        Ok(RustMathValue::Integer(lucas_signed(n_i64)))
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
                        let n_u32 = int_arg::<u32>("bell", "n", n, 0..=repl_limits::BELL_N, "REPL limit")?;
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
                        // (n+1) x (k+1) table, bounded in memory by `check_table`; k > n returns 0
                        // (and the other trivial cases) before any table is allocated.
                        let n_u32 = int_arg::<u32>("stirling1", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("stirling1", "k", k, 0..=u32::MAX, "")?;
                        check_table("stirling1", TableKind::Stirling1, n_u32, k_u32)?;
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
                        // (n+1) x (k+1) table, bounded in memory by `check_table`; k > n returns 0
                        // (and the other trivial cases) before any table is allocated.
                        let n_u32 = int_arg::<u32>("stirling2", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("stirling2", "k", k, 0..=u32::MAX, "")?;
                        check_table("stirling2", TableKind::Stirling2, n_u32, k_u32)?;
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
                        let n_u32 = int_arg::<u32>("multinomial", "n", n, 0..=u32::MAX, "")?;
                        let mut k_ints: Vec<&Integer> = Vec::with_capacity(ks.len());
                        for k in ks {
                            match k {
                                RustMathValue::Integer(ki) => {
                                    if ki.signum() < 0 {
                                        return Err(EvalError::new("ValueError", format!(
                                            "multinomial: every k must be a non-negative integer, got {}", show_int(ki))));
                                    }
                                    k_ints.push(ki);
                                }
                                _ => return Err(EvalError::new("TypeError", "k values must be integers")),
                            }
                        }
                        // The sum is taken in Integer: the callee's u32 sum could overflow.
                        let sum = k_ints.iter().fold(Integer::zero(), |acc, ki| acc + (*ki).clone());
                        if sum != *n {
                            return Err(EvalError::new("ValueError",
                                format!("multinomial requires sum of k values ({}) to equal n ({})", show_int(&sum), n_u32)));
                        }
                        // Every k_i is now in 0..=n, and the callee's partial sums stay <= n.
                        let mut ks_u32 = Vec::with_capacity(k_ints.len());
                        for ki in k_ints {
                            ks_u32.push(int_arg::<u32>("multinomial", "k", ki, 0..=n_u32, "")?);
                        }
                        // n! and every k_i! by sequential products (the k_i! cost less than n!).
                        let work = ks_u32.iter().fold(product_work(n_u32, n_u32), |acc, &ki| acc.saturating_add(product_work(ki, ki)));
                        if work > repl_limits::FACTORIAL_WORK {
                            return Err(work_error("multinomial", &format!(
                                "n! for n = {} takes n multiplications on numbers of up to {} bits", n_u32,
                                u64::from(n_u32) * u64::from(u32::BITS - n_u32.leading_zeros())), work, repl_limits::FACTORIAL_WORK));
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
                        let n_u32 = int_arg::<u32>("falling_factorial", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("falling_factorial", "k", k, 0..=u32::MAX, "")?;
                        // k > n returns 0 at once; otherwise k multiplications.
                        if k_u32 <= n_u32 {
                            let work = product_work(k_u32, n_u32);
                            if work > repl_limits::FACTORIAL_WORK {
                                return Err(work_error("falling_factorial", &format!(
                                    "(n)_k for n = {}, k = {} takes k multiplications on numbers of up to {} bits", n_u32, k_u32,
                                    u64::from(k_u32) * u64::from(u32::BITS - n_u32.leading_zeros())), work, repl_limits::FACTORIAL_WORK));
                            }
                        }
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
                        let k_u32 = int_arg::<u32>("rising_factorial", "k", k, 0..=u32::MAX, "")?;
                        // The callee computes n + i for i < k in u32.
                        let n_u32 = int_arg::<u32>("rising_factorial", "n", n, 0..=u32::MAX - k_u32.saturating_sub(1),
                            "n + k - 1 must fit in 32 bits")?;
                        if n_u32 == 0 && k_u32 > 0 {
                            // 0·1·…·(k-1) = 0; the callee would still run k iterations.
                            return Ok(RustMathValue::Integer(Integer::zero()));
                        }
                        let top = n_u32 + k_u32.saturating_sub(1);
                        let work = product_work(k_u32, top);
                        if work > repl_limits::FACTORIAL_WORK {
                            return Err(work_error("rising_factorial", &format!(
                                "n^(k) for n = {}, k = {} takes k multiplications on numbers of up to {} bits", n_u32, k_u32,
                                u64::from(k_u32) * u64::from(u32::BITS - top.leading_zeros())), work, repl_limits::FACTORIAL_WORK));
                        }
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
                        // (n+1) x (k+1) table, bounded in memory by `check_table`; k >= n returns 0
                        // (and the other trivial cases) before any table is allocated.
                        let n_u32 = int_arg::<u32>("eulerian", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("eulerian", "k", k, 0..=u32::MAX, "")?;
                        check_table("eulerian", TableKind::Eulerian, n_u32, k_u32)?;
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
                        let n_u32 = int_arg::<u32>("narayana", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("narayana", "k", k, 0..=u32::MAX, "")?;
                        // The callee computes C(n, k) and C(n, k-1) only for 2 <= k <= n.
                        if n_u32 >= 1 && (2..=n_u32).contains(&k_u32) {
                            let work = binomial_work_u32(n_u32, k_u32).saturating_add(binomial_work_u32(n_u32, k_u32 - 1));
                            if work > repl_limits::BINOMIAL_WORK {
                                return Err(work_error("narayana", &format!(
                                    "N(n, k) for n = {}, k = {} computes C(n, k) and C(n, k-1)", n_u32, k_u32),
                                    work, repl_limits::BINOMIAL_WORK));
                            }
                        }
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
                        let n_u32 = int_arg::<u32>("motzkin", "n", n, 0..=repl_limits::MOTZKIN_N, "REPL limit")?;
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
                                let n_u32 = int_arg::<u32>("delannoy", "n", n, 0..=u32::MAX, "")?;
                                let work = delannoy_work(n_u32, n_u32, true);
                                if work > repl_limits::BINOMIAL_WORK {
                                    return Err(work_error("delannoy", &format!(
                                        "D(n) for n = {} sums n + 1 squared binomials C(n, k)", n_u32),
                                        work, repl_limits::BINOMIAL_WORK));
                                }
                                Ok(RustMathValue::Integer(delannoy_central(n_u32)))
                            }
                            _ => Err(EvalError::new("TypeError", "delannoy requires integer arguments")),
                        }
                    }
                    2 => {
                        match (&args[0], &args[1]) {
                            (RustMathValue::Integer(m), RustMathValue::Integer(n)) => {
                                let m_u32 = int_arg::<u32>("delannoy", "m", m, 0..=u32::MAX, "")?;
                                let n_u32 = int_arg::<u32>("delannoy", "n", n, 0..=u32::MAX, "")?;
                                let work = delannoy_work(m_u32, n_u32, false);
                                if work > repl_limits::BINOMIAL_WORK {
                                    return Err(work_error("delannoy", &format!(
                                        "D(m, n) for m = {}, n = {} sums min(m, n) + 1 products C(m, k) C(n, k)", m_u32, n_u32),
                                        work, repl_limits::BINOMIAL_WORK));
                                }
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
                        // The callee keeps all n+1 values; the bound also keeps its
                        // `3 * (2i - 1) as u32` far from overflow.
                        let n_u32 = int_arg::<u32>("schroder", "n", n, 0..=repl_limits::SCHRODER_N, "REPL limit")?;
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
                        let n_u32 = int_arg::<u32>("schroder_small", "n", n, 0..=repl_limits::SCHRODER_N, "REPL limit")?;
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
                        // partition_count is usize-valued: p(417) overflows 64 bits.
                        let n_usize = int_arg::<usize>("partitions", "n", n, 0..=repl_limits::PARTITIONS_N,
                            "p(n) must fit in a machine word")?;
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
                        let n_u32 = int_arg::<u32>("derangements", "n", n, 0..=repl_limits::DERANGEMENTS_N, "REPL limit")?;
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
                                let k_u32 = int_arg::<u32>("sigma", "k", k, 0..=u32::MAX, "")?;
                                // n = ±1 returns 1, n = 0 an error and k = 0 the divisor count,
                                // before any power is taken.
                                if n.bit_length() > 1 && k_u32 > 0 {
                                    let log2_n = log2_abs(n);
                                    // Also keeps (a+1)k below 2^29 (a <= log2|n|).
                                    let all_bits = f64::from(k_u32) * log2_n;
                                    if all_bits > repl_limits::SIGMA_BITS {
                                        return Err(EvalError::new("ValueError", format!(
                                            "sigma: sigma_k(n) for k = {} has about {:.0} bits (REPL limit {} bits, a memory bound)",
                                            k_u32, all_bits, repl_limits::SIGMA_BITS)));
                                    }
                                    let twos = n.as_bigint().trailing_zeros().unwrap_or(0) as f64;
                                    if f64::from(k_u32) * (log2_n - twos) > repl_limits::SIGMA_FACTOR_BITS {
                                        // The time depends on the factorisation: factor |n| (the
                                        // trial division Integer::sigma runs), estimate, and
                                        // compute from the factors, so n is factored once.
                                        let factors = rustmath_integers::prime::factor(n);
                                        let secs = sigma_seconds(&factors, k_u32);
                                        if secs > repl_limits::SIGMA_SECONDS {
                                            return Err(EvalError::new("ValueError", format!(
                                                "sigma: sigma_k(n) for k = {} needs about {:.1} s or more for the powers \
                                                 p^((a+1)k) and the divisions (REPL limit {} s, a lower estimate)",
                                                k_u32, secs, repl_limits::SIGMA_SECONDS)));
                                        }
                                        return Ok(RustMathValue::Integer(sigma_from_factors(&factors, k_u32)));
                                    }
                                }
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
                        check_proof_bits("next_prime", n)?;
                        try_next_prime(n)
                            .map(RustMathValue::Integer)
                            .map_err(|e| EvalError::new("ValueError", format!("next_prime: {}", e)))
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
                        // The callee's domain (it panics for 0). No REPL limit (S6).
                        let n_usize = int_arg::<usize>("nth_prime", "n", n, 1..=usize::MAX, "")?;
                        if n_usize as u64 > rustmath_integers::sieve::PRIME_PI_2_64 {
                            return Err(EvalError::new("ValueError", format!(
                                "nth_prime: n = {} exceeds pi(2^64) = {}; the n-th prime does not fit \
                                 a u64 and is not computed",
                                n, rustmath_integers::sieve::PRIME_PI_2_64)));
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
                        if *n > Integer::from(u64::MAX) {
                            return Err(EvalError::new("ValueError", format!(
                                "prime_pi: x = {} exceeds 2^64; the sieve cannot reach it and \
                                 sublinear counting is not implemented", n)));
                        }
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
                        let primes = try_primes_between(a, b)
                            .map_err(|e| EvalError::new("ValueError", format!("primes: {}", e)))?;
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

            // Numerical root finding with multiple syntaxes:
            // SageMath style: find_root(cos(x)==sin(x), 0, pi/2) or find_root(f(x), 0, 1)
            // Extended: find_root(expr, var, a, b)
            "find_root" | "nsolve" => {
                // First, check if the first argument contains "==" (equation)
                // Parse the args_str to detect equation syntax
                let trimmed = args_str.trim();

                // Try to find "==" in the expression (for equation syntax)
                let (expr_to_solve, bounds_start) = if let Some(eq_pos) = self.find_equation_operator(trimmed) {
                    // Equation syntax: lhs == rhs, a, b
                    // Convert to lhs - rhs and find where it's zero
                    let lhs_str = &trimmed[..eq_pos];
                    let rhs_and_bounds = &trimmed[eq_pos + 2..];

                    // Find the comma that separates rhs from bounds
                    let (rhs_str, bounds_str) = self.split_rhs_and_bounds(rhs_and_bounds)?;

                    // Parse lhs and rhs as expressions
                    let lhs = self.eval_expr(lhs_str.trim())?;
                    let rhs = self.eval_expr(rhs_str.trim())?;

                    // Convert to lhs - rhs
                    let diff_expr = match (lhs, rhs) {
                        (RustMathValue::Expr(l), RustMathValue::Expr(r)) => l - r,
                        (RustMathValue::Expr(l), RustMathValue::Integer(n)) => l - Expr::Integer(n),
                        (RustMathValue::Expr(l), RustMathValue::Float(f)) => l - Expr::Real(f),
                        (RustMathValue::Integer(n), RustMathValue::Expr(r)) => Expr::Integer(n) - r,
                        (RustMathValue::Float(f), RustMathValue::Expr(r)) => Expr::Real(f) - r,
                        _ => return Err(EvalError::new("TypeError", "equation must involve symbolic expressions")),
                    };
                    (diff_expr, bounds_str)
                } else {
                    // No equation, use standard parsing
                    let args = self.parse_args(args_str)?;

                    if args.len() == 3 {
                        // find_root(expr, a, b) - auto-detect variable
                        let expr_clone = match &args[0] {
                            RustMathValue::Expr(e) => e.clone(),
                            _ => return Err(EvalError::new("TypeError", "first argument must be an expression")),
                        };
                        let a = self.value_to_f64(&args[1])?;
                        let b = self.value_to_f64(&args[2])?;

                        // Auto-detect variable
                        let symbols = expr_clone.symbols();
                        if symbols.is_empty() {
                            return Err(EvalError::new("ValueError", "expression has no variables"));
                        }
                        let var_sym = symbols[0].clone();

                        let f = move |x: f64| -> f64 {
                            let val = Expr::Real(x);
                            let substituted = expr_clone.substitute(&var_sym, &val);
                            substituted.eval_float().unwrap_or(f64::NAN)
                        };

                        return match bisection(f, a, b, 1e-10, 1000) {
                            Some(result) => Ok(RustMathValue::Float(result.root)),
                            None => Err(EvalError::new("ValueError", "No root found in interval")),
                        };
                    } else if args.len() >= 4 {
                        // find_root(expr, var, a, b) - original syntax
                        let expr_clone = match &args[0] {
                            RustMathValue::Expr(e) => e.clone(),
                            _ => return Err(EvalError::new("TypeError", "first argument must be an expression")),
                        };
                        let var_name = match &args[1] {
                            RustMathValue::Symbol(s) => s.name().to_string(),
                            RustMathValue::String(s) => s.clone(),
                            _ => return Err(EvalError::new("TypeError", "second argument must be a variable")),
                        };
                        let expr_symbols = expr_clone.symbols();
                        let var_sym = expr_symbols.iter()
                            .find(|s| s.name() == var_name)
                            .cloned()
                            .ok_or_else(|| EvalError::new("ValueError",
                                format!("Variable '{}' not found in expression", var_name)))?;
                        let a = self.value_to_f64(&args[2])?;
                        let b = self.value_to_f64(&args[3])?;

                        let f = move |x: f64| -> f64 {
                            let val = Expr::Real(x);
                            let substituted = expr_clone.substitute(&var_sym, &val);
                            substituted.eval_float().unwrap_or(f64::NAN)
                        };

                        return match bisection(f, a, b, 1e-10, 1000) {
                            Some(result) => Ok(RustMathValue::Float(result.root)),
                            None => Err(EvalError::new("ValueError", "No root found in interval")),
                        };
                    } else {
                        return Err(EvalError::new("ArgumentError",
                            "find_root requires: (equation, a, b) or (expr, a, b) or (expr, var, a, b)"));
                    }
                };

                // Handle equation syntax result
                let symbols = expr_to_solve.symbols();
                if symbols.is_empty() {
                    return Err(EvalError::new("ValueError", "equation has no variables"));
                }
                let var_sym = symbols[0].clone();

                // Parse bounds from bounds_start
                let bounds: Vec<&str> = bounds_start.split(',').map(|s| s.trim()).collect();
                if bounds.len() < 2 {
                    return Err(EvalError::new("ArgumentError", "need lower and upper bounds"));
                }
                let a = self.eval_to_f64(bounds[0])?;
                let b = self.eval_to_f64(bounds[1])?;

                let f = move |x: f64| -> f64 {
                    let val = Expr::Real(x);
                    let substituted = expr_to_solve.substitute(&var_sym, &val);
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
                    // An integral value in 2..=PLOT_POINTS; `as usize` is exact there (NaN
                    // and ±inf fail both tests), so 2.5 no longer becomes 2.
                    let v = self.expr_to_f64(&args[4])?;
                    if !(v.fract() == 0.0 && (2.0..=repl_limits::PLOT_POINTS as f64).contains(&v)) {
                        return Err(EvalError::new("ValueError", format!(
                            "plot_function: n_points must be an integer in 2..={}, got {}", repl_limits::PLOT_POINTS, v)));
                    }
                    v as usize
                } else {
                    100
                };

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
                        RustMathValue::Integer(n) => int_arg::<usize>("histogram", "bins", n, 1..=repl_limits::HISTOGRAM_BINS, "")?,
                        _ => return Err(EvalError::new("TypeError", "bins must be an integer")),
                    }
                } else {
                    10
                };

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
            // OR surface plot: plot3d(f(x,y), (x, xmin, xmax), (y, ymin, ymax))
            "plot3d" => {
                return self.eval_plot3d(args_str);
            }

            // Scatter3d is specifically for point lists
            "scatter3d" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 1 {
                    return Err(EvalError::new("ArgumentError", "scatter3d requires 1 argument (list of 3D points)"));
                }
                match &args[0] {
                    RustMathValue::List(pts) => {
                        let coords = self.extract_3d_coords(pts)?;
                        let svg = self.generate_3d_scatter_svg(&coords, "3D Scatter Plot");
                        let description = format!("3D plot with {} points", coords.len());
                        return Ok(RustMathValue::Plot { description, svg });
                    }
                    _ => return Err(EvalError::new("TypeError", "scatter3d requires a list of 3D points")),
                }
            }

            // Parametric 3D plot: parametric_plot3d((x(t), y(t), z(t)), (t, tmin, tmax))
            // Or parametric surface: parametric_plot3d((x(u,v), y(u,v), z(u,v)), (u, umin, umax), (v, vmin, vmax))
            "parametric_plot3d" => {
                return self.eval_parametric_plot3d(args_str);
            }

            // Implicit 3D plot: implicit_plot3d(f(x,y,z), (x, xmin, xmax), (y, ymin, ymax), (z, zmin, zmax))
            "implicit_plot3d" => {
                return self.eval_implicit_plot3d(args_str);
            }

            // 3D line plot: line3d([(x1,y1,z1), (x2,y2,z2), ...])
            "line3d" => {
                return self.eval_line3d(args_str);
            }

            // 3D arrow: arrow3d((x1,y1,z1), (x2,y2,z2))
            "arrow3d" => {
                return self.eval_arrow3d(args_str);
            }

            // Sphere: sphere(center=(0,0,0), radius=1)
            "sphere" => {
                return self.eval_sphere(args_str);
            }

            // Cylinder: cylinder(start, end, radius)
            "cylinder" => {
                return self.eval_cylinder(args_str);
            }

            // Revolution surface: revolution_plot3d(curve, (t, tmin, tmax), axis='z')
            "revolution_plot3d" => {
                return self.eval_revolution_plot3d(args_str);
            }

            // Spherical plot: spherical_plot3d(f(theta, phi), (theta, 0, pi), (phi, 0, 2*pi))
            "spherical_plot3d" => {
                return self.eval_spherical_plot3d(args_str);
            }

            // Cylindrical plot: cylindrical_plot3d(f(r, theta), (r, rmin, rmax), (theta, 0, 2*pi))
            "cylindrical_plot3d" => {
                return self.eval_cylindrical_plot3d(args_str);
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
                        let dim = int_arg::<usize>("CliffordAlgebra", "dimension", &n, 0..=repl_limits::CLIFFORD_DIM, "REPL limit")?;
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
                        let dim = int_arg::<usize>("ExteriorAlgebra", "dimension", &n, 0..=repl_limits::CLIFFORD_DIM, "REPL limit")?;
                        // Create exterior algebra as Clifford with Q=0
                        let q_form: Vec<Rational> = (0..dim).map(|_| Rational::from(0i64)).collect();
                        let ext = CliffordAlgebra::new(q_form);
                        Ok(RustMathValue::CliffordAlg(ext))
                    }
                    _ => Err(EvalError::new("TypeError", "ExteriorAlgebra requires an integer dimension")),
                }
            }

            // e(i) or e(algebra, i) - Create basis element e_i for Clifford/Exterior algebra
            "e" => {
                let args = self.parse_args(args_str)?;
                match args.len() {
                    1 => {
                        // e(i) - create with default Euclidean metric, dimension 10
                        match &args[0] {
                            RustMathValue::Integer(i) => {
                                let idx = int_arg::<usize>("e", "index", i, 0..=9,
                                    "e(i) is a generator of the 10-dimensional Euclidean Clifford algebra")?;
                                let basis = CliffordBasisElement::generator(idx);
                                // Use dimension 10 with Euclidean metric to avoid index out of bounds
                                let dim = 10;
                                let qform: Vec<Rational> = (0..dim).map(|_| Rational::from(1i64)).collect();
                                let elem = CliffordAlgebraElement::from_term(
                                    Rational::from(1i64),
                                    basis,
                                    dim,
                                    qform,
                                );
                                Ok(RustMathValue::CliffordElem(elem))
                            }
                            _ => Err(EvalError::new("TypeError", "e requires an integer index")),
                        }
                    }
                    2 => {
                        // e(algebra, i) - use algebra's dimension and quadratic form
                        match (&args[0], &args[1]) {
                            (RustMathValue::CliffordAlg(alg), RustMathValue::Integer(i)) => {
                                let idx = index_arg("e", "index", i, alg.dimension(), "generators")?;
                                let basis = CliffordBasisElement::generator(idx);
                                let elem = CliffordAlgebraElement::from_term(
                                    Rational::from(1i64),
                                    basis,
                                    alg.dimension(),
                                    alg.quadratic_form().to_vec(),
                                );
                                Ok(RustMathValue::CliffordElem(elem))
                            }
                            _ => Err(EvalError::new("TypeError", "e(algebra, index) requires (CliffordAlgebra, integer)")),
                        }
                    }
                    _ => Err(EvalError::new("ArgumentError", "e requires 1 argument (index) or 2 arguments (algebra, index)")),
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
                        // Total in k: no basis blade has a negative grade or one above the
                        // dimension, so those give the zero element (grade_component filters).
                        let graded = match k.to_usize() {
                            Some(g) => elem.grade_component(g),
                            None => CliffordAlgebraElement::new(elem.dimension(), elem.quadratic_form().to_vec()),
                        };
                        Ok(RustMathValue::CliffordElem(graded))
                    }
                    _ => Err(EvalError::new("TypeError", "grade requires (CliffordElement, integer)")),
                }
            }

            // pseudoscalar(algebra) - Get the pseudoscalar (top-degree element)
            "pseudoscalar" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let ps = cl.pseudoscalar();
                        Ok(RustMathValue::CliffordElem(ps))
                    }
                    _ => Err(EvalError::new("TypeError", "pseudoscalar requires a CliffordAlgebra")),
                }
            }

            // volume_form(algebra) - Get the volume form (alias for pseudoscalar)
            "volume_form" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let vf = cl.volume_form();
                        Ok(RustMathValue::CliffordElem(vf))
                    }
                    _ => Err(EvalError::new("TypeError", "volume_form requires a CliffordAlgebra")),
                }
            }

            // even_part(elem) - Get the even-graded component
            "even_part" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        let ep = elem.even_part();
                        Ok(RustMathValue::CliffordElem(ep))
                    }
                    _ => Err(EvalError::new("TypeError", "even_part requires a CliffordElement")),
                }
            }

            // odd_part(elem) - Get the odd-graded component
            "odd_part" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        let op = elem.odd_part();
                        Ok(RustMathValue::CliffordElem(op))
                    }
                    _ => Err(EvalError::new("TypeError", "odd_part requires a CliffordElement")),
                }
            }

            // is_even(elem) - Check if element is even-graded
            "is_even" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        Ok(RustMathValue::Bool(elem.is_even()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_even requires a CliffordElement")),
                }
            }

            // is_odd(elem) - Check if element is odd-graded
            "is_odd" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        Ok(RustMathValue::Bool(elem.is_odd()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_odd requires a CliffordElement")),
                }
            }

            // is_homogeneous(elem) - Check if element is homogeneous
            "is_homogeneous" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        Ok(RustMathValue::Bool(elem.is_homogeneous()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_homogeneous requires a CliffordElement")),
                }
            }

            // reverse(elem) - Clifford reverse (reverses basis element order)
            "reverse" | "clifford_reverse" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        let rev = elem.reverse();
                        Ok(RustMathValue::CliffordElem(rev))
                    }
                    _ => Err(EvalError::new("TypeError", "reverse requires a CliffordElement")),
                }
            }

            // grade_involution(elem) - Grade involution α (negates odd-graded parts)
            "grade_involution" | "alpha" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        let gi = elem.grade_involution();
                        Ok(RustMathValue::CliffordElem(gi))
                    }
                    _ => Err(EvalError::new("TypeError", "grade_involution requires a CliffordElement")),
                }
            }

            // clifford_conjugate(elem) - Clifford conjugate (reverse + grade involution)
            "clifford_conjugate" | "conjugate_clifford" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordElem(ref elem) => {
                        let cc = elem.clifford_conjugate();
                        Ok(RustMathValue::CliffordElem(cc))
                    }
                    _ => Err(EvalError::new("TypeError", "clifford_conjugate requires a CliffordElement")),
                }
            }

            // counit(algebra, elem) - Hopf algebra counit (extract scalar part)
            "counit" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "counit requires 2 arguments: algebra and element"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::CliffordAlg(alg), RustMathValue::CliffordElem(elem)) => {
                        let cu = alg.counit(elem);
                        Ok(RustMathValue::Rational(cu))
                    }
                    _ => Err(EvalError::new("TypeError", "counit requires (CliffordAlgebra, CliffordElement)")),
                }
            }

            // antipode(algebra, elem) - Hopf algebra antipode
            "antipode" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "antipode requires 2 arguments: algebra and element"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::CliffordAlg(alg), RustMathValue::CliffordElem(elem)) => {
                        let ap = alg.antipode(elem);
                        Ok(RustMathValue::CliffordElem(ap))
                    }
                    _ => Err(EvalError::new("TypeError", "antipode requires (CliffordAlgebra, CliffordElement)")),
                }
            }

            // interior_product(algebra, form, vector) - Interior product (contraction)
            "interior_product" | "contract" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "interior_product requires 3 arguments: algebra, form, and vector"));
                }
                match (&args[0], &args[1], &args[2]) {
                    (RustMathValue::CliffordAlg(alg), RustMathValue::CliffordElem(form), RustMathValue::CliffordElem(vec)) => {
                        let ip = alg.interior_product(form, vec);
                        Ok(RustMathValue::CliffordElem(ip))
                    }
                    _ => Err(EvalError::new("TypeError", "interior_product requires (CliffordAlgebra, CliffordElement, CliffordElement)")),
                }
            }

            // center(algebra) - Get basis for center of algebra
            "center" | "center_basis" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let cb = cl.center_basis();
                        let list: Vec<RustMathValue> = cb.into_iter()
                            .map(|e| RustMathValue::CliffordElem(e))
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "center requires a CliffordAlgebra")),
                }
            }

            // supercenter(algebra) - Get basis for supercenter of algebra
            "supercenter" | "supercenter_basis" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let scb = cl.supercenter_basis();
                        let list: Vec<RustMathValue> = scb.into_iter()
                            .map(|e| RustMathValue::CliffordElem(e))
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "supercenter requires a CliffordAlgebra")),
                }
            }

            // algebra_basis(algebra) - Get all basis elements
            "algebra_basis" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let basis = cl.basis();
                        let list: Vec<RustMathValue> = basis.into_iter()
                            .map(|e| RustMathValue::CliffordElem(e))
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "algebra_basis requires a CliffordAlgebra")),
                }
            }

            // basis_of_grade(algebra, k) - Get basis elements of grade k
            "basis_of_grade" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "basis_of_grade requires 2 arguments: algebra and grade"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::CliffordAlg(alg), RustMathValue::Integer(k)) => {
                        // Total in k: no basis blade of a negative grade or one above the
                        // dimension (basis_of_grade is empty there).
                        let basis = match k.to_usize() {
                            Some(g) => alg.basis_of_grade(g),
                            None => Vec::new(),
                        };
                        let list: Vec<RustMathValue> = basis.into_iter()
                            .map(|e| RustMathValue::CliffordElem(e))
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "basis_of_grade requires (CliffordAlgebra, integer)")),
                }
            }

            // quadratic_form(algebra) - Get the quadratic form values
            "quadratic_form" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        let qf = cl.quadratic_form();
                        let list: Vec<RustMathValue> = qf.iter()
                            .map(|r| RustMathValue::Rational(r.clone()))
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "quadratic_form requires a CliffordAlgebra")),
                }
            }

            // is_exterior(algebra) - Check if this is an exterior algebra (Q=0)
            "is_exterior" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::CliffordAlg(ref cl) => {
                        Ok(RustMathValue::Bool(cl.is_exterior()))
                    }
                    _ => Err(EvalError::new("TypeError", "is_exterior requires a CliffordAlgebra")),
                }
            }

            // ===== JORDAN ALGEBRA FUNCTIONS =====

            // JordanAlgebra(form_matrix) - Create Jordan algebra from symmetric bilinear form
            // JordanAlgebra(n) - Create with identity form
            "JordanAlgebra" => {
                let args = self.parse_args(args_str)?;
                match args.len() {
                    1 => {
                        match &args[0] {
                            RustMathValue::Integer(n) => {
                                // Create with identity form of dimension n
                                let dim = int_arg::<usize>("JordanAlgebra", "dimension", n, 0..=repl_limits::JORDAN_DIM, "REPL limit")?;
                                let j = JordanAlgebraSymmetricBilinear::<Rational>::standard(dim);
                                Ok(RustMathValue::JordanSymBilinear(j))
                            }
                            RustMathValue::Matrix(m) => {
                                // Create from form matrix
                                if m.rows() != m.cols() {
                                    return Err(EvalError::new("ValueError", "Form matrix must be square"));
                                }
                                let n = m.rows();
                                let mut form = vec![vec![Rational::from(0i64); n]; n];
                                for i in 0..n {
                                    for k in 0..n {
                                        if let Ok(val) = m.get(i, k) {
                                            form[i][k] = Rational::from_integer(val.clone());
                                        }
                                    }
                                }
                                let ja = JordanAlgebraSymmetricBilinear::new(form);
                                Ok(RustMathValue::JordanSymBilinear(ja))
                            }
                            _ => Err(EvalError::new("TypeError", "JordanAlgebra requires integer (dimension) or matrix (form)")),
                        }
                    }
                    _ => Err(EvalError::new("ArgumentError", "JordanAlgebra requires 1 argument")),
                }
            }

            // SpecialJordanAlgebra(n) - Create special Jordan algebra from n x n matrices
            "SpecialJordanAlgebra" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let size = int_arg::<usize>("SpecialJordanAlgebra", "size", &n, 0..=repl_limits::SPECIAL_JORDAN_N, "REPL limit")?;
                        let j = SpecialJordanAlgebra::<Rational>::new(size);
                        Ok(RustMathValue::SpecialJordan(j))
                    }
                    _ => Err(EvalError::new("TypeError", "SpecialJordanAlgebra requires integer matrix size")),
                }
            }

            // AlbertAlgebra() - Create exceptional Jordan algebra (27-dimensional)
            "AlbertAlgebra" | "ExceptionalJordanAlgebra" => {
                let j = ExceptionalJordanAlgebra::<Rational>::new();
                Ok(RustMathValue::ExceptionalJordan(j))
            }

            // jordan_one(algebra) - Get identity element
            "jordan_one" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinear(ref j) => {
                        Ok(RustMathValue::JordanSymBilinearElem(j.one()))
                    }
                    RustMathValue::SpecialJordan(ref j) => {
                        Ok(RustMathValue::SpecialJordanElem(j.one()))
                    }
                    RustMathValue::ExceptionalJordan(ref j) => {
                        Ok(RustMathValue::AlbertElem(j.one()))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_one requires a Jordan algebra")),
                }
            }

            // jordan_zero(algebra) - Get zero element
            "jordan_zero" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinear(ref j) => {
                        Ok(RustMathValue::JordanSymBilinearElem(j.zero()))
                    }
                    RustMathValue::SpecialJordan(ref j) => {
                        Ok(RustMathValue::SpecialJordanElem(j.zero()))
                    }
                    RustMathValue::ExceptionalJordan(ref j) => {
                        Ok(RustMathValue::AlbertElem(j.zero()))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_zero requires a Jordan algebra")),
                }
            }

            // jordan_basis(algebra) - Get basis elements
            "jordan_basis" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinear(ref j) => {
                        let basis = j.basis();
                        let list: Vec<RustMathValue> = basis.into_iter()
                            .map(RustMathValue::JordanSymBilinearElem)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    RustMathValue::ExceptionalJordan(ref j) => {
                        let basis = j.basis();
                        let list: Vec<RustMathValue> = basis.into_iter()
                            .map(RustMathValue::AlbertElem)
                            .collect();
                        Ok(RustMathValue::List(list))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_basis requires a Jordan algebra")),
                }
            }

            // jordan_multiply(algebra, a, b) - Jordan product
            "jordan_multiply" | "jordan_product" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 3 {
                    return Err(EvalError::new("ArgumentError", "jordan_multiply requires 3 arguments: algebra, element, element"));
                }
                match (&args[0], &args[1], &args[2]) {
                    (RustMathValue::JordanSymBilinear(j), RustMathValue::JordanSymBilinearElem(a), RustMathValue::JordanSymBilinearElem(b)) => {
                        let product = j.multiply(a, b);
                        Ok(RustMathValue::JordanSymBilinearElem(product))
                    }
                    (RustMathValue::SpecialJordan(j), RustMathValue::SpecialJordanElem(a), RustMathValue::SpecialJordanElem(b)) => {
                        let product = j.multiply(a, b);
                        Ok(RustMathValue::SpecialJordanElem(product))
                    }
                    (RustMathValue::ExceptionalJordan(j), RustMathValue::AlbertElem(a), RustMathValue::AlbertElem(b)) => {
                        let product = j.multiply(a, b);
                        Ok(RustMathValue::AlbertElem(product))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_multiply requires matching algebra and element types")),
                }
            }

            // jordan_trace(element) - Get trace of Jordan element
            "jordan_trace" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinearElem(ref e) => {
                        let tr = e.trace();
                        Ok(RustMathValue::Rational(tr))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_trace requires a SymmetricBilinear Jordan element")),
                }
            }

            // jordan_norm(algebra, element) - Get norm of Jordan element
            "jordan_norm" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "jordan_norm requires 2 arguments: algebra, element"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::JordanSymBilinear(j), RustMathValue::JordanSymBilinearElem(e)) => {
                        let norm = e.norm(j);
                        Ok(RustMathValue::Rational(norm))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_norm requires (JordanAlgebra, JordanElement)")),
                }
            }

            // jordan_bar(element) - Bar involution
            "jordan_bar" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinearElem(ref e) => {
                        let bar = e.bar();
                        Ok(RustMathValue::JordanSymBilinearElem(bar))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_bar requires a SymmetricBilinear Jordan element")),
                }
            }

            // jordan_dimension(algebra) - Get dimension
            "jordan_dimension" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::JordanSymBilinear(ref j) => {
                        Ok(RustMathValue::Integer(Integer::from(j.dimension() as i64)))
                    }
                    RustMathValue::SpecialJordan(ref j) => {
                        Ok(RustMathValue::Integer(Integer::from(j.dimension() as i64)))
                    }
                    RustMathValue::ExceptionalJordan(ref j) => {
                        Ok(RustMathValue::Integer(Integer::from(j.dimension() as i64)))
                    }
                    _ => Err(EvalError::new("TypeError", "jordan_dimension requires a Jordan algebra")),
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
                        // Domain only: a manifold stores its dimension, nothing of that size
                        // (EuclideanSpace and coordinate_form bound what they allocate).
                        let d = int_arg::<usize>("Manifold", "dimension", dim, 0..=usize::MAX, "")?;
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
                        // The callee builds one coordinate name per dimension.
                        let dim = int_arg::<usize>("EuclideanSpace", "dimension", &n, 0..=repl_limits::EUCLIDEAN_DIM, "REPL limit")?;
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
                        // A degree above dim(M) is the zero form, as the library itself builds;
                        // the bound keeps p + 1 and p + q in rustmath-manifolds from overflowing.
                        let degree = int_arg::<usize>("diff_form", "degree", deg, 0..=repl_limits::DIFF_FORM_DEGREE,
                            "p + 1 and p + q must fit in a machine word")?;
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
                        // The callee stores dim(M) components.
                        if m.dimension() > repl_limits::MANIFOLD_COORDS {
                            return Err(EvalError::new("ValueError", format!(
                                "coordinate_form: the manifold has dimension {}; a coordinate form stores that many \
                                 components (REPL limit {})", m.dimension(), repl_limits::MANIFOLD_COORDS)));
                        }
                        let i = index_arg("coordinate_form", "index", idx, m.dimension(), "coordinates")?;
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
                        let (n, p) = (form.manifold().dimension(), form.degree());
                        let degree = p.checked_add(1).ok_or_else(|| EvalError::new("ValueError", format!(
                            "exterior_derivative: the degree p + 1 of d(omega) for p = {} does not fit in a machine word", p)))?;
                        // For p < n the callee builds all n^(p+1) components (diff_form.rs:139).
                        if p < n {
                            if let Err(limit) = check_form_components("exterior_derivative", n, degree) {
                                return Err(missing_components_error(&[form], chart).unwrap_or(limit));
                            }
                        }
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
                                // Forms on different manifolds are an error before p + q is formed.
                                if Arc::ptr_eq(f1.manifold(), f2.manifold()) {
                                    let (n, p, q) = (f1.manifold().dimension(), f1.degree(), f2.degree());
                                    let degree = p.checked_add(q).ok_or_else(|| EvalError::new("ValueError", format!(
                                        "wedge: the degree p + q of the product for p = {}, q = {} does not fit in a machine word", p, q)))?;
                                    // For p + q <= n the callee builds all n^(p+q) products
                                    // (tensor_field.rs:268-292).
                                    if degree <= n {
                                        if let Err(limit) = check_form_components("wedge", n, degree) {
                                            return Err(missing_components_error(&[f1, f2], chart).unwrap_or(limit));
                                        }
                                    }
                                }
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
        let expr = parse_expr(s)
            .map_err(|e| EvalError::new("ParseError", format!("Cannot parse expression: {}", e)))?;

        // Substitute any REPL variables that have numeric values
        let expr = self.substitute_repl_variables(expr);

        Ok(expr)
    }

    /// Substitute REPL variables with their numeric values into an expression
    fn substitute_repl_variables(&self, expr: Expr) -> Expr {
        let mut result = expr;
        for (name, value) in &self.variables {
            let replacement = match value {
                RustMathValue::Integer(n) => Some(Expr::Integer(n.clone())),
                RustMathValue::Float(f) => Some(Expr::Real(*f)),
                RustMathValue::Rational(r) => Some(Expr::Rational(r.clone())),
                _ => None,
            };
            if let Some(repl_expr) = replacement {
                result = result.substitute_by_name(name, &repl_expr);
            }
        }
        result
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

    // ===== 3D PLOTTING IMPLEMENTATIONS =====

    /// Evaluate plot3d - either surface plot or scatter plot
    /// plot3d(f(x,y), (x, xmin, xmax), (y, ymin, ymax)) - surface
    /// plot3d([Point3D(...)]) - scatter
    fn eval_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        // Check if it's a list (scatter plot)
        if parts.len() == 1 {
            let arg = self.eval_expr(parts[0].trim())?;
            if let RustMathValue::List(pts) = arg {
                let coords = self.extract_3d_coords(&pts)?;
                let svg = self.generate_3d_scatter_svg(&coords, "3D Scatter Plot");
                let description = format!("3D plot with {} points", coords.len());
                return Ok(RustMathValue::Plot { description, svg });
            }
        }

        // Otherwise it's a surface plot: plot3d(f, (x, xmin, xmax), (y, ymin, ymax))
        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "plot3d requires f(x,y), (x, xmin, xmax), (y, ymin, ymax) or a list of 3D points"));
        }

        let expr = self.parse_symbolic_arg(&parts[0])?;

        // Parse x range
        let x_range_str = parts[1].trim();
        if !x_range_str.starts_with('(') || !x_range_str.ends_with(')') {
            return Err(EvalError::new("SyntaxError", "x range must be (var, min, max)"));
        }
        let x_inner = &x_range_str[1..x_range_str.len()-1];
        let x_parts: Vec<&str> = self.split_at_depth_zero(x_inner, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must have 3 parts: (var, min, max)"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;

        // Parse y range
        let y_range_str = parts[2].trim();
        if !y_range_str.starts_with('(') || !y_range_str.ends_with(')') {
            return Err(EvalError::new("SyntaxError", "y range must be (var, min, max)"));
        }
        let y_inner = &y_range_str[1..y_range_str.len()-1];
        let y_parts: Vec<&str> = self.split_at_depth_zero(y_inner, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must have 3 parts: (var, min, max)"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;

        let x_sym = Symbol::new(x_var_name);
        let y_sym = Symbol::new(y_var_name);
        let expr_symbols = expr.symbols();

        // Generate surface
        let n_grid = 30;
        let mut z_values: Vec<Vec<f64>> = Vec::new();

        for j in 0..n_grid {
            let y = y_min + (y_max - y_min) * j as f64 / (n_grid - 1) as f64;
            let mut row = Vec::new();
            for i in 0..n_grid {
                let x = x_min + (x_max - x_min) * i as f64 / (n_grid - 1) as f64;

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

        let svg = self.generate_surface_svg(&z_values, x_min, x_max, y_min, y_max, "Surface Plot z = f(x,y)");
        let description = format!("Surface plot over [{:.2}, {:.2}] x [{:.2}, {:.2}]", x_min, x_max, y_min, y_max);
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate parametric_plot3d
    /// Curve: parametric_plot3d((x(t), y(t), z(t)), (t, tmin, tmax))
    /// Surface: parametric_plot3d((x(u,v), y(u,v), z(u,v)), (u, umin, umax), (v, vmin, vmax))
    fn eval_parametric_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 2 {
            return Err(EvalError::new("ArgumentError",
                "parametric_plot3d requires (x,y,z), (t, tmin, tmax) or (x,y,z), (u, umin, umax), (v, vmin, vmax)"));
        }

        // Parse the (x, y, z) expressions tuple
        let xyz_str = parts[0].trim();
        let xyz_str = xyz_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "first argument must be (x(t), y(t), z(t))"))?;
        let xyz_parts = self.split_at_depth_zero(xyz_str, ',');
        if xyz_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "tuple must have 3 expressions: (x, y, z)"));
        }

        // Parse expressions symbolically
        let x_expr = self.parse_symbolic_arg(xyz_parts[0].trim())?;
        let y_expr = self.parse_symbolic_arg(xyz_parts[1].trim())?;
        let z_expr = self.parse_symbolic_arg(xyz_parts[2].trim())?;

        // Check if it's a curve (2 args) or surface (3 args)
        if parts.len() == 2 {
            // Parametric curve
            let t_range_str = parts[1].trim();
            let t_range_str = t_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                .ok_or_else(|| EvalError::new("SyntaxError", "parameter range must be (var, min, max)"))?;
            let t_parts = self.split_at_depth_zero(t_range_str, ',');
            if t_parts.len() != 3 {
                return Err(EvalError::new("ArgumentError", "parameter range must have 3 parts"));
            }
            let t_var_name = t_parts[0].trim();
            let t_min = self.eval_to_f64(t_parts[1].trim())?;
            let t_max = self.eval_to_f64(t_parts[2].trim())?;
            let t_sym = Symbol::new(t_var_name);

            // Generate curve points
            let n_points = 200;
            let mut coords = Vec::new();

            for i in 0..=n_points {
                let t = t_min + (t_max - t_min) * i as f64 / n_points as f64;

                let x_eval = x_expr.substitute(&t_sym, &Expr::Real(t));
                let y_eval = y_expr.substitute(&t_sym, &Expr::Real(t));
                let z_eval = z_expr.substitute(&t_sym, &Expr::Real(t));

                let x = try_eval_to_f64(&x_eval).unwrap_or(f64::NAN);
                let y = try_eval_to_f64(&y_eval).unwrap_or(f64::NAN);
                let z = try_eval_to_f64(&z_eval).unwrap_or(f64::NAN);

                if x.is_finite() && y.is_finite() && z.is_finite() {
                    coords.push((x, y, z));
                }
            }

            let svg = self.generate_3d_curve_svg(&coords, "3D Parametric Curve");
            let description = format!("Parametric curve with {} points", coords.len());
            Ok(RustMathValue::Plot { description, svg })
        } else {
            // Parametric surface (3 args including the tuple)
            // Parse u range
            let u_range_str = parts[1].trim();
            let u_range_str = u_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                .ok_or_else(|| EvalError::new("SyntaxError", "u range must be (var, min, max)"))?;
            let u_parts = self.split_at_depth_zero(u_range_str, ',');
            if u_parts.len() != 3 {
                return Err(EvalError::new("ArgumentError", "u range must have 3 parts"));
            }
            let u_var_name = u_parts[0].trim();
            let u_min = self.eval_to_f64(u_parts[1].trim())?;
            let u_max = self.eval_to_f64(u_parts[2].trim())?;
            let u_sym = Symbol::new(u_var_name);

            // Parse v range
            let v_range_str = parts[2].trim();
            let v_range_str = v_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
                .ok_or_else(|| EvalError::new("SyntaxError", "v range must be (var, min, max)"))?;
            let v_parts = self.split_at_depth_zero(v_range_str, ',');
            if v_parts.len() != 3 {
                return Err(EvalError::new("ArgumentError", "v range must have 3 parts"));
            }
            let v_var_name = v_parts[0].trim();
            let v_min = self.eval_to_f64(v_parts[1].trim())?;
            let v_max = self.eval_to_f64(v_parts[2].trim())?;
            let v_sym = Symbol::new(v_var_name);

            // Generate parametric surface
            let n_u = 30;
            let n_v = 30;
            let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

            for j in 0..n_v {
                let v = v_min + (v_max - v_min) * j as f64 / (n_v - 1) as f64;
                let mut row = Vec::new();
                for i in 0..n_u {
                    let u = u_min + (u_max - u_min) * i as f64 / (n_u - 1) as f64;

                    let x_eval = x_expr.substitute(&u_sym, &Expr::Real(u))
                                       .substitute(&v_sym, &Expr::Real(v));
                    let y_eval = y_expr.substitute(&u_sym, &Expr::Real(u))
                                       .substitute(&v_sym, &Expr::Real(v));
                    let z_eval = z_expr.substitute(&u_sym, &Expr::Real(u))
                                       .substitute(&v_sym, &Expr::Real(v));

                    let x = try_eval_to_f64(&x_eval).unwrap_or(f64::NAN);
                    let y = try_eval_to_f64(&y_eval).unwrap_or(f64::NAN);
                    let z = try_eval_to_f64(&z_eval).unwrap_or(f64::NAN);

                    row.push((x, y, z));
                }
                surface_points.push(row);
            }

            let svg = self.generate_parametric_surface_svg(&surface_points, "3D Parametric Surface");
            let description = format!("Parametric surface ({}×{} grid)", n_u, n_v);
            Ok(RustMathValue::Plot { description, svg })
        }
    }

    /// Evaluate implicit_plot3d
    /// implicit_plot3d(f(x,y,z), (x, xmin, xmax), (y, ymin, ymax), (z, zmin, zmax))
    fn eval_implicit_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        if parts.len() < 4 {
            return Err(EvalError::new("ArgumentError",
                "implicit_plot3d requires f(x,y,z), (x, xmin, xmax), (y, ymin, ymax), (z, zmin, zmax)"));
        }

        // Parse expression symbolically
        let f_expr = self.parse_symbolic_arg(parts[0].trim())?;

        // Parse x range
        let x_range_str = parts[1].trim();
        let x_range_str = x_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "x range must be (var, min, max)"))?;
        let x_parts = self.split_at_depth_zero(x_range_str, ',');
        if x_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "x range must have 3 parts"));
        }
        let x_var_name = x_parts[0].trim();
        let x_min = self.eval_to_f64(x_parts[1].trim())?;
        let x_max = self.eval_to_f64(x_parts[2].trim())?;
        let x_sym = Symbol::new(x_var_name);

        // Parse y range
        let y_range_str = parts[2].trim();
        let y_range_str = y_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "y range must be (var, min, max)"))?;
        let y_parts = self.split_at_depth_zero(y_range_str, ',');
        if y_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "y range must have 3 parts"));
        }
        let y_var_name = y_parts[0].trim();
        let y_min = self.eval_to_f64(y_parts[1].trim())?;
        let y_max = self.eval_to_f64(y_parts[2].trim())?;
        let y_sym = Symbol::new(y_var_name);

        // Parse z range
        let z_range_str = parts[3].trim();
        let z_range_str = z_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "z range must be (var, min, max)"))?;
        let z_parts = self.split_at_depth_zero(z_range_str, ',');
        if z_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "z range must have 3 parts"));
        }
        let z_var_name = z_parts[0].trim();
        let z_min = self.eval_to_f64(z_parts[1].trim())?;
        let z_max = self.eval_to_f64(z_parts[2].trim())?;
        let z_sym = Symbol::new(z_var_name);

        // Use marching cubes to find the isosurface
        let n = 20; // Grid resolution
        let mut triangles: Vec<[(f64, f64, f64); 3]> = Vec::new();

        // Evaluate function on grid
        let mut grid: Vec<Vec<Vec<f64>>> = Vec::new();
        for k in 0..=n {
            let zv = z_min + (z_max - z_min) * k as f64 / n as f64;
            let mut plane = Vec::new();
            for j in 0..=n {
                let yv = y_min + (y_max - y_min) * j as f64 / n as f64;
                let mut row = Vec::new();
                for i in 0..=n {
                    let xv = x_min + (x_max - x_min) * i as f64 / n as f64;

                    let eval_expr = f_expr.substitute(&x_sym, &Expr::Real(xv))
                                          .substitute(&y_sym, &Expr::Real(yv))
                                          .substitute(&z_sym, &Expr::Real(zv));
                    let val = try_eval_to_f64(&eval_expr).unwrap_or(f64::NAN);
                    row.push(val);
                }
                plane.push(row);
            }
            grid.push(plane);
        }

        // Marching cubes (simplified - just extract isosurface points)
        let level = 0.0;
        let dx = (x_max - x_min) / n as f64;
        let dy = (y_max - y_min) / n as f64;
        let dz = (z_max - z_min) / n as f64;

        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    // Get 8 corner values
                    let v = [
                        grid[k][j][i], grid[k][j][i+1], grid[k][j+1][i+1], grid[k][j+1][i],
                        grid[k+1][j][i], grid[k+1][j][i+1], grid[k+1][j+1][i+1], grid[k+1][j+1][i],
                    ];

                    // Simple crossing detection: if any edge has sign change, add a point
                    let x0 = x_min + i as f64 * dx;
                    let y0 = y_min + j as f64 * dy;
                    let z0 = z_min + k as f64 * dz;

                    // Check for sign changes and interpolate vertices
                    let mut vertices = Vec::new();

                    // Edge 0-1 (x direction, bottom front)
                    if (v[0] - level) * (v[1] - level) < 0.0 && v[0].is_finite() && v[1].is_finite() {
                        let t = (level - v[0]) / (v[1] - v[0]);
                        vertices.push((x0 + t * dx, y0, z0));
                    }
                    // Edge 1-2 (y direction, right front)
                    if (v[1] - level) * (v[2] - level) < 0.0 && v[1].is_finite() && v[2].is_finite() {
                        let t = (level - v[1]) / (v[2] - v[1]);
                        vertices.push((x0 + dx, y0 + t * dy, z0));
                    }
                    // Edge 2-3 (x direction, bottom back)
                    if (v[2] - level) * (v[3] - level) < 0.0 && v[2].is_finite() && v[3].is_finite() {
                        let t = (level - v[2]) / (v[3] - v[2]);
                        vertices.push((x0 + (1.0 - t) * dx, y0 + dy, z0));
                    }
                    // Edge 3-0 (y direction, left front)
                    if (v[3] - level) * (v[0] - level) < 0.0 && v[3].is_finite() && v[0].is_finite() {
                        let t = (level - v[3]) / (v[0] - v[3]);
                        vertices.push((x0, y0 + (1.0 - t) * dy, z0));
                    }
                    // Edge 4-5 (x direction, top front)
                    if (v[4] - level) * (v[5] - level) < 0.0 && v[4].is_finite() && v[5].is_finite() {
                        let t = (level - v[4]) / (v[5] - v[4]);
                        vertices.push((x0 + t * dx, y0, z0 + dz));
                    }
                    // Edge 5-6 (y direction, right back)
                    if (v[5] - level) * (v[6] - level) < 0.0 && v[5].is_finite() && v[6].is_finite() {
                        let t = (level - v[5]) / (v[6] - v[5]);
                        vertices.push((x0 + dx, y0 + t * dy, z0 + dz));
                    }
                    // Edge 6-7 (x direction, top back)
                    if (v[6] - level) * (v[7] - level) < 0.0 && v[6].is_finite() && v[7].is_finite() {
                        let t = (level - v[6]) / (v[7] - v[6]);
                        vertices.push((x0 + (1.0 - t) * dx, y0 + dy, z0 + dz));
                    }
                    // Edge 7-4 (y direction, left back)
                    if (v[7] - level) * (v[4] - level) < 0.0 && v[7].is_finite() && v[4].is_finite() {
                        let t = (level - v[7]) / (v[4] - v[7]);
                        vertices.push((x0, y0 + (1.0 - t) * dy, z0 + dz));
                    }
                    // Edge 0-4 (z direction, front left)
                    if (v[0] - level) * (v[4] - level) < 0.0 && v[0].is_finite() && v[4].is_finite() {
                        let t = (level - v[0]) / (v[4] - v[0]);
                        vertices.push((x0, y0, z0 + t * dz));
                    }
                    // Edge 1-5 (z direction, front right)
                    if (v[1] - level) * (v[5] - level) < 0.0 && v[1].is_finite() && v[5].is_finite() {
                        let t = (level - v[1]) / (v[5] - v[1]);
                        vertices.push((x0 + dx, y0, z0 + t * dz));
                    }
                    // Edge 2-6 (z direction, back right)
                    if (v[2] - level) * (v[6] - level) < 0.0 && v[2].is_finite() && v[6].is_finite() {
                        let t = (level - v[2]) / (v[6] - v[2]);
                        vertices.push((x0 + dx, y0 + dy, z0 + t * dz));
                    }
                    // Edge 3-7 (z direction, back left)
                    if (v[3] - level) * (v[7] - level) < 0.0 && v[3].is_finite() && v[7].is_finite() {
                        let t = (level - v[3]) / (v[7] - v[3]);
                        vertices.push((x0, y0 + dy, z0 + t * dz));
                    }

                    // Create triangles from vertices (fan triangulation from centroid)
                    if vertices.len() >= 3 {
                        let centroid = (
                            vertices.iter().map(|v| v.0).sum::<f64>() / vertices.len() as f64,
                            vertices.iter().map(|v| v.1).sum::<f64>() / vertices.len() as f64,
                            vertices.iter().map(|v| v.2).sum::<f64>() / vertices.len() as f64,
                        );
                        for w in 0..vertices.len() {
                            let next = (w + 1) % vertices.len();
                            triangles.push([vertices[w], vertices[next], centroid]);
                        }
                    }
                }
            }
        }

        let svg = self.generate_implicit_surface_svg(&triangles, "Implicit Surface f(x,y,z) = 0");
        let description = format!("Implicit surface with {} triangles", triangles.len());
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate line3d
    fn eval_line3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let arg = self.eval_expr(args_str.trim())?;
        match arg {
            RustMathValue::List(pts) => {
                let coords = self.extract_3d_coords(&pts)?;
                let svg = self.generate_3d_curve_svg(&coords, "3D Line");
                let description = format!("3D line with {} vertices", coords.len());
                Ok(RustMathValue::Plot { description, svg })
            }
            _ => Err(EvalError::new("TypeError", "line3d requires a list of points")),
        }
    }

    /// Evaluate arrow3d
    fn eval_arrow3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let parts = self.split_at_depth_zero(args_str.trim(), ',');
        if parts.len() < 2 {
            return Err(EvalError::new("ArgumentError", "arrow3d requires start and end points"));
        }

        // Parse start point
        let start_val = self.eval_expr(parts[0].trim())?;
        let start = match &start_val {
            RustMathValue::Point3D(p) => (p.x, p.y, p.z),
            RustMathValue::List(v) if v.len() == 3 => {
                (self.value_to_f64(&v[0])?, self.value_to_f64(&v[1])?, self.value_to_f64(&v[2])?)
            }
            _ => return Err(EvalError::new("TypeError", "start must be Point3D or [x,y,z]")),
        };

        // Parse end point
        let end_val = self.eval_expr(parts[1].trim())?;
        let end = match &end_val {
            RustMathValue::Point3D(p) => (p.x, p.y, p.z),
            RustMathValue::List(v) if v.len() == 3 => {
                (self.value_to_f64(&v[0])?, self.value_to_f64(&v[1])?, self.value_to_f64(&v[2])?)
            }
            _ => return Err(EvalError::new("TypeError", "end must be Point3D or [x,y,z]")),
        };

        let svg = self.generate_3d_arrow_svg(start, end, "3D Arrow");
        let description = format!("Arrow from ({:.2},{:.2},{:.2}) to ({:.2},{:.2},{:.2})",
            start.0, start.1, start.2, end.0, end.1, end.2);
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate sphere
    fn eval_sphere(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let args_str = args_str.trim();
        let parts = self.split_at_depth_zero(args_str, ',');

        let (center, radius) = if parts.is_empty() || args_str.is_empty() {
            // Default: unit sphere at origin
            ((0.0, 0.0, 0.0), 1.0)
        } else if parts.len() == 1 {
            // Just radius
            let r = self.eval_to_f64(parts[0].trim())?;
            ((0.0, 0.0, 0.0), r)
        } else {
            // center, radius
            let center_val = self.eval_expr(parts[0].trim())?;
            let center = match &center_val {
                RustMathValue::Point3D(p) => (p.x, p.y, p.z),
                RustMathValue::List(v) if v.len() == 3 => {
                    (self.value_to_f64(&v[0])?, self.value_to_f64(&v[1])?, self.value_to_f64(&v[2])?)
                }
                _ => return Err(EvalError::new("TypeError", "center must be Point3D or [x,y,z]")),
            };
            let r = self.eval_to_f64(parts[1].trim())?;
            (center, r)
        };

        // Generate sphere as parametric surface
        let n = 30;
        let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        for j in 0..=n {
            let phi = std::f64::consts::PI * j as f64 / n as f64;
            let mut row = Vec::new();
            for i in 0..=n {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n as f64;

                let x = center.0 + radius * phi.sin() * theta.cos();
                let y = center.1 + radius * phi.sin() * theta.sin();
                let z = center.2 + radius * phi.cos();
                row.push((x, y, z));
            }
            surface_points.push(row);
        }

        let svg = self.generate_parametric_surface_svg(&surface_points, "Sphere");
        let description = format!("Sphere center=({:.2},{:.2},{:.2}), radius={:.2}",
            center.0, center.1, center.2, radius);
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate cylinder
    fn eval_cylinder(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let parts = self.split_at_depth_zero(args_str.trim(), ',');

        let (start, end, radius) = if parts.len() >= 3 {
            let start_val = self.eval_expr(parts[0].trim())?;
            let start = match &start_val {
                RustMathValue::Point3D(p) => (p.x, p.y, p.z),
                RustMathValue::List(v) if v.len() == 3 => {
                    (self.value_to_f64(&v[0])?, self.value_to_f64(&v[1])?, self.value_to_f64(&v[2])?)
                }
                _ => return Err(EvalError::new("TypeError", "start must be Point3D or [x,y,z]")),
            };
            let end_val = self.eval_expr(parts[1].trim())?;
            let end = match &end_val {
                RustMathValue::Point3D(p) => (p.x, p.y, p.z),
                RustMathValue::List(v) if v.len() == 3 => {
                    (self.value_to_f64(&v[0])?, self.value_to_f64(&v[1])?, self.value_to_f64(&v[2])?)
                }
                _ => return Err(EvalError::new("TypeError", "end must be Point3D or [x,y,z]")),
            };
            let r = self.eval_to_f64(parts[2].trim())?;
            (start, end, r)
        } else {
            // Default: vertical unit cylinder
            ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
        };

        // Generate cylinder as parametric surface
        let n_theta = 30;
        let n_h = 10;
        let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        // Compute axis direction
        let axis = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
        let axis_len = (axis.0*axis.0 + axis.1*axis.1 + axis.2*axis.2).sqrt();

        // Find perpendicular vectors
        let (perp1, perp2) = if axis.0.abs() < 0.9 {
            let p1 = (0.0, -axis.2, axis.1);
            let p1_len = (p1.1*p1.1 + p1.2*p1.2).sqrt();
            let p1 = (p1.0/p1_len, p1.1/p1_len, p1.2/p1_len);
            let p2 = (axis.1*p1.2 - axis.2*p1.1, axis.2*p1.0 - axis.0*p1.2, axis.0*p1.1 - axis.1*p1.0);
            let p2_len = (p2.0*p2.0 + p2.1*p2.1 + p2.2*p2.2).sqrt();
            (p1, (p2.0/p2_len, p2.1/p2_len, p2.2/p2_len))
        } else {
            let p1 = (-axis.1, axis.0, 0.0);
            let p1_len = (p1.0*p1.0 + p1.1*p1.1).sqrt();
            let p1 = (p1.0/p1_len, p1.1/p1_len, p1.2/p1_len);
            let p2 = (axis.1*p1.2 - axis.2*p1.1, axis.2*p1.0 - axis.0*p1.2, axis.0*p1.1 - axis.1*p1.0);
            let p2_len = (p2.0*p2.0 + p2.1*p2.1 + p2.2*p2.2).sqrt();
            (p1, (p2.0/p2_len, p2.1/p2_len, p2.2/p2_len))
        };

        for j in 0..=n_h {
            let t = j as f64 / n_h as f64;
            let base = (
                start.0 + t * axis.0,
                start.1 + t * axis.1,
                start.2 + t * axis.2,
            );
            let mut row = Vec::new();
            for i in 0..=n_theta {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / n_theta as f64;
                let x = base.0 + radius * (theta.cos() * perp1.0 + theta.sin() * perp2.0);
                let y = base.1 + radius * (theta.cos() * perp1.1 + theta.sin() * perp2.1);
                let z = base.2 + radius * (theta.cos() * perp1.2 + theta.sin() * perp2.2);
                row.push((x, y, z));
            }
            surface_points.push(row);
        }

        let svg = self.generate_parametric_surface_svg(&surface_points, "Cylinder");
        let description = format!("Cylinder radius={:.2}, height={:.2}", radius, axis_len);
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate revolution_plot3d
    /// revolution_plot3d(curve, (t, tmin, tmax), axis='z')
    fn eval_revolution_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let parts = self.split_at_depth_zero(args_str.trim(), ',');
        if parts.len() < 2 {
            return Err(EvalError::new("ArgumentError",
                "revolution_plot3d requires curve and (t, tmin, tmax)"));
        }

        // Parse curve: either (r(t), z(t)) or just r(t) where z = t
        let curve_str = parts[0].trim();
        let (r_expr_str, h_expr_str) = if curve_str.starts_with('(') && curve_str.ends_with(')') {
            let inner = &curve_str[1..curve_str.len()-1];
            let curve_parts = self.split_at_depth_zero(inner, ',');
            if curve_parts.len() >= 2 {
                (curve_parts[0].trim(), curve_parts[1].trim())
            } else {
                (curve_parts[0].trim(), "t")
            }
        } else {
            (curve_str, "t")
        };

        // Parse expressions symbolically
        let r_expr = self.parse_symbolic_arg(r_expr_str)?;
        let h_expr = self.parse_symbolic_arg(h_expr_str)?;

        // Parse t range
        let t_range_str = parts[1].trim();
        let t_range_str = t_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "t range must be (var, min, max)"))?;
        let t_parts = self.split_at_depth_zero(t_range_str, ',');
        if t_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "t range must have 3 parts"));
        }
        let t_var_name = t_parts[0].trim();
        let t_min = self.eval_to_f64(t_parts[1].trim())?;
        let t_max = self.eval_to_f64(t_parts[2].trim())?;
        let t_sym = Symbol::new(t_var_name);

        // Parse axis (default z)
        let axis = if parts.len() > 2 {
            let axis_str = parts[2].trim().trim_matches('"').trim_matches('\'');
            axis_str.to_lowercase()
        } else {
            "z".to_string()
        };

        // Generate surface of revolution
        let n_t = 50;
        let n_theta = 40;
        let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        for j in 0..=n_theta {
            let theta = 2.0 * std::f64::consts::PI * j as f64 / n_theta as f64;
            let mut row = Vec::new();
            for i in 0..=n_t {
                let t = t_min + (t_max - t_min) * i as f64 / n_t as f64;

                let r_eval = r_expr.substitute(&t_sym, &Expr::Real(t));
                let h_eval = h_expr.substitute(&t_sym, &Expr::Real(t));
                let r = try_eval_to_f64(&r_eval).unwrap_or(f64::NAN);
                let h = try_eval_to_f64(&h_eval).unwrap_or(f64::NAN);

                let (x, y, z) = match axis.as_str() {
                    "x" => (h, r * theta.cos(), r * theta.sin()),
                    "y" => (r * theta.cos(), h, r * theta.sin()),
                    _ => (r * theta.cos(), r * theta.sin(), h), // z axis
                };
                row.push((x, y, z));
            }
            surface_points.push(row);
        }

        let svg = self.generate_parametric_surface_svg(&surface_points, "Surface of Revolution");
        let description = "Surface of revolution".to_string();
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate spherical_plot3d
    /// spherical_plot3d(r(theta, phi), (theta, 0, pi), (phi, 0, 2*pi))
    fn eval_spherical_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let parts = self.split_at_depth_zero(args_str.trim(), ',');
        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "spherical_plot3d requires r(theta,phi), (theta, min, max), (phi, min, max)"));
        }

        // Parse expression symbolically
        let r_expr = self.parse_symbolic_arg(parts[0].trim())?;

        // Parse theta range
        let theta_range_str = parts[1].trim();
        let theta_range_str = theta_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "theta range must be (var, min, max)"))?;
        let theta_parts = self.split_at_depth_zero(theta_range_str, ',');
        if theta_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "theta range must have 3 parts"));
        }
        let theta_var_name = theta_parts[0].trim();
        let theta_min = self.eval_to_f64(theta_parts[1].trim())?;
        let theta_max = self.eval_to_f64(theta_parts[2].trim())?;
        let theta_sym = Symbol::new(theta_var_name);

        // Parse phi range
        let phi_range_str = parts[2].trim();
        let phi_range_str = phi_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "phi range must be (var, min, max)"))?;
        let phi_parts = self.split_at_depth_zero(phi_range_str, ',');
        if phi_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "phi range must have 3 parts"));
        }
        let phi_var_name = phi_parts[0].trim();
        let phi_min = self.eval_to_f64(phi_parts[1].trim())?;
        let phi_max = self.eval_to_f64(phi_parts[2].trim())?;
        let phi_sym = Symbol::new(phi_var_name);

        // Generate surface
        let n_theta = 40;
        let n_phi = 40;
        let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        for j in 0..=n_phi {
            let phi = phi_min + (phi_max - phi_min) * j as f64 / n_phi as f64;
            let mut row = Vec::new();
            for i in 0..=n_theta {
                let theta = theta_min + (theta_max - theta_min) * i as f64 / n_theta as f64;

                let r_eval = r_expr.substitute(&theta_sym, &Expr::Real(theta))
                                   .substitute(&phi_sym, &Expr::Real(phi));
                let r = try_eval_to_f64(&r_eval).unwrap_or(1.0);

                let x = r * theta.sin() * phi.cos();
                let y = r * theta.sin() * phi.sin();
                let z = r * theta.cos();
                row.push((x, y, z));
            }
            surface_points.push(row);
        }

        let svg = self.generate_parametric_surface_svg(&surface_points, "Spherical Plot");
        let description = "Spherical coordinate surface".to_string();
        Ok(RustMathValue::Plot { description, svg })
    }

    /// Evaluate cylindrical_plot3d
    /// cylindrical_plot3d(z(r, theta), (r, rmin, rmax), (theta, 0, 2*pi))
    fn eval_cylindrical_plot3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let parts = self.split_at_depth_zero(args_str.trim(), ',');
        if parts.len() < 3 {
            return Err(EvalError::new("ArgumentError",
                "cylindrical_plot3d requires z(r,theta), (r, min, max), (theta, min, max)"));
        }

        // Parse expression symbolically
        let z_expr = self.parse_symbolic_arg(parts[0].trim())?;

        // Parse r range
        let r_range_str = parts[1].trim();
        let r_range_str = r_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "r range must be (var, min, max)"))?;
        let r_parts = self.split_at_depth_zero(r_range_str, ',');
        if r_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "r range must have 3 parts"));
        }
        let r_var_name = r_parts[0].trim();
        let r_min = self.eval_to_f64(r_parts[1].trim())?;
        let r_max = self.eval_to_f64(r_parts[2].trim())?;
        let r_sym = Symbol::new(r_var_name);

        // Parse theta range
        let theta_range_str = parts[2].trim();
        let theta_range_str = theta_range_str.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
            .ok_or_else(|| EvalError::new("SyntaxError", "theta range must be (var, min, max)"))?;
        let theta_parts = self.split_at_depth_zero(theta_range_str, ',');
        if theta_parts.len() != 3 {
            return Err(EvalError::new("ArgumentError", "theta range must have 3 parts"));
        }
        let theta_var_name = theta_parts[0].trim();
        let theta_min = self.eval_to_f64(theta_parts[1].trim())?;
        let theta_max = self.eval_to_f64(theta_parts[2].trim())?;
        let theta_sym = Symbol::new(theta_var_name);

        // Generate surface
        let n_r = 30;
        let n_theta = 40;
        let mut surface_points: Vec<Vec<(f64, f64, f64)>> = Vec::new();

        for j in 0..=n_theta {
            let theta = theta_min + (theta_max - theta_min) * j as f64 / n_theta as f64;
            let mut row = Vec::new();
            for i in 0..=n_r {
                let r = r_min + (r_max - r_min) * i as f64 / n_r as f64;

                let z_eval = z_expr.substitute(&r_sym, &Expr::Real(r))
                                   .substitute(&theta_sym, &Expr::Real(theta));
                let z = try_eval_to_f64(&z_eval).unwrap_or(f64::NAN);

                let x = r * theta.cos();
                let y = r * theta.sin();
                row.push((x, y, z));
            }
            surface_points.push(row);
        }

        let svg = self.generate_parametric_surface_svg(&surface_points, "Cylindrical Plot");
        let description = "Cylindrical coordinate surface".to_string();
        Ok(RustMathValue::Plot { description, svg })
    }

    // ===== 3D SVG GENERATION HELPERS =====

    /// Generate SVG for a 3D surface z = f(x, y) with SageMath-style defaults
    fn generate_surface_svg(&self, z_values: &Vec<Vec<f64>>, x_min: f64, x_max: f64,
                            y_min: f64, y_max: f64, title: &str) -> String {
        // Use default options (SageMath-like)
        self.generate_surface_svg_with_options(z_values, x_min, x_max, y_min, y_max, title, &Plot3DOptions::default())
    }

    /// Generate SVG for a 3D surface with configurable options
    fn generate_surface_svg_with_options(&self, z_values: &Vec<Vec<f64>>, x_min: f64, x_max: f64,
                            y_min: f64, y_max: f64, title: &str, opts: &Plot3DOptions) -> String {
        let n_y = z_values.len();
        let n_x = if n_y > 0 { z_values[0].len() } else { 0 };
        if n_x == 0 || n_y == 0 { return String::new(); }

        // Find z bounds
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for row in z_values {
            for &z in row {
                if z.is_finite() {
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
            }
        }
        if !z_min.is_finite() { z_min = 0.0; }
        if !z_max.is_finite() { z_max = 1.0; }
        let z_range = if (z_max - z_min).abs() < 1e-10 { 1.0 } else { z_max - z_min };

        let width = 600.0;
        let height = 500.0;
        let scale = 150.0;

        // Light direction for shading (normalized, pointing from upper-left-front)
        let light_dir = (0.5_f64, -0.5_f64, 0.7_f64);
        let light_len = (light_dir.0.powi(2) + light_dir.1.powi(2) + light_dir.2.powi(2)).sqrt();
        let light_dir = (light_dir.0 / light_len, light_dir.1 / light_len, light_dir.2 / light_len);

        // Parse base color
        let base_color = Self::parse_hex_color(&opts.color);

        let dx = (x_max - x_min) / (n_x - 1) as f64;
        let dy = (y_max - y_min) / (n_y - 1) as f64;

        // Build mesh data for interactive rotation
        // Normalize all coordinates to [-0.5, 0.5] range
        let mut quads_data: Vec<((f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), String)> = Vec::new();

        for j in 0..n_y-1 {
            for i in 0..n_x-1 {
                let x0 = x_min + i as f64 * dx;
                let x1 = x_min + (i + 1) as f64 * dx;
                let y0 = y_min + j as f64 * dy;
                let y1 = y_min + (j + 1) as f64 * dy;

                let z00 = z_values[j][i];
                let z10 = z_values[j][i+1];
                let z01 = z_values[j+1][i];
                let z11 = z_values[j+1][i+1];

                if !z00.is_finite() || !z10.is_finite() || !z01.is_finite() || !z11.is_finite() {
                    continue;
                }

                // Normalize coordinates to [-0.5, 0.5]
                let nx00 = (x0 - x_min) / (x_max - x_min) - 0.5;
                let ny00 = (y0 - y_min) / (y_max - y_min) - 0.5;
                let nz00 = (z00 - z_min) / z_range - 0.5;

                let nx10 = (x1 - x_min) / (x_max - x_min) - 0.5;
                let ny10 = (y0 - y_min) / (y_max - y_min) - 0.5;
                let nz10 = (z10 - z_min) / z_range - 0.5;

                let nx01 = (x0 - x_min) / (x_max - x_min) - 0.5;
                let ny01 = (y1 - y_min) / (y_max - y_min) - 0.5;
                let nz01 = (z01 - z_min) / z_range - 0.5;

                let nx11 = (x1 - x_min) / (x_max - x_min) - 0.5;
                let ny11 = (y1 - y_min) / (y_max - y_min) - 0.5;
                let nz11 = (z11 - z_min) / z_range - 0.5;

                // Compute surface normal for shading
                let v1 = (dx, 0.0, z10 - z00);
                let v2 = (0.0, dy, z01 - z00);
                let normal_x = v1.1 * v2.2 - v1.2 * v2.1;
                let normal_y = v1.2 * v2.0 - v1.0 * v2.2;
                let normal_z = v1.0 * v2.1 - v1.1 * v2.0;
                let n_len = (normal_x * normal_x + normal_y * normal_y + normal_z * normal_z).sqrt();
                let (normal_x, normal_y, normal_z) = if n_len > 1e-10 {
                    (normal_x / n_len, normal_y / n_len, normal_z / n_len)
                } else {
                    (0.0, 0.0, 1.0)
                };

                let intensity = if opts.shading {
                    let dot = (normal_x * light_dir.0 + normal_y * light_dir.1 + normal_z * light_dir.2).abs();
                    0.3 + 0.7 * dot
                } else {
                    1.0
                };

                let shaded_color = Self::shade_color(&base_color, intensity);

                quads_data.push((
                    (nx00, ny00, nz00),
                    (nx10, ny10, nz10),
                    (nx01, ny01, nz01),
                    (nx11, ny11, nz11),
                    shaded_color,
                ));
            }
        }

        // Build bounding box corners (normalized)
        let box_corners = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
            (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        ];

        // Start SVG with interactive rotation
        let svg = if opts.interactive {
            // Build mesh data as JSON for JavaScript
            let mut mesh_json = String::from("[");
            for (idx, (v0, v1, v2, v3, color)) in quads_data.iter().enumerate() {
                if idx > 0 { mesh_json.push(','); }
                mesh_json.push_str(&format!(
                    "{{\"v\":[[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}]],\"c\":\"{}\"}}",
                    v0.0, v0.1, v0.2, v1.0, v1.1, v1.2, v3.0, v3.1, v3.2, v2.0, v2.1, v2.2, color
                ));
            }
            mesh_json.push(']');

            // Box corners JSON
            let box_json = format!(
                "[[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}]]",
                box_corners[0].0, box_corners[0].1, box_corners[0].2,
                box_corners[1].0, box_corners[1].1, box_corners[1].2,
                box_corners[2].0, box_corners[2].1, box_corners[2].2,
                box_corners[3].0, box_corners[3].1, box_corners[3].2,
                box_corners[4].0, box_corners[4].1, box_corners[4].2,
                box_corners[5].0, box_corners[5].1, box_corners[5].2,
                box_corners[6].0, box_corners[6].1, box_corners[6].2,
                box_corners[7].0, box_corners[7].1, box_corners[7].2,
            );

            let mesh_enabled = if opts.mesh { "true" } else { "false" };

            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="cursor: grab;">
<rect width="{width}" height="{height}" fill="{bg}"/>
<text x="{title_x}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">{title}</text>
<g id="mesh-group"></g>
<g id="box-group"></g>
<text x="{title_x}" y="{hint_y}" font-size="10" fill="#999" text-anchor="middle">Drag to rotate, scroll to zoom</text>
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.currentScript.parentElement;
    var meshGroup = svg.getElementById('mesh-group');
    var boxGroup = svg.getElementById('box-group');
    var mesh = {mesh_json};
    var boxCorners = {box_json};
    var boxEdges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    var width = {width}, height = {height}, scale = {scale}, centerX = width/2, centerY = height/2 + 30;
    var theta = 0.7854, phi = 0.6155; // Initial rotation (45°, 35°)
    var zoom = 1.0, isDragging = false, lastX, lastY;
    var showMesh = {mesh_enabled};

    function rotate(x, y, z) {{
        var cosT = Math.cos(theta), sinT = Math.sin(theta);
        var cosP = Math.cos(phi), sinP = Math.sin(phi);
        var x1 = x * cosT - y * sinT;
        var y1 = x * sinT + y * cosT;
        var z1 = z;
        var x2 = x1;
        var y2 = y1 * cosP - z1 * sinP;
        var z2 = y1 * sinP + z1 * cosP;
        return [x2, y2, z2];
    }}

    function project(x, y, z) {{
        var r = rotate(x, y, z);
        return {{
            px: centerX + scale * zoom * r[0],
            py: centerY - scale * zoom * r[2],
            depth: r[1]
        }};
    }}

    function render() {{
        // Project and sort quads
        var quads = mesh.map(function(q, idx) {{
            var pts = q.v.map(function(v) {{ return project(v[0], v[1], v[2]); }});
            var avgDepth = (pts[0].depth + pts[1].depth + pts[2].depth + pts[3].depth) / 4;
            return {{ pts: pts, color: q.c, depth: avgDepth, idx: idx }};
        }});
        quads.sort(function(a, b) {{ return a.depth - b.depth; }});

        // Clear and redraw mesh
        meshGroup.innerHTML = '';
        quads.forEach(function(q) {{
            var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            var points = q.pts.map(function(p) {{ return p.px.toFixed(1) + ',' + p.py.toFixed(1); }}).join(' ');
            poly.setAttribute('points', points);
            poly.setAttribute('fill', q.color);
            poly.setAttribute('fill-opacity', '{opacity}');
            if (showMesh) {{
                poly.setAttribute('stroke', '{mesh_color}');
                poly.setAttribute('stroke-width', '0.5');
            }}
            meshGroup.appendChild(poly);
        }});

        // Redraw bounding box
        boxGroup.innerHTML = '';
        var projBox = boxCorners.map(function(c) {{ return project(c[0], c[1], c[2]); }});
        boxEdges.forEach(function(e) {{
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', projBox[e[0]].px.toFixed(1));
            line.setAttribute('y1', projBox[e[0]].py.toFixed(1));
            line.setAttribute('x2', projBox[e[1]].px.toFixed(1));
            line.setAttribute('y2', projBox[e[1]].py.toFixed(1));
            line.setAttribute('stroke', '#888');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,2');
            boxGroup.appendChild(line);
        }});

        // Axis labels
        var labels = [['x', 1], ['y', 2], ['z', 4]];
        labels.forEach(function(l) {{
            var p = projBox[l[1]];
            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (p.px + (l[0]==='y' ? -12 : 8)).toFixed(0));
            text.setAttribute('y', (p.py + (l[0]==='z' ? -5 : 4)).toFixed(0));
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');
            text.setAttribute('font-weight', 'bold');
            text.textContent = l[0];
            boxGroup.appendChild(text);
        }});
    }}

    svg.addEventListener('mousedown', function(e) {{
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
        svg.style.cursor = 'grabbing';
    }});

    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            var dx = e.clientX - lastX;
            var dy = e.clientY - lastY;
            theta += dx * 0.01;
            phi += dy * 0.01;
            phi = Math.max(-1.5, Math.min(1.5, phi));
            lastX = e.clientX;
            lastY = e.clientY;
            render();
        }}
    }});

    svg.addEventListener('mouseup', function() {{
        isDragging = false;
        svg.style.cursor = 'grab';
    }});

    svg.addEventListener('mouseleave', function() {{
        isDragging = false;
        svg.style.cursor = 'grab';
    }});

    svg.addEventListener('wheel', function(e) {{
        e.preventDefault();
        zoom *= e.deltaY > 0 ? 0.9 : 1.1;
        zoom = Math.max(0.3, Math.min(3.0, zoom));
        render();
    }});

    render();
}})();
]]></script></svg>"##,
                width = width,
                height = height,
                bg = opts.background,
                title_x = width / 2.0,
                title = title,
                hint_y = height - 10.0,
                mesh_json = mesh_json,
                box_json = box_json,
                scale = scale,
                opacity = opts.opacity,
                mesh_color = opts.mesh_color,
                mesh_enabled = mesh_enabled,
            )
        } else {
            // Non-interactive: use static isometric projection
            let center_x = width / 2.0;
            let center_y = height / 2.0 + 30.0;

            let project = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
                let px = center_x + scale * (x - y) * 0.866;
                let py = center_y - scale * (x + y) * 0.5 - scale * z;
                let depth = x + y - z;
                (px, py, depth)
            };

            let mut svg_str = format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
                width, height, width, height
            );

            svg_str.push_str(&format!(r#"<rect width="{}" height="{}" fill="{}"/>"#, width * 2.0, height * 2.0, opts.background));
            svg_str.push_str(&format!(
                r##"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">{}</text>"##,
                width / 2.0, title
            ));

            // Collect and sort quads
            let mut quads: Vec<(f64, String)> = Vec::new();
            for (v0, v1, v2, v3, color) in &quads_data {
                let (px0, py0, d0) = project(v0.0, v0.1, v0.2);
                let (px1, py1, _) = project(v1.0, v1.1, v1.2);
                let (px2, py2, _) = project(v2.0, v2.1, v2.2);
                let (px3, py3, _) = project(v3.0, v3.1, v3.2);

                let stroke = if opts.mesh {
                    format!(r#" stroke="{}" stroke-width="0.5""#, opts.mesh_color)
                } else {
                    String::new()
                };

                let quad_svg = format!(
                    r##"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="{}" fill-opacity="{}"{}/>"##,
                    px0, py0, px1, py1, px2, py2, px3, py3, color, opts.opacity, stroke
                );
                quads.push((d0, quad_svg));
            }

            quads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, quad_svg) in quads {
                svg_str.push_str(&quad_svg);
            }

            // Draw bounding box
            if opts.axes {
                let projected: Vec<(f64, f64)> = box_corners.iter()
                    .map(|&(x, y, z)| { let (px, py, _) = project(x, y, z); (px, py) })
                    .collect();

                let edges = [
                    (0, 1), (2, 3), (4, 5), (6, 7),
                    (0, 2), (1, 3), (4, 6), (5, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ];

                for (i, j) in edges {
                    svg_str.push_str(&format!(
                        r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#888" stroke-width="1" stroke-dasharray="4,2"/>"##,
                        projected[i].0, projected[i].1, projected[j].0, projected[j].1
                    ));
                }

                let (xx, xy) = projected[1];
                let (yx, yy) = projected[2];
                let (zx, zy) = projected[4];
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">x</text>"##, xx + 8.0, xy + 4.0));
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">y</text>"##, yx - 12.0, yy + 4.0));
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">z</text>"##, zx - 12.0, zy - 5.0));
            }

            svg_str.push_str("</svg>");
            return svg_str;
        };

        svg
    }

    /// Parse hex color string to (r, g, b) tuple
    fn parse_hex_color(hex: &str) -> (u8, u8, u8) {
        let hex = hex.trim_start_matches('#');
        if hex.len() >= 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(102);
            let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(102);
            let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);
            (r, g, b)
        } else {
            (102, 102, 255)  // Default SageMath blue
        }
    }

    /// Apply shading intensity to a color
    fn shade_color(color: &(u8, u8, u8), intensity: f64) -> String {
        let r = (color.0 as f64 * intensity).min(255.0) as u8;
        let g = (color.1 as f64 * intensity).min(255.0) as u8;
        let b = (color.2 as f64 * intensity).min(255.0) as u8;
        format!("#{:02x}{:02x}{:02x}", r, g, b)
    }

    /// Generate SVG for a 3D parametric curve with rotation support
    fn generate_3d_curve_svg(&self, coords: &[(f64, f64, f64)], title: &str) -> String {
        if coords.is_empty() { return String::new(); }

        let width = 500.0;
        let height = 400.0;
        let scale = 100.0;

        // Generate unique ID for this SVG
        use std::time::{SystemTime, UNIX_EPOCH};
        let svg_id = format!("curve3d-{}", SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0));

        // Find bounds
        let (min_x, max_x) = coords.iter().map(|p| p.0).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));
        let (min_y, max_y) = coords.iter().map(|p| p.1).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));
        let (min_z, max_z) = coords.iter().map(|p| p.2).fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), v| (min.min(v), max.max(v)));

        let range_x = if (max_x - min_x).abs() < 1e-10 { 1.0 } else { max_x - min_x };
        let range_y = if (max_y - min_y).abs() < 1e-10 { 1.0 } else { max_y - min_y };
        let range_z = if (max_z - min_z).abs() < 1e-10 { 1.0 } else { max_z - min_z };
        let max_range = range_x.max(range_y).max(range_z);
        let mid_x = (min_x + max_x) / 2.0;
        let mid_y = (min_y + max_y) / 2.0;
        let mid_z = (min_z + max_z) / 2.0;

        // Normalize coordinates
        let normalized: Vec<(f64, f64, f64)> = coords.iter()
            .map(|&(x, y, z)| ((x - mid_x) / max_range, (y - mid_y) / max_range, (z - mid_z) / max_range))
            .collect();

        // Build points JSON
        let mut points_json = String::from("[");
        for (idx, &(nx, ny, nz)) in normalized.iter().enumerate() {
            if idx > 0 { points_json.push(','); }
            points_json.push_str(&format!("[{:.4},{:.4},{:.4}]", nx, ny, nz));
        }
        points_json.push(']');

        // Bounding box corners (normalized)
        let box_json = "[[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[-0.5,-0.5,0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5]]";

        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" id="{svg_id}" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="cursor: grab;">
<rect width="{width}" height="{height}" fill="white"/>
<text x="{title_x}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">{title}</text>
<g id="{svg_id}-box"></g>
<path id="{svg_id}-curve" fill="none" stroke="#2563eb" stroke-width="2"/>
<text x="{title_x}" y="{hint_y}" font-size="10" fill="#999" text-anchor="middle">Drag to rotate, scroll to zoom</text>
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.getElementById('{svg_id}');
    if (!svg) return;
    var boxGroup = document.getElementById('{svg_id}-box');
    var curvePath = document.getElementById('{svg_id}-curve');
    var points = {points_json};
    var boxCorners = {box_json};
    var boxEdges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    var width = {width}, height = {height}, scale = {scale}, centerX = width/2, centerY = height/2;
    var theta = 0.7854, phi = 0.6155;
    var zoom = 1.0, isDragging = false, lastX, lastY;

    function rotate(x, y, z) {{
        var cosT = Math.cos(theta), sinT = Math.sin(theta);
        var cosP = Math.cos(phi), sinP = Math.sin(phi);
        var x1 = x * cosT - y * sinT, y1 = x * sinT + y * cosT, z1 = z;
        return [x1, y1 * cosP - z1 * sinP, y1 * sinP + z1 * cosP];
    }}

    function project(x, y, z) {{
        var r = rotate(x, y, z);
        return {{ px: centerX + scale * zoom * r[0], py: centerY - scale * zoom * r[2] }};
    }}

    function render() {{
        var d = points.map(function(p, i) {{
            var proj = project(p[0], p[1], p[2]);
            return (i === 0 ? 'M ' : ' L ') + proj.px.toFixed(1) + ' ' + proj.py.toFixed(1);
        }}).join('');
        curvePath.setAttribute('d', d);

        boxGroup.innerHTML = '';
        var projBox = boxCorners.map(function(c) {{ return project(c[0], c[1], c[2]); }});
        boxEdges.forEach(function(e) {{
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', projBox[e[0]].px.toFixed(1));
            line.setAttribute('y1', projBox[e[0]].py.toFixed(1));
            line.setAttribute('x2', projBox[e[1]].px.toFixed(1));
            line.setAttribute('y2', projBox[e[1]].py.toFixed(1));
            line.setAttribute('stroke', '#888');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,2');
            boxGroup.appendChild(line);
        }});

        var labels = [['x', 1], ['y', 2], ['z', 4]];
        labels.forEach(function(l) {{
            var p = projBox[l[1]];
            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (p.px + (l[0]==='y' ? -12 : 8)).toFixed(0));
            text.setAttribute('y', (p.py + (l[0]==='z' ? -5 : 4)).toFixed(0));
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');
            text.setAttribute('font-weight', 'bold');
            text.textContent = l[0];
            boxGroup.appendChild(text);
        }});
    }}

    svg.addEventListener('mousedown', function(e) {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; svg.style.cursor = 'grabbing'; }});
    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            theta += (e.clientX - lastX) * 0.01;
            phi += (e.clientY - lastY) * 0.01;
            phi = Math.max(-1.5, Math.min(1.5, phi));
            lastX = e.clientX; lastY = e.clientY;
            render();
        }}
    }});
    svg.addEventListener('mouseup', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('mouseleave', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('wheel', function(e) {{ e.preventDefault(); zoom *= e.deltaY > 0 ? 0.9 : 1.1; zoom = Math.max(0.3, Math.min(3.0, zoom)); render(); }});
    render();
}})();
]]></script></svg>"##,
            svg_id = svg_id,
            width = width,
            height = height,
            title_x = width / 2.0,
            title = title,
            hint_y = height - 10.0,
            points_json = points_json,
            box_json = box_json,
            scale = scale,
        )
    }

    /// Generate SVG for a 3D parametric surface with SageMath-style defaults
    fn generate_parametric_surface_svg(&self, surface: &Vec<Vec<(f64, f64, f64)>>, title: &str) -> String {
        self.generate_parametric_surface_svg_with_options(surface, title, &Plot3DOptions::default())
    }

    /// Generate SVG for a 3D parametric surface with configurable options
    fn generate_parametric_surface_svg_with_options(&self, surface: &Vec<Vec<(f64, f64, f64)>>, title: &str, opts: &Plot3DOptions) -> String {
        if surface.is_empty() || surface[0].is_empty() { return String::new(); }

        let width = 600.0;
        let height = 500.0;
        let scale = 140.0;

        // Light direction for shading
        let light_dir = (0.5_f64, -0.5_f64, 0.7_f64);
        let light_len = (light_dir.0.powi(2) + light_dir.1.powi(2) + light_dir.2.powi(2)).sqrt();
        let light_dir = (light_dir.0 / light_len, light_dir.1 / light_len, light_dir.2 / light_len);

        // Parse base color
        let base_color = Self::parse_hex_color(&opts.color);

        // Find bounds
        let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY; let mut max_z = f64::NEG_INFINITY;

        for row in surface {
            for &(x, y, z) in row {
                if x.is_finite() { min_x = min_x.min(x); max_x = max_x.max(x); }
                if y.is_finite() { min_y = min_y.min(y); max_y = max_y.max(y); }
                if z.is_finite() { min_z = min_z.min(z); max_z = max_z.max(z); }
            }
        }

        let max_range = (max_x - min_x).max(max_y - min_y).max(max_z - min_z).max(1e-10);
        let mid_x = (min_x + max_x) / 2.0;
        let mid_y = (min_y + max_y) / 2.0;
        let mid_z = (min_z + max_z) / 2.0;

        // Collect normalized quads with their shaded colors
        let mut quads_data: Vec<((f64, f64, f64), (f64, f64, f64), (f64, f64, f64), (f64, f64, f64), String)> = Vec::new();

        for j in 0..surface.len()-1 {
            for i in 0..surface[j].len()-1 {
                let p00 = surface[j][i];
                let p10 = surface[j][i+1];
                let p01 = surface[j+1][i];
                let p11 = surface[j+1][i+1];

                if !p00.0.is_finite() || !p10.0.is_finite() || !p01.0.is_finite() || !p11.0.is_finite() {
                    continue;
                }

                // Normalize coordinates
                let n00 = ((p00.0 - mid_x) / max_range, (p00.1 - mid_y) / max_range, (p00.2 - mid_z) / max_range);
                let n10 = ((p10.0 - mid_x) / max_range, (p10.1 - mid_y) / max_range, (p10.2 - mid_z) / max_range);
                let n01 = ((p01.0 - mid_x) / max_range, (p01.1 - mid_y) / max_range, (p01.2 - mid_z) / max_range);
                let n11 = ((p11.0 - mid_x) / max_range, (p11.1 - mid_y) / max_range, (p11.2 - mid_z) / max_range);

                // Compute surface normal for shading
                let v1 = (p10.0 - p00.0, p10.1 - p00.1, p10.2 - p00.2);
                let v2 = (p01.0 - p00.0, p01.1 - p00.1, p01.2 - p00.2);
                let nx = v1.1 * v2.2 - v1.2 * v2.1;
                let ny = v1.2 * v2.0 - v1.0 * v2.2;
                let nz = v1.0 * v2.1 - v1.1 * v2.0;
                let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
                let (nx, ny, nz) = if n_len > 1e-10 {
                    (nx / n_len, ny / n_len, nz / n_len)
                } else {
                    (0.0, 0.0, 1.0)
                };

                let intensity = if opts.shading {
                    let dot = (nx * light_dir.0 + ny * light_dir.1 + nz * light_dir.2).abs();
                    0.3 + 0.7 * dot
                } else {
                    1.0
                };

                let shaded_color = Self::shade_color(&base_color, intensity);
                quads_data.push((n00, n10, n01, n11, shaded_color));
            }
        }

        // Bounding box corners (normalized)
        let box_corners = [
            (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
            (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        ];

        let svg = if opts.interactive {
            // Build mesh data as JSON
            let mut mesh_json = String::from("[");
            for (idx, (v0, v1, v2, v3, color)) in quads_data.iter().enumerate() {
                if idx > 0 { mesh_json.push(','); }
                mesh_json.push_str(&format!(
                    "{{\"v\":[[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}]],\"c\":\"{}\"}}",
                    v0.0, v0.1, v0.2, v1.0, v1.1, v1.2, v3.0, v3.1, v3.2, v2.0, v2.1, v2.2, color
                ));
            }
            mesh_json.push(']');

            let box_json = format!(
                "[[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}]]",
                box_corners[0].0, box_corners[0].1, box_corners[0].2,
                box_corners[1].0, box_corners[1].1, box_corners[1].2,
                box_corners[2].0, box_corners[2].1, box_corners[2].2,
                box_corners[3].0, box_corners[3].1, box_corners[3].2,
                box_corners[4].0, box_corners[4].1, box_corners[4].2,
                box_corners[5].0, box_corners[5].1, box_corners[5].2,
                box_corners[6].0, box_corners[6].1, box_corners[6].2,
                box_corners[7].0, box_corners[7].1, box_corners[7].2,
            );

            let mesh_enabled = if opts.mesh { "true" } else { "false" };

            format!(
                r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="cursor: grab;">
<rect width="{width}" height="{height}" fill="{bg}"/>
<text x="{title_x}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">{title}</text>
<g id="mesh-group"></g>
<g id="box-group"></g>
<text x="{title_x}" y="{hint_y}" font-size="10" fill="#999" text-anchor="middle">Drag to rotate, scroll to zoom</text>
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.currentScript.parentElement;
    var meshGroup = svg.getElementById('mesh-group');
    var boxGroup = svg.getElementById('box-group');
    var mesh = {mesh_json};
    var boxCorners = {box_json};
    var boxEdges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    var width = {width}, height = {height}, scale = {scale}, centerX = width/2, centerY = height/2 + 30;
    var theta = 0.7854, phi = 0.6155;
    var zoom = 1.0, isDragging = false, lastX, lastY;
    var showMesh = {mesh_enabled};

    function rotate(x, y, z) {{
        var cosT = Math.cos(theta), sinT = Math.sin(theta);
        var cosP = Math.cos(phi), sinP = Math.sin(phi);
        var x1 = x * cosT - y * sinT;
        var y1 = x * sinT + y * cosT;
        var z1 = z;
        var x2 = x1;
        var y2 = y1 * cosP - z1 * sinP;
        var z2 = y1 * sinP + z1 * cosP;
        return [x2, y2, z2];
    }}

    function project(x, y, z) {{
        var r = rotate(x, y, z);
        return {{ px: centerX + scale * zoom * r[0], py: centerY - scale * zoom * r[2], depth: r[1] }};
    }}

    function render() {{
        var quads = mesh.map(function(q) {{
            var pts = q.v.map(function(v) {{ return project(v[0], v[1], v[2]); }});
            var avgDepth = (pts[0].depth + pts[1].depth + pts[2].depth + pts[3].depth) / 4;
            return {{ pts: pts, color: q.c, depth: avgDepth }};
        }});
        quads.sort(function(a, b) {{ return a.depth - b.depth; }});

        meshGroup.innerHTML = '';
        quads.forEach(function(q) {{
            var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            var points = q.pts.map(function(p) {{ return p.px.toFixed(1) + ',' + p.py.toFixed(1); }}).join(' ');
            poly.setAttribute('points', points);
            poly.setAttribute('fill', q.color);
            poly.setAttribute('fill-opacity', '{opacity}');
            if (showMesh) {{ poly.setAttribute('stroke', '{mesh_color}'); poly.setAttribute('stroke-width', '0.5'); }}
            meshGroup.appendChild(poly);
        }});

        boxGroup.innerHTML = '';
        var projBox = boxCorners.map(function(c) {{ return project(c[0], c[1], c[2]); }});
        boxEdges.forEach(function(e) {{
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', projBox[e[0]].px.toFixed(1));
            line.setAttribute('y1', projBox[e[0]].py.toFixed(1));
            line.setAttribute('x2', projBox[e[1]].px.toFixed(1));
            line.setAttribute('y2', projBox[e[1]].py.toFixed(1));
            line.setAttribute('stroke', '#888');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,2');
            boxGroup.appendChild(line);
        }});

        var labels = [['x', 1], ['y', 2], ['z', 4]];
        labels.forEach(function(l) {{
            var p = projBox[l[1]];
            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (p.px + (l[0]==='y' ? -12 : 8)).toFixed(0));
            text.setAttribute('y', (p.py + (l[0]==='z' ? -5 : 4)).toFixed(0));
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');
            text.setAttribute('font-weight', 'bold');
            text.textContent = l[0];
            boxGroup.appendChild(text);
        }});
    }}

    svg.addEventListener('mousedown', function(e) {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; svg.style.cursor = 'grabbing'; }});
    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            theta += (e.clientX - lastX) * 0.01;
            phi += (e.clientY - lastY) * 0.01;
            phi = Math.max(-1.5, Math.min(1.5, phi));
            lastX = e.clientX; lastY = e.clientY;
            render();
        }}
    }});
    svg.addEventListener('mouseup', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('mouseleave', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('wheel', function(e) {{ e.preventDefault(); zoom *= e.deltaY > 0 ? 0.9 : 1.1; zoom = Math.max(0.3, Math.min(3.0, zoom)); render(); }});
    render();
}})();
]]></script></svg>"##,
                width = width,
                height = height,
                bg = opts.background,
                title_x = width / 2.0,
                title = title,
                hint_y = height - 10.0,
                mesh_json = mesh_json,
                box_json = box_json,
                scale = scale,
                opacity = opts.opacity,
                mesh_color = opts.mesh_color,
                mesh_enabled = mesh_enabled,
            )
        } else {
            // Non-interactive: static isometric projection
            let center_x = width / 2.0;
            let center_y = height / 2.0 + 30.0;

            let project = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
                let px = center_x + scale * (x - y) * 0.866;
                let py = center_y - scale * (x + y) * 0.5 - scale * z;
                let depth = x + y - z;
                (px, py, depth)
            };

            let mut svg_str = format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
                width, height, width, height
            );

            svg_str.push_str(&format!(r#"<rect width="{}" height="{}" fill="{}"/>"#, width * 2.0, height * 2.0, opts.background));
            svg_str.push_str(&format!(
                r##"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">{}</text>"##,
                width / 2.0, title
            ));

            let mut quads: Vec<(f64, String)> = Vec::new();
            for (v0, v1, v2, v3, color) in &quads_data {
                let (px0, py0, d0) = project(v0.0, v0.1, v0.2);
                let (px1, py1, _) = project(v1.0, v1.1, v1.2);
                let (px2, py2, _) = project(v2.0, v2.1, v2.2);
                let (px3, py3, _) = project(v3.0, v3.1, v3.2);

                let stroke = if opts.mesh {
                    format!(r#" stroke="{}" stroke-width="0.5""#, opts.mesh_color)
                } else {
                    String::new()
                };

                let quad_svg = format!(
                    r##"<polygon points="{:.1},{:.1} {:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" fill="{}" fill-opacity="{}"{}/>"##,
                    px0, py0, px1, py1, px3, py3, px2, py2, color, opts.opacity, stroke
                );
                quads.push((d0, quad_svg));
            }

            quads.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, quad_svg) in quads {
                svg_str.push_str(&quad_svg);
            }

            if opts.axes {
                let projected: Vec<(f64, f64)> = box_corners.iter()
                    .map(|&(x, y, z)| { let (px, py, _) = project(x, y, z); (px, py) })
                    .collect();

                let edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
                for (i, j) in edges {
                    svg_str.push_str(&format!(
                        r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#888" stroke-width="1" stroke-dasharray="4,2"/>"##,
                        projected[i].0, projected[i].1, projected[j].0, projected[j].1
                    ));
                }

                let (xx, xy) = projected[1]; let (yx, yy) = projected[2]; let (zx, zy) = projected[4];
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">x</text>"##, xx + 8.0, xy + 4.0));
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">y</text>"##, yx - 12.0, yy + 4.0));
                svg_str.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">z</text>"##, zx - 12.0, zy - 5.0));
            }

            svg_str.push_str("</svg>");
            svg_str
        };

        svg
    }

    /// Generate SVG for a 3D arrow
    fn generate_3d_arrow_svg(&self, start: (f64, f64, f64), end: (f64, f64, f64), title: &str) -> String {
        let width = 400.0;
        let height = 350.0;
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        let scale = 80.0;

        let all_coords = vec![start, end, (0.0, 0.0, 0.0)];
        let min_x = all_coords.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = all_coords.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = all_coords.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = all_coords.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let min_z = all_coords.iter().map(|p| p.2).fold(f64::INFINITY, f64::min);
        let max_z = all_coords.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);

        let range = (max_x - min_x).max(max_y - min_y).max(max_z - min_z).max(1.0);
        let mid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);

        let project = |x: f64, y: f64, z: f64| -> (f64, f64) {
            let nx = (x - mid.0) / range;
            let ny = (y - mid.1) / range;
            let nz = (z - mid.2) / range;

            let px = center_x + scale * (nx - ny) * 0.866;
            let py = center_y - scale * (nx + ny) * 0.5 - scale * nz;
            (px, py)
        };

        let mut svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}" style="cursor: grab;">
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.currentScript.parentElement;
    var viewBox = svg.viewBox.baseVal;
    var zoom = 1.0, panX = 0, panY = 0, isDragging = false, startX, startY;
    svg.addEventListener('wheel', function(e) {{
        e.preventDefault();
        zoom *= e.deltaY > 0 ? 1.1 : 0.9;
        zoom = Math.max(0.5, Math.min(3.0, zoom));
        var w = {} * zoom, h = {} * zoom;
        viewBox.x = ({} - w) / 2 + panX; viewBox.y = ({} - h) / 2 + panY;
        viewBox.width = w; viewBox.height = h;
    }});
    svg.addEventListener('mousedown', function(e) {{ isDragging = true; startX = e.clientX; startY = e.clientY; }});
    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            panX -= (e.clientX - startX) * zoom; panY -= (e.clientY - startY) * zoom;
            startX = e.clientX; startY = e.clientY;
            var w = {} * zoom, h = {} * zoom;
            viewBox.x = ({} - w) / 2 + panX; viewBox.y = ({} - h) / 2 + panY;
            viewBox.width = w; viewBox.height = h;
        }}
    }});
    svg.addEventListener('mouseup', function() {{ isDragging = false; }});
    svg.addEventListener('mouseleave', function() {{ isDragging = false; }});
}})();
]]></script>"##,
            width, height, width, height, width, height, width, height, width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
            width / 2.0, title
        ));

        // Draw 3D bounding box
        let corners = [
            (min_x, min_y, min_z), (max_x, min_y, min_z),
            (min_x, max_y, min_z), (max_x, max_y, min_z),
            (min_x, min_y, max_z), (max_x, min_y, max_z),
            (min_x, max_y, max_z), (max_x, max_y, max_z),
        ];
        let projected_corners: Vec<(f64, f64)> = corners.iter()
            .map(|&(x, y, z)| project(x, y, z))
            .collect();

        // 12 edges of the box
        let edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  // x-direction edges
            (0, 2), (1, 3), (4, 6), (5, 7),  // y-direction edges
            (0, 4), (1, 5), (2, 6), (3, 7),  // z-direction edges
        ];

        for (i, j) in edges {
            svg.push_str(&format!(
                r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#888" stroke-width="1" stroke-dasharray="4,2"/>"##,
                projected_corners[i].0, projected_corners[i].1, projected_corners[j].0, projected_corners[j].1
            ));
        }

        // Axis labels
        let (xx, xy) = projected_corners[1];
        let (yx, yy) = projected_corners[2];
        let (zx, zy) = projected_corners[4];
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">x</text>"##, xx + 8.0, xy + 4.0));
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">y</text>"##, yx - 12.0, yy + 4.0));
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">z</text>"##, zx - 12.0, zy - 5.0));

        let (sx, sy) = project(start.0, start.1, start.2);
        let (ex, ey) = project(end.0, end.1, end.2);

        svg.push_str(&format!(
            r##"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#2563eb" stroke-width="3" marker-end="url(#arrowhead)"/>"##,
            sx, sy, ex, ey
        ));

        // Arrowhead marker
        svg.push_str(r##"<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/></marker></defs>"##);

        svg.push_str(&format!(
            r##"<text x="{}" y="{}" font-size="10" fill="#999" text-anchor="middle">Scroll to zoom, drag to pan</text>"##,
            width / 2.0, height - 10.0
        ));

        svg.push_str("</svg>");
        svg
    }

    /// Generate SVG for implicit surface (triangles) with rotation support
    fn generate_implicit_surface_svg(&self, triangles: &[[(f64, f64, f64); 3]], title: &str) -> String {
        if triangles.is_empty() { return String::new(); }

        let opts = Plot3DOptions::default();
        let width = 600.0;
        let height = 500.0;
        let scale = 140.0;

        // Generate unique ID for this SVG
        use std::time::{SystemTime, UNIX_EPOCH};
        let svg_id = format!("implicit3d-{}", SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0));

        // Light direction for shading
        let light_dir = (0.5_f64, -0.5_f64, 0.7_f64);
        let light_len = (light_dir.0.powi(2) + light_dir.1.powi(2) + light_dir.2.powi(2)).sqrt();
        let light_dir = (light_dir.0 / light_len, light_dir.1 / light_len, light_dir.2 / light_len);
        let base_color = Self::parse_hex_color(&opts.color);

        // Find bounds
        let mut min_x = f64::INFINITY; let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
        let mut min_z = f64::INFINITY; let mut max_z = f64::NEG_INFINITY;

        for tri in triangles {
            for &(x, y, z) in tri {
                if x.is_finite() { min_x = min_x.min(x); max_x = max_x.max(x); }
                if y.is_finite() { min_y = min_y.min(y); max_y = max_y.max(y); }
                if z.is_finite() { min_z = min_z.min(z); max_z = max_z.max(z); }
            }
        }

        let max_range = (max_x - min_x).max(max_y - min_y).max(max_z - min_z).max(1.0);
        let mid = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0);

        // Build triangles JSON with normalized coordinates and shaded colors
        let mut tris_json = String::from("[");
        for (idx, tri) in triangles.iter().enumerate() {
            if idx > 0 { tris_json.push(','); }

            // Normalize vertices
            let v0 = ((tri[0].0 - mid.0) / max_range, (tri[0].1 - mid.1) / max_range, (tri[0].2 - mid.2) / max_range);
            let v1 = ((tri[1].0 - mid.0) / max_range, (tri[1].1 - mid.1) / max_range, (tri[1].2 - mid.2) / max_range);
            let v2 = ((tri[2].0 - mid.0) / max_range, (tri[2].1 - mid.1) / max_range, (tri[2].2 - mid.2) / max_range);

            // Compute surface normal for shading
            let e1 = (tri[1].0 - tri[0].0, tri[1].1 - tri[0].1, tri[1].2 - tri[0].2);
            let e2 = (tri[2].0 - tri[0].0, tri[2].1 - tri[0].1, tri[2].2 - tri[0].2);
            let nx = e1.1 * e2.2 - e1.2 * e2.1;
            let ny = e1.2 * e2.0 - e1.0 * e2.2;
            let nz = e1.0 * e2.1 - e1.1 * e2.0;
            let n_len = (nx * nx + ny * ny + nz * nz).sqrt();
            let (nx, ny, nz) = if n_len > 1e-10 { (nx / n_len, ny / n_len, nz / n_len) } else { (0.0, 0.0, 1.0) };

            let dot = (nx * light_dir.0 + ny * light_dir.1 + nz * light_dir.2).abs();
            let intensity = 0.3 + 0.7 * dot;
            let shaded_color = Self::shade_color(&base_color, intensity);

            tris_json.push_str(&format!(
                "{{\"v\":[[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}]],\"c\":\"{}\"}}",
                v0.0, v0.1, v0.2, v1.0, v1.1, v1.2, v2.0, v2.1, v2.2, shaded_color
            ));
        }
        tris_json.push(']');

        let box_json = "[[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[-0.5,-0.5,0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5]]";

        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" id="{svg_id}" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="cursor: grab;">
<rect width="{width}" height="{height}" fill="{bg}"/>
<text x="{title_x}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#333">{title}</text>
<g id="{svg_id}-mesh"></g>
<g id="{svg_id}-box"></g>
<text x="{title_x}" y="{hint_y}" font-size="10" fill="#999" text-anchor="middle">Drag to rotate, scroll to zoom</text>
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.getElementById('{svg_id}');
    if (!svg) return;
    var meshGroup = document.getElementById('{svg_id}-mesh');
    var boxGroup = document.getElementById('{svg_id}-box');
    var tris = {tris_json};
    var boxCorners = {box_json};
    var boxEdges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
    var width = {width}, height = {height}, scale = {scale}, centerX = width/2, centerY = height/2 + 30;
    var theta = 0.7854, phi = 0.6155;
    var zoom = 1.0, isDragging = false, lastX, lastY;

    function rotate(x, y, z) {{
        var cosT = Math.cos(theta), sinT = Math.sin(theta);
        var cosP = Math.cos(phi), sinP = Math.sin(phi);
        var x1 = x * cosT - y * sinT, y1 = x * sinT + y * cosT, z1 = z;
        return [x1, y1 * cosP - z1 * sinP, y1 * sinP + z1 * cosP];
    }}

    function project(x, y, z) {{
        var r = rotate(x, y, z);
        return {{ px: centerX + scale * zoom * r[0], py: centerY - scale * zoom * r[2], depth: r[1] }};
    }}

    function render() {{
        var projected = tris.map(function(t) {{
            var pts = t.v.map(function(v) {{ return project(v[0], v[1], v[2]); }});
            var avgDepth = (pts[0].depth + pts[1].depth + pts[2].depth) / 3;
            return {{ pts: pts, color: t.c, depth: avgDepth }};
        }});
        projected.sort(function(a, b) {{ return a.depth - b.depth; }});

        meshGroup.innerHTML = '';
        projected.forEach(function(t) {{
            var poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
            var points = t.pts.map(function(p) {{ return p.px.toFixed(1) + ',' + p.py.toFixed(1); }}).join(' ');
            poly.setAttribute('points', points);
            poly.setAttribute('fill', t.color);
            poly.setAttribute('fill-opacity', '{opacity}');
            meshGroup.appendChild(poly);
        }});

        boxGroup.innerHTML = '';
        var projBox = boxCorners.map(function(c) {{ return project(c[0], c[1], c[2]); }});
        boxEdges.forEach(function(e) {{
            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', projBox[e[0]].px.toFixed(1));
            line.setAttribute('y1', projBox[e[0]].py.toFixed(1));
            line.setAttribute('x2', projBox[e[1]].px.toFixed(1));
            line.setAttribute('y2', projBox[e[1]].py.toFixed(1));
            line.setAttribute('stroke', '#888');
            line.setAttribute('stroke-width', '1');
            line.setAttribute('stroke-dasharray', '4,2');
            boxGroup.appendChild(line);
        }});

        var labels = [['x', 1], ['y', 2], ['z', 4]];
        labels.forEach(function(l) {{
            var p = projBox[l[1]];
            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', (p.px + (l[0]==='y' ? -12 : 8)).toFixed(0));
            text.setAttribute('y', (p.py + (l[0]==='z' ? -5 : 4)).toFixed(0));
            text.setAttribute('font-size', '12');
            text.setAttribute('fill', '#333');
            text.setAttribute('font-weight', 'bold');
            text.textContent = l[0];
            boxGroup.appendChild(text);
        }});
    }}

    svg.addEventListener('mousedown', function(e) {{ isDragging = true; lastX = e.clientX; lastY = e.clientY; svg.style.cursor = 'grabbing'; }});
    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            theta += (e.clientX - lastX) * 0.01;
            phi += (e.clientY - lastY) * 0.01;
            phi = Math.max(-1.5, Math.min(1.5, phi));
            lastX = e.clientX; lastY = e.clientY;
            render();
        }}
    }});
    svg.addEventListener('mouseup', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('mouseleave', function() {{ isDragging = false; svg.style.cursor = 'grab'; }});
    svg.addEventListener('wheel', function(e) {{ e.preventDefault(); zoom *= e.deltaY > 0 ? 0.9 : 1.1; zoom = Math.max(0.3, Math.min(3.0, zoom)); render(); }});
    render();
}})();
]]></script></svg>"##,
            svg_id = svg_id,
            width = width,
            height = height,
            bg = opts.background,
            title_x = width / 2.0,
            title = title,
            hint_y = height - 10.0,
            tris_json = tris_json,
            box_json = box_json,
            scale = scale,
            opacity = opts.opacity,
        )
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

    /// Find the position of "==" operator in a string, respecting parentheses
    fn find_equation_operator(&self, s: &str) -> Option<usize> {
        let chars: Vec<char> = s.chars().collect();
        let mut depth = 0;

        for i in 0..chars.len().saturating_sub(1) {
            match chars[i] {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                '=' if depth == 0 && chars.get(i + 1) == Some(&'=') => {
                    return Some(i);
                }
                _ => {}
            }
        }
        None
    }

    /// Split "rhs, a, b" into (rhs, "a, b")
    fn split_rhs_and_bounds(&self, s: &str) -> Result<(String, String), EvalError> {
        let mut depth = 0;
        let chars: Vec<char> = s.chars().collect();

        for (i, &ch) in chars.iter().enumerate() {
            match ch {
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                ',' if depth == 0 => {
                    let rhs = s[..i].to_string();
                    let bounds = s[i + 1..].to_string();
                    return Ok((rhs, bounds));
                }
                _ => {}
            }
        }
        Err(EvalError::new("SyntaxError", "could not parse equation bounds"))
    }

    /// Convert a list of RustMathValues to Vec<f64>
    fn list_to_f64s(&self, list: &[RustMathValue]) -> Result<Vec<f64>, EvalError> {
        list.iter().map(|v| self.value_to_f64(v)).collect()
    }

    /// Format a Rational matrix as a string
    fn format_rational_matrix(&self, m: &Matrix<Rational>) -> String {
        let mut rows = Vec::new();
        for i in 0..m.rows() {
            let mut row = Vec::new();
            for j in 0..m.cols() {
                if let Ok(val) = m.get(i, j) {
                    row.push(format!("{}", val));
                }
            }
            rows.push(format!("[{}]", row.join(", ")));
        }
        format!("[{}]", rows.join(", "))
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

    /// Substitute a variable in an expression string with a numeric value
    /// This is a simple textual substitution for use in plotting functions
    fn substitute_var(&self, expr_str: &str, var_name: &str, value: f64) -> String {
        // Use regex-like pattern matching to replace variable occurrences
        // We need to be careful to only replace whole variable names, not partial matches
        let mut result = String::new();
        let mut chars = expr_str.chars().peekable();
        let var_chars: Vec<char> = var_name.chars().collect();

        while let Some(c) = chars.next() {
            // Check if this might be the start of our variable
            if c == var_chars[0] {
                let mut matched = true;
                let mut consumed: Vec<char> = vec![c];

                // Try to match rest of variable name
                for &vc in &var_chars[1..] {
                    if let Some(&next) = chars.peek() {
                        if next == vc {
                            consumed.push(chars.next().unwrap());
                        } else {
                            matched = false;
                            break;
                        }
                    } else {
                        matched = false;
                        break;
                    }
                }

                if matched {
                    // Check that we're not in the middle of a larger identifier
                    let is_word_boundary = match chars.peek() {
                        Some(&next) => !next.is_alphanumeric() && next != '_',
                        None => true,
                    };

                    // Also check if previous char was a word char (harder - we use a simpler heuristic)
                    // For now, check if result ends with an alphanumeric
                    let prev_is_boundary = result.chars().last()
                        .map(|prev| !prev.is_alphanumeric() && prev != '_')
                        .unwrap_or(true);

                    if is_word_boundary && prev_is_boundary {
                        // Wrap in parentheses for safety with negative numbers
                        result.push_str(&format!("({})", value));
                    } else {
                        // Not a standalone variable, push back consumed chars
                        for ch in consumed {
                            result.push(ch);
                        }
                    }
                } else {
                    // Didn't fully match, push back what we consumed
                    for ch in consumed {
                        result.push(ch);
                    }
                }
            } else {
                result.push(c);
            }
        }

        result
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
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}" style="cursor: grab;">
<script type="text/javascript"><![CDATA[
(function() {{
    var svg = document.currentScript.parentElement;
    var viewBox = svg.viewBox.baseVal;
    var zoom = 1.0, panX = 0, panY = 0, isDragging = false, startX, startY;
    svg.addEventListener('wheel', function(e) {{
        e.preventDefault();
        zoom *= e.deltaY > 0 ? 1.1 : 0.9;
        zoom = Math.max(0.5, Math.min(3.0, zoom));
        var w = {} * zoom, h = {} * zoom;
        viewBox.x = ({} - w) / 2 + panX; viewBox.y = ({} - h) / 2 + panY;
        viewBox.width = w; viewBox.height = h;
    }});
    svg.addEventListener('mousedown', function(e) {{ isDragging = true; startX = e.clientX; startY = e.clientY; }});
    svg.addEventListener('mousemove', function(e) {{
        if (isDragging) {{
            panX -= (e.clientX - startX) * zoom; panY -= (e.clientY - startY) * zoom;
            startX = e.clientX; startY = e.clientY;
            var w = {} * zoom, h = {} * zoom;
            viewBox.x = ({} - w) / 2 + panX; viewBox.y = ({} - h) / 2 + panY;
            viewBox.width = w; viewBox.height = h;
        }}
    }});
    svg.addEventListener('mouseup', function() {{ isDragging = false; }});
    svg.addEventListener('mouseleave', function() {{ isDragging = false; }});
}})();
]]></script>"##,
            width, height, width, height, width, height, width, height, width, height, width, height
        );

        svg.push_str(&format!(r#"<rect width="{}" height="{}" fill="white"/>"#, width, height));
        svg.push_str(&format!(
            r#"<text x="{}" y="25" text-anchor="middle" font-size="14" font-family="sans-serif">{}</text>"#,
            width / 2.0, title
        ));

        // Draw 3D bounding box
        let corners = [
            (min_x, min_y, min_z), (max_x, min_y, min_z),
            (min_x, max_y, min_z), (max_x, max_y, min_z),
            (min_x, min_y, max_z), (max_x, min_y, max_z),
            (min_x, max_y, max_z), (max_x, max_y, max_z),
        ];
        let projected_corners: Vec<(f64, f64)> = corners.iter()
            .map(|&(x, y, z)| project(x, y, z))
            .collect();

        // 12 edges of the box
        let edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),  // x-direction edges
            (0, 2), (1, 3), (4, 6), (5, 7),  // y-direction edges
            (0, 4), (1, 5), (2, 6), (3, 7),  // z-direction edges
        ];

        for (i, j) in edges {
            svg.push_str(&format!(
                r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#888" stroke-width="1" stroke-dasharray="4,2"/>"##,
                projected_corners[i].0, projected_corners[i].1, projected_corners[j].0, projected_corners[j].1
            ));
        }

        // Axis labels
        let (xx, xy) = projected_corners[1];
        let (yx, yy) = projected_corners[2];
        let (zx, zy) = projected_corners[4];
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">x</text>"##, xx + 8.0, xy + 4.0));
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">y</text>"##, yx - 12.0, yy + 4.0));
        svg.push_str(&format!(r##"<text x="{:.0}" y="{:.0}" font-size="12" fill="#333" font-weight="bold">z</text>"##, zx - 12.0, zy - 5.0));

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

        svg.push_str(&format!(
            r##"<text x="{}" y="{}" font-size="10" fill="#999" text-anchor="middle">Scroll to zoom, drag to pan</text>"##,
            width / 2.0, height - 10.0
        ));

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
                let exp = int_arg::<u32>(op, "exponent", b, 0..=repl_limits::OP_POW_EXP, "REPL limit")?;
                Ok(Integer(a.pow(exp)))
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
            // Integer + CliffordElem: create scalar element with matching dimension and add
            (Integer(k), "+", CliffordElem(a)) | (CliffordElem(a), "+", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from_integer(k.clone());
                let scalar = CliffordAlgebraElement::scalar(coeff, a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar + a.clone()))
            }
            (Integer(k), "-", CliffordElem(a)) => {
                let coeff = rustmath_rationals::Rational::from_integer(k.clone());
                let scalar = CliffordAlgebraElement::scalar(coeff, a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar - a.clone()))
            }
            (CliffordElem(a), "-", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from_integer(k.clone());
                let scalar = CliffordAlgebraElement::scalar(coeff, a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(a.clone() - scalar))
            }
            // Scalar multiplication: Integer * CliffordElem
            (Integer(k), "*", CliffordElem(a)) | (CliffordElem(a), "*", Integer(k)) => {
                let coeff = rustmath_rationals::Rational::from_integer(k.clone());
                let scalar = CliffordAlgebraElement::scalar(coeff, a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar * a.clone()))
            }
            // Scalar division: CliffordElem / Integer
            (CliffordElem(a), "/", Integer(k)) => {
                if k.is_zero() {
                    return Err(EvalError::new("ZeroDivisionError", "Division by zero"));
                }
                // Create 1/k as a Rational
                let one = rustmath_rationals::Rational::from(1i64);
                let divisor = rustmath_rationals::Rational::from_integer(k.clone());
                let coeff = one / divisor;
                let scalar = CliffordAlgebraElement::scalar(coeff, a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar * a.clone()))
            }
            // Rational operations with Clifford elements
            (Rational(r), "+", CliffordElem(a)) | (CliffordElem(a), "+", Rational(r)) => {
                let scalar = CliffordAlgebraElement::scalar(r.clone(), a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar + a.clone()))
            }
            (Rational(r), "*", CliffordElem(a)) | (CliffordElem(a), "*", Rational(r)) => {
                let scalar = CliffordAlgebraElement::scalar(r.clone(), a.dimension(), a.quadratic_form().to_vec());
                Ok(CliffordElem(scalar * a.clone()))
            }

            // String operations (Python-like)
            // String repetition: "=" * 60 or 60 * "="
            (String(s), "*", Integer(n)) | (Integer(n), "*", String(s)) => {
                // A negative count gives "" (Python), for every negative n.
                if n.signum() < 0 {
                    return Ok(String("".to_string()));
                }
                let count = int_arg::<usize>("*", "repetition count", n, 0..=repl_limits::STRING_REPEAT, "REPL limit")?;
                Ok(String(s.repeat(count)))
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
  is_prime(n)        - Test primality (a proof; refused above 1024 bits unless n is composite)
  is_probable_prime(n) - Baillie-PSW probable-prime test (exact below 3.19e23)
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
  sqrt_mod(a, n)       - Square roots of a mod n: a root, then all roots (at most 16)
                         or their number; also sqrt_mod(Mod(a, n)). A composite n is
                         factored by trial division (a large prime cofactor never finishes)
  pow_mod(b, e, m)     - Compute b^e mod m

Combinatorics Functions:
  binomial(n, k)       - Binomial coefficient C(n,k) for all integers (C(n,k) = (-1)^k C(k-n-1,k) for n < 0, 0 for k < 0)
  catalan(n)           - n-th Catalan number
  fibonacci(n)         - n-th Fibonacci number (n may be negative)
  lucas(n)             - n-th Lucas number (n may be negative)
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
  Integer arguments are range-checked; a ValueError names the range. REPL limits, set so
  that a refused call would take more than about 10 s or a few hundred MB: binomial, narayana,
  delannoy and falling/rising_factorial/multinomial bound a work estimate (the error shows it);
  stirling1/stirling2/eulerian bound the size of their (n+1) x (k+1) table (400 MB);
  catalan n <= 316227, fibonacci/lucas |n| <= 1300000, bell n <= 600, motzkin n <= 4500,
  schroder n <= 50000, derangements n <= 1400, partitions n <= 416 (127 on 32-bit targets;
  p(n) must fit in a machine word); sigma(n, k) about k*log2|n| <= 2^28 bits and, from the
  factorisation of n, a lower estimate of its time <= 10 s (sigma's trial division itself is
  not bounded)

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

// ---------------------------------------------------------------------------------------------
// Checked integer arguments and REPL resource limits (T-250).
//
// MAGMA Handbook ch. 18 §18.8 (`Binomial`, `Multinomial`, `Factorial`, `NumberOfPartitions`),
// §18.7 (`DivisorSigma`), §18.9.2 (`NthPrime`); ch. 144 §144.2 (`Binomial` and `Multinomial`
// for all integers, `Fibonacci`/`Lucas` for all integers n, `Catalan`, `StirlingFirst`,
// `StirlingSecond`, `Bell`, `EulerianNumber`, `NumberOfPermutations`); digests
// `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md:245-308` and
// `Chapter_144_Enumerative_Combinatorics.md:36-47`. Binomial coefficients with a negative upper
// index use Graham, Knuth, Patashnik, Concrete Mathematics, 2nd ed., (5.14).
//
// Every user integer reaches a machine type through `int_arg`/`index_arg`, `checked_binomial`
// or an exact conversion; nothing is wrapped, truncated or allowed to panic.
// ---------------------------------------------------------------------------------------------

/// Resource limits of REPL commands (T-250). The REPL has no timeout or interrupt, and a panic
/// or an out-of-memory abort kills the kernel, so these limits are its only protection.
///
/// The limits added for T-250 are set so that a refused call would take more than about 10 s
/// (release build, the development container) or a few hundred MB (about 400 MB; so the memory
/// limits do refuse some fast calls, e.g. stirling1(1200, 1199) on 64-bit targets: 0.45 s,
/// 463 MB), and so that an admitted call needs at most a few hundred MB (except where noted
/// below: taylor, derived differential forms). They were sized from rustc -O mirrors of the
/// exact callee loops over num-bigint 0.4.8 (the X2-jupyter implementation report has the
/// timings). Where the cost depends on more than one argument, the limit bounds a work or
/// size estimate (`binomial_work_u32`, `product_work`, `table_bytes`, `sigma_seconds`,
/// `check_form_components`) rather than one argument. Some bounds do not bound the time:
/// `matrix_power` and `nth_prime` get their callee's domain only, `taylor`'s cost and the
/// size of derived differential forms depend on the expressions, and `sigma` must factor n,
/// which no limit bounds. The older
/// limits (the first block) are unchanged and are listed here so that every limit is in one
/// place.
mod repl_limits {
    // Existing limits, unchanged.
    pub const FACTORIAL_N: u32 = 10_000;
    pub const POW_EXP: u32 = 1_000; // pow(b, e)
    pub const OP_POW_EXP: u32 = 10_000; // b ** e, b ^ e
    pub const MATRIX_DIM: usize = 1_000; // identity, zeros
    pub const ONES_DIM: usize = 100;
    pub const HISTOGRAM_BINS: usize = 1_000;
    pub const PLOT_POINTS: usize = 10_000;
    pub const CLIFFORD_DIM: usize = 10; // CliffordAlgebra, ExteriorAlgebra
    pub const JORDAN_DIM: usize = 20;
    pub const SPECIAL_JORDAN_N: usize = 10;
    pub const STRING_REPEAT: usize = 10_000;

    // Limits added for T-250 (time: refused calls take more than about 10 s; memory: admitted
    // calls stay under a few hundred MB).
    /// Size bound of C(N, K) in bits. Inside the old u32 domain it never refuses a call that
    /// `BINOMIAL_WORK` admits (see `checked_binomial`); above it, it caps the result at 512 KB.
    pub const BINOMIAL_BITS: u64 = 1 << 22;
    /// K · (size bound) for C(N, K), a model of the loop's time: 3.1e-11 to 6.3e-11 s per unit
    /// (binomial, narayana, delannoy), so a refused call takes more than about 12 s and an
    /// admitted one with N < 2^64 at most about 25 s. C(524288, 262144) (work 2^38, 9.8 s) is
    /// admitted. The model counts one-word factors N - i; for N >= 2^64 every step multiplies
    /// by a factor of words(N) words, which it leaves out, so such an admitted call can take
    /// longer (not measured; BINOMIAL_BITS keeps K · bits(N) below 2^22). That errs towards
    /// admitting.
    pub const BINOMIAL_WORK: u64 = 400_000_000_000;
    /// The largest n with 4n² <= BINOMIAL_WORK: catalan(n) computes C(2n, n), whose work
    /// estimate is 4n², so catalan(n) is admitted exactly when binomial(2n, n) is.
    pub const CATALAN_N: u32 = 316_227;
    /// fibonacci and lucas, |n|: n additions of numbers of up to 0.69n bits (10^6: 6.7 s,
    /// 1.5·10^6: 15.1 s).
    pub const FIBONACCI_N: i64 = 1_300_000;
    /// bell(600): 11.2 s and 53 MB (n+1 Stirling tables).
    pub const BELL_N: u32 = 600;
    /// stirling1, stirling2 and eulerian allocate an (n+1) x (k+1) table of Integers, whose
    /// size `table_bytes` computes. (1000, 999) needs 244, 212 and 333 MB on 64-bit targets
    /// (peaks 268, 228 and 380 MB, 0.2-0.4 s) and is admitted; peaks stay below about
    /// 1.15 · 400 MB = 460 MB.
    /// Small k is cheap: stirling1(10000, 1) needs 71 MB (0.05 s).
    pub const TABLE_BYTES: u64 = 400_000_000;
    /// motzkin(4000): 7.8 s, motzkin(4500): 12.1 s.
    pub const MOTZKIN_N: u32 = 4_500;
    /// schroder keeps n+1 Integers of up to 2.54n bits: 258 MB at 40000 (0.41 s), 400 MB at
    /// 50000 (0.59 s), the budget of TABLE_BYTES.
    pub const SCHRODER_N: u32 = 50_000;
    /// derangements(1300): 8.9 s, derangements(1400): 11.4 s (a gcd in every Rational addition).
    pub const DERANGEMENTS_N: u32 = 1_400;
    /// k · (k · bits of the largest factor) for falling_factorial, rising_factorial and the
    /// factorials of multinomial: 5.0e-12 to 6.0e-12 s per unit, so a refused call takes more
    /// than about 12 s and an admitted product at most about 15 s. multinomial also divides n!
    /// by every k_i!, one division of a number of up to bits(n!) bits per k_i, which the model
    /// leaves out (it was calibrated with one and two k_i); with many k_i an admitted call can
    /// take longer. That errs towards admitting.
    pub const FACTORIAL_WORK: u64 = 2_500_000_000_000;
    /// sigma(n, k): when k · log2 of the odd part of |n| exceeds this, n is factored and the
    /// time estimated by `sigma_seconds`. Below it the estimate is at most 4.75 s for every
    /// factorisation (the largest is a single prime with a = 1: the estimate of one term
    /// falls with a, and both of its powers are superadditive), so n is not factored first.
    pub const SIGMA_FACTOR_BITS: f64 = 16_777_216.0; // 2^24
    /// sigma(n, k) is refused when `sigma_seconds`, a lower estimate of its time, exceeds this.
    pub const SIGMA_SECONDS: f64 = 10.0;
    /// About k · log2|n|, the bits of sigma_k(n): a memory bound (sigma(2, 2^28): 445 MB).
    pub const SIGMA_BITS: f64 = 268_435_456.0; // 2^28
    /// Bounds the order only. Every term keeps n!, so even f = x needs 320 MB at order 20000
    /// (0.6 s); for most f the unsimplified derivatives grow exponentially and exhaust memory
    /// far below this order (x^2: 233 MB at order 8; 2*x+1 at order 3000: over 3 GB), which
    /// no order bound can prevent (rustmath-symbolic `Expr::taylor`, a split-out). The series
    /// is a left-deep sum that `simplify` and its drop walk recursively: order 20000 ran in a
    /// release build on an 8 MB thread, but overflows that stack in a debug build (order
    /// 10000 does not); the recursion is part of the same split-out.
    pub const TAYLOR_ORDER: usize = 20_000;
    /// EuclideanSpace(n) builds n coordinate names: 221 MB at 4·10^6, 385 MB at 7·10^6 (0.49 s).
    /// A Manifold value is shared (Arc), so the names are never copied.
    pub const EUCLIDEAN_DIM: usize = 7_000_000;
    /// coordinate_form stores dim(M) components inline, 64 bytes each (an `Expr`), and the REPL
    /// copies the whole form on every assignment (the value, the variable and `last_result`:
    /// three copies at the peak) and on every argument lookup of a variable holding it.
    /// Measured with the real `ReplContext` (VmHWM): `w = coordinate_form(M, C, 0)` then
    /// `wedge(w, w, C)` peaks at 287 MB and 381 MB at 1.5·10^6, at 381 MB and 506 MB at 2·10^6,
    /// and at 756 MB and 1006 MB at 4·10^6. 1.5·10^6 keeps an assignment plus a two-argument
    /// use inside the 400 MB budget of TABLE_BYTES.
    pub const MANIFOLD_COORDS: usize = 1_500_000;
    /// diff_form degree: keeps `p + 1` and `p + q` of forms built directly from overflowing
    /// (exterior_derivative and wedge check the degrees of derived forms themselves).
    pub const DIFF_FORM_DEGREE: usize = usize::MAX / 2;
    /// exterior_derivative and wedge build all n^degree components of the result: at 10^6
    /// components d(dx_0) (n = 1000) and dx_0 ∧ dx_0 (n = 1000) peak at 382 MB, d(d(dx_0))
    /// (n = 100) at 574 MB, because rustmath-symbolic does not simplify and each further d
    /// enlarges the expressions; that growth is not bounded here (a split-out).
    pub const FORM_COMPONENTS: usize = 1_000_000;
    /// p(n) <= usize::MAX for `partition_count`: p(416) = 17873792969689876004 < 2^64 <=
    /// p(417), and p(127) < 2^32 <= p(128) (PARI numbpart).
    pub const PARTITIONS_N: usize = if usize::BITS >= 64 { 416 } else { 127 };
    /// Above this min(m, n) the work of delannoy(m, n) is certainly above BINOMIAL_WORK
    /// (it is at least min(m, n)^3 / 3), so `delannoy_work` does not sum it.
    pub const DELANNOY_SUM_CAP: u32 = 20_000;
}

/// `v` for an error message: in full up to 128 bits, otherwise its sign and bit length (so a
/// 10^6-digit argument is not printed in decimal).
fn show_int(v: &Integer) -> String {
    if v.bit_length() <= 128 {
        v.to_string()
    } else {
        let sign = if v.signum() < 0 { "a negative" } else { "an" };
        format!("{} integer of {} bits", sign, v.bit_length())
    }
}

/// The integer argument `name` of command `cmd`, converted to `T` and accepted only inside
/// `range`. Nothing is wrapped, truncated or allowed to panic. Any other value (negative,
/// too large for `T`, or above a REPL limit) is a ValueError that names the command, the
/// argument and the range, with `why` in parentheses when it is not empty.
fn int_arg<T>(cmd: &str, name: &str, v: &Integer, range: RangeInclusive<T>, why: &str) -> Result<T, EvalError>
where
    T: Copy + PartialOrd + std::fmt::Display + for<'a> TryFrom<&'a BigInt>,
{
    // num-bigint's TryFrom<&BigInt> (0.4.8, bigint/convert.rs) covers u32/i64/usize alike;
    // rustmath-core's NumericConversion has no to_u32, and BigInt is already a direct dependency.
    match T::try_from(v.as_bigint()) {
        Ok(x) if range.contains(&x) => Ok(x),
        _ => {
            let why = if why.is_empty() { String::new() } else { format!(" ({})", why) };
            Err(EvalError::new("ValueError", format!(
                "{}: {} must be an integer in {}..={}{}, got {}",
                cmd, name, range.start(), range.end(), why, show_int(v))))
        }
    }
}

/// A 0-based index `name` of command `cmd` into `len` items described by `what`.
fn index_arg(cmd: &str, name: &str, v: &Integer, len: usize, what: &str) -> Result<usize, EvalError> {
    if len == 0 {
        return Err(EvalError::new("ValueError", format!(
            "{}: there are no {} to index, got {} = {}", cmd, what, name, show_int(v))));
    }
    // `what` is a plural ("rows"); one item reads "1 row".
    let what = if len == 1 { what.strip_suffix('s').unwrap_or(what) } else { what };
    int_arg(cmd, name, v, 0..=len - 1, &format!("{} {}", len, what))
}

/// MAGMA `Binomial(n, r)` (Handbook ch. 18 §18.8, ch. 144 §144.2) for all integers n, k, with
/// the values of PARI/GP `binomial` and GAP `Binomial`, computed by `SageInteger::binomial`
/// (rustmath-integers, `2a46bd0`):
///   k < 0                 -> 0
///   0 <= n < k            -> 0
///   n < 0 <= k            -> (-1)^k C(k-n-1, k)   (Graham, Knuth, Patashnik, Concrete
///                                                  Mathematics, 2nd ed., (5.14))
///   0 <= k <= n           -> C(n, k), the exact product over K = min(k, n-k) factors
/// With N the upper index after the transformation (N = n or k-n-1) and K = min(k, N-k), the
/// result has fewer than K·(bits(N) - bits(K) + 3) bits, because C(N, K) <= (eN/K)^K (e.g.
/// CLRS 3rd ed., (C.5)), log2 N < bits(N) and log2 K >= bits(K) - 1. For K <= 1 the result
/// is 1 or ±N, computed at once, so no limit applies and n may have any size.
///
/// Otherwise the REPL refuses the call when this size bound exceeds BINOMIAL_BITS (2^22), or
/// when the work K · bound exceeds BINOMIAL_WORK (4·10^11): the loop runs K steps, and step i
/// multiplies and divides C(N, i), which has at most bound + bits(N) bits, by one word. For
/// N < 2^32 (the domain of the old u32 command) the size check never refuses a call that the
/// work check admits: bits(N) - bits(K) + 3 <= 35 - bits(K), so bits(K) <= 17 gives a bound of
/// at most 18·2^17 < 2^22, and bound > 2^22 needs K > 2^22/17, hence K · bound > 2^44/17 > 10^12.
/// Measured in release (rustc -O mirror of the loop), the work model gives 3.1e-11 to
/// 5.3e-11 s per unit: C(524288, 262143) (work 3.4·10^11) takes 10.5 s and is admitted, a
/// refused call takes more than about 12 s.
fn checked_binomial(n: &Integer, k: &Integer) -> Result<Integer, EvalError> {
    if k.signum() < 0 {
        return Ok(Integer::zero());
    }
    let upper = if n.signum() < 0 { k.clone() - n.clone() - Integer::one() } else { n.clone() };
    if *k > upper {
        return Ok(Integer::zero());
    }
    let rest = upper.clone() - k.clone();
    let kk = if *k < rest { k.clone() } else { rest };
    if kk > Integer::one() {
        let (bound, work) = binomial_estimate(&upper, &kk);
        if bound > repl_limits::BINOMIAL_BITS {
            return Err(EvalError::new("ValueError", format!(
                "binomial: C(n, k) for n = {}, k = {} is a product of {} factors and may have \
                 more than {} bits (REPL limit)",
                show_int(n), show_int(k), show_int(&kk), repl_limits::BINOMIAL_BITS)));
        }
        if work > repl_limits::BINOMIAL_WORK {
            return Err(EvalError::new("ValueError", format!(
                "binomial: C(n, k) for n = {}, k = {} takes {} steps on numbers of up to {} bits; \
                 steps x bits = {} exceeds {} (REPL limit, about 10 s)",
                show_int(n), show_int(k), show_int(&kk), bound, work, repl_limits::BINOMIAL_WORK)));
        }
    }
    Ok(<Integer as SageInteger>::binomial(n, k))
}

/// (bound, work) for C(N, K) with N >= 2K >= 2: bound = K·(bits(N) - bits(K) + 3), an upper
/// bound on the bits of the result, and work = K · bound. Both saturate at u64::MAX. The bound
/// is at least 3K, so for K > BINOMIAL_BITS / 3 it is reported as u64::MAX without computing it.
fn binomial_estimate(upper: &Integer, kk: &Integer) -> (u64, u64) {
    match kk.to_u64() {
        Some(kk64) if kk64 <= repl_limits::BINOMIAL_BITS / 3 => {
            // upper >= 2K, so bits(upper) > bits(K) and the subtraction cannot underflow
            let per_factor = (upper.bit_length() + 3).saturating_sub(kk.bit_length());
            let bound = kk64.saturating_mul(per_factor);
            (bound, kk64.saturating_mul(bound))
        }
        _ => (u64::MAX, u64::MAX),
    }
}

/// The work estimate of `binomial_estimate` for `rustmath_combinatorics::binomial(n, k)`, the
/// u32 callee of catalan, narayana and delannoy: 0 when it returns at once (k > n, k = 0 or
/// k = n), otherwise K · K·(bits(n) - bits(K) + 3) with K = min(k, n - k).
fn binomial_work_u32(n: u32, k: u32) -> u64 {
    if k > n || k == 0 || k == n {
        return 0;
    }
    let kk = u64::from(k.min(n - k));
    let per_factor = u64::from(u32::BITS - n.leading_zeros()) + 3 - u64::from(u32::BITS - (kk as u32).leading_zeros());
    kk.saturating_mul(kk.saturating_mul(per_factor))
}

/// The work of `delannoy(m, n)` (`central`: `delannoy_central(n)`, m = n): the sum of
/// `binomial_work_u32` over the binomials the callee computes, C(m, k) and C(n, k) for
/// k <= min(m, n) (one of them for the central number). The products and the sum come on top;
/// the measured release time is 3.8e-11 (delannoy_central(10000): 14.7 s for 3.857e11) to
/// 6.3e-11 s per unit. Above DELANNOY_SUM_CAP the
/// sum is at least min(m, n)^3 / 3 > BINOMIAL_WORK and is reported as u64::MAX.
fn delannoy_work(m: u32, n: u32, central: bool) -> u64 {
    let s = m.min(n);
    if s > repl_limits::DELANNOY_SUM_CAP {
        return u64::MAX;
    }
    (0..=s).fold(0u64, |acc, k| {
        let w = if central { binomial_work_u32(n, k) } else { binomial_work_u32(m, k).saturating_add(binomial_work_u32(n, k)) };
        acc.saturating_add(w)
    })
}

/// k · bound for a product of k one-word factors whose largest is `top`: bound = k · bits(top)
/// bounds the bits of the product, and the i-th multiplication costs about its size. Used for
/// falling_factorial, rising_factorial and the factorials in multinomial; saturates.
fn product_work(k: u32, top: u32) -> u64 {
    let k = u64::from(k);
    k.saturating_mul(k.saturating_mul(u64::from(u32::BITS - top.leading_zeros())))
}

/// ValueError for a call whose work estimate exceeds a REPL limit.
fn work_error(cmd: &str, what: &str, work: u64, limit: u64) -> EvalError {
    EvalError::new("ValueError", format!(
        "{}: {}; steps x bits = {} exceeds {} (REPL limit, about 10 s)",
        cmd, what, if work == u64::MAX { "more than 2^64".to_string() } else { work.to_string() }, limit))
}

/// log2|v| for v != 0, from the leading 64 bits (exact to about 1e-16 relative).
fn log2_abs(v: &Integer) -> f64 {
    let bits = v.bit_length();
    if bits <= 64 {
        let x = u64::try_from(v.as_bigint().magnitude()).unwrap_or(u64::MAX);
        return (x as f64).log2();
    }
    let top = v.as_bigint().magnitude() >> ((bits - 64) as usize);
    (u64::try_from(&top).unwrap_or(u64::MAX) as f64).log2() + (bits - 64) as f64
}

/// The tables of rustmath-combinatorics sized by `table_bytes`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TableKind {
    /// `stirling_first`: s(i, j) = (i-1) s(i-1, j) + s(i-1, j-1) (lib.rs:506).
    Stirling1,
    /// `stirling_second`: S(i, j) = j S(i-1, j) + S(i-1, j-1) (lib.rs:538).
    Stirling2,
    /// `eulerian`: A(i, j) = (j+1) A(i-1, j) + (i-j) A(i-1, j-1) (lib.rs:591).
    Eulerian,
}

/// log2(2^x + 2^y), with -inf for the value 0.
fn log2_add(x: f64, y: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        return y;
    }
    if y == f64::NEG_INFINITY {
        return x;
    }
    let (hi, lo) = if x >= y { (x, y) } else { (y, x) };
    hi + (lo - hi).exp2().ln_1p() / std::f64::consts::LN_2
}

/// Bytes that `stirling_first(n, k)`, `stirling_second(n, k)` or `eulerian(n, k)` allocate
/// for their table: `vec![vec![Integer::zero(); k + 1]; n + 1]` (the Vec and Integer headers,
/// by `size_of`: 24 and 32 bytes on 64-bit targets, 12 and 16 on 32-bit ones) plus, for every
/// nonzero entry, its 64-bit words and 16 bytes of allocator overhead (the 64-bit layout; on
/// 32-bit targets num-bigint uses u32 digits and the overhead is smaller, so the formula
/// overestimates there, by up to about 12 bytes per entry). None when the
/// callee returns without a table (k > n, k = n, k = 0, and for stirling2 k = 1; eulerian:
/// k >= n, k = 0). The size of every entry comes from the callee's own recurrence run on
/// log2 values in f64 over one rolling row, so on 64-bit targets the estimate is the table's
/// real size, not a bound on it: for (1000, 999) it gives 243.7, 212.0 and 332.6 MB on 64-bit targets, the exact
/// sizes computed with Python big integers; the measured peaks (rustc -O mirrors, VmHWM) are 268, 228 and
/// 380 MB, i.e. 1.07 to 1.15 times the estimate (temporaries and capacity slack). This costs
/// O(n k) float operations, far less than the callee's O(n k) big-integer steps, and stops
/// early with a value above `limit` at the end of the first row where the running total passes
/// it. The headers are checked first, every nonzero entry adds at least 8 + 16 bytes, a visited
/// row has at most one zero entry and the last row at most k + 1 entries, so at most about
/// limit / 24 + 2n entries are visited.
fn table_bytes(kind: TableKind, n: u32, k: u32, limit: u64) -> Option<u64> {
    let no_table = match kind {
        TableKind::Stirling1 => n == 0 || k == 0 || k >= n,
        TableKind::Stirling2 => n == 0 || k <= 1 || k >= n,
        TableKind::Eulerian => k == 0 || k >= n,
    };
    if no_table {
        return None;
    }
    let int_size = std::mem::size_of::<Integer>() as u128;
    let row_size = std::mem::size_of::<Vec<Integer>>() as u128 + int_size * (u128::from(k) + 1);
    let headers = (u128::from(n) + 1) * row_size;
    if headers > u128::from(limit) {
        return Some(u64::try_from(headers).unwrap_or(u64::MAX));
    }
    let entry = |l: f64| -> u64 { 8 * ((l / 64.0).floor() as u64 + 1) + 16 };
    // logs[j] = log2 of the entry j of the current row; row 0 is [1, 0, ..., 0].
    let mut logs = vec![f64::NEG_INFINITY; k as usize + 1];
    logs[0] = 0.0;
    let mut total = headers as u64 + entry(0.0);
    for i in 1..=n as usize {
        match kind {
            TableKind::Stirling1 => {
                let li = ((i - 1) as f64).log2();
                for j in (1..=i.min(k as usize)).rev() {
                    logs[j] = if j == i { 0.0 } else { log2_add(li + logs[j], logs[j - 1]) };
                }
                logs[0] = f64::NEG_INFINITY;
            }
            TableKind::Stirling2 => {
                for j in (1..=i.min(k as usize)).rev() {
                    logs[j] = if j == 1 || j == i {
                        0.0
                    } else {
                        log2_add((j as f64).log2() + logs[j], logs[j - 1])
                    };
                }
                logs[0] = f64::NEG_INFINITY;
            }
            TableKind::Eulerian => {
                for j in (1..=(i - 1).min(k as usize)).rev() {
                    logs[j] = log2_add(((j + 1) as f64).log2() + logs[j], ((i - j) as f64).log2() + logs[j - 1]);
                }
                logs[0] = 0.0;
            }
        }
        for &l in logs.iter().take(i.min(k as usize) + 1) {
            if l != f64::NEG_INFINITY {
                total = total.saturating_add(entry(l));
            }
        }
        if total > limit {
            return Some(total);
        }
    }
    Some(total)
}

/// A lower estimate, in seconds (release build, the development container), of computing
/// sigma_k(n) = prod over p^a || n of (p^((a+1)k) - 1) / (p^k - 1) from the factorisation
/// `factors` of |n| (`sigma_from_factors`). For an odd p^a, with b = k log2 p bits in p^k and
/// B = (a+1) b in p^((a+1)k), the two powers cost about C_POW (B / 2^25)^ALPHA and the
/// division, a times longer than its divisor, about C_DIV a (b / 2^24)^BETA (num-bigint 0.4.8:
/// Toom-3 products, Burnikel-Ziegler division). The constants are fitted so that the estimate
/// is at most the measured time on all 39 calibration runs (a rustc -O mirror of
/// `sigma_from_factors`, at 2^24.5 to 2^26.5 bits: a = 1 for p = 3, 1000003, 2^61-1 and
/// 2^1279-1; a = 2 to 10000 for p = 3; a = 5, 3, 2 and 1000 for the other primes; products of
/// 2, 3, 5 and 16 primes): measured / estimate is 1.00 to 1.38 for a prime power and up to 2.6
/// for 16 primes. Left out, so that the estimate stays a lower bound: the power of 2 (whose
/// products skip zero words; at most about 2 s under SIGMA_BITS), the products of the
/// per-prime factors, and the trial division. A refused call therefore takes more than
/// SIGMA_SECONDS, also in the old `Integer::sigma`, which is slower (it divides twice); an
/// admitted one can take longer, e.g. sigma(2^65535 * 3^1000, 4096), 12 s, is admitted.
fn sigma_seconds(factors: &[(Integer, u32)], k: u32) -> f64 {
    const C_POW: f64 = 2.15;
    const ALPHA: f64 = 1.25;
    const C_DIV: f64 = 2.6;
    const BETA: f64 = 1.7;
    let two = Integer::from(2);
    factors
        .iter()
        .filter(|(p, _)| *p != two)
        .map(|(p, a)| {
            let b = f64::from(k) * log2_abs(p);
            let big = f64::from(*a + 1) * b;
            C_POW * (big / 33_554_432.0).powf(ALPHA) + C_DIV * f64::from(*a) * (b / 16_777_216.0).powf(BETA)
        })
        .sum()
}

/// MAGMA `DivisorSigma(k, n)` (Handbook ch. 18 §18.7) for k >= 1 from the factorisation of
/// |n|: the product over p^a of (p^((a+1)k) - 1) / (p^k - 1), the formula of `Integer::sigma`
/// (rustmath-integers integer.rs:384-404) with one exact division per prime instead of
/// `div_rem`'s two. The REPL uses it when it has factored n for `sigma_seconds`, so the trial
/// division runs once. The caller's SIGMA_BITS check keeps (a+1)k below 2^29.
fn sigma_from_factors(factors: &[(Integer, u32)], k: u32) -> Integer {
    let one = Integer::one();
    factors.iter().fold(Integer::one(), |acc, (p, a)| {
        let num = p.pow((*a + 1) * k) - one.clone();
        let den = p.pow(k) - one.clone();
        acc * (num / den)
    })
}

/// The REPL check for exterior_derivative and wedge, whose callees build every one of the
/// n^degree components of the result: refuses unless n^degree <= FORM_COMPONENTS.
fn check_form_components(cmd: &str, n: usize, degree: usize) -> Result<(), EvalError> {
    match u32::try_from(degree).ok().and_then(|d| n.checked_pow(d)) {
        Some(c) if c <= repl_limits::FORM_COMPONENTS => Ok(()),
        count => Err(EvalError::new("ValueError", format!(
            "{}: the result, a {}-form on a manifold of dimension {}, has {} components (REPL limit {}, \
             a memory bound)", cmd, degree, n,
            count.map_or_else(|| format!("more than {}", usize::MAX), |c| c.to_string()), repl_limits::FORM_COMPONENTS))),
    }
}

/// For a call of exterior_derivative or wedge that `check_form_components` refuses: the
/// callee's own error when one of `forms` (in the callee's order) has no components in
/// `chart`. Both callees look the components up before they allocate anything
/// (`TensorField::components`, tensor_field.rs:162-166, `NoComponentsInChart`), so such a call
/// fails there whatever the dimension, and that error is the accurate answer (in the REPL these
/// are the forms built by diff_form: every REPL chart is named "user_chart"). The lookup clones
/// the components, so it is made only on this refusal path, and only for a form of at most
/// FORM_COMPONENTS components (n^degree; only a coordinate form on a manifold of dimension above
/// 10^6 has more). None when the forms have components in `chart` or the lookup is skipped:
/// the component limit then answers.
fn missing_components_error(forms: &[&DiffForm], chart: &Chart) -> Option<EvalError> {
    for form in forms {
        let n = form.manifold().dimension();
        let count = u32::try_from(form.degree()).ok().and_then(|d| n.checked_pow(d));
        if !count.is_some_and(|c| c <= repl_limits::FORM_COMPONENTS) {
            return None;
        }
        if let Err(e) = form.tensor().components(chart) {
            return Some(EvalError::new("ComputationError", format!("{}", e)));
        }
    }
    None
}

/// The REPL check for stirling1, stirling2 and eulerian: refuses a table above TABLE_BYTES.
fn check_table(cmd: &str, kind: TableKind, n: u32, k: u32) -> Result<(), EvalError> {
    match table_bytes(kind, n, k, repl_limits::TABLE_BYTES) {
        Some(bytes) if bytes > repl_limits::TABLE_BYTES => Err(EvalError::new("ValueError", format!(
            "{}: the recurrence table for n = {}, k = {} needs at least {} bytes (REPL limit {} bytes, \
             a memory bound)", cmd, n, k, bytes, repl_limits::TABLE_BYTES))),
        _ => Ok(()),
    }
}

/// `is_prime`, `next_prime` and `sqrt_mod` prove primality (ECPP, rustmath-integers `ecpp`) only
/// up to this many bits: a proof takes about 3.3 s at 1024 bits and 21 s at 1536 bits in a
/// release build (I1 design §2.3.7, random primes), and the REPL has no time limit. A number
/// that Baillie-PSW refutes is decided at every size (`false` is a proof).
const PROOF_BITS: u64 = 1024;

/// Refuses an argument above [`PROOF_BITS`] for a command that must prove a prime.
fn check_proof_bits(cmd: &str, n: &Integer) -> Result<(), EvalError> {
    if n.bits() > PROOF_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the argument has {} bits; a primality proof (ECPP) above {} bits can take \
             minutes, so it is not attempted (is_probable_prime gives the Baillie-PSW answer)",
            cmd, n.bits(), PROOF_BITS)));
    }
    Ok(())
}

/// The REPL's `is_prime`: Baillie-PSW first (a `false` is a proof at every size), then the
/// proof (`try_is_prime`) up to [`PROOF_BITS`]; an unresolved proof is a ValueError.
fn checked_is_prime(cmd: &str, n: &Integer) -> Result<bool, EvalError> {
    if !is_probable_prime(n) {
        return Ok(false);
    }
    check_proof_bits(cmd, n)?;
    try_is_prime(n).map_err(|e| EvalError::new("ValueError", format!("{}: {}", cmd, e)))
}

/// At most this many roots are listed by `sqrt_mod`; above it only their number is printed.
const SQRT_MOD_LIST_LIMIT: u64 = 16;

/// The text of the REPL command `sqrt_mod(a, n)`: the square roots of a in Z/nZ.
///
/// MAGMA: `Modsqrt(n, m)` (Handbook ch. 18 §18.12.1); `IsSquare`, `Sqrt`, `AllSqrts` in Z/mZ
/// (ch. 19 §19.5.6, worked example H19E4: the 12 roots of 1404 in Z/2340Z).
///
/// Output, with b = a mod n:
/// - `"{r}; square roots of {b} mod {n}: [r_1, ..., r_N]"` when b has N <= 16 roots, listed in
///   ascending order; r is `IntegerMod::sqrt`'s root (the least root for a prime power, the CRT
///   of the per-prime-power least roots otherwise, e.g. 7 for 4 mod 15);
/// - `"{r}; {b} has {N} square roots mod {n} (not listed)"` when N > 16;
/// - `"no square root: {b} is not a square mod {n}"`, which is always a proof.
///
/// Algorithm:
/// - n <= 0 is an error; n = 1 gives the single root 0 (Z/1Z = {0}).
/// - Prime n (`try_is_prime`, a proof, up to [`PROOF_BITS`] bits): if (b/n) = -1 there is no
///   root (the Jacobi symbol proves it for every odd n); otherwise
///   `Integer::try_sqrt_mod_prime` gives a root r, squared back, and the roots are {r, n - r}
///   (H. Cohen, GTM 138, Algorithms 1.4.10 and 1.5.1). This repeats the k = 1 decision of
///   `IntegerMod::sqrt` without its trial-division factoring of n, which would take minutes
///   near 1e18 and never finish for 2^127 - 1; a REPL test checks that the two agree for every
///   n <= 60. A Baillie-PSW probable prime above [`PROOF_BITS`] bits is not proven: the roots
///   are computed the same way and the text says that the list is complete if n is prime. An
///   unresolved proof is a ValueError.
/// - Composite n: for odd n with Jacobi symbol (b/n) = -1 there is no root (a proof, and no
///   factoring is needed). Otherwise `IntegerMod::sqrt`, `num_sqrts` and `all_sqrts` (CRT over the prime-power
///   factors; `None` is a proof). They factor n by trial division, each call again (up to three
///   times here). A composite n with a large prime cofactor is therefore out of reach: for
///   n = 2^127 + 1 = 3 · 56713727820156410577229101238628035243 trial division would run to
///   about 7.5e18, so the command does not finish. A proven factorization (MAGMA's
///   `Factorization :=` parameter, or rho/ECM with a primality proof) is not implemented.
fn sqrt_mod_report(a: &Integer, n: &Integer) -> Result<String, EvalError> {
    if n.signum() <= 0 {
        return Err(EvalError::new(
            "ValueError",
            format!("sqrt_mod: the modulus must be at least 1, got {}", n),
        ));
    }
    if n.is_one() {
        return Ok("0; square roots of 0 mod 1: [0]".to_string());
    }
    let b = a.modulo(n);
    let join = |roots: &[Integer]| {
        roots.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
    };

    let probable = is_probable_prime(n);
    let proven = probable && n.bits() <= PROOF_BITS;
    let is_prime = if proven {
        try_is_prime(n).map_err(|e| EvalError::new("ValueError", format!("sqrt_mod: {}", e)))?
    } else {
        probable
    };
    if is_prime {
        let two = Integer::from(2);
        let roots: Vec<Integer> = if b.is_zero() || *n == two {
            vec![b.clone()]
        } else if b.kronecker_symbol(n) == -1 {
            return Ok(format!("no square root: {} is not a square mod {}", b, n));
        } else {
            match b.try_sqrt_mod_prime(n) {
                Ok(Some(r)) => {
                    let mut v = vec![r.clone(), n.clone() - r];
                    v.sort();
                    v
                }
                Ok(None) => {
                    return Ok(format!("no square root: {} is not a square mod {}", b, n));
                }
                Err(e) => {
                    return Err(EvalError::new(
                        "ValueError",
                        format!(
                            "sqrt_mod: n = {} passed the probable-prime test but is composite ({}); \
                             factor n to decide",
                            n, e
                        ),
                    ))
                }
            }
        };
        let mut text = format!("{}; square roots of {} mod {}: [{}]", roots[0], b, n, join(&roots));
        if !proven {
            text.push_str(&format!(
                " (n is only a probable prime: its primality proof is not attempted above {} \
                 bits; the list is complete if n is prime)",
                PROOF_BITS
            ));
        }
        return Ok(text);
    }

    // Composite n. A Jacobi symbol -1 proves that b is not a square mod n (no factoring).
    if n.is_odd() && b.kronecker_symbol(n) == -1 {
        return Ok(format!("no square root: {} is not a square mod {}", b, n));
    }
    // IntegerMod handles any modulus (trial-division factoring, see above).
    let x = IntegerMod::new(b.clone(), n.clone())
        .map_err(|e| EvalError::new("ValueError", format!("sqrt_mod: {}", e)))?;
    let r = match x.sqrt() {
        None => return Ok(format!("no square root: {} is not a square mod {}", b, n)),
        Some(r) => r.value().clone(),
    };
    let count = x.num_sqrts();
    if count <= Integer::from(SQRT_MOD_LIST_LIMIT) {
        let roots: Vec<Integer> = x.all_sqrts().iter().map(|y| y.value().clone()).collect();
        Ok(format!("{}; square roots of {} mod {}: [{}]", r, b, n, join(&roots)))
    } else {
        Ok(format!("{}; {} has {} square roots mod {} (not listed)", r, b, count, n))
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
    fn test_clifford_new_functions() {
        let mut ctx = ReplContext::new();

        // Create Cl(R^3)
        ctx.eval("Cl3 = CliffordAlgebra(3)").unwrap();

        // Test quadratic_form
        let qf = ctx.eval("quadratic_form(Cl3)").unwrap();
        assert!(qf.text.contains("1"), "Expected quadratic form with 1s: {}", qf.text);

        // Test is_exterior
        let is_ext = ctx.eval("is_exterior(Cl3)").unwrap();
        assert!(is_ext.text.contains("false") || is_ext.text.contains("False"),
            "Expected false for Clifford algebra: {}", is_ext.text);

        // Test pseudoscalar
        let ps = ctx.eval("pseudoscalar(Cl3)").unwrap();
        assert!(ps.text.contains("0, 1, 2") || ps.text.contains("CliffordAlgebraElement"),
            "Expected pseudoscalar: {}", ps.text);

        // Test basis_of_grade
        let grade2 = ctx.eval("basis_of_grade(Cl3, 2)").unwrap();
        assert!(grade2.text.contains("[") || grade2.text.contains("CliffordAlgebraElement"),
            "Expected list of grade-2 basis elements: {}", grade2.text);

        // Test algebra_basis
        let basis = ctx.eval("algebra_basis(Cl3)").unwrap();
        assert!(basis.text.contains("[") || basis.text.contains("CliffordAlgebraElement"),
            "Expected list of all basis elements: {}", basis.text);

        // Test center
        let center_result = ctx.eval("center(Cl3)").unwrap();
        assert!(center_result.text.contains("[") || center_result.text.contains("CliffordAlgebraElement"),
            "Expected center basis: {}", center_result.text);

        // Create element and test element functions
        ctx.eval("e0 = e(0)").unwrap();
        ctx.eval("e1 = e(1)").unwrap();
        ctx.eval("mixed = e0 + e0*e1").unwrap();

        // Test even_part
        let even = ctx.eval("even_part(mixed)").unwrap();
        assert!(even.text.contains("CliffordAlgebraElement"),
            "Expected even part: {}", even.text);

        // Test odd_part
        let odd = ctx.eval("odd_part(mixed)").unwrap();
        assert!(odd.text.contains("CliffordAlgebraElement"),
            "Expected odd part: {}", odd.text);

        // Test is_homogeneous
        let homog = ctx.eval("is_homogeneous(e0)").unwrap();
        assert!(homog.text.contains("true") || homog.text.contains("True"),
            "Expected true for generator: {}", homog.text);

        // Test reverse
        let rev = ctx.eval("reverse(e0)").unwrap();
        assert!(rev.text.contains("CliffordAlgebraElement"),
            "Expected reverse: {}", rev.text);

        // Test grade_involution
        let gi = ctx.eval("grade_involution(e0)").unwrap();
        assert!(gi.text.contains("CliffordAlgebraElement"),
            "Expected grade involution: {}", gi.text);

        // Test clifford_conjugate
        let cc = ctx.eval("clifford_conjugate(e0)").unwrap();
        assert!(cc.text.contains("CliffordAlgebraElement"),
            "Expected clifford conjugate: {}", cc.text);

        // Test exterior algebra specific functions
        ctx.eval("E3 = ExteriorAlgebra(3)").unwrap();

        // Test is_exterior on exterior algebra
        let is_ext2 = ctx.eval("is_exterior(E3)").unwrap();
        assert!(is_ext2.text.contains("true") || is_ext2.text.contains("True"),
            "Expected true for exterior algebra: {}", is_ext2.text);

        // Test counit
        ctx.eval("elem = e(0)").unwrap();
        let cu = ctx.eval("counit(E3, elem)").unwrap();
        assert!(cu.text.contains("0") || cu.text.contains("Rational"),
            "Expected counit: {}", cu.text);

        // Test antipode
        let ap = ctx.eval("antipode(E3, elem)").unwrap();
        assert!(ap.text.contains("CliffordAlgebraElement"),
            "Expected antipode: {}", ap.text);
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

    // ---------------------------------------------------------------------------------
    // sqrt_mod (card T-249): square roots modulo any n >= 1.
    //
    // MAGMA: `Modsqrt(n, m)` (ch. 18 §18.12.1); `IsSquare`, `Sqrt`, `AllSqrts` in Z/mZ
    // (ch. 19 §19.5.6, worked example H19E4). Reference values: PARI/GP 2.15.4 (`gp -q`):
    //   brute force  select(x -> Mod(x,m)^2 == Mod(a,m), [0..m-1])        (root lists)
    //   existence    issquare(Mod(a,m))
    //   the root     lift(chinese(...)) of the per-prime-power least roots (IntegerMod::sqrt's
    //                rule), e.g. chinese(Mod(1,3), Mod(2,5)) = Mod(7,15) for 4 mod 15
    //   primes       issquare(Mod(a,p),&r); vecsort([lift(r), p-lift(r)])
    //   2^64         the four roots of 17 by Hensel structure, each checked by Mod(r,2^64)^2==17
    //   psi12        factor(318665857834031151167461) = 399165290221 * 798330580441;
    //                kronecker(41, psi12) = -1, issquare(Mod(73, psi12)) = 1
    // ---------------------------------------------------------------------------------

    fn sqrt_mod_text(ctx: &mut ReplContext, cmd: &str) -> String {
        ctx.eval(cmd)
            .unwrap_or_else(|e| panic!("{cmd}: {} {}", e.name, e.message))
            .text
    }

    fn sqrt_mod_err(ctx: &mut ReplContext, cmd: &str) -> EvalError {
        match ctx.eval(cmd) {
            Ok(r) => panic!("{cmd}: expected an error, got {}", r.text),
            Err(e) => e,
        }
    }

    /// A String value prints with quotes.
    fn quoted(s: &str) -> String {
        format!("\"{}\"", s)
    }

    // The old command called the prime-only `square_root_mod_prime` and printed a false
    // "No square root exists" for every one of these.
    #[test]
    fn test_sqrt_mod_composite_moduli() {
        let mut ctx = ReplContext::new();
        let cases: &[(&str, &str)] = &[
            // PARI brute force [2, 7, 8, 13]; CRT root chinese(Mod(1,3), Mod(2,5)) = 7.
            ("sqrt_mod(4, 15)", "7; square roots of 4 mod 15: [2, 7, 8, 13]"),
            ("sqrt_mod(Mod(4, 15))", "7; square roots of 4 mod 15: [2, 7, 8, 13]"),
            ("sqrt_mod(Mod(4, 15), 15)", "7; square roots of 4 mod 15: [2, 7, 8, 13]"),
            // PARI issquare(Mod(2,15)) = 0.
            ("sqrt_mod(2, 15)", "no square root: 2 is not a square mod 15"),
            ("sqrt_mod(0, 15)", "0; square roots of 0 mod 15: [0]"),
            ("sqrt_mod(4, 9)", "2; square roots of 4 mod 9: [2, 7]"),
            ("sqrt_mod(19, 15)", "7; square roots of 4 mod 15: [2, 7, 8, 13]"),
            // MAGMA H19E4: the 12 square roots of 1404 in Z/2340Z (PARI brute force).
            (
                "sqrt_mod(1404, 2340)",
                "1872; square roots of 1404 mod 2340: [78, 312, 468, 702, 858, 1092, 1248, 1482, 1638, 1872, 2028, 2262]",
            ),
            // Z/2^64: PARI, four roots of 17, each squares back.
            (
                "sqrt_mod(17, 18446744073709551616)",
                "405959429219100393; square roots of 17 mod 18446744073709551616: [405959429219100393, 8817412607635675415, 9629331466073876201, 18040784644490451223]",
            ),
            ("sqrt_mod(1, 1)", "0; square roots of 0 mod 1: [0]"),
        ];
        for (cmd, expected) in cases {
            assert_eq!(sqrt_mod_text(&mut ctx, cmd), quoted(expected), "{cmd}");
        }
    }

    #[test]
    fn test_sqrt_mod_prime_moduli() {
        let mut ctx = ReplContext::new();
        let cases: &[(&str, &str)] = &[
            ("sqrt_mod(4, 7)", "2; square roots of 4 mod 7: [2, 5]"),
            ("sqrt_mod(Mod(2, 7))", "3; square roots of 2 mod 7: [3, 4]"),
            ("sqrt_mod(3, 7)", "no square root: 3 is not a square mod 7"),
            ("sqrt_mod(-1, 5)", "2; square roots of 4 mod 5: [2, 3]"),
            ("sqrt_mod(-1, 7)", "no square root: 6 is not a square mod 7"),
            ("sqrt_mod(3, 2)", "1; square roots of 1 mod 2: [1]"),
            ("sqrt_mod(0, 13)", "0; square roots of 0 mod 13: [0]"),
            ("sqrt_mod(2, 17)", "6; square roots of 2 mod 17: [6, 11]"),
        ];
        for (cmd, expected) in cases {
            assert_eq!(sqrt_mod_text(&mut ctx, cmd), quoted(expected), "{cmd}");
        }
    }

    #[test]
    fn test_sqrt_mod_many_roots_not_listed() {
        let mut ctx = ReplContext::new();
        // 5040 = 2^4 * 3^2 * 5 * 7: 1 has 4 * 2 * 2 * 2 = 32 roots (PARI brute force: 32).
        assert_eq!(
            sqrt_mod_text(&mut ctx, "sqrt_mod(1, 5040)"),
            quoted("1; 1 has 32 square roots mod 5040 (not listed)")
        );
        // 0 mod 2^64 has 2^32 roots (the multiples of 2^32).
        assert_eq!(
            sqrt_mod_text(&mut ctx, "sqrt_mod(0, 18446744073709551616)"),
            quoted("0; 0 has 4294967296 square roots mod 18446744073709551616 (not listed)")
        );
    }

    // Prime moduli far beyond trial division: PARI issquare(Mod(a,p),&r).
    #[test]
    fn test_sqrt_mod_large_primes_fast_path() {
        let mut ctx = ReplContext::new();
        let cases: &[(&str, &str)] = &[
            (
                "sqrt_mod(2, 2305843009213693951)",
                "2147483648; square roots of 2 mod 2305843009213693951: [2147483648, 2305843007066210303]",
            ),
            (
                "sqrt_mod(3, 1000000000000000009)",
                "60248662633209561; square roots of 3 mod 1000000000000000009: [60248662633209561, 939751337366790448]",
            ),
            (
                "sqrt_mod(-1, 1000000000000000009)",
                "333333333000000003; square roots of 1000000000000000008 mod 1000000000000000009: [333333333000000003, 666666667000000006]",
            ),
            (
                "sqrt_mod(7, 1000000000000000009)",
                "no square root: 7 is not a square mod 1000000000000000009",
            ),
        ];
        for (cmd, expected) in cases {
            assert_eq!(sqrt_mod_text(&mut ctx, cmd), quoted(expected), "{cmd}");
        }
    }

    #[test]
    fn test_sqrt_mod_large_prime_and_strong_pseudoprime() {
        let mut ctx = ReplContext::new();
        // 2^127 - 1 is prime (PARI isprime = 1) and proven now: no caveat.
        let t = sqrt_mod_text(&mut ctx, "sqrt_mod(2, 170141183460469231731687303715884105727)");
        assert_eq!(
            t,
            quoted("18446744073709551616; square roots of 2 mod 170141183460469231731687303715884105727: [18446744073709551616, 170141183460469231713240559642174554111]")
        );
        // psi12 is composite (PARI isprime = 0, factor = 399165290221 * 798330580441), and
        // kronecker(41, psi12) = -1: a proof without factoring.
        assert_eq!(
            sqrt_mod_text(&mut ctx, "sqrt_mod(41, 318665857834031151167461)"),
            quoted("no square root: 41 is not a square mod 318665857834031151167461")
        );
        assert_eq!(ctx.eval("is_prime(318665857834031151167461)").unwrap().text, "False");
        // The old rows sqrt_mod(4, psi12) and sqrt_mod(73, psi12) took the prime path and
        // listed two of the four roots of 4 with a caveat (PARI: 4 has four roots mod psi12).
        // On the composite path they need psi12 factored, which IntegerMod does by trial
        // division (about 2e11 divisions), so they are left out here.
        //
        // 2^1279 - 1 is a Mersenne prime (PARI isprime = 1) above the REPL's proof size: the
        // roots are computed and squared back, the text says the proof was not attempted.
        let t = sqrt_mod_text(&mut ctx, "sqrt_mod(4, 2^1279 - 1)");
        assert!(t.starts_with("\"2; square roots of 4 mod "), "{t}");
        assert!(t.contains("probable prime"), "{t}");
    }

    /// The primality commands after the proving `is_prime` (gp: isprime(2^127-1) = 1,
    /// isprime(318665857834031151167461) = 0, nextprime(2) = 2, isprime(2^1279-1) = 1,
    /// 17 | 2^1100 + 1 because 1100 = 4 · 275).
    #[test]
    fn test_prime_commands_with_proofs() {
        let mut ctx = ReplContext::new();
        let text = |ctx: &mut ReplContext, cmd: &str| {
            ctx.eval(cmd).unwrap_or_else(|e| panic!("{cmd}: {} {}", e.name, e.message)).text
        };
        assert_eq!(text(&mut ctx, "is_prime(2^127 - 1)"), "True");
        assert_eq!(text(&mut ctx, "is_prime(318665857834031151167461)"), "False");
        assert_eq!(text(&mut ctx, "is_probable_prime(318665857834031151167461)"), "False");
        assert_eq!(text(&mut ctx, "is_probable_prime(2^1279 - 1)"), "True");
        assert_eq!(text(&mut ctx, "is_prime(2^1100 + 1)"), "False");
        assert_eq!(text(&mut ctx, "next_prime(1)"), "2");
        assert_eq!(text(&mut ctx, "next_prime(0)"), "2");
        for cmd in ["is_prime(2^1279 - 1)", "next_prime(2^1100)", "prime_pi(2^70)", "nth_prime(10^18)"] {
            let e = sqrt_mod_err(&mut ctx, cmd);
            assert_eq!(e.name, "ValueError", "{cmd}: {}", e.message);
        }
        assert!(sqrt_mod_err(&mut ctx, "is_prime(2^1279 - 1)").message.contains("is_probable_prime"));
    }

    #[test]
    fn test_sqrt_mod_argument_checks() {
        let mut ctx = ReplContext::new();
        for cmd in ["sqrt_mod(Mod(4, 15), 7)", "sqrt_mod(4, 0)", "sqrt_mod(4, -7)"] {
            assert_eq!(sqrt_mod_err(&mut ctx, cmd).name, "ValueError", "{cmd}");
        }
        for cmd in ["sqrt_mod(4)", "sqrt_mod(1.5, 7)"] {
            assert_eq!(sqrt_mod_err(&mut ctx, cmd).name, "TypeError", "{cmd}");
        }
        for cmd in ["sqrt_mod()", "sqrt_mod(1, 2, 3)"] {
            assert_eq!(sqrt_mod_err(&mut ctx, cmd).name, "ArgumentError", "{cmd}");
        }
    }

    // The prime path repeats the k = 1 decision of IntegerMod::sqrt; for every n in 2..=60 and
    // a in 0..n the command must agree with IntegerMod: the leading root is `sqrt()`, the
    // listed roots are `all_sqrts()` (at most 8 here), and "no square root" appears exactly
    // when `sqrt()` is `None`.
    #[test]
    fn test_sqrt_mod_agrees_with_integer_mod() {
        let mut ctx = ReplContext::new();
        for n in 2i64..=60 {
            for a in 0..n {
                let text = sqrt_mod_text(&mut ctx, &format!("sqrt_mod({a}, {n})"));
                let x = IntegerMod::new(Integer::from(a), Integer::from(n)).unwrap();
                let expected = match x.sqrt() {
                    None => format!("no square root: {a} is not a square mod {n}"),
                    Some(r) => {
                        let all: Vec<String> =
                            x.all_sqrts().iter().map(|y| y.value().to_string()).collect();
                        format!("{}; square roots of {a} mod {n}: [{}]", r.value(), all.join(", "))
                    }
                };
                assert_eq!(text, quoted(&expected), "sqrt_mod({a}, {n})");
                assert_eq!(text.contains("no square root"), x.sqrt().is_none(), "sqrt_mod({a}, {n})");
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // T-250: checked integer arguments (X2). Reference values: PARI/GP 2.15.4 unless noted;
    // the gp expression is in the comment next to each group. GAP `Binomial` and Python
    // `math.comb` with (5.14) for the binomials PARI refuses (k >= 2^63: "overflow in
    // t_INT-->long assignment").
    // ---------------------------------------------------------------------------------

    fn int_text(ctx: &mut ReplContext, cmd: &str) -> String {
        ctx.eval(cmd)
            .unwrap_or_else(|e| panic!("{cmd}: {} {}", e.name, e.message))
            .text
    }

    /// Asserts a ValueError whose message contains `frag`.
    fn value_err(ctx: &mut ReplContext, cmd: &str, frag: &str) {
        match ctx.eval(cmd) {
            Ok(r) => panic!("{cmd}: expected a ValueError, got {}", r.text),
            Err(e) => {
                assert_eq!(e.name, "ValueError", "{cmd}: {}", e.message);
                assert!(e.message.contains(frag), "{cmd}: message {:?} lacks {:?}", e.message, frag);
            }
        }
    }

    // binomial for all integers (T-250). The old command truncated n and k to 32 bits.
    #[test]
    fn test_binomial_generalised() {
        let mut ctx = ReplContext::new();
        let cases = [
            // gp: binomial(10,3), binomial(100,50), binomial(0,0), binomial(3,5)
            ("binomial(10, 3)", "120"),
            ("binomial(100, 50)", "100891344545564193334812497256"),
            ("binomial(0, 0)", "1"),
            ("binomial(3, 5)", "0"),
            // gp: [binomial(4294967297,k) | k<-[1..3]]  (old: 1, 0, ...)
            ("binomial(4294967297, 1)", "4294967297"),
            ("binomial(4294967297, 2)", "9223372039002259456"),
            ("binomial(4294967297, 3)", "13204693752377389598208163840"),
            // gp: binomial(2^32, 2^32-1)  (old: 0)
            ("binomial(4294967296, 4294967295)", "4294967296"),
            // gp: [binomial(18446744073709551619,k) | k<-[1..3]]  (old: ValueError)
            ("binomial(18446744073709551619, 1)", "18446744073709551619"),
            ("binomial(18446744073709551619, 2)", "170141183460469231777804163900157984771"),
            ("binomial(18446744073709551619, 3)", "1046183622564446794312913937455549532847586212310908141569"),
            // GAP Binomial(2^64+3, 2^64+2); Python math.comb (PARI refuses)
            ("binomial(18446744073709551619, 18446744073709551618)", "18446744073709551619"),
            // gp: binomial(-1,2), binomial(-1,3), binomial(-5,3), binomial(-7,4)
            ("binomial(-1, 2)", "1"),
            ("binomial(-1, 3)", "-1"),
            ("binomial(-5, 3)", "-35"),
            ("binomial(-7, 4)", "210"),
            // gp: binomial(-4294967297,2), binomial(-18446744073709551619,3)
            ("binomial(-4294967297, 2)", "9223372043297226753"),
            ("binomial(-18446744073709551619, 3)", "-1046183622564446794653196304376487996421641284184933662730"),
            // gp: binomial(-1,-1), binomial(5,-1), binomial(-3,-5)
            ("binomial(-1, -1)", "0"),
            ("binomial(5, -1)", "0"),
            ("binomial(-3, -5)", "0"),
            // GAP Binomial(-1,2^64), Binomial(-1,2^64+1), Binomial(-2,2^40)
            ("binomial(-1, 18446744073709551616)", "1"),
            ("binomial(-1, 18446744073709551617)", "-1"),
            ("binomial(-2, 1099511627776)", "1099511627777"),
        ];
        for (cmd, expected) in cases {
            assert_eq!(int_text(&mut ctx, cmd), expected, "{cmd}");
        }
        // K = min(k, N - k) <= 1 skips the limits, so n may have any size (review 2, N1).
        // gp: binomial(2^1100,1) == 2^1100; binomial(2^4200000,1) == 2^4200000 and
        // binomial(-2^4200000,1) == -2^4200000 (both 1); C(N, N-1) = C(N, 1) = N.
        ctx.eval("a = 2 ** 1100").unwrap();
        assert_eq!(int_text(&mut ctx, "binomial(a, 1) - a"), "0");
        ctx.eval("t = 2 ** 10000").unwrap();
        ctx.eval("b = t ** 420").unwrap(); // 2^4200000, more than BINOMIAL_BITS = 2^22 bits
        assert_eq!(int_text(&mut ctx, "binomial(b, 0)"), "1");
        assert_eq!(int_text(&mut ctx, "binomial(b, 1) - b"), "0");
        assert_eq!(int_text(&mut ctx, "binomial(b, b - 1) - b"), "0");
        assert_eq!(int_text(&mut ctx, "binomial(0 - b, 1) + b"), "0");
        value_err(&mut ctx, "binomial(b, 2)", "4194304 bits");
        // Unchanged errors.
        assert_eq!(ctx.eval("binomial(1, 2, 3)").unwrap_err().name, "ArgumentError");
        assert_eq!(ctx.eval("binomial(1.5, 2)").unwrap_err().name, "TypeError");
    }

    // The size and work limits of binomial. Each refused call is refused before any big
    // multiplication. The admitted calls near the limit take 2-21 s in release, so the
    // estimate is checked instead of running them.
    #[test]
    fn test_binomial_limits() {
        let mut ctx = ReplContext::new();
        // Size: K·(bits(N) - bits(K) + 3) > 2^22, or K > 2^22/3.
        for cmd in [
            "binomial(18446744073709551616, 4611686018427387904)",
            "binomial(18446744073709551616, 90000)",
            "binomial(18446744073709551616, 18446744073709461616)",
            "binomial(-18446744073709551616, 90000)",
        ] {
            value_err(&mut ctx, cmd, "binomial:");
            value_err(&mut ctx, cmd, "may have more than 4194304 bits (REPL limit)");
        }
        // Work: K · bound > 4·10^11 (release: more than about 12 s).
        for cmd in [
            "binomial(1000000, 500000)",
            "binomial(1048576, 524288)",
            "binomial(-1000000, 500000)",
            "binomial(4294967295, 200000)",
        ] {
            value_err(&mut ctx, cmd, "binomial:");
            value_err(&mut ctx, cmd, "exceeds 400000000000 (REPL limit");
        }
        let int = |s: &str| Integer::from(s.parse::<BigInt>().unwrap());
        let est = |n: &str, k: &str| binomial_estimate(&int(n), &int(k));
        let ok = |(b, w): (u64, u64)| b <= repl_limits::BINOMIAL_BITS && w <= repl_limits::BINOMIAL_WORK;
        // Admitted (release times of the mirror loop in parentheses).
        assert_eq!(est("524288", "262144"), (1048576, 274877906944)); // 9.8 s
        assert_eq!(est("524288", "262143"), (1310715, 343594762245)); // 10.5 s
        assert_eq!(est("4294967295", "90000"), (1620000, 145800000000)); // 7.4 s
        assert_eq!(est("200000", "100000"), (400000, 40000000000)); // 1.5 s, = catalan(100000)
        assert_eq!(est("18446744073709551616", "30000"), (1590000, 47700000000));
        for (n, k) in [("524288", "262144"), ("524288", "262143"), ("4294967295", "90000"),
                       ("200000", "100000"), ("18446744073709551616", "30000")] {
            assert!(ok(est(n, k)), "C({n}, {k}) must be admitted");
        }
        // Refused.
        assert_eq!(est("18446744073709551616", "4611686018427387904"), (u64::MAX, u64::MAX));
        assert_eq!(est("18446744073709551616", "90000"), (4590000, 413100000000));
        assert_eq!(est("1000000", "500000"), (2000000, 1000000000000));
        assert!(!ok(est("18446744073709551616", "90000")) && !ok(est("1000000", "500000")));
        // The u32 variant used by catalan, narayana and delannoy agrees with it.
        for (n, k) in [(524288u32, 262144u32), (200000, 100000), (4294967295, 90000), (10, 3)] {
            assert_eq!(binomial_work_u32(n, k), est(&n.to_string(), &k.to_string()).1, "C({n}, {k})");
        }
        assert_eq!(binomial_work_u32(5, 0), 0);
        assert_eq!(binomial_work_u32(5, 5), 0);
        assert_eq!(binomial_work_u32(5, 7), 0);
    }

    // Consistency of the REPL limits with the work models they stand for.
    #[test]
    fn test_repl_limit_models() {
        // catalan(n) = C(2n, n)/(n+1) is admitted exactly when binomial(2n, n) is.
        let n = repl_limits::CATALAN_N;
        assert!(binomial_work_u32(2 * n, n) <= repl_limits::BINOMIAL_WORK);
        assert!(binomial_work_u32(2 * (n + 1), n + 1) > repl_limits::BINOMIAL_WORK);
        // delannoy: above the cap the work is at least sum 4·min(k, s-k)^2 > BINOMIAL_WORK, so
        // reporting u64::MAX there refuses nothing the sum would admit.
        let s = u64::from(repl_limits::DELANNOY_SUM_CAP) + 1;
        let lower: u64 = (0..=s).map(|k| 4 * k.min(s - k).pow(2)).sum();
        assert!(lower > repl_limits::BINOMIAL_WORK);
        assert_eq!(delannoy_work(repl_limits::DELANNOY_SUM_CAP + 1, u32::MAX, false), u64::MAX);
        assert_eq!(delannoy_work(1, u32::MAX, false), 34); // only C(n, 1): one step, 1·1·(32 - 1 + 3)
        assert_eq!(delannoy_work(10000, 10000, false), 771337339340);
        assert_eq!(delannoy_work(10000, 10000, true), 385668669670);
        // product_work: k · k · bits(top).
        assert_eq!(product_work(300000, u32::MAX), 2880000000000);
        assert_eq!(product_work(250000, 250000), 1125000000000);
        assert_eq!(product_work(0, 7), 0);
    }

    // Each of these was a silent wrong answer before (truncation to 32 bits or a usize wrap).
    #[test]
    fn test_int_args_truncation_fixed() {
        let mut ctx = ReplContext::new();
        let catalan_range = format!("catalan: n must be an integer in 0..={}", repl_limits::CATALAN_N);
        let fib_range = format!("fibonacci: n must be an integer in -{0}..={0}", repl_limits::FIBONACCI_N);
        let partitions_range = format!("partitions: n must be an integer in 0..={}", repl_limits::PARTITIONS_N);
        let cases: Vec<(&str, String)> = vec![
            ("catalan(4294967297)", catalan_range.clone()),           // old: 1
            ("catalan(2147483648)", catalan_range),                   // old: 0 (release)
            ("fibonacci(4294967296)", fib_range),                     // old: 0
            ("multinomial(4294967302, [2, 2, 2])", "multinomial: n must be an integer in 0..=4294967295".into()), // old: 90
            ("multinomial(0, [2147483648, 2147483648])",
             "multinomial requires sum of k values (4294967296) to equal n (0)".into()), // old: panic / hang
            ("sigma(2, 4294967296)", "sigma: k must be an integer in 0..=4294967295".into()), // old: sigma_0(2) = 2
            ("matrix_power(Matrix([[1, 1], [0, 1]]), 4294967296)",
             "matrix_power: exponent must be an integer in 0..=4294967295".into()), // old: identity
            ("stirling2(4294967298, 2)", "stirling2: n must be an integer in 0..=4294967295".into()), // old: 1
            ("falling_factorial(4294967297, 2)", "falling_factorial: n must be an integer in 0..=4294967295".into()), // old: 0
            ("rising_factorial(4294967295, 2)",
             "rising_factorial: n must be an integer in 0..=4294967294 (n + k - 1 must fit in 32 bits)".into()), // old: panic / 0
            ("partitions(417)", partitions_range), // old: panic / 541220193622112941
        ];
        for (cmd, frag) in &cases {
            value_err(&mut ctx, cmd, frag);
        }
        assert_eq!(int_text(&mut ctx, "row(matrix_power(Matrix([[1, 1], [0, 1]]), 3), 0)"), "[1, 3]");
    }

    // Each of these panicked before (inherent Integer::to_i64), which killed the kernel.
    #[test]
    fn test_int_args_no_panic() {
        let mut ctx = ReplContext::new();
        let big = "9223372036854775808"; // 2^63
        for (cmd, frag) in [
            (format!("factorial({big})"), "factorial: n must be an integer in 0..=10000"),
            (format!("pow(2, {big})"), "pow: exponent must be an integer in 0..=1000"),
            ("2 ** 18446744073709551616".to_string(), "**: exponent must be an integer in 0..=10000"),
            (format!("identity({big})"), "identity: n must be an integer in 0..=1000"),
            (format!("zeros(2, {big})"), "zeros: cols must be"),
            (format!("ones({big}, 2)"), "ones: rows must be an integer in 0..=100"),
            (format!("matrix_power(Matrix([[1, 1], [0, 1]]), {big})"), "matrix_power: exponent must be"),
            (format!("row(Matrix([[1, 2], [3, 4]]), {big})"), "row: i must be an integer in 0..=1 (2 rows)"),
            (format!("histogram([1, 2, 3], {big})"), "histogram: bins must be an integer in 1..=1000"),
            (format!("CliffordAlgebra({big})"), "CliffordAlgebra: dimension must be an integer in 0..=10"),
            (format!("e({big})"), "e: index must be an integer in 0..=9"),
            ("Manifold(18446744073709551616, \"M\")".to_string(), "Manifold: dimension must be an integer in 0..="),
        ] {
            value_err(&mut ctx, &cmd, frag);
        }
        // gp: 2.^63 = 9223372036854775808.0 (exact in a double). Rust prints the shortest
        // round-trip digits padded with zeros, so the real part is parsed back.
        let c = int_text(&mut ctx, &format!("Complex({big}, 0)"));
        assert!(c.ends_with(" + 0i"), "{c}");
        let re: f64 = c.trim_end_matches(" + 0i").parse().unwrap();
        assert_eq!(re, 9223372036854775808.0);
        ctx.eval("c = 2 ** 1100").unwrap();
        value_err(&mut ctx, "Complex(c, 0)", "Complex: re");
        // gp: matrank([2^64,1;1,1]) = 2
        assert_eq!(int_text(&mut ctx, "rank(Matrix([[18446744073709551616, 1], [1, 1]]))"), "2");
        assert!(ctx.eval("solve_linear(Matrix([[18446744073709551616, 0], [0, 1]]), [18446744073709551616, 3])").is_ok());
        assert!(ctx.eval("JordanAlgebra(Matrix([[18446744073709551616, 0], [0, 1]]))").is_ok());
        ctx.eval("e1 = e(0)").unwrap();
        assert!(int_text(&mut ctx, "e1 * 18446744073709551616").contains("18446744073709551616"));
        assert!(ctx.eval("e1 / 18446744073709551616").is_ok());
        ctx.eval(&format!("c = {big}")).unwrap();
        value_err(&mut ctx, "\"ab\" * c", "*: repetition count must be an integer in 0..=10000");
        ctx.eval("c = -9223372036854775809").unwrap();
        assert_eq!(int_text(&mut ctx, "\"ab\" * c"), quoted(""));
        // grade and basis_of_grade are total in k: outside 0..=dim there is nothing.
        let zero = int_text(&mut ctx, "grade(e1, 5)");
        assert_eq!(int_text(&mut ctx, &format!("grade(e1, {big})")), zero);
        assert_eq!(int_text(&mut ctx, "grade(e1, -1)"), zero);
        ctx.eval("Cl3 = CliffordAlgebra(3)").unwrap();
        let empty = int_text(&mut ctx, "basis_of_grade(Cl3, 4)");
        assert_eq!(int_text(&mut ctx, "basis_of_grade(Cl3, -1)"), empty);
        assert_eq!(int_text(&mut ctx, "basis_of_grade(Cl3, 18446744073709551616)"), empty);
    }

    // Values inside the ranges, from the existing callees (binomial, fibonacci and lucas from
    // the generalised ones).
    #[test]
    fn test_combinatorics_reference_values() {
        let mut ctx = ReplContext::new();
        let cases = [
            // gp: [binomial(2*n,n)/(n+1) | n<-[0,10,100]]
            ("catalan(0)", "1"),
            ("catalan(10)", "16796"),
            ("catalan(100)", "896519947090131496687170070074100632420837521538745909320"),
            // gp: [fibonacci(n) | n<-[50,-5,-6,-100]]
            ("fibonacci(50)", "12586269025"),
            ("fibonacci(-5)", "5"),
            ("fibonacci(-6)", "-8"),
            ("fibonacci(-100)", "-354224848179261915075"),
            // gp: L(n)=fibonacci(n-1)+fibonacci(n+1); [L(n) | n<-[10,-5,-6,-100]]
            ("lucas(10)", "123"),
            ("lucas(-5)", "-11"),
            ("lucas(-6)", "18"),
            ("lucas(-100)", "792070839848372253127"),
            // gp: B(n)=sum(k=0,n,stirling(n,k,2)); [B(10), B(50)]
            ("bell(10)", "115975"),
            ("bell(50)", "185724268771078270438257767181908917499221852770"),
            // gp: abs(stirling(10,3,1)), stirling(10,3,2), stirling(5,7,1), stirling(0,0,2)
            ("stirling1(10, 3)", "1172700"),
            ("stirling2(10, 3)", "9330"),
            ("stirling1(5, 7)", "0"),
            ("stirling1(5, 4294967295)", "0"),
            ("stirling2(0, 0)", "1"),
            // gp: prod(i=0,k-1,n-i) and prod(i=0,k-1,n+i)
            ("falling_factorial(10, 3)", "720"),
            ("falling_factorial(5, 7)", "0"),
            ("falling_factorial(5, 1000000)", "0"), // k > n: no work limit
            ("falling_factorial(4294967295, 2)", "18446744060824649730"),
            ("rising_factorial(3, 4)", "360"),
            ("rising_factorial(4294967294, 2)", "18446744060824649730"),
            ("rising_factorial(0, 0)", "1"),
            ("rising_factorial(0, 5)", "0"),
            ("rising_factorial(0, 4000000000)", "0"),
            // gp: 20!/(10!*10!)
            ("multinomial(20, [10, 10])", "184756"),
            // gp: eulerianpol(5) = x^4 + 26*x^3 + 66*x^2 + 26*x + 1
            ("eulerian(5, 1)", "26"),
            ("eulerian(5, 2)", "66"),
            // gp: N(n,k)=binomial(n,k)*binomial(n,k-1)/n
            ("narayana(5, 2)", "10"),
            ("narayana(5, 3)", "20"),
            ("narayana(4294967295, 1)", "1"),
            ("narayana(4294967295, 2)", "9223372030412324865"),
            // gp: sum(k=0,n\2,binomial(n,2k)*binomial(2k,k)/(k+1))
            ("motzkin(10)", "2188"),
            // gp: D(m,n)=sum(k=0,min(m,n),binomial(m,k)*binomial(n,k)*2^k)
            ("delannoy(3)", "63"),
            ("delannoy(2, 3)", "25"),
            ("delannoy(10, 10)", "8097453"),
            ("delannoy(1, 20001)", "40003"),
            ("delannoy(1, 4294967295)", "8589934591"),
            // gp: sum(k=0,n,binomial(n+k,n-k)*binomial(2k,k)/(k+1))
            ("schroder(6)", "1806"),
            ("schroder_small(6)", "903"),
            // gp: numbpart(100)
            ("partitions(100)", "190569292"),
            // gp: n!*sum(i=0,n,(-1)^i/i!)
            ("derangements(10)", "1334961"),
            // gp: sigma(12,2), sigma(12,0), sigma(-12,2), sigma(1,4294967295)
            ("sigma(12, 2)", "210"),
            ("sigma(12, 0)", "6"),
            ("sigma(-12, 2)", "210"),
            ("sigma(1, 4294967295)", "1"),
            // gp: prime(100), 10!
            ("nth_prime(100)", "541"),
            ("factorial(10)", "3628800"),
        ];
        for (cmd, expected) in cases {
            assert_eq!(int_text(&mut ctx, cmd), expected, "{cmd}");
        }
        // gp: numbpart(416) = 17873792969689876004, the top of the 64-bit range.
        #[cfg(target_pointer_width = "64")]
        assert_eq!(int_text(&mut ctx, "partitions(416)"), "17873792969689876004");
        // gp: sigma(2,10001) == 2^10001+1. A power of 2 is cheap at any k (no k-only limit).
        ctx.eval("c = 2 ** 10000").unwrap();
        assert_eq!(int_text(&mut ctx, "sigma(2, 10001) - c * 2"), "1");
    }

    // Out-of-range inputs name the command, the argument and the range; limits + 1.
    #[test]
    fn test_int_arg_ranges_named() {
        let mut ctx = ReplContext::new();
        let cases: Vec<(String, String)> = vec![
            ("factorial(-1)".into(), "factorial: n must be an integer in 0..=10000".into()),
            ("catalan(-1)".into(), format!("catalan: n must be an integer in 0..={}", repl_limits::CATALAN_N)),
            (format!("catalan({})", repl_limits::CATALAN_N + 1), "catalan: n must be".into()),
            (format!("fibonacci({})", repl_limits::FIBONACCI_N + 1),
             format!("fibonacci: n must be an integer in -{0}..={0}", repl_limits::FIBONACCI_N)),
            (format!("lucas(-{})", repl_limits::FIBONACCI_N + 1),
             format!("lucas: n must be an integer in -{0}..={0}", repl_limits::FIBONACCI_N)),
            (format!("bell({})", repl_limits::BELL_N + 1), format!("bell: n must be an integer in 0..={}", repl_limits::BELL_N)),
            // Near the diagonal, refused on 64-bit and 32-bit targets (test_table_limits):
            // 472.8 / 447.8, 461.7 / 434.6 and 445.1 / 425.7 MB.
            ("stirling1(1250, 1249)".into(), "stirling1: the recurrence table for n = 1250, k = 1249 needs at least".into()),
            ("stirling2(1300, 1299)".into(), "(REPL limit 400000000 bytes, a memory bound)".into()),
            ("eulerian(1100, 1099)".into(), "eulerian: the recurrence table for n = 1100, k = 1099".into()),
            ("stirling1(-1, 1)".into(), "stirling1: n must be an integer in 0..=4294967295".into()),
            ("stirling2(5, -1)".into(), "stirling2: k must be an integer in 0..=4294967295".into()),
            ("multinomial(4, [5, -1])".into(), "multinomial: every k must be a non-negative integer, got -1".into()),
            ("multinomial(300000, [300000])".into(), "multinomial: n! for n = 300000".into()),
            ("multinomial(300000, [300000])".into(), "exceeds 2500000000000 (REPL limit".into()),
            ("falling_factorial(4294967295, 300000)".into(), "falling_factorial: (n)_k for n = 4294967295, k = 300000".into()),
            ("rising_factorial(1, 400000)".into(), "rising_factorial: n^(k) for n = 1, k = 400000".into()),
            ("rising_factorial(1, 400000)".into(), "steps x bits = 3040000000000 exceeds 2500000000000".into()),
            ("narayana(2000000, 1000000)".into(), "narayana: N(n, k) for n = 2000000, k = 1000000 computes C(n, k) and C(n, k-1)".into()),
            ("narayana(2000000, 1000000)".into(), "exceeds 400000000000".into()),
            (format!("motzkin({})", repl_limits::MOTZKIN_N + 1), format!("motzkin: n must be an integer in 0..={}", repl_limits::MOTZKIN_N)),
            ("delannoy(20001)".into(), "delannoy: D(n) for n = 20001".into()),
            ("delannoy(20001)".into(), "more than 2^64".into()),
            ("delannoy(10000, 10000)".into(), "steps x bits = 771337339340 exceeds 400000000000".into()),
            ("delannoy(-1, 2)".into(), "delannoy: m must be an integer in 0..=4294967295".into()),
            (format!("schroder({})", repl_limits::SCHRODER_N + 1), format!("schroder: n must be an integer in 0..={}", repl_limits::SCHRODER_N)),
            ("schroder_small(-1)".into(), format!("schroder_small: n must be an integer in 0..={}", repl_limits::SCHRODER_N)),
            ("partitions(-1)".into(), format!("partitions: n must be an integer in 0..={}", repl_limits::PARTITIONS_N)),
            (format!("derangements({})", repl_limits::DERANGEMENTS_N + 1),
             format!("derangements: n must be an integer in 0..={}", repl_limits::DERANGEMENTS_N)),
            ("sigma(12, -1)".into(), "sigma: k must be an integer in 0..=4294967295".into()),
            ("nth_prime(0)".into(), format!("nth_prime: n must be an integer in 1..={}", usize::MAX)),
            ("nth_prime(-5)".into(), format!("nth_prime: n must be an integer in 1..={}", usize::MAX)),
            ("identity(-1)".into(), "identity: n must be an integer in 0..=1000".into()),
            ("zeros(-1, 2)".into(), "zeros: rows must be an integer in 0..=1000".into()),
            ("ones(2, 101)".into(), "ones: cols must be an integer in 0..=100".into()),
            ("pow(2, -1)".into(), "pow: exponent must be an integer in 0..=1000".into()),
            ("2 ** 10001".into(), "**: exponent must be an integer in 0..=10000".into()),
            ("row(Matrix([[1, 2], [3, 4]]), 2)".into(), "row: i must be an integer in 0..=1 (2 rows), got 2".into()),
            ("col(Matrix([[1, 2], [3, 4]]), -1)".into(), "col: j must be an integer in 0..=1 (2 columns), got -1".into()),
            ("histogram([1, 2, 3], 0)".into(), "histogram: bins must be an integer in 1..=1000".into()),
            ("CliffordAlgebra(11)".into(), "CliffordAlgebra: dimension must be an integer in 0..=10".into()),
            ("ExteriorAlgebra(-1)".into(), "ExteriorAlgebra: dimension must be an integer in 0..=10".into()),
            ("e(10)".into(), "e: index must be an integer in 0..=9".into()),
            ("JordanAlgebra(21)".into(), "JordanAlgebra: dimension must be an integer in 0..=20".into()),
            ("SpecialJordanAlgebra(-1)".into(), "SpecialJordanAlgebra: size must be an integer in 0..=10".into()),
            ("EuclideanSpace(-1)".into(), format!("EuclideanSpace: dimension must be an integer in 0..={}", repl_limits::EUCLIDEAN_DIM)),
            (format!("EuclideanSpace({})", repl_limits::EUCLIDEAN_DIM + 1), "EuclideanSpace: dimension must be".into()),
            ("Manifold(-1, \"M\")".into(), format!("Manifold: dimension must be an integer in 0..={}", usize::MAX)),
        ];
        for (cmd, frag) in &cases {
            value_err(&mut ctx, cmd, frag);
        }
        // sigma: the memory bound (the time estimate is in test_sigma_limits).
        ctx.eval("c = 2 ** 2000").unwrap();
        value_err(&mut ctx, "sigma(c, 140000)", "a memory bound"); // 140000·2000 > 2^28
        // Clifford generators.
        ctx.eval("Cl3 = CliffordAlgebra(3)").unwrap();
        value_err(&mut ctx, "e(Cl3, 3)", "e: index must be an integer in 0..=2 (3 generators), got 3");
        ctx.eval("C0 = CliffordAlgebra(0)").unwrap();
        value_err(&mut ctx, "e(C0, 0)", "e: there are no generators to index");
        // Manifolds: a manifold allocates nothing of its dimension, so any usize is accepted.
        assert_eq!(int_text(&mut ctx, "Manifold(5000, \"M\")"), "Manifold('M', dim=5000)");
        ctx.eval("M = Manifold(3, \"M\")").unwrap();
        value_err(&mut ctx, "diff_form(M, -1)", "diff_form: degree must be an integer in 0..=");
        value_err(&mut ctx, &format!("diff_form(M, {})", repl_limits::DIFF_FORM_DEGREE as u128 + 1),
            "(p + 1 and p + q must fit in a machine word)");
        assert_eq!(int_text(&mut ctx, "diff_form(M, 5)"), "DiffForm(degree=5)");
        assert_eq!(int_text(&mut ctx, "diff_form(M, 1001)"), "DiffForm(degree=1001)");
        // A one-name chart: `chart(E3, "x,y,z")` cannot be written (commas inside a string are
        // split, S9), and `chart` does not compare the names with dim(M) (S14).
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("C = chart(E3, \"t\")").unwrap();
        assert!(ctx.eval("coordinate_form(E3, C, 0)").is_ok());
        value_err(&mut ctx, "coordinate_form(E3, C, 3)", "coordinate_form: index must be an integer in 0..=2 (3 coordinates)");
        value_err(&mut ctx, "coordinate_form(E3, C, -1)", "coordinate_form: index must be an integer in 0..=2 (3 coordinates)");
        let big = repl_limits::MANIFOLD_COORDS + 1;
        ctx.eval(&format!("Mbig = Manifold({}, \"B\")", big)).unwrap();
        ctx.eval("Cb = chart(Mbig, \"t\")").unwrap();
        value_err(&mut ctx, "coordinate_form(Mbig, Cb, 0)", &format!("coordinate_form: the manifold has dimension {}", big));
        // plot_function's n_points (a float before the cast): 2.5 silently became 2.
        assert!(ctx.eval("plot_function(x^2, x, 0, 1, 3)").is_ok());
        value_err(&mut ctx, "plot_function(x^2, x, 0, 1, 2.5)", "plot_function: n_points must be an integer in 2..=10000");
        value_err(&mut ctx, "plot_function(x^2, x, 0, 1, 1)", "plot_function: n_points must be an integer in 2..=10000");
    }

    #[test]
    fn test_taylor_order_checked() {
        let mut ctx = ReplContext::new();
        assert!(ctx.eval("taylor(\"x^2\", x, 0, 2)").is_ok());
        // The parser reads -1 as Neg(1), which silently became order 5 before.
        let range = format!("taylor: order must be an integer in 0..={}", repl_limits::TAYLOR_ORDER);
        value_err(&mut ctx, "taylor(\"x^2\", x, 0, -1)", &range);
        value_err(&mut ctx, &format!("taylor(\"x^2\", x, 0, {})", repl_limits::TAYLOR_ORDER + 1), &range);
        let e = ctx.eval("taylor(\"x^2\", x, 0, y)").unwrap_err();
        assert_eq!(e.name, "TypeError");
        assert!(e.message.contains("taylor: order must be an integer"), "{}", e.message);
    }

    // stirling1/stirling2/eulerian: the limit is the table's size in (n, k), not n (fix-up B2).
    #[test]
    fn test_table_limits() {
        use TableKind::*;
        let lim = repl_limits::TABLE_BYTES;
        // Exact sizes with the formula of `table_bytes`, from Python big integers (fix-up
        // x2_fix1/st2.py, x2_fix2/st3.py): (kind, n, k, entry bytes, total on 64-bit targets).
        // The entry part (8 bytes per 64-bit word + 16 per nonzero entry) is the same on every
        // target; the headers come from size_of (Vec and Integer: 24 and 32 bytes on 64-bit
        // targets, 12 and 16 on 32-bit ones), so the expected total is built the same way.
        let headers = |n: u32, k: u32| -> u64 {
            let row = std::mem::size_of::<Vec<Integer>>() + std::mem::size_of::<Integer>() * (k as usize + 1);
            (u64::from(n) + 1) * row as u64
        };
        for (kind, n, k, entries, total64) in [
            (Stirling1, 1000, 999, 211_698_192u64, 243_754_216u64),
            (Stirling2, 1000, 999, 179_950_824, 212_006_848),
            (Eulerian, 1000, 999, 300_533_720, 332_589_744),
            (Stirling1, 10000, 1, 69_724_496, 70_604_584),
            (Stirling2, 20000, 2, 25_877_568, 28_277_688),
            (Eulerian, 20000, 1, 25_880_064, 27_640_152),
            (Stirling2, 5000, 10, 35_016_104, 36_896_480),
            (Eulerian, 1060, 1059, 361_370_144, 397_384_728),
            (Stirling1, 1180, 1179, 353_545_064, 398_167_968),
        ] {
            let exact = entries + headers(n, k);
            if cfg!(target_pointer_width = "64") {
                assert_eq!(exact, total64, "{kind:?}({n}, {k})");
            }
            let est = table_bytes(kind, n, k, lim).unwrap();
            assert!(est.abs_diff(exact) <= exact / 10_000, "{kind:?}({n}, {k}): {est} vs {exact}");
            assert!(est <= lim, "{kind:?}({n}, {k}) must be admitted");
        }
        // Refused on 64-bit and 32-bit targets: 472.8 / 447.8, 461.7 / 434.6 and
        // 445.1 / 425.7 MB (st3.py; the estimate stops once above the limit).
        for (kind, n, k) in [(Stirling1, 1250, 1249), (Stirling2, 1300, 1299), (Eulerian, 1100, 1099)] {
            assert!(table_bytes(kind, n, k, lim).unwrap() > lim, "{kind:?}({n}, {k}) must be refused");
        }
        // Just above the limit on 64-bit targets only: 418.6 and 410.7 MB there, but 395.5 and
        // 385.7 MB (admitted) with the 32-bit headers.
        #[cfg(target_pointer_width = "64")]
        for (kind, n, k) in [(Stirling1, 1200, 1199), (Stirling2, 1250, 1249)] {
            assert!(table_bytes(kind, n, k, lim).unwrap() > lim, "{kind:?}({n}, {k}) must be refused");
        }
        // Headers alone: (2^32 - 1) rows.
        assert!(table_bytes(Stirling2, u32::MAX, 2, lim).unwrap() > lim);
        // No table: the callee answers at once.
        for (kind, n, k) in [(Stirling1, 5, 7), (Stirling1, u32::MAX, u32::MAX), (Stirling1, u32::MAX, 0),
                             (Stirling2, u32::MAX, 1), (Stirling2, 0, 0), (Eulerian, u32::MAX, 0), (Eulerian, 7, 7)] {
            assert_eq!(table_bytes(kind, n, k, lim), None, "{kind:?}({n}, {k})");
        }
        let mut ctx = ReplContext::new();
        // Refused in round 0 (n > 1000) although cheap. gp: abs(stirling(1001,1,1)) == 1000!
        assert_eq!(int_text(&mut ctx, "stirling1(1001, 1)"), int_text(&mut ctx, "factorial(1000)"));
        // gp: A(n,k)=sum(j=0,k,(-1)^j*binomial(n+1,j)*(k+1-j)^n); A(1001,1) - 2^1001 = -1002
        assert_eq!(int_text(&mut ctx, "eulerian(1001, 1) - 2 ** 1001"), "-1002");
        // gp: stirling(n,1,2) = 1, stirling(n,n,1) = 1, A(n,0) = 1 (no table at any n).
        assert_eq!(int_text(&mut ctx, "stirling2(4294967295, 1)"), "1");
        assert_eq!(int_text(&mut ctx, "stirling1(4294967295, 4294967295)"), "1");
        assert_eq!(int_text(&mut ctx, "eulerian(4294967295, 0)"), "1");
        value_err(&mut ctx, "stirling2(4294967295, 2)", "stirling2: the recurrence table for n = 4294967295, k = 2");
    }

    // sigma(n, k): the time estimate sees the factorisation (fix-up B1).
    #[test]
    fn test_sigma_limits() {
        let f = |pa: &[(u64, u32)]| -> Vec<(Integer, u32)> { pa.iter().map(|&(p, a)| (Integer::from(p), a)).collect() };
        // Admitted: odd prime powers that the old Integer::sigma computed in 2-10 s (release;
        // fix-up report §B1). The first was refused in round 0.
        for (pa, k) in [
            (vec![(3, 10000)], 2200),
            (vec![(3, 1000)], 21200),
            (vec![(3, 100)], 212000),
            (vec![(3, 20)], 1060000),
            (vec![(3, 5)], 4234097),
            (vec![(3, 1), (5, 1), (7, 1), (11, 1), (13, 1)], 1710131),
            (vec![(3, 1)], 10585244),
        ] {
            let s = sigma_seconds(&f(&pa), k);
            assert!(s <= repl_limits::SIGMA_SECONDS, "{pa:?}, k = {k}: {s} s");
        }
        // Refused: a = 1 at 2^25 bits (old code 24.3 s).
        assert!(sigma_seconds(&f(&[(3, 1)]), 21170489) > repl_limits::SIGMA_SECONDS);
        // The power of 2 is left out (a lower estimate).
        assert_eq!(sigma_seconds(&f(&[(2, 65535)]), 4096), 0.0);
        let mut ctx = ReplContext::new();
        value_err(&mut ctx, "sigma(3, 25000000)", "sigma: sigma_k(n) for k = 25000000 needs about");
        value_err(&mut ctx, "sigma(3, 25000000)", &format!("(REPL limit {} s, a lower estimate)", repl_limits::SIGMA_SECONDS));
        // `sigma_from_factors` against PARI and Integer::sigma:
        // gp: sigma(720,3), sigma(2^10*3*1000003,2), sigma(3^5*7^2,7), sigma(2,1)
        for (n, k, expected) in [
            (720i64, 3u32, "446483142"),
            (-720, 3, "446483142"),
            (1024 * 3 * 1000003, 2, "13981093886199810100"),
            (243 * 49, 7, "33948112144628376652738148004"),
            (2, 1, "3"),
        ] {
            let n = Integer::from(n);
            let factors = rustmath_integers::prime::factor(&n);
            assert_eq!(sigma_from_factors(&factors, k).to_string(), expected, "sigma({n}, {k})");
            assert_eq!(n.sigma(k).unwrap().to_string(), expected, "Integer::sigma({n}, {k})");
        }
    }

    // exterior_derivative and wedge build every component of the result (fix-up B3).
    #[test]
    fn test_form_component_limits() {
        let mut ctx = ReplContext::new();
        let lim = format!("(REPL limit {}, a memory bound)", repl_limits::FORM_COMPONENTS);
        // Manifold(100000) is accepted and coordinate_form stores 10^5 components, but d(w) and
        // w ∧ w would build 10^10 (an allocation failure, which aborts the kernel).
        ctx.eval("M = Manifold(100000, \"M\")").unwrap();
        ctx.eval("C = chart(M, \"t\")").unwrap();
        ctx.eval("w = coordinate_form(M, C, 0)").unwrap();
        for cmd in ["exterior_derivative(w, C)", "wedge(w, w, C)"] {
            let name = cmd.split('(').next().unwrap();
            value_err(&mut ctx, cmd, &format!("{name}: the result, a 2-form on a manifold of dimension 100000, has "));
            value_err(&mut ctx, cmd, &lim);
        }
        // Small forms on R^3 work, also derived ones.
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("C3 = chart(E3, \"t\")").unwrap();
        ctx.eval("a = coordinate_form(E3, C3, 0)").unwrap();
        ctx.eval("b = coordinate_form(E3, C3, 1)").unwrap();
        ctx.eval("da = exterior_derivative(a, C3)").unwrap();
        assert!(ctx.eval("exterior_derivative(da, C3)").is_ok());
        assert!(ctx.eval("wedge(a, b, C3)").is_ok());
        assert!(ctx.eval("wedge(da, b, C3)").is_ok());
        // The check itself: 1000^2 = 10^6 and 100^3 = 10^6 components are admitted.
        assert!(check_form_components("x", 1000, 2).is_ok());
        assert!(check_form_components("x", 100, 3).is_ok());
        assert!(check_form_components("x", 0, 0).is_ok());
        assert!(check_form_components("x", 1, 1_000_000).is_ok());
        assert!(check_form_components("x", repl_limits::FORM_COMPONENTS + 1, 1).is_err());
        assert!(check_form_components("x", 2, 64).is_err());
        // Degrees of derived forms (the p >= n and p + q > n branches build no components):
        // g = d(f) has degree usize::MAX/2 + 1, so g ∧ g would overflow p + q.
        ctx.eval("M3 = Manifold(3, \"M3\")").unwrap();
        ctx.eval("Cm = chart(M3, \"t\")").unwrap();
        ctx.eval(&format!("f = diff_form(M3, {})", repl_limits::DIFF_FORM_DEGREE)).unwrap();
        ctx.eval("g = exterior_derivative(f, Cm)").unwrap();
        let half = repl_limits::DIFF_FORM_DEGREE + 1;
        value_err(&mut ctx, "wedge(g, g, Cm)",
            &format!("wedge: the degree p + q of the product for p = {half}, q = {half} does not fit in a machine word"));
        // h = f ∧ f has degree usize::MAX - 1, d(h) has usize::MAX, and d(d(h)) would overflow.
        ctx.eval("h = wedge(f, f, Cm)").unwrap();
        ctx.eval("h1 = exterior_derivative(h, Cm)").unwrap();
        value_err(&mut ctx, "exterior_derivative(h1, Cm)",
            &format!("exterior_derivative: the degree p + 1 of d(omega) for p = {} does not fit in a machine word", usize::MAX));
    }

    // A form without components in the chart is the callee's error (as at HEAD), not the
    // component limit's, also where the result would be over the limit (fix-up 2, m3).
    #[test]
    fn test_form_without_components() {
        let mut ctx = ReplContext::new();
        let no_comps = |ctx: &mut ReplContext, cmd: &str| {
            let e = ctx.eval(cmd).unwrap_err();
            assert_eq!(e.name, "ComputationError", "{cmd}: {}", e.message);
            assert_eq!(e.message, "No components defined in this chart", "{cmd}");
        };
        ctx.eval("N = Manifold(2000, \"N\")").unwrap();
        ctx.eval("CN = chart(N, \"t\")").unwrap();
        ctx.eval("f = diff_form(N, 1)").unwrap(); // no components
        ctx.eval("w = coordinate_form(N, CN, 0)").unwrap();
        // d and ∧ would have 2000^2 = 4·10^6 components, above FORM_COMPONENTS. (Every REPL
        // chart is named "user_chart", so a form with components has them in every REPL chart.)
        for cmd in ["exterior_derivative(f, CN)", "wedge(f, f, CN)", "wedge(f, w, CN)", "wedge(w, f, CN)"] {
            no_comps(&mut ctx, cmd);
        }
        // With components in the chart, the limit answers.
        let lim = format!("(REPL limit {}, a memory bound)", repl_limits::FORM_COMPONENTS);
        value_err(&mut ctx, "exterior_derivative(w, CN)", &lim);
        value_err(&mut ctx, "wedge(w, w, CN)", &lim);
        // Below the limit the callee answers as before.
        ctx.eval("S = Manifold(3, \"S\")").unwrap();
        ctx.eval("CS = chart(S, \"t\")").unwrap();
        ctx.eval("fs = diff_form(S, 1)").unwrap();
        no_comps(&mut ctx, "exterior_derivative(fs, CS)");
        // The lookup clones the components, so it is skipped for a form of more than
        // FORM_COMPONENTS components (n^degree); the limit then answers.
        let big = Arc::new(DifferentiableManifold::new("B", repl_limits::FORM_COMPONENTS + 1));
        let t = Chart::new("t", 1, vec!["t"]).unwrap();
        assert!(missing_components_error(&[&DiffForm::new(big, 1)], &t).is_none());
        let small = Arc::new(DifferentiableManifold::new("A", 1000));
        assert!(missing_components_error(&[&DiffForm::new(small, 2)], &t).is_some());
    }
}
