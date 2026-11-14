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
use rustmath_finitefields::{
    all_square_roots_mod_prime_power, num_square_roots_mod_prime_power, square_root_mod_prime_power,
    IntegerMod,
};
use rustmath_combinatorics::{
    catalan, fibonacci_signed, lucas_signed, bell_number, stirling_first, stirling_second,
    multinomial, eulerian, narayana, motzkin,
    delannoy, delannoy_central, schroder_large, schroder_small,
    count_derangements,
};
use rustmath_combinatorics::factorials::{falling_factorial_integer, rising_factorial_integer};
use rustmath_combinatorics::partition_number::{number_of_partitions, NUMBER_OF_PARTITIONS_MAX_N};
use rustmath_integers::{
    prime::{
        is_probable_prime, nth_prime, prime_pi, try_is_prime, try_next_prime,
        try_prime_range as try_primes_between,
    },
    crt::chinese_remainder_theorem,
    factorization_with_limits, FactorizationLimits,
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Interrupts the evaluation running in a `ReplContext` from another thread (S13). The
/// evaluation stops at its next checkpoint with a `KeyboardInterrupt` error. A request made
/// while nothing runs applies to the next evaluation, unless `clear` discards it.
#[derive(Clone, Debug, Default)]
pub struct InterruptHandle(Arc<AtomicBool>);

impl InterruptHandle {
    pub fn interrupt(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
    pub fn clear(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
    pub fn is_pending(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

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
    Matrix(Arc<Matrix<Integer>>),
    Polynomial(Arc<UnivariatePolynomial<Integer>>),
    IntegerMod(IntegerMod),
    List(Arc<Vec<RustMathValue>>),
    Graph(Arc<Graph>),
    Point2D(Point2D),
    Point3D(Point3D),
    /// A plot with SVG content and description
    Plot { description: String, svg: Arc<str> },
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
    ChartVal(Arc<Chart>),
    /// Differential form on a manifold
    DiffFormVal(Arc<DiffForm>),
    /// Vector field on a manifold
    VectorFieldVal(Arc<VectorField>),
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
            interactive: true,  // Enable zoom/rotate by default
            background: "#ffffff".to_string(),
            axes: true,
            shading: true,  // Enable realistic shading
        }
    }
}

impl Plot3DOptions {
    /// Parse options from keyword arguments like "color='red', mesh=True, opacity=0.8".
    ///
    /// No command calls this yet (the 3D plots use the defaults and fixed grids). Every
    /// argument must be `key=value` with a known key: `color`/`rgbcolor` (a color name, or 3 or
    /// 6 hex digits after `#` or `0x`), `opacity` (a number in [0, 1]), and the booleans `mesh`,
    /// `interactive`, `axes`, `shading` (`True`/`true`/`1` or `False`/`false`/`0`). Anything
    /// else is a ValueError naming the key (N5): nothing is clamped, defaulted or ignored. The
    /// former `plot_points` key is refused, since no generator read it.
    pub fn parse_from_args(args: &[&str]) -> Result<Self, EvalError> {
        let mut opts = Self::default();
        let boolean = |key: &str, value: &str| -> Result<bool, EvalError> {
            match value {
                "True" | "true" | "1" => Ok(true),
                "False" | "false" | "0" => Ok(false),
                _ => Err(EvalError::new("ValueError", format!(
                    "plot3d option {}: expected True or False, got {}", key, value))),
            }
        };

        for arg in args {
            let arg = arg.trim();
            let Some(eq_pos) = arg.find('=') else {
                return Err(EvalError::new("ValueError", format!(
                    "plot3d option {}: options are given as key=value", arg)));
            };
            let key = arg[..eq_pos].trim().to_lowercase();
            let value = arg[eq_pos + 1..].trim();

            match key.as_str() {
                "color" | "rgbcolor" => {
                    // Remove quotes if present
                    let color = value.trim_matches(|c| c == '\'' || c == '"');
                    opts.color = Self::parse_color(color).ok_or_else(|| EvalError::new("ValueError", format!(
                        "plot3d option {}: unknown color {} (a color name, or 3 or 6 hex digits after # or 0x)",
                        key, color)))?;
                }
                "opacity" => {
                    opts.opacity = match value.parse::<f64>() {
                        Ok(o) if (0.0..=1.0).contains(&o) => o,
                        _ => return Err(EvalError::new("ValueError", format!(
                            "plot3d option opacity: expected a number in [0, 1], got {}", value))),
                    };
                }
                "mesh" => opts.mesh = boolean("mesh", value)?,
                "interactive" => opts.interactive = boolean("interactive", value)?,
                "axes" => opts.axes = boolean("axes", value)?,
                "shading" => opts.shading = boolean("shading", value)?,
                _ => {
                    return Err(EvalError::new("ValueError", format!(
                        "plot3d: unknown option {} (options: color, rgbcolor, opacity, mesh, interactive, axes, shading)",
                        key)));
                }
            }
        }

        Ok(opts)
    }

    /// Parse color names to hex codes; `None` for an unknown name or a hex value that is not 3
    /// or 6 hex digits (review 3 of J1a, m1: it was the default blue).
    fn parse_color(color: &str) -> Option<String> {
        let hex = |digits: &str| -> Option<String> {
            ((digits.len() == 3 || digits.len() == 6) && digits.chars().all(|c| c.is_ascii_hexdigit()))
                .then(|| format!("#{}", digits))
        };
        let named = match color.to_lowercase().as_str() {
            "blue" => "#6666ff",
            "red" => "#ff6666",
            "green" => "#66ff66",
            "yellow" => "#ffff66",
            "orange" => "#ff9933",
            "purple" => "#9966ff",
            "cyan" => "#66ffff",
            "magenta" => "#ff66ff",
            "white" => "#ffffff",
            "black" => "#333333",
            "gray" | "grey" => "#999999",
            _ => {
                return if let Some(d) = color.strip_prefix('#') {
                    hex(d)
                } else if let Some(d) = color.strip_prefix("0x") {
                    hex(d)
                } else {
                    None
                };
            }
        };
        Some(named.to_string())
    }
}

impl RustMathValue {
    /// The output of a value. A value whose text would exceed OUTPUT_BYTES (estimated before
    /// anything is formatted, display_size) is shown as a one-line summary; the value is kept.
    pub fn to_display(&self) -> EvalResult {
        let size = display_size(self, repl_limits::OUTPUT_BYTES);
        if size.bytes > repl_limits::OUTPUT_BYTES || size.library_int_bits > repl_limits::LIBRARY_INT_DISPLAY_BITS {
            return EvalResult::text(display_summary(self, &size));
        }
        self.to_display_unchecked()
    }

    fn to_display_unchecked(&self) -> EvalResult {
        match self {
            RustMathValue::Integer(n) => EvalResult::text(decimal_string(n)),
            RustMathValue::Rational(r) => {
                let (num, den) = (decimal_string(r.numerator()), decimal_string(r.denominator()));
                let text = if r.denominator().is_one() { num.clone() } else { format!("{}/{}", num, den) };
                let latex = format!("$\\frac{{{}}}{{{}}}$", num, den);
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
                // One pass: each entry is converted to decimal once and appended to both the text
                // and the LaTeX form (no table of entry strings is kept).
                let (mut text, mut latex) = (String::from("["), String::from("$\\begin{pmatrix} "));
                for i in 0..m.rows() {
                    if i > 0 { text.push_str(", "); latex.push_str(" \\\\ "); }
                    text.push('[');
                    for j in 0..m.cols() {
                        let cell = m.get(i, j).map(decimal_string).unwrap_or_default();
                        if j > 0 { text.push_str(", "); latex.push_str(" & "); }
                        text.push_str(&cell);
                        latex.push_str(&cell);
                    }
                    text.push(']');
                }
                text.push(']');
                latex.push_str(" \\end{pmatrix}$");
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::Polynomial(p) => {
                // Each coefficient is converted to decimal once, for the text and the LaTeX form.
                let coeffs = p.coefficients();
                let strs: Vec<String> = coeffs.iter().map(|c| if c.is_zero() { String::new() } else { decimal_string(c) }).collect();
                let text = poly_text(coeffs, &strs);
                let mut latex_terms = Vec::new();
                for (i, coeff) in coeffs.iter().enumerate().rev() {
                    if coeff.is_zero() {
                        continue;
                    }
                    let coeff_str = strs[i].clone();
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
                let (v, m) = (decimal_string(a.value()), decimal_string(a.modulus()));
                let text = format!("{} (mod {})", v, m);
                let latex = format!("${} \\pmod{{{}}}$", v, m);
                EvalResult::text(text).with_latex(latex)
            }
            RustMathValue::List(items) => {
                // The whole list was measured by to_display.
                let parts: Vec<String> = items.iter()
                    .map(|v| v.to_display_unchecked().text)
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
                    result.svg = Some(svg.to_string());
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
    /// Nesting depth of `eval_expr` calls in the command being evaluated (bounded by
    /// `repl_limits::EVAL_DEPTH`, so deep nesting is an error instead of a stack overflow).
    eval_depth: usize,
    /// Interrupt requests from other threads (S13)
    interrupt: InterruptHandle,
    /// Optional time limit of one `eval` call, and the deadline of the running one
    time_limit: Option<Duration>,
    deadline: Option<Instant>,
}

impl ReplContext {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            execution_count: 0,
            last_result: None,
            stdout: String::new(),
            stderr: String::new(),
            eval_depth: 0,
            interrupt: InterruptHandle::default(),
            time_limit: None,
            deadline: None,
        }
    }

    /// A handle that interrupts this context's evaluations from another thread.
    pub fn interrupt_handle(&self) -> InterruptHandle {
        self.interrupt.clone()
    }

    /// Sets (or with `None` removes) the time limit of each later `eval` call.
    pub fn set_time_limit(&mut self, limit: Option<Duration>) {
        self.time_limit = limit;
    }

    pub fn time_limit(&self) -> Option<Duration> {
        self.time_limit
    }

    /// A cancellation point: Err while an interrupt is pending or after the deadline. The flag is
    /// sticky: it is not cleared here, so a caller that swallows the error fails again at the next
    /// checkpoint; `eval` clears it when the cell ends.
    fn checkpoint(&self) -> Result<(), EvalError> {
        if self.interrupt.is_pending() {
            return Err(EvalError::new("KeyboardInterrupt", "interrupted"));
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                let secs = self.time_limit.map(|d| d.as_secs_f64()).unwrap_or(0.0);
                return Err(EvalError::new("TimeoutError", format!(
                    "the evaluation reached its time limit of {} s (set with time_limit)", secs)));
            }
        }
        Ok(())
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
        self.deadline = self.time_limit.map(|d| Instant::now() + d);
        let result = self.eval_cell(code);
        self.deadline = None;
        // The request has stopped this evaluation, or came after its last checkpoint.
        self.interrupt.clear();
        result
    }

    fn eval_cell(&mut self, code: &str) -> Result<EvalResult, EvalError> {
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

            self.checkpoint()?;
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
                        self.checkpoint()?;

                        // Unpack the list into individual variables
                        match &value {
                            RustMathValue::List(items) => {
                                if items.len() != var_names.len() {
                                    return Err(EvalError::new("ValueError",
                                        format!("Cannot unpack {} values into {} variables",
                                            items.len(), var_names.len())));
                                }
                                // The same order as a single assignment: display, then a
                                // checkpoint, then the stores (an interrupt before the stores
                                // discards the line). Each clone is O(1) for the Arc variants (F2).
                                let display_text = items.iter()
                                    .map(|v| v.to_display().text)
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                self.checkpoint()?;
                                for (name, val) in var_names.iter().zip(items.iter()) {
                                    self.variables.insert(name.to_string(), val.clone());
                                }
                                self.last_result = Some(value.clone());
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
                    // F2: the value is computed once; the variable and `_` share it (one O(1)
                    // clone for the Arc variants). S13: a checkpoint before the display and one
                    // after it, so an interrupt that arrives before the store discards the line.
                    let value = self.eval_expr(expr)?;
                    self.checkpoint()?;
                    let display = value.to_display();
                    self.checkpoint()?;
                    self.variables.insert(var_name, value.clone());
                    self.last_result = Some(value);
                    return Ok(display);
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

        // Evaluate as expression: the same order as an assignment (display, checkpoint, store).
        let value = self.eval_expr(line)?;
        self.checkpoint()?;
        let display = value.to_display();
        self.checkpoint()?;
        self.last_result = Some(value);
        Ok(display)
    }

    /// Evaluates one expression. The nesting of these calls is bounded by
    /// `repl_limits::EVAL_DEPTH` (review 3 of J1a, m4): grouping parentheses and unary signs
    /// (S9) made inputs such as `-(-(...(1)...))` evaluable, and each level recurses, so without a
    /// bound a few thousand levels overflow the stack, which kills the kernel (no `catch_unwind`
    /// can catch it). Runs of signs and of parentheses are read in one step and do not count.
    fn eval_expr(&mut self, expr: &str) -> Result<RustMathValue, EvalError> {
        self.checkpoint()?;
        if self.eval_depth >= repl_limits::EVAL_DEPTH {
            return Err(EvalError::new("SyntaxError", format!(
                "the expression is nested more than {} levels deep (REPL limit; deeper nesting \
                 would overflow the stack)", repl_limits::EVAL_DEPTH)));
        }
        self.eval_depth += 1;
        let result = self.eval_expr_nested(expr);
        self.eval_depth -= 1;
        result
    }

    fn eval_expr_nested(&mut self, expr: &str) -> Result<RustMathValue, EvalError> {
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
                return Ok(RustMathValue::List(elements?.into()));
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
                if !f.is_finite() {
                    return Err(EvalError::new("ValueError", format!("the float literal {} is outside the range of a double", expr)));
                }
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
                    let mut lex = LexState::default();
                    let mut valid = true;
                    for (i, ch) in args_part.char_indices() {
                        lex.top_level(ch);
                        // The '(' after the name closes before the end: not a single call.
                        if ch == ')' && lex.quote.is_none() && lex.depth == 0 && i != args_part.len() - 1 {
                            valid = false;
                            break;
                        }
                    }
                    if valid && lex.depth == 0 && lex.quote.is_none() {
                        return self.eval_function_call(expr);
                    }
                }
            }
        }

        // String literal
        if is_string_literal(expr) {
            return Ok(RustMathValue::String(expr[1..expr.len() - 1].to_string()));
        }

        // List literal [a, b, c, ...]
        if expr.starts_with('[') && expr.ends_with(']') {
            let inner = &expr[1..expr.len() - 1];
            let elements = self.parse_list_elements(inner)?;
            return Ok(RustMathValue::List(elements.into()));
        }

        // Grouping parentheses (S9): "(e)" is e when the '(' closes at the last character and no
        // top-level comma is inside (a tuple such as "(x, 0, 1)" stays a command argument).
        if let Some(inner) = group_inner(expr) {
            return self.eval_expr(inner);
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

            // time_limit(s): stop later evaluations after s seconds (0: no limit); time_limit()
            // shows it. A cooperative limit (S13), like MAGMA's Alarm (ch. 3 §3.12) but the
            // kernel keeps running.
            "time_limit" => {
                let args = self.parse_args(args_str)?;
                match args.as_slice() {
                    [] => {}
                    [RustMathValue::Integer(s)] if s.signum() >= 0 => {
                        let secs = int_arg::<u64>("time_limit", "seconds", s, 0..=TIME_LIMIT_MAX_SECS, "at most one year")?;
                        self.time_limit = if secs == 0 { None } else { Some(Duration::from_secs(secs)) };
                    }
                    [RustMathValue::Float(f)] if f.is_finite() && *f >= 0.0 && *f <= TIME_LIMIT_MAX_SECS as f64 => {
                        self.time_limit = if *f == 0.0 { None } else { Some(Duration::from_secs_f64(*f)) };
                    }
                    _ => return Err(EvalError::new("ValueError", format!(
                        "time_limit: give a number of seconds in 0..={} (0: no limit)", TIME_LIMIT_MAX_SECS))),
                }
                Ok(RustMathValue::String(match self.time_limit {
                    Some(d) => format!("time limit: {} s for each later cell", d.as_secs_f64()),
                    None => "no time limit".to_string(),
                }))
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
                        ].into()))
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
                        check_rational_gcd("Rational", n.bit_length(), d.bit_length())?;
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
                    Ok(RustMathValue::List(symbols.into()))
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
                    (RustMathValue::Integer(base), RustMathValue::Integer(exp)) => int_power_value("pow", base, exp),
                    (RustMathValue::Rational(base), RustMathValue::Integer(exp)) => rational_power_value("pow", base, exp),
                    _ => Err(EvalError::new("TypeError", "pow requires an integer or rational base and an integer exponent")),
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
                Ok(RustMathValue::Matrix(m.into()))
            }

            // Identity matrix: identity(n)
            "identity" => {
                let arg = self.eval_expr(args_str)?;
                match arg {
                    RustMathValue::Integer(n) => {
                        let size = int_arg::<usize>("identity", "n", &n, 0..=repl_limits::MATRIX_DIM, "REPL limit")?;
                        Ok(RustMathValue::Matrix(Matrix::identity(size).into()))
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
                        Ok(RustMathValue::Matrix(Matrix::zeros(rows, cols).into()))
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
                        Ok(RustMathValue::Matrix(m.transpose().into()))
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
                        ].into()))
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
                                            row.push(rational_text(val));
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
                                    row.push(rational_text(val));
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
                        Ok(RustMathValue::Matrix(m.into()))
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
                        Ok(RustMathValue::Matrix(m.into()))
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
                        let result = self.checked_matrix_power(m, n)?;
                        Ok(RustMathValue::Matrix(result.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        check_scalar_product("scalar_mul", "a matrix", m.rows() * m.cols(), max_entry_bits(m), k)?;
                        let result = m.scalar_mul(k);
                        Ok(RustMathValue::Matrix(result.into()))
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
                Ok(RustMathValue::Polynomial(p.into()))
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
                        Ok(RustMathValue::Polynomial(p.derivative().into()))
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
                        Ok(RustMathValue::List(coeffs.into()))
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
                        Ok(RustMathValue::List(result.into()))
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
                        Ok(RustMathValue::Polynomial(p.gcd(q).into()))
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
                        Ok(RustMathValue::Polynomial(p.lcm(q).into()))
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
                        Ok(RustMathValue::Polynomial(p.compose(q).into()))
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
                        let (quo, rem) = p.div_rem(q).map_err(|e| match e {
                            rustmath_core::MathError::DivisionByZero =>
                                EvalError::new("ZeroDivisionError", "divmod: division by the zero polynomial"),
                            other => EvalError::new("MathError", format!("divmod: {}", other)),
                        })?;
                        Ok(RustMathValue::List(vec![
                            RustMathValue::Polynomial(quo.into()),
                            RustMathValue::Polynomial(rem.into()),
                        ].into()))
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
                    Some(val) if val.is_finite() => Ok(RustMathValue::String(format!("{:.10}", val))),
                    // S11 (review 3, m2): a value beyond the double range is refused, as in
                    // `exact_to_f64`, instead of the text "inf" or "NaN".
                    Some(val) => Err(EvalError::new("ValueError", format!(
                        "{}: the value is outside the range of a double (the double computation gives {})",
                        func_name, val))),
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
                        Ok(RustMathValue::List(list.into()))
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
                        // C1's band of rolling rows: time and memory bounded by
                        // `check_stirling`; the trivial cases are answered at once.
                        let n_u32 = int_arg::<u32>("stirling1", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("stirling1", "k", k, 0..=u32::MAX, "")?;
                        check_stirling("stirling1", StirlingKind::Stirling1, n_u32, k_u32)?;
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
                        // C1's band of rolling rows: time and memory bounded by
                        // `check_stirling`; the trivial cases are answered at once.
                        let n_u32 = int_arg::<u32>("stirling2", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("stirling2", "k", k, 0..=u32::MAX, "")?;
                        // {n 1} = 1 for n >= 1 (one block); the callee would run n rows.
                        if k_u32 == 1 && n_u32 >= 1 {
                            return Ok(RustMathValue::Integer(Integer::one()));
                        }
                        check_stirling("stirling2", StirlingKind::Stirling2, n_u32, k_u32)?;
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
                        for k in ks.iter() {
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
                        // C1 multiplies one binomial per k_i (Goetgheluck) in a product tree. All their
                        // numerator factors together are at most the n factors of n!, so the time
                        // model of the n! product tree bounds the call (J1b ladder `multinomial`:
                        // multinomial(3000000, [1500000, 1500000]) 0.23 s against 9.6 s estimated,
                        // multinomial(100000, [1, ..., 1]) 0.09 s against 0.05 s plus the parse).
                        check_factorial_product("multinomial", u64::from(n_u32), u64::from(u32::BITS - n_u32.leading_zeros()))?;
                        Ok(RustMathValue::Integer(multinomial(n_u32, &ks_u32)))
                    }
                    _ => Err(EvalError::new("TypeError", "multinomial requires (integer, list) arguments")),
                }
            }

            // Falling factorial x(x-1)...(x-k+1) for any integer x (Sage falling_factorial;
            // MAGMA ch. 144 NumberOfPermutations(n, k) for x = n >= 0), through C1's
            // falling_factorial_integer (one balanced product of k factors).
            "falling_factorial" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", "falling_factorial requires 2 arguments (n, k)"));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(x), RustMathValue::Integer(k)) => {
                        let k = int_arg::<u64>("falling_factorial", "k", k, 0..=u64::MAX, "")?;
                        if k == 0 {
                            return Ok(RustMathValue::Integer(Integer::one()));
                        }
                        // For x >= 0 and k > x the factor x - x = 0 occurs: answered at once.
                        if x.signum() >= 0 && Integer::from(k) > *x {
                            return Ok(RustMathValue::Integer(Integer::zero()));
                        }
                        // The largest factor: x for x >= 0; |x| + k - 1 for x < 0.
                        let top_bits = if x.signum() >= 0 {
                            x.bit_length()
                        } else {
                            (-x.clone() + Integer::from(k - 1)).bit_length()
                        };
                        check_factorial_product("falling_factorial", k, top_bits)?;
                        Ok(RustMathValue::Integer(falling_factorial_integer(x, k)))
                    }
                    _ => Err(EvalError::new("TypeError", "falling_factorial requires integer arguments")),
                }
            }

            // Rising factorial x(x+1)...(x+k-1) for any integer x. pochhammer(x, k) is the same
            // function: the Pochhammer symbol (a)_n = a(a+1)...(a+n-1) of MAGMA ch. 49
            // (`HypergeometricSeries`, Chapter_49_Power_Laurent_and_Puiseux_Series.md:292), Sage
            // `rising_factorial` and rustmath-calculus `pochhammer.rs` (S10: it was an alias of
            // falling_factorial). Through C1's rising_factorial_integer.
            "rising_factorial" | "pochhammer" => {
                let args = self.parse_args(args_str)?;
                if args.len() != 2 {
                    return Err(EvalError::new("ArgumentError", format!("{} requires 2 arguments (n, k)", func_name)));
                }
                match (&args[0], &args[1]) {
                    (RustMathValue::Integer(x), RustMathValue::Integer(k)) => {
                        let k = int_arg::<u64>(func_name, "k", k, 0..=u64::MAX, "")?;
                        if k == 0 {
                            return Ok(RustMathValue::Integer(Integer::one()));
                        }
                        // For x <= 0 and k > -x the factor x + (-x) = 0 occurs: answered at once.
                        if x.signum() <= 0 && Integer::from(k) > -x.clone() {
                            return Ok(RustMathValue::Integer(Integer::zero()));
                        }
                        // The largest factor: x + k - 1 for x > 0; |x| for x <= 0.
                        let top_bits = if x.signum() > 0 {
                            (x.clone() + Integer::from(k - 1)).bit_length()
                        } else {
                            x.bit_length()
                        };
                        check_factorial_product(func_name, k, top_bits)?;
                        Ok(RustMathValue::Integer(rising_factorial_integer(x, k)))
                    }
                    _ => Err(EvalError::new("TypeError", format!("{} requires integer arguments", func_name))),
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
                        // C1's band of rolling rows: time and memory bounded by
                        // `check_stirling`; the trivial cases are answered at once.
                        let n_u32 = int_arg::<u32>("eulerian", "n", n, 0..=u32::MAX, "")?;
                        let k_u32 = int_arg::<u32>("eulerian", "k", k, 0..=u32::MAX, "")?;
                        check_stirling("eulerian", StirlingKind::Eulerian, n_u32, k_u32)?;
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
                        // MAGMA NumberOfPartitions(n) (ch. 18, ch. 145): C1's
                        // number_of_partitions (Euler's recurrence for small n, Hardy-Ramanujan-
                        // Rademacher with a rigorous enclosure above; exact for every
                        // n <= NUMBER_OF_PARTITIONS_MAX_N, which PARTITIONS_N is below).
                        let n = int_arg::<u64>("partitions", "n", n, 0..=repl_limits::PARTITIONS_N, "REPL limit, about 10 s")?;
                        Ok(RustMathValue::Integer(number_of_partitions(n)))
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
                        Ok(RustMathValue::List(list.into()))
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
                        // S15: the sieve is linear in x (about 1.75 s per 10^9 in release).
                        if *n > Integer::from(repl_limits::PRIME_PI_X) {
                            return Err(EvalError::new("ValueError", format!(
                                "prime_pi: x = {} is above the REPL limit {} (the sieve takes about 10 s there)",
                                show_int(n), repl_limits::PRIME_PI_X)));
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
                        let primes = checked_primes(a, b)?;
                        let list: Vec<RustMathValue> = primes.into_iter()
                            .map(RustMathValue::Integer)
                            .collect();
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                // The bounds as doubles, each rounded once (S11): a bound outside the double
                // range is a ValueError, never inf or NaN.
                let a = exact_to_f64(&args[2], "integrate_num: lower bound")
                    .unwrap_or_else(|| Err(EvalError::new("TypeError", "lower bound must be a number")))?;
                let b = exact_to_f64(&args[3], "integrate_num: upper bound")
                    .unwrap_or_else(|| Err(EvalError::new("TypeError", "upper bound must be a number")))?;

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
                let bounds: Vec<&str> = self.split_at_depth_zero(&bounds_start, ',').into_iter().map(|s| s.trim()).collect();
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
                // S15: every constructor is checked against the memory model first.
                check_graph_size(func_name, n as u128, 0, &[])?;
                Ok(RustMathValue::Graph(Graph::new(n).into()))
            }

            // Complete graph K_n
            "complete_graph" | "CompleteGraph" | "K" => {
                let n = self.eval_as_usize(args_str)?;
                let v = n as u128;
                let degrees: &[(u128, u128)] = if n == 0 { &[] } else { &[(v, v - 1)] };
                check_graph_size(func_name, v, v * v.saturating_sub(1) / 2, degrees)?;
                Ok(RustMathValue::Graph(complete_graph(n).into()))
            }

            // Cycle graph C_n
            "cycle_graph" | "CycleGraph" | "C" => {
                let n = self.eval_as_usize(args_str)?;
                if n < 3 {
                    return Err(EvalError::new("ValueError", "Cycle graph requires at least 3 vertices"));
                }
                check_graph_size(func_name, n as u128, n as u128, &[(n as u128, 2)])?;
                Ok(RustMathValue::Graph(cycle_graph(n).into()))
            }

            // Path graph P_n
            "path_graph" | "PathGraph" | "P" => {
                let n = self.eval_as_usize(args_str)?;
                let v = n as u128;
                let degrees: &[(u128, u128)] = if n < 2 { &[] } else { &[(2, 1), (v - 2, 2)] };
                check_graph_size(func_name, v, v.saturating_sub(1), degrees)?;
                Ok(RustMathValue::Graph(path_graph(n).into()))
            }

            // Star graph S_n (one center connected to n outer vertices)
            "star_graph" | "StarGraph" => {
                let n = self.eval_as_usize(args_str)?;
                let v = n as u128;
                let degrees: &[(u128, u128)] = if n == 0 { &[] } else { &[(v, 1), (1, v)] };
                check_graph_size(func_name, v + 1, v, degrees)?;
                Ok(RustMathValue::Graph(star_graph(n).into()))
            }

            // Wheel graph W_n
            "wheel_graph" | "WheelGraph" => {
                let n = self.eval_as_usize(args_str)?;
                if n < 3 {
                    return Err(EvalError::new("ValueError", "Wheel graph requires at least 3 outer vertices"));
                }
                let v = n as u128;
                check_graph_size(func_name, v + 1, 2 * v, &[(v, 3), (1, v)])?;
                Ok(RustMathValue::Graph(wheel_graph(n).into()))
            }

            // Petersen graph
            "petersen_graph" | "PetersenGraph" | "petersen" => {
                Ok(RustMathValue::Graph(petersen_graph().into()))
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
                        Ok(RustMathValue::List(values.into()))
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
                        Ok(RustMathValue::List(values.into()))
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
                        Ok(RustMathValue::List(values.into()))
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
                        Ok(RustMathValue::List(values.into()))
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
                    RustMathValue::Graph(g) => Graph::clone(g),
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
                    Ok(()) => Ok(RustMathValue::Graph(Arc::new(g))),
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
                        Ok(RustMathValue::List(hull_values.into()))
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
                                Ok(RustMathValue::List(vertices.into()))
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
                    Err(e) if looks_like_function_plot || is_cancellation(&e) => {
                        // If it looks like a function plot but failed, return the error (an
                        // interrupt or the time limit always, S13)
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
                            return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                            return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                        return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                    self.checkpoint()?; // S13: one cancellation point per sample
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
                return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                            RustMathValue::Integer(n) => Ok(decimal_string(n)),
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
                return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                        return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                        return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
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
                        Ok(RustMathValue::List(list.into()))
                    }
                    RustMathValue::ExceptionalJordan(ref j) => {
                        let basis = j.basis();
                        let list: Vec<RustMathValue> = basis.into_iter()
                            .map(RustMathValue::AlbertElem)
                            .collect();
                        Ok(RustMathValue::List(list.into()))
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
                        let names = chart_coordinate_names(coords, m.dimension())?;
                        let chart = Chart::new("user_chart", m.dimension(), names)
                            .map_err(|e| EvalError::new("ValueError", format!("chart: {}", e)))?;
                        Ok(RustMathValue::ChartVal(chart.into()))
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
                        Ok(RustMathValue::DiffFormVal(form.into()))
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
                        check_chart("coordinate_form", chart, m.dimension())?;
                        let i = index_arg("coordinate_form", "index", idx, m.dimension(), "coordinates")?;
                        let form = DiffForm::coordinate_form(m.clone(), &chart, i)
                            .map_err(|e| EvalError::new("ValueError", format!("{}", e)))?;
                        Ok(RustMathValue::DiffFormVal(form.into()))
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
                        check_chart("exterior_derivative", chart, form.manifold().dimension())?;
                        let (n, p) = (form.manifold().dimension(), form.degree());
                        let degree = p.checked_add(1).ok_or_else(|| EvalError::new("ValueError", format!(
                            "exterior_derivative: the degree p + 1 of d(omega) for p = {} does not fit in a machine word", p)))?;
                        // For p < n the callee builds all n^(p+1) components
                        // (rustmath-manifolds `DiffForm::exterior_derivative`).
                        if p < n {
                            if let Err(limit) = check_form_components("exterior_derivative", n, degree) {
                                return Err(missing_components_error(&[form], chart).unwrap_or(limit));
                            }
                        }
                        let d_form = form.exterior_derivative(chart)
                            .map_err(|e| EvalError::new("ComputationError", format!("{}", e)))?;
                        Ok(RustMathValue::DiffFormVal(d_form.into()))
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
                                check_chart("wedge", chart, f1.manifold().dimension())?;
                                // Forms on different manifolds are an error before p + q is formed.
                                if Arc::ptr_eq(f1.manifold(), f2.manifold()) {
                                    let (n, p, q) = (f1.manifold().dimension(), f1.degree(), f2.degree());
                                    let degree = p.checked_add(q).ok_or_else(|| EvalError::new("ValueError", format!(
                                        "wedge: the degree p + q of the product for p = {}, q = {} does not fit in a machine word", p, q)))?;
                                    // For p + q <= n the callee builds all n^(p+q) products
                                    // (rustmath-manifolds `DiffForm::wedge` through `TensorField::tensor_product`).
                                    if degree <= n {
                                        if let Err(limit) = check_form_components("wedge", n, degree) {
                                            return Err(missing_components_error(&[f1, f2], chart).unwrap_or(limit));
                                        }
                                    }
                                }
                                let result = f1.wedge(f2, chart)
                                    .map_err(|e| EvalError::new("ComputationError", format!("{}", e)))?;
                                Ok(RustMathValue::DiffFormVal(result.into()))
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
        let mut current = String::new();
        let mut lex = LexState::default();
        for ch in args_str.chars() {
            let top = lex.top_level(ch);
            match ch {
                ',' if top => {
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
        let mut current_row = String::new();

        // Rows are the bracketed parts at depth 0; brackets inside a string literal do not
        // count (S9, `LexState`). Commas between rows are skipped.
        let mut lex = LexState::default();
        for ch in inner.chars() {
            let outside = lex.quote.is_none();
            let before = lex.depth;
            lex.top_level(ch);
            if outside && ch == '[' && before == 0 {
                current_row.clear();
            } else if outside && ch == ']' && before == 1 {
                let row = self.parse_list_elements(&current_row)?;
                result.push(row);
                current_row.clear();
            } else if before > 0 {
                current_row.push(ch);
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
        let mut current = String::new();
        let mut lex = LexState::default();
        for ch in s.chars() {
            let top = lex.top_level(ch);
            match ch {
                ',' if top => {
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
        for part in self.split_at_depth_zero(inner, ',') {
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
        let mut current = String::new();
        let mut lex = LexState::default();
        for ch in args_str.chars() {
            let top = lex.top_level(ch);
            match ch {
                ',' if top => {
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
            // Try to evaluate it as an expression first. An interrupt or the time limit (S13)
            // is passed on, so the line stops before the library call of the caller.
            match self.eval_expr(s) {
                Ok(RustMathValue::Expr(e)) => return Ok(e),
                Ok(RustMathValue::Symbol(sym)) => return Ok(Expr::Symbol(sym)),
                Ok(RustMathValue::Integer(n)) => return Ok(Expr::Integer(n)),
                Ok(_) => {}
                Err(e) if is_cancellation(&e) => return Err(e),
                Err(_) => {}
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
        let mut current = String::new();
        let mut lex = LexState::default();
        for ch in args_str.chars() {
            let top = lex.top_level(ch);
            match ch {
                ',' if top => {
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
        let mut start = 0;
        for (i, ch) in top_level_chars(s) {
            if ch == delimiter {
                result.push(&s[start..i]);
                start = i + ch.len_utf8();
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
            self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
    }

    /// A real number written as text: a plot range, a find_root bound, a point coordinate (S9,
    /// S11; revision 2). The text is evaluated exactly first (`eval_expr`): an Integer or a
    /// Rational is rounded once to the nearest double (`exact_to_f64`), a Float is taken as it is.
    /// Only a text whose value is not an exact number (it uses pi, e, tau, a decimal literal or a
    /// Float variable) goes to `float_arith`, the real field of doubles. The result is a finite
    /// double or an error.
    fn eval_to_f64(&mut self, s: &str) -> Result<f64, EvalError> {
        let s = s.trim();
        let exact = self.eval_expr(s);
        if let Ok(v) = &exact {
            if let Some(r) = exact_to_f64(v, s) {
                return r;
            }
        }
        match self.float_arith(s, 0)? {
            Some(RealArith::Real(f)) => Ok(f),
            Some(RealArith::Exact(v)) => exact_to_f64(&v, s)
                .unwrap_or_else(|| Err(EvalError::new("TypeError", "Cannot convert to number"))),
            None => match exact {
                Err(e) => Err(e),
                Ok(_) => Err(EvalError::new("TypeError", "Cannot convert to number")),
            },
        }
    }

    /// The value of an arithmetic text in the real field of doubles (S9 revision 2), by MAGMA's
    /// coercion rule (ch. 25 §25.1.2): integers and rationals combine exactly (`apply_binary_op`,
    /// the REPL's own arithmetic), and an operation with a real operand rounds the exact operand to
    /// the nearest double and is done in IEEE double arithmetic, the real field of 53 bits. Real
    /// operands are the constants pi/PI, e/E, tau/TAU (a variable of the same name wins), decimal
    /// literals and Float variables. So `10^400/10^399 * pi` is 10 pi, while `10^20 + pi - 10^20` is
    /// 0, as in MAGMA at this precision: cancellation after a real operand is part of the semantics.
    /// An exact power with a non-integral rational exponent is real (`2^(1/2)`, as in MAGMA).
    /// Precedence, associativity and unary signs as in `try_eval_binary_op`.
    /// Ok(None): `s` uses something else (a call, another variable, a string). Err: the value
    /// overflows or underflows a double (MPFR's exponent range is far wider, so this is a limit of
    /// the double, not an answer), a division by zero, or an error of the exact arithmetic.
    /// `depth` is the nesting depth of this call, bounded by `repl_limits::EVAL_DEPTH` as in
    /// `eval_expr`.
    fn float_arith(&self, s: &str, depth: usize) -> Result<Option<RealArith>, EvalError> {
        if depth >= repl_limits::EVAL_DEPTH {
            return Err(EvalError::new("SyntaxError", format!(
                "the expression is nested more than {} levels deep (REPL limit; deeper nesting \
                 would overflow the stack)", repl_limits::EVAL_DEPTH)));
        }
        let s = s.trim();
        let top = top_level_chars(s);
        // 1. and 2. The + - chain, else the * / chain, folded from the left.
        for level in [ChainLevel::Sum, ChainLevel::Product] {
            if let Some((operands, ops)) = binary_chain(s, &top, level) {
                let Some(mut acc) = self.float_arith(operands[0], depth + 1)? else { return Ok(None) };
                for (op, operand) in ops.iter().zip(&operands[1..]) {
                    let Some(r) = self.float_arith(operand, depth + 1)? else { return Ok(None) };
                    acc = self.real_op(acc, op, r, s)?;
                }
                return Ok(Some(acc));
            }
        }
        // 3. A run of leading signs, read in one step.
        let mut rest = s;
        let mut negative = false;
        while let Some(sign) = rest.chars().next().filter(|c| *c == '-' || *c == '+') {
            negative ^= sign == '-';
            rest = rest[1..].trim_start();
        }
        if rest.len() < s.len() {
            if rest.is_empty() {
                return Ok(None);
            }
            let Some(v) = self.float_arith(rest, depth + 1)? else { return Ok(None) };
            return Ok(Some(match (negative, v) {
                (false, v) => v,
                (true, RealArith::Real(f)) => RealArith::Real(-f),
                (true, RealArith::Exact(x)) => RealArith::Exact(self.negate(&x)?),
            }));
        }
        // 4. ** and ^, the leftmost (right associative).
        for (i, &(pos, ch)) in top.iter().enumerate() {
            let len = if ch == '^' {
                1
            } else if ch == '*' && top.get(i + 1) == Some(&(pos + 1, '*')) {
                2
            } else {
                continue;
            };
            let (l, r) = (s[..pos].trim(), s[pos + len..].trim());
            if l.is_empty() || r.is_empty() {
                return Ok(None);
            }
            let Some(lv) = self.float_arith(l, depth + 1)? else { return Ok(None) };
            let Some(rv) = self.float_arith(r, depth + 1)? else { return Ok(None) };
            return self.real_op(lv, "^", rv, s).map(Some);
        }
        // 5. Grouping parentheses.
        if let Some(inner) = group_inner(s) {
            return self.float_arith(inner, depth + 1);
        }
        // 6. Atoms.
        if let Some(v) = self.variables.get(s) {
            return Ok(match v {
                RustMathValue::Integer(_) | RustMathValue::Rational(_) => Some(RealArith::Exact(v.clone())),
                RustMathValue::Float(f) if f.is_finite() => Some(RealArith::Real(*f)),
                _ => None,
            });
        }
        match s {
            "pi" | "PI" => return Ok(Some(RealArith::Real(std::f64::consts::PI))),
            "e" | "E" => return Ok(Some(RealArith::Real(std::f64::consts::E))),
            "tau" | "TAU" => return Ok(Some(RealArith::Real(std::f64::consts::TAU))),
            _ => {}
        }
        if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(BigInt::from_str(s).ok().map(|n| RealArith::Exact(RustMathValue::Integer(Integer::new(n)))));
        }
        let decimal = s.starts_with(|c: char| c.is_ascii_digit() || c == '.')
            && s.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-'));
        if !decimal {
            return Ok(None);
        }
        let Ok(f) = s.parse::<f64>() else { return Ok(None) };
        let nonzero = s.split(['e', 'E']).next().unwrap_or("").bytes().any(|b| (b'1'..=b'9').contains(&b));
        if !f.is_finite() {
            return Err(EvalError::new("ValueError", format!("the float literal {} is outside the range of a double", s)));
        }
        if nonzero && (f == 0.0 || f.is_subnormal()) {
            return Err(EvalError::new("ValueError", format!(
                "{}: the float literal is below the normal range of a double (underflow)", s)));
        }
        Ok(Some(RealArith::Real(f)))
    }

    /// One operation of `float_arith`: exact when both operands are exact (a power with a
    /// non-integral rational exponent excepted), else in doubles. `whole` names the text in errors.
    fn real_op(&self, l: RealArith, op: &str, r: RealArith, whole: &str) -> Result<RealArith, EvalError> {
        use RustMathValue as V;
        let pow = op == "^" || op == "**";
        if let (RealArith::Exact(a), RealArith::Exact(b)) = (&l, &r) {
            let b_int = match b {
                V::Rational(q) if q.denominator().is_one() => Some(V::Integer(q.numerator().clone())),
                V::Rational(_) => None,
                other => Some(other.clone()),
            };
            match (pow, b_int) {
                (true, None) => {} // 2^(1/2): a real number, below
                (_, Some(bi)) if pow => return self.apply_binary_op(a, "^", &bi).map(RealArith::Exact),
                _ => return self.apply_binary_op(a, op, b).map(RealArith::Exact),
            }
        }
        let to_real = |v: RealArith| -> Result<f64, EvalError> {
            match v {
                RealArith::Real(f) => Ok(f),
                RealArith::Exact(x) => {
                    let f = exact_to_f64(&x, whole).unwrap_or_else(|| Err(EvalError::new("TypeError", "Cannot convert to number")))?;
                    if !is_exact_zero(&x) && (f == 0.0 || f.is_subnormal()) {
                        return Err(EvalError::new("ValueError", format!(
                            "{}: an exact value below the normal range of a double meets a real operand (underflow)", whole)));
                    }
                    Ok(f)
                }
            }
        };
        let (x, y) = (to_real(l)?, to_real(r)?);
        let z = match op {
            "+" => x + y,
            "-" => x - y,
            "*" => x * y,
            "/" => {
                if y == 0.0 {
                    return Err(EvalError::new("ZeroDivisionError", "division by zero"));
                }
                x / y
            }
            _ => x.powf(y),
        };
        let shown = if pow { "^" } else { op };
        if !z.is_finite() {
            return Err(EvalError::new("ValueError", format!(
                "{} is not a finite real number in double precision ({:e} {} {:e} gives {})", whole, x, shown, y, z)));
        }
        // + and - of doubles are exact when the result is 0 or subnormal (gradual underflow), so only
        // products, quotients and powers of nonzero operands can underflow (0 * y and 0^y are exact).
        if ((op == "*" && y != 0.0) || op == "/" || pow) && x != 0.0 && (z == 0.0 || z.is_subnormal()) {
            return Err(EvalError::new("ValueError", format!(
                "{} underflows double precision ({:e} {} {:e} gives {:e})", whole, x, shown, y, z)));
        }
        Ok(RealArith::Real(z))
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

    /// The integer options of a plot command (N5): `parts` are the arguments after the positional
    /// ones. Each is `name=value` with `name` one of `options` (exact match), and `value`
    /// evaluates to an integer inside that option's range; `bare` names the option a bare
    /// integer argument sets (the old positional n_points form). Anything else is an error:
    /// nothing is clamped, defaulted or ignored. Returns (option, value) pairs; an option given
    /// twice is an error.
    fn plot_int_options<S: AsRef<str>>(&mut self, cmd: &str, parts: &[S],
        options: &[(&str, RangeInclusive<usize>)], bare: Option<&str>)
        -> Result<HashMap<String, usize>, EvalError>
    {
        let mut out: HashMap<String, usize> = HashMap::new();
        for part in parts {
            let part = part.as_ref().trim();
            let (name, value) = match top_level_chars(part).into_iter().find(|&(_, c)| c == '=') {
                Some((i, _)) => (part[..i].trim(), part[i + 1..].trim()),
                None => match bare {
                    Some(b) => (b, part),
                    None => return Err(EvalError::new("ArgumentError", format!(
                        "{}: unexpected argument {}; options are given as name=value", cmd, part))),
                },
            };
            let range = match options.iter().find(|(n, _)| *n == name) {
                Some((_, r)) => r.clone(),
                None => {
                    let names: Vec<&str> = options.iter().map(|(n, _)| *n).collect();
                    return Err(EvalError::new("ValueError", format!(
                        "{}: unknown option {} (options: {})", cmd, name, names.join(", "))));
                }
            };
            let v = match self.eval_expr(value)? {
                RustMathValue::Integer(n) => int_arg::<usize>(cmd, name, &n, range, "")?,
                RustMathValue::Rational(r) if r.denominator().is_one() => {
                    int_arg::<usize>(cmd, name, r.numerator(), range, "")?
                }
                other => return Err(EvalError::new("ValueError", format!(
                    "{}: {} must be an integer in {}..={}, got {}", cmd, name, range.start(), range.end(),
                    other.to_display().text))),
            };
            if out.insert(name.to_string(), v).is_some() {
                return Err(EvalError::new("ValueError", format!("{}: option {} is given twice", cmd, name)));
            }
        }
        Ok(out)
    }

    /// Evaluate slope field plot
    /// Syntax: plot_slope_field(f, (x, xmin, xmax), (y, ymin, ymax))
    ///         plot_slope_field(f, (x, xmin, xmax), (y, ymin, ymax), plot_points=25)
    fn eval_slope_field(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        // Parse arguments - looking for: expr, (var1, min1, max1), (var2, min2, max2), optional_kwargs
        let args_str = args_str.trim();

        // Split by comma at depth 0
        let mut parts: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut lex = LexState::default();
        for ch in args_str.chars() {
            let top = lex.top_level(ch);
            match ch {
                ',' if top => {
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

        // Optional plot_points (N5: checked, not clamped).
        let opts = self.plot_int_options("plot_slope_field", &parts[3..],
            &[("plot_points", 2..=repl_limits::SLOPE_FIELD_POINTS)], Some("plot_points"))?;
        let n_points = opts.get("plot_points").copied().unwrap_or(20);

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
                self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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

        // Optional plot_points (N5: checked, not clamped).
        let opts = self.plot_int_options("parametric_plot", &parts[2..],
            &[("plot_points", 2..=repl_limits::PARAMETRIC_POINTS)], Some("plot_points"))?;
        let n_points = opts.get("plot_points").copied().unwrap_or(200);

        let t_sym = Symbol::new(t_var_name);
        let x_symbols = x_expr.symbols();
        let y_symbols = y_expr.symbols();

        // Generate curve points
        let mut coords: Vec<(f64, f64)> = Vec::new();
        let t_step = (t_max - t_min) / (n_points - 1) as f64;

        for i in 0..n_points {
            self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
        let opts = self.plot_int_options("contour_plot", &parts[3..], &[
            ("contours", 1..=repl_limits::CONTOURS),
            ("n_contours", 1..=repl_limits::CONTOURS),
            ("plot_points", 2..=repl_limits::CONTOUR_POINTS),
        ], None)?;
        if opts.contains_key("contours") && opts.contains_key("n_contours") {
            return Err(EvalError::new("ValueError", "contour_plot: give contours or n_contours, not both"));
        }
        let n_contours = opts.get("contours").or(opts.get("n_contours")).copied().unwrap_or(10);
        let n_grid = opts.get("plot_points").copied().unwrap_or(50);

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
                self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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

        // Optional plot_points (N5: checked, not clamped).
        let opts = self.plot_int_options("vector_field", &parts[3..],
            &[("plot_points", 2..=repl_limits::VECTOR_FIELD_POINTS)], None)?;
        let n_points = opts.get("plot_points").copied().unwrap_or(15);

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
                self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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

        // Optional plot_points (N5: checked, not clamped).
        let opts = self.plot_int_options("implicit_plot", &parts[3..],
            &[("plot_points", 2..=repl_limits::IMPLICIT_POINTS)], None)?;
        let n_grid = opts.get("plot_points").copied().unwrap_or(100);

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
                self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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

        // Optional plot_points (N5: checked, not clamped).
        let opts = self.plot_int_options("region_plot", &parts[3..],
            &[("plot_points", 2..=repl_limits::IMPLICIT_POINTS)], None)?;
        let n_grid = opts.get("plot_points").copied().unwrap_or(100);

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
                self.checkpoint()?; // S13: one cancellation point per sample
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

        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                return Ok(RustMathValue::Plot { description, svg: svg.into() });
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
                self.checkpoint()?; // S13: one cancellation point per sample
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                self.checkpoint()?; // S13: one cancellation point per sample
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
            Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                    self.checkpoint()?; // S13: one cancellation point per sample
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
            Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                    self.checkpoint()?; // S13: one cancellation point per sample
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
    }

    /// Evaluate line3d
    fn eval_line3d(&mut self, args_str: &str) -> Result<RustMathValue, EvalError> {
        let arg = self.eval_expr(args_str.trim())?;
        match arg {
            RustMathValue::List(pts) => {
                let coords = self.extract_3d_coords(&pts)?;
                let svg = self.generate_3d_curve_svg(&coords, "3D Line");
                let description = format!("3D line with {} vertices", coords.len());
                Ok(RustMathValue::Plot { description, svg: svg.into() })
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                self.checkpoint()?; // S13: one cancellation point per sample
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                self.checkpoint()?; // S13: one cancellation point per sample
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
                self.checkpoint()?; // S13: one cancellation point per sample
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
        Ok(RustMathValue::Plot { description, svg: svg.into() })
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
    /// Every number is rounded once to the nearest double (S11, `exact_to_f64`); a value outside
    /// the double range is a ValueError, so no inf or NaN reaches rustmath-stats.
    fn list_to_floats(&self, values: &[RustMathValue]) -> Result<Vec<f64>, EvalError> {
        values.iter().map(|v| {
            exact_to_f64(v, "list element")
                .unwrap_or_else(|| Err(EvalError::new("TypeError", "List must contain only numbers")))
        }).collect()
    }

    /// Convert a list of RustMathValues to Vec<Complex> for FFT functions
    fn list_to_complexes(&self, values: &[RustMathValue]) -> Result<Vec<Complex>, EvalError> {
        values.iter().map(|v| {
            match v {
                RustMathValue::Complex(c) => Ok(c.clone()),
                // A real number, rounded once (S11).
                _ => exact_to_f64(v, "list element")
                    .unwrap_or_else(|| Err(EvalError::new("TypeError", "List must contain only numbers or complex values")))
                    .map(|f| Complex::new(f, 0.0)),
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
    /// The number rounded once to the nearest double (S11, `exact_to_f64`); a value outside the
    /// double range is a ValueError.
    fn value_to_f64(&self, val: &RustMathValue) -> Result<f64, EvalError> {
        exact_to_f64(val, "argument").unwrap_or_else(|| Err(EvalError::new("TypeError", "expected a number")))
    }

    /// Find the position of "==" operator in a string, respecting parentheses
    fn find_equation_operator(&self, s: &str) -> Option<usize> {
        let top = top_level_chars(s);
        top.windows(2)
            .find(|w| w[0].1 == '=' && w[1] == (w[0].0 + 1, '='))
            .map(|w| w[0].0)
    }

    /// Split "rhs, a, b" into (rhs, "a, b")
    fn split_rhs_and_bounds(&self, s: &str) -> Result<(String, String), EvalError> {
        match top_level_chars(s).into_iter().find(|&(_, c)| c == ',') {
            Some((i, _)) => Ok((s[..i].to_string(), s[i + 1..].to_string())),
            None => Err(EvalError::new("SyntaxError", "could not parse equation bounds")),
        }
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
                    row.push(rational_text(val));
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
        // Split at the operator of lowest precedence, among the top-level characters only
        // (outside brackets and string literals, `LexState`):
        //   1. binary + and -: the rightmost one (left associative);
        //   2. * and /: the rightmost one that is not part of ** (left associative);
        //   3. a leading unary - or +, which applies to the rest: -2**2 = -(2**2) as in Python;
        //   4. ** and ^: the leftmost one (right associative); its right operand may be signed.
        // A candidate whose left or right operand is empty is skipped; the caller then reports
        // the text as unparsable.
        let top = top_level_chars(expr);

        // 1. and 2. A chain of binary + and -, else of * and /: every operator of the level at
        //    once, folded from the left ((a - b) + c, (a * b) / c), so the number of terms adds no
        //    recursion.
        for level in [ChainLevel::Sum, ChainLevel::Product] {
            if let Some((operands, ops)) = binary_chain(expr, &top, level) {
                let mut acc = self.eval_expr(operands[0])?;
                for (op, operand) in ops.iter().zip(&operands[1..]) {
                    let right_val = self.eval_expr(operand)?;
                    acc = self.apply_binary_op(&acc, op, &right_val)?;
                }
                return Ok(Some(acc));
            }
        }

        // 3. A leading sign. Integer and float literals such as -5 never get here (eval_expr
        //    reads them first).
        //    A run of signs is read in one step (--x = x, -+-x = x), so its length adds no
        //    recursion.
        let mut rest = expr.trim_start();
        let mut negative = false;
        let mut signed = false;
        while let Some(sign) = rest.chars().next().filter(|c| *c == '-' || *c == '+') {
            negative ^= sign == '-';
            signed = true;
            rest = rest[1..].trim_start();
        }
        if signed && !rest.is_empty() {
            let v = self.eval_expr(rest)?;
            return Ok(Some(if negative { self.negate(&v)? } else { v }));
        }

        // 4. ** and ^, the leftmost one.
        for (i, &(pos, ch)) in top.iter().enumerate() {
            let op = if ch == '*' && top.get(i + 1).map(|&(p, c)| (p, c)) == Some((pos + 1, '*')) {
                "**"
            } else if ch == '^' {
                "^"
            } else {
                continue;
            };
            let (left, right) = (expr[..pos].trim(), expr[pos + op.len()..].trim());
            if !left.is_empty() && !right.is_empty() {
                let left_val = self.eval_expr(left)?;
                let right_val = self.eval_expr(right)?;
                return Ok(Some(self.apply_binary_op(&left_val, op, &right_val)?));
            }
            break;
        }

        Ok(None)
    }

    /// Unary minus (S9): -v for numbers, symbols, expressions and the module-like values; a
    /// TypeError otherwise.
    fn negate(&self, v: &RustMathValue) -> Result<RustMathValue, EvalError> {
        use RustMathValue::*;
        match v {
            Integer(n) => Ok(Integer(-n.clone())),
            Rational(r) => Ok(Rational(-r.clone())),
            Float(f) => Ok(Float(-f)),
            Complex(c) => Ok(Complex(-c.clone())),
            IntegerMod(a) => Ok(IntegerMod(-a.clone())),
            Symbol(s) => Ok(Expr(-rustmath_symbolic::Expr::Symbol(s.clone()))),
            Expr(e) => Ok(Expr(-e.clone())),
            Matrix(_) | Polynomial(_) | CliffordElem(_) => {
                self.apply_binary_op(&Integer(rustmath_integers::Integer::from(-1)), "*", v)
            }
            _ => Err(EvalError::new("TypeError", format!("bad operand type for unary -: {}", value_type_name(v)))),
        }
    }

    fn apply_binary_op(&self, left: &RustMathValue, op: &str, right: &RustMathValue) -> Result<RustMathValue, EvalError> {
        use RustMathValue::*;

        match (left, op, right) {
            // Integer operations
            (Integer(a), "+", Integer(b)) => Ok(Integer(a.clone() + b.clone())),
            (Integer(a), "-", Integer(b)) => Ok(Integer(a.clone() - b.clone())),
            (Integer(a), "*", Integer(b)) => Ok(Integer(checked_int_product(op, a, b)?)),
            (Integer(a), "**", Integer(b)) | (Integer(a), "^", Integer(b)) => int_power_value(op, a, b),
            (Rational(a), "**", Integer(b)) | (Rational(a), "^", Integer(b)) => rational_power_value(op, a, b),

            // Integer and Rational together (S9 revision 1). Z is a subring of Q, so a mixed
            // operation and the quotient of two integers are computed in Q, as in MAGMA (ch. 20),
            // Sage (QQ) and PARI. A zero divisor is a ZeroDivisionError (Rational's Div panics).
            (Integer(_) | Rational(_), "/", Integer(_) | Rational(_)) if is_exact_zero(right) => {
                Err(EvalError::new("ZeroDivisionError", "division by zero"))
            }
            (Integer(a), "/", Integer(b)) => {
                check_rational_gcd(op, a.bit_length(), b.bit_length())?;
                rustmath_rationals::Rational::new(a.clone(), b.clone())
                    .map(Rational)
                    .map_err(|e| EvalError::new("ZeroDivisionError", format!("{:?}", e)))
            }
            (Integer(a), "+" | "-" | "*" | "/", Rational(_)) => {
                self.apply_binary_op(&Rational(rustmath_rationals::Rational::from_integer(a.clone())), op, right)
            }
            (Rational(_), "+" | "-" | "*" | "/", Integer(b)) => {
                self.apply_binary_op(left, op, &Rational(rustmath_rationals::Rational::from_integer(b.clone())))
            }
            // Rational operations
            (Rational(a), "+", Rational(b)) => {
                check_rational_op(op, a, b)?;
                Ok(Rational(a.clone() + b.clone()))
            }
            (Rational(a), "-", Rational(b)) => {
                check_rational_op(op, a, b)?;
                Ok(Rational(a.clone() - b.clone()))
            }
            (Rational(a), "*", Rational(b)) => {
                check_rational_op(op, a, b)?;
                Ok(Rational(a.clone() * b.clone()))
            }
            (Rational(a), "/", Rational(b)) => {
                check_rational_op(op, a, b)?;
                Ok(Rational(a.clone() / b.clone()))
            }

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
                Ok(Matrix(result.into()))
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
                Ok(Matrix(result.into()))
            }

            (Matrix(a), "*", Matrix(b)) => {
                // Matrix multiplication, refused above the REPL's size and time limits (S12)
                if a.cols() == b.rows() {
                    check_matrix_product("*", a, b, 0.0)?;
                }
                let result = a.mul(b)
                    .map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
                Ok(Matrix(result.into()))
            }

            // Matrix scalar multiplication
            (Matrix(m), "*", Integer(k)) | (Integer(k), "*", Matrix(m)) => {
                check_scalar_product(op, "a matrix", m.rows() * m.cols(), max_entry_bits(m), k)?;
                Ok(Matrix(m.scalar_mul(k).into()))
            }

            // Polynomial operations
            (Polynomial(p), "+", Polynomial(q)) => {
                Ok(Polynomial(Arc::new((**p).clone() + (**q).clone())))
            }
            (Polynomial(p), "-", Polynomial(q)) => {
                Ok(Polynomial(Arc::new((**p).clone() - (**q).clone())))
            }
            (Polynomial(p), "*", Polynomial(q)) => {
                check_poly_product(op, p, q)?;
                Ok(Polynomial(Arc::new((**p).clone() * (**q).clone())))
            }
            // Polynomial scalar multiplication
            (Polynomial(p), "*", Integer(k)) | (Integer(k), "*", Polynomial(p)) => {
                check_scalar_product(op, "a polynomial", p.coefficients().len(), max_coeff_bits(p), k)?;
                Ok(Polynomial(p.scalar_mul(k).into()))
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
                if s.is_empty() {
                    return Ok(String(std::string::String::new()));
                }
                // S12: the result, not the count, is bounded.
                let max = repl_limits::STRING_BYTES / s.len();
                let count = int_arg::<usize>("*", "repetition count", n, 0..=max,
                    &format!("REPL limit: the result has at most {} bytes", repl_limits::STRING_BYTES))?;
                Ok(String(s.repeat(count)))
            }
            // String concatenation: "a" + "b"
            (String(a), "+", String(b)) => {
                if a.len() + b.len() > repl_limits::STRING_BYTES {
                    return Err(EvalError::new("ValueError", format!(
                        "+: the concatenation would have {} bytes, more than {} (REPL limit)",
                        a.len() + b.len(), repl_limits::STRING_BYTES)));
                }
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

    /// M^e (MAGMA `A ^ n`, ch. 26) for any non-negative integer e, by left-to-right binary
    /// powering (Knuth, TAOCP vol. 2, 4.6.3: square, then multiply by M when the next bit is 1;
    /// no final squaring), with a growth guard (S12) in three parts:
    ///
    /// (a) A certain early refusal from the trace. At the start of iteration i the bits
    ///     ebits-1 .. i+1 of e are read, so r = M^t with t = floor(e / 2^(i+1)) and
    ///     e/t >= 2^(i+1). |tr r| <= d·rho(M)^t, and rho(M)^e = rho(M^e) <= ||M^e||_inf <=
    ///     d·max|entry| (Horn and Johnson, Matrix Analysis, 2nd ed., Thm. 5.6.9), so some entry
    ///     of M^e has log2|entry| >= 2^(i+1)·g - log2 d with g = bit_length(tr r) - 1 - log2 d,
    ///     when g > 0. When that bound, less one bit for the f64 rounding, exceeds RESULT_BITS
    ///     the call is refused before any large product. It does not apply when every tr(M^t)
    ///     is 0 (nilpotent M, or eigenvalues whose powers cancel); then (b) protects.
    /// (b) A size and time check before every product (`check_matrix_product`): rigorous for
    ///     the size, an estimate for the time, summed over the call.
    /// (c) An up-front floor: ebits - 1 squarings and popcount(e) - 1 products by M, each at
    ///     least matrix_product_seconds(d, d, d, 1). The loop keeps no exponent counter, so each
    ///     iteration does O(d^2) work besides its products, which the per-product term covers.
    ///
    /// Checkpoints (S13) before every product. Invariant: at the end of iteration i,
    /// r = M^floor(e / 2^i).
    fn checked_matrix_power(&self, m: &Matrix<Integer>, e: &Integer) -> Result<Matrix<Integer>, EvalError> {
        if m.rows() != m.cols() {
            return Err(EvalError::new("MatrixError", "Matrix power is only defined for square matrices"));
        }
        if e.signum() < 0 {
            return Err(EvalError::new("ValueError", format!(
                "matrix_power: the exponent must be a non-negative integer, got {} (MAGMA allows n < 0 for \
                 an invertible A; the REPL does not compute integer inverses)", show_int(e))));
        }
        let d = m.rows();
        if e.is_zero() || d == 0 {
            return Ok(Matrix::identity(d));
        }
        let ebits = e.bit_length();
        // Binary powering needs ebits - 1 squarings and popcount(e) - 1 products by M, each at
        // least matrix_product_seconds(d, d, d, 1): refuse at once when that floor is too long.
        let products = (ebits - 1) + (e.as_bigint().magnitude().count_ones() - 1);
        let floor_secs = products as f64 * matrix_product_seconds(d, d, d, 1);
        if floor_secs > repl_limits::PRODUCT_SECONDS {
            return Err(EvalError::new("ValueError", format!(
                "matrix_power: an exponent of {} bits needs {} products of {}x{} matrices, about {:.1} s even if \
                 the entries stay small (estimate), more than {} s (REPL limit)",
                ebits, products, d, d, floor_secs, repl_limits::PRODUCT_SECONDS)));
        }
        let log2_d = (d as f64).log2();
        let mut r = m.clone();
        let mut spent = 0.0;
        for i in (0..ebits - 1).rev() {
            self.checkpoint()?;
            // r = M^t with t = floor(e / 2^(i+1)), so e/t >= 2^(i+1): a certain lower bound on
            // the bits of the largest entry of M^e from the trace of r ((a) above).
            let tr = (0..d).fold(Integer::zero(), |acc, j| acc + r.get(j, j).cloned().unwrap_or_else(|_| Integer::zero()));
            let g = (tr.bit_length() as f64 - 1.0) - log2_d;
            if g > 0.0 {
                let lb = ((i + 1) as f64).exp2() * g - log2_d;
                if lb - 1.0 > repl_limits::RESULT_BITS {
                    return Err(EvalError::new("ValueError", format!(
                        "matrix_power: M^e for e = {} has an entry of more than {:.3e} bits (from the trace), \
                         more than {} (REPL limit)", show_int(e), lb - 1.0, repl_limits::RESULT_BITS)));
                }
            }
            spent += check_matrix_product("matrix_power", &r, &r, spent)?;
            r = r.mul(&r).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
            if e.as_bigint().bit(i) {
                self.checkpoint()?;
                spent += check_matrix_product("matrix_power", &r, m, spent)?;
                r = r.mul(m).map_err(|e| EvalError::new("MatrixError", format!("{:?}", e)))?;
            }
        }
        Ok(r)
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
  pow(base, exp)     - base^exp for an integer or rational base and any integer exp (a
                       negative exp gives a rational: pow(2, -1) = 1/2); refused when the
                       result would have more than 2^25 bits (for exp < 0: its denominator)

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
  matrix_power(M, e) - M^e for any integer e >= 0 (binary powering); refused when an entry
                       certainly exceeds 2^25 bits (from the trace), when a product may
                       exceed 2^25 bits in all, or when the products would take more than
                       about 10 s (estimate)

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
                         factored by trial division, Pollard rho and the quadratic sieve
                         within REPL limits; an unfactored part is reported, not guessed
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
  pochhammer(x, n)        - Pochhammer symbol (x)_n = x(x+1)...(x+n-1), the rising factorial (MAGMA ch. 49)
  eulerian(n, k)       - Eulerian number
  narayana(n, k)       - Narayana number
  motzkin(n)           - n-th Motzkin number
  delannoy(m, n)       - Delannoy number
  schroder(n)          - Large Schröder number
  partitions(n)        - Number of partitions of n
  derangements(n)      - Number of derangements of n
  Integer arguments are range-checked; a ValueError names the range. REPL limits, set so
  that an admitted call takes at most about 10 s and a few hundred MB (release build):
  binomial, narayana and delannoy bound a work estimate (the error shows it);
  falling/rising_factorial (any integer n) and multinomial bound an estimate of their
  product tree (10 s); stirling1/stirling2/eulerian bound an estimate of their recurrence
  (10 s); catalan n <= 316227, fibonacci/lucas |n| <= 70000000, bell n <= 5000,
  motzkin n <= 300000, schroder n <= 250000, derangements n <= 2000000,
  partitions n <= 20000000000; sigma(n, k) about k*log2|n| <= 2^28 bits and, from the
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
  prime_pi(n)          - Count of primes ≤ n (n up to 6·10^9)
  primes(a, b)         - List of primes in [a, b) (half-open, as Sage prime_range), each proven
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
  +, -, *            - Arithmetic (integers, matrices, polynomials). Integer and rational
                       products are refused above 2^25 bits; matrix and polynomial products
                       above 2^25 bits in all or about 10 s (estimate); string repetition
                       and concatenation above 16777216 bytes
  ** or ^            - Exponentiation, as pow (negative exponents give rationals)
  /                  - Division (for rationals)

Commands:
  help               - Show this help
  vars               - Show defined variables
  time_limit(s)      - Stop each later cell after s seconds (0: no limit); time_limit()
                       shows the limit
  print(a, b, ...)   - Print the values; as a function it returns the printed text, so a
                       value above the display limit gives its one-line summary

Interrupts and the display:
  The kernel answers while a cell runs. An interrupt (or the time limit) stops the cell at
  the next checkpoint: between lines and sub-expressions, between the products of
  matrix_power and between plot samples. A single library call (factor, det, simplify,
  taylor, nintegrate, find_root, one large product, one plot sample) runs to its end; the
  kernel says so after 2 s, and a restart stops it at once (all variables are lost). An
  interrupted line has no effect. An output whose text would exceed 16777216 bytes is
  shown as a one-line summary with its size; the value is kept.

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

/// A value inside `float_arith` (S9 revision 2): an exact number (Integer or Rational), or an
/// element of the real field of doubles.
enum RealArith {
    Exact(RustMathValue),
    Real(f64),
}

/// The two left-associative precedence levels of the S9 grammar.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChainLevel {
    Sum,
    Product,
}

/// The chain of `s` at one level (S9): every top-level binary `+`/`-` (`is_binary_sign`), or every
/// top-level `*` that is not part of `**` and every `/`, with the operands between them (trimmed).
/// None when the level has no operator or an operand is blank ("3 -": the caller then reports the
/// text as unparsable). `top` is `top_level_chars(s)`.
fn binary_chain<'a>(s: &'a str, top: &[(usize, char)], level: ChainLevel) -> Option<(Vec<&'a str>, Vec<&'static str>)> {
    let bytes = s.as_bytes();
    let mut operands = Vec::new();
    let mut ops = Vec::new();
    let mut start = 0;
    for &(pos, ch) in top {
        let op = match (level, ch) {
            (ChainLevel::Sum, '+') if is_binary_sign(s, pos) => "+",
            (ChainLevel::Sum, '-') if is_binary_sign(s, pos) => "-",
            (ChainLevel::Product, '/') => "/",
            (ChainLevel::Product, '*')
                if !(pos > 0 && bytes[pos - 1] == b'*') && bytes.get(pos + 1) != Some(&b'*') => "*",
            _ => continue,
        };
        operands.push(s[start..pos].trim());
        ops.push(op);
        start = pos + 1;
    }
    operands.push(s[start..].trim());
    (!ops.is_empty() && operands.iter().all(|t| !t.is_empty())).then_some((operands, ops))
}

/// Whether `v` is the Integer or Rational zero (a zero divisor of `/`).
fn is_exact_zero(v: &RustMathValue) -> bool {
    match v {
        RustMathValue::Integer(n) => n.is_zero(),
        RustMathValue::Rational(r) => r.numerator().is_zero(),
        _ => false,
    }
}

/// The text inside the grouping parentheses around `s` (S9): Some(e) when s = "(..(e)..)" with
/// j >= 1 pairs that each enclose all of e, e is not blank and has no top-level comma (a tuple such
/// as "(x, 0, 1)" is not a group; "((x, 0, 1))" gives "(x, 0, 1)"). All j pairs go in one pass over
/// the depth profile (LexState), so a run of any length costs O(len) and no recursion: the k-th
/// '(' of the leading run encloses everything up to the trailing run iff the bracket depth stays
/// >= k between the two runs.
fn group_inner(s: &str) -> Option<&str> {
    let s = s.trim();
    let lead: Vec<usize> = s.char_indices().take_while(|&(_, c)| c == '(' || c.is_whitespace())
        .filter(|&(_, c)| c == '(').map(|(i, _)| i).collect();
    let trail: Vec<usize> = s.char_indices().rev().take_while(|&(_, c)| c == ')' || c.is_whitespace())
        .filter(|&(_, c)| c == ')').map(|(i, _)| i).collect();
    if lead.is_empty() || trail.is_empty() {
        return None;
    }
    let core_start = lead[lead.len() - 1] + 1;
    let core_end = trail[trail.len() - 1]; // byte offset of the first ')' of the trailing run
    if core_start > core_end {
        return None; // "()" or ")(": nothing inside
    }
    let mut lex = LexState::default();
    let mut min_depth = i32::MAX;
    for (i, ch) in s.char_indices() {
        lex.top_level(ch);
        if i >= core_start && i < core_end {
            min_depth = min_depth.min(lex.depth);
        }
    }
    if lex.depth != 0 || lex.quote.is_some() {
        return None; // unbalanced or an open literal
    }
    let core_min = if core_start == core_end { lead.len() as i32 } else { min_depth };
    let mut j = (core_min.max(0) as usize).min(lead.len()).min(trail.len());
    while j > 0 {
        let inner = &s[lead[j - 1] + 1..trail[j - 1]];
        if inner.trim().is_empty() {
            return None;
        }
        if !top_level_chars(inner).iter().any(|&(_, c)| c == ',') {
            return Some(inner);
        }
        j -= 1; // the innermost pair holds a tuple: keep it
    }
    None
}

/// The double nearest to an exact number (S11), `what` naming it in errors. Integer and Rational
/// round once to nearest (correctly rounded); a value whose magnitude rounds above f64::MAX is a
/// ValueError "{what}: {v} is outside the range of a double", never inf or NaN. A nonzero value
/// below the least subnormal (about 4.9e-324) rounds to 0.0, the IEEE result. A Float passes when
/// finite. Other values: None, so each caller keeps its own TypeError.
fn exact_to_f64(v: &RustMathValue, what: &str) -> Option<Result<f64, EvalError>> {
    let (f, shown) = match v {
        RustMathValue::Integer(n) => (n.to_f64().unwrap_or(f64::NAN), show_int(n)),
        RustMathValue::Rational(r) => (
            r.to_f64().unwrap_or(f64::NAN),
            if r.numerator().bit_length() > 128 || r.denominator().bit_length() > 128 {
                format!("a rational with a {}-bit numerator", r.numerator().bit_length())
            } else {
                r.to_string()
            },
        ),
        RustMathValue::Float(x) => (*x, x.to_string()),
        _ => return None,
    };
    Some(if f.is_finite() {
        Ok(f)
    } else {
        Err(EvalError::new("ValueError", format!("{}: {} is outside the range of a double", what, shown)))
    })
}

/// Bracket depth and string-literal state of a left-to-right scan of REPL source text (S9).
/// A string literal starts with `"` or `'` outside a literal and ends at the next unescaped copy
/// of the same quote; inside it a backslash escapes the next character, as in
/// `strip_inline_comment`. Brackets inside a literal do not count.
#[derive(Clone, Copy, Debug, Default)]
struct LexState {
    depth: i32,
    quote: Option<char>,
    escaped: bool,
}

impl LexState {
    /// Advances over `ch`. True when `ch` lies at bracket depth 0 outside every string literal
    /// and is neither a bracket nor a quote: the characters at which a splitter may cut.
    fn top_level(&mut self, ch: char) -> bool {
        if let Some(q) = self.quote {
            if self.escaped {
                self.escaped = false;
            } else if ch == '\\' {
                self.escaped = true;
            } else if ch == q {
                self.quote = None;
            }
            return false;
        }
        match ch {
            '"' | '\'' => {
                self.quote = Some(ch);
                false
            }
            '(' | '[' | '{' => {
                self.depth += 1;
                false
            }
            ')' | ']' | '}' => {
                self.depth -= 1;
                false
            }
            _ => self.depth == 0,
        }
    }
}

/// The byte offset and the character of every top-level character of `s` (`LexState::top_level`).
fn top_level_chars(s: &str) -> Vec<(usize, char)> {
    let mut lex = LexState::default();
    s.char_indices().filter(|&(_, ch)| lex.top_level(ch)).collect()
}

/// Whether `s` is exactly one string literal: it opens with a quote, and that literal closes at
/// the last character of `s`.
fn is_string_literal(s: &str) -> bool {
    let mut lex = LexState::default();
    let mut chars = s.char_indices();
    match chars.next() {
        Some((_, q)) if q == '"' || q == '\'' => {
            lex.top_level(q);
        }
        _ => return false,
    }
    for (i, ch) in chars {
        lex.top_level(ch);
        if lex.quote.is_none() {
            return i + ch.len_utf8() == s.len();
        }
    }
    false
}

/// Whether the '+' or '-' at byte `pos` of `s` is a binary operator (S9): the previous non-blank
/// character ends an operand (a name or number, a closing bracket or a closing quote), and the
/// sign is not the exponent sign of a float literal such as 1e-3 or 2.5E+10. Otherwise it is a
/// unary sign: at the start, or after an operator, a comma, '=' or an opening bracket.
fn is_binary_sign(s: &str, pos: usize) -> bool {
    let before = s[..pos].trim_end();
    let prev = match before.chars().last() {
        Some(c) => c,
        None => return false,
    };
    let ends_operand = prev.is_alphanumeric()
        || matches!(prev, '_' | '.' | ')' | ']' | '}' | '"' | '\'');
    if !ends_operand {
        return false;
    }
    // An exponent sign follows the 'e' directly, and the 'e' ends a decimal mantissa that is not
    // the tail of a name (x1e - 3 is a subtraction).
    if (prev == 'e' || prev == 'E') && before.len() == pos {
        let mantissa = &before[..before.len() - 1];
        let start = mantissa
            .char_indices()
            .rev()
            .take_while(|&(_, c)| c.is_ascii_digit() || c == '.')
            .last()
            .map_or(mantissa.len(), |(i, _)| i);
        let run = &mantissa[start..];
        let after_name = mantissa[..start].chars().last().is_some_and(|c| c.is_alphanumeric() || c == '_');
        let digit_follows = s[pos + 1..].starts_with(|c: char| c.is_ascii_digit());
        if run.chars().any(|c| c.is_ascii_digit()) && !after_name && digit_follows {
            return false;
        }
    }
    true
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
        Expr::Integer(n) => decimal_string(n),
        Expr::Rational(r) => format!("\\frac{{{}}}{{{}}}", decimal_string(r.numerator()), decimal_string(r.denominator())),
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
//
// Follow-ups unit J1a (parser, conversions, limits, sqrt_mod):
// - Arithmetic in Q: ch. 18 §18.4.1 ("Division `/` gives a rational result"), ch. 20 §20.1.2
//   (coercion between Z and Q), §20.2.2 (a / b in reduced form), §20.4.2 (a ^ k); digests
//   `Chapter_18_Ring_of_Integers.md:125`, `Chapter_20_Rational_Field.md:44-46, 88, 157`.
// - Range bounds in the real field of 53 bits (`float_arith`): ch. 25 §25.1.1-§25.1.2, an
//   operation with a real operand is done in the real field (`Chapter_25_Real_and_Complex_Fields.md:56-58`).
// - The Pochhammer symbol (a)_n = a(a+1)...(a+n-1): ch. 49 (`HypergeometricSeries`,
//   `Chapter_49_Power_Laurent_and_Puiseux_Series.md:292`).
// - sqrt_mod: ch. 18 §18.10.1 (`Factorization` with TrialDivisionLimit, PollardRhoLimit,
//   MPQSLimit, Proof), §18.12.1 (`Modsqrt`), ch. 19 §19.5.6 (`Sqrt`, `AllSqrts` with a known
//   factorization). Primes: §18.9.2 (`PrimesInInterval`), with H. L. Montgomery, R. C. Vaughan,
//   *The large sieve*, Mathematika 20 (1973) 119-134 (the count bound) and J. Sorenson,
//   J. Webster, Math. Comp. 86 (2017) 985-1003 (the twelve bases below psi_12).
//
// Follow-ups unit J1b (shared values, growth guards, display, interrupts):
// - Powers: ch. 18 `n ^ k` (`Chapter_18_Ring_of_Integers.md:125`), ch. 20 `a ^ k`
//   (`Chapter_20_Rational_Field.md:157`), ch. 26 `A ^ n` ("A^0 is the identity. If n < 0, A must
//   be invertible", `Chapter_26_Matrices.md:204`). Matrix powers by left-to-right binary powering
//   (D. E. Knuth, TAOCP vol. 2, 3rd ed., 4.6.3), with the spectral-radius bound
//   rho(A) <= |||A||| (R. A. Horn, C. R. Johnson, Matrix Analysis, 2nd ed., Thm. 5.6.9).
// - Display: radix conversion cost, R. P. Brent, P. Zimmermann, Modern Computer Arithmetic, 1.7.
// - Interrupts: ch. 1 1.2 (`<Ctrl>-C` "interrupts at a convenient point",
//   `Chapter_01_Statements_and_Expressions.md:24-31`) and ch. 3 3.12 (`Alarm(s)`,
//   `Chapter_03_Input_and_Output.md:400`).
// - Counting arms: ch. 18 and ch. 145 `NumberOfPartitions` (`Chapter_18_Ring_of_Integers.md:270`,
//   `Chapter_145_Partitions_Words_and_Young_Tableaux.md:55`), ch. 144 `NumberOfPermutations`
//   (the falling factorial, `Chapter_144_Enumerative_Combinatorics.md:37`), ch. 49 Pochhammer
//   symbol; balanced product trees, J. von zur Gathen, J. Gerhard, Modern Computer Algebra,
//   3rd ed., 10.1.
// ---------------------------------------------------------------------------------------------

/// Resource limits of REPL commands (T-250). Interrupts and the optional time limit (S13,
/// `time_limit`) stop a cell at its next checkpoint, between library calls; a single library
/// call runs to its end, and an out-of-memory abort still kills the kernel. So these limits
/// still bound each single call.
///
/// The limits added for T-250 are set so that a refused call would take more than about 10 s
/// (release build, the development container) or a few hundred MB (about 400 MB), and so that
/// an admitted call needs at most a few hundred MB (except where noted below: taylor, derived
/// differential forms). They were sized from rustc -O mirrors of the exact callee loops over
/// num-bigint 0.4.8 (the X2-jupyter implementation report has the timings). Where the cost
/// depends on more than one argument, the limit bounds a work, size or time estimate
/// (`binomial_work_u32`, `stirling_seconds`, `factorial_product_seconds`, `sigma_seconds`,
/// `matrix_product_seconds`, `check_form_components`) rather than one argument.
///
/// Unit J1b re-measured the limits whose callees C1 changed, and those that counted copies of
/// a value (F2 removed them), end to end with the real `ReplContext` in release (a harness that
/// runs one REPL line per argument in its own process and reads VmHWM; the computation and the
/// display together). Each limit is the largest ladder value whose admitted calls take at most
/// about 10 s and 400 MB; where that and a round number disagree, the time bound wins, so the
/// first refused value of a one-argument limit costs close to 10 s (FIBONACCI_N: 7·10^7 takes
/// 8.4 s, 8·10^7 10.5 s). TAYLOR_ORDER, FORM_COMPONENTS and DIFF_FORM_DEGREE wait for the
/// rustmath-symbolic and rustmath-manifolds units (S1, MF1); the binomial family (BINOMIAL_*,
/// CATALAN_N, DELANNOY_SUM_CAP) and PRIMES_COUNT were not re-measured: their callees got faster
/// (catalan(316227) 0.03 s) or their values are now held once (primes), so they only refuse
/// more than needed. Some bounds do not bound the time:
/// `nth_prime` gets its callee's domain only, `taylor`'s cost and the
/// size of derived differential forms depend on the expressions, and `sigma` must factor n,
/// which no limit bounds. The older
/// limits (the first block) are unchanged and are listed here so that every limit is in one
/// place.
///
/// The limits of unit J1a (the last block) follow the same policy and were measured on the
/// same container: the nesting depth of `eval_expr` (a stack bound, CPython's 200); the plot
/// options (the old clamp maxima, now refused above); the graph constructors (a memory model of
/// the hash tables, within 2 % of the REPL's peak); `prime_pi` (the sieve's linear time);
/// `primes` (a proven bound on the number of primes for memory, and above 2^64 a time model of
/// Baillie-PSW per odd candidate and of ECPP per probable prime); and the factorization behind
/// `sqrt_mod` (MAGMA's TrialDivisionLimit and PollardRhoLimit defaults, the sieve up to 60
/// digits, trial division to the reach of the old command, ECPP up to 1024 bits).
mod repl_limits {
    // Existing limits, unchanged.
    pub const FACTORIAL_N: u32 = 10_000;
    pub const MATRIX_DIM: usize = 1_000; // identity, zeros
    pub const ONES_DIM: usize = 100;
    pub const HISTOGRAM_BINS: usize = 1_000;
    pub const PLOT_POINTS: usize = 10_000;
    pub const CLIFFORD_DIM: usize = 10; // CliffordAlgebra, ExteriorAlgebra
    pub const JORDAN_DIM: usize = 20;
    pub const SPECIAL_JORDAN_N: usize = 10;

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
    /// fibonacci and lucas, |n| (C1: doubling). End to end with the display (J1b ladder `fib`,
    /// release): 5·10^7 5.2 s, 6·10^7 6.7 s, 7·10^7 8.4 s and 87 MB, 8·10^7 10.5 s.
    pub const FIBONACCI_N: i64 = 70_000_000;
    /// bell (C1: Bell triangle in place). Ladder `bell`: 4000 2.7 s, 5000 5.1 s and 33 MB,
    /// 6000 10.6 s.
    pub const BELL_N: u32 = 5_000;
    /// stirling1, stirling2 and eulerian keep C1's rolling band of w + 1 entries of at most B
    /// bits (`stirling_band`): a memory bound on (w + 1)(B/8 + 32) bytes.
    pub const TABLE_BYTES: u64 = 400_000_000;
    /// stirling1, stirling2 and eulerian: the estimated seconds of the recurrence
    /// (`stirling_seconds`).
    pub const STIRLING_SECONDS: f64 = 10.0;
    /// motzkin (C1: three-term recurrence, a time bound). Ladder `motzkin`: 250000 5.8 s,
    /// 300000 8.4 s and 5 MB.
    pub const MOTZKIN_N: u32 = 300_000;
    /// schroder and schroder_small (C1: rolling recurrence, a time bound, a few MB). Ladder
    /// `schroder`: 200000 6.0 s, 250000 9.7 s, 300000 14.0 s.
    pub const SCHRODER_N: u32 = 250_000;
    /// derangements (C1: binary splitting). Ladder `derange`: 1.5·10^6 5.2 s, 2·10^6 8.7 s and
    /// 54 MB.
    pub const DERANGEMENTS_N: u32 = 2_000_000;
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
    /// 10000 does not); the recursion is part of the same split-out. Since J1b the kernel runs
    /// the REPL on a thread of 64 MiB (`kernel::EVAL_STACK_BYTES`); the limit waits for the
    /// rustmath-symbolic unit (S1) to be re-measured.
    pub const TAYLOR_ORDER: usize = 20_000;
    /// EuclideanSpace(n) builds n coordinate names: 221 MB at 4·10^6, 385 MB at 7·10^6 (0.49 s).
    /// A Manifold value is shared (Arc), so the names are never copied.
    pub const EUCLIDEAN_DIM: usize = 7_000_000;
    /// coordinate_form stores dim(M) components, 64 bytes each (an `Expr`); since F2 (J1b) a
    /// form is held once (behind an Arc, shared by the variable, `_` and every argument
    /// lookup). It needs a chart of dim(M) names (J1a S14), about 56 bytes a name plus the
    /// vector. Measured with the real `ReplContext` (VmHWM after the chart line / after
    /// `w = coordinate_form(M, C, 0)`; `v = w` and `wedge(w, w, C)` add nothing): 1.5·10^6
    /// 192 / 232 MB, 2·10^6 285 / 289 MB, 2.5·10^6 343 / 374 MB, 3·10^6 383 / 441 MB. A second
    /// form adds 64 bytes per coordinate (530 MB at 2.5·10^6).
    pub const MANIFOLD_COORDS: usize = 2_500_000;
    /// diff_form degree: keeps `p + 1` and `p + q` of forms built directly from overflowing
    /// (exterior_derivative and wedge check the degrees of derived forms themselves).
    pub const DIFF_FORM_DEGREE: usize = usize::MAX / 2;
    /// exterior_derivative and wedge build all n^degree components of the result: at 10^6
    /// components d(dx_0) (n = 1000) and dx_0 ∧ dx_0 (n = 1000) peak at 382 MB, d(d(dx_0))
    /// (n = 100) at 574 MB, because rustmath-symbolic does not simplify and each further d
    /// enlarges the expressions; that growth is not bounded here (a split-out). These peaks
    /// were measured with the copies the REPL made before F2 (J1b), so they are upper bounds;
    /// the limit is re-measured once MF1 (rustmath-manifolds) lands.
    pub const FORM_COMPONENTS: usize = 1_000_000;
    /// partitions(n) through C1's number_of_partitions (Hardy-Ramanujan-Rademacher, exact).
    /// Ladder `partitions` (release, end to end): 10^9 0.7 s, 10^10 4.0 s, 1.5·10^10 7.6 s,
    /// 1.7·10^10 6.6 s, 1.8·10^10 7.3 s, 1.9·10^10 8.3 s, 2·10^10 - 1 8.4 s, 2·10^10 7.2 to 7.4 s
    /// (the cost is not monotone in n), 10 MB.
    pub const PARTITIONS_N: u64 = 20_000_000_000;
    /// Above this min(m, n) the work of delannoy(m, n) is certainly above BINOMIAL_WORK
    /// (it is at least min(m, n)^3 / 3), so `delannoy_work` does not sum it.
    pub const DELANNOY_SUM_CAP: u32 = 20_000;

    // Limits added for the parser and the commands of the follow-ups unit J1a.
    /// Nesting depth of `eval_expr` (and of `float_arith`), as CPython's parser refuses more
    /// than 200 nested parentheses. Each level is a recursive call, so deeper nesting overflows
    /// the stack and kills the kernel. Measured on the J1a prototype (review 3, m4): alternating
    /// nesting such as `-(-(...(1)...))` aborted at 600 to 1000 levels in a debug build and at
    /// 3000 to 5000 in an optimized one (8 MB thread); it costs two calls per level, so it is
    /// refused from 100 levels. Runs of signs and of parentheses are read in one step and do
    /// not count. Nested calls `abs(abs(...))` use far larger frames in a debug build (about
    /// 240 KB per level), where they can still overflow below this bound.
    pub const EVAL_DEPTH: usize = 200;
    /// Plot options (N5): the old clamp maxima, now refused above. They bound the number of SVG
    /// elements. The lower bounds are those of the drawing code (2 grid points, 1 contour).
    pub const SLOPE_FIELD_POINTS: usize = 50;
    pub const PARAMETRIC_POINTS: usize = 1_000;
    pub const CONTOURS: usize = 30;
    pub const CONTOUR_POINTS: usize = 100;
    pub const VECTOR_FIELD_POINTS: usize = 30;
    pub const IMPLICIT_POINTS: usize = 200; // implicit_plot, region_plot
    /// Graph constructors (S15): `graph_bytes` of the graph, which since F2 (J1b) is held once;
    /// `add_edge` copies it, so two copies at its peak, about 400 MB at the limit. Admitted
    /// maxima: Graph(4166666), path_graph and cycle_graph(1785714), star_graph and
    /// wheel_graph(1617192), complete_graph(3585) (at 3586 the tables double to 8192 buckets).
    /// Measured (release, VmHWM of `G = ...` / of `H = add_edge(G, 0, 2)`): path_graph(1780000)
    /// 199 / 394 MB, Graph(4100000) 196 / 388 MB, complete_graph(3585) 134 / 264 MB,
    /// star_graph(1780000) 217 / 430 MB (model 218 MB).
    pub const GRAPH_BYTES: u128 = 200_000_000;
    /// prime_pi(x): the segmented sieve is linear in x, 1.75 s per 10^9 in release (I1's sieve:
    /// prime_pi(4·10^9) in 7.0 s), so about 10.5 s here; gp primepi(6*10^9) = 279545368.
    pub const PRIME_PI_X: u64 = 6_000_000_000;
    /// primes(a, b): at most this many entries (`count_bound`), a memory bound: a listed prime
    /// costs about 375 bytes at the peak of an assignment with its display text.
    pub const PRIMES_COUNT: u64 = 1_000_000;
    /// Spans above this have count_bound > PRIMES_COUNT (count_bound(8·10^6) = 1006608).
    pub const PRIMES_SPAN: u64 = 8_000_000;
    /// primes(a, b) above 2^64: modelled seconds of the two passes (`checked_primes`).
    pub const PRIMES_SECONDS: f64 = 10.0;
    /// One strong (Miller-Rabin) test by bit size, release: num-bigint modpow (J1a design §6.3,
    /// `tcost.rs`).
    pub const PRIMES_STRONG_TEST_SECONDS: [(u64, f64); 9] = [(64, 8.4e-6), (128, 1.4e-5), (256, 3.4e-5),
        (512, 1.5e-4), (1024, 7.3e-4), (2048, 4.7e-3), (4096, 4.5e-2), (8192, 2.6e-1), (16384, 2.2)];
    /// One ECPP proof with its verification (`try_is_prime`), release: means over random primes.
    pub const PRIMES_PROOF_SECONDS: [(u64, f64); 9] = [(100, 0.008), (128, 0.024), (256, 0.070),
        (512, 0.41), (768, 1.7), (1024, 3.3), (1536, 20.6), (2048, 75.8), (3072, 957.0)];
    /// sqrt_mod's factorization (MAGMA `Factorization` parameters, ch. 18 §18.10.1;
    /// `sqrt_mod_factor`). Phase 1: trial division below this (MAGMA's TrialDivisionLimit).
    pub const SQRT_MOD_TRIAL: u64 = 10_000;
    /// Phase 2: Brent rho iterations per composite (MAGMA's PollardRhoLimit).
    pub const SQRT_MOD_RHO_STEPS: u64 = 8_191;
    /// Phase 2 only on cofactors this small: its BPSW and rho cost grow as bits^3 and bits^2.
    /// HEAD needed more than 17 s for one prime n of 4096 bits (J1a review 3).
    pub const SQRT_MOD_I2_MAX_BITS: u64 = 4_096;
    /// The quadratic sieve up to this many digits: about 2.4 s in release at 60 digits (I2).
    pub const SQRT_MOD_QS_DIGITS: u32 = 60;
    /// Phase 3: modelled nanoseconds of trial division; the prime bound is
    /// T(bits) = SQRT_MOD_TRIAL_WORK / (55 + bits / 11) in u64 integer arithmetic, the reach of
    /// the old trial division (every odd candidate, 55 + bits/11 ns each) in 10 s.
    pub const SQRT_MOD_TRIAL_WORK: u64 = 20_000_000_000;
    /// ECPP proofs up to this size, the bound of is_prime (I1: 3.3 s mean, 4.3 s max at 1024 bits
    /// in release).
    pub const SQRT_MOD_PROOF_BITS: u64 = super::PROOF_BITS;

    // Limits of the follow-ups unit J1b (growth guards S12, display D). They bound the size of
    // a result, not an argument: the size of a power or product is known before it is computed.
    /// b ** e, b ^ e and pow(b, e) for integer and rational b: the bits of the result (for e < 0,
    /// of its denominator; for a rational base, of its numerator and denominator together), and
    /// Integer * Integer and the Rational products. 3^e at 2^25 bits takes 2.4 s for the power
    /// and 1.4 s for the display (release; `x = 3 ** 21170489` 3.82 s end to end, 50 MB).
    pub const POW_BITS: f64 = 33_554_432.0; // 2^25
    /// Products of matrices and polynomials, matrix powers and scalar products: a bound on the
    /// total bits of the entries of each product.
    pub const RESULT_BITS: f64 = 33_554_432.0; // 2^25
    /// matrix_power, Matrix * Matrix and Polynomial * Polynomial: the estimated seconds of the
    /// products of one call (`matrix_product_seconds`; release fit).
    pub const PRODUCT_SECONDS: f64 = 10.0;
    /// String repetition and concatenation: the bytes of the result.
    pub const STRING_BYTES: usize = 1 << 24;
    /// The text of one REPL output in bytes (D): a value whose text, estimated before anything
    /// is formatted (`display_size`), would be larger is shown as a one-line summary that gives
    /// its size; the value itself is kept and can be used. One integer under this budget has at
    /// most about 5.57·10^7 bits and prints in about 3 s (release; `sigma(2, 55700000)` 2.93 s
    /// end to end).
    pub const OUTPUT_BYTES: u64 = 1 << 24; // 16 MiB
    /// Symbolic expressions are printed by rustmath-symbolic, whose integers convert with
    /// num-bigint (quadratic: 0.44 s at 2^22 bits, 3.6 s at 2^24, release): an expression with a
    /// larger integer is summarised.
    pub const LIBRARY_INT_DISPLAY_BITS: u64 = 1 << 22;
}

// C1's number_of_partitions panics above its domain; the REPL limit keeps every call inside it.
const _: () = assert!(repl_limits::PARTITIONS_N <= NUMBER_OF_PARTITIONS_MAX_N);

/// psi_12 = 318665857834031151167461, the least strong pseudoprime to the twelve prime bases
/// 2, ..., 37 (J. Sorenson, J. Webster, *Strong pseudoprimes to twelve prime bases*, Math. Comp.
/// 86 (2017) 985-1003). Below it `is_probable_prime` runs those twelve tests, which decide.
const PSI_12: u128 = 318_665_857_834_031_151_167_461;

/// An upper bound on the number of primes in [x, x + y) for every x >= 0 (S15): 0 for y <= 0,
/// 1 for y = 1, else min(ceil(y/2) + 1, floor(2y / ln y)). The second term is H. L. Montgomery
/// and R. C. Vaughan, *The large sieve*, Mathematika 20 (1973) 119-134, Corollary 2:
/// pi(x + y) - pi(x) < 2y / log y for y > 1. `None` when y > `repl_limits::PRIMES_SPAN`
/// (8·10^6, an exact Integer comparison): there the bound already exceeds
/// `repl_limits::PRIMES_COUNT`, since count_bound(8·10^6) = 1006608 and the bound increases
/// with y. Below it the quotient is computed in f64 on y as u64 (exact, no overflow, no NaN;
/// its rounding error, about 10^-10, can move the floor only at a quotient that close to an
/// integer).
fn count_bound(y: &Integer) -> Option<u64> {
    if y.signum() <= 0 {
        return Some(0);
    }
    if *y > Integer::from(repl_limits::PRIMES_SPAN) {
        return None;
    }
    let y = y.to_u64().expect("0 < y <= PRIMES_SPAN");
    if y == 1 {
        return Some(1);
    }
    let yf = y as f64;
    let mv = (2.0 * yf / yf.ln()).floor() as u64;
    Some((y.div_ceil(2) + 1).min(mv))
}

/// `table` is a list of (bits, seconds) measurements, ascending; the value at `bits` by
/// log-log interpolation (a power law between two points), the first value below the table,
/// and the last segment's power law extended above it (S15).
fn table_seconds(table: &[(u64, f64)], bits: u64) -> f64 {
    let (b0, t0) = table[0];
    if bits <= b0 {
        return t0;
    }
    let power_law = |(b1, t1): (u64, f64), (b2, t2): (u64, f64)| -> f64 {
        let e = (t2 / t1).ln() / (b2 as f64 / b1 as f64).ln();
        t1 * (bits as f64 / b1 as f64).powf(e)
    };
    for w in table.windows(2) {
        if bits <= w[1].0 {
            return power_law(w[0], w[1]);
        }
    }
    power_law(table[table.len() - 2], table[table.len() - 1])
}

/// Modelled seconds of `is_probable_prime` on one odd candidate of `bits` bits (S15; release
/// build): trial division by the primes below 1000 (the 2e-6 term below 1000 bits, inside
/// T_strong above), one strong test for the 16.3 % of odd numbers that pass it (Mertens'
/// product over the odd primes below 1000), and for the primes, a fraction of about
/// 2.885 / bits of the odd candidates, at most 12 more strong tests (34.6 / bits). That is at
/// most 0.163 + 34.6 / bits strong tests, below the factor 0.25 + 40 / bits used here.
fn primes_t_test(bits: u64) -> f64 {
    2e-6 + table_seconds(&repl_limits::PRIMES_STRONG_TEST_SECONDS, bits) * (0.25 + 40.0 / bits as f64)
}

/// Modelled seconds of one ECPP proof (`try_is_prime`) of a prime of `bits` bits (S15).
fn primes_t_proof(bits: u64) -> f64 {
    table_seconds(&repl_limits::PRIMES_PROOF_SECONDS, bits)
}

/// The REPL command `primes(a, b)`: the primes in [a, b), half-open as Sage `prime_range`
/// (MAGMA `PrimesInInterval(a, b - 1)`, ch. 18 §18.9.2), each proven (S15).
///
/// 1. Memory: the list has at most [`count_bound`]`(b - max(a, 2))` entries, which must be at
///    most `repl_limits::PRIMES_COUNT`.
/// 2. Below 2^64: the segmented sieve (rustmath-integers `try_prime_range`, exact).
/// 3. From 2^64 up, two passes over the odd candidates c of [max(a, 2^64), b), with B = bits of
///    b - 1. The call is refused before any test when n_odd · t_test(B) exceeds
///    `repl_limits::PRIMES_SECONDS`. Pass 1 runs `is_probable_prime` (Baillie-PSW, exact below
///    psi_12) on each c; `false` is a proof of compositeness. After each probable prime at or
///    above psi_12 (k of them so far), the call is refused as soon as
///    n_odd · t_test(B) + k · t_proof(B) exceeds the limit. Pass 2 proves each of those with
///    `try_is_prime` (ECPP): `Ok(false)` drops c (a BPSW pseudoprime), an unresolved proof is a
///    ValueError. The answer never depends on the time models: a refusal answers nothing.
fn checked_primes(a: &Integer, b: &Integer) -> Result<Vec<Integer>, EvalError> {
    let two = Integer::from(2);
    let lo = if *a > two { a.clone() } else { two.clone() };
    let span = b.clone() - lo.clone();
    if span.signum() <= 0 {
        return Ok(Vec::new());
    }
    match count_bound(&span) {
        None => {
            return Err(EvalError::new("ValueError", format!(
                "primes: the span b - max(a, 2) is {}, above {}, so [a, b) may contain more than {} primes \
                 (Montgomery-Vaughan bound), more than the REPL limit {} (a memory bound)",
                show_int(&span), repl_limits::PRIMES_SPAN, repl_limits::PRIMES_COUNT, repl_limits::PRIMES_COUNT)));
        }
        Some(c) if c > repl_limits::PRIMES_COUNT => {
            return Err(EvalError::new("ValueError", format!(
                "primes: [a, b) may contain up to {} primes (Montgomery-Vaughan bound), more than the \
                 REPL limit {} (a memory bound)",
                c, repl_limits::PRIMES_COUNT)));
        }
        Some(_) => {}
    }
    let two_64 = Integer::from(u64::MAX) + Integer::one();
    let mut out = Vec::new();
    if lo < two_64 {
        let hi = if *b < two_64 { b.clone() } else { two_64.clone() };
        // The sieve part of try_prime_range: exact, no proof can fail.
        out = try_primes_between(&lo, &hi)
            .map_err(|e| EvalError::new("ValueError", format!("primes: {}", e)))?;
    }
    if *b <= two_64 {
        return Ok(out);
    }
    let big_lo = if lo > two_64 { lo } else { two_64 };
    let bits = (b.clone() - Integer::one()).bits();
    // Odd integers in [L, H): floor(H/2) - floor(L/2) (both positive).
    let n_odd = (b.clone() / two.clone() - big_lo.clone() / two.clone())
        .to_u64()
        .expect("at most PRIMES_SPAN candidates");
    let t_pass1 = n_odd as f64 * primes_t_test(bits);
    if t_pass1 > repl_limits::PRIMES_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "primes: testing the {} odd candidates of [max(a, 2^64), b), numbers of {} bits, needs about \
             {:.0} s (REPL limit {} s)",
            n_odd, bits, t_pass1, repl_limits::PRIMES_SECONDS)));
    }
    let psi12 = Integer::from_decimal_str(&PSI_12.to_string()).expect("decimal literal");
    let t_proof = primes_t_proof(bits);
    // Pass 1: Baillie-PSW on every odd candidate.
    let mut probable = Vec::new();
    let mut k_big = 0u64;
    let mut c = if big_lo.is_odd() { big_lo } else { big_lo + Integer::one() };
    while c < *b {
        if is_probable_prime(&c) {
            if c >= psi12 {
                k_big += 1;
                let t = t_pass1 + k_big as f64 * t_proof;
                if t > repl_limits::PRIMES_SECONDS {
                    return Err(EvalError::new("ValueError", format!(
                        "primes: [a, b) has at least {} BPSW probable primes that need a proof (at or above \
                         psi_12 = {}); proving them prime needs about {:.1} s or more (REPL limit {} s)",
                        k_big, PSI_12, t, repl_limits::PRIMES_SECONDS)));
                }
            }
            probable.push(c.clone());
        }
        c = c + two.clone();
    }
    // Pass 2: an ECPP proof for each probable prime at or above psi_12 (below it the twelve
    // strong tests of pass 1 decide).
    for c in probable {
        if c < psi12 {
            out.push(c);
            continue;
        }
        match try_is_prime(&c) {
            Ok(true) => out.push(c),
            Ok(false) => {}
            Err(e) => {
                return Err(EvalError::new("ValueError", format!(
                    "primes: {} is a BPSW probable prime that could not be proven prime ({})",
                    show_int(&c), e)));
            }
        }
    }
    Ok(out)
}

/// Bytes of a std `HashSet<usize>` table holding d entries (S15). std's `HashSet` is hashbrown:
/// an empty set allocates nothing; a set of d entries has 4 buckets for d <= 3, 8 for d <= 7,
/// else the power of two >= 8d/7, with 9 bytes per bucket (the entry and its control byte),
/// plus the trailing control bytes and the allocator's header (32 bytes together, measured).
/// std does not promise this layout, so the model is empirical: it matched the real REPL peak
/// within 2 % (J1a design §6.1, VmHWM of `path_graph(10^6)`, `complete_graph(3000)`,
/// `Graph(2·10^6)`, `star_graph(10^6)`). `None` when the value does not fit in u128.
fn hash_set_bytes(d: u128) -> Option<u128> {
    match d {
        0 => Some(0),
        1..=3 => Some(64),
        4..=7 => Some(96),
        _ => 8u128
            .checked_mul(d)?
            .div_ceil(7)
            .checked_next_power_of_two()?
            .checked_mul(9)?
            .checked_add(32),
    }
}

/// The memory of one rustmath-graphs `Graph` with `v` vertices whose degrees come in
/// (count, degree) groups (S15): 48 bytes per vertex (the set in the adjacency `Vec`) plus each
/// vertex's table ([`hash_set_bytes`]). `None` when the model overflows u128, which is far
/// above every limit.
fn graph_bytes(v: u128, groups: &[(u128, u128)]) -> Option<u128> {
    let mut b = v.checked_mul(48)?;
    for &(c, d) in groups {
        b = b.checked_add(c.checked_mul(hash_set_bytes(d)?)?)?;
    }
    Some(b)
}

/// Refuses a graph constructor whose graph needs more than `repl_limits::GRAPH_BYTES` bytes by
/// [`graph_bytes`] (S15); `e` is the number of edges, for the message. A model that
/// overflows u128 is refused too.
fn check_graph_size(cmd: &str, v: u128, e: u128, groups: &[(u128, u128)]) -> Result<(), EvalError> {
    let need = match graph_bytes(v, groups) {
        Some(b) if b <= repl_limits::GRAPH_BYTES => return Ok(()),
        Some(b) => format!("about {} bytes", b),
        None => "more than 2^128 bytes".to_string(),
    };
    Err(EvalError::new("ValueError", format!(
        "{}: a graph with {} vertices and {} edges needs {} (REPL limit {}, a memory bound; \
         add_edge copies the graph)",
        cmd, v, e, need, repl_limits::GRAPH_BYTES)))
}

/// The coordinate names of `chart(M, names)` (S14): `names` separated by commas, exactly dim(M) of
/// them (Sage `Chart`: "the list of coordinates must contain dim elements"), each a valid
/// identifier, pairwise distinct (each names a coordinate symbol). For dim(M) = 0 the string is
/// empty.
fn chart_coordinate_names(names: &str, dim: usize) -> Result<Vec<String>, EvalError> {
    let list: Vec<String> = if names.trim().is_empty() {
        Vec::new()
    } else {
        names.split(',').map(|s| s.trim().to_string()).collect()
    };
    if list.len() != dim {
        return Err(EvalError::new("ValueError", format!(
            "chart: the manifold has dimension {}, so the chart needs {} coordinate names, got {}",
            dim, dim, list.len())));
    }
    let mut seen = std::collections::HashSet::new();
    for name in &list {
        if !is_valid_identifier(name) {
            return Err(EvalError::new("ValueError", format!(
                "chart: coordinate name {:?} is not an identifier", name)));
        }
        if !seen.insert(name.as_str()) {
            return Err(EvalError::new("ValueError", format!(
                "chart: coordinate name {} appears twice", name)));
        }
    }
    Ok(list)
}

/// The REPL check that `chart` has one coordinate per dimension of the manifold it is used on
/// (S14): a chart is made for one dimension, but REPL values can be combined freely.
fn check_chart(cmd: &str, chart: &Chart, dim: usize) -> Result<(), EvalError> {
    if chart.dimension() != dim {
        return Err(EvalError::new("ValueError", format!(
            "{}: the chart has {} coordinates, but the manifold has dimension {}",
            cmd, chart.dimension(), dim)));
    }
    Ok(())
}

/// The type name of a REPL value, for error messages.
fn value_type_name(v: &RustMathValue) -> &'static str {
    use RustMathValue::*;
    match v {
        Integer(_) => "Integer",
        Rational(_) => "Rational",
        Complex(_) => "Complex",
        Float(_) => "Float",
        Symbol(_) => "Symbol",
        Expr(_) => "Expr",
        String(_) => "String",
        Bool(_) => "Bool",
        Matrix(_) => "Matrix",
        Polynomial(_) => "Polynomial",
        IntegerMod(_) => "IntegerMod",
        List(_) => "List",
        Graph(_) => "Graph",
        Point2D(_) => "Point2D",
        Point3D(_) => "Point3D",
        Plot { .. } => "Plot",
        CliffordAlg(_) => "CliffordAlg",
        CliffordElem(_) => "CliffordElem",
        ExteriorAlg(_) => "ExteriorAlg",
        JordanSymBilinear(_) => "JordanSymBilinear",
        JordanSymBilinearElem(_) => "JordanSymBilinearElem",
        SpecialJordan(_) => "SpecialJordan",
        SpecialJordanElem(_) => "SpecialJordanElem",
        ExceptionalJordan(_) => "ExceptionalJordan",
        AlbertElem(_) => "AlbertElem",
        Manifold(_) => "Manifold",
        ChartVal(_) => "ChartVal",
        DiffFormVal(_) => "DiffFormVal",
        VectorFieldVal(_) => "VectorFieldVal",
        None => "None",
    }
}

/// The largest time_limit(s): one year.
const TIME_LIMIT_MAX_SECS: u64 = 365 * 86_400;

/// True for the errors of `ReplContext::checkpoint` (S13). Code that tries one evaluation and
/// falls back to another on error passes these on instead.
fn is_cancellation(e: &EvalError) -> bool {
    e.name == "KeyboardInterrupt" || e.name == "TimeoutError"
}

/// The decimal text of n, the same text as `n.to_string()`. num-bigint 0.4.8 converts by divide
/// and conquer with schoolbook division, which is quadratic (9.3 s at 2^25 bits, release); above
/// 2^15 bits the conversion goes through GMP's mpz_get_str (rug), which is subquadratic (Brent and
/// Zimmermann, Modern Computer Arithmetic, 1.7).
fn decimal_string(n: &Integer) -> String {
    if n.bit_length() <= 1 << 15 {
        return n.to_string();
    }
    let mag = rug::Integer::from_digits(&n.as_bigint().magnitude().to_u64_digits(), rug::integer::Order::Lsf);
    if n.signum() < 0 { (-mag).to_string() } else { mag.to_string() }
}

/// The text of rustmath-rationals' Display (`n` or `n/d`, rational.rs:232-240), with both parts
/// converted by `decimal_string`.
fn rational_text(r: &Rational) -> String {
    let num = decimal_string(r.numerator());
    if r.denominator().is_one() { num } else { format!("{}/{}", num, decimal_string(r.denominator())) }
}

/// The text of rustmath-polynomials' Display for a polynomial over Z (univariate.rs:790-824),
/// built from coefficient strings that are already converted (`strs[i]` for the coefficient of
/// x^i; empty for zero coefficients).
fn poly_text(coeffs: &[Integer], strs: &[String]) -> String {
    let mut out = String::new();
    for (i, c) in coeffs.iter().enumerate().rev() {
        if c.is_zero() {
            continue;
        }
        if !out.is_empty() {
            out.push_str(" + ");
        }
        match (i, c.is_one()) {
            (0, _) => out.push_str(&strs[i]),
            (1, true) => out.push('x'),
            (1, false) => { out.push_str(&strs[i]); out.push_str("*x"); }
            (_, true) => { out.push_str("x^"); out.push_str(&i.to_string()); }
            (_, false) => { out.push_str(&strs[i]); out.push_str("*x^"); out.push_str(&i.to_string()); }
        }
    }
    if out.is_empty() { "0".to_string() } else { out }
}

/// An upper bound on the bytes of the text form of a value (D): `bytes`, which stops growing
/// soon after it passes the limit, and the largest Integer that a library formatter (the Display
/// of rustmath-symbolic's Expr, which converts with num-bigint) would print.
#[derive(Clone, Copy, Debug, Default)]
struct DisplaySize {
    bytes: u64,
    library_int_bits: u64,
}

/// Bytes of the decimal text of n, sign included: at most floor(bits·log10 2) + 1 digits.
fn int_text_bytes(n: &Integer) -> u64 {
    (n.bit_length() as f64 * std::f64::consts::LOG10_2) as u64 + 2
}

/// display_size of a value: an upper bound on its text, computed without formatting. The walk
/// stops once the bound passes `limit`, so it costs O(limit) at most (a symbolic expression is a
/// DAG whose text can be exponentially larger than the value).
fn display_size(v: &RustMathValue, limit: u64) -> DisplaySize {
    let mut s = DisplaySize::default();
    let mut stack: Vec<&RustMathValue> = vec![v];
    while let Some(v) = stack.pop() {
        if s.bytes > limit {
            break;
        }
        match v {
            RustMathValue::Integer(n) => s.bytes += int_text_bytes(n),
            RustMathValue::Rational(r) => s.bytes += int_text_bytes(r.numerator()) + int_text_bytes(r.denominator()) + 1,
            RustMathValue::IntegerMod(a) => s.bytes += int_text_bytes(a.value()) + int_text_bytes(a.modulus()) + 8,
            RustMathValue::String(t) => s.bytes += t.len() as u64 + 2,
            RustMathValue::Matrix(m) => {
                s.bytes += 2 + 2 * m.rows() as u64;
                for i in 0..m.rows() {
                    for j in 0..m.cols() {
                        if let Ok(x) = m.get(i, j) {
                            s.bytes += int_text_bytes(x) + 2;
                        }
                    }
                    if s.bytes > limit {
                        break;
                    }
                }
            }
            RustMathValue::Polynomial(p) => {
                // A zero coefficient has no term in the text (review 2 of J1b, m2).
                s.bytes += 1;
                for c in p.coefficients().iter().filter(|c| !c.is_zero()) {
                    s.bytes += int_text_bytes(c) + 24;   // " + ", "*x^", the exponent
                }
            }
            RustMathValue::List(items) => {
                s.bytes += 2 + 2 * items.len() as u64;
                stack.extend(items.iter());
            }
            RustMathValue::Expr(e) => expr_display_size(e, limit, &mut s),
            RustMathValue::Symbol(sym) => s.bytes += sym.name().len() as u64,
            RustMathValue::ChartVal(c) => {
                s.bytes += 8 + c.coordinate_names().iter().map(|n| n.len() as u64 + 2).sum::<u64>();
            }
            // Fixed-size texts (Bool, Float, Complex, points, the algebra and manifold headers,
            // Graph, Plot description). Clifford and Jordan elements are not walked (section 6).
            _ => s.bytes += 256,
        }
    }
    s
}

/// Adds the text bound of a symbolic expression (Display of rustmath-symbolic `Expr`:
/// "(l op r)", "name(x)", "f(a, b)"), iteratively, stopping past `limit`.
fn expr_display_size(e: &Expr, limit: u64, s: &mut DisplaySize) {
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(e) = stack.pop() {
        if s.bytes > limit {
            return;
        }
        match e {
            // Numeric leaves get 2 more bytes: a Display may put parentheses around a negative
            // or rational base of `^` (review 2 of J1b).
            Expr::Integer(n) => {
                s.bytes += int_text_bytes(n) + 2;
                s.library_int_bits = s.library_int_bits.max(n.bit_length());
            }
            Expr::Rational(r) => {
                s.bytes += int_text_bytes(r.numerator()) + int_text_bytes(r.denominator()) + 3;
                s.library_int_bits = s.library_int_bits.max(r.numerator().bit_length()).max(r.denominator().bit_length());
            }
            Expr::Real(_) => s.bytes += 332,      // the longest f64 Display is about 320 characters
            Expr::Symbol(sym) => s.bytes += sym.name().len() as u64,
            Expr::Binary(_, l, r) => {
                s.bytes += 5;
                stack.push(l);
                stack.push(r);
            }
            Expr::Unary(_, x) => {
                s.bytes += 16;
                stack.push(x);
            }
            Expr::Function(name, args) => {
                s.bytes += name.len() as u64 + 2 + 2 * args.len() as u64;
                stack.extend(args.iter().map(|a| &**a));
            }
        }
    }
}

/// The one-line summary of a value that to_display does not print (D).
fn display_summary(v: &RustMathValue, size: &DisplaySize) -> String {
    let what = match v {
        RustMathValue::Integer(n) => {
            return format!("<{}integer of {} bits, about {:.4e} decimal digits: not printed (REPL display limit {} bytes)>",
                if n.signum() < 0 { "negative " } else { "" }, n.bit_length(),
                n.bit_length() as f64 * std::f64::consts::LOG10_2, repl_limits::OUTPUT_BYTES);
        }
        RustMathValue::Rational(r) => format!("rational number with a numerator of {} bits and a denominator of {} bits",
            r.numerator().bit_length(), r.denominator().bit_length()),
        RustMathValue::IntegerMod(a) => format!("integer modulo an integer of {} bits", a.modulus().bit_length()),
        RustMathValue::String(t) => format!("string of {} bytes", t.len()),
        RustMathValue::Matrix(m) => format!("{}x{} matrix with entries of up to {} bits", m.rows(), m.cols(), max_entry_bits(m)),
        RustMathValue::Polynomial(p) => format!("polynomial with {} coefficients of up to {} bits",
            p.coefficients().len(), max_coeff_bits(p)),
        RustMathValue::List(items) => format!("list of {} elements", items.len()),
        RustMathValue::Expr(_) => "symbolic expression".to_string(),
        RustMathValue::ChartVal(c) => format!("chart with {} coordinates", c.dimension()),
        _ => "value".to_string(),
    };
    if size.bytes > repl_limits::OUTPUT_BYTES {
        format!("<{}: its text has more than {} bytes, not printed (REPL display limit)>", what, repl_limits::OUTPUT_BYTES)
    } else {
        format!("<{}: it contains an integer of {} bits, more than {} (REPL display limit for symbolic expressions)>",
            what, size.library_int_bits, repl_limits::LIBRARY_INT_DISPLAY_BITS)
    }
}

/// b^k for k >= 0, refused when the result would have more than POW_BITS bits (S12). Callers
/// pass k = |e| and the exponent `e` as written (for the message; for e < 0 the power is the
/// denominator of b^e). |b| <= 1 needs no power, so k may be any non-negative integer there.
fn checked_int_pow(cmd: &str, b: &Integer, k: &Integer, e: &Integer) -> Result<Integer, EvalError> {
    debug_assert!(k.signum() >= 0, "callers pass |e|");
    if b.is_zero() {
        return Ok(if k.is_zero() { Integer::one() } else { Integer::zero() });
    }
    if b.bit_length() == 1 {
        // b = 1 or b = -1
        let odd = k.as_bigint().bit(0);
        return Ok(if b.signum() < 0 && odd { b.clone() } else { Integer::one() });
    }
    let bits = int_pow_bits(b, k);
    // NaN cannot occur (k >= 0, |b| >= 2), but is refused too.
    if bits.is_nan() || bits > repl_limits::POW_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: b^e for b = {}, e = {} has {}about {:.3e} bits, more than {} (REPL limit)",
            cmd, show_int(b), show_int(e), if e.signum() < 0 { "a denominator of " } else { "" },
            bits, repl_limits::POW_BITS)));
    }
    // |b| >= 2, so k <= bits <= POW_BITS < 2^32.
    let k32 = u32::try_from(k.as_bigint()).expect("k <= POW_BITS");
    Ok(b.pow(k32))
}

/// b^e for an integer b and any integer e (MAGMA ch. 18 `n ^ k`): an Integer for e >= 0; for
/// e < 0 the rational 1 / b^|e| (J1a's semantics; MAGMA, Sage and PARI give `2^-1 = 1/2`), with
/// b^|e| bounded by POW_BITS; 0^e for e < 0 is a ZeroDivisionError. `**`, `^` and `pow` all
/// call it, so they agree for integer bases.
fn int_power_value(cmd: &str, b: &Integer, e: &Integer) -> Result<RustMathValue, EvalError> {
    if e.signum() >= 0 {
        return Ok(RustMathValue::Integer(checked_int_pow(cmd, b, e, e)?));
    }
    if b.is_zero() {
        return Err(EvalError::new("ZeroDivisionError", format!("{}: 0 has no inverse (exponent {})", cmd, show_int(e))));
    }
    let d = checked_int_pow(cmd, b, &(-e.clone()), e)?;
    Rational::new(Integer::one(), d)
        .map(RustMathValue::Rational)
        .map_err(|err| EvalError::new("MathError", format!("{:?}", err)))
}

/// r^e for a rational r = n/d and any integer e (MAGMA ch. 20 `a ^ k`): n^|e| / d^|e|, inverted
/// for e < 0. Refused when the two powers together would have more than POW_BITS bits; 0^e for
/// e < 0 is a ZeroDivisionError. `**`, `^` and `pow` all call it, so they agree for rational bases.
fn rational_power_value(cmd: &str, r: &Rational, e: &Integer) -> Result<RustMathValue, EvalError> {
    let k = if e.signum() < 0 { -e.clone() } else { e.clone() };
    let (n, d) = (r.numerator(), r.denominator());
    if e.signum() < 0 && n.is_zero() {
        return Err(EvalError::new("ZeroDivisionError", format!("{}: 0 has no inverse (exponent {})", cmd, show_int(e))));
    }
    // |x| <= 1 has a power of one bit; otherwise int_pow_bits.
    let bits_of = |x: &Integer| if x.bit_length() <= 1 { 1.0 } else { int_pow_bits(x, &k) };
    let bits = bits_of(n) + bits_of(d);
    // NaN cannot occur (k >= 0, |b| >= 2), but is refused too.
    if bits.is_nan() || bits > repl_limits::POW_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: r^e for r = {}/{}, e = {} has about {:.3e} bits in its numerator and denominator, \
             more than {} (REPL limit)",
            cmd, show_int(n), show_int(d), show_int(e), bits, repl_limits::POW_BITS)));
    }
    // Rational::new reduces n^k / d^k with a gcd although the parts are coprime.
    check_rational_gcd(cmd, bits_of(n) as u64, bits_of(d) as u64)?;
    let (pn, pd) = (checked_int_pow(cmd, n, &k, e)?, checked_int_pow(cmd, d, &k, e)?);
    let (pn, pd) = if e.signum() < 0 { (pd, pn) } else { (pn, pd) };
    Rational::new(pn, pd)
        .map(RustMathValue::Rational)
        .map_err(|err| EvalError::new("MathError", format!("{:?}", err)))
}

/// The bit length of |b|^k for |b| >= 2, k >= 0: floor(k·log2|b|) + 1, as an f64 (+inf when k
/// does not fit an f64). log2_abs reads the top 64 bits, so the relative error is about 1e-16;
/// for a power of two it is exact. At the boundary, gp gives `#binary(3^21170489)` = 33554432
/// and `#binary(3^21170490)` = 33554433, and so does this function.
fn int_pow_bits(b: &Integer, k: &Integer) -> f64 {
    (k.to_f64().unwrap_or(f64::INFINITY) * log2_abs(b)).floor() + 1.0
}

/// `a * b` for integers, refused when the product has more than POW_BITS bits (review 2 of J1b,
/// m3: the same bound as `**`, so `x * x` cannot build what `x ** 2` refuses). The product has
/// bits(a) + bits(b) - 1 or bits(a) + bits(b) bits; the first refuses at once, the second only
/// after the product (of at most POW_BITS + 1 bits) is known.
fn checked_int_product(op: &str, a: &Integer, b: &Integer) -> Result<Integer, EvalError> {
    let (ba, bb) = (a.bit_length(), b.bit_length());
    let refuse = |bits: u64, at_least: bool| EvalError::new("ValueError", format!(
        "{}: the product of integers of {} and {} bits has {}{} bits, more than {} (REPL limit)",
        op, ba, bb, if at_least { "at least " } else { "" }, bits, repl_limits::POW_BITS));
    if ba == 0 || bb == 0 {
        return Ok(Integer::zero());
    }
    if (ba + bb - 1) as f64 > repl_limits::POW_BITS {
        return Err(refuse(ba + bb - 1, true));
    }
    let p = a.clone() * b.clone();
    if p.bit_length() as f64 > repl_limits::POW_BITS {
        return Err(refuse(p.bit_length(), false));
    }
    Ok(p)
}

/// `a op b` for rationals (+, -, *, /): refused when the unreduced numerator and denominator
/// that rustmath-rationals forms may have more than POW_BITS bits in all (an upper bound; the
/// reduced result can be smaller, hence "may"), or when reducing them would take too long
/// (`check_rational_gcd`).
fn check_rational_op(op: &str, a: &Rational, b: &Rational) -> Result<(), EvalError> {
    let (na, da) = (a.numerator().bit_length(), a.denominator().bit_length());
    let (nb, db) = (b.numerator().bit_length(), b.denominator().bit_length());
    // The unreduced parts: n/d for a + b and a - b is (na db ± nb da) / (da db).
    let (n, d) = match op {
        "*" => (na + nb, da + db),
        "/" => (na + db, da + nb),
        _ => ((na + db).max(nb + da) + 1, da + db),
    };
    if (n + d) as f64 > repl_limits::POW_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the result may have {} bits in its numerator and denominator, more than {} (REPL limit)",
            op, n + d, repl_limits::POW_BITS)));
    }
    check_rational_gcd(op, n, d)
}

/// Every rustmath-rationals operation reduces its result with a gcd (`Rational::new`,
/// Lehmer's algorithm in rustmath-integers), which is quadratic: measured (release, J1b)
/// Rational(a, b) with a, b of 2^20, 2^21 and 2^22 bits: 1.8, 7.0 and 27.7 s. The estimate
/// GCD_SECONDS_PER_BIT2 · bits(n) · bits(d) is above all three; for unequal sizes the first
/// division step dominates and costs less. Refused above PRODUCT_SECONDS. This also bounds
/// r^e, whose parts are already coprime: rustmath-rationals has no constructor that skips the
/// reduction (a handoff of unit J1b).
fn check_rational_gcd(cmd: &str, nbits: u64, dbits: u64) -> Result<(), EvalError> {
    let t = GCD_SECONDS_PER_BIT2 * nbits as f64 * dbits as f64;
    if t > repl_limits::PRODUCT_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "{}: reducing a fraction with a numerator of {} bits and a denominator of {} bits (a quadratic \
             gcd in rustmath-rationals) would take about {:.1} s (estimate), more than {} s (REPL limit)",
            cmd, nbits, dbits, t, repl_limits::PRODUCT_SECONDS)));
    }
    Ok(())
}
const GCD_SECONDS_PER_BIT2: f64 = 1.7e-12;

/// The largest bit length of an entry.
fn max_entry_bits(m: &Matrix<Integer>) -> u64 {
    let mut best = 0;
    for i in 0..m.rows() {
        for j in 0..m.cols() {
            if let Ok(v) = m.get(i, j) {
                best = best.max(v.bit_length());
            }
        }
    }
    best
}

/// Estimated seconds of `Matrix::mul` (rustmath-matrix, schoolbook, one clone pair and one
/// multiplication per inner step) on an (r x k)·(k x c) product whose operands have entries of
/// at most `bits` bits: (MUL_PER_PRODUCT_NS + r·k·c·(MUL_STEP_NS + MUL_WORD_NS·w^1.465)) ns with
/// w = ceil(bits/64) words (1.465 = log 5 / log 3, the Toom-3 exponent of num-bigint; Knuth,
/// TAOCP vol. 2, 4.3.3). The per-product term covers one product's trace, its two entry scans
/// and its allocation, so the sum of the estimates also bounds the loop of
/// `checked_matrix_power`. It charges w_max^1.465 per step also when one operand has one-word
/// entries (R·M in matrix_power, A * identity(n)), which over-estimates those products: it errs
/// towards refusing.
///
/// Fit (release, this container; ns per inner step, d^3 steps, entries of b bits):
///
/// | d, b       | 200, 8 | 100, 64 | 100, 256 | 60, 1024 | 40, 4096 | 20, 16384 | 12, 65536 | 6, 262144 | 4, 2^20 | 2, 2^22 |
/// |------------|--------|---------|----------|----------|----------|-----------|-----------|-----------|---------|---------|
/// | measured   | 33.7   | 45      | 92       | 297      | 2843     | 29204     | 2.5e5     | 1.9e6     | 2.0e7   | 1.3e8   |
/// | model      | 44     | 44      | 110      | 614      | 4460     | 33860     | 2.6e5     | 2.0e6     | 1.5e7   | 1.2e8   |
///
/// Per product, entries that do not grow (`matrix_power` at e = 2^k): 1x1 100 ns, 2x2 390 ns,
/// 3x3 1070 ns, 4x4 2340 ns, 8x8 37.5 ns a step, 32x32 34 ns a step. Large one-word products
/// (J1b ladder `mmul`, `identity(d) * identity(d)` end to end): d = 300 to 550 at 30-34 ns a
/// step, d = 575, 584, 600, 650, 700 at 55.6, 47.1, 46.4, 48.6, 49.6 ns a step (10.6, 9.4, 10.0,
/// 13.4, 17.0 s; the operands no longer fit the caches, and the times vary by about 10 %
/// between runs). MUL_STEP_NS = 46 gives 56 ns a step at one word, so the largest admitted
/// product of one-word matrices is 563 x 563 (estimate 9.99 s; 560 x 560 took 6.1 s). The matrix_power
/// ladder (`mpow`) stays under the estimates: [[1]] to the 2^33554431 2.6 s (estimate 5.2 s),
/// [[0,1],[1,0]] to the 2^20000000 6.0 s (11.0 s, refused by the floor), identity(8) to the
/// 2^300000 4.6 s (8.6 s), identity(32) to the 2^4000 3.5 s (7.3 s), the Fibonacci matrix to
/// the 1.2·10^7 2.1 s.
fn matrix_product_seconds(r: usize, k: usize, c: usize, bits: u64) -> f64 {
    let w = bits.max(1).div_ceil(64) as f64;
    (MUL_PER_PRODUCT_NS + (r as f64) * (k as f64) * (c as f64) * (MUL_STEP_NS + MUL_WORD_NS * w.powf(1.465))) * 1e-9
}
/// The constants of matrix_product_seconds (ns): per product, per inner step, per inner step
/// and word^1.465 (the fit above).
const MUL_PER_PRODUCT_NS: f64 = 100.0;
const MUL_STEP_NS: f64 = 46.0;
const MUL_WORD_NS: f64 = 10.0;

/// ceil(log2 k) for k >= 1 (0 for k <= 1): a sum of k terms below 2^b is below 2^(b + this).
fn ceil_log2(k: usize) -> u64 {
    if k <= 1 { 0 } else { (usize::BITS - (k - 1).leading_zeros()) as u64 }
}

/// Refuses a product A·B (A: r x k, B: k x c) whose entries may exceed RESULT_BITS in total, or
/// whose estimated time, added to the `spent` seconds already estimated for this call, exceeds
/// PRODUCT_SECONDS (S12). With |a| < 2^bA and |b| < 2^bB an entry of A·B is below
/// k·2^(bA + bB), so it has at most bA + bB + ceil(log2 k) bits (a rigorous size bound; "may"
/// because cancellation can make the true size smaller). Returns the estimated seconds.
fn check_matrix_product(cmd: &str, a: &Matrix<Integer>, b: &Matrix<Integer>, spent: f64) -> Result<f64, EvalError> {
    let (ba, bb) = (max_entry_bits(a), max_entry_bits(b));
    let k = a.cols();
    let entry_bits = ba + bb + ceil_log2(k);
    let total = entry_bits as f64 * a.rows() as f64 * b.cols() as f64;
    if total > repl_limits::RESULT_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the product of a {}x{} and a {}x{} matrix with entries of up to {} and {} bits may have \
             entries of {} bits, {:.3e} bits in all, more than {} (REPL limit)",
            cmd, a.rows(), k, k, b.cols(), ba, bb, entry_bits, total, repl_limits::RESULT_BITS)));
    }
    let secs = matrix_product_seconds(a.rows(), k, b.cols(), ba.max(bb));
    if spent + secs > repl_limits::PRODUCT_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the products would take about {:.1} s (estimate), more than {} s (REPL limit)",
            cmd, spent + secs, repl_limits::PRODUCT_SECONDS)));
    }
    Ok(secs)
}

/// Refuses k·A for a matrix or polynomial with `count` entries of at most `bits` bits when the
/// result may have more than RESULT_BITS bits in all (an entry of k·a has at most bits(k) + bits(a)
/// bits). One multiplication per entry, so the size also bounds the time (review 1, m8).
fn check_scalar_product(cmd: &str, what: &str, count: usize, bits: u64, k: &Integer) -> Result<(), EvalError> {
    let entry_bits = bits + k.bit_length();
    let total = entry_bits as f64 * count as f64;
    if total > repl_limits::RESULT_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: {} with {} entries of up to {} bits times an integer of {} bits may have {:.3e} bits \
             in all, more than {} (REPL limit)", cmd, what, count, bits, k.bit_length(), total, repl_limits::RESULT_BITS)));
    }
    Ok(())
}

/// The largest bit length of a coefficient.
fn max_coeff_bits(p: &UnivariatePolynomial<Integer>) -> u64 {
    p.coefficients().iter().map(|c| c.bit_length()).max().unwrap_or(0)
}

/// Refuses p·q (schoolbook, (m+1)(n+1) coefficient products, rustmath-polynomials Mul) when the
/// result may have more than RESULT_BITS bits in all, or when its estimated time
/// matrix_product_seconds(1, m+1, n+1, max(bp, bq))·POLY_STEP_FACTOR exceeds PRODUCT_SECONDS.
/// A coefficient of p·q is a sum of at most min(m, n) + 1 products, so it has at most
/// bp + bq + ceil(log2(min(m, n) + 1)) bits; there are m + n + 1 of them. Measured (J1b ladder
/// `pmul`, release, unit coefficients, `Q = P * P`): 2000 terms 0.14 s, 5000 0.70 s, 10000
/// 2.80 s, 12000 4.05 s, 15000 6.48 s, 28-35 ns per coefficient product against 44 ns in the
/// model.
fn check_poly_product(cmd: &str, p: &UnivariatePolynomial<Integer>, q: &UnivariatePolynomial<Integer>) -> Result<(), EvalError> {
    let (lp, lq) = (p.coefficients().len(), q.coefficients().len());
    let (bp, bq) = (max_coeff_bits(p), max_coeff_bits(q));
    let short = lp.min(lq);
    let entry_bits = bp + bq + ceil_log2(short);
    let total = entry_bits as f64 * (lp + lq).saturating_sub(1) as f64;
    if total > repl_limits::RESULT_BITS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the product of polynomials with {} and {} coefficients of up to {} and {} bits may have \
             {:.3e} bits in all, more than {} (REPL limit)", cmd, lp, lq, bp, bq, total, repl_limits::RESULT_BITS)));
    }
    let secs = matrix_product_seconds(1, lp, lq, bp.max(bq)) * POLY_STEP_FACTOR;
    if secs > repl_limits::PRODUCT_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the product of polynomials with {} and {} coefficients would take about {:.1} s (estimate), \
             more than {} s (REPL limit)", cmd, lp, lq, secs, repl_limits::PRODUCT_SECONDS)));
    }
    Ok(())
}
/// Polynomial products cost this many times a matrix inner step: 44 ns per one-word
/// coefficient product, which errs towards refusing by a factor of about 1.5 (fit above).
const POLY_STEP_FACTOR: f64 = 0.79;

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

/// Estimated seconds of a balanced product of k factors of at most `top_bits` bits (C1's
/// product_range_u64 / product_tree; J. von zur Gathen, J. Gerhard, Modern Computer Algebra,
/// 3rd ed., 10.1): FACT_FACTOR_NS·k + FACT_WORD_NS·W^1.465 ns with W = k·top_bits/64 words,
/// the size of the result; the top levels of the tree cost about M(W) and dominate (1.465 =
/// log 5 / log 3, num-bigint's Toom-3). All in f64, so no overflow for k up to u64::MAX. The
/// result has at most k·top_bits bits, so the time bound also bounds its memory.
fn factorial_product_seconds(k: u64, top_bits: u64) -> f64 {
    let w = k as f64 * top_bits.max(1) as f64 / 64.0;
    (FACT_FACTOR_NS * k as f64 + FACT_WORD_NS * w.powf(1.465)) * 1e-9
}
/// The constants of factorial_product_seconds (ns), release fit. J1b ladder `falling` (end to
/// end, the display included up to 5.57·10^7 bits): falling(2^32 - 1, k) at k = 10^5, 10^6,
/// 1.5·10^6, 2·10^6: 0.15, 3.7, 7.0, 7.6 s (estimates 0.12, 3.4, 6.1, 9.3 s); falling(2^64, 10^6)
/// 8.0 s (9.5 s); falling(2^1000, 10^5) 16.1 s (17.8 s, refused); rising(1, 3·10^6) 6.9 s
/// (9.6 s); rising(1, 5·10^6) 15.5 s (25 s, refused).
const FACT_FACTOR_NS: f64 = 20.0;
const FACT_WORD_NS: f64 = 15.0;

/// Refuses a falling or rising factorial whose product would take more than PRODUCT_SECONDS
/// (estimate).
fn check_factorial_product(cmd: &str, k: u64, top_bits: u64) -> Result<(), EvalError> {
    let t = factorial_product_seconds(k, top_bits);
    if t > repl_limits::PRODUCT_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the product of {} factors of up to {} bits would take about {:.1} s (estimate), more than {} s \
             (REPL limit)", cmd, k, top_bits, t, repl_limits::PRODUCT_SECONDS)));
    }
    Ok(())
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

/// The three row recurrences of C1's `band_rows` (rustmath-combinatorics lib.rs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StirlingKind {
    /// `stirling_first`: [i j] = (i-1) [i-1 j] + [i-1 j-1].
    Stirling1,
    /// `stirling_second`: {i j} = j {i-1 j} + {i-1 j-1}.
    Stirling2,
    /// `eulerian`: <i j> = (j+1) <i-1 j> + (i-j) <i-1 j-1>, run on k' = min(k, n - 1 - k).
    Eulerian,
}

/// log2(n!) by Stirling's series with the term 1/(12n) (0 for n <= 1). Its error is below
/// 1/(360 n^3) in the natural log, far below one bit.
fn log2_factorial(n: u64) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let x = n as f64;
    (x * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI * x).ln() + 1.0 / (12.0 * x)) / std::f64::consts::LN_2
}

/// log2 C(n, m) for m <= n, from `log2_factorial`.
fn log2_binomial(n: u64, m: u64) -> f64 {
    (log2_factorial(n) - log2_factorial(m) - log2_factorial(n - m)).max(0.0)
}

/// The band of (n, k) in C1's `band_rows`: the number of big-integer steps, the width w (the
/// rolling buffer has at most w + 1 entries), and an a-priori upper bound B on the bits of the
/// result, which bounds every entry of the band (the recurrences have positive coefficients, so
/// each entry is at most the final value). None for the cases the callee answers at once
/// (stirling: k = 0, k >= n; eulerian: k >= n or k' = 0).
///
/// Steps: k (n - k + 1) for Stirling (j in 1..=k, i in j..=j + n - k), (k' + 1)(n - k') for
/// Eulerian. Bounds on the result, by kind:
/// - [n k] = (n-1)! e_{k-1}(1, 1/2, ..., 1/(n-1)), and e_m(x) <= min(H^m / m!, C(N, m) / m!) for
///   N terms in (0, 1] with sum H <= ln(n - 1) + 1 and largest product 1/m!. So
///   log2 [n k] <= log2 (n-1)! - log2 (k-1)! + min((k-1) log2 H, log2 C(n-1, k-1)).
/// - {n k} <= min(k^n / k!, C(n, k) k^(n-k)) (surjections; B. C. Rennie, A. J. Dobson,
///   J. Combin. Theory 7 (1969) 116-121).
/// - <n k'> <= min((k'+1)^n, n!) (Worpitzky's identity at x = k' + 1 has the term <n, n-1-k'>
///   = <n k'> with coefficient 1).
fn stirling_band(kind: StirlingKind, n: u32, k: u32) -> Option<(f64, u64, f64)> {
    let (n, k) = (u64::from(n), u64::from(k));
    match kind {
        StirlingKind::Stirling1 | StirlingKind::Stirling2 => {
            if k == 0 || k >= n {
                return None;
            }
            let steps = k as f64 * (n - k + 1) as f64;
            let w = k.min(n - k + 1);
            let bits = if kind == StirlingKind::Stirling1 {
                let h = if n >= 2 { ((n - 1) as f64).ln() + 1.0 } else { 1.0 };
                log2_factorial(n - 1) - log2_factorial(k - 1)
                    + ((k - 1) as f64 * h.log2()).min(log2_binomial(n - 1, k - 1))
            } else {
                let lk = (k as f64).log2();
                (n as f64 * lk - log2_factorial(k)).min(log2_binomial(n, k) + (n - k) as f64 * lk)
            };
            Some((steps, w, bits.max(1.0)))
        }
        StirlingKind::Eulerian => {
            if k >= n {
                return None;
            }
            let k1 = k.min(n - 1 - k);
            if k1 == 0 {
                return None;
            }
            let steps = (k1 + 1) as f64 * (n - k1) as f64;
            let bits = (n as f64 * ((k1 + 1) as f64).log2()).min(log2_factorial(n));
            Some((steps, k1 + 1, bits.max(1.0)))
        }
    }
}

/// Estimated seconds (release) of `stirling_first`, `stirling_second` or `eulerian` from
/// `stirling_band`: steps · (c0 + c1 · ceil(B / 64)) ns. One step multiplies an entry by a
/// small integer and adds another, so its cost is linear in the words of the entries; the
/// entries grow along the band, so c1 also absorbs the ratio of their mean size to B.
fn stirling_seconds(kind: StirlingKind, steps: f64, bits: f64) -> f64 {
    let (c0, c1) = match kind {
        StirlingKind::Stirling1 => STIRLING1_NS,
        StirlingKind::Stirling2 => STIRLING2_NS,
        StirlingKind::Eulerian => EULERIAN_NS,
    };
    steps * (c0 + c1 * (bits / 64.0).ceil().max(1.0)) * 1e-9
}
/// (c0, c1) of `stirling_seconds` in ns, per kind: the smallest round constants under which
/// the estimate is above every measured time of the J1b ladder (release, this container; 21
/// (n, k) per kind, from (1000, 500) to (200000, 2) and (20000, 19900)). Measured ns per step
/// against W = ceil(B/64): stirling1 80 at W = 90, 685 at 662, 1543 at 1770, 43800 at 50522;
/// stirling2 329 at 281, 617 at 609, 1129 at 1424; eulerian 857 at 474, 1182 at 848, 3868 at
/// 3115. One constant pair cannot fit both the centre and the edges of a row with these a-priori
/// bounds, so the fit follows the centre (the worst admitted region) and over-estimates the
/// edges (stirling2(200000, 2): 0.15 s, estimated 1.3 s). At the boundary:
/// stirling1(6000, 3000) 6.17 s (estimate 6.7 s), stirling1(10000, 1000) 13.9 s (17.2 s, refused),
/// stirling1(100000, 3) 5.86 s (7.5 s), stirling1(200000, 2) 17.5 s (21.2 s, refused),
/// stirling2(10000, 1000) 10.2 s (13.9 s, refused), eulerian(5000, 2500) 7.39 s (8.9 s),
/// eulerian(6000, 3000) 12.3 s (15.4 s, refused). Memory is small: the band of
/// stirling1(6000, 3000) peaks at 12 MB.
const STIRLING1_NS: (f64, f64) = (50.0, 1.05);
const STIRLING2_NS: (f64, f64) = (50.0, 1.05);
const EULERIAN_NS: (f64, f64) = (150.0, 1.5);

/// Refuses stirling1, stirling2 or eulerian when `stirling_seconds` exceeds STIRLING_SECONDS or
/// the rolling buffer, (w + 1) entries of at most B bits, would exceed TABLE_BYTES.
fn check_stirling(cmd: &str, kind: StirlingKind, n: u32, k: u32) -> Result<(), EvalError> {
    let Some((steps, w, bits)) = stirling_band(kind, n, k) else {
        return Ok(());
    };
    let t = stirling_seconds(kind, steps, bits);
    if t > repl_limits::STIRLING_SECONDS {
        return Err(EvalError::new("ValueError", format!(
            "{}: the recurrence for n = {}, k = {} takes {:.3e} steps on numbers of up to {:.0} bits, about {:.1} s \
             (estimate), more than {} s (REPL limit)", cmd, n, k, steps, bits, t, repl_limits::STIRLING_SECONDS)));
    }
    let bytes = (w as f64 + 1.0) * (bits / 8.0 + 32.0);
    if bytes > repl_limits::TABLE_BYTES as f64 {
        return Err(EvalError::new("ValueError", format!(
            "{}: the rolling rows for n = {}, k = {} need about {:.3e} bytes, more than {} (REPL limit, a memory bound)",
            cmd, n, k, bytes, repl_limits::TABLE_BYTES)));
    }
    Ok(())
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
/// (`TensorField::components`, `NoComponentsInChart`), so such a call
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

/// `is_prime`, `next_prime` and `sqrt_mod` prove primality (ECPP, rustmath-integers `ecpp`) only
/// up to this many bits: a proof takes about 3.3 s at 1024 bits and 21 s at 1536 bits in a
/// release build (I1 design §2.3.7, random primes), and a proof is one library call that an
/// interrupt does not stop. A number that Baillie-PSW refutes is decided at every size (`false`
/// is a proof).
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

/// The limits of `sqrt_mod`'s factorization (see [`sqrt_mod_factor`]). Production code uses
/// [`SQRT_MOD_LIMITS`]; the unit tests build smaller ones.
struct SqrtModLimits {
    /// Phase 1: trial division by the primes below this.
    trial: u64,
    /// Phase 2: Pollard rho iterations per composite (I2 `pollard_rho_limit`).
    rho_steps: u64,
    /// Phase 2 runs only on cofactors of at most this many bits.
    i2_max_bits: u64,
    /// The quadratic sieve on composites of at most this many digits (I2 `mpqs_limit`; < 25:
    /// none).
    qs_digits: u32,
    /// Phase 3: modelled nanoseconds of trial division; the prime bound is
    /// T(bits) = trial_work / (55 + bits / 11) in u64 integer arithmetic.
    trial_work: u64,
    /// Probable primes of at most this many bits get an ECPP proof.
    proof_bits: u64,
}

/// The production limits of `sqrt_mod` (constants in `repl_limits`).
const SQRT_MOD_LIMITS: SqrtModLimits = SqrtModLimits {
    trial: repl_limits::SQRT_MOD_TRIAL,
    rho_steps: repl_limits::SQRT_MOD_RHO_STEPS,
    i2_max_bits: repl_limits::SQRT_MOD_I2_MAX_BITS,
    qs_digits: repl_limits::SQRT_MOD_QS_DIGITS,
    trial_work: repl_limits::SQRT_MOD_TRIAL_WORK,
    proof_bits: repl_limits::SQRT_MOD_PROOF_BITS,
};

/// One factor q^e of n for `sqrt_mod`; the bases of all pieces are pairwise coprime and the
/// product of the pieces is |n|.
#[derive(Clone, Debug, PartialEq, Eq)]
enum SqrtModPiece {
    /// p proven prime (trial division, the twelve strong tests below psi_12, or ECPP).
    Prime { p: Integer, e: u32 },
    /// q passed Baillie-PSW and was not proven prime (`why`).
    Probable { q: Integer, e: u32, why: String },
    /// c proven composite, or not tested, and not split within the limits (`why`).
    Composite { c: Integer, e: u32, why: String },
}

/// Phase 3 trial division bound: T(bits) = work / (55 + bits / 11), in u64 integer arithmetic.
/// 55 + bits/11 ns is the measured release cost of one trial-division step (J1a design §7.1:
/// 61 ns at 64 bits, 103 ns at 512, 834 ns at 8192).
fn sqrt_mod_trial_bound(work: u64, bits: u64) -> u64 {
    work / (55 + bits / 11)
}

/// The factorization of |n| > 1 behind `sqrt_mod`, within `lim` (MAGMA `Factorization` with
/// `TrialDivisionLimit`, `PollardRhoLimit`, `MPQSLimit` and `Proof`, ch. 18 §18.10.1).
///
/// 1. Trial division by the primes below `lim.trial` (stopping when p^2 exceeds the cofactor).
/// 2. The cofactor m: prime if the loop stopped at p^2 > m or m < trial^2 (every smaller prime
///    was tried). Else, if it has at most `lim.i2_max_bits` bits, rustmath-integers
///    `factorization_with_limits` (Brent rho with `rho_steps` iterations, the quadratic sieve up
///    to `qs_digits` digits, no proof): its primes are kept, its probable primes go to step 4,
///    its composites to step 3. A larger m goes to step 3 untested.
/// 3. Each composite or untested piece c: trial division by the primes from `lim.trial` while
///    p <= T(bits of the cofactor) ([`sqrt_mod_trial_bound`]; the bound grows as the cofactor
///    shrinks) and the modelled cost of the steps taken stays within `lim.trial_work` (review 3
///    of J1a, m5). A cofactor below the square of the last prime tried is prime. A composite of
///    step 2 with no factor found is not given to I2 again (it already ran rho and the sieve on
///    it); a proper or untested cofactor of at most `i2_max_bits` bits is classified by
///    Baillie-PSW and sieved again (no rho).
/// 4. Each probable prime of at most `lim.proof_bits` bits gets an ECPP proof (`try_is_prime`);
///    a larger one, or an unresolved proof, stays `Probable`.
///
/// Every step keeps the product of the pieces equal to |n| and splits a piece only into
/// divisors of it, so the bases stay pairwise coprime.
fn sqrt_mod_factor(n: &Integer, lim: &SqrtModLimits) -> Vec<SqrtModPiece> {
    use std::ops::ControlFlow;
    let zero = BigInt::from(0u32);
    let mut pieces = Vec::new();
    let mut m = n.abs();
    debug_assert!(m > Integer::one());
    // Phase 1.
    let mut below_sqrt = false;
    rustmath_integers::sieve::for_each_prime(2, lim.trial, |p| {
        let pi = Integer::from(p);
        if pi.clone() * pi.clone() > m {
            below_sqrt = true;
            return ControlFlow::Break(());
        }
        let mut e = 0u32;
        while m.as_bigint() % p == zero {
            m = Integer::new(m.as_bigint() / p);
            e += 1;
        }
        if e > 0 {
            pieces.push(SqrtModPiece::Prime { p: pi, e });
        }
        ControlFlow::Continue(())
    });
    if m.is_one() {
        return pieces;
    }
    // Phase 2.
    let trial_sq = Integer::from(lim.trial) * Integer::from(lim.trial);
    let mut probable: Vec<(Integer, u32)> = Vec::new();
    // (c, e, whether I2 already classified c as composite and ran rho and the sieve on it)
    let mut rest: Vec<(Integer, u32, bool)> = Vec::new();
    if below_sqrt || m < trial_sq {
        pieces.push(SqrtModPiece::Prime { p: m, e: 1 });
    } else if m.bits() <= lim.i2_max_bits {
        let f = factorization_with_limits(&m, &FactorizationLimits {
            trial_division_limit: 0,
            pollard_rho_limit: lim.rho_steps,
            mpqs_limit: Some(lim.qs_digits),
            proof: false,
        });
        pieces.extend(f.primes.into_iter().map(|(p, e)| SqrtModPiece::Prime { p, e }));
        probable.extend(f.probable_primes);
        rest.extend(f.composites.into_iter().map(|(c, e)| (c, e, true)));
    } else {
        rest.push((m, 1, false));
    }
    // Phase 3.
    let start = lim.trial.max(2);
    for (c, e, classified) in rest {
        let mut cur = c.clone();
        let mut sqrt_cap = isqrt_cap(&cur);
        let mut used = 0u64;
        let mut bound = sqrt_mod_trial_bound(lim.trial_work, cur.bits());
        let mut proven_prime = false;
        rustmath_integers::sieve::for_each_prime(start, lim.trial_work / 55 + 1, |p| {
            if p > sqrt_cap {
                proven_prime = true;
                return ControlFlow::Break(());
            }
            let step = 55 + cur.bits() / 11;
            bound = lim.trial_work / step;
            if p > bound {
                return ControlFlow::Break(());
            }
            used = used.saturating_add(step);
            if used > lim.trial_work {
                bound = p - 1;
                return ControlFlow::Break(());
            }
            let mut f = 0u32;
            while cur.as_bigint() % p == zero {
                cur = Integer::new(cur.as_bigint() / p);
                f += 1;
            }
            if f > 0 {
                pieces.push(SqrtModPiece::Prime { p: Integer::from(p), e: e * f });
                sqrt_cap = isqrt_cap(&cur);
            }
            ControlFlow::Continue(())
        });
        if cur.is_one() {
            continue;
        }
        if proven_prime {
            pieces.push(SqrtModPiece::Prime { p: cur, e });
        } else if cur == c && classified {
            pieces.push(SqrtModPiece::Composite { c: cur, e, why: format!(
                "no prime factor up to {}, not split by Pollard rho ({} steps) or the quadratic sieve (up to {} digits)",
                bound, lim.rho_steps, lim.qs_digits) });
        } else if cur.bits() <= lim.i2_max_bits {
            let f = factorization_with_limits(&cur, &FactorizationLimits {
                trial_division_limit: 0,
                pollard_rho_limit: 0,
                mpqs_limit: Some(lim.qs_digits),
                proof: false,
            });
            pieces.extend(f.primes.into_iter().map(|(p, f)| SqrtModPiece::Prime { p, e: e * f }));
            probable.extend(f.probable_primes.into_iter().map(|(q, f)| (q, e * f)));
            pieces.extend(f.composites.into_iter().map(|(c, f)| SqrtModPiece::Composite { c, e: e * f, why: format!(
                "no prime factor up to {}, not split by the quadratic sieve", bound) }));
        } else {
            let bits = cur.bits();
            pieces.push(SqrtModPiece::Composite { c: cur, e, why: format!(
                "{} bits with no prime factor up to {}; testing it exceeds the REPL limit ({} bits)",
                bits, bound, lim.i2_max_bits) });
        }
    }
    // Step 4: proofs.
    for (q, e) in probable {
        if q.bits() > lim.proof_bits {
            pieces.push(SqrtModPiece::Probable { q, e, why: format!(
                "above the REPL proof limit of {} bits", lim.proof_bits) });
            continue;
        }
        match try_is_prime(&q) {
            Ok(true) => pieces.push(SqrtModPiece::Prime { p: q, e }),
            Ok(false) => pieces.push(SqrtModPiece::Composite { c: q, e, why: "a BPSW pseudoprime".to_string() }),
            Err(err) => pieces.push(SqrtModPiece::Probable { q, e, why: format!("ECPP unresolved: {}", err) }),
        }
    }
    pieces
}

/// floor(sqrt(c)) as a u64, or u64::MAX when c has more than 128 bits (then every u64 prime p
/// has p^2 < c).
fn isqrt_cap(c: &Integer) -> u64 {
    if c.bits() > 128 {
        return u64::MAX;
    }
    c.sqrt().ok().and_then(|r| r.to_u64()).unwrap_or(u64::MAX)
}

/// The text of the REPL command `sqrt_mod(a, n)`: the square roots of a in Z/nZ, within the
/// production limits [`SQRT_MOD_LIMITS`].
///
/// MAGMA: `Modsqrt(n, m)` (Handbook ch. 18 §18.12.1); `IsSquare`, `Sqrt`, `AllSqrts` in Z/mZ
/// (ch. 19 §19.5.6, with the optional `Factorization` parameter "to speed up computation when m
/// is large"; worked example H19E4: the 12 roots of 1404 in Z/2340Z).
///
/// Output, with b = a mod n:
/// - `"{r}; square roots of {b} mod {n}: [r_1, ..., r_N]"` when b has N <= 16 roots, listed in
///   ascending order; r is the CRT of the per-prime-power least roots (the same root as
///   `IntegerMod::sqrt`, e.g. 7 for 4 mod 15);
/// - `"{r}; {b} has {N} square roots mod {n} (not listed)"` when N > 16;
/// - `"no square root: {b} is not a square mod {n}"`, which is always a proof.
///
/// When a factor q of n is a Baillie-PSW probable prime that was not proven prime, the text
/// ends with a clause saying so: the list is complete if q is prime.
fn sqrt_mod_report(a: &Integer, n: &Integer) -> Result<String, EvalError> {
    sqrt_mod_report_with(a, n, &SQRT_MOD_LIMITS)
}

/// [`sqrt_mod_report`] with explicit limits.
///
/// Algorithm (the roots mod n are the CRT combinations of the roots mod the pieces, since the
/// pieces are pairwise coprime with product n):
/// 1. n <= 0 is an error; n = 1 gives the single root 0 (Z/1Z = {0}).
/// 2. If the Jacobi symbol (b/m) of the odd part m of n is -1, there is no root (a proof: the
///    Jacobi symbol of a square unit is 1), and n is not factored.
/// 3. n is factored once by [`sqrt_mod_factor`]: trial division, Brent rho and the quadratic
///    sieve (rustmath-integers I2), ECPP proofs (I1), within `lim`.
/// 4. A piece with odd base q and (b/q) = -1, or a proven prime power p^e with no root
///    (`square_root_mod_prime_power`, rustmath-finitefields), proves that there is no root.
/// 5. Otherwise an unsplit composite piece makes the answer undecided here: a ValueError that
///    names it (the roots need its factorization).
/// 6. A probable prime q^e with e >= 2 is a ValueError (Hensel lifting needs q prime); with
///    e = 1 the roots mod q come from `Integer::try_sqrt_mod_prime` (Tonelli-Shanks, each root
///    squared back), and a failure there proves q composite.
/// 7. The count is the product of the per-prime-power counts
///    (`num_square_roots_mod_prime_power`); r is the CRT of the least roots; at most 16 roots are
///    listed (`all_square_roots_mod_prime_power` per prime power, combined by CRT).
///
/// References: H. Cohen, GTM 138, Algorithms 1.5.1 (Tonelli-Shanks) and §1.6 (Hensel lifting),
/// §1.3.3 (CRT); R. P. Brent, *An improved Monte Carlo factorization algorithm*, BIT 20 (1980)
/// 176-184; A. O. L. Atkin, F. Morain, *Elliptic curves and primality proving*, Math. Comp. 61
/// (1993) 29-68.
fn sqrt_mod_report_with(a: &Integer, n: &Integer, lim: &SqrtModLimits) -> Result<String, EvalError> {
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
    let no_root = || Ok(format!("no square root: {} is not a square mod {}", b, n));
    let two = Integer::from(2);
    // 2. The odd part of n.
    let mut odd = n.clone();
    while odd.is_even() {
        odd = odd / two.clone();
    }
    if !odd.is_one() && b.kronecker_symbol(&odd) == -1 {
        return no_root();
    }
    // 3. and 4.
    let pieces = sqrt_mod_factor(n, lim);
    for piece in &pieces {
        let q = match piece {
            SqrtModPiece::Prime { p, .. } => p,
            SqrtModPiece::Probable { q, .. } => q,
            SqrtModPiece::Composite { c, .. } => c,
        };
        if q.is_odd() && b.modulo(q).kronecker_symbol(q) == -1 {
            return no_root();
        }
    }
    for piece in &pieces {
        if let SqrtModPiece::Prime { p, e } = piece {
            if square_root_mod_prime_power(&b, p, *e as usize).is_none() {
                return no_root();
            }
        }
    }
    // 5. The unfactored part is printed in full up to 1024 bits (the user may factor it
    // elsewhere and supply the roots), above that by its size.
    let full = |v: &Integer| if v.bits() <= 1024 { v.to_string() } else { show_int(v) };
    for piece in &pieces {
        if let SqrtModPiece::Composite { c, why, .. } = piece {
            return Err(EvalError::new("ValueError", format!(
                "sqrt_mod: n = {} is not factored completely within the REPL limits: {} ({}); the square \
                 roots mod n need its factorization",
                full(n), full(c), why)));
        }
    }
    // 6. and 7.: the roots of each piece (as a list when short, else only its least root and
    // count).
    struct Local {
        modulus: Integer,
        least: Integer,
        count: Integer,
        all: Option<Vec<Integer>>,
    }
    let list_limit = Integer::from(SQRT_MOD_LIST_LIMIT);
    let mut locals = Vec::new();
    let mut caveats = Vec::new();
    for piece in &pieces {
        match piece {
            SqrtModPiece::Prime { p, e } => {
                let k = *e as usize;
                let count = num_square_roots_mod_prime_power(&b, p, k);
                let least = square_root_mod_prime_power(&b, p, k).expect("checked in step 4");
                let all = (count <= list_limit).then(|| all_square_roots_mod_prime_power(&b, p, k));
                locals.push(Local { modulus: p.pow(*e), least, count, all });
            }
            SqrtModPiece::Probable { q, e, why } => {
                let shown = if q == n { "n".to_string() } else { full(q) };
                if *e >= 2 {
                    return Err(EvalError::new("ValueError", format!(
                        "sqrt_mod: n has the factor {}^{}, and {} is a BPSW probable prime that was not \
                         proven prime ({}); the roots mod {}^{} need a proof",
                        shown, e, shown, why, shown, e)));
                }
                let bq = b.modulo(q);
                let roots = if bq.is_zero() {
                    vec![Integer::zero()]
                } else {
                    match bq.try_sqrt_mod_prime(q) {
                        Ok(Some(r)) => {
                            let s = q.clone() - r.clone();
                            if r < s { vec![r, s] } else { vec![s, r] }
                        }
                        Ok(None) => return no_root(),
                        Err(_) => {
                            return Err(EvalError::new("ValueError", format!(
                                "sqrt_mod: {} passed BPSW but Tonelli-Shanks failed, so it is composite; \
                                 factor n to decide",
                                shown)));
                        }
                    }
                };
                caveats.push(format!(
                    " (unproven: {} is a BPSW probable prime of {} bits that was not proven prime ({}); \
                     the list is complete if it is prime)",
                    shown, q.bits(), why));
                locals.push(Local {
                    modulus: q.clone(),
                    least: roots[0].clone(),
                    count: Integer::from(roots.len() as u64),
                    all: Some(roots),
                });
            }
            SqrtModPiece::Composite { .. } => unreachable!("refused in step 5"),
        }
    }
    let crt = |x: &Integer, m: &Integer, y: &Integer, k: &Integer| {
        rustmath_integers::crt_two(x, m, y, k).expect("the pieces are pairwise coprime")
    };
    let mut r = Integer::zero();
    let mut count = Integer::one();
    let mut modulus = Integer::one();
    for l in &locals {
        r = crt(&r, &modulus, &l.least, &l.modulus);
        count = count * l.count.clone();
        modulus = modulus * l.modulus.clone();
    }
    let mut text = if count <= list_limit {
        // Every count is >= 1 here, so each local list has at most 16 entries.
        let mut cur = vec![Integer::zero()];
        let mut cur_mod = Integer::one();
        for l in &locals {
            let roots = l.all.as_ref().expect("a count <= 16 is listed");
            let mut next = Vec::with_capacity(cur.len() * roots.len());
            for x in &cur {
                for y in roots {
                    next.push(crt(x, &cur_mod, y, &l.modulus));
                }
            }
            cur = next;
            cur_mod = cur_mod * l.modulus.clone();
        }
        cur.sort();
        let list = cur.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        format!("{}; square roots of {} mod {}: [{}]", r, b, n, list)
    } else {
        format!("{}; {} has {} square roots mod {} (not listed)", r, b, count, n)
    };
    for c in caveats {
        text.push_str(&c);
    }
    Ok(text)
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

    /// The text of `cmd`, or "Name: message" for an error.
    fn eval_text(ctx: &mut ReplContext, cmd: &str) -> String {
        match ctx.eval(cmd) {
            Ok(v) => v.text,
            Err(e) => format!("{}: {}", e.name, e.message),
        }
    }

    #[test]
    fn test_rational_products_kept() {
        // Regression (pass on HEAD): gp 1/2*1/3 = 1/6, 2/3*3/4 = 1/2, 1/2+1/3*3/4 = 3/4.
        let mut ctx = ReplContext::new();
        for (cmd, want) in [("1/2 * 1/3", "1/6"), ("2/3 * 3/4", "1/2"), ("1/2 + 1/3 * 3/4", "3/4"),
                            ("r = Rational(1, 3)", "1/3"), ("r * 3 / 2", "1/2"), ("e1 = 1/2", "1/2"),
                            ("e1 * 2/3", "1/3")] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{}", cmd);
        }
    }

    #[test]
    fn test_integer_rational_arithmetic() {
        // gp: 2^-2*4 = 1, 8/4*2 = 4, 8*2/4 = 4, 2*3/4 = 3/2, (2/3)^-2 = 9/4, (-2/3)^-3 = -27/8,
        // 10^20/3, 10^400/10^399 = 10; 1/0 and (0/1)^-1: error (gp "impossible inverse").
        let mut ctx = ReplContext::new();
        for (cmd, want) in [("2**-2 * 4", "1"), ("8 / 4 * 2", "4"), ("8 * 2 / 4", "4"), ("2 * 3 / 4", "3/2"),
                            ("a = 8", "8"), ("b = 4", "4"), ("a / b", "2"), ("h = 1/2", "1/2"),
                            ("h + 1", "3/2"), ("1 - h", "1/2"), ("h ^ 2", "1/4"), ("(1/2)^2", "1/4"),
                            ("(2/3)^-2", "9/4"), ("(-2/3)^-3", "-27/8"), ("h ** 0", "1"),
                            ("10^20/3", "100000000000000000000/3"), ("10^400/10^399", "10")] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{}", cmd);
        }
        for cmd in ["1/0", "a / 0", "z0 = 0/5", "h / z0", "1/2 / 0", "(0/1)^-1"] {
            let t = eval_text(&mut ctx, cmd);
            if cmd == "z0 = 0/5" { assert_eq!(t, "0"); continue; }
            assert!(t.starts_with("ZeroDivisionError"), "{}: {}", cmd, t);
        }
    }

    #[test]
    fn test_range_bounds_float_arith() {
        let mut ctx = ReplContext::new();
        ctx.eval("w = var(\"w\")").unwrap();
        let num = |ctx: &mut ReplContext, cmd: &str| -> f64 {
            let t = eval_text(ctx, cmd);
            t.parse::<f64>().unwrap_or_else(|_| panic!("{}: {}", cmd, t))
        };
        // gp: -3+2 = -1, -Pi-1 = -4.14.., -1-3 = -4, 2*-1+5 = 3; the roots are 2, -3, -3, 2.
        assert!((num(&mut ctx, "find_root(w^2 == 4, -3+2, 3)") - 2.0).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 9, -pi-1, 0)") + 3.0).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 9, -1-3, 0)") + 3.0).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 4, 0, 2*-1+5)") - 2.0).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 2, -2^2, 0)") + 2f64.sqrt()).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 2, 0, 2*e)") - 2f64.sqrt()).abs() < 1e-9);
        // A name is a whole identifier: pi2 is the variable, not 3.14159...2.
        ctx.eval("pi2 = 5").unwrap();
        assert!((num(&mut ctx, "find_root(w^2 == 16, 0, pi2)") - 4.0).abs() < 1e-9);
        ctx.eval("e = 5").unwrap();
        assert!((num(&mut ctx, "find_root(w^2 == 16, 0, e)") - 4.0).abs() < 1e-9);
        assert!((num(&mut ctx, "find_root(w^2 == 4, 0, 10^400/10^399)") - 2.0).abs() < 1e-9);
        for cmd in ["find_root(w^2 == 4, 0, 1e400)", "find_root(w^2 == 4, 0, 10^400)"] {
            let t = eval_text(&mut ctx, cmd);
            assert!(t.starts_with("ValueError"), "{}: {}", cmd, t);
        }
        // Revision 2: exact first, so 1/0 and 0/0 are the ZeroDivisionError of the exact arithmetic.
        for cmd in ["find_root(w^2 == 4, 0, 1/0)", "find_root(w^2 == 4, 0, 0/0)"] {
            let t = eval_text(&mut ctx, cmd);
            assert!(t.starts_with("ZeroDivisionError"), "{}: {}", cmd, t);
        }
        assert!(eval_text(&mut ctx, "find_root(w^2 == 2, 0, 2pi)").starts_with("SyntaxError"));
    }

    #[test]
    fn test_range_bounds_exact_first() {
        // Revision 2, B1 (a): texts without a real operand are exact, so no cancellation.
        let mut ctx = ReplContext::new();
        ctx.eval("w = var(\"w\")").unwrap();
        // HEAD regression rows: the bound is 1 (exact), and the root 0.5 lies outside [1, 3].
        for cmd in ["find_root(2*w == 1, 10^20 + 1 - 10^20, 3)", "find_root(2*w == 1, 2^53 + 1 - 2^53, 3)"] {
            let t = eval_text(&mut ctx, cmd);
            assert!(t.starts_with("ValueError: No root found"), "{}: {}", cmd, t);
        }
        ctx.eval("a = 10^20").unwrap();
        let t = eval_text(&mut ctx, "find_root(2*w == 1, a + 1 - a, 3)");
        assert!(t.starts_with("ValueError: No root found"), "{}", t);
        // B1 (b): absorbed overflow or underflow; the exact value of the bound is 3, the root 2.
        for cmd in ["find_root(w^2 == 4, 1, 3 * 10^300 / 10^400 * 10^100)",
                    "find_root(w^2 == 4, 1, 10^-200 * 10^-200 * 10^300 * 10^100 * 3)"] {
            let t = eval_text(&mut ctx, cmd);
            let v: f64 = t.parse().unwrap_or_else(|_| panic!("{}: {}", cmd, t));
            assert!((v - 2.0).abs() < 1e-9, "{}: {}", cmd, t);
        }
    }

    #[test]
    fn test_float_arith_real_field() {
        // Revision 2: MAGMA's coercion rule with the real field of doubles.
        let mut ctx = ReplContext::new();
        let f = |ctx: &mut ReplContext, s: &str| ctx.eval_to_f64(s);
        // Exact subresults stay exact until a real operand (gp 10^400/10^399*Pi, (10^20+1-10^20)*Pi).
        assert_eq!(f(&mut ctx, "10^400/10^399 * pi").unwrap(), 10.0 * std::f64::consts::PI);
        assert_eq!(f(&mut ctx, "(10^20 + 1 - 10^20) * pi").unwrap(), std::f64::consts::PI);
        assert_eq!(f(&mut ctx, "pi/2").unwrap(), std::f64::consts::FRAC_PI_2);
        assert_eq!(f(&mut ctx, "-pi").unwrap(), -std::f64::consts::PI);
        assert_eq!(f(&mut ctx, "pi - pi").unwrap(), 0.0);
        // A zero operand makes a product exact, not an underflow (HEAD 2.14159..., gp the same).
        assert_eq!(f(&mut ctx, "pi + 12157665459056928801 * (pi - pi) - pi / (pi)").unwrap(), std::f64::consts::PI - 1.0);
        // 3^40 is rounded once (gp 3^40*Pi = 38194432490975628713.208...).
        assert_eq!(f(&mut ctx, "3^40 * pi").unwrap(), 12157665459056928801.0 * std::f64::consts::PI);
        assert!((f(&mut ctx, "2^(1/2)").unwrap() - 2f64.sqrt()).abs() < 1e-15);
        // A real operand makes the rest a 53-bit computation, as in MAGMA at this precision.
        assert_eq!(f(&mut ctx, "10^20 + pi - 10^20").unwrap(), 0.0);
        // Overflow and underflow after a real operand are errors, never 0 or inf.
        for s in ["pi * 10^-400 * 10^400", "pi * 1e-200 * 1e-200 * 1e300", "pi^-1000", "1e308 * 10 * pi", "pi * 2^1100"] {
            let e = f(&mut ctx, s).unwrap_err();
            assert_eq!(e.name, "ValueError", "{}: {}", s, e.message);
        }
        assert_eq!(f(&mut ctx, "pi / (pi - pi)").unwrap_err().name, "ZeroDivisionError");
        assert_eq!(f(&mut ctx, "pi + 1/0").unwrap_err().name, "ZeroDivisionError");
        assert_eq!(f(&mut ctx, "(-pi)^(1/2)").unwrap_err().name, "ValueError");
        // A run of signs before a real operand is one step (no stack growth).
        let deep = format!("{}pi", "-".repeat(100_001));
        assert_eq!(f(&mut ctx, &deep).unwrap(), -std::f64::consts::PI);
        // Float literals and Float variables are real operands (HEAD's doubles).
        assert_eq!(f(&mut ctx, "0.1 + 0.2").unwrap(), 0.1 + 0.2);
        ctx.eval("x = 1.5").unwrap();
        assert_eq!(f(&mut ctx, "x * pi").unwrap(), 1.5 * std::f64::consts::PI);
    }

    #[test]
    fn test_negative_exponent_message() {
        // J1b replaced J1a's exponent bound (|e| <= 10000) by a bound on the bits of the result
        // (S12): these were refused and are now computed. gp: 2^-10001 == 1/2^10001 and
        // (1/2)^-10001 == 2^10001; #digits(2^10001) = 3011, 2^10001 % 10^20 = 63548609585193418752.
        let mut ctx = ReplContext::new();
        let p = eval_text(&mut ctx, "2 ** 10001");
        assert_eq!(p.len(), 3011);
        assert!(p.ends_with("63548609585193418752"), "{p}");
        assert_eq!(eval_text(&mut ctx, "2 ** -10001"), format!("1/{p}"));
        assert_eq!(eval_text(&mut ctx, "(1/2) ^ -10001"), p);
        // gp: #binary(2^33554432) = 33554433, one bit above the limit.
        assert_eq!(eval_text(&mut ctx, "2 ** -33554432"),
                   "ValueError: **: b^e for b = 2, e = -33554432 has a denominator of about 3.355e7 bits, more than 33554432 (REPL limit)");
        assert!(eval_text(&mut ctx, "(1/2) ^ -33554432").starts_with("ValueError: ^: r^e for r = 1/2, e = -33554432 has about"));
    }

    /// S14: a chart has exactly dim(M) distinct identifier names (Sage `Chart.__init__`: "the
    /// list of coordinates must contain dim elements").
    #[test]
    fn test_chart_matches_dimension() {
        let mut ctx = ReplContext::new();
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        assert_eq!(eval_text(&mut ctx, "chart(E3, \"x,y,z\")"), "Chart(x, y, z)");
        assert_eq!(eval_text(&mut ctx, "chart(E3, \"r,θ,φ\")"), "Chart(r, θ, φ)");
        for (cmd, want) in [
            ("chart(E3, \"t\")", "chart: the manifold has dimension 3, so the chart needs 3 coordinate names, got 1"),
            ("chart(E3, \"x,y,z,w\")", "chart: the manifold has dimension 3, so the chart needs 3 coordinate names, got 4"),
            ("chart(E3, \"x,,z\")", "chart: coordinate name \"\" is not an identifier"),
            ("chart(E3, \"x,x,y\")", "chart: coordinate name x appears twice"),
            ("chart(E3, \"x,1y,z\")", "chart: coordinate name \"1y\" is not an identifier"),
        ] {
            assert_eq!(eval_text(&mut ctx, cmd), format!("ValueError: {want}"), "{cmd}");
        }
        ctx.eval("M0 = Manifold(0, \"P\")").unwrap();
        assert_eq!(eval_text(&mut ctx, "chart(M0, \"\")"), "Chart()");
        assert_eq!(
            eval_text(&mut ctx, "chart(M0, \"t\")"),
            "ValueError: chart: the manifold has dimension 0, so the chart needs 0 coordinate names, got 1"
        );
        assert_eq!(chart_coordinate_names("", 0).unwrap(), Vec::<String>::new());
        assert!(chart_coordinate_names("", 1).is_err());
        assert_eq!(chart_coordinate_names(" x , y ", 2).unwrap(), vec!["x".to_string(), "y".to_string()]);
        ctx.eval("E4 = EuclideanSpace(4)").unwrap();
        ctx.eval("C3 = chart(E3, \"x,y,z\")").unwrap();
        assert_eq!(
            eval_text(&mut ctx, "coordinate_form(E4, C3, 0)"),
            "ValueError: coordinate_form: the chart has 3 coordinates, but the manifold has dimension 4"
        );
    }

    /// N5: plot options are checked, not clamped or ignored. The expected values are counts of
    /// grid points (n^2 arrows) and the error texts.
    #[test]
    fn test_plot_options_checked() {
        let mut ctx = ReplContext::new();
        ctx.eval("x = var(\"x\")").unwrap();
        ctx.eval("y = var(\"y\")").unwrap();
        let slope = |opt: &str| format!("plot_slope_field(x - y, (x, -2, 2), (y, -2, 2), {opt})");
        let contour = |opt: &str| format!("contour_plot(x^2 + y^2, (x, -2, 2), (y, -2, 2), {opt})");
        for (opt, want) in [("plot_points=25", 625), ("25", 625), ("plot_points=3", 9), ("plot_points=50/2", 625)] {
            assert_eq!(eval_text(&mut ctx, &slope(opt)), format!("Slope field with {want} arrows"), "{opt}");
        }
        let range = "ValueError: plot_slope_field: plot_points must be an integer in 2..=50, got";
        for (opt, got) in [("plot_points=1000", "1000"), ("plot_points=abc", "abc"), ("plot_points=2.5", "2.5"),
                           ("plot_points=5/2", "5/2")] {
            assert_eq!(eval_text(&mut ctx, &slope(opt)), format!("{range} {got}"), "{opt}");
        }
        assert_eq!(eval_text(&mut ctx, &slope("plot_pionts=25")),
                   "ValueError: plot_slope_field: unknown option plot_pionts (options: plot_points)");
        assert_eq!(eval_text(&mut ctx, &slope("plot_points=4, plot_points=5")),
                   "ValueError: plot_slope_field: option plot_points is given twice");
        assert_eq!(eval_text(&mut ctx, &contour("contours=5")), "Contour plot with 5 levels");
        assert!(ctx.eval(&contour("plot_points=2*10")).is_ok());
        for (opt, want) in [
            ("contours=100", "ValueError: contour_plot: contours must be an integer in 1..=30, got 100"),
            ("cmap=\"jet\"", "ValueError: contour_plot: unknown option cmap (options: contours, n_contours, plot_points)"),
            ("contours=5, n_contours=6", "ValueError: contour_plot: give contours or n_contours, not both"),
            ("7", "ArgumentError: contour_plot: unexpected argument 7; options are given as name=value"),
        ] {
            assert_eq!(eval_text(&mut ctx, &contour(opt)), want, "{opt}");
        }
        for (cmd, range) in [
            ("parametric_plot((x, x^2), (x, 0, 1), plot_points=1001)", "parametric_plot: plot_points must be an integer in 2..=1000, got 1001"),
            ("vector_field((y, -x), (x, -1, 1), (y, -1, 1), plot_points=31)", "vector_field: plot_points must be an integer in 2..=30, got 31"),
            ("implicit_plot(x^2 + y^2 - 1, (x, -2, 2), (y, -2, 2), plot_points=201)", "implicit_plot: plot_points must be an integer in 2..=200, got 201"),
            ("region_plot(x^2 + y^2 - 1, (x, -2, 2), (y, -2, 2), plot_points=201)", "region_plot: plot_points must be an integer in 2..=200, got 201"),
        ] {
            assert_eq!(eval_text(&mut ctx, cmd), format!("ValueError: {range}"), "{cmd}");
        }
        // The least value the drawing code can use (the old minima silently raised it).
        assert_eq!(eval_text(&mut ctx, "parametric_plot((x, x^2), (x, 0, 1), plot_points=2)"), "Parametric curve with 2 points");
        assert_eq!(eval_text(&mut ctx, "vector_field((y, -x), (x, -1, 1), (y, -1, 1), plot_points=2)"), "Vector field with 4 arrows");
        assert!(ctx.eval("implicit_plot(x^2 + y^2 - 1, (x, -2, 2), (y, -2, 2), plot_points=2)").is_ok());
        assert!(ctx.eval("region_plot(x^2 + y^2 - 1, (x, -2, 2), (y, -2, 2), plot_points=2)").is_ok());
        // Plot3DOptions: every key and value is checked (review 3 of J1a, m1: colors too).
        for args in [&["opacity=1.5"][..], &["plot_points=40"], &["foo=1"], &["mesh=maybe"], &["color='foo'"],
                     &["color=#12345"], &["color=0xggg"], &["opacity"]] {
            assert!(Plot3DOptions::parse_from_args(args).is_err(), "{args:?}");
        }
        let o = Plot3DOptions::parse_from_args(&["color='red'", "opacity=0.5", "mesh=True"]).unwrap();
        assert_eq!((o.color.as_str(), o.opacity, o.mesh), ("#ff6666", 0.5, true));
        let o = Plot3DOptions::parse_from_args(&["rgbcolor=0x0f0", "axes=false"]).unwrap();
        assert_eq!((o.color.as_str(), o.axes), ("#0f0", false));
    }

    /// S10: pochhammer is the rising factorial. gp: prod(i=0,3,3+i) = 360,
    /// prod(i=0,2,5+i) = 210, prod(i=0,4,10+i) = 240240, prod(i=0,3,3-i) = 0.
    #[test]
    fn test_pochhammer_is_rising() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [("pochhammer(3, 4)", "360"), ("pochhammer(5, 3)", "210"), ("pochhammer(10, 5)", "240240"),
                            ("pochhammer(0, 0)", "1"), ("falling_factorial(3, 4)", "0"), ("rising_factorial(3, 4)", "360")] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{cmd}");
        }
        // J1b: any integer x (C1's rising_factorial_integer); gp: prod(i=0,1,-1+i) = 0. This was
        // refused before (u32 n). Its errors name the command that was called.
        assert_eq!(eval_text(&mut ctx, "pochhammer(-1, 2)"), "0");
        assert!(eval_text(&mut ctx, "pochhammer(1, -1)").starts_with("ValueError: pochhammer: k must be"));
    }

    #[test]
    fn test_chart_checked_in_form_arms() {
        let mut ctx = ReplContext::new();
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("E4 = EuclideanSpace(4)").unwrap();
        ctx.eval("C3 = chart(E3, \"x,y,z\")").unwrap();
        ctx.eval("C4 = chart(E4, \"p,q,r,s\")").unwrap();
        ctx.eval("a = coordinate_form(E3, C3, 0)").unwrap();
        assert_eq!(eval_text(&mut ctx, "exterior_derivative(a, C4)"),
                   "ValueError: exterior_derivative: the chart has 4 coordinates, but the manifold has dimension 3");
        assert_eq!(eval_text(&mut ctx, "wedge(a, a, C4)"),
                   "ValueError: wedge: the chart has 4 coordinates, but the manifold has dimension 3");
        assert!(ctx.eval("exterior_derivative(a, C3)").is_ok());
        assert!(ctx.eval("wedge(a, a, C3)").is_ok());
    }

    #[test]
    fn test_median_large_rational() {
        // HEAD panics in rustmath-stats (statistics.rs:60, partial_cmp of NaN).
        let mut ctx = ReplContext::new();
        ctx.eval("big = 10^400").unwrap();
        ctx.eval("r2 = Rational(big + 1, big)").unwrap();
        assert_eq!(eval_text(&mut ctx, "median([r2, 1, 2])").parse::<f64>().unwrap(), 1.0);
    }

    #[test]
    fn test_value_type_name_by_variant() {
        let mut ctx = ReplContext::new();
        assert_eq!(eval_text(&mut ctx, "-\"a\""), "TypeError: bad operand type for unary -: String");
        assert_eq!(eval_text(&mut ctx, "-[1, 2]"), "TypeError: bad operand type for unary -: List");
    }

    #[test]
    fn test_float_literal_range_and_underflow() {
        let mut ctx = ReplContext::new();
        assert!(eval_text(&mut ctx, "1e400").starts_with("ValueError"));
        assert!(eval_text(&mut ctx, "-1e400").starts_with("ValueError"));
        assert_eq!(eval_text(&mut ctx, "1e-400"), "0");
        // Python float(Fraction(1, 10**400)) = 0.0: below the least subnormal.
        assert_eq!(eval_text(&mut ctx, "mean([Rational(1, 10^400)])").parse::<f64>().unwrap(), 0.0);
        assert_eq!(eval_text(&mut ctx, "mean([Rational(1, 10^310)])").parse::<f64>().unwrap(), 1e-310);
    }

    #[test]
    fn test_sign_and_group_runs() {
        let mut ctx = ReplContext::new();
        let deep_signs = format!("{}1", "-".repeat(100_001));
        assert_eq!(eval_text(&mut ctx, &deep_signs), "-1");
        let deep_group = format!("{}7{}", "(".repeat(100_000), ")".repeat(100_000));
        assert_eq!(eval_text(&mut ctx, &deep_group), "7");
        // Chains are folded in a loop: HEAD's debug build overflowed the stack at 732 terms.
        let long_sum = format!("1{}", "+1".repeat(100_000));
        assert_eq!(eval_text(&mut ctx, &long_sum), "100001");
        let long_product = format!("2{}/2", "*2".repeat(1_000));
        assert_eq!(eval_text(&mut ctx, &long_product), (rustmath_integers::Integer::from(2).pow(1000)).to_string());
        let w = "w = var(\"w\")";
        ctx.eval(w).unwrap();
        assert_eq!(eval_text(&mut ctx, "w - w + w - w"), "(((w - w) + w) - w)");
        for (cmd, want) in [("-+-5", "5"), ("((a)+(b))", "(a + b)"), ("(1)+(2)", "3"), ("(\")\")", "\")\""),
                            ("((1)+(2))*((3))", "9")] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{}", cmd);
        }
        assert!(eval_text(&mut ctx, "((1, 2))").starts_with("SyntaxError"));
        assert!(eval_text(&mut ctx, "(())").starts_with("SyntaxError"));
    }

    /// S9: unary signs, precedence and associativity. References: gp `2*-3` = -6, `-2^2` = -4,
    /// `2^-1` = 1/2, `-2^-2` = -1/4, `0^-1` (error e_INV), `2^3^2` = 512, `1-2-3` = -4,
    /// `2-3+4` = 3, `(2+3)*4` = 20, `2*(3-1)^2` = 8, `-(2+3)` = -5; Python `3 - -2` = 5,
    /// `--5` = 5, `-+-5` = 5.
    #[test]
    fn test_parser_signs_and_precedence() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [
            ("2 * -3", "-6"), ("3 - -2", "5"), ("--5", "5"), ("-2 ** 2", "-4"), ("-2^2", "-4"),
            ("2 ** -1", "1/2"), ("2 ^ -1", "1/2"), ("-2**-2", "-1/4"), ("2^3^2", "512"),
            ("2**3**2", "512"), ("1 - 2 - 3", "-4"), ("2 - 3 + 4", "3"), ("(2+3)*4", "20"),
            ("2*(3-1)^2", "8"), ("-(2+3)", "-5"), ("x = 5", "5"), ("-x", "-5"), ("-x^2", "-25"),
            ("-x * 2", "-10"), ("-+-5", "5"), ("x1e = 7", "7"), ("x1e-3", "4"),
        ] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{cmd}");
        }
        assert!(eval_text(&mut ctx, "0 ** -1").starts_with("ZeroDivisionError"));
        // Left associativity: ((y + z) - y), not (y + (z - y)).
        ctx.eval("y = var(\"y\")").unwrap();
        ctx.eval("z = var(\"z\")").unwrap();
        assert_eq!(eval_text(&mut ctx, "y + z - y"), "((y + z) - y)");
        // 1e-3 is one float literal; Float arithmetic is missing (a TypeError, not a SyntaxError).
        assert_eq!(ctx.eval("2 * 1e-3").unwrap_err().name, "TypeError");
        assert_eq!(eval_text(&mut ctx, "-\"a\""), "TypeError: bad operand type for unary -: String");
    }

    /// S9: operators, commas and brackets inside string literals; non-ASCII text.
    #[test]
    fn test_parser_string_literals() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [
            ("\"a-b\"", "\"a-b\""), ("\"a+b\"", "\"a+b\""), ("\"x^2\"", "\"x^2\""),
            ("s = \"x,y,z\"", "\"x,y,z\""), ("s", "\"x,y,z\""), ("'x,y'", "\"x,y\""),
            ("[ \"a,b\", \"c\" ]", "[\"a,b\", \"c\"]"), ("print(\"x,y\", 1)", "\"x,y\" 1"),
            ("\"ab\" * -5", "\"\""), ("\"é\" + \"a\"", "\"éa\""), ("\"é\"*2", "\"éé\""),
            ("Polynomial([gcd(4, 6), 1])", "x + 2"),
        ] {
            assert_eq!(eval_text(&mut ctx, cmd), want, "{cmd}");
        }
        // A char index was used as a byte index (the old parser panicked here).
        assert!(ctx.eval("π*2").is_ok());
        ctx.eval("x = var(\"x\")").unwrap();
        let v: f64 = eval_text(&mut ctx, "find_root(x^2 == 2, 0, 2)").parse().unwrap();
        assert!((v - 2f64.sqrt()).abs() < 1e-9, "{v}");
    }

    #[test]
    fn test_lex_state() {
        let commas: Vec<usize> = top_level_chars("f(\"a,b\", [1, 2]), c")
            .into_iter()
            .filter(|&(_, c)| c == ',')
            .map(|(i, _)| i)
            .collect();
        assert_eq!(commas, vec![16]);
        assert!(!is_string_literal("\"a\" + \"b\""));
        assert!(is_string_literal("\"a\\\"b\""));
        assert!(is_string_literal("'x'"));
        assert!(!is_string_literal("\"x"));
        assert!(!is_binary_sign("1e-3", 2));
        assert!(is_binary_sign("x1e-3", 3));
        assert!(!is_binary_sign("2 * -3", 4));
        assert!(is_binary_sign("a - b", 2));
    }

    /// The nesting bound of `eval_expr` and `float_arith` (review 3 of J1a, m4). The body runs
    /// on a thread with a large stack: debug frames are several times larger than release ones.
    #[test]
    fn test_eval_depth_limit() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let mut ctx = ReplContext::new();
                let nest = |k: usize| format!("{}1{}", "-(".repeat(k), ")".repeat(k));
                // Two calls per level: 99 levels pass, 100 are refused (a SyntaxError, no abort).
                assert_eq!(eval_text(&mut ctx, &nest(99)), "-1");
                let e = ctx.eval(&nest(100)).unwrap_err();
                assert_eq!(e.name, "SyntaxError");
                assert!(e.message.contains("nested more than 200 levels"), "{}", e.message);
                let e = ctx.eval(&nest(100_000)).unwrap_err();
                assert_eq!(e.name, "SyntaxError");
                // The same bound in the float evaluator of ranges and bounds.
                assert_eq!(ctx.eval_to_f64(&format!("{}pi{}", "-(".repeat(99), ")".repeat(99))).unwrap(),
                           -std::f64::consts::PI);
                assert_eq!(ctx.eval_to_f64(&format!("{}pi{}", "-(".repeat(100), ")".repeat(100)))
                               .unwrap_err().name, "SyntaxError");
                // The counter is back at 0 after an error: a shallow command still works.
                assert_eq!(eval_text(&mut ctx, "-(-(2))"), "2");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// S11: exact numbers beyond the double range. References: Python's correctly rounded
    /// `float(Fraction(10**400 + 1, 10**400))` = 1.0, `float(Fraction(1, 10**310))` = 1e-310,
    /// `float(Fraction(1, 10**400))` = 0.0, `float(2**1100)` OverflowError; gp
    /// `intnum(x=0, 2, x)` = 2. The old num/den conversion gave NaN, [NaN + 0i], 0, inf.
    #[test]
    fn test_large_rational_to_float() {
        let mut ctx = ReplContext::new();
        let num = |ctx: &mut ReplContext, cmd: &str| -> f64 {
            let t = eval_text(ctx, cmd);
            t.parse::<f64>().unwrap_or_else(|_| panic!("{cmd}: {t}"))
        };
        ctx.eval("big = 10^400").unwrap();
        ctx.eval("r2 = Rational(big + 1, big)").unwrap();
        assert_eq!(num(&mut ctx, "mean([r2])"), 1.0);
        assert_eq!(eval_text(&mut ctx, "variance([r2, 1])"), "0");
        assert_eq!(eval_text(&mut ctx, "fft([r2])"), "[1 + 0i]");
        assert_eq!(num(&mut ctx, "mean([Rational(1, 10^310)])"), 1e-310);
        assert_eq!(num(&mut ctx, "mean([Rational(1, 10^400)])"), 0.0);
        let t = eval_text(&mut ctx, "mean([2^1100])");
        assert!(t.starts_with("ValueError") && t.contains("is outside the range of a double"), "{t}");
        ctx.eval("x = var(\"x\")").unwrap();
        assert!((num(&mut ctx, "find_root(x - 1, x, 0, Rational(2*big + 1, big))") - 1.0).abs() < 1e-9);
        // The integrate_num bounds (review 3, m3): the bound rounds to 2.0.
        assert!((num(&mut ctx, "integrate_num(x*1, x, 0, Rational(2*big + 1, big))") - 2.0).abs() < 1e-12);
        assert_eq!(
            eval_text(&mut ctx, "integrate_num(x*1, x, 0, 2^1100)"),
            "ValueError: integrate_num: upper bound: an integer of 1101 bits is outside the range of a double"
        );
        // N of a value beyond the double range (review 3, m2); N(r2) is 1 after the library fix.
        assert!(eval_text(&mut ctx, "N(2^1100)").starts_with("ValueError"));
        assert_eq!(eval_text(&mut ctx, "N(r2)"), "\"1.0000000000\"");
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

    /// S15: graph constructors are bounded by the memory model. The large boundary cases are
    /// checked on `graph_bytes` without building the graphs. gp `binomial(5,2)` = 10.
    #[test]
    fn test_graph_limits() {
        let mut ctx = ReplContext::new();
        assert_eq!(eval_text(&mut ctx, "G = path_graph(100000)"), "Graph with 100000 vertices and 99999 edges");
        assert_eq!(eval_text(&mut ctx, "complete_graph(5)"), "Graph with 5 vertices and 10 edges");
        assert_eq!(eval_text(&mut ctx, "path_graph(1)"), "Graph with 1 vertices and 0 edges");
        assert_eq!(eval_text(&mut ctx, "star_graph(0)"), "Graph with 1 vertices and 0 edges");
        // J1b: a graph is held once (F2), so GRAPH_BYTES = 200 MB; the admitted maxima below.
        assert_eq!(graph_bytes(1_785_714, &[(2, 1), (1_785_712, 2)]), Some(199_999_968));
        assert_eq!(graph_bytes(3585, &[(3585, 3584)]), Some(132_444_240));
        assert_eq!(graph_bytes(4_166_666, &[]), Some(199_999_968));
        assert_eq!(graph_bytes(1_617_193, &[(1_617_192, 1), (1, 1_617_192)]), Some(199_999_952));
        assert_eq!(graph_bytes(1 << 62, &[(1 << 62, (1 << 62) - 1)]), None);
        let lim = format!("(REPL limit {}, a memory bound; add_edge copies the graph)", repl_limits::GRAPH_BYTES);
        for (cmd, need) in [
            ("path_graph(1785715)", "a graph with 1785715 vertices and 1785714 edges needs about 200000080 bytes"),
            ("complete_graph(3586)", "a graph with 3586 vertices and 6427905 edges needs about 264675488 bytes"),
            ("Graph(4166667)", "a graph with 4166667 vertices and 0 edges needs about 200000016 bytes"),
            ("star_graph(1617193)", "a graph with 1617194 vertices and 1617193 edges needs about 200000064 bytes"),
            ("complete_graph(1000000)", "a graph with 1000000 vertices and 499999500000 edges needs about 18874448000000 bytes"),
            ("complete_graph(4611686018427387904)", "a graph with 4611686018427387904 vertices and \
              10633823966279326980924613473029062656 edges needs more than 2^128 bytes"),
            ("complete_graph(18446744073709551615)", "a graph with 18446744073709551615 vertices and \
              170141183460469231704017187605319778305 edges needs more than 2^128 bytes"),
            ("cycle_graph(1785715)", "a graph with 1785715 vertices and 1785715 edges needs about 200000080 bytes"),
            ("wheel_graph(1617193)", "a graph with 1617194 vertices and 3234386 edges needs about 200000064 bytes"),
        ] {
            let name = cmd.split('(').next().unwrap();
            assert_eq!(eval_text(&mut ctx, cmd), format!("ValueError: {name}: {need} {lim}"), "{cmd}");
        }
        // n + 1 overflowed usize in rustmath-graphs (a debug panic); now the model refuses it.
        assert!(eval_text(&mut ctx, "star_graph(18446744073709551615)").starts_with("ValueError: star_graph: a graph with 18446744073709551616 vertices"));
    }

    /// The hash table model against the peak memory of the real REPL (VmHWM of `G = ...`,
    /// release, J1a design §6.1): path_graph(10^6) 334 MB, complete_graph(3000) 331 MB,
    /// Graph(2·10^6) 287 MB, star_graph(10^6) 389 MB. An assignment held three copies before
    /// F2 (J1b); these peaks were measured then.
    #[test]
    fn test_hash_set_bytes_model() {
        assert_eq!(hash_set_bytes(0), Some(0));
        assert_eq!(hash_set_bytes(3), Some(64));
        assert_eq!(hash_set_bytes(7), Some(96));
        assert_eq!(hash_set_bytes(8), Some(16 * 9 + 32));
        assert_eq!(hash_set_bytes(u128::MAX), None);
        let n = 1_000_000u128;
        for (model, measured_mb) in [
            (graph_bytes(n, &[(2, 1), (n - 2, 2)]), 334.0),
            (graph_bytes(3000, &[(3000, 2999)]), 331.0),
            (graph_bytes(2 * n, &[]), 287.0),
            (graph_bytes(n + 1, &[(n, 1), (1, n)]), 389.0),
        ] {
            let mb = 3.0 * model.unwrap() as f64 / 1e6;
            assert!((mb / measured_mb - 1.0).abs() < 0.03, "{mb} MB vs {measured_mb} MB");
        }
    }

    /// S15: prime_pi and primes. References (gp): `primepi(10^6)` = 78498, `primes([11,28])`,
    /// `primes([2^64, 2^64+99])`, `primes([10^30, 10^30+299])`, `nextprime(10^400) - 10^400` = 69,
    /// `nextprime(2^1000) - 2^1000` = 297; above 2^1024 the first primes are at +643, +1081,
    /// +2113 (29 below +20000).
    #[test]
    fn test_prime_counting_limits() {
        let mut ctx = ReplContext::new();
        assert_eq!(eval_text(&mut ctx, "prime_pi(10^6)"), "78498");
        assert_eq!(eval_text(&mut ctx, "prime_pi(-5)"), "0");
        assert_eq!(eval_text(&mut ctx, "prime_pi(6000000001)"),
                   "ValueError: prime_pi: x = 6000000001 is above the REPL limit 6000000000 (the sieve takes about 10 s there)");
        assert_eq!(eval_text(&mut ctx, "primes(11, 29)"), "[11, 13, 17, 19, 23]");
        assert_eq!(eval_text(&mut ctx, "primes(-10, 3)"), "[2]");
        assert_eq!(eval_text(&mut ctx, "primes(5, 3)"), "[]");
        assert_eq!(eval_text(&mut ctx, "primes(2^64, 2^64 + 100)"),
                   "[18446744073709551629, 18446744073709551653, 18446744073709551667, 18446744073709551697, 18446744073709551709]");
        assert_eq!(eval_text(&mut ctx, "primes(10^30, 10^30 + 300)"),
                   "[1000000000000000000000000000057, 1000000000000000000000000000099, 1000000000000000000000000000211, \
                    1000000000000000000000000000231, 1000000000000000000000000000271]");
        // Memory: count_bound(10^8 - 2) = 10857362 > 10^6; 10^400 is far above the span limit.
        for cmd in ["primes(0, 10^8)", "primes(0, 10^400)"] {
            let t = eval_text(&mut ctx, cmd);
            assert!(t.starts_with("ValueError: primes: the span") && t.contains("more than 1000000 primes"), "{cmd}: {t}");
        }
        // Above 2^64: answers without primes are cheap (no proof is needed).
        assert_eq!(eval_text(&mut ctx, "primes(10^400, 10^400 + 10)"), "[]");
        assert_eq!(eval_text(&mut ctx, "primes(2^1000, 2^1000 + 200)"), "[]");
        let t = eval_text(&mut ctx, "primes(2^1000, 2^1000 + 10^6)");
        assert!(t.starts_with("ValueError: primes: testing the 500000 odd candidates"), "{t}");
        // Refused at the third probable prime of pass 1 (the exact k depends on t_proof).
        let t = eval_text(&mut ctx, "primes(2^1024, 2^1024 + 20000)");
        assert!(t.starts_with("ValueError: primes: [a, b) has at least ") && t.contains("BPSW probable primes that need a proof")
                && t.contains("REPL limit 10 s"), "{t}");
    }

    /// The time model of `primes` above 2^64 and the Montgomery-Vaughan bound.
    #[test]
    fn test_primes_time_model() {
        let proof = &repl_limits::PRIMES_PROOF_SECONDS;
        assert_eq!(table_seconds(proof, 512), 0.41);
        assert_eq!(table_seconds(proof, 50), 0.008);
        let mid = table_seconds(proof, 1200);
        assert!(mid > 3.3 && mid < 20.6, "{mid}");
        assert!(table_seconds(proof, 1300) > mid);
        assert!(table_seconds(proof, 4096) > 957.0);
        // HEAD's release cost per odd candidate (J1a design §6.3, `t_head_primes.txt`), by height
        // 2^h: the old prime_range was at least three times t_test everywhere, so the prior
        // check refuses nothing that it finished within 30 s.
        for (h, head_seconds) in [(64u64, 61e-6), (100, 74e-6), (128, 70e-6), (256, 150e-6), (512, 295e-6),
                                  (1024, 1.31e-3), (2048, 4.75e-3), (4096, 37e-3)] {
            let t = primes_t_test(h + 1);
            assert!(3.0 * t <= head_seconds, "2^{h}: 3 * {t} > {head_seconds}");
        }
        // gp: cb(y) = min(ceil(y/2)+1, floor(2*y/log(y))); 2*7943969/log(7943969) = 1000000.907.
        let cb = |y: i64| count_bound(&Integer::from(y));
        for (y, want) in [(1, 1), (2, 2), (10, 6), (100, 43), (10_000, 2171), (1_000_000, 144_764),
                          (7_943_969, 1_000_000), (7_943_970, 1_000_001), (8_000_000, 1_006_608), (-3, 0)] {
            assert_eq!(cb(y), Some(want), "count_bound({y})");
        }
        assert_eq!(cb(8_000_001), None);
        assert_eq!(count_bound(&Integer::from(10).pow(400)), None);
        // An upper bound: gp primepi(10), primepi(100), primepi(10^4) = 4, 25, 1229.
        for (y, pi) in [(10, 4), (100, 25), (10_000, 1229)] {
            assert!(cb(y).unwrap() >= pi);
        }
    }

    /// sqrt_mod factors n once (trial division, rho, the quadratic sieve, ECPP); the old trial
    /// division never finished on the first three rows. References (gp):
    /// `factor(2^127+1)` = 3 * 56713727820156410577229101238628035243, the roots by `chinese` over
    /// {1, 2} mod 3 and {2, p - 2} mod p, r = `chinese(Mod(1,3), Mod(2,p))`;
    /// 1000000000000000005490000000000000001989 = `nextprime(10^19)*nextprime(10^20)`, roots by
    /// `chinese` over {2, P - 2} and {2, Q - 2}; `kronecker(3, m)` = 1 but `issquare(Mod(3, m))` = 0
    /// (`kronecker(3, P)` = -1); `factor(318665857834031151167461)` = 399165290221 * 798330580441,
    /// `issquare(Mod(73, psi12))` = 1, the four roots of 73 and of 4 by `chinese`, r of 73 =
    /// `chinese(Mod(79487054939, p), Mod(173837379523, q))` (the least roots);
    /// `kronecker(41, psi12)` = -1; `isprime(2^1279-1)` = 1.
    #[test]
    fn test_sqrt_mod_factored() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [
            ("sqrt_mod(4, 2^127+1)", "113427455640312821154458202477256070488; square roots of 4 mod 170141183460469231731687303715884105729: [2, 56713727820156410577229101238628035241, 113427455640312821154458202477256070488, 170141183460469231731687303715884105727]"),
            ("sqrt_mod(4, 1000000000000000005490000000000000001989)", "2; square roots of 4 mod 1000000000000000005490000000000000001989: [2, 208067940552016986281146496815286624620, 791932059447983019208853503184713377369, 1000000000000000005490000000000000001987]"),
            ("sqrt_mod(3, 1000000000000000005490000000000000001989)", "no square root: 3 is not a square mod 1000000000000000005490000000000000001989"),
            ("sqrt_mod(4, 15)", "7; square roots of 4 mod 15: [2, 7, 8, 13]"),
            ("sqrt_mod(1404, 2340)", "1872; square roots of 1404 mod 2340: [78, 312, 468, 702, 858, 1092, 1248, 1482, 1638, 1872, 2028, 2262]"),
            ("sqrt_mod(4, 318665857834031151167461)", "2; square roots of 4 mod 318665857834031151167461: [2, 3193322321766, 318665857830837828845695, 318665857834031151167459]"),
            ("sqrt_mod(73, 318665857834031151167461)", "75322749390115309241067; square roots of 73 mod 318665857834031151167461: [75322749390115309241067, 116429215029920790230196, 202236642804110360937265, 243343108443915841926394]"),
            ("sqrt_mod(41, 318665857834031151167461)", "no square root: 41 is not a square mod 318665857834031151167461"),
        ] {
            assert_eq!(sqrt_mod_text(&mut ctx, cmd), quoted(want), "{cmd}");
        }
        // Above the proof limit the roots of a probable prime are listed with the caveat.
        let t = sqrt_mod_text(&mut ctx, "sqrt_mod(4, 2^1279-1)");
        let n = Integer::from(2).pow(1279) - Integer::one();
        let roots = format!("[2, {}]", n.clone() - Integer::from(2));
        assert!(t.starts_with(&format!("\"2; square roots of 4 mod {}: {}", n, roots)), "{t}");
        assert!(t.ends_with(" (unproven: n is a BPSW probable prime of 1279 bits that was not proven prime \
                             (above the REPL proof limit of 1024 bits); the list is complete if it is prime)\""), "{t}");
    }

    /// The phases of `sqrt_mod_factor` with small limits. n = 1009 * 1013 * (2^89 - 1) =
    /// 632659779567127515219534076208987 (gp `factor`; `isprime(2^89-1)` = 1). The eight roots of
    /// 9 mod n: gp `chinese` over the roots 3 and -3 mod each prime, sorted.
    #[test]
    fn test_sqrt_mod_factor_phases() {
        let n = Integer::from_decimal_str("632659779567127515219534076208987").unwrap();
        let m89 = Integer::from_decimal_str("618970019642690137449562111").unwrap();
        let lim = |work: u64| SqrtModLimits {
            trial: 100,
            rho_steps: 0,
            i2_max_bits: 4096,
            qs_digits: 0,
            trial_work: work,
            proof_bits: 1024,
        };
        // Phase 1 finds nothing below 100, phase 2 only classifies n (no rho, no sieve), phase 3
        // runs to T = 128000 / (55 + 109/11) = 2000 and finds 1009 and 1013; the cofactor
        // 2^89 - 1 goes to I2 again (a probable prime) and step 4 proves it.
        let mut pieces = sqrt_mod_factor(&n, &lim(2000 * 64));
        pieces.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
        let mut want = vec![
            SqrtModPiece::Prime { p: Integer::from(1009), e: 1 },
            SqrtModPiece::Prime { p: Integer::from(1013), e: 1 },
            SqrtModPiece::Prime { p: m89, e: 1 },
        ];
        want.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
        assert_eq!(pieces, want);
        assert_eq!(
            sqrt_mod_report_with(&Integer::from(9), &n, &lim(2000 * 64)).unwrap(),
            "3; square roots of 9 mod 632659779567127515219534076208987: [3, 89309327224184831862184968309860, \
             95033561965840430253318518712382, 184342889190025262115503487022239, 448316890377102253104030589186748, \
             537626217601287084966215557496605, 543350452342942683357349107899127, 632659779567127515219534076208984]"
        );
        // T = 1010: phase 3 finds 1009 and stops at 1013 > T; the proper cofactor 1013 * (2^89 - 1)
        // is classified by I2 again (composite).
        let e = sqrt_mod_report_with(&Integer::from(9), &n, &lim(1010 * 64)).unwrap_err();
        assert_eq!(e.name, "ValueError");
        assert!(e.message.starts_with("sqrt_mod: n = 632659779567127515219534076208987 is not factored completely \
                                       within the REPL limits: 627016629898045109236406418443 (no prime factor up to \
                                       1010, not split by the quadratic sieve)"), "{}", e.message);
        // T = 500: no factor of the phase-2 composite, so no second I2 call.
        let e = sqrt_mod_report_with(&Integer::from(9), &n, &lim(500 * 64)).unwrap_err();
        assert!(e.message.ends_with(": 632659779567127515219534076208987 (no prime factor up to 500, not split by \
                                     Pollard rho (0 steps) or the quadratic sieve (up to 0 digits)); the square \
                                     roots mod n need its factorization"), "{}", e.message);
        // A Jacobi symbol -1 decides before any factoring (gp kronecker(2, n) = -1).
        assert_eq!(
            sqrt_mod_report_with(&Integer::from(2), &n, &lim(2000 * 64)).unwrap(),
            "no square root: 2 is not a square mod 632659779567127515219534076208987"
        );
    }

    /// Release only (phase 3 trial division runs to T = 2.5·10^8): p40 = nextprime(10^39),
    /// q40 = nextprime(2*10^39) (gp), so n = p40 * q40 =
    /// 2000000000000000000000000000000000000017000000000000000000000000000000000000033 is above
    /// the sieve's 60 digits and has no factor below T: refused, naming the cofactor.
    #[test]
    #[ignore]
    fn test_sqrt_mod_refusal_release() {
        let mut ctx = ReplContext::new();
        let start = std::time::Instant::now();
        let e = sqrt_mod_err(&mut ctx, "sqrt_mod(4, 2000000000000000000000000000000000000017000000000000000000000000000000000000033)");
        assert_eq!(e.name, "ValueError");
        assert!(e.message.contains("is not factored completely within the REPL limits: \
                                    2000000000000000000000000000000000000017000000000000000000000000000000000000033 \
                                    (no prime factor up to"), "{}", e.message);
        assert!(start.elapsed().as_secs_f64() < 5.0, "{:?}", start.elapsed());
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
        // sqrt_mod now factors psi12 (I2's sieve), so test_sqrt_mod_factored has them with all
        // four roots.
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
        // factorial_product_seconds (J1b): 20 ns · k + 15 ns · (k · bits / 64)^1.465.
        // falling(2^32 - 1, 10^6) took 3.7 s with its display; rising(1, 5·10^6) 15.5 s.
        assert!((factorial_product_seconds(1_000_000, 32) - 3.37).abs() < 0.02);
        assert!(factorial_product_seconds(5_000_000, 23) > 10.0);
        assert!(factorial_product_seconds(3_000_000, 22) < 10.0);
        assert!(factorial_product_seconds(u64::MAX, u64::MAX).is_finite());
    }

    // Each of these was a silent wrong answer before (truncation to 32 bits or a usize wrap).
    #[test]
    fn test_int_args_truncation_fixed() {
        let mut ctx = ReplContext::new();
        let catalan_range = format!("catalan: n must be an integer in 0..={}", repl_limits::CATALAN_N);
        let fib_range = format!("fibonacci: n must be an integer in -{0}..={0}", repl_limits::FIBONACCI_N);
        let cases: Vec<(&str, String)> = vec![
            ("catalan(4294967297)", catalan_range.clone()),           // old: 1
            ("catalan(2147483648)", catalan_range),                   // old: 0 (release)
            ("fibonacci(4294967296)", fib_range),                     // old: 0
            ("multinomial(4294967302, [2, 2, 2])", "multinomial: n must be an integer in 0..=4294967295".into()), // old: 90
            ("multinomial(0, [2147483648, 2147483648])",
             "multinomial requires sum of k values (4294967296) to equal n (0)".into()), // old: panic / hang
            ("sigma(2, 4294967296)", "sigma: k must be an integer in 0..=4294967295".into()), // old: sigma_0(2) = 2
            ("stirling2(4294967298, 2)", "stirling2: n must be an integer in 0..=4294967295".into()), // old: 1
        ];
        for (cmd, frag) in &cases {
            value_err(&mut ctx, cmd, frag);
        }
        // J1b: these were refused (u32 arguments, a machine-word p(n)); now computed exactly.
        // Old wrong values: identity, 0, panic / 0, panic / 541220193622112941.
        for (cmd, want) in [
            // gp: [1,1;0,1]^4294967296
            ("matrix_power(Matrix([[1, 1], [0, 1]]), 4294967296)", "[[1, 4294967296], [0, 1]]"),
            // gp: prod(i=0,1,4294967297-i), prod(i=0,1,4294967295+i), numbpart(417)
            ("falling_factorial(4294967297, 2)", "18446744078004518912"),
            ("rising_factorial(4294967295, 2)", "18446744069414584320"),
            ("partitions(417)", "18987964267331664557"),
        ] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
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
            // J1b: the size of the result is bounded, not the exponent.
            (format!("pow(2, {big})"), "pow: b^e for b = 2, e = 9223372036854775808 has about 9.223e18 bits, more than 33554432 (REPL limit)"),
            ("2 ** 18446744073709551616".to_string(), "**: b^e for b = 2, e = 18446744073709551616 has about"),
            (format!("identity({big})"), "identity: n must be an integer in 0..=1000"),
            (format!("zeros(2, {big})"), "zeros: cols must be"),
            (format!("ones({big}, 2)"), "ones: rows must be an integer in 0..=100"),
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
        // gp: [1,1;0,1]^(2^63) (J1b: any non-negative exponent; refused before).
        assert_eq!(int_text(&mut ctx, &format!("matrix_power(Matrix([[1, 1], [0, 1]]), {big})")),
                   "[[1, 9223372036854775808], [0, 1]]");
        ctx.eval(&format!("c = {big}")).unwrap();
        value_err(&mut ctx, "\"ab\" * c",
            "*: repetition count must be an integer in 0..=8388608 (REPL limit: the result has at most 16777216 bytes)");
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
            // J1b: C1's rolling rows need little memory, so the Stirling numbers are bounded by
            // a time model (stirling_seconds); the centre of a row is the costly part.
            ("stirling1(7000, 3500)".into(), "stirling1: the recurrence for n = 7000, k = 3500 takes 1.225e7 steps on numbers of up to 50148 bits, about 10.7 s (estimate), more than 10 s".into()),
            ("stirling2(10000, 1000)".into(), "stirling2: the recurrence for n = 10000, k = 1000 takes 9.001e6 steps".into()),
            ("eulerian(6000, 3000)".into(), "eulerian: the recurrence for n = 6000, k = 3000 takes 9.003e6 steps on numbers of up to 66656 bits, about 15.4 s".into()),
            ("stirling1(-1, 1)".into(), "stirling1: n must be an integer in 0..=4294967295".into()),
            ("stirling2(5, -1)".into(), "stirling2: k must be an integer in 0..=4294967295".into()),
            ("multinomial(4, [5, -1])".into(), "multinomial: every k must be a non-negative integer, got -1".into()),
            // J1b: the product-tree time model of C1's functions (factorial_product_seconds).
            ("multinomial(4000000, [2000000, 2000000])".into(), "multinomial: the product of 4000000 factors of up to 22 bits would take about 14.8 s (estimate)".into()),
            ("falling_factorial(4294967295, 3000000)".into(), "falling_factorial: the product of 3000000 factors of up to 32 bits would take about 16.8 s (estimate), more than 10 s (REPL limit)".into()),
            ("rising_factorial(1, 10000000)".into(), "rising_factorial: the product of 10000000 factors of up to 24 bits would take about".into()),
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
        // Refused by the old table and work models, admitted now (J1b). gp:
        // abs(stirling(1250,1249,1)) = 780625, stirling(1300,1299,2) = 844350, A(1100,1099) = 1,
        // 300000!/300000! = 1.
        for (cmd, want) in [("stirling1(1250, 1249)", "780625"), ("stirling2(1300, 1299)", "844350"),
                            ("eulerian(1100, 1099)", "1"), ("multinomial(300000, [300000])", "1")] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
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
        // A chart has one coordinate name per dimension (S14); the names are one string literal
        // with commas (S9).
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("C = chart(E3, \"x,y,z\")").unwrap();
        assert!(ctx.eval("coordinate_form(E3, C, 0)").is_ok());
        value_err(&mut ctx, "coordinate_form(E3, C, 3)", "coordinate_form: index must be an integer in 0..=2 (3 coordinates)");
        value_err(&mut ctx, "coordinate_form(E3, C, -1)", "coordinate_form: index must be an integer in 0..=2 (3 coordinates)");
        let big = repl_limits::MANIFOLD_COORDS + 1;
        ctx.eval(&format!("Mbig = Manifold({}, \"B\")", big)).unwrap();
        // The size check comes before the chart check, so the R^3 chart shows the limit.
        value_err(&mut ctx, "coordinate_form(Mbig, C, 0)", &format!("coordinate_form: the manifold has dimension {}", big));
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
    fn test_stirling_limits() {
        use StirlingKind::*;
        // The a-priori bounds of stirling_band are upper bounds. gp:
        // #binary(abs(stirling(1000,500,1))) = 5404, #binary(stirling(1000,500,2)) = 5070,
        // #binary(A(1000,500)) = 8525 with A(n,k)=sum(j=0,k,(-1)^j*binomial(n+1,j)*(k+1-j)^n),
        // #binary(abs(stirling(1000,999,1))) = 19, #binary(stirling(1000,999,2)) = 19,
        // #binary(abs(stirling(5000,2500,1))) = 32846.
        for (kind, n, k, true_bits) in [(Stirling1, 1000, 500, 5404.0), (Stirling2, 1000, 500, 5070.0),
                                        (Eulerian, 1000, 500, 8525.0), (Stirling1, 1000, 999, 19.0),
                                        (Stirling2, 1000, 999, 19.0), (Stirling1, 5000, 2500, 32846.0)] {
            let (_, _, bits) = stirling_band(kind, n, k).unwrap();
            assert!(bits >= true_bits - 1.0, "{kind:?}({n}, {k}): bound {bits} below {true_bits}");
        }
        // The cases the callee answers at once have no band.
        for (kind, n, k) in [(Stirling1, 5, 7), (Stirling1, u32::MAX, u32::MAX), (Stirling1, u32::MAX, 0),
                             (Stirling2, 0, 0), (Eulerian, u32::MAX, 0), (Eulerian, 7, 7), (Eulerian, 7, 6)] {
            assert_eq!(stirling_band(kind, n, k), None, "{kind:?}({n}, {k})");
        }
        let mut ctx = ReplContext::new();
        // gp: abs(stirling(1001,1,1)) == 1000!
        assert_eq!(int_text(&mut ctx, "stirling1(1001, 1)"), int_text(&mut ctx, "factorial(1000)"));
        // gp: A(n,k)=sum(j=0,k,(-1)^j*binomial(n+1,j)*(k+1-j)^n); A(1001,1) - 2^1001 = -1002
        assert_eq!(int_text(&mut ctx, "eulerian(1001, 1) - 2 ** 1001"), "-1002");
        // gp: stirling(n,1,2) = 1, stirling(n,n,1) = 1, A(n,0) = 1, answered at once at any n.
        assert_eq!(int_text(&mut ctx, "stirling2(4294967295, 1)"), "1");
        assert_eq!(int_text(&mut ctx, "stirling1(4294967295, 4294967295)"), "1");
        assert_eq!(int_text(&mut ctx, "eulerian(4294967295, 0)"), "1");
        // 8.6e9 steps: refused at once by the time model (HEAD and C1 ran it for minutes).
        value_err(&mut ctx, "stirling2(4294967295, 2)", "stirling2: the recurrence for n = 4294967295, k = 2 takes");
        value_err(&mut ctx, "stirling1(4294967295, 1)", "stirling1: the recurrence for n = 4294967295, k = 1 takes");
        value_err(&mut ctx, "eulerian(4294967295, 1)", "eulerian: the recurrence for n = 4294967295, k = 1 takes");
        // The edges of the band are cheap (C1: stirling1(20000, 19900) 0.33 s in release).
        // gp: stirling(20000,19998,2) = 19995333683325000 = binomial(20000,3)*(3*20000-5)/4.
        assert_eq!(int_text(&mut ctx, "stirling2(20000, 19998)"), "19995333683325000");
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

    /// "x0,x1,...": n distinct coordinate names for `chart` (S14).
    fn chart_names(n: usize) -> String {
        (0..n).map(|i| format!("x{i}")).collect::<Vec<_>>().join(",")
    }

    // exterior_derivative and wedge build every component of the result (fix-up B3).
    #[test]
    fn test_form_component_limits() {
        let mut ctx = ReplContext::new();
        let lim = format!("(REPL limit {}, a memory bound)", repl_limits::FORM_COMPONENTS);
        // Manifold(100000) is accepted and coordinate_form stores 10^5 components, but d(w) and
        // w ∧ w would build 10^10 (an allocation failure, which aborts the kernel).
        ctx.eval("M = Manifold(100000, \"M\")").unwrap();
        ctx.eval(&format!("C = chart(M, \"{}\")", chart_names(100000))).unwrap();
        ctx.eval("w = coordinate_form(M, C, 0)").unwrap();
        for cmd in ["exterior_derivative(w, C)", "wedge(w, w, C)"] {
            let name = cmd.split('(').next().unwrap();
            value_err(&mut ctx, cmd, &format!("{name}: the result, a 2-form on a manifold of dimension 100000, has "));
            value_err(&mut ctx, cmd, &lim);
        }
        // Small forms on R^3 work, also derived ones.
        ctx.eval("E3 = EuclideanSpace(3)").unwrap();
        ctx.eval("C3 = chart(E3, \"x,y,z\")").unwrap();
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
        ctx.eval("Cm = chart(M3, \"a,b,c\")").unwrap();
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
        ctx.eval(&format!("CN = chart(N, \"{}\")", chart_names(2000))).unwrap();
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
        ctx.eval("CS = chart(S, \"a,b,c\")").unwrap();
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

    // ---------------------------------------------------------------------------------
    // Follow-ups unit J1b: shared values (F2), growth guards (S12), the display budget (D),
    // interrupts and the time limit (S13), the C1 arms. Reference values from PARI/GP 2.15
    // (`ref_j1b.gp` in the J1b implementation report; each command is in the comment).
    // ---------------------------------------------------------------------------------

    #[test]
    fn test_values_shared_not_copied() {
        let mut ctx = ReplContext::new();
        ctx.eval("M = Manifold(3, \"M\")").unwrap();
        ctx.eval("C = chart(M, \"a,b,c\")").unwrap();
        ctx.eval("w = coordinate_form(M, C, 0)").unwrap();
        ctx.eval("v = w").unwrap();
        match (&ctx.variables["w"], &ctx.variables["v"], ctx.last_result.as_ref().unwrap()) {
            (RustMathValue::DiffFormVal(a), RustMathValue::DiffFormVal(b), RustMathValue::DiffFormVal(c)) => {
                assert!(Arc::ptr_eq(a, b) && Arc::ptr_eq(a, c));
            }
            _ => panic!("not forms"),
        }
        ctx.eval("L = [1, 2, 3]").unwrap();
        ctx.eval("P = [L, L]").unwrap();
        let l = match &ctx.variables["L"] { RustMathValue::List(l) => l.clone(), _ => panic!("L") };
        match &ctx.variables["P"] {
            RustMathValue::List(p) => {
                assert_eq!(p.len(), 2);
                for item in p.iter() {
                    match item { RustMathValue::List(x) => assert!(Arc::ptr_eq(x, &l)), _ => panic!("P item") }
                }
            }
            _ => panic!("P"),
        }
        ctx.eval("A = identity(3)").unwrap();
        ctx.eval("B = A").unwrap();
        match (&ctx.variables["A"], &ctx.variables["B"]) {
            (RustMathValue::Matrix(a), RustMathValue::Matrix(b)) => assert!(Arc::ptr_eq(a, b)),
            _ => panic!("matrices"),
        }
        // Tuple unpacking shares the elements too.
        ctx.eval("X, Y = [A, L]").unwrap();
        match (&ctx.variables["X"], &ctx.variables["A"]) {
            (RustMathValue::Matrix(x), RustMathValue::Matrix(a)) => assert!(Arc::ptr_eq(x, a)),
            _ => panic!("X"),
        }
    }

    #[test]
    fn test_value_semantics_kept() {
        let mut ctx = ReplContext::new();
        ctx.eval("G = Graph(3)").unwrap();
        ctx.eval("H = add_edge(G, 0, 1)").unwrap();
        assert_eq!(int_text(&mut ctx, "num_edges(G)"), "0");
        assert_eq!(int_text(&mut ctx, "num_edges(H)"), "1");
        ctx.eval("p = Polynomial([1, 1])").unwrap();
        ctx.eval("q = p * p").unwrap();
        // gp: (x+1)^2 = x^2 + 2*x + 1
        assert_eq!(int_text(&mut ctx, "q"), "x^2 + 2*x + 1");
        assert_eq!(int_text(&mut ctx, "p"), "x + 1");
    }

    #[test]
    fn test_int_pow_growth_guard() {
        let mut ctx = ReplContext::new();
        // gp: #digits(2^10001) = 3011, 2^10001 % 10^20 = 63548609585193418752 (HEAD: refused)
        let x = int_text(&mut ctx, "2 ** 10001");
        assert_eq!(x.len(), 3011);
        assert!(x.ends_with("63548609585193418752"));
        // gp: #digits(2^1001) = 302, 2^1001 % 10^20 = 48773674411336138752 (HEAD: refused)
        let y = int_text(&mut ctx, "pow(2, 1001)");
        assert_eq!(y.len(), 302);
        assert!(y.ends_with("48773674411336138752"));
        // gp: #binary(3^21170489) = 33554432, #binary(3^21170490) = 33554433,
        // #binary(2^33554431) = 33554432
        assert_eq!(int_pow_bits(&Integer::from(3), &Integer::from(21170489)), 33554432.0);
        assert_eq!(int_pow_bits(&Integer::from(3), &Integer::from(21170490)), 33554433.0);
        assert_eq!(int_pow_bits(&Integer::from(2), &Integer::from(33554431)), 33554432.0);
        value_err(&mut ctx, "3 ** 21170490", "**: b^e for b = 3, e = 21170490 has about 3.355e7 bits, more than 33554432");
        // gp: #binary(2^33554432) = 33554433
        value_err(&mut ctx, "2 ** 33554432", "more than 33554432 (REPL limit)");
        // gp: (-1)^(10^30) = 1, (-1)^(10^30+1) = -1, 0^0 = 1, 0^(10^30) = 0, 1^(10^30) = 1
        for (cmd, want) in [("(-1) ** (10 ** 30)", "1"), ("(-1) ** (10 ** 30 + 1)", "-1"), ("0 ** 0", "1"),
                            ("0 ** (10 ** 30)", "0"), ("1 ** (10 ** 30)", "1")] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
        }
        // gp: 10000*#binary(3^10000) = 158500000 bits (HEAD admitted it: minutes)
        ctx.eval("a = 3 ** 10000").unwrap();
        value_err(&mut ctx, "a ** 10000", "**: b^e for b = an integer of 15850 bits, e = 10000 has about 1.585e8 bits");
        // Integer products have the same bound (review 2 of J1b, m3).
        // gp: #binary(2^16777215*2^16777216) = 33554432, #binary(2^33554431*2) = 33554433
        ctx.eval("h = 2 ** 16777215").unwrap();
        ctx.eval("g = 2 ** 16777216").unwrap();
        ctx.eval("q = h * g").unwrap();
        ctx.eval("f = 2 ** 33554431").unwrap();
        assert_eq!(int_text(&mut ctx, "q - f"), "0");
        value_err(&mut ctx, "f * 2", "*: the product of integers of 33554432 and 2 bits has at least 33554433 bits, more than 33554432");
        value_err(&mut ctx, "f * f", "has at least 67108863 bits");
        value_err(&mut ctx, "Rational(f, 3) * Rational(f, 5)", "*: the result may have");
    }

    #[test]
    fn test_negative_and_rational_powers() {
        let mut ctx = ReplContext::new();
        // gp: 2^-1, (-2)^-3, 2^-100, (-1)^-(10^30+1), (2/3)^-2, (2/3)^3
        for (cmd, want) in [
            ("2 ** -1", "1/2"), ("pow(2, -1)", "1/2"), ("pow(-2, -3)", "-1/8"), ("(-2) ** -3", "-1/8"),
            ("pow(2, -100)", "1/1267650600228229401496703205376"), ("(-1) ** -(10 ** 30 + 1)", "-1"),
            ("pow(Rational(2, 3), -2)", "9/4"), ("(2/3) ** 3", "8/27"), ("pow(Rational(2, 3), 3)", "8/27"),
        ] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
        }
        // gp: 0^-1 is an error
        assert_eq!(ctx.eval("0 ** -1").unwrap_err().name, "ZeroDivisionError");
        assert_eq!(ctx.eval("pow(0, -1)").unwrap_err().name, "ZeroDivisionError");
        value_err(&mut ctx, "3 ** -21170490", "has a denominator of about 3.355e7 bits, more than 33554432");
        // gp: 21170490 + ceil(21170489*log(3)/log(2)) = 54724922 bits in all
        value_err(&mut ctx, "pow(Rational(2, 3), 21170489)", "more than 33554432");
        let e = ctx.eval("pow(Matrix([[1]]), 2)").unwrap_err();
        assert_eq!(e.name, "TypeError");
        assert!(e.message.contains("integer or rational base"), "{}", e.message);
        // rustmath-rationals reduces every result with a quadratic gcd, so a fraction with two
        // large parts is refused by time (release: 27.7 s at 2^22 bits each; the J1b prototype
        // ran pow(Rational(2, 3), 15000000) for more than 120 s).
        value_err(&mut ctx, "pow(Rational(2, 3), 10000000)",
            "pow: reducing a fraction with a numerator of 10000001 bits and a denominator of 15849626 bits");
        ctx.eval("a = 2 ** 4194304 + 1").unwrap();
        ctx.eval("b = 2 ** 4194305 + 1").unwrap();
        value_err(&mut ctx, "Rational(a, b)", "Rational: reducing a fraction with a numerator of 4194305 bits");
        value_err(&mut ctx, "a / b", "/: reducing a fraction");
        // (1/a) + (1/b) = (a + b) / (a b): parts of 4194308 and 8388611 bits.
        value_err(&mut ctx, "(1 / a) + (1 / b)", "+: reducing a fraction with a numerator of 4194308 bits and a denominator of 8388611 bits");
        // One small part: cheap.
        assert!(ctx.eval("c = a / 2").is_ok());
        assert!(ctx.eval("(a / 3) + (b / 5)").is_ok());
        assert!(ctx.eval("pow(Rational(1, 3), 200000)").is_ok());
    }

    #[test]
    fn test_matrix_power_growth_guard() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [
            // gp: [1,1;0,1]^(10^9), [1,1;0,1]^(10^30)
            ("matrix_power(Matrix([[1,1],[0,1]]), 10 ** 9)", "[[1, 1000000000], [0, 1]]"),
            ("matrix_power(Matrix([[1,1],[0,1]]), 10 ** 30)", "[[1, 1000000000000000000000000000000], [0, 1]]"),
            // gp: [2,-1,0;1,3,1;0,1,-2]^7
            ("matrix_power(Matrix([[2,-1,0],[1,3,1],[0,1,-2]]), 7)", "[[-1215, -1171, -259], [1171, 215, 135], [259, 135, -201]]"),
            // gp: [0,0,8;1,0,0;0,1,0]^30 == 8^10*matid(3), 8^10 = 1073741824
            ("matrix_power(Matrix([[0,0,8],[1,0,0],[0,1,0]]), 30)", "[[1073741824, 0, 0], [0, 1073741824, 0], [0, 0, 1073741824]]"),
            // gp: [3,1;1,2]^20
            ("matrix_power(Matrix([[3,1],[1,2]]), 20)", "[[106894531250, 66064453125], [66064453125, 40830078125]]"),
            ("matrix_power(Matrix([[1,2],[3,4]]), 0)", "[[1, 0], [0, 1]]"),
        ] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
        }
        // gp: F=[1,1;1,0]^1000; #digits(F[1,2]) = 209, F[1,2] % 10^20 = 76137795166849228875,
        // F[1,1] % 10^20 = 91902245245323403501, F[2,2] % 10^20 = 15764450078474174626
        let f = ctx.checked_matrix_power(
            &Matrix::from_vec(2, 2, vec![Integer::from(1), Integer::from(1), Integer::from(1), Integer::from(0)]).unwrap(),
            &Integer::from(1000)).unwrap();
        let f01 = f.get(0, 1).unwrap().to_string();
        assert_eq!(f01.len(), 209);
        assert!(f01.ends_with("76137795166849228875"));
        assert!(f.get(0, 0).unwrap().to_string().ends_with("91902245245323403501"));
        assert!(f.get(1, 1).unwrap().to_string().ends_with("15764450078474174626"));
        // (a): refused from the trace before any large product.
        let start = Instant::now();
        value_err(&mut ctx, "matrix_power(Matrix([[2]]), 4294967295)", "(from the trace)");
        assert!(start.elapsed() < Duration::from_secs(1), "{:?}", start.elapsed());
        value_err(&mut ctx, "matrix_power(Matrix([[0,0,8],[1,0,0],[0,1,0]]), 10 ** 9)", "(from the trace)");
        // Exponents of a million bits with entries that do not grow (no exponent counter).
        // gp: Mat(1)^(2^1000000) = Mat(1), [0,1;1,0]^(2^1000001+1) = [0,1;1,0]
        ctx.eval("e = 2 ** 1000000").unwrap();
        ctx.eval("f = 2 ** 1000001 + 1").unwrap();
        let start = Instant::now();
        assert_eq!(int_text(&mut ctx, "matrix_power(Matrix([[1]]), e)"), "[[1]]");
        assert_eq!(int_text(&mut ctx, "matrix_power(Matrix([[0,1],[1,0]]), f)"), "[[0, 1], [1, 0]]");
        assert!(start.elapsed() < Duration::from_secs(30), "{:?}", start.elapsed());
        // (c): the floor refuses at once.
        ctx.eval("g = 2 ** 33554431").unwrap();
        let start = Instant::now();
        value_err(&mut ctx, "matrix_power(Matrix([[0,1],[1,0]]), g)", "even if the entries stay small");
        assert!(start.elapsed() < Duration::from_secs(1), "{:?}", start.elapsed());
        // (b): about 56 s for one product of 1000 x 1000 one-word matrices.
        value_err(&mut ctx, "identity(1000) * identity(1000)", "s (estimate), more than 10 s (REPL limit)");
        value_err(&mut ctx, "matrix_power(identity(1000), 2)", "even if the entries stay small");
        value_err(&mut ctx, "matrix_power(Matrix([[1,2],[3,4]]), -1)", "matrix_power: the exponent must be a non-negative integer, got -1");
        assert_eq!(ctx.eval("matrix_power(Matrix([[1,2,3],[4,5,6]]), 2)").unwrap_err().name, "MatrixError");
        // The REPL's binary powering agrees with the library's Matrix::pow.
        let mats = [
            Matrix::from_vec(2, 2, vec![Integer::from(1), Integer::from(1), Integer::from(1), Integer::from(0)]).unwrap(),
            Matrix::from_vec(3, 3, [-3, 2, 0, 1, -1, 3, 2, 2, -2].iter().map(|&v| Integer::from(v)).collect()).unwrap(),
            Matrix::from_vec(3, 3, [0, 1, 0, 0, 0, 1, 0, 0, 0].iter().map(|&v| Integer::from(v)).collect()).unwrap(),
            Matrix::from_vec(3, 3, [3, -3, 1, 2, 0, -1, -2, 1, 3].iter().map(|&v| Integer::from(v)).collect()).unwrap(),
        ];
        for m in &mats {
            for e in 0..40u32 {
                assert_eq!(ctx.checked_matrix_power(m, &Integer::from(e)).unwrap(), m.pow(e).unwrap(), "{m:?}^{e}");
            }
        }
        // The time model: (100 + 600^3 · 56) ns = 12.1 s (measured 10.0 s, J1b ladder `mmul`);
        // increasing in the size and the bits.
        let t = matrix_product_seconds(600, 600, 600, 64);
        assert!((t - 12.096).abs() < 0.001, "{t}");
        assert!(matrix_product_seconds(601, 601, 601, 64) > t && matrix_product_seconds(600, 600, 600, 65) > t);
        // The size bound: 2x2 entries of 2^23 bits: 4 · (2^24 + 1) bits > 2^25.
        let big = Integer::new(BigInt::from(1) << 8_388_607usize);
        let m = Matrix::from_vec(2, 2, vec![big.clone(), big.clone(), big.clone(), big]).unwrap();
        let e = check_matrix_product("*", &m, &m, 0.0).unwrap_err();
        assert!(e.message.contains("may have entries of 16777217 bits"), "{}", e.message);
    }

    #[test]
    fn test_product_size_guards() {
        let mut ctx = ReplContext::new();
        ctx.eval("x = 2 ** 3355443").unwrap();
        // HEAD: 10^4 entries of 3355444 bits (4.2 GB) were allocated.
        value_err(&mut ctx, "ones(100, 100) * x", "*: a matrix with 10000 entries of up to 1 bits times an integer of 3355444 bits may have");
        value_err(&mut ctx, "x * ones(100, 100)", "may have");
        value_err(&mut ctx, "scalar_mul(ones(100, 100), x)", "scalar_mul: a matrix with 10000 entries");
        assert!(ctx.eval("Matrix([[1]]) * x").is_ok());
        ctx.eval("y = 2 ** 20000000").unwrap();
        ctx.eval("P = Polynomial([y, y])").unwrap();
        value_err(&mut ctx, "P * P", "*: the product of polynomials with 2 and 2 coefficients of up to 20000001 and 20000001 bits may have");
        value_err(&mut ctx, "P * x", "*: a polynomial with 2 entries");
        // gp: (x+1)^2
        assert_eq!(int_text(&mut ctx, "Polynomial([1, 1]) * Polynomial([1, 1])"), "x^2 + 2*x + 1");
        // The time model for polynomials, without building them in the REPL: 16000 unit
        // coefficients: 16000^2 · 44 ns = 11.3 s (15000: 9.9 s, admitted; measured 6.5 s).
        let p = |n: usize| UnivariatePolynomial::new(vec![Integer::one(); n]);
        let e = check_poly_product("*", &p(16000), &p(16000)).unwrap_err();
        assert!(e.message.contains("would take about 11.3 s (estimate)"), "{}", e.message);
        assert!(check_poly_product("*", &p(15000), &p(15000)).is_ok());
    }

    #[test]
    fn test_string_growth_guard() {
        let mut ctx = ReplContext::new();
        // HEAD refused counts above 10000 whatever the length, and aborted on 2e12 bytes.
        assert!(ctx.eval("x = \"ab\" * 10001").is_ok());
        // 16777216 bytes: admitted, and shown as a summary (its quoted text has 2^24 + 2 bytes).
        let t = int_text(&mut ctx, "\"a\" * 16777216");
        assert!(t.starts_with("<string of 16777216 bytes: its text has more than 16777216 bytes"), "{t}");
        value_err(&mut ctx, "\"a\" * 16777217", "*: repetition count must be an integer in 0..=16777216");
        ctx.eval("x = \"ab\" * 10000").unwrap();
        assert!(ctx.eval("y = x * 838").is_ok());
        value_err(&mut ctx, "x * 839",
            "*: repetition count must be an integer in 0..=838 (REPL limit: the result has at most 16777216 bytes), got 839");
        ctx.eval("c = 2 ** 1100").unwrap();
        assert_eq!(int_text(&mut ctx, "\"\" * c"), quoted(""));
        assert_eq!(int_text(&mut ctx, "\"ab\" * -3"), quoted(""));
        ctx.eval("s = \"ab\" * 8388608").unwrap();
        value_err(&mut ctx, "s + \"c\"", "+: the concatenation would have 16777217 bytes, more than 16777216");
    }

    #[test]
    fn test_integer_display_limit() {
        // 2^(2^26) has 67108865 bits: about 2.02e7 digits, more than OUTPUT_BYTES.
        let big = RustMathValue::Integer(Integer::new(BigInt::from(1) << 67108864usize));
        let s = big.to_display().text;
        assert!(s.starts_with("<integer of 67108865 bits, about 2.0202e7 decimal digits: not printed"), "{s}");
        let neg = RustMathValue::Integer(Integer::new(-(BigInt::from(1) << 67108864usize)));
        assert!(neg.to_display().text.starts_with("<negative integer of 67108865 bits"));
        let small = RustMathValue::Integer(Integer::new(BigInt::from(1) << 99usize));
        assert_eq!(small.to_display().text, "633825300114114700748351602688");
        // gp: #binary(sigma(2, 2^26)) = 67108865 (HEAD took minutes in debug to print it)
        let mut ctx = ReplContext::new();
        let t = int_text(&mut ctx, "sigma(2, 67108864)");
        assert!(t.starts_with("<integer of 67108865 bits"), "{t}");
        // decimal_string agrees with num-bigint on both sides of the 2^15-bit switch.
        for k in [32767usize, 32768, 32769, 40000, 100003] {
            let a = Integer::new((BigInt::from(3) << k) - BigInt::from(12345));
            assert_eq!(decimal_string(&a), a.to_string());
            let b = -a.clone();
            assert_eq!(decimal_string(&b), b.to_string());
        }
    }

    #[test]
    fn test_output_budget() {
        let mut ctx = ReplContext::new();
        ctx.eval("x = var(\"x\")").unwrap();
        ctx.eval("e = x + x").unwrap();
        for _ in 0..40 {
            ctx.eval("e = e + e").unwrap();
        }
        // HEAD: the text doubles per line (2^40 leaves), an allocation failure.
        let start = Instant::now();
        let t = int_text(&mut ctx, "e");
        assert!(t.starts_with("<symbolic expression: its text has more than 16777216 bytes"), "{t}");
        assert!(start.elapsed() < Duration::from_secs(5), "{:?}", start.elapsed());
        // gp: #digits(2^16777215) = 5050445, four of them: 20201780 bytes.
        ctx.eval("a = 2 ** 16777215").unwrap();
        let t = int_text(&mut ctx, "[a, a, a, a]");
        assert!(t.starts_with("<list of 4 elements: its text has more than"), "{t}");
        let r = ctx.eval("Matrix([[1,2],[3,4]])").unwrap();
        assert_eq!(r.text, "[[1, 2], [3, 4]]");
        assert_eq!(r.latex.as_deref(), Some("$\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}$"));
        let r = ctx.eval("Rational(6, 4)").unwrap();
        assert_eq!(r.text, "3/2");
        assert_eq!(r.latex.as_deref(), Some("$\\frac{3}{2}$"));
        // poly_text equals the library Display.
        for coeffs in [vec![0i64], vec![1], vec![0, 1], vec![3, 0, 1], vec![-5, 2, 0, 7], vec![1, 1, 1, -1]] {
            let p = UnivariatePolynomial::new(coeffs.iter().map(|&c| Integer::from(c)).collect::<Vec<_>>());
            let strs: Vec<String> = p.coefficients().iter().map(|c| if c.is_zero() { String::new() } else { decimal_string(c) }).collect();
            assert_eq!(poly_text(p.coefficients(), &strs), p.to_string());
        }
        // Zero coefficients have no term (review 2 of J1b, m2): x^700000 has 8 bytes of text.
        let v = RustMathValue::Polynomial(Arc::new(UnivariatePolynomial::new(
            (0..=700_000).map(|i| if i == 700_000 { Integer::one() } else { Integer::zero() }).collect())));
        assert_eq!(v.to_display().text, "x^700000");
        // An expression with a 4194305-bit integer is summarised (the library prints it with
        // num-bigint); gp: #binary(2^4194304) = 4194305.
        ctx.eval("y = 2 ** 4194304").unwrap();
        let t = int_text(&mut ctx, "x * y");
        assert!(t.contains("it contains an integer of 4194305 bits, more than 4194304"), "{t}");
        // One bit less is printed; checked on the size estimate only (printing it takes seconds
        // in a debug build).
        let leaf = Expr::Symbol(Symbol::new("x")) * Expr::Integer(Integer::new(BigInt::from(1) << 4194303usize));
        let size = display_size(&RustMathValue::Expr(leaf), repl_limits::OUTPUT_BYTES);
        assert_eq!(size.library_int_bits, 4194304);
        assert!(size.bytes < repl_limits::OUTPUT_BYTES);
    }

    #[test]
    fn test_pending_interrupt_stops_next_eval() {
        let mut ctx = ReplContext::new();
        let h = ctx.interrupt_handle();
        h.interrupt();
        let e = ctx.eval("a = 1").unwrap_err();
        assert_eq!(e.name, "KeyboardInterrupt");
        assert!(!ctx.variables.contains_key("a"));
        assert!(!h.is_pending());
        assert_eq!(int_text(&mut ctx, "a = 1"), "1");
    }

    #[test]
    fn test_interrupt_from_another_thread() {
        let mut ctx = ReplContext::new();
        let h = ctx.interrupt_handle();
        let cell = format!("first = 1\n{}\nlast = 2", vec!["x = factorial(3000)"; 5000].join("\n"));
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            h.interrupt();
        });
        let start = Instant::now();
        let e = ctx.eval(&cell).unwrap_err();
        t.join().unwrap();
        assert_eq!(e.name, "KeyboardInterrupt");
        assert!(start.elapsed() < Duration::from_secs(5), "{:?}", start.elapsed());
        assert!(ctx.variables.contains_key("first"));
        assert!(!ctx.variables.contains_key("last"));
    }

    #[test]
    fn test_time_limit() {
        let mut ctx = ReplContext::new();
        ctx.set_time_limit(Some(Duration::ZERO));
        assert_eq!(ctx.eval("1 + 1").unwrap_err().name, "TimeoutError");
        ctx.set_time_limit(None);
        assert_eq!(int_text(&mut ctx, "1 + 1"), "2");
        assert_eq!(int_text(&mut ctx, "time_limit(1)"), quoted("time limit: 1 s for each later cell"));
        assert_eq!(ctx.time_limit(), Some(Duration::from_secs(1)));
        let cell = vec!["x = factorial(3000)"; 100000].join("\n");
        let start = Instant::now();
        let e = ctx.eval(&cell).unwrap_err();
        assert_eq!(e.name, "TimeoutError");
        assert!(e.message.contains("time limit of 1 s"), "{}", e.message);
        assert!(start.elapsed() < Duration::from_secs(4), "{:?}", start.elapsed());
        assert_eq!(int_text(&mut ctx, "time_limit()"), quoted("time limit: 1 s for each later cell"));
        assert_eq!(int_text(&mut ctx, "time_limit(0)"), quoted("no time limit"));
        assert_eq!(int_text(&mut ctx, "1 + 1"), "2");
        assert_eq!(int_text(&mut ctx, "time_limit(0.5)"), quoted("time limit: 0.5 s for each later cell"));
        assert_eq!(int_text(&mut ctx, "time_limit(0)"), quoted("no time limit"));
        value_err(&mut ctx, "time_limit(-1)", "time_limit: give a number of seconds");
        value_err(&mut ctx, "time_limit(\"x\")", "time_limit: give a number of seconds");
        value_err(&mut ctx, "time_limit(31536001)", "time_limit: seconds must be an integer in 0..=31536000");
    }

    #[test]
    fn test_matrix_power_checkpoint() {
        let ctx = ReplContext::new();
        let m = Matrix::from_vec(2, 2, vec![Integer::from(1), Integer::from(1), Integer::from(1), Integer::from(0)]).unwrap();
        ctx.interrupt_handle().interrupt();
        assert_eq!(ctx.checked_matrix_power(&m, &Integer::from(1000)).unwrap_err().name, "KeyboardInterrupt");
        ctx.interrupt_handle().clear();
        assert!(ctx.checked_matrix_power(&m, &Integer::from(1000)).is_ok());
    }

    /// One plot sample is a library call; the loops check for interrupts between samples
    /// (review 2 of J1b, m1: plot_function too).
    #[test]
    fn test_plot_checkpoint() {
        for cmd in ["plot3d(e, (x, 0, 1), (y, 0, 1))", "plot_function(e, x, 0, 1, 10000)"] {
            let mut ctx = ReplContext::new();
            ctx.eval("x = var(\"x\")").unwrap();
            ctx.eval("y = var(\"y\")").unwrap();
            ctx.eval("e = x + x").unwrap();
            for _ in 0..12 {
                ctx.eval("e = e + e").unwrap();
            }
            let h = ctx.interrupt_handle();
            let t = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                h.interrupt();
            });
            let start = Instant::now();
            let r = ctx.eval(cmd);
            t.join().unwrap();
            let elapsed = start.elapsed();
            // A machine fast enough to finish before the interrupt proves nothing either way.
            if elapsed > Duration::from_millis(400) {
                let e = r.expect_err(cmd);
                assert_eq!(e.name, "KeyboardInterrupt", "{cmd}: {}", e.message);
                assert!(elapsed < Duration::from_millis(2200), "{cmd}: {elapsed:?}");
            }
        }
    }

    #[test]
    fn test_divmod_by_zero_polynomial() {
        let mut ctx = ReplContext::new();
        // HEAD: a panic in rustmath-polynomials, which killed the kernel.
        let e = ctx.eval("divmod(Polynomial([1]), Polynomial([0]))").unwrap_err();
        assert_eq!(e.name, "ZeroDivisionError");
        assert_eq!(e.message, "divmod: division by the zero polynomial");
        // gp: divrem(x^2-1, x+1) = [x - 1, 0]; rustmath-polynomials prints x - 1 as "x + -1".
        assert_eq!(int_text(&mut ctx, "divmod(Polynomial([-1, 0, 1]), Polynomial([1, 1]))"), "[x + -1, 0]");
    }

    #[test]
    fn test_c1_arms() {
        let mut ctx = ReplContext::new();
        for (cmd, want) in [
            // gp: numbpart(417), numbpart(10^4), numbpart(416), numbpart(0), numbpart(100)
            ("partitions(417)", "18987964267331664557"),
            ("partitions(10000)", "36167251325636293988820471890953695495016030339315650422081868605887952568754066420592310556052906916435144"),
            ("partitions(416)", "17873792969689876004"),
            ("partitions(0)", "1"),
            ("partitions(100)", "190569292"),
            // gp: ff(x,k)=prod(i=0,k-1,x-i); rf(x,k)=prod(i=0,k-1,x+i)
            ("falling_factorial(2 ** 40, 3)", "1329227995781289095444965371779481600"),
            ("falling_factorial(4294967297, 2)", "18446744078004518912"),
            ("falling_factorial(-3, 4)", "360"),
            ("falling_factorial(-4294967296, 2)", "18446744078004518912"),
            ("falling_factorial(10 ** 30, 3)", "999999999999999999999999999997000000000000000000000000000002000000000000000000000000000000"),
            ("falling_factorial(5, 6)", "0"),
            ("falling_factorial(5, 10 ** 12)", "0"),
            ("falling_factorial(-3, 0)", "1"),
            ("rising_factorial(-5, 3)", "-60"),
            ("pochhammer(-5, 3)", "-60"),
            ("rising_factorial(4294967295, 2)", "18446744069414584320"),
            ("rising_factorial(2 ** 64, 2)", "340282366920938463481821351505477763072"),
            ("rising_factorial(-(2 ** 64 + 5), 3)", "-6277101735386680767919177826258927978529847705109601976380"),
            ("rising_factorial(-5, 5)", "-120"),
            ("rising_factorial(-5, 6)", "0"),
            ("rising_factorial(-5, 10 ** 12)", "0"),
        ] {
            assert_eq!(int_text(&mut ctx, cmd), want, "{cmd}");
        }
        value_err(&mut ctx, "partitions(-1)", &format!("partitions: n must be an integer in 0..={}", repl_limits::PARTITIONS_N));
        value_err(&mut ctx, &format!("partitions({})", repl_limits::PARTITIONS_N + 1), "partitions: n must be an integer in 0..=");
        // The time model of the product tree: 16.7 s and 64 s (estimates).
        value_err(&mut ctx, "falling_factorial(4294967295, 3000000)",
            "falling_factorial: the product of 3000000 factors of up to 32 bits would take about");
        value_err(&mut ctx, "rising_factorial(1, 10000000)", "rising_factorial: the product of 10000000 factors of up to 24 bits");
        value_err(&mut ctx, "falling_factorial(3, -1)", "falling_factorial: k must be an integer in 0..=18446744073709551615");
    }

}
