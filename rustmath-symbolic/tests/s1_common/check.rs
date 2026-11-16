//! Outcome checks shared by the S1 tests: an expected value (coefficients from gp) or an expected
//! refusal kind, checked through `series_expansion` (with the REPL budget), `taylor_with_budget`
//! (the same budget) and the limit path.
use super::api::*;
use super::eval::{self, Num};
use rustmath_rationals::Rational;
use std::collections::HashMap;

/// The kind of a `SeriesError` (design §2.9), without its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    NotLaurent,
    Unsupported,
    UndecidedZero,
    DivisionByZero,
    InvalidPoint,
    PrecisionExhausted,
    BudgetExceeded,
    LimitExceeded(StructureLimit),
    ValuationBelowMinimum,
    Pole,
}

pub fn kind(e: &SeriesError) -> Kind {
    match e {
        SeriesError::NotLaurent(_) => Kind::NotLaurent,
        SeriesError::Unsupported(_) => Kind::Unsupported,
        SeriesError::UndecidedZero(_) => Kind::UndecidedZero,
        SeriesError::DivisionByZero(_) => Kind::DivisionByZero,
        SeriesError::InvalidPoint(_) => Kind::InvalidPoint,
        SeriesError::PrecisionExhausted { .. } => Kind::PrecisionExhausted,
        SeriesError::BudgetExceeded { .. } => Kind::BudgetExceeded,
        SeriesError::LimitExceeded { limit, .. } => Kind::LimitExceeded(*limit),
        SeriesError::ValuationBelowMinimum { .. } => Kind::ValuationBelowMinimum,
        SeriesError::Pole { .. } => Kind::Pole,
    }
}

/// Parse a kind name as the data files write it ("NotLaurent", "LimitExceeded(TreeSize)", ...).
pub fn kind_from_name(s: &str) -> Kind {
    match s {
        "NotLaurent" => Kind::NotLaurent,
        "Unsupported" => Kind::Unsupported,
        "UndecidedZero" => Kind::UndecidedZero,
        "DivisionByZero" => Kind::DivisionByZero,
        "InvalidPoint" => Kind::InvalidPoint,
        "PrecisionExhausted" => Kind::PrecisionExhausted,
        "BudgetExceeded" => Kind::BudgetExceeded,
        "LimitExceeded(MonomialFactors)" => Kind::LimitExceeded(StructureLimit::MonomialFactors),
        "LimitExceeded(Kernels)" => Kind::LimitExceeded(StructureLimit::Kernels),
        "LimitExceeded(ConstantTerms)" => Kind::LimitExceeded(StructureLimit::ConstantTerms),
        "LimitExceeded(KernelDepth)" => Kind::LimitExceeded(StructureLimit::KernelDepth),
        "LimitExceeded(TreeSize)" => Kind::LimitExceeded(StructureLimit::TreeSize),
        "ValuationBelowMinimum" => Kind::ValuationBelowMinimum,
        "Pole" => Kind::Pole,
        other => panic!("unknown refusal kind {other:?}"),
    }
}

pub fn x() -> Symbol {
    Symbol::new("x")
}

/// One expected coefficient: exponent, real part, imaginary part, as gp prints them
/// (exact rationals or decimals with 70 significant digits).
pub type Coef<'a> = (i64, &'a str, &'a str);

/// What the limit path must return.
#[derive(Debug, Clone, PartialEq)]
pub enum LimWant {
    /// `Finite(q)` with q this rational ("0", "1", "-3/2").
    Finite(String),
    /// `Unknown` (an engine error, a pole, or a constant term that is not rational, §2.13).
    Unknown,
    /// Not checked.
    Skip,
    /// Any one of these. The data files write the list with commas, `Finite(16),Unknown`: a
    /// constant term whose value is the rational q but which the design's constant field keeps as
    /// a kernel (`exp(4 log 2)`, `1 + (sin(1)^2 + cos(1)^2)`), or which an optional fold of the
    /// owner's rule may make rational (`s1_common/mod.rs`). `Unknown` is always honest, and
    /// `Finite(q)` is the true limit.
    AnyOf(Vec<LimWant>),
}

pub fn lim_want_from(s: &str) -> LimWant {
    if s.contains(',') {
        return LimWant::AnyOf(s.split(',').map(|t| lim_want_from(t.trim())).collect());
    }
    match s {
        "Unknown" => LimWant::Unknown,
        "-" => LimWant::Skip,
        s if s.starts_with("Finite(") && s.ends_with(')') => {
            LimWant::Finite(s["Finite(".len()..s.len() - 1].to_string())
        }
        other => panic!("bad limit expectation {other:?}"),
    }
}

fn rational_expr(s: &str) -> Expr {
    let q: Rational = eval::parse_decimal(s);
    if q.denominator().is_one() {
        Expr::Integer(q.numerator().clone())
    } else {
        Expr::Rational(q)
    }
}

/// Whether a limit result is one that `want` accepts.
pub fn limit_matches(got: &LimitResult, want: &LimWant) -> bool {
    match want {
        LimWant::Skip => true,
        LimWant::Unknown => matches!(got, LimitResult::Unknown),
        LimWant::Finite(q) => matches!(got, LimitResult::Finite(v) if *v == rational_expr(q)),
        LimWant::AnyOf(ws) => ws.iter().any(|w| limit_matches(got, w)),
    }
}

/// The limit path's result checked against `want`; Err(message) on a mismatch.
pub fn check_limit(f: &Expr, a: &Expr, want: &LimWant) -> Result<(), String> {
    if matches!(want, LimWant::Skip) {
        return Ok(());
    }
    let got = lim(f, &x(), a);
    if limit_matches(&got, want) {
        Ok(())
    } else {
        Err(format!("limit path: expected {want:?}, got {got:?}"))
    }
}

/// The value of the engine's coefficient of exponent k (0 when absent).
fn coefficient_value(s: &SeriesExpansion, k: i64) -> Result<Num, String> {
    match s.coefficient(k) {
        None => Err(format!(
            "coefficient {k} at or above the precision {}",
            s.precision
        )),
        Some(c) => eval::eval(&c).map_err(|e| format!("coefficient {k} = {c}: {e}")),
    }
}

/// `series_expansion` of f at a up to `order` (REPL budget) has exactly the coefficients `want`
/// (absent exponents are 0), each to `digits` digits, and precision order + 1; then
/// `taylor_with_budget` is `Ok` and equals `to_expr()`, and the limit path gives `lim_want`.
pub fn check_value(
    f_text: &str,
    a_text: &str,
    order: i64,
    want: &[Coef<'_>],
    lim_want: &LimWant,
    digits: u32,
) -> Result<(), String> {
    let f = parse(f_text);
    let a = parse(a_text);
    let s = se(
        &f,
        &x(),
        &ExpansionPoint::At(a.clone()),
        order,
        Some(REPL_TAYLOR_WORK),
    )
    .map_err(|e| format!("series_expansion: expected a value, got Err({e:?})"))?;
    if s.precision != order + 1 {
        return Err(format!(
            "precision {} instead of {}",
            s.precision,
            order + 1
        ));
    }
    let mut want_map: HashMap<i64, Num> = HashMap::new();
    for (k, re, im) in want {
        want_map.insert(*k, eval::parse_coefficient(re, im));
    }
    let lo = want
        .iter()
        .map(|w| w.0)
        .chain(s.first_exponent())
        .min()
        .unwrap_or(0)
        .min(0);
    for k in lo..=order {
        let got = coefficient_value(&s, k)?;
        let w = want_map.remove(&k).unwrap_or_else(eval::zero);
        if !eval::close(&got, &w, digits) {
            return Err(format!(
                "coefficient of h^{k}: engine {} (as {:?}), gp {}",
                eval::show(&got),
                s.coefficient(k).map(|c| c.to_string()),
                eval::show(&w)
            ));
        }
    }
    let t = tw(&f, &x(), &a, order as usize, REPL_TAYLOR_WORK)
        .map_err(|e| format!("taylor_with_budget: expected Ok, got Err({e:?})"))?;
    if t != s.to_expr() {
        return Err("taylor_with_budget differs from series_expansion().to_expr()".into());
    }
    check_limit(&f, &a, lim_want)
}

/// `series_expansion` and `taylor_with_budget` (REPL budget) both refuse with `want`, and the
/// limit path gives `lim_want` (normally `Unknown`).
pub fn check_refusal(
    f_text: &str,
    a_text: &str,
    order: i64,
    want: Kind,
    lim_want: &LimWant,
) -> Result<(), String> {
    let f = parse(f_text);
    let a = parse(a_text);
    match se(
        &f,
        &x(),
        &ExpansionPoint::At(a.clone()),
        order,
        Some(REPL_TAYLOR_WORK),
    ) {
        Err(e) if kind(&e) == want => {}
        Err(e) => {
            return Err(format!(
                "series_expansion: expected {want:?}, got Err({e:?})"
            ))
        }
        Ok(s) => {
            return Err(format!(
                "series_expansion: expected {want:?}, got Ok({})",
                s.to_expr()
            ))
        }
    }
    match tw(&f, &x(), &a, order as usize, REPL_TAYLOR_WORK) {
        Err(e) if kind(&e) == want => {}
        Err(e) => {
            return Err(format!(
                "taylor_with_budget: expected {want:?}, got Err({e:?})"
            ))
        }
        Ok(r) => {
            return Err(format!(
                "taylor_with_budget: expected {want:?}, got Ok({r})"
            ))
        }
    }
    check_limit(&f, &a, lim_want)
}

/// One accepted outcome of an input with several honest outcomes (a row whose outcome depends on
/// a fold that the scope of the owner's rule leaves optional, `s1_common/mod.rs`).
#[derive(Debug, Clone)]
pub enum Outcome<'a> {
    /// The value (gp's coefficients) and what the limit path must then return.
    Value(&'a [Coef<'a>], LimWant),
    /// The refusal kind and what the limit path must then return.
    Refusal(Kind, LimWant),
}

/// `series_expansion` of f at a (REPL budget) has one of `outcomes`: an `Ok` is checked as
/// `check_value` against each `Value` outcome (one must hold), an `Err` must have the kind of a
/// `Refusal` outcome and is then checked as `check_refusal`. Every other result fails, so a row
/// with several outcomes still rejects a wrong value, a refusal of another kind and a panic.
pub fn check_outcomes(
    f_text: &str,
    a_text: &str,
    order: i64,
    outcomes: &[Outcome<'_>],
    digits: u32,
) -> Result<(), String> {
    let f = parse(f_text);
    let a = parse(a_text);
    let accepted = || {
        outcomes
            .iter()
            .map(|o| match o {
                Outcome::Value(..) => "a value".to_string(),
                Outcome::Refusal(k, _) => format!("{k:?}"),
            })
            .collect::<Vec<_>>()
            .join(" or ")
    };
    match se(
        &f,
        &x(),
        &ExpansionPoint::At(a.clone()),
        order,
        Some(REPL_TAYLOR_WORK),
    ) {
        Ok(s) => {
            let mut mismatches = Vec::new();
            for o in outcomes {
                if let Outcome::Value(c, l) = o {
                    match check_value(f_text, a_text, order, c, l, digits) {
                        Ok(()) => return Ok(()),
                        Err(m) => mismatches.push(m),
                    }
                }
            }
            if mismatches.is_empty() {
                Err(format!(
                    "series_expansion: expected {}, got Ok({})",
                    accepted(),
                    s.to_expr()
                ))
            } else {
                Err(mismatches.join("; "))
            }
        }
        Err(e) => {
            for o in outcomes {
                if let Outcome::Refusal(k, l) = o {
                    if *k == kind(&e) {
                        return check_refusal(f_text, a_text, order, *k, l);
                    }
                }
            }
            Err(format!(
                "series_expansion: expected {}, got Err({e:?})",
                accepted()
            ))
        }
    }
}

/// `check_value` for a sparse expansion (exponents such as -(2^63 - 1), where `check_value`'s
/// loop over every exponent from the first one up to `order` cannot run): every stored term of
/// `series_expansion` (REPL budget) equals the wanted coefficient of its exponent (0 when absent)
/// to `digits` digits, every wanted coefficient that is not stored is 0, the precision is
/// order + 1, `taylor_with_budget` equals `to_expr()` (at a finite point) and the limit path gives
/// `lim_want`.
pub fn check_value_sparse(
    f_text: &str,
    point: &ExpansionPoint,
    order: i64,
    want: &[Coef<'_>],
    lim_want: &LimWant,
    digits: u32,
) -> Result<(), String> {
    let f = parse(f_text);
    let s = se(&f, &x(), point, order, Some(REPL_TAYLOR_WORK))
        .map_err(|e| format!("series_expansion: expected a value, got Err({e:?})"))?;
    if s.precision != order + 1 {
        return Err(format!(
            "precision {} instead of {}",
            s.precision,
            order + 1
        ));
    }
    let mut want_map: HashMap<i64, Num> = HashMap::new();
    for (k, re, im) in want {
        want_map.insert(*k, eval::parse_coefficient(re, im));
    }
    for (k, c) in &s.terms {
        let got = eval::eval(c).map_err(|e| format!("coefficient {k} = {c}: {e}"))?;
        let w = want_map.remove(k).unwrap_or_else(eval::zero);
        if !eval::close(&got, &w, digits) {
            return Err(format!(
                "coefficient of h^{k}: engine {} (as {c}), gp {}",
                eval::show(&got),
                eval::show(&w)
            ));
        }
    }
    for (k, w) in want_map {
        if !eval::close(&eval::zero(), &w, digits) {
            return Err(format!("no term of exponent {k}, gp {}", eval::show(&w)));
        }
    }
    if let ExpansionPoint::At(a) = point {
        let usize_order = usize::try_from(order).map_err(|_| "negative order".to_string())?;
        let t = tw(&f, &x(), a, usize_order, REPL_TAYLOR_WORK)
            .map_err(|e| format!("taylor_with_budget: expected Ok, got Err({e:?})"))?;
        if t != s.to_expr() {
            return Err("taylor_with_budget differs from series_expansion().to_expr()".into());
        }
        check_limit(&f, a, lim_want)?;
    }
    Ok(())
}

/// Run `f` on a thread with `stack` bytes of stack (the bar: 2 MB in debug, 8 MB in release).
pub fn on_stack<T: Send + 'static>(stack: usize, f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .stack_size(stack)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("the test thread panicked (or overflowed its stack)")
}

/// The number of nodes of e as a tree (every shared subtree counted at each occurrence),
/// computed over the DAG (memoized by address, explicit stack), saturating.
pub fn tree_size(root: &Expr) -> u64 {
    let mut memo: HashMap<*const Expr, u64> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    while let Some((n, done)) = stack.pop() {
        let key = n as *const Expr;
        if memo.contains_key(&key) {
            continue;
        }
        let children: Vec<&Expr> = match n {
            Expr::Binary(_, l, r) => vec![&**l, &**r],
            Expr::Unary(_, a) => vec![&**a],
            Expr::Function(_, args) => args.iter().map(|a| &**a).collect(),
            _ => vec![],
        };
        if !done {
            stack.push((n, true));
            for c in children {
                stack.push((c, false));
            }
            continue;
        }
        let s = children.iter().fold(1u64, |acc, c| {
            acc.saturating_add(memo[&(*c as *const Expr)])
        });
        memo.insert(key, s);
    }
    memo[&(root as *const Expr)]
}

/// The number of distinct nodes (by address) of e.
pub fn dag_size(root: &Expr) -> usize {
    let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
    let mut stack: Vec<&Expr> = vec![root];
    while let Some(n) = stack.pop() {
        if !seen.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Binary(_, l, r) => {
                stack.push(&**l);
                stack.push(&**r);
            }
            Expr::Unary(_, a) => stack.push(&**a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    seen.len()
}

/// Whether e mentions the symbol `name` (DAG walk).
pub fn mentions(root: &Expr, name: &str) -> bool {
    let mut seen: std::collections::HashSet<*const Expr> = std::collections::HashSet::new();
    let mut stack: Vec<&Expr> = vec![root];
    while let Some(n) = stack.pop() {
        if !seen.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Symbol(s) if s.name() == name => return true,
            Expr::Binary(_, l, r) => {
                stack.push(&**l);
                stack.push(&**r);
            }
            Expr::Unary(_, a) => stack.push(&**a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    false
}

/// The peak resident set (VmHWM) of this process in bytes (Linux), for the release budget tests.
pub fn vm_hwm_bytes() -> Option<u64> {
    let st = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = st.lines().find(|l| l.starts_with("VmHWM"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The bar's budget-hole limits (S1_BAR (c)): 20 s in release, 512 MB.
pub const BAR_SECONDS: f64 = 20.0;
pub const BAR_BYTES: u64 = 512 * 1024 * 1024;

/// A data-file row: `id \t input \t point \t order \t want \t limit \t note`, where want is
/// `value v re:im ...` (gp's valuation, then one coefficient per exponent from v to order) or
/// `err Kind`, and limit is `Finite(q)`, `Unknown`, `-` (not checked) or a comma list of these (any
/// one of them). A row with several honest outcomes (a fold the scope of the owner's rule leaves
/// optional, `s1_common/mod.rs`) lists them in want separated by ` | `, and its limit column lists
/// the limit for each outcome in the same order, separated by ` | ` (or one limit for all).
#[derive(Debug, Clone)]
pub struct Row {
    pub id: String,
    pub input: String,
    pub point: String,
    pub order: i64,
    pub want: RowWant,
    pub limit: LimWant,
    pub note: String,
}

#[derive(Debug, Clone)]
pub enum RowWant {
    Value(Vec<(i64, String, String)>),
    Refusal(Kind),
    /// Several honest outcomes, each with its limit (the row's `limit` is then `Skip`).
    OneOf(Vec<(RowWant, LimWant)>),
}

fn parse_want(s: &str, ln: usize) -> RowWant {
    if let Some(rest) = s.strip_prefix("value ") {
        let mut it = rest.split(' ');
        let v: i64 = it.next().expect("valuation").parse().expect("valuation");
        let coefs = it
            .enumerate()
            .map(|(j, c)| {
                let (re, im) = c
                    .split_once(':')
                    .unwrap_or_else(|| panic!("line {ln}: {c}"));
                (v + j as i64, re.to_string(), im.to_string())
            })
            .collect();
        RowWant::Value(coefs)
    } else if let Some(k) = s.strip_prefix("err ") {
        RowWant::Refusal(kind_from_name(k))
    } else {
        panic!("line {ln}: bad expectation {s:?}");
    }
}

pub fn parse_rows(text: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() >= 6,
            "line {}: {} fields: {line:?}",
            ln + 1,
            f.len()
        );
        let (want, limit) = if f[4].contains(" | ") {
            let wants: Vec<&str> = f[4].split(" | ").collect();
            let lims: Vec<&str> = f[5].split(" | ").collect();
            assert!(
                lims.len() == 1 || lims.len() == wants.len(),
                "line {}: {} outcomes but {} limits",
                ln + 1,
                wants.len(),
                lims.len()
            );
            let alts = wants
                .iter()
                .enumerate()
                .map(|(j, w)| {
                    let l = lims[if lims.len() == 1 { 0 } else { j }];
                    (parse_want(w, ln + 1), lim_want_from(l))
                })
                .collect();
            (RowWant::OneOf(alts), LimWant::Skip)
        } else {
            (parse_want(f[4], ln + 1), lim_want_from(f[5]))
        };
        rows.push(Row {
            id: f[0].to_string(),
            input: f[1].to_string(),
            point: f[2].to_string(),
            order: f[3].parse().expect("order"),
            want,
            limit,
            note: f.get(6).map(|s| s.to_string()).unwrap_or_default(),
        });
    }
    rows
}

/// Check one data row; Err(message) on a mismatch.
pub fn check_row(r: &Row, digits: u32) -> Result<(), String> {
    match &r.want {
        RowWant::Value(c) => {
            let coefs: Vec<Coef<'_>> = c
                .iter()
                .map(|(k, re, im)| (*k, re.as_str(), im.as_str()))
                .collect();
            check_value(&r.input, &r.point, r.order, &coefs, &r.limit, digits)
        }
        RowWant::Refusal(k) => check_refusal(&r.input, &r.point, r.order, *k, &r.limit),
        RowWant::OneOf(alts) => {
            let coefs: Vec<Vec<Coef<'_>>> = alts
                .iter()
                .map(|(w, _)| match w {
                    RowWant::Value(c) => c
                        .iter()
                        .map(|(k, re, im)| (*k, re.as_str(), im.as_str()))
                        .collect(),
                    _ => Vec::new(),
                })
                .collect();
            let outcomes: Vec<Outcome<'_>> = alts
                .iter()
                .zip(&coefs)
                .map(|((w, l), c)| match w {
                    RowWant::Value(_) => Outcome::Value(c.as_slice(), l.clone()),
                    RowWant::Refusal(k) => Outcome::Refusal(*k, l.clone()),
                    RowWant::OneOf(_) => panic!("row {}: nested outcomes", r.id),
                })
                .collect();
            check_outcomes(&r.input, &r.point, r.order, &outcomes, digits)
        }
    }
}

/// Check every row (each on a thread with `stack` bytes), and fail with the list of mismatches.
pub fn check_all(rows: Vec<Row>, digits: u32, stack: usize) {
    let total = rows.len();
    let mut failures = Vec::new();
    for r in rows {
        let id = r.id.clone();
        let desc = format!("{} at {}, order {}", r.input, r.point, r.order);
        let res = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || check_row(&r, digits))
            .expect("spawn")
            .join();
        match res {
            Ok(Ok(())) => {}
            Ok(Err(m)) => failures.push(format!("{id} [{desc}]: {m}")),
            Err(_) => failures.push(format!("{id} [{desc}]: panicked (or overflowed its stack)")),
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} of {} rows failed:\n{}",
            failures.len(),
            total,
            failures.join("\n")
        );
    }
}
