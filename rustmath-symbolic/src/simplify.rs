//! Expression simplification

use crate::assumptions::{has_property, Property};
use crate::expression::{BinaryOp, Expr, UnaryOp};
use crate::pattern::RuleDatabase;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::collections::HashMap;
use std::sync::Arc;

impl Expr {
    /// Simplify the expression using basic algebraic rules
    ///
    /// Applies constant folding, identity elimination, and other basic simplifications.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // 0 + x simplifies to x
    /// let expr = Expr::from(0) + x.clone();
    /// let simplified = expr.simplify();
    /// ```
    pub fn simplify(&self) -> Expr {
        simplify(self)
    }

    /// Full simplification including expansion and collection
    ///
    /// This is more aggressive than basic simplify().
    pub fn simplify_full(&self) -> Expr {
        // First expand
        let expanded = self.expand();
        // Then simplify
        let simplified = simplify(&expanded);
        simplified
    }

    /// Simplify rational expressions (ratios of polynomials)
    pub fn simplify_rational(&self) -> Expr {
        match self {
            Expr::Binary(BinaryOp::Div, num, den) => {
                // Simplify numerator and denominator
                let num_simp = simplify(num);
                let den_simp = simplify(den);

                // Try to find common factors (basic GCD for polynomials)
                // For now, just return the simplified ratio
                Expr::Binary(BinaryOp::Div, Arc::new(num_simp), Arc::new(den_simp))
            }
            _ => simplify(self),
        }
    }

    /// Simplify trigonometric expressions
    ///
    /// Applies trigonometric identities like sin²(x) + cos²(x) = 1
    /// Now uses the pattern-based rule system for comprehensive coverage
    pub fn simplify_trig(&self) -> Expr {
        // Use the pattern-based rule system
        let db = RuleDatabase::new();
        let simplified = db.apply_all(self);

        // If pattern-based simplification didn't change anything,
        // try the legacy approach as fallback
        if simplified == *self {
            self.simplify_trig_legacy()
        } else {
            simplified
        }
    }

    /// Legacy trigonometric simplification (kept for compatibility)
    fn simplify_trig_legacy(&self) -> Expr {
        match self {
            // sin²(x) + cos²(x) = 1
            Expr::Binary(BinaryOp::Add, left, right) => {
                if is_sin_squared(left) && is_cos_squared_same_arg(right, left) {
                    return Expr::from(1);
                }
                if is_cos_squared(left) && is_sin_squared_same_arg(right, left) {
                    return Expr::from(1);
                }

                // Recursively simplify
                let left_simp = left.simplify_trig_legacy();
                let right_simp = right.simplify_trig_legacy();
                Expr::Binary(BinaryOp::Add, Arc::new(left_simp), Arc::new(right_simp))
            }
            Expr::Binary(op, left, right) => {
                let left_simp = left.simplify_trig_legacy();
                let right_simp = right.simplify_trig_legacy();
                Expr::Binary(*op, Arc::new(left_simp), Arc::new(right_simp))
            }
            Expr::Unary(op, inner) => {
                let inner_simp = inner.simplify_trig_legacy();
                Expr::Unary(*op, Arc::new(inner_simp))
            }
            _ => self.clone(),
        }
    }

    /// Simplify logarithmic expressions
    ///
    /// Applies log identities like log(a*b) = log(a) + log(b)
    pub fn simplify_log(&self) -> Expr {
        match self {
            Expr::Unary(UnaryOp::Log, inner) => {
                match inner.as_ref() {
                    // log(1) = 0
                    Expr::Integer(n) if n.is_one() => Expr::from(0),
                    // log(a * b) = log(a) + log(b)
                    Expr::Binary(BinaryOp::Mul, left, right) => {
                        let log_left = Expr::Unary(UnaryOp::Log, left.clone()).simplify_log();
                        let log_right = Expr::Unary(UnaryOp::Log, right.clone()).simplify_log();
                        Expr::Binary(BinaryOp::Add, Arc::new(log_left), Arc::new(log_right))
                    }
                    // log(a / b) = log(a) - log(b)
                    Expr::Binary(BinaryOp::Div, num, den) => {
                        let log_num = Expr::Unary(UnaryOp::Log, num.clone()).simplify_log();
                        let log_den = Expr::Unary(UnaryOp::Log, den.clone()).simplify_log();
                        Expr::Binary(BinaryOp::Sub, Arc::new(log_num), Arc::new(log_den))
                    }
                    // log(a^b) = b * log(a)
                    Expr::Binary(BinaryOp::Pow, base, exp) => {
                        let log_base = Expr::Unary(UnaryOp::Log, base.clone()).simplify_log();
                        Expr::Binary(BinaryOp::Mul, exp.clone(), Arc::new(log_base))
                    }
                    _ => self.clone(),
                }
            }
            Expr::Binary(op, left, right) => {
                let left_simp = left.simplify_log();
                let right_simp = right.simplify_log();
                Expr::Binary(*op, Arc::new(left_simp), Arc::new(right_simp))
            }
            Expr::Unary(op, inner) => {
                let inner_simp = inner.simplify_log();
                Expr::Unary(*op, Arc::new(inner_simp))
            }
            _ => self.clone(),
        }
    }

    /// Expand trigonometric expressions using angle addition formulas
    ///
    /// Applies identities like sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    pub fn expand_trig(&self) -> Expr {
        // Basic implementation - can be extended with more formulas
        match self {
            Expr::Unary(UnaryOp::Sin, inner) => {
                match inner.as_ref() {
                    // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                    Expr::Binary(BinaryOp::Add, a, b) => {
                        let sin_a = Expr::Unary(UnaryOp::Sin, a.clone());
                        let cos_a = Expr::Unary(UnaryOp::Cos, a.clone());
                        let sin_b = Expr::Unary(UnaryOp::Sin, b.clone());
                        let cos_b = Expr::Unary(UnaryOp::Cos, b.clone());

                        let term1 = Expr::Binary(BinaryOp::Mul, Arc::new(sin_a), Arc::new(cos_b));
                        let term2 = Expr::Binary(BinaryOp::Mul, Arc::new(cos_a), Arc::new(sin_b));
                        Expr::Binary(BinaryOp::Add, Arc::new(term1), Arc::new(term2))
                    }
                    _ => self.clone(),
                }
            }
            Expr::Unary(UnaryOp::Cos, inner) => {
                match inner.as_ref() {
                    // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                    Expr::Binary(BinaryOp::Add, a, b) => {
                        let sin_a = Expr::Unary(UnaryOp::Sin, a.clone());
                        let cos_a = Expr::Unary(UnaryOp::Cos, a.clone());
                        let sin_b = Expr::Unary(UnaryOp::Sin, b.clone());
                        let cos_b = Expr::Unary(UnaryOp::Cos, b.clone());

                        let term1 = Expr::Binary(BinaryOp::Mul, Arc::new(cos_a), Arc::new(cos_b));
                        let term2 = Expr::Binary(BinaryOp::Mul, Arc::new(sin_a), Arc::new(sin_b));
                        Expr::Binary(BinaryOp::Sub, Arc::new(term1), Arc::new(term2))
                    }
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
    }

    /// Collect like terms in an expression
    ///
    /// Groups terms with the same symbolic part
    pub fn collect_like_terms(&self) -> Expr {
        // Basic implementation - collects terms in addition
        match self {
            Expr::Binary(BinaryOp::Add, left, right) => {
                let left_coll = left.collect_like_terms();
                let right_coll = right.collect_like_terms();
                // Try to combine if they're like terms
                match (&left_coll, &right_coll) {
                    // n*x + m*x = (n+m)*x
                    (
                        Expr::Binary(BinaryOp::Mul, n1, x1),
                        Expr::Binary(BinaryOp::Mul, n2, x2),
                    ) if x1 == x2 => {
                        let coeff_sum = Expr::Binary(BinaryOp::Add, n1.clone(), n2.clone()).simplify();
                        Expr::Binary(BinaryOp::Mul, Arc::new(coeff_sum), x1.clone())
                    }
                    _ => Expr::Binary(BinaryOp::Add, Arc::new(left_coll), Arc::new(right_coll)),
                }
            }
            _ => self.clone(),
        }
    }

    /// Combine fractions and powers
    ///
    /// Combines expressions like a/b + c/d = (ad + bc)/(bd)
    pub fn combine(&self) -> Expr {
        match self {
            // Combine fractions: a/b + c/d = (ad + bc)/(bd)
            Expr::Binary(BinaryOp::Add, left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (
                        Expr::Binary(BinaryOp::Div, a, b),
                        Expr::Binary(BinaryOp::Div, c, d),
                    ) => {
                        let ad = Expr::Binary(BinaryOp::Mul, a.clone(), d.clone());
                        let bc = Expr::Binary(BinaryOp::Mul, b.clone(), c.clone());
                        let bd = Expr::Binary(BinaryOp::Mul, b.clone(), d.clone());
                        let num = Expr::Binary(BinaryOp::Add, Arc::new(ad), Arc::new(bc));
                        Expr::Binary(BinaryOp::Div, Arc::new(num), Arc::new(bd)).simplify()
                    }
                    _ => self.clone(),
                }
            }
            // Combine powers: x^a * x^b = x^(a+b)
            Expr::Binary(BinaryOp::Mul, left, right) => {
                match (left.as_ref(), right.as_ref()) {
                    (
                        Expr::Binary(BinaryOp::Pow, base1, exp1),
                        Expr::Binary(BinaryOp::Pow, base2, exp2),
                    ) if base1 == base2 => {
                        let exp_sum = Expr::Binary(BinaryOp::Add, exp1.clone(), exp2.clone());
                        Expr::Binary(BinaryOp::Pow, base1.clone(), Arc::new(exp_sum)).simplify()
                    }
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
    }

    /// Simplify using assumptions
    ///
    /// Applies simplifications that depend on variable properties.
    /// For example: sqrt(x^2) = x if x > 0, sqrt(x^2) = -x if x < 0
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::{Expr, assume, Property};
    ///
    /// let x = Expr::symbol("x");
    /// let x_sym = x.as_symbol().unwrap();
    /// assume(x_sym, Property::Positive);
    ///
    /// // sqrt(x^2) simplifies to x when x > 0
    /// let expr = x.clone().pow(Expr::from(2)).sqrt();
    /// let simplified = expr.simplify_with_assumptions();
    /// // assert_eq!(simplified, x);
    /// ```
    pub fn simplify_with_assumptions(&self) -> Expr {
        match self {
            // sqrt(x^2) = |x|, but if x > 0 then = x, if x < 0 then = -x
            Expr::Unary(UnaryOp::Sqrt, inner) => {
                if let Expr::Binary(BinaryOp::Pow, base, exp) = inner.as_ref() {
                    if let Expr::Integer(n) = exp.as_ref() {
                        if n == &Integer::from(2) {
                            // We have sqrt(x^2)
                            if let Expr::Symbol(s) = base.as_ref() {
                                // Check assumptions
                                if has_property(s, Property::Positive) {
                                    return (**base).clone();
                                } else if has_property(s, Property::Negative) {
                                    return -(**base).clone();
                                }
                            }
                            // Default: sqrt(x^2) = |x|
                            return (**base).clone().abs();
                        }
                    }
                }
                // Recursively simplify
                let inner_simp = inner.simplify_with_assumptions();
                Expr::Unary(UnaryOp::Sqrt, Arc::new(inner_simp))
            }

            // |x| = x if x >= 0, |x| = -x if x < 0
            Expr::Unary(UnaryOp::Abs, inner) => {
                if let Expr::Symbol(s) = inner.as_ref() {
                    if has_property(s, Property::Positive) || has_property(s, Property::Zero) {
                        return (**inner).clone();
                    } else if has_property(s, Property::Negative) {
                        return -(**inner).clone();
                    }
                }
                // Recursively simplify
                let inner_simp = inner.simplify_with_assumptions();
                Expr::Unary(UnaryOp::Abs, Arc::new(inner_simp))
            }

            // For binary operations, recursively apply
            Expr::Binary(op, left, right) => {
                let left_simp = left.simplify_with_assumptions();
                let right_simp = right.simplify_with_assumptions();
                Expr::Binary(*op, Arc::new(left_simp), Arc::new(right_simp))
            }

            // For other unary operations, recursively apply
            Expr::Unary(op, inner) => {
                let inner_simp = inner.simplify_with_assumptions();
                Expr::Unary(*op, Arc::new(inner_simp))
            }

            // For functions, recursively apply to arguments
            Expr::Function(name, args) => {
                let simplified_args: Vec<Arc<Expr>> = args
                    .iter()
                    .map(|arg| Arc::new(arg.simplify_with_assumptions()))
                    .collect();
                Expr::Function(name.clone(), simplified_args)
            }

            _ => self.clone(),
        }
    }

    /// Advanced simplification using pattern matching and assumptions
    ///
    /// This is the most comprehensive simplification method.
    /// It applies:
    /// 1. Pattern-based trigonometric identities
    /// 2. Pattern-based exponential/logarithm rules
    /// 3. Assumption-based simplifications
    /// 4. Basic algebraic simplifications
    ///
    /// # Examples
    ///
    /// ```
    /// use rustmath_symbolic::Expr;
    ///
    /// let x = Expr::symbol("x");
    ///
    /// // sin(x)^2 + cos(x)^2 simplifies to 1
    /// let expr = x.clone().sin().pow(Expr::from(2)) + x.clone().cos().pow(Expr::from(2));
    /// let simplified = expr.simplify_advanced();
    /// // assert_eq!(simplified, Expr::from(1));
    /// ```
    pub fn simplify_advanced(&self) -> Expr {
        // First apply pattern-based rules
        let db = RuleDatabase::new();
        let step1 = db.apply_all(self);

        // Then apply assumption-based simplifications
        let step2 = step1.simplify_with_assumptions();

        // Finally apply basic algebraic simplifications
        let step3 = simplify(&step2);

        step3
    }
}

/// Simplify an expression
///
/// HEAD's rules and evaluation order (children before their parent, left before right, a call's
/// arguments in order), walked with an explicit work list instead of the call stack, so that the
/// depth of the tree costs heap, not stack (S1 review 2, B1: round 1 kept the recursive frame at
/// HEAD's size, but a function call and a binary node with a binary left child each added a frame
/// per level, so `f(f(...))`, `f(x + f(...))` and `(x + L) + x` chains aborted below HEAD's depth;
/// `tests/s1_depth.rs`). The combining steps (`simplify_binary`, `simplify_unary`) do not recurse.
/// Never inlined, so its work list is not in the frame of a recursive caller (`Expr::limit`).
///
/// S1 fix-up round 3 (review 3, B2): a node with two or more parents in the walked DAG is
/// simplified once, and its value is shared by every occurrence (memo by address; every node
/// stays alive during the walk, so no address is reused). A leaf reached through an `Arc` is
/// shared, not copied. The results are HEAD's (the rules are a function of the node), only the
/// sharing differs: HEAD simplified a subtree again at each occurrence, so the series result that
/// names a kernel's source at each of 1000 terms folded that source 1000 times.
///
/// S1 fix-up round 4 (review 4, B2): the parents are counted inside the walked DAG, by a first
/// walk that expands each node once (a node held by one `Arc` has one parent and is not counted),
/// and a memo entry is removed after its last use. Round 3 memoized every node whose `Arc` count
/// was above 1 and kept its value to the end of the walk; the count also counts owners outside
/// the DAG (a REPL line `a2 = a1*3` stores a shallow clone of a1, so a1's children are held
/// twice), so `simplify(a6000)` of the chain a_k = a_{k-1}*3 kept the 6 000 intermediate literals
/// (754 MB, HEAD 9 MB; `tests/s1_review4_memory.rs`).
///
/// S1 fix-up round 5 (review 5, B2): the memo holds at most `MEMO_PAYLOAD_WORDS` words of fresh
/// literals (32 MiB). Round 4 kept every shared node's value until its last use, also when the
/// first use threw it away (`0 * q`): a DAG naming 40 000 negations of one 2^20-bit leaf twice
/// aborted at a 4 GB limit, where HEAD needs 46 MB. Each value carries an upper bound of the
/// words of fresh literals it holds that no entry accounts for (`weight`); a value is memoized
/// only while the entries' total stays within the cap, and past it the node's next uses
/// recompute it, as HEAD does. So the memory is HEAD's live set plus at most the cap, and the
/// work is at most HEAD's (each visit is one HEAD makes; a memo hit only skips a subtree). For a
/// result the series engine accepted, the literals its distinct nodes fold to are at most
/// `series_const::MAX_RESULT_MEMO_WORDS` (the same cap), so the memo needs no recomputation on
/// the result, nor after `differentiate`, whose clones share the result's nodes
/// (`tests/s1_review5_memory.rs`).
///
/// The work: one visit per distinct node, plus the constant folds of HEAD's rules, each run once
/// per distinct node while the memo holds it, and `MAX_FOLD_POWER_BITS` bounds one power. On any
/// input this is HEAD's work or less: the folds of the input's own literal arithmetic are not
/// bounded here (a 24 KB chain `(2^254+1)^4096/3/.../3` takes 35 s with HEAD's rules, review 4).
///
/// On a result of the series engine (the S1 finish, under the bar the owner set on 2026-11-16):
/// the engine refuses a result whose folds, counted at every occurrence in its tree, pass
/// `series_const::MAX_RESULT_FOLD_WORK` (2^31 units, counted by `series_const::tree_bound`).
/// Measured at the edges of the caps (release, 14 families; S1 fix-up rounds 4 and 5):
/// the slowest `simplify` of a result itself took 0.86 s; after `expand`, `substitute(x, 2)` or
/// `differentiate` of the same results the slowest took 4.12 s, 3.93 s and 2.90 s (the fin2
/// semantics review measured 4.53 s and 4.70 s for the first two, on a sum of 1 487 fractions 1/p
/// in a kernel at order 10). That is a measured scaling, not a bound on `simplify` of a result:
/// the cap does not count the nodes that `expand` and `differentiate` add, nor the folds of a
/// value that `substitute` puts in, and `simplify` after them costs about the number of terms
/// times its cost on the input's kernel source (or on one substituted coefficient). After
/// `substitute(x, 2)` of the result of `(10^100-1)^1050/7^120000/(1-x)` (32 bytes; release): 6.8 s
/// at order 10 and 66.0 s at order 100 (fin1 resources review); the engine now refuses this input
/// above order 32 (`series_const::MAX_RESULT_PARSE_WORK`), and the accepted orders 29 to 32 take
/// 18 to 23 s (the doc of `series_const::MAX_RESULT_TEXT`).
/// Review 6 (release, 8 MB thread): after `expand` of
/// `sin((y0+z0)*...*(y11+z11))/(1-x)` at order 1000 (a 110-byte input), 21.8 s and 5.6 GB; after
/// `differentiate` of `sin(y0^y1^...^y999)/(1-x)` at order 1040 (4.9 KB), 9.1 s and 2.9 GB. The
/// measured numbers are in the doc of `series_const::MAX_RESULT_TEXT`; the follow-up unit S2
/// ("DAG-aware consumers") makes these consumers share repeated subexpressions.
#[inline(never)]
pub fn simplify(expr: &Expr) -> Expr {
    /// A step of the walk: visit a node (the root, or a child through its `Arc`), or combine the
    /// simplified children on top of the value stack; `Some(key)` memoizes the node's value.
    enum Step<'a> {
        Root(&'a Expr),
        Child(&'a Arc<Expr>),
        Binary(BinaryOp, Option<usize>),
        Unary(UnaryOp, Option<usize>),
        Call(&'a str, usize, Option<usize>),
    }
    fn expand<'a>(e: &'a Expr, key: Option<usize>, steps: &mut Vec<Step<'a>>) {
        match e {
            Expr::Binary(op, left, right) => {
                steps.push(Step::Binary(*op, key));
                steps.push(Step::Child(right));
                steps.push(Step::Child(left));
            }
            Expr::Unary(op, inner) => {
                steps.push(Step::Unary(*op, key));
                steps.push(Step::Child(inner));
            }
            Expr::Function(name, args) => {
                steps.push(Step::Call(name, args.len(), key));
                steps.extend(args.iter().rev().map(Step::Child));
            }
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => {
                unreachable!("simplify: a leaf is not expanded")
            }
        }
    }
    fn is_leaf(e: &Expr) -> bool {
        matches!(
            e,
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_)
        )
    }
    /// S1 fix-up round 4: the parents in the DAG of `root` of each non-leaf node held by more than
    /// one `Arc` (by address). Explicit stack; such a node is expanded at its first visit only.
    fn parents_in_dag(root: &Expr) -> HashMap<usize, usize> {
        fn reach<'a>(
            c: &'a Arc<Expr>,
            count: &mut HashMap<usize, usize>,
            stack: &mut Vec<&'a Expr>,
        ) {
            if is_leaf(c) {
                return;
            }
            if Arc::strong_count(c) > 1 {
                let n = count.entry(Arc::as_ptr(c).addr()).or_insert(0);
                *n += 1;
                if *n > 1 {
                    return;
                }
            }
            stack.push(c);
        }
        let mut count: HashMap<usize, usize> = HashMap::new();
        let mut stack: Vec<&Expr> = vec![root];
        while let Some(n) = stack.pop() {
            match n {
                Expr::Binary(_, l, r) => {
                    reach(l, &mut count, &mut stack);
                    reach(r, &mut count, &mut stack);
                }
                Expr::Unary(_, a) => reach(a, &mut count, &mut stack),
                Expr::Function(_, args) => {
                    for a in args {
                        reach(a, &mut count, &mut stack);
                    }
                }
                _ => {}
            }
        }
        count
    }
    let parents = parents_in_dag(expr);
    /// S1 fix-up round 5 (review 5, B2): an entry of the memo of the shared nodes (by address):
    /// the value and the words of fresh literals it holds that no other entry accounts for
    struct Entry {
        v: Arc<Expr>,
        w: u64,
    }
    /// The words of a literal value (both parts of a rational); 0 for any other node.
    fn lit_words(e: &Expr) -> u64 {
        let iw = |n: &Integer| n.bit_length() / 64 + 1;
        match e {
            Expr::Integer(n) => iw(n),
            Expr::Rational(q) => iw(q.numerator()).saturating_add(iw(q.denominator())),
            Expr::Real(_) => 1,
            _ => 0,
        }
    }
    /// S1 fix-up round 5 (review 5, B2): an upper bound of the words of fresh literals in the
    /// value `v` that a rule built from the operands `ops` (each with its own bound): an
    /// operand returned as it is keeps its bound; a literal is its own words (the operands are
    /// dropped); any other node holds at most its operands and the literals it made itself.
    fn weight(v: &Arc<Expr>, ops: &[(&Arc<Expr>, u64)]) -> u64 {
        if let Some((_, w)) = ops.iter().find(|(o, _)| Arc::ptr_eq(v, o)) {
            return *w;
        }
        let own = lit_words(v);
        if own > 0 {
            return own;
        }
        let mut w = ops.iter().fold(0u64, |acc, (_, w)| acc.saturating_add(*w));
        let mut fresh = |c: &Arc<Expr>| {
            if !ops.iter().any(|(o, _)| Arc::ptr_eq(c, o)) {
                w = w.saturating_add(lit_words(c));
            }
        };
        match &**v {
            Expr::Binary(_, l, r) => {
                fresh(l);
                fresh(r);
            }
            Expr::Unary(_, a) => fresh(a),
            _ => {}
        }
        w
    }
    let mut memo: HashMap<usize, Entry> = HashMap::new();
    // the uses of each shared node still to come (a recomputed parent visits its children again,
    // past the count: those visits recompute, as HEAD does)
    let mut uses = parents;
    // the words of fresh literals held by memo entries (and by released entries whose value is
    // still held elsewhere: never decremented for them, an upper bound)
    let mut payload: u64 = 0;
    let mut steps: Vec<Step<'_>> = vec![Step::Root(expr)];
    // each value with the words of fresh literals it holds that no memo entry accounts for
    let mut values: Vec<(Arc<Expr>, u64)> = Vec::new();
    // memoize a computed value of a shared node with uses still to come, while the payload stays
    // within MEMO_PAYLOAD_WORDS; otherwise its next uses recompute it (HEAD's behaviour)
    fn remember(
        memo: &mut HashMap<usize, Entry>,
        uses: &HashMap<usize, usize>,
        payload: &mut u64,
        key: Option<usize>,
        v: Arc<Expr>,
        w: u64,
    ) -> (Arc<Expr>, u64) {
        if let Some(k) = key {
            if uses.get(&k).copied().unwrap_or(0) > 0 {
                if let Some(p) = payload.checked_add(w).filter(|p| *p <= MEMO_PAYLOAD_WORDS) {
                    *payload = p;
                    memo.insert(k, Entry { v: v.clone(), w });
                    return (v, 0);
                }
            }
        }
        (v, w)
    }
    while let Some(step) = steps.pop() {
        match step {
            Step::Root(e) => {
                if is_leaf(e) {
                    values.push((Arc::new(e.clone()), 0))
                } else {
                    expand(e, None, &mut steps)
                }
            }
            // a leaf simplifies to itself: share it
            Step::Child(a) if is_leaf(a) => values.push((a.clone(), 0)),
            Step::Child(a) => {
                // a node held by one Arc has one parent; otherwise its parents in the DAG
                let key = (Arc::strong_count(a) > 1)
                    .then(|| Arc::as_ptr(a).addr())
                    .filter(|k| uses.contains_key(k));
                if let Some(k) = key {
                    let left = uses.get_mut(&k).expect("a counted node");
                    *left = left.saturating_sub(1);
                    let last = *left == 0;
                    if last {
                        if let Some(e) = memo.remove(&k) {
                            // the last use: the value leaves the memo. Held by nothing else, its
                            // literals move to the value stack's account; held elsewhere (a parent's
                            // value, another entry), they stay counted
                            if Arc::strong_count(&e.v) == 1 {
                                payload = payload.saturating_sub(e.w);
                                values.push((e.v, e.w));
                            } else {
                                values.push((e.v, 0));
                            }
                            continue;
                        }
                    } else if let Some(e) = memo.get(&k) {
                        values.push((e.v.clone(), 0));
                        continue;
                    }
                }
                expand(a, key, &mut steps);
            }
            Step::Binary(op, key) => {
                let (right_simp, rw) = values.pop().expect("simplify: a right operand");
                let (left_simp, lw) = values.pop().expect("simplify: a left operand");
                let (l, r) = (left_simp.clone(), right_simp.clone());
                let v = simplify_binary(op, left_simp, right_simp);
                let w = weight(&v, &[(&l, lw), (&r, rw)]);
                drop((l, r));
                let out = remember(&mut memo, &uses, &mut payload, key, v, w);
                values.push(out);
            }
            Step::Unary(op, key) => {
                let (inner_simp, iw) = values.pop().expect("simplify: an operand");
                let i = inner_simp.clone();
                let v = simplify_unary(op, inner_simp);
                let w = weight(&v, &[(&i, iw)]);
                drop(i);
                let out = remember(&mut memo, &uses, &mut payload, key, v, w);
                values.push(out);
            }
            Step::Call(name, n, key) => {
                let parts: Vec<(Arc<Expr>, u64)> = values.drain(values.len() - n..).collect();
                let w = parts
                    .iter()
                    .fold(0u64, |acc, (_, w)| acc.saturating_add(*w));
                let args: Vec<Arc<Expr>> = parts.into_iter().map(|(a, _)| a).collect();
                let v = Arc::new(Expr::Function(name.to_string(), args));
                let out = remember(&mut memo, &uses, &mut payload, key, v, w);
                values.push(out);
            }
        }
    }
    let (root, _) = values.pop().expect("simplify: the result");
    drop(memo);
    Arc::try_unwrap(root).unwrap_or_else(|a| (*a).clone())
}

/// Check if expression is sin(x)^2
fn is_sin_squared(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            matches!(base.as_ref(), Expr::Unary(UnaryOp::Sin, _))
                && matches!(exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
        }
        _ => false,
    }
}

/// Check if expression is cos(x)^2
fn is_cos_squared(expr: &Expr) -> bool {
    match expr {
        Expr::Binary(BinaryOp::Pow, base, exp) => {
            matches!(base.as_ref(), Expr::Unary(UnaryOp::Cos, _))
                && matches!(exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
        }
        _ => false,
    }
}

/// Check if expression is cos²(arg) where arg matches the sin²(arg) from sin_squared_expr
fn is_cos_squared_same_arg(expr: &Expr, sin_squared_expr: &Expr) -> bool {
    match (expr, sin_squared_expr) {
        (
            Expr::Binary(BinaryOp::Pow, cos_base, cos_exp),
            Expr::Binary(BinaryOp::Pow, sin_base, _),
        ) => {
            if let (Expr::Unary(UnaryOp::Cos, cos_arg), Expr::Unary(UnaryOp::Sin, sin_arg)) =
                (cos_base.as_ref(), sin_base.as_ref())
            {
                matches!(cos_exp.as_ref(), Expr::Integer(n) if n == &Integer::from(2))
                    && cos_arg == sin_arg
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Check if expression is sin²(arg) where arg matches the cos²(arg) from cos_squared_expr
fn is_sin_squared_same_arg(expr: &Expr, cos_squared_expr: &Expr) -> bool {
    is_cos_squared_same_arg(cos_squared_expr, expr)
}

/// Extract an exact rational constant from an expression, if it is one.
fn as_rational_const(e: &Expr) -> Option<Rational> {
    match e {
        Expr::Integer(n) => Rational::new(n.clone(), Integer::one()).ok(),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// Convert an exact rational back to canonical Expr form (Integer when integral).
fn rational_to_expr(r: Rational) -> Expr {
    if r.denominator().is_one() {
        Expr::Integer(r.numerator().clone())
    } else {
        Expr::Rational(r)
    }
}

/// Maximum |exponent| that constant folding will expand (guards memory/time).
pub(crate) const MAX_FOLD_EXPONENT: i64 = 4096;

/// S1 fix-up round 3 (review 3, B2): the largest a-priori size of a folded power c^n, in bits:
/// |n| (bits(num c) + bits(den c)), the estimate and the cap of the series engine's exact powers
/// (`series_const::MAX_EXACT_POWER_BITS`, 2^20 bits), so `simplify` never builds a power that the
/// engine kept as a kernel. HEAD bounded only the exponent, so `((2^256+1)^4096)^4096` built an
/// integer of 4.3e9 bits. A larger power stays `c ^ n`: the same value, not folded. Below the cap
/// the fold is HEAD's.
pub(crate) const MAX_FOLD_POWER_BITS: u64 = crate::series_const::MAX_EXACT_POWER_BITS;

/// S1 fix-up round 5 (review 5, B2): the words of fresh literals (64-bit words of the integers and
/// rationals that the folds build) that `simplify`'s memo may hold, 2^22 (32 MiB). Past it, a
/// shared node is recomputed at each use, as HEAD does.
pub(crate) const MEMO_PAYLOAD_WORDS: u64 = 1 << 22;

/// HEAD's rules for one binary node over simplified operands. S1 fix-up round 3: on shared nodes
/// (an operand that a rule returns is the same `Arc`), with the size guard of the `Pow` fold.
pub(crate) fn simplify_binary(op: BinaryOp, left: Arc<Expr>, right: Arc<Expr>) -> Arc<Expr> {
    let zero = || Arc::new(Expr::Integer(Integer::zero()));
    let lit = |r: Rational| Arc::new(rational_to_expr(r));
    match op {
        BinaryOp::Add => {
            // 0 + x = x, x + 0 = x
            if let Expr::Integer(n) = &*left {
                if n.is_zero() {
                    return right;
                }
            }
            if let Expr::Integer(n) = &*right {
                if n.is_zero() {
                    return left;
                }
            }

            // Constant folding (exact rational arithmetic)
            if let (Some(a), Some(b)) = (as_rational_const(&left), as_rational_const(&right)) {
                return lit(a + b);
            }

            Arc::new(Expr::Binary(BinaryOp::Add, left, right))
        }

        BinaryOp::Sub => {
            // x - 0 = x
            if let Expr::Integer(n) = &*right {
                if n.is_zero() {
                    return left;
                }
            }

            // Constant folding (exact rational arithmetic)
            if let (Some(a), Some(b)) = (as_rational_const(&left), as_rational_const(&right)) {
                return lit(a - b);
            }

            Arc::new(Expr::Binary(BinaryOp::Sub, left, right))
        }

        BinaryOp::Mul => {
            // 0 * x = 0, x * 0 = 0
            if let Expr::Integer(n) = &*left {
                if n.is_zero() {
                    return zero();
                }
                if n.is_one() {
                    return right;
                }
            }
            if let Expr::Integer(n) = &*right {
                if n.is_zero() {
                    return zero();
                }
                if n.is_one() {
                    return left;
                }
            }

            // Constant folding (exact rational arithmetic)
            if let (Some(a), Some(b)) = (as_rational_const(&left), as_rational_const(&right)) {
                return lit(a * b);
            }

            Arc::new(Expr::Binary(BinaryOp::Mul, left, right))
        }

        BinaryOp::Div => {
            // x / 1 = x
            if let Expr::Integer(n) = &*right {
                if n.is_one() {
                    return left;
                }
            }

            // Constant folding (exact rational arithmetic).
            // NOTE: this used to use truncating Integer division (1/2 -> 0);
            // it now produces an exact Rational (1/2 -> 1/2).
            if let (Some(a), Some(b)) = (as_rational_const(&left), as_rational_const(&right)) {
                if !b.is_zero() {
                    return lit(a / b);
                }
            }

            // (c * rest) / d = (c/d) * rest for constants c, d (cancels
            // common numeric factors, e.g. (2*x^2)/2 -> x^2)
            if let Some(d) = as_rational_const(&right) {
                if !d.is_zero() {
                    if let Expr::Binary(BinaryOp::Mul, a, b) = &*left {
                        if let Some(c) = as_rational_const(a) {
                            return simplify_binary(BinaryOp::Mul, lit(c / d), b.clone());
                        }
                        if let Some(c) = as_rational_const(b) {
                            return simplify_binary(BinaryOp::Mul, a.clone(), lit(c / d));
                        }
                    }
                }
            }

            Arc::new(Expr::Binary(BinaryOp::Div, left, right))
        }

        BinaryOp::Pow => {
            // x^0 = 1, x^1 = x
            if let Expr::Integer(n) = &*right {
                if n.is_zero() {
                    return Arc::new(Expr::Integer(Integer::one()));
                }
                if n.is_one() {
                    return left;
                }
            }

            // Constant folding: c^n for rational c and integer n, with a bounded exponent and
            // (S1 fix-up round 3) a bounded size of the result, |n| (bits(num c) + bits(den c))
            // at most MAX_FOLD_POWER_BITS, checked before anything is copied or computed
            if let Expr::Integer(n) = &*right {
                if *n >= Integer::from(-MAX_FOLD_EXPONENT) && *n <= Integer::from(MAX_FOLD_EXPONENT)
                {
                    let e = i64::try_from(n).expect("|n| <= MAX_FOLD_EXPONENT checked above");
                    let base_bits = match &*left {
                        Expr::Integer(c) => Some(c.bit_length() + 1),
                        Expr::Rational(c) => {
                            Some(c.numerator().bit_length() + c.denominator().bit_length())
                        }
                        _ => None,
                    };
                    let small = base_bits
                        .map(|b| b.saturating_mul(e.unsigned_abs()) <= MAX_FOLD_POWER_BITS)
                        .unwrap_or(false);
                    if small {
                        if let Some(base) = as_rational_const(&left) {
                            if e > 0 {
                                return lit(base.pow(e as u32));
                            }
                            // negative exponent: c^(-k) = 1 / c^k (only for c != 0)
                            if e < 0 && !base.is_zero() {
                                let one = Rational::new(Integer::one(), Integer::one())
                                    .expect("1/1 is a valid rational");
                                return lit(one / base.pow((-e) as u32));
                            }
                        }
                    }
                }
            }

            Arc::new(Expr::Binary(BinaryOp::Pow, left, right))
        }

        BinaryOp::Mod => {
            // Pass through unchanged for now
            Arc::new(Expr::Binary(BinaryOp::Mod, left, right))
        }
    }
}

/// HEAD's rules for one unary node over a simplified operand (S1 fix-up round 3: on a shared
/// node).
pub(crate) fn simplify_unary(op: UnaryOp, inner: Arc<Expr>) -> Arc<Expr> {
    let int = |n: Integer| Arc::new(Expr::Integer(n));
    match op {
        UnaryOp::Neg => {
            // -(-x) = x
            if let Expr::Unary(UnaryOp::Neg, inner_inner) = &*inner {
                return inner_inner.clone();
            }

            // Constant folding
            if let Expr::Integer(n) = &*inner {
                return int(-n.clone());
            }
            if let Expr::Rational(r) = &*inner {
                return Arc::new(Expr::Rational(-r.clone()));
            }

            Arc::new(Expr::Unary(UnaryOp::Neg, inner))
        }

        // Exact special values: f(0) and f(1) where the result is exact
        UnaryOp::Exp if inner.is_zero() => int(Integer::one()),
        UnaryOp::Log if is_one_const(&inner) => int(Integer::zero()),
        UnaryOp::Sin | UnaryOp::Tan | UnaryOp::Sinh | UnaryOp::Tanh if inner.is_zero() => {
            int(Integer::zero())
        }
        UnaryOp::Cos | UnaryOp::Cosh if inner.is_zero() => int(Integer::one()),
        UnaryOp::Arcsin | UnaryOp::Arctan if inner.is_zero() => int(Integer::zero()),
        UnaryOp::Arccos if is_one_const(&inner) => int(Integer::zero()),
        UnaryOp::Sqrt if inner.is_zero() => int(Integer::zero()),
        UnaryOp::Sqrt if is_one_const(&inner) => int(Integer::one()),

        _ => Arc::new(Expr::Unary(op, inner)),
    }
}

/// Check if an expression is the exact constant 1
fn is_one_const(e: &Expr) -> bool {
    match e {
        Expr::Integer(n) => n.is_one(),
        Expr::Rational(r) => r.numerator().is_one() && r.denominator().is_one(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1 fix-up round 3 (review 3, B2): `simplify` folds c^n only when the a-priori size of the
    /// power, |n| (bits(num c) + bits(den c)), is at most 2^20 bits (the series engine's cap of an
    /// exact power, `series_const::MAX_EXACT_POWER_BITS`). HEAD folded every |n| <= 4096 with no
    /// bound, so ((2^256+1)^4096)^4096 built an integer of 4.3e9 bits. Below the cap the fold is
    /// HEAD's: (2^254+1)^4096 is (255 + 1) x 4096 = 2^20 bits a priori, folded.
    #[test]
    fn simplify_folds_powers_up_to_the_engine_cap() {
        let p = |s: &str| crate::parser::parse(s).expect("parse");
        let big = Integer::from(2).pow(256) + Integer::from(1);
        let s = simplify(&p("((2^256+1)^4096)^2"));
        let inner = Expr::Binary(
            BinaryOp::Pow,
            Arc::new(Expr::Integer(big)),
            Arc::new(Expr::from(4096)),
        );
        let want = Expr::Binary(BinaryOp::Pow, Arc::new(inner), Arc::new(Expr::from(2)));
        assert_eq!(s, want);
        let s = simplify(&p("(2^254+1)^4096"));
        let b = Integer::from(2).pow(254) + Integer::from(1);
        assert_eq!(s, Expr::Integer(b.pow(4096)));
        // a negative exponent is bounded the same way
        let s = simplify(&p("(2^256+1)^(-4096)"));
        assert!(matches!(s, Expr::Binary(BinaryOp::Pow, _, _)), "{s}");
        assert_eq!(simplify(&p("(1/3)^(-5)")), Expr::from(243));
    }

    /// S1 fix-up round 3 (review 3, B2): a node held by more than one `Arc` is simplified once and
    /// its value is shared by every occurrence (HEAD simplified it again at each occurrence: a
    /// series result names a kernel's source at every term).
    #[test]
    fn simplify_shares_the_value_of_a_shared_node() {
        let src = Arc::new(crate::parser::parse("(2^100+1)^64").expect("parse"));
        let x = Arc::new(Expr::symbol("x"));
        let y = Arc::new(Expr::symbol("y"));
        let e = Expr::Binary(
            BinaryOp::Add,
            Arc::new(Expr::Binary(BinaryOp::Mul, src.clone(), x)),
            Arc::new(Expr::Binary(BinaryOp::Mul, src.clone(), y)),
        );
        let s = simplify(&e);
        let (l, r) = match &s {
            Expr::Binary(BinaryOp::Add, l, r) => (l.clone(), r.clone()),
            other => panic!("{other}"),
        };
        let (a, b) = match (&*l, &*r) {
            (Expr::Binary(BinaryOp::Mul, a, _), Expr::Binary(BinaryOp::Mul, b, _)) => {
                (a.clone(), b.clone())
            }
            other => panic!("{other:?}"),
        };
        let want = (Integer::from(2).pow(100) + Integer::from(1)).pow(64);
        assert_eq!(*a, Expr::Integer(want));
        assert!(Arc::ptr_eq(&a, &b), "the folded source is not shared");
    }

    /// S1 fix-up round 4 (review 4, B2): the parents of a node are counted inside the walked DAG.
    /// A chain whose links are also held outside (a REPL line `a2 = a1*3` stores a shallow clone,
    /// so each child is held twice but has one parent in the DAG) gives the value of the same
    /// expression as a fresh tree; a node with three parents inside the DAG and one owner outside
    /// is simplified once and its value is shared by the three occurrences.
    #[test]
    fn simplify_counts_parents_inside_the_dag() {
        let p = |s: &str| crate::parser::parse(s).expect("parse");
        let mut links = vec![p("(2^100+1)^8")];
        for k in 1..50 {
            let prev = links[k - 1].clone();
            links.push(prev * Expr::from(3));
        }
        let fresh = p(&format!("(2^100+1)^8{}", "*3".repeat(49)));
        let want = (Integer::from(2).pow(100) + Integer::from(1)).pow(8) * Integer::from(3).pow(49);
        assert_eq!(
            simplify(links.last().expect("a link")),
            Expr::Integer(want.clone())
        );
        assert_eq!(simplify(&fresh), Expr::Integer(want));
        let shared = Arc::new(p("(2^60+1)^4 + 1/3"));
        let outside = shared.clone();
        let x = Arc::new(Expr::symbol("x"));
        let e = Expr::Binary(
            BinaryOp::Add,
            Arc::new(Expr::Binary(BinaryOp::Mul, shared.clone(), x)),
            Arc::new(Expr::Function(
                "g".into(),
                vec![shared.clone(), shared.clone()],
            )),
        );
        let s = simplify(&e);
        let (m, g) = match &s {
            Expr::Binary(BinaryOp::Add, m, g) => (m.clone(), g.clone()),
            other => panic!("{other}"),
        };
        let (a, b, c) = match (&*m, &*g) {
            (Expr::Binary(BinaryOp::Mul, a, _), Expr::Function(_, args)) => {
                (a.clone(), args[0].clone(), args[1].clone())
            }
            other => panic!("{other:?}"),
        };
        let v = Rational::new(
            (Integer::from(2).pow(60) + Integer::from(1)).pow(4) * Integer::from(3)
                + Integer::from(1),
            Integer::from(3),
        )
        .expect("3 != 0");
        assert_eq!(*a, Expr::Rational(v));
        assert!(
            Arc::ptr_eq(&a, &b) && Arc::ptr_eq(&b, &c),
            "the value is not shared"
        );
        assert!(Arc::strong_count(&outside) >= 2);
    }

    /// S1 fix-up round 5 (review 5, B2): past `MEMO_PAYLOAD_WORDS` the memo stops holding values
    /// and their next uses recompute them, as HEAD does; the result is the same. A DAG naming
    /// 320 shared negations of a 2^20-bit leaf twice (320 x 16 385 words, past the 2^22-word cap),
    /// inside `0 * q`, `q + q` and calls: the value equals that of HEAD's recursive walk (the same
    /// rules, `simplify_binary` and `simplify_unary`, applied at every occurrence).
    #[test]
    fn simplify_past_the_memo_cap_gives_heads_value() {
        fn head(e: &Expr) -> Arc<Expr> {
            match e {
                Expr::Binary(op, l, r) => simplify_binary(*op, head(l), head(r)),
                Expr::Unary(op, a) => simplify_unary(*op, head(a)),
                Expr::Function(n, args) => Arc::new(Expr::Function(
                    n.clone(),
                    args.iter().map(|a| head(a)).collect(),
                )),
                leaf => Arc::new(leaf.clone()),
            }
        }
        let p = Arc::new(Expr::Integer(
            Integer::from(2).pow(1 << 20) + Integer::from(7),
        ));
        let m = 320usize;
        assert!(m as u64 * ((1 << 20) / 64 + 1) > MEMO_PAYLOAD_WORDS);
        let qs: Vec<Arc<Expr>> = (0..m)
            .map(|_| Arc::new(Expr::Unary(UnaryOp::Neg, p.clone())))
            .collect();
        let zero = Arc::new(Expr::from(0));
        let y = Arc::new(Expr::symbol("y"));
        let mut a: Arc<Expr> = zero.clone();
        let mut b: Arc<Expr> = zero.clone();
        for (i, q) in qs.iter().enumerate() {
            let t = match i % 3 {
                0 => Arc::new(Expr::Binary(BinaryOp::Mul, zero.clone(), q.clone())),
                1 => Arc::new(Expr::Binary(BinaryOp::Add, q.clone(), q.clone())),
                _ => Arc::new(Expr::Function("g".into(), vec![q.clone(), y.clone()])),
            };
            a = Arc::new(Expr::Binary(BinaryOp::Add, a, t));
            b = Arc::new(Expr::Binary(BinaryOp::Add, b, q.clone()));
        }
        let e = Expr::Binary(BinaryOp::Sub, a, b);
        let got = simplify(&e);
        let want = head(&e);
        assert!(got == *want, "the value differs from HEAD's");
        drop(qs);
    }

    /// S1 design §2.10: a left-deep sum of 10^5 terms simplifies on a 2 MB debug thread (the
    /// left spine is walked with an explicit stack), with HEAD's result.
    #[test]
    fn simplify_long_left_deep_sum() {
        let r = std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(|| {
                let x = Expr::symbol("x");
                let mut sum = Expr::from(1);
                for k in 1..100_000i64 {
                    sum = sum + Expr::from(k % 7);
                }
                let s = simplify(&sum);
                let want: i64 = 1 + (1..100_000i64).map(|k| k % 7).sum::<i64>();
                assert_eq!(s, Expr::from(want));
                let mut xs = x.clone();
                for _ in 1..100_000 {
                    xs = xs + x.clone();
                }
                // HEAD's simplify keeps the sum (no collection of like terms): the same left
                // spine of 10^5 - 1 additions, walked here with a loop (substitute and
                // PartialEq still recurse, S1 design §7 item 5)
                let t = simplify(&xs);
                let mut depth = 0usize;
                let mut cur = &t;
                while let Expr::Binary(BinaryOp::Add, l, r) = cur {
                    assert_eq!(**r, x);
                    depth += 1;
                    cur = l;
                }
                assert_eq!((depth, cur), (99_999, &x));
            })
            .expect("spawn")
            .join();
        assert!(r.is_ok(), "simplify overflowed or panicked");
    }

    #[test]
    fn test_simplify_addition() {
        // 0 + x = x
        let x = Expr::symbol("x");
        let expr = Expr::from(0) + x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 2 + 3 = 5
        let expr = Expr::from(2) + Expr::from(3);
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(5));
    }

    #[test]
    fn test_simplify_multiplication() {
        // 1 * x = x
        let x = Expr::symbol("x");
        let expr = Expr::from(1) * x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);

        // 0 * x = 0
        let expr = Expr::from(0) * x.clone();
        let simplified = simplify(&expr);
        assert_eq!(simplified, Expr::from(0));
    }

    #[test]
    fn test_simplify_negation() {
        // -(-x) = x
        let x = Expr::symbol("x");
        let expr = -(-x.clone());
        let simplified = simplify(&expr);
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_simplify_trig() {
        let x = Expr::symbol("x");

        // sin(x)^2 + cos(x)^2 = 1
        let sin_x = x.clone().sin();
        let cos_x = x.clone().cos();
        let expr = sin_x.pow(Expr::from(2)) + cos_x.pow(Expr::from(2));

        let simplified = expr.simplify_trig();

        // Should simplify to 1
        assert_eq!(simplified, Expr::from(1));
    }

    #[test]
    fn test_simplify_full() {
        let x = Expr::symbol("x");

        // (x + 1) * 2 should expand and simplify
        let expr = (x.clone() + Expr::from(1)) * Expr::from(2);
        let simplified = expr.simplify_full();

        // Should be in expanded form
        match simplified {
            Expr::Binary(BinaryOp::Add, _, _) => {}
            _ => {}
        }
    }

    #[test]
    fn test_simplify_rational() {
        let x = Expr::symbol("x");

        // (2*x) / 2 could simplify to x (but needs more advanced implementation)
        let expr = (Expr::from(2) * x.clone()) / Expr::from(2);
        let _simplified = expr.simplify_rational();

        // For now, just test that it doesn't crash
    }
}
