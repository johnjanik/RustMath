//! The constant field K of the series engine (S1 design §2.2, §2.7, §2.15): a constant is a
//! finite Q-linear combination of monomials in kernels; kernels are expressions free of the
//! expansion variable that are not rationals and not sums, products or integer powers of simpler
//! ones (radicals n^(1/d), pi, e, I, f(c) for the unary functions, 1/P, generic parameters). Every
//! big-number operation is charged against the work budget before it runs, and the zero test is
//! three-valued (decided zero, decided nonzero, undecided).
//!
//! Specification: MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series" (§49.1.3, §49.1.4,
//! §49.5), digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`; the S1
//! design (frozen at revision 7rt4 with revision 8), ported from its prototype `proto13`.
//!
//! References: Knuth, TAOCP vol. 2, 3rd ed., §4.7; R. P. Brent and H. T. Kung, "Fast algorithms
//! for manipulating formal power series", J. ACM 25(4), 1978, 581-595; S. C. Johnson, "Sparse
//! polynomial arithmetic", ACM SIGSAM Bulletin 8(3), 1974, 63-71; M. Monagan and R. Pearce,
//! "Sparse polynomial division using a heap", J. Symbolic Comput. 46(7), 2011, 807-822 (the heap
//! merges); D. Richardson, "Some undecidable problems involving elementary functions of a real
//! variable", J. Symbolic Logic 33(4), 1968, 514-520 (zero equivalence is undecidable in general,
//! hence the three-valued test); F. Lindemann, "Ueber die Zahl pi", Math. Ann. 20, 1882, 213-225,
//! and C. Hermite, "Sur la fonction exponentielle", C. R. Acad. Sci. Paris 77, 1873 (the
//! transcendence facts the nonzero rules use).
//!
//! Port note (a deviation from the design, recorded in the S1 report): the design threads an
//! explicit `&mut Work` through every operation and returns errors with `?`. The prototype, which
//! every review and red-team round measured, emulates that with thread-local state (the work
//! counter, the fault, the kernel table, the hint table); the port keeps the prototype's state and
//! charge points, scoped to one `series_engine::Session` per call.

// Prototype of the constant field for the S1 series engine.
// A constant is a finite Q-linear combination of monomials in kernels.
// Kernels are expressions free of the expansion variable that are not
// rational numbers and not sums / products / integer powers of simpler ones.

use crate::{BinaryOp, Expr, Symbol, UnaryOp};
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub enum SErr {
    /// f has no Laurent expansion at the point (branch point, log or essential singularity).
    NotLaurent(String),
    /// a function or operation the engine does not expand
    Unsupported(String),
    /// a constant must be inverted (or its sign read) and its zero test is undecided
    Undecided(Expr),
    /// the input divides by an exact zero
    DivisionByZero(String),
    /// the expansion point is not usable
    BadPoint(String),
    /// internal: working precision too low (a denominator is weakly zero)
    NeedPrecision,
    /// bounded precision search found no answer
    PrecisionExhausted(i64),
    /// work budget exceeded
    Budget(u64),
    /// revision 5: a structural cap (design section 2.15) was reached before the work it bounds
    Limit(&'static str, u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KClass {
    /// a free symbol: a generic parameter (indeterminate)
    Indet,
    /// the imaginary unit `I` (I^2 = -1 is applied)
    Imag,
    /// `pi` or `e`: transcendental real numbers
    Named,
    /// a constant proved nonzero (exp(c), f(q) for rational q, 1/P, ...)
    NonZero,
    /// a constant whose zero test is unknown
    Opaque,
    /// 1/P for a constant P (rendered as a denominator)
    Inverse,
}

pub struct KData {
    /// interning id: unique per distinct key within one expansion (revision 2, m2)
    pub id: u64,
    /// revision 6 (review 6, B3): shared, so a rendering refers to it by an Arc increment and
    /// never clones its top node (a Symbol node owns its name's String)
    pub expr: Arc<Expr>,
    pub class: KClass,
    /// the kernel's expression contains an indeterminate (only matters for non-Indet kernels)
    pub has_indet: bool,
    /// for Inverse kernels: the constant P with kernel = 1/P
    pub inv_of: Option<Konst>,
    /// for radical kernels n^(1/d) (integer n > 1, not a perfect d-th power): (n, d)
    pub root: Option<(Integer, u32)>,
    /// revision 5: an upper bound of the depth of `expr` (computed from the arguments' normal
    /// forms before the expression is built; capped, section 2.15)
    pub depth: u32,
    /// revision 5: for a kernel f(c) of a unary function, (f, c) (log_exact reads it; the
    /// expression may be the input's own subtree, section 2.2)
    pub unary: Option<(UnaryOp, Konst)>,
    /// revision 7rt3 (red team 3 structure, H2): an upper bound of the number of nodes of `expr`
    /// as a tree (saturating), from the arguments' cached tree sizes or the input node's
    pub tree: u64,
}

#[derive(Clone)]
pub struct Kernel(pub Arc<KData>);

impl PartialEq for Kernel {
    fn eq(&self, o: &Self) -> bool {
        self.0.id == o.0.id
    }
}
impl Eq for Kernel {}
impl PartialOrd for Kernel {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Kernel {
    fn cmp(&self, o: &Self) -> Ordering {
        // indeterminates first, then by key
        let ra = if self.0.class == KClass::Indet { 0 } else { 1 };
        let rb = if o.0.class == KClass::Indet { 0 } else { 1 };
        ra.cmp(&rb).then_with(|| self.0.id.cmp(&o.0.id))
    }
}

pub type Mono = Vec<(Kernel, i64)>;
pub type Term = (Mono, Rational);

#[derive(Clone)]
pub struct Konst {
    /// revision 4 (review 4, B1): shared, so a copy of a constant is O(1); every operation that
    /// builds a new constant charges first
    /// revision 5: the terms carry their sizes, computed once when they are built, so every
    /// a-priori charge and every class test is O(1)
    pub t: Arc<KMap>,
}
impl PartialEq for Konst {
    fn eq(&self, o: &Self) -> bool {
        self.t.len() == o.t.len() && self.t.iter().eq(o.t.iter())
    }
}
pub struct KMap {
    pub m: Terms,
    pub sz: KSize,
}
/// revision 7rt3 (red team 3 structure, H2): a rendering of the same value with a smaller tree (section
/// 2.5, "value at the point": the input node with the variable replaced by the point, when that node
/// is regular), with its tree size and depth. Kept in a side table of the expansion keyed by the
/// constant's address (the table holds the constant, so the address stays valid), not in the header:
/// hints are rare, and a field would add 32 bytes to every constant (a 240-byte chunk for CONST_BASE)
pub struct Hint {
    pub e: Arc<Expr>,
    pub tree: u64,
    pub depth: u32,
}
impl std::ops::Deref for KMap {
    type Target = Terms;
    fn deref(&self) -> &Terms {
        &self.m
    }
}
/// revision 5: a constant with at most one term holds it inline.
/// revision 6 (review 6, B2): several terms are a sorted boxed slice, not a BTreeMap: 88 bytes
/// per term and no node slack (a BTreeMap of 2 to 11 terms allocated a leaf of about 1 KB).
/// Lookups are binary searches; every operation builds its result in monomial order (merges,
/// maps, a sort), so no operation inserts into a sorted container.
pub enum Terms {
    Zero,
    One(Term),
    Many(Box<[Term]>),
}
impl Terms {
    pub fn as_slice(&self) -> &[Term] {
        match self {
            Terms::Zero => &[],
            Terms::One(t) => std::slice::from_ref(t),
            Terms::Many(v) => v,
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Mono, &Rational)> {
        self.as_slice().iter().map(|(m, q)| (m, q))
    }
    pub fn keys(&self) -> impl Iterator<Item = &Mono> {
        self.as_slice().iter().map(|(k, _)| k)
    }
    pub fn values(&self) -> impl Iterator<Item = &Rational> {
        self.as_slice().iter().map(|(_, v)| v)
    }
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// binary search, O(log t) monomial comparisons
    pub fn get(&self, k: &Mono) -> Option<&Rational> {
        let v = self.as_slice();
        v.binary_search_by(|(m, _)| m.cmp(k)).ok().map(|i| &v[i].1)
    }
}
/// revision 5: sizes of a constant, computed in one pass over its terms when it is built
#[derive(Clone, Copy, Default)]
pub struct KSize {
    pub words: u64,
    pub copy: u64,
    pub expr: u64,
    /// sum of the monomial lengths
    pub factors: u64,
    /// sum over terms of max(0, L - SEARCH_FREE)
    pub excess: u64,
    /// sum of the bit lengths of all numerators and denominators
    pub bits: u64,
    /// an upper bound of the depth of to_expr_raw
    pub depth: u32,
    pub max_l: u64,
    /// power_size: the largest bit size of a term (q != +-1, plus its radical factors)
    pub max_term_bits: u64,
    pub max_abs_exp: u64,
    /// some kernel is an indeterminate or depends on one
    pub any_indet: bool,
    /// some non-indeterminate kernel depends on an indeterminate
    pub nonindet_dep: bool,
    /// every kernel is an indeterminate
    pub only_indet: bool,
    /// no indeterminate, no I, no kernel depending on an indeterminate
    pub real_free: bool,
    pub any_opaque: bool,
    /// revision 6: some factor is the kernel I (split_imag copies nothing otherwise)
    pub any_imag: bool,
    /// revision 6: every kernel is I or one and the same named constant (rule 4 of the zero test,
    /// read in O(1); revision 5 walked the factors at every test, uncharged)
    pub one_named_imag: bool,
    /// revision 7rt3 (red team 3 structure, H2): the nodes of to_expr_raw as a tree (saturating; a
    /// kernel counts its expression's tree at every occurrence), as mono_expr and balanced_arc build it
    pub tree: u64,
}
/// depth added by balanced_sum / balanced_product over n leaves
pub fn bal_depth(n: u64) -> u32 {
    if n <= 1 {
        0
    } else if n <= 4 {
        (n - 1) as u32
    } else {
        1 + bal_depth(n.div_ceil(2))
    }
}
fn kpow_depth(k: &Kernel, e: i64) -> u32 {
    k.0.depth
        .saturating_add(if e == 1 || e == -1 { 0 } else { 1 })
}
/// revision 6: the units of one term, as copy_units and expr_units sum them (section 2.7)
pub fn term_copy_units(l: u64, wn: u64, wd: u64) -> u64 {
    COPY_TERM + COPY_FACTOR * l + COPY_WORD * (wn + wd)
}
fn ksize(m: &Terms) -> KSize {
    let mut z = KSize {
        only_indet: true,
        real_free: true,
        one_named_imag: true,
        ..Default::default()
    };
    let mut max_term: u32 = 0;
    let mut named: Option<u64> = None;
    let mut tree: u64 = 0;
    for (mono, q) in m.iter() {
        let l = mono.len() as u64;
        // revision 7rt3: the term's tree as mono_expr builds it: each factor's kernel tree (plus a
        // Pow and an Integer node when its exponent is not +-1); the numerator's nn - 1 products and
        // its coefficient (nothing for 1, a negation for -1, a product and a rational node else), or
        // a rational node alone; the denominator's dn - 1 products and the quotient
        let (mut nn, mut dn, mut tt) = (0u64, 0u64, 0u64);
        for (k, e) in mono {
            tt = tt
                .saturating_add(k.0.tree)
                .saturating_add(if *e == 1 || *e == -1 { 0 } else { 2 });
            if k.0.class == KClass::Inverse || *e < 0 {
                dn += 1
            } else {
                nn += 1
            }
        }
        let qone = q.is_one();
        let qm1 = q.numerator().signum() < 0
            && q.numerator().bit_length() == 1
            && q.denominator().is_one();
        tt = tt.saturating_add(if nn == 0 {
            1
        } else {
            nn - 1
                + if qone {
                    0
                } else if qm1 {
                    1
                } else {
                    2
                }
        });
        tt = tt.saturating_add(dn);
        tree = tree.saturating_add(tt);
        let (wn, wd) = (iw(q.numerator()), iw(q.denominator()));
        z.words += rat_words(q) + l;
        z.copy += term_copy_units(l, wn, wd);
        // revision 6: the nodes mono_expr allocates for one term: a kernel factor with exponent
        // +-1 is the kernel's shared Arc (no node); L - 1 products at most; the coefficient node
        // and its product (2); the quotient (1); the term's share of the balanced sum (1); per
        // factor whose exponent is not +-1, a Pow node and an Integer node with its digit vector.
        // So at most 2 + L + 2 pf nodes (revision 5: 3 + 2 L, for a rendering that cloned each
        // kernel's top node)
        let pf = mono.iter().filter(|(_, e)| *e != 1 && *e != -1).count() as u64;
        z.expr += EXPR_NODE * (2 + l + 2 * pf) + EXPR_WORD * (wn + wd) + EXPR_INT * (2 + pf);
        z.factors += l;
        z.excess += l.saturating_sub(SEARCH_FREE);
        z.bits += q.numerator().bit_length() + q.denominator().bit_length();
        z.max_l = z.max_l.max(l);
        let unit = q.is_one()
            || (q.numerator().signum() < 0
                && q.numerator().bit_length() == 1
                && q.denominator().is_one());
        let mut tb = if unit { 0 } else { rat_bits(q) };
        let (mut nn, mut nd, mut dn, mut dd) = (0u64, 0u32, 0u64, 0u32);
        for (k, e) in mono {
            z.max_abs_exp = z.max_abs_exp.max(e.unsigned_abs());
            let c = k.0.class;
            if c == KClass::Indet || k.0.has_indet {
                z.any_indet = true;
            }
            if c != KClass::Indet && k.0.has_indet {
                z.nonindet_dep = true;
            }
            if c != KClass::Indet {
                z.only_indet = false;
            }
            if c == KClass::Indet || c == KClass::Imag || k.0.has_indet {
                z.real_free = false;
            }
            if c == KClass::Opaque {
                z.any_opaque = true;
            }
            if c == KClass::Imag {
                z.any_imag = true;
            }
            match c {
                KClass::Imag => {}
                KClass::Named if named.is_none_or(|n| n == k.0.id) => named = Some(k.0.id),
                _ => z.one_named_imag = false,
            }
            if let Some((nr, d)) = &k.0.root {
                tb = tb.saturating_add(
                    nr.bit_length().saturating_mul(e.unsigned_abs()) / (*d as u64) + 1,
                );
            }
            let fd = kpow_depth(k, *e);
            if c == KClass::Inverse || *e < 0 {
                dn += 1;
                dd = dd.max(fd);
            } else {
                nn += 1;
                nd = nd.max(fd);
            }
        }
        z.max_term_bits = z.max_term_bits.max(tb);
        // the term's expression (mono_expr): q * (balanced numerator) / (balanced denominator)
        // q = 1 prints the product alone, q = -1 as a negation, any other q as q * product
        let numer = if nn == 0 {
            1
        } else {
            bal_depth(nn) + nd + if q.is_one() { 0 } else { 1 }
        };
        let td = if dn == 0 {
            numer
        } else {
            1 + numer.max(bal_depth(dn) + dd)
        };
        max_term = max_term.max(td);
    }
    z.expr += 16;
    z.words = z.words.max(1);
    z.depth = if m.is_empty() {
        1
    } else {
        bal_depth(m.len() as u64) + max_term
    };
    // the balanced sum adds t - 1 nodes
    z.tree = if m.is_empty() {
        1
    } else {
        tree.saturating_add(m.len() as u64 - 1)
    };
    if m.is_empty() {
        z.only_indet = false;
    }
    z
}
thread_local! {
    /// revision 6: 0 and 1 are shared, so the many temporaries 0 and 1 allocate nothing
    static ZERO_K: Konst = Konst::raw(Terms::Zero);
    static ONE_K: Konst = Konst::raw(Terms::One((Vec::new(), Rational::from_integer(Integer::from(1)))));
}
impl Konst {
    fn raw(terms: Terms) -> Konst {
        let sz = ksize(&terms);
        Konst {
            t: Arc::new(KMap { m: terms, sz }),
        }
    }
    /// revision 6 (review 6, B2): the one constructor. `v` is in monomial order, without equal
    /// monomials or zero coefficients. It charges CONST_BASE (the Arc, the header and the cached
    /// sizes, about 200 bytes) before it allocates them; the terms themselves were charged by the
    /// operation that built them (copy_units, or the per-pair charges of a product). The pass of
    /// ksize, O(t + F), is covered by those charges (section 2.16, fact 2).
    pub fn from_sorted(v: Vec<Term>) -> Konst {
        if v.is_empty() {
            return Konst::zero();
        }
        if !charge(CONST_BASE) {
            return Konst::zero();
        }
        let terms = if v.len() == 1 {
            Terms::One(v.into_iter().next().unwrap())
        } else {
            Terms::Many(v.into_boxed_slice())
        };
        Konst::raw(terms)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZS {
    Zero,
    NonZero,
    Unknown,
}

pub fn qi(n: i64) -> Rational {
    Rational::from_integer(Integer::from(n))
}
/// 1/2 and 1/4
fn qhalf() -> Rational {
    Rational::new(Integer::from(1), Integer::from(2)).expect("2 != 0")
}
fn qquarter() -> Rational {
    Rational::new(Integer::from(1), Integer::from(4)).expect("4 != 0")
}

fn rat_expr(q: &Rational) -> Expr {
    if q.denominator().is_one() {
        Expr::Integer(q.numerator().clone())
    } else {
        Expr::Rational(q.clone())
    }
}

pub fn balanced_sum(mut terms: Vec<Expr>) -> Expr {
    // depth ceil(log2 n); short sums keep the left-deep shape of a + b + c
    fn build(t: &mut [Option<Expr>]) -> Expr {
        let n = t.len();
        if n == 1 {
            return t[0].take().unwrap();
        }
        if n <= 4 {
            let mut acc = t[0].take().unwrap();
            for x in t[1..].iter_mut() {
                acc = acc + x.take().unwrap();
            }
            return acc;
        }
        let mid = n.div_ceil(2);
        let (l, r) = t.split_at_mut(mid);
        build(l) + build(r)
    }
    if terms.is_empty() {
        return Expr::from(0);
    }
    let mut v: Vec<Option<Expr>> = terms.drain(..).map(Some).collect();
    build(&mut v)
}

/// Work counter. The design threads `&mut Work` through every constant operation and returns
/// `Err(BudgetExceeded)` with `?`. The prototype emulated that with a thread-local counter and a
/// thread-local fault, and the port keeps that state (a deviation, see `series_engine::Session`):
/// once a fault is set (budget passed, exponent overflow, a structural cap), every constant
/// operation returns at once, and the engine reports the fault at its next check. The budget is
/// the thread-local `MAXCOST`, set by the driver; `used` mirrors the counter after each charge.
pub struct Work {
    pub used: u64,
}

impl Work {
    pub fn charge(&mut self, w: u64) -> Result<(), SErr> {
        // revision 7rt4 (red team 4 series, H2): saturating; `16 + w` wrapped a saturated
        // u64::MAX to 15 units (a panic in debug), so the budget admitted a buffer of 2^60 slots
        charge(w.saturating_add(16));
        self.used = COST.with(|c| c.get());
        check()
    }
}

thread_local! {
    /// kernel interning table of the current expansion (the design keeps it in the Ctx)
    pub static KTABLE: std::cell::RefCell<std::collections::HashMap<String, Kernel>> = std::cell::RefCell::new(std::collections::HashMap::new());
    pub static COST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub static MAXCOST: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
    pub static FAULT: std::cell::RefCell<Option<SErr>> = const { std::cell::RefCell::new(None) };
}
/// add w to the work counter; false (and a Budget fault) once it passes the budget
pub fn charge(w: u64) -> bool {
    let v = COST.with(|c| {
        let v = c.get().saturating_add(w);
        c.set(v);
        v
    });
    if v > MAXCOST.with(|m| m.get()) {
        set_fault(SErr::Budget(v));
        return false;
    }
    !faulted()
}
pub fn set_fault(e: SErr) {
    FAULT.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_none() {
            *f = Some(e);
        }
    });
}
pub fn faulted() -> bool {
    FAULT.with(|f| f.borrow().is_some())
}
/// revision 4: an UndecidedZero error for c, or the pending fault (a budget refusal inside the
/// zero test wins, as `?` does in the design)
pub fn undecided(c: &Konst) -> SErr {
    // revision 7rt3 (red team 3 structure, H2): the constant of UndecidedZero is a value the
    // caller can walk, so it is bounded as a result is
    if c.tree() > MAX_RESULT_TREE {
        if let Err(e) = check() {
            return e;
        }
        return SErr::Limit("nodes of the result's tree", MAX_RESULT_TREE);
    }
    undecided_expr(|| c.to_expr())
}
pub fn undecided_expr(f: impl FnOnce() -> Expr) -> SErr {
    if let Err(e) = check() {
        return e;
    }
    let ex = f();
    match check() {
        Err(e) => e,
        // S1 fix-up round 2 (review 2, B2): the constant is bounded by its leaf text too; round 4
        // (review 4, B1): and by the fold work of its `simplify`
        Ok(()) => match result_caps(&ex) {
            Some(e) => e,
            None => SErr::Undecided(ex),
        },
    }
}
pub fn check() -> Result<(), SErr> {
    match FAULT.with(|f| f.borrow().clone()) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
/// budget left (for the a-priori walks of ball_cost and key_size)
pub fn remaining() -> u64 {
    MAXCOST
        .with(|m| m.get())
        .saturating_sub(COST.with(|c| c.get()))
}
pub fn reset_work(max: u64) {
    COST.with(|c| c.set(0));
    MAXCOST.with(|m| m.set(max));
    FAULT.with(|f| *f.borrow_mut() = None);
    drop_table();
}
/// revision 5: the table is dropped newest kernel first. An Inverse kernel holds its P0, whose
/// monomials hold older kernels; dropping the newest first leaves every older kernel still owned
/// by the table, so no drop recurses through a chain of kernels.
pub fn drop_table() {
    let mut v: Vec<Kernel> = KTABLE.with(|t| t.borrow_mut().drain().map(|(_, k)| k).collect());
    v.sort_by_key(|k| std::cmp::Reverse(k.0.id));
    for k in v {
        drop(k);
    }
}
/// the kernel with this key: the existing one, or a new one built by `make` (O(1) comparisons
/// afterwards: kernels compare by (class rank, id))
/// revision 4 (review 4, B2): the table holds the only copy of each key; `make` runs only for a
/// new key (it charges the kernel's expression before building it)
pub fn intern(key: String, make: impl FnOnce(u64) -> KData) -> Kernel {
    if let Some(k) = KTABLE.with(|t| t.borrow().get(&key).cloned()) {
        return k;
    }
    let id = KTABLE.with(|t| t.borrow().len() as u64);
    // revision 5: the number of distinct kernels of one expansion is capped, and a new kernel's
    // memory is charged before it is created
    if id as usize >= MAX_KERNELS {
        limit("distinct kernels of one expansion", MAX_KERNELS as u64);
        return fault_kernel();
    }
    if !charge(KERNEL_UNITS) {
        return fault_kernel();
    }
    let k = Kernel(Arc::new(make(id)));
    KTABLE.with(|t| t.borrow_mut().insert(key, k.clone()));
    k
}

// ---------------------------------------------------------------------------
// Revision 3 (review 3, B1, B2, m1, m2): charged big-number arithmetic. Every Rational or
// Integer operation on coefficients goes through one of these functions, which charges an
// a-priori bound of its cost immediately before it runs. Costs, in units (64-bit words):
//   OP_BASE per operation (the per-call overhead of a Lehmer gcd and of the allocations),
//   a * b for a product of a-word and b-word integers (schoolbook bound),
//   N * D for the normalization of an unreduced N-word / D-word fraction (Lehmer gcd and two
//     exact divisions; N when the denominator is 1),
//   W * W / 64 + W for an integer power with a W-word result (binary powering, Karatsuba),
//   W * W for an exact root of a W-word radicand (Newton square roots).
// On a fault (budget passed) they return 0 / None at once; the caller reports the fault.
pub const OP_BASE: u64 = 64;

// ---------------------------------------------------------------------------
// Revision 5: structural caps (design section 2.15). Each is checked before the work it bounds
// and gives a typed error (SErr::Limit; in the zero test the error propagates like a budget
// refusal, it is never read as a decision).
/// factors of one monomial (a monomial's factors are distinct kernels)
pub const MAX_MONO_FACTORS: usize = 1 << 15;
/// distinct kernels of one expansion (the interning table)
pub const MAX_KERNELS: usize = 1 << 16;
/// terms of one constant
pub const MAX_CONST_TERMS: usize = 1 << 20;
/// depth of a kernel's expression: at most max(MAX_KERNEL_DEPTH, KERNEL_DEPTH_FACTOR * input depth)
pub const MAX_KERNEL_DEPTH: u32 = 1024;
/// revision 7rt3 (red team 3 structure, H2): nodes of a result as a tree (its rendering, or the
/// constant of UndecidedZero), so that a consumer that walks the tree node by node (substitute,
/// PartialEq, the walk of Display, expand, differentiate) is bounded by the cap times its cost per
/// node. The payload of the nodes is bounded by the other caps. S1 fix-up round 5 (review 5, B1):
/// each of them is counted at every occurrence in the tree, so that it also bounds a consumer
/// that rebuilds the tree without its sharing (`substitute`, `expand`, `differentiate`, which make
/// a fresh node at every occurrence): `MAX_RESULT_TEXT` the digits `Display` converts, in the
/// result, in its `simplify` and in the literals its nodes may fold to, and
/// `MAX_RESULT_FOLD_WORK` the folds of `simplify`; `MAX_RESULT_MEMO_WORDS` bounds the literals of
/// the distinct nodes, which `simplify`'s memo holds when `differentiate` shares them. The
/// consumers measured, and the bound that holds, are in the doc of `MAX_RESULT_TEXT`.
pub const MAX_RESULT_TREE: u64 = 1 << 21;
/// S1 fix-up round 2 (review 2, B2), round 3 (review 3, B1, B2 and m1): the printing weight of a
/// result's tree and of the tree `simplify` makes of it (`tree_text`), the payload that the node
/// count of `MAX_RESULT_TREE` does not see. A shared kernel with a long leaf is rendered once and
/// named by every term, so its tree repeats the leaf at each term; a kernel made from an input
/// node renders as that node, which `simplify` folds (a 15-character power can fold to 3e5
/// digits). The weight is `node_text` at every occurrence of a leaf, with `print_weight` for the
/// digits of an integer, and the folded literal in place of a subtree that `simplify` may fold
/// (`Fold`). Release, 8 MB thread (num-bigint 0.4.8): an integer of d digits prints at about
/// 0.21 d^(1/2) ns per digit above 10^4 digits, and `print_weight` grows as d^(3/2) above
/// 2^16 digits, so 2^27 units print in about 7 to 8 s whatever the size of the integers
/// (measured at the cap: 10^5-digit leaves 6.9 s, 3.1e5-digit folded leaves 7.6 s, 4.2e5-digit
/// leaves 7.3 s; `fu2/s1impl/fix3/probe/edge1.log`). This cap bounds the printing of a result
/// and of `simplify`'s output (at most about 8 s, 7.8 s measured for `sin((2^254+1)^4096)/(1-x)`
/// at order 192, its largest accepted order, review 4), and the size of the literals `simplify`
/// builds. It does not bound the time of `simplify`'s folds: the input's own literal arithmetic in
/// a kernel's source folds left to right (review 4, B1: 35 s for a 24 KB chain whose folded
/// literal prints 638 KB). `MAX_RESULT_FOLD_WORK` bounds that (S1 fix-up round 4).
///
/// S1 fix-up round 5 (review 5, B1): the same cap also bounds `TextBound::lit_text`, the printing
/// weight of the literal at every node of the tree and every occurrence (a leaf's, and the one an
/// inner node may fold to), the payload that the output of `simplify` after `differentiate` may
/// print: the quotient and power rules keep the values of inner nodes (d(3/a) holds a^2), so the
/// derivative of the chain `3/(3/(...(3/K)...))` with 255 levels printed 160 MB in 20.6 s before
/// this count. Its edge is the text cap's for a result whose literals are all printed
/// (`sin((2^254+1)^4096)/(1-x)` stays a value up to order 192).
///
/// The bound that holds (the node cap, the three per-occurrence caps, `MAX_RESULT_MEMO_WORDS` and
/// `simplify`'s memo): for an accepted result, `simplify` of it and of the output of
/// `substitute(x, 2)`, `expand` and `differentiate(x)` each fold at most about 2^31 units of the
/// model (plus the new folds of the substituted value, and of the derivative's own constants),
/// `Display` of each of these outputs takes at most about 8 s, and `substitute` and `==` one visit
/// per node of the tree. Measured at the edges of the caps (release, 8 MB thread, 14 families of
/// literal-heavy kernel sources at orders 1 and 10, bisected to the largest accepted size,
/// `fu2/s1impl/fix5/probe/edges_final.log`): `simplify` after `substitute` or `expand` at most
/// 4.1 s (a sum of 1 487 fractions 1/p in a kernel at order 10, 1.9 ns per unit of the model) and
/// after `differentiate` at most 0.9 s; `Display` of the derivative's `simplify` at most 7.9 s
/// (60 MB, the 16-level `3/(...)` chain at order 10); `sin((2^254+1)^4096)/(1-x)` at order 192:
/// `simplify` after `substitute` 1.5 s and after `differentiate` 2.9 s, each output printed in
/// about 8 s; VmHWM at most 320 MB (`exp(x)` at order 8996, every consumer in one process). All
/// within S1_BAR (c). `exp(x)`, whose coefficients 1/k! are not shared, stays a value up to
/// order 8996 (the work budget of the REPL refuses it from about 9150 anyway); `simplify` after
/// `substitute(x, 2)` folds its 8 997 terms in 1.1 s.
pub const MAX_RESULT_TEXT: u64 = 1 << 27;
/// S1 fix-up round 4 (review 4, B1): the work of `simplify`'s constant folds on a result's tree
/// (in the units of the engine's charged arithmetic, the sizes from the bounds of `tree_bound`),
/// so that `simplify` of every accepted result stays within S1_BAR (c) whatever order the input's
/// literal arithmetic is written in. S1 fix-up round 5 (review 5, B1): counted at every
/// occurrence in the tree (`TextBound::total` of the root; round 4 counted each distinct node
/// once, `TextMemo::work`), so that it also bounds `simplify` after `substitute`, `expand` or
/// `differentiate`, which rebuild the tree and fold a shared kernel source once per term:
/// `sin(K/3 x 2535)/(1-x)` at order 10 took 24.1 s after `substitute` and is now refused. See
/// `fold_binary_work` for why the engine's own charges do not bound it. Release, 8 MB thread
/// (`series_const::tests::fold_work_calibration`): 0.37 to 1.11 ns per unit on 24 shapes (chains of
/// quotients by 3, by 2^61 - 1 and under pi, sums with growing denominators, products and sums of
/// 1 to 4 rational powers with 2^19-bit parts, a balanced gcd of two 2^20-bit integers, chains of
/// `* 3` and `+ 1`), and 1.3 for a chain of quotients by 7, so the cap is at most about 3 s of
/// folding. At the cap (review 4's B1 families at their largest accepted length,
/// `tests/s1_review4.rs`): `simplify` 0.8 to 2.8 s. S1 fix-up round 5: a sum of 1 487 fractions
/// 1/p (products of primes as denominators) at order 10, at the cap, folds in 4.1 s after
/// `expand` (1.9 ns per unit), the slowest shape measured.
pub const MAX_RESULT_FOLD_WORK: u64 = 1 << 31;
/// S1 fix-up round 5 (review 5, B2): the words of the literals that `simplify` may build at the
/// distinct nodes of a result's DAG (`TextMemo::payload`, `TextBound::lit`), at most the payload
/// of `simplify`'s memo (`simplify::MEMO_PAYLOAD_WORDS`, 2^22 words, 32 MiB). A consumer that
/// clones the result's nodes and so makes them shared (`differentiate`: the product rule clones
/// both factors, sharing their children) needs every such literal in the memo at once.
pub const MAX_RESULT_MEMO_WORDS: u64 = crate::simplify::MEMO_PAYLOAD_WORDS;
/// S1 fix-up round 3 (review 3, minor m1): the digits of one integer up to which `print_weight`
/// counts one unit per character. num-bigint's decimal conversion takes 7.5 ns per digit at 10^3
/// digits, 22 at 10^4, 63 at 10^5, 130 at 4x10^5 and 264 at 1.6x10^6 (release, `fix3/probe`), so
/// a long integer costs more per character than a short one.
pub const PRINT_LINEAR_DIGITS: u64 = 1 << 16;
/// The printing weight of d characters of one integer: d up to `PRINT_LINEAR_DIGITS`, d (d /
/// `PRINT_LINEAR_DIGITS`)^(1/2) above. Convex, so splitting a literal never raises the weight.
pub fn print_weight(d: u64) -> u64 {
    if d <= PRINT_LINEAR_DIGITS {
        d
    } else {
        let w = (d as f64) * ((d as f64) / (PRINT_LINEAR_DIGITS as f64)).sqrt();
        if w >= u64::MAX as f64 {
            u64::MAX
        } else {
            w as u64
        }
    }
}
/// The weight's growth rate at t characters, minus 1: w'(t) - 1 (0 up to `PRINT_LINEAR_DIGITS`)
fn print_weight_slope_excess(t: u64) -> f64 {
    if t <= PRINT_LINEAR_DIGITS {
        0.0
    } else {
        1.5 * ((t as f64) / (PRINT_LINEAR_DIGITS as f64)).sqrt() - 1.0
    }
}
fn f64_to_u64(w: f64) -> u64 {
    if w >= u64::MAX as f64 {
        u64::MAX
    } else if w > 0.0 {
        w.ceil() as u64
    } else {
        0
    }
}
/// The characters of an integer of `bits` bits as `Display` prints it, its sign included: at most
/// 0.302 bits + 2.
fn int_chars(bits: u64) -> u64 {
    bits.saturating_mul(302) / 1000 + 2
}
/// The printing weight of an integer of `bits` bits.
fn int_text(bits: u64) -> u64 {
    print_weight(int_chars(bits))
}
/// The printing weight of a leaf, or of a call's name: `int_text` per integer and per part of a
/// rational, the length `Display` prints for a float (at most 327, review 3 B1), the name of a
/// symbol, the name of a call and its parentheses; 0 for an operator node.
fn node_text(e: &Expr) -> u64 {
    match e {
        Expr::Integer(i) => int_text(i.bit_length()),
        Expr::Rational(q) => int_text(q.numerator().bit_length())
            .saturating_add(int_text(q.denominator().bit_length())),
        Expr::Real(f) => crate::expression::real_display_len(*f),
        Expr::Symbol(s) => s.name().len() as u64,
        Expr::Function(name, _) => name.len() as u64 + 2,
        Expr::Binary(..) | Expr::Unary(..) => 0,
    }
}
/// S1 fix-up round 3 (review 3, B2): upper bounds of a rational literal that `simplify` may fold a
/// node to: a numerator of at most `num` bits, a denominator of at most 2^`den` (`den` 0: an
/// integer), and whether it may be 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fold {
    num: u64,
    den: u64,
    zero: bool,
}
/// the value of a call with special values (exp(0) = 1, sin(0) = 0, arccos(1) = 0, ...)
const ZERO_OR_ONE: Fold = Fold {
    num: 1,
    den: 0,
    zero: true,
};
/// x^0
const ONE: Fold = Fold {
    num: 1,
    den: 0,
    zero: false,
};
/// the largest literal (bits of its numerator and denominator) that `tree_text` folds exactly as
/// `simplify` does; a larger one is followed by its bit sizes only
const EXACT_FOLD_BITS: u64 = 1 << 12;
fn q_bits(q: &Rational) -> u64 {
    q.numerator().bit_length() + q.denominator().bit_length()
}
impl Fold {
    fn of_integer(i: &Integer) -> Fold {
        Fold {
            num: i.bit_length(),
            den: 0,
            zero: i.is_zero(),
        }
    }
    fn of_rational(q: &Rational) -> Fold {
        Fold {
            num: q.numerator().bit_length(),
            den: if q.denominator().is_one() {
                0
            } else {
                q.denominator().bit_length()
            },
            zero: q.numerator().is_zero(),
        }
    }
    fn merge(self, o: Fold) -> Fold {
        Fold {
            num: self.num.max(o.num),
            den: self.den.max(o.den),
            zero: self.zero || o.zero,
        }
    }
    /// the characters of its numerator and of its denominator
    fn parts(self) -> (u64, u64) {
        (
            int_chars(self.num),
            if self.den > 0 {
                int_chars(self.den.saturating_add(1))
            } else {
                0
            },
        )
    }
    /// the printing weight of the literal
    fn text(self) -> u64 {
        let (n, d) = self.parts();
        print_weight(n).saturating_add(if d > 0 { print_weight(d) } else { 0 })
    }
    fn add(self, b: Fold) -> Fold {
        Fold {
            num: self
                .num
                .saturating_add(b.den)
                .max(b.num.saturating_add(self.den))
                .saturating_add(1),
            den: self.den.saturating_add(b.den),
            zero: true,
        }
    }
    fn mul(self, b: Fold) -> Fold {
        Fold {
            num: self.num.saturating_add(b.num),
            den: self.den.saturating_add(b.den),
            zero: self.zero || b.zero,
        }
    }
    fn div(self, b: Fold) -> Fold {
        Fold {
            num: self.num.saturating_add(b.den),
            den: self.den.saturating_add(b.num),
            zero: self.zero,
        }
    }
    /// c^k (k >= 2) or c^-k, as `simplify` folds it: only when its a-priori size, k (bits(num c)
    /// + bits(den c)), is at most `MAX_FOLD_POWER_BITS`, so each part has at most that many bits
    fn power(self, k: u64, negative: bool) -> Fold {
        let cap = crate::simplify::MAX_FOLD_POWER_BITS;
        if negative {
            Fold {
                num: if self.den == 0 {
                    1
                } else {
                    k.saturating_mul(self.den.saturating_add(1)).min(cap)
                },
                den: k.saturating_mul(self.num).min(cap),
                zero: false,
            }
        } else {
            Fold {
                // |c| <= 1 (num <= 1 bit): c^k is 0 or +-1
                num: if self.num <= 1 {
                    1
                } else {
                    k.saturating_mul(self.num).min(cap)
                },
                den: k.saturating_mul(self.den).min(cap),
                zero: self.zero,
            }
        }
    }
}
/// S1 fix-up round 3 (review 3, B2): what `simplify` makes of a node, as far as `tree_text` needs
/// it: the literal it surely folds to (`int`: an `Integer` node, else a `Rational` node), or
/// bounds of a literal it may fold to.
#[derive(Clone, Debug)]
enum Val {
    Exact { v: Rational, int: bool },
    Bound(Fold),
}
impl Val {
    fn bounds(&self) -> Fold {
        match self {
            Val::Exact { v, .. } => Fold::of_rational(v),
            Val::Bound(f) => *f,
        }
    }
    /// the literal `rational_to_expr(v)` (an `Integer` when integral), followed exactly while it
    /// is small
    fn folded(v: Rational) -> Val {
        if q_bits(&v) <= EXACT_FOLD_BITS {
            let int = v.denominator().is_one();
            Val::Exact { v, int }
        } else {
            Val::Bound(Fold::of_rational(&v))
        }
    }
    fn int(n: i64) -> Val {
        Val::Exact {
            v: Rational::from_integer(Integer::from(n)),
            int: true,
        }
    }
    /// an `Integer` node equal to n
    fn is_int(&self, n: i64) -> bool {
        matches!(self, Val::Exact { v, int: true } if *v == Rational::from_integer(Integer::from(n)))
    }
    fn text(&self) -> u64 {
        match self {
            Val::Exact { v, .. } => {
                let n = int_text(v.numerator().bit_length());
                if v.denominator().is_one() {
                    n
                } else {
                    n.saturating_add(int_text(v.denominator().bit_length()))
                }
            }
            Val::Bound(f) => f.text(),
        }
    }
    fn chars(&self) -> u64 {
        let (n, d) = self.bounds().parts();
        n.max(d)
    }
}
/// S1 fix-up round 4 (review 4, B1): the work of `simplify`'s constant folds, in the units of the
/// engine's charged arithmetic (one unit is about one 64-bit word operation: `q_add`, `q_mul`,
/// `q_pow` and `COPY_WORD`), from bounds of the operands' sizes. `simplify` folds with HEAD's
/// rules in HEAD's order (left to right), the engine in its own (`product_many` multiplies a
/// `* /` chain in pairwise rounds), so the engine's own charges do not bound `simplify`'s folds of
/// a kernel's source or a hint: a chain K/3/3/.../3 with K of 2^20 bits costs the engine one
/// division by 3^n and `simplify` n divisions of a 2^20-bit numerator by a growing 3^k, each
/// normalized by a gcd (review 4: 0.037 s against 35 s at n = 12 000). The words of an integer
/// of at most `bits` bits:
fn fold_words(bits: u64) -> u64 {
    bits / 64 + 1
}
/// (numerator words, denominator words, surely an integer) of a literal bounded by f
fn fold_parts_words(f: Fold) -> (u64, u64, bool) {
    (
        fold_words(f.num),
        if f.den == 0 { 1 } else { fold_words(f.den) },
        f.den == 0,
    )
}
/// S1 fix-up round 5 (review 5, B2): the words of a literal bounded by v, as `simplify`'s memo
/// counts them (`simplify::lit_words`: bits / 64 + 1 per part, the denominator only when there is
/// one)
fn val_words(v: &Val) -> u64 {
    let f = v.bounds();
    fold_words(f.num).saturating_add(if f.den == 0 { 0 } else { fold_words(f.den) })
}
/// `as_rational_const` of a literal: a copy (and for an `Integer`, `Rational::new(n, 1)`)
fn fold_copy_work(f: Fold) -> u64 {
    let (n, d, _) = fold_parts_words(f);
    COPY_WORD.saturating_mul(n.saturating_add(d).saturating_add(n))
}
/// a fold of `+ - * /` on two rational literals bounded by a and b: the two `as_rational_const`
/// copies, the products and the normalization as `q_add` and `q_mul` charge them (a quotient
/// as a product by the reciprocal), and the copy of the result
fn fold_binary_work(op: BinaryOp, a: Fold, b: Fold) -> u64 {
    let (na, da, ia) = fold_parts_words(a);
    let (nb, db, ib) = fold_parts_words(b);
    let (core, rn, rd) = match op {
        BinaryOp::Add | BinaryOp::Sub => {
            let n = na
                .saturating_add(db)
                .max(nb.saturating_add(da))
                .saturating_add(1);
            let d = da.saturating_add(db);
            let prod = na
                .saturating_mul(db)
                .saturating_add(nb.saturating_mul(da))
                .saturating_add(da.saturating_mul(db));
            (prod.saturating_add(norm_simplify(n, d, ia && ib)), n, d)
        }
        BinaryOp::Mul => {
            let (n, d) = (na.saturating_add(nb), da.saturating_add(db));
            let prod = na.saturating_mul(nb).saturating_add(da.saturating_mul(db));
            (prod.saturating_add(norm_simplify(n, d, ia && ib)), n, d)
        }
        BinaryOp::Div => {
            let (n, d) = (na.saturating_add(db), da.saturating_add(nb));
            let prod = na.saturating_mul(db).saturating_add(da.saturating_mul(nb));
            (
                prod.saturating_add(norm_simplify(n, d, ia && b.num <= 1)),
                n,
                d,
            )
        }
        BinaryOp::Pow | BinaryOp::Mod => (0, 0, 0),
    };
    OP_BASE
        .saturating_add(fold_copy_work(a))
        .saturating_add(fold_copy_work(b))
        .saturating_add(core)
        .saturating_add(result_units(rn, rd))
}
/// the fold c^k (k >= 2, or c^-k with k >= 1) of a literal bounded by f, as `simplify_binary` runs it: HEAD's
/// `Ring::pow` (binary powering with a normalized `Rational` product at every step) on parts of at
/// most `MAX_FOLD_POWER_BITS` bits (the size guard); `q_pow`'s charge for the last step (with
/// `norm_simplify`), twice for the steps before it and for the products by c at the odd steps,
/// plus the reciprocal for k < 0
fn fold_pow_work(f: Fold, k: u64, negative: bool) -> u64 {
    let cap = crate::simplify::MAX_FOLD_POWER_BITS;
    let int = f.den == 0;
    let wn = fold_words(k.saturating_mul(f.num).min(cap));
    let wd = if int {
        1
    } else {
        fold_words(k.saturating_mul(f.den).min(cap))
    };
    let last = pow_cost(wn)
        .saturating_add(pow_cost(wd))
        .saturating_add(norm_simplify(wn, wd, int));
    let inv = if negative {
        norm_simplify(wd, wn, false).saturating_add(result_units(wn, wd))
    } else {
        0
    };
    OP_BASE
        .saturating_mul(2 + 64 - k.leading_zeros() as u64)
        .saturating_add(fold_copy_work(f))
        .saturating_add(last.saturating_mul(2))
        .saturating_add(inv)
        .saturating_add(result_units(wn, wd))
}
/// `Rational::new` of an n-word over a d-word integer in `simplify` (HEAD's `Rational`: Lehmer's
/// gcd, `rustmath_integers::lehmer`): one division of the longer by the shorter (`q_new`'s
/// n d), then Lehmer steps on two numbers of min(n, d) words, which cost about four times as much
/// per word product in release (measured in `fold_work_calibration`: a balanced gcd of two
/// 16 384-word integers 1.4 s for 2.7e8 word products, 5 ns each; a division of a 16 384-word
/// integer by a growing one about 1 ns per word product); n for an integer (d = 1)
fn norm_simplify(n: u64, d: u64, den_one: bool) -> u64 {
    if den_one {
        n
    } else {
        let short = n.min(d);
        n.saturating_mul(d)
            .saturating_add(short.saturating_mul(short).saturating_mul(4))
    }
}
/// the characters of an integer of `chars` characters as 64-bit words (0.302 characters a bit)
fn chars_words(chars: u64) -> u64 {
    chars.saturating_mul(1000) / 19_264 + 1
}

/// S1 fix-up round 3 (review 3, B1 and B2): what `tree_text` knows of one node.
#[derive(Clone, Debug)]
pub struct TextBound {
    /// an upper bound of the printing weight (`node_text` at every occurrence of a leaf) of the
    /// node's tree, and of the tree that `simplify` makes of it
    text: u64,
    /// `Some` when `simplify` may fold the node to a rational literal (`None`: it surely does not)
    val: Option<Val>,
    /// an upper bound of the characters of every integer, numerator and denominator in the
    /// node's tree and in `simplify`'s
    chars: u64,
    /// S1 fix-up round 4 (review 4, B1): an upper bound of the characters of every denominator in
    /// the node's tree and in `simplify`'s (0: none), for the literal c of the rule
    /// (c * rest) / d = (c/d) * rest
    den_chars: u64,
    /// S1 fix-up round 4 (review 4, B1): an upper bound of the work of `simplify`'s folds at this
    /// node alone (its children's are counted at the children: `TextMemo::work`)
    work: u64,
    /// S1 fix-up round 5 (review 5, B1): an upper bound of the work of `simplify`'s folds on the
    /// node's tree, every occurrence of a shared subtree counted (`work` plus the children's
    /// `total`): the folds of `simplify` after a consumer that rebuilds the tree
    total: u64,
    /// S1 fix-up round 5 (review 5, B2): an upper bound of the words of the literals `simplify`
    /// builds at this node alone (the literal it may fold to, and the c/d of (c * rest) / d)
    lit: u64,
    /// S1 fix-up round 5 (review 5, B1): an upper bound of the printing weight of the literals at
    /// every node of the tree, every occurrence counted (a leaf's, and the one a node may fold
    /// to): what the output of `simplify` after `differentiate` may print, whose quotient and
    /// power rules keep the values of inner nodes (d(3/a) has a^2 in its denominator)
    lit_text: u64,
}
impl TextBound {
    pub fn text(&self) -> u64 {
        self.text
    }
    /// S1 fix-up round 5: the fold work of the node's tree, per occurrence
    pub fn total(&self) -> u64 {
        self.total
    }
    /// S1 fix-up round 5: the literal words built at this node alone
    pub fn lit(&self) -> u64 {
        self.lit
    }
    /// S1 fix-up round 5: the printing weight of the literals at every node of the tree
    pub fn lit_text(&self) -> u64 {
        self.lit_text
    }
    fn with_val(text: u64, chars: u64, den_chars: u64, val: Option<Val>, work: u64) -> TextBound {
        let (text, chars, den_chars) = match &val {
            Some(v) => (
                text.max(v.text()),
                chars.max(v.chars()),
                den_chars.max(v.bounds().parts().1),
            ),
            None => (text, chars, den_chars),
        };
        let lit = val.as_ref().map(val_words).unwrap_or(0);
        let lit_text = val.as_ref().map(|v| v.text()).unwrap_or(0);
        TextBound {
            text,
            val,
            chars,
            den_chars,
            work,
            total: work,
            lit,
            lit_text,
        }
    }
    /// a leaf: its value when it is an integer or a rational
    fn leaf(e: &Expr) -> TextBound {
        let val = match e {
            Expr::Integer(i) => Some(if i.bit_length() <= EXACT_FOLD_BITS {
                Val::Exact {
                    v: Rational::from_integer(i.clone()),
                    int: true,
                }
            } else {
                Val::Bound(Fold::of_integer(i))
            }),
            Expr::Rational(q) => Some(if q_bits(q) <= EXACT_FOLD_BITS {
                Val::Exact {
                    v: q.clone(),
                    int: false,
                }
            } else {
                Val::Bound(Fold::of_rational(q))
            }),
            _ => None,
        };
        // S1 fix-up round 5: a leaf is shared by `simplify`, not built, so it adds no literal
        // words to the memo (`lit`); its text counts in `lit_text`
        let mut b = TextBound::with_val(node_text(e), 0, 0, val, 0);
        b.lit = 0;
        b
    }
    /// the integer literal m (an exponent of h)
    pub fn int(m: i64) -> TextBound {
        TextBound::leaf(&Expr::from(m))
    }
    /// a unary node over `a`, by the rules of `simplify_unary`: a negation of a literal is a
    /// literal (a copy, S1 fix-up round 4); exp, log, sin, tan, sinh, tanh, cos, cosh, arcsin,
    /// arctan, arccos and sqrt fold their special values 0 and 1 at 0 or 1
    pub fn unary(op: UnaryOp, a: &TextBound) -> TextBound {
        let val = match (&a.val, op) {
            (None, _) => None,
            (Some(Val::Exact { v, int }), UnaryOp::Neg) => Some(Val::Exact {
                v: -v.clone(),
                int: *int,
            }),
            (Some(Val::Bound(f)), UnaryOp::Neg) => Some(Val::Bound(*f)),
            (Some(Val::Exact { v, .. }), op) => {
                let zero = v.is_zero();
                let one = v.is_one();
                match op {
                    UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh if zero => Some(Val::int(1)),
                    UnaryOp::Sin
                    | UnaryOp::Tan
                    | UnaryOp::Sinh
                    | UnaryOp::Tanh
                    | UnaryOp::Arcsin
                    | UnaryOp::Arctan
                    | UnaryOp::Sqrt
                        if zero =>
                    {
                        Some(Val::int(0))
                    }
                    UnaryOp::Log | UnaryOp::Arccos if one => Some(Val::int(0)),
                    UnaryOp::Sqrt if one => Some(Val::int(1)),
                    _ => None,
                }
            }
            (Some(Val::Bound(_)), _) => Some(Val::Bound(ZERO_OR_ONE)),
        };
        let work = match (&a.val, op) {
            (Some(v), UnaryOp::Neg) => {
                let (n, d, _) = fold_parts_words(v.bounds());
                OP_BASE.saturating_add(result_units(n, d))
            }
            _ => 0,
        };
        let mut v = TextBound::with_val(a.text, a.chars, a.den_chars, val, work);
        v.total = work.saturating_add(a.total);
        v.lit_text = v.lit_text.saturating_add(a.lit_text);
        v
    }
    /// a call: its name, its parentheses and its arguments (`simplify` folds no call; it copies
    /// the name and the argument pointers)
    fn call(name_len: u64, args: &[TextBound]) -> TextBound {
        TextBound {
            text: args
                .iter()
                .fold(name_len + 2, |acc, a| acc.saturating_add(a.text)),
            val: None,
            chars: args.iter().map(|a| a.chars).max().unwrap_or(0),
            den_chars: args.iter().map(|a| a.den_chars).max().unwrap_or(0),
            work: OP_BASE.saturating_add(
                COPY_WORD
                    .saturating_mul((name_len.saturating_add(8 * args.len() as u64)).div_ceil(8)),
            ),
            total: 0,
            lit: 0,
            lit_text: 0,
        }
        .with_children(args)
    }
    /// S1 fix-up round 5: `total` = `work` plus the children's totals, and the children's
    /// `lit_text` added to the node's own
    fn with_children(mut self, children: &[TextBound]) -> TextBound {
        self.total = children
            .iter()
            .fold(self.work, |acc, c| acc.saturating_add(c.total));
        self.lit_text = children
            .iter()
            .fold(self.lit_text, |acc, c| acc.saturating_add(c.lit_text));
        self
    }
    /// a binary node over `l` and `r`, by the rules of `simplify_binary`: 0 + x = x, x - 0 = x,
    /// 0 * x = 0, 1 * x = x, x / 1 = x, x^0 = 1, x^1 = x, the folds of `+ - * /` on two rational
    /// literals, c^n for a rational c and an `Integer` n with |n| <= 4096 within the size guard,
    /// and (c * rest) / d = (c/d) * rest, which makes a new literal from a literal c of the left's
    /// tree and d. S1 fix-up round 4: with the work of the fold, counted as if every fold that
    /// may run runs (an upper bound)
    pub fn binary(op: BinaryOp, l: &TextBound, r: &TextBound) -> TextBound {
        use BinaryOp::*;
        let (lv, rv) = (l.val.as_ref(), r.val.as_ref());
        let both_exact = match (lv, rv) {
            (Some(Val::Exact { v: a, .. }), Some(Val::Exact { v: b, .. })) => Some((a, b)),
            _ => None,
        };
        let val = match op {
            Add | Sub => {
                if op == Add && lv.is_some_and(|v| v.is_int(0)) {
                    rv.cloned()
                } else if rv.is_some_and(|v| v.is_int(0)) {
                    lv.cloned()
                } else if let Some((a, b)) = both_exact {
                    let s = if op == Add {
                        a.clone() + b.clone()
                    } else {
                        a.clone() - b.clone()
                    };
                    Some(Val::folded(s))
                } else {
                    match (lv, rv) {
                        (Some(a), Some(b)) => Some(Val::Bound(a.bounds().add(b.bounds()))),
                        _ => None,
                    }
                }
            }
            Mul => {
                if lv.is_some_and(|v| v.is_int(0)) || rv.is_some_and(|v| v.is_int(0)) {
                    Some(Val::int(0))
                } else if lv.is_some_and(|v| v.is_int(1)) {
                    rv.cloned()
                } else if rv.is_some_and(|v| v.is_int(1)) {
                    lv.cloned()
                } else if let Some((a, b)) = both_exact {
                    if q_bits(a).saturating_add(q_bits(b)) <= EXACT_FOLD_BITS {
                        Some(Val::folded(a.clone() * b.clone()))
                    } else {
                        Some(Val::Bound(Fold::of_rational(a).mul(Fold::of_rational(b))))
                    }
                } else {
                    match (lv, rv) {
                        (Some(a), Some(b)) => Some(Val::Bound(a.bounds().mul(b.bounds()))),
                        // 0 * x = 0 when the literal side is the Integer 0
                        (Some(Val::Bound(f)), None) | (None, Some(Val::Bound(f))) if f.zero => {
                            Some(Val::Bound(ZERO_OR_ONE))
                        }
                        _ => None,
                    }
                }
            }
            Div => {
                if rv.is_some_and(|v| v.is_int(1)) {
                    lv.cloned()
                } else if let Some((a, b)) = both_exact {
                    if b.is_zero() {
                        None
                    } else if q_bits(a).saturating_add(q_bits(b)) <= EXACT_FOLD_BITS {
                        Some(Val::folded(a.clone() / b.clone()))
                    } else {
                        Some(Val::Bound(Fold::of_rational(a).div(Fold::of_rational(b))))
                    }
                } else {
                    match (lv, rv) {
                        (Some(a), Some(b)) => Some(Val::Bound(a.bounds().div(b.bounds()))),
                        _ => None,
                    }
                }
            }
            Pow => pow_val(lv, rv),
            Mod => None,
        };
        // the work: a fold of two literals (every rule before it returns an operand, which costs
        // nothing, so the fold is the bound), or the copies `as_rational_const` makes of the one
        // literal operand of a node that does not fold
        let mut work = match (op, lv, rv) {
            (Add | Sub | Mul | Div, Some(a), Some(b)) => {
                fold_binary_work(op, a.bounds(), b.bounds())
            }
            (Add | Sub | Mul | Div, Some(a), None) | (Add | Sub | Mul | Div, None, Some(a)) => {
                OP_BASE.saturating_add(fold_copy_work(a.bounds()))
            }
            (Pow, Some(b), Some(e)) => pow_fold_work(b, e),
            _ => 0,
        };
        let mut text = l.text.saturating_add(r.text);
        let mut chars = l.chars.max(r.chars);
        let mut den_chars = l.den_chars.max(r.den_chars);
        let mut div_lit = (0u64, 0u64);
        if let (Div, None, Some(b)) = (op, lv, rv) {
            // (c * rest) / d = (c/d) * rest: the parts of c/d are products of a part of c (at most
            // l.chars characters) and a part of d; the weight of a product of parts of a and b
            // characters exceeds the weights of the parts by at most b (w'(a + b) - 1)
            let (dn, dd) = b.bounds().parts();
            let db = dn.saturating_add(dd);
            let grown = l.chars.saturating_add(db);
            let extra = (db as f64) * print_weight_slope_excess(grown);
            text = text.saturating_add(f64_to_u64(extra)).saturating_add(4);
            chars = chars.max(grown);
            // S1 fix-up round 4: the denominator of c/d divides den(c) num(d), and the fold c/d
            // costs a quotient of c (a numerator of at most l.chars characters, a denominator of
            // at most l.den_chars) by d
            den_chars = den_chars.max(l.den_chars.saturating_add(dn));
            let c = Fold {
                num: chars_words(l.chars).saturating_mul(64),
                den: if l.den_chars == 0 {
                    0
                } else {
                    chars_words(l.den_chars).saturating_mul(64)
                },
                zero: true,
            };
            work = work.saturating_add(fold_binary_work(Div, c, b.bounds()));
            // S1 fix-up round 5: c/d is a new literal inside the node's value
            let cd = Val::Bound(c.div(b.bounds()));
            div_lit = (val_words(&cd), cd.text());
        }
        let mut v = TextBound::with_val(text, chars, den_chars, val, work);
        v.lit = v.lit.saturating_add(div_lit.0);
        v.total = work.saturating_add(l.total).saturating_add(r.total);
        v.lit_text = v
            .lit_text
            .saturating_add(div_lit.1)
            .saturating_add(l.lit_text)
            .saturating_add(r.lit_text);
        v
    }
    /// the bound of the tree that `balanced_arc(items, op)` builds (the same shape); its fold
    /// work is the sum of its nodes' (`TextBound::work` of the result), and (S1 fix-up round 5)
    /// its `lit` the sum of its nodes' literal words; `total` counts the items' too
    pub fn balanced(items: &[TextBound], op: BinaryOp) -> Option<TextBound> {
        fn build(t: &[TextBound], op: BinaryOp, acc: &mut (u64, u64)) -> TextBound {
            let n = t.len();
            if n == 1 {
                return t[0].clone();
            }
            if n <= 4 {
                let mut a = t[0].clone();
                for x in &t[1..] {
                    a = TextBound::binary(op, &a, x);
                    acc.0 = acc.0.saturating_add(a.work);
                    acc.1 = acc.1.saturating_add(a.lit);
                }
                return a;
            }
            let mid = n.div_ceil(2);
            let a = build(&t[..mid], op, acc);
            let b = build(&t[mid..], op, acc);
            let v = TextBound::binary(op, &a, &b);
            acc.0 = acc.0.saturating_add(v.work);
            acc.1 = acc.1.saturating_add(v.lit);
            v
        }
        if items.is_empty() {
            None
        } else if items.len() == 1 {
            let mut v = items[0].clone();
            v.work = 0;
            v.lit = 0;
            Some(v)
        } else {
            let mut acc = (0u64, 0u64);
            let mut v = build(items, op, &mut acc);
            v.work = acc.0;
            v.lit = acc.1;
            Some(v)
        }
    }
}
/// S1 fix-up round 4 (review 4, B1): the fold work of a `Pow` node whose base may be a rational
/// literal and whose exponent may be an `Integer` (the cases of `pow_val`): c^k with
/// 2 <= |k| <= 4096 or k = -1, within the size guard; an exponent bound by its bits is counted at
/// its largest value
fn pow_fold_work(base: &Val, exp: &Val) -> u64 {
    let max_k = crate::simplify::MAX_FOLD_EXPONENT as u64;
    let (k, neg) = match exp {
        Val::Exact { v, int } => {
            if !*int || !v.denominator().is_one() {
                return 0;
            }
            match NumericConversion::to_i64(v.numerator()) {
                // x^1 = x; c^-1 = 1/c is a fold
                Some(k) if k != 1 && (1..=max_k).contains(&k.unsigned_abs()) => {
                    (k.unsigned_abs(), k < 0)
                }
                _ => return 0,
            }
        }
        Val::Bound(f) => {
            let k = if f.num >= 13 {
                max_k
            } else {
                ((1u64 << f.num) - 1).clamp(1, max_k)
            };
            return fold_pow_work(base.bounds(), k, true);
        }
    };
    // the size guard of `simplify_binary`, decided for a known base (as `pow_val` does)
    if let Val::Exact { v: c, .. } = base {
        let cap = crate::simplify::MAX_FOLD_POWER_BITS;
        if q_bits(c).saturating_mul(k) > cap || (neg && c.is_zero()) {
            return 0;
        }
    }
    fold_pow_work(base.bounds(), k, neg)
}
/// `Pow` of `simplify_binary`: x^0 = 1 and x^1 = x for an `Integer` exponent 0 or 1; c^n for a
/// rational literal c and an `Integer` n with 0 < |n| <= 4096 when |n| (bits(num c) + bits(den
/// c)) <= `MAX_FOLD_POWER_BITS` (and c != 0 for n < 0)
fn pow_val(base: Option<&Val>, exp: Option<&Val>) -> Option<Val> {
    let max_k = crate::simplify::MAX_FOLD_EXPONENT as u64;
    let cap = crate::simplify::MAX_FOLD_POWER_BITS;
    match exp {
        None => None,
        // a `Rational` node, or a non-integral value: no rule of `Pow` applies
        Some(Val::Exact { v, int }) if !*int || !v.denominator().is_one() => None,
        Some(Val::Exact { v, .. }) => {
            let n = v.numerator();
            if n.is_zero() {
                return Some(Val::int(1));
            }
            if n.is_one() {
                return base.cloned();
            }
            let k = match NumericConversion::to_i64(n) {
                Some(k) if k.unsigned_abs() <= max_k => k,
                _ => return None,
            };
            let (ku, neg) = (k.unsigned_abs(), k < 0);
            match base? {
                Val::Exact { v: c, .. } => {
                    if q_bits(c).saturating_mul(ku) > cap || (neg && c.is_zero()) {
                        return None;
                    }
                    if q_bits(c).saturating_mul(ku) <= EXACT_FOLD_BITS {
                        let p = Rational::from_integer(c.numerator().pow(ku as u32));
                        let d = Rational::from_integer(c.denominator().pow(ku as u32));
                        let value = if neg { d / p } else { p / d };
                        return Some(Val::folded(value));
                    }
                    Some(Val::Bound(Fold::of_rational(c).power(ku, neg)))
                }
                Val::Bound(f) => Some(Val::Bound(f.power(ku, neg))),
            }
        }
        // an exponent that may simplify to an `Integer` n, |n| < 2^num
        Some(Val::Bound(f)) => {
            let k = if f.num >= 13 {
                max_k
            } else {
                ((1u64 << f.num) - 1).clamp(1, max_k)
            };
            let mut out = ONE;
            if let Some(b) = base {
                let b = b.bounds();
                out = out
                    .merge(b)
                    .merge(b.power(k, false))
                    .merge(b.power(k, true));
            }
            Some(Val::Bound(out))
        }
    }
}
/// S1 fix-up round 4 (review 4, B1): the memo of `tree_bound` (a node held by more than one `Arc`,
/// and each root, by address), and the fold work of the distinct nodes it has bounded so far.
/// `simplify` simplifies each distinct node once (S1 fix-up rounds 3 and 4), so `work` bounds the
/// work of its folds on every root bounded with this memo.
#[derive(Default)]
pub struct TextMemo {
    nodes: std::collections::HashMap<usize, TextBound>,
    /// the sum of `TextBound::work` over the distinct nodes bounded so far (saturating)
    pub work: u64,
    /// S1 fix-up round 5 (review 5, B2): the sum of `TextBound::lit` over the distinct nodes
    /// bounded so far (saturating): the literal words `simplify` builds at them, which its memo
    /// holds at most once each
    pub payload: u64,
}
/// S1 fix-up round 2 (review 2, B2), round 3 (review 3, B1, B2): the bound of `root`'s tree and of
/// the tree of `simplify(root)` (`TextBound`), by an explicit-stack walk of its DAG. A node held by
/// more than one `Arc` (and `root`) is memoized by address in `memo`, so a shared subtree costs
/// O(1) at each further occurrence and the walk is linear in the DAG. `memo` may be shared by
/// several roots that are alive together (the coefficients of one expansion). Round 4 (review 4,
/// B1): `memo.work` gains the fold work of each node the first time it is bounded (a node held by
/// one `Arc` has one parent, so it is bounded once with its parent).
pub fn tree_bound(root: &Expr, memo: &mut TextMemo) -> TextBound {
    fn key(e: &Expr) -> usize {
        std::ptr::from_ref(e).addr()
    }
    if let Some(v) = memo.nodes.get(&key(root)) {
        return v.clone();
    }
    /// visit a node (whether it is memoized), or combine its `n` children's bounds
    enum Step<'a> {
        Visit(&'a Expr, bool),
        Combine(&'a Expr, usize, bool),
    }
    let mut steps: Vec<Step<'_>> = vec![Step::Visit(root, true)];
    let mut vals: Vec<TextBound> = Vec::new();
    while let Some(st) = steps.pop() {
        match st {
            Step::Visit(n, m) => {
                if m {
                    if let Some(v) = memo.nodes.get(&key(n)) {
                        vals.push(v.clone());
                        continue;
                    }
                }
                let children: &[Arc<Expr>] = match n {
                    Expr::Binary(_, l, r) => {
                        steps.push(Step::Combine(n, 2, m));
                        steps.push(Step::Visit(r, Arc::strong_count(r) > 1));
                        steps.push(Step::Visit(l, Arc::strong_count(l) > 1));
                        continue;
                    }
                    Expr::Unary(_, a) => std::slice::from_ref(a),
                    Expr::Function(_, args) => args,
                    _ => &[],
                };
                steps.push(Step::Combine(n, children.len(), m));
                for c in children.iter().rev() {
                    steps.push(Step::Visit(c, Arc::strong_count(c) > 1));
                }
            }
            Step::Combine(n, k, m) => {
                let at = vals.len() - k;
                let v = match n {
                    Expr::Binary(op, _, _) => TextBound::binary(*op, &vals[at], &vals[at + 1]),
                    Expr::Unary(op, _) => TextBound::unary(*op, &vals[at]),
                    Expr::Function(name, _) => TextBound::call(name.len() as u64, &vals[at..]),
                    _ => TextBound::leaf(n),
                };
                vals.truncate(at);
                memo.work = memo.work.saturating_add(v.work);
                memo.payload = memo.payload.saturating_add(v.lit);
                if m {
                    memo.nodes.insert(key(n), v.clone());
                }
                vals.push(v);
            }
        }
    }
    vals.pop().expect("tree_bound: the root's bound")
}
/// The printing weight of `root`'s tree and of `simplify(root)`'s (`tree_bound`); `memo.work`
/// gains the fold work of the nodes bounded (used by the unit tests).
#[cfg(test)]
pub fn tree_text(root: &Expr, memo: &mut TextMemo) -> u64 {
    tree_bound(root, memo).text
}
/// the refusal of a result (or of the constant of `UndecidedZero`) whose tree, or the tree
/// `simplify` makes of it, weighs more than `MAX_RESULT_TEXT` (`tree_text`)
pub fn text_limit() -> SErr {
    SErr::Limit(
        "characters of leaf text in the result's tree",
        MAX_RESULT_TEXT,
    )
}
/// S1 fix-up round 5 (review 5, B1): the refusal of a result (or of the constant of
/// `UndecidedZero`) whose literals at every node and occurrence (`TextBound::lit_text`) weigh
/// more than `MAX_RESULT_TEXT`
pub fn literal_text_limit() -> SErr {
    SErr::Limit(
        "characters of the literals at every node of the result's tree",
        MAX_RESULT_TEXT,
    )
}
/// S1 fix-up round 4 (review 4, B1): the refusal of a result (or of the constant of
/// `UndecidedZero`) whose `simplify` may do more than `MAX_RESULT_FOLD_WORK` units of folding
/// (`TextMemo::work`)
pub fn fold_limit() -> SErr {
    SErr::Limit(
        "units of constant folding in the simplification of the result",
        MAX_RESULT_FOLD_WORK,
    )
}
/// S1 fix-up round 5 (review 5, B2): the refusal of a result (or of the constant of
/// `UndecidedZero`) whose distinct nodes `simplify` may fold to more than `MAX_RESULT_MEMO_WORDS`
/// words of literals (`TextMemo::payload`)
pub fn memo_limit() -> SErr {
    SErr::Limit(
        "words of the literals that the simplification of the result builds at its distinct nodes",
        MAX_RESULT_MEMO_WORDS,
    )
}
/// The caps of one rendered value's tree (`tree_bound`), the refusal if any: the text
/// (`MAX_RESULT_TEXT`), the fold work per occurrence (`MAX_RESULT_FOLD_WORK`, S1 fix-up round 5)
/// and the literal words of the distinct nodes (`MAX_RESULT_MEMO_WORDS`, round 5).
pub fn bound_caps(text: u64, total: u64, payload: u64, lit_text: u64) -> Option<SErr> {
    if text > MAX_RESULT_TEXT {
        Some(text_limit())
    } else if lit_text > MAX_RESULT_TEXT {
        Some(literal_text_limit())
    } else if total > MAX_RESULT_FOLD_WORK {
        Some(fold_limit())
    } else if payload > MAX_RESULT_MEMO_WORDS {
        Some(memo_limit())
    } else {
        None
    }
}
/// The caps of one rendered value (the constant of `UndecidedZero`): the refusal, if any
pub fn result_caps(e: &Expr) -> Option<SErr> {
    let mut memo = TextMemo::default();
    let b = tree_bound(e, &mut memo);
    bound_caps(b.text, b.total, memo.payload, b.lit_text)
}
pub const KERNEL_DEPTH_FACTOR: u32 = 4;
/// revision 5 (review 5, B2): units per factor of a monomial that a product, a power or an
/// inverse builds: 16 bytes of heap (at about 4 bytes per unit) plus the merge step
/// revision 6: 6 (= COPY_FACTOR; revision 5: 5, 3.2 bytes per unit)
pub const MONO_FACTOR_UNITS: u64 = 6;
/// map searches (review 5, B2): comparing two monomials walks their common prefix. The first
/// SEARCH_FREE factors of each operand's monomials are covered by the per-pair base (the q_mul and
/// q_add bases and MONO_FACTOR_UNITS); each further factor is charged SEARCH_CMP comparisons per
/// level of the B-tree (bits(t) + 1 levels for a map of t keys).
pub const SEARCH_FREE: u64 = 16;
pub const SEARCH_CMP: u64 = 3;
pub fn bits_u(t: u64) -> u64 {
    64 - t.leading_zeros() as u64 + 1
}
/// units of the searches of `n_ins` insertions of monomials with total excess `excess` into a
/// map that holds at most `t` keys
pub fn search_units(excess: u64, t: u64) -> u64 {
    SEARCH_CMP.saturating_mul(bits_u(t)).saturating_mul(excess)
}
/// revision 7rt3: units of one entry of the hint table (about 40 bytes, at most doubled while it grows)
pub const HINT_UNITS: u64 = 32;
thread_local! {
    /// revision 7rt3: the rendering hints of the current attempt (the design keeps them in Work)
    pub static HINT_T: std::cell::RefCell<std::collections::HashMap<usize, (Konst, Hint)>> = std::cell::RefCell::new(std::collections::HashMap::new());
    pub static HINT_N: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
/// revision 7rt3: the hint table is cleared at the start of every attempt (its constants are that
/// attempt's), and after the result is rendered
pub fn clear_hints() {
    let old = HINT_T.with(|t| std::mem::take(&mut *t.borrow_mut()));
    HINT_N.with(|c| c.set(0));
    drop(old);
}
thread_local! {
    /// revision 5: the allowed kernel depth of the current expansion
    pub static KDEPTH_ALLOWED: std::cell::Cell<u32> = const { std::cell::Cell::new(MAX_KERNEL_DEPTH) };
}
pub fn limit(what: &'static str, v: u64) {
    set_fault(SErr::Limit(what, v));
}
pub fn iw(n: &Integer) -> u64 {
    n.bit_length() / 64 + 1
}
fn norm_cost(nw: u64, dw: u64, den_one: bool) -> u64 {
    if den_one {
        nw
    } else {
        nw.saturating_mul(dw)
    }
}
/// an upper bound of the bit length of b^n for an integer b: n log2|b| + 1, with log2 taken
/// from the leading 53 bits (rounded up) and the exponent of the rest
pub fn pow_bits(b: &Integer, n: u64) -> u64 {
    let l = b.bit_length();
    if l <= 1 {
        return 1;
    }
    let shift = l.saturating_sub(53);
    let top = b.abs().as_bigint().magnitude() >> shift as usize;
    let topf: f64 = rustmath_core::NumericConversion::to_f64(&Integer::from(
        num_traits::ToPrimitive::to_u64(&top).unwrap_or(u64::MAX) as i64,
    ))
    .unwrap_or(9.007199254740992e15);
    // |b| = top exactly when shift = 0, else |b| < (top + 1) 2^shift
    let lg = if shift == 0 {
        topf.log2()
    } else {
        (topf + 1.0).log2() + shift as f64
    };
    let bits = (n as f64) * lg * (1.0 + 1e-12) + 2.0;
    if bits >= u64::MAX as f64 {
        u64::MAX
    } else {
        bits.ceil() as u64
    }
}
/// revision 7 (review 7, m4): the floor of a charged operation's units, COPY_WORD per word of its
/// result (num words + den words), so that every result is charged at most 3 bytes per unit
fn result_units(nw: u64, dw: u64) -> u64 {
    COPY_WORD.saturating_mul(nw.saturating_add(dw))
}
fn pow_cost(w: u64) -> u64 {
    (w.saturating_mul(w) / 64).saturating_add(w)
}
pub fn q_mul(a: &Rational, b: &Rational) -> Rational {
    let (na, da, nb, db) = (
        iw(a.numerator()),
        iw(a.denominator()),
        iw(b.numerator()),
        iw(b.denominator()),
    );
    let den_one = a.denominator().is_one() && b.denominator().is_one();
    // revision 7rt4: every units formula saturates (section 2.7)
    let c = OP_BASE
        .saturating_add(na.saturating_mul(nb))
        .saturating_add(da.saturating_mul(db))
        .saturating_add(norm_cost(na + nb, da + db, den_one));
    // revision 7 (review 7, m4): at least COPY_WORD per word of the result (a product by a
    // one-word operand was charged about 2 units per result word, 4 bytes per unit)
    let c = c.max(result_units(na + nb, da + db));
    if !charge(c) {
        return Rational::zero();
    }
    a * b
}
pub fn q_add(a: &Rational, b: &Rational) -> Rational {
    let (na, da, nb, db) = (
        iw(a.numerator()),
        iw(a.denominator()),
        iw(b.numerator()),
        iw(b.denominator()),
    );
    let den_one = a.denominator().is_one() && b.denominator().is_one();
    let prod = na
        .saturating_mul(db)
        .saturating_add(nb.saturating_mul(da))
        .saturating_add(da.saturating_mul(db));
    let c = OP_BASE.saturating_add(prod).saturating_add(norm_cost(
        (na + db).max(nb + da) + 1,
        da + db,
        den_one,
    ));
    // revision 7 (review 7, m4): at least COPY_WORD per word of the result
    let c = c.max(result_units((na + db).max(nb + da) + 1, da + db));
    if !charge(c) {
        return Rational::zero();
    }
    a + b
}
pub fn q_sub(a: &Rational, b: &Rational) -> Rational {
    q_add(a, &-b)
}
/// q^n for n >= 0 (the result size is bounded by the callers' caps first)
pub fn q_pow(q: &Rational, n: u64) -> Rational {
    // result sizes: bits(q^n) <= n bits(q)
    let wn = pow_bits(q.numerator(), n) / 64 + 1;
    let wd = pow_bits(q.denominator(), n) / 64 + 1;
    let c = OP_BASE
        .saturating_add(pow_cost(wn))
        .saturating_add(pow_cost(wd))
        .saturating_add(norm_cost(wn, wd, q.denominator().is_one()));
    if !charge(c) || n > u32::MAX as u64 {
        return Rational::zero();
    }
    let num = q.numerator().pow(n as u32);
    let den = q.denominator().pow(n as u32);
    Rational::new(num, den).unwrap()
}
pub fn i_pow(n: &Integer, e: u32) -> Integer {
    if !charge(OP_BASE.saturating_add(pow_cost(pow_bits(n, e as u64) / 64 + 1))) {
        return Integer::from(1);
    }
    n.pow(e)
}
pub fn i_mul(a: &Integer, b: &Integer) -> Integer {
    // revision 7 (review 7, m4): at least COPY_WORD per word of the result
    if !charge(
        OP_BASE
            .saturating_add(iw(a).saturating_mul(iw(b)))
            .max(COPY_WORD.saturating_mul(iw(a) + iw(b))),
    ) {
        return Integer::from(0);
    }
    a.clone() * b.clone()
}
pub fn i_gcd(a: &Integer, b: &Integer) -> Integer {
    if !charge(OP_BASE.saturating_add(iw(a).saturating_mul(iw(b)))) {
        return Integer::from(1);
    }
    a.gcd(b)
}
pub fn i_lcm(a: &Integer, b: &Integer) -> Integer {
    // gcd, product, exact division
    if !charge(OP_BASE.saturating_add(iw(a).saturating_mul(iw(b)).saturating_mul(3))) {
        return Integer::from(1);
    }
    a.lcm(b)
}
/// Rational::new(n, d) for d != 0 (the gcd of n and d)
pub fn q_new(n: Integer, d: Integer) -> Rational {
    if !charge(OP_BASE.saturating_add(norm_cost(iw(&n), iw(&d), d.is_one()))) {
        return Rational::zero();
    }
    Rational::new(n, d).unwrap()
}
/// exact d-th root of n >= 0 (Integer::exact_root, I3: power-residue sieve, 2-adic root, then
/// at most log2 d Newton square roots)
pub fn i_exact_root(n: &Integer, d: u32) -> Option<Integer> {
    let w = iw(n);
    if !charge(OP_BASE.saturating_add(w.saturating_mul(w))) {
        return None;
    }
    n.exact_root(d)
}

// revision 4 (review 4, B1, m6): sizes of copies. Revision 6 (review 6, B2 and B3): every heap
// object the engine allocates is charged at most about 3 bytes per unit (section 2.16, "Memory"),
// glibc chunks included (request + 8, rounded up to 16, at least 32 bytes).
/// per term of a constant: its 88-byte slot (Mono, Rational), the allocator overhead of the
/// monomial vector and of the two digit vectors (152 bytes for a rational term: 2.9 bytes per unit)
pub const COPY_TERM: u64 = 52;
/// per factor of a monomial: 16 bytes
pub const COPY_FACTOR: u64 = 6;
/// per 64-bit word of a numerator or a denominator: 8 bytes
pub const COPY_WORD: u64 = 3;
/// revision 6: per constant built (its Arc, KMap header and cached sizes, a 208-byte chunk)
pub const CONST_BASE: u64 = 72;
/// per Expr node that a rendering allocates: an Arc<Expr> is an 80-byte request, a 96-byte chunk
/// (revision 5: 20, about 4.8 bytes per unit)
pub const EXPR_NODE: u64 = 32;
/// per word of an integer an Expr node copies, and per integer (its digit vector's overhead)
pub const EXPR_WORD: u64 = 3;
pub const EXPR_INT: u64 = 11;
/// revision 5: per new kernel, before it is created (its KernelData, the table entry; the key is
/// charged per byte and the constant that holds the kernel by CONST_BASE)
pub const KERNEL_UNITS: u64 = 128;
/// revision 5: per node visit of konst_of and of expand
pub const KONST_NODE: u64 = 16;
/// revision 6: per series term a series operation copies or produces (a 16-byte slot; revision 5: 4)
pub const SER_TERM: u64 = 6;
/// revision 6: per pair of a product accumulated in a map (its share of a B-tree node, up to
/// 176 bytes at half fill, with the monomial's allocation)
pub const ACC_ENTRY: u64 = 32;
fn rw(q: &Rational) -> u64 {
    iw(q.numerator()) + iw(q.denominator())
}
/// a-priori cost of copying every term of c into a new map (merge, negation, grouping)
pub fn copy_units(c: &Konst) -> u64 {
    c.t.sz.copy
}
/// a-priori cost of to_expr(c): the nodes it allocates and the integer words it copies (the
/// kernels' expressions are shared, not copied)
pub fn expr_units(c: &Konst) -> u64 {
    c.t.sz.expr
}
fn rat_words(q: &Rational) -> u64 {
    (q.numerator().bit_length() + q.denominator().bit_length()) / 64 + 1
}
fn rat_bits(q: &Rational) -> u64 {
    q.numerator().bit_length() + q.denominator().bit_length()
}

/// size limits of exact constant powers (larger results stay a kernel c^n)
pub const MAX_EXACT_POWER_BITS: u64 = 1 << 20;
pub const MAX_EXACT_POWER_TERMS: u128 = 512;
pub const MAX_EXACT_EXPONENT: i64 = 4096;
/// a radical factor n^q is folded into the rational coefficient only below this size
pub const MAX_FOLD_BITS: u64 = 1 << 20;

fn pow_rat_i(q: &Rational, n: i64) -> Rational {
    // exact q^n, charged (revision 3); callers bound the size first
    let r = q_pow(q, n.unsigned_abs());
    if n >= 0 || r.is_zero() {
        r
    } else {
        r.reciprocal().unwrap()
    }
}

impl Konst {
    pub fn zero() -> Self {
        ZERO_K.with(|z| z.clone())
    }
    pub fn rat(q: Rational) -> Self {
        if q.is_zero() {
            return Konst::zero();
        }
        if q.is_one() {
            return Konst::one();
        }
        // revision 6: the digit vectors of its numerator and denominator (the header and the
        // inline term are CONST_BASE): 272 bytes for 94 units for a one-word rational
        if !charge(COPY_WORD.saturating_mul(rw(&q)).saturating_add(16)) {
            return Konst::zero();
        }
        Konst::from_sorted(vec![(Vec::new(), q)])
    }
    pub fn int(n: i64) -> Self {
        Konst::rat(qi(n))
    }
    pub fn one() -> Self {
        ONE_K.with(|o| o.clone())
    }
    pub fn kernel(k: Kernel) -> Self {
        // revision 6: the one-factor monomial and the digit vectors of the coefficient 1 (the header
        // and the inline term are CONST_BASE): 304 bytes for 104 units
        if !charge(32) {
            return Konst::zero();
        }
        Konst::from_sorted(vec![(vec![(k, 1)], qi(1))])
    }
    fn hint_key(&self) -> usize {
        Arc::as_ptr(&self.t).addr()
    }
    /// revision 7rt3: the hint of this constant, if any (one lookup; none while the table is empty)
    pub fn hint(&self) -> Option<(Arc<Expr>, u64, u32)> {
        if HINT_N.with(|c| c.get()) == 0 {
            return None;
        }
        HINT_T.with(|t| {
            t.borrow()
                .get(&self.hint_key())
                .map(|(_, h)| (h.e.clone(), h.tree, h.depth))
        })
    }
    /// revision 7rt3: the tree size of this constant's rendering (its hint's when it has one)
    pub fn tree(&self) -> u64 {
        match self.hint() {
            Some(h) => h.1,
            None => self.t.sz.tree,
        }
    }
    /// revision 7rt3: the depth of this constant's rendering (its hint's when it has one)
    pub fn depth_eff(&self) -> u32 {
        match self.hint() {
            Some(h) => h.2,
            None => self.t.sz.depth,
        }
    }
    pub fn has_hint(&self) -> bool {
        self.hint().is_some()
    }
    /// revision 7rt3: set the rendering hint (the first one stays), charged HINT_UNITS for the entry
    pub fn set_hint(&self, h: Hint) {
        if self.has_hint() || !charge(HINT_UNITS) {
            return;
        }
        let k = self.hint_key();
        HINT_T.with(|t| t.borrow_mut().insert(k, (self.clone(), h)));
        HINT_N.with(|c| c.set(c.get() + 1));
    }
    pub fn is_zero(&self) -> bool {
        self.t.is_empty()
    }
    pub fn as_rational(&self) -> Option<Rational> {
        if self.t.is_empty() {
            return Some(Rational::zero());
        }
        if self.t.len() == 1 {
            if let Some(q) = self.t.get(&Vec::new()) {
                return Some(q.clone());
            }
        }
        None
    }
    pub fn words(&self) -> u64 {
        self.t.sz.words
    }
    /// revision 6: a + b or a - b by one two-pointer merge. Charged first: copy_units of both (the
    /// copies, and the comparisons: each walks a common prefix no longer than the monomial it
    /// advances past, so the merge compares at most F factors in all), then q_add per common
    /// monomial; the term cap before each new term.
    // revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
    #[inline(never)]
    fn merge2(&self, o: &Konst, neg_o: bool) -> Konst {
        if o.t.is_empty() {
            return self.clone();
        }
        if self.t.is_empty() {
            return if neg_o { o.neg() } else { o.clone() };
        }
        if !charge(copy_units(self).saturating_add(copy_units(o))) {
            return Konst::zero();
        }
        let (a, b) = (self.t.as_slice(), o.t.as_slice());
        let mut out: Vec<Term> = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            let ord = if i >= a.len() {
                Ordering::Greater
            } else if j >= b.len() {
                Ordering::Less
            } else {
                a[i].0.cmp(&b[j].0)
            };
            let t = match ord {
                Ordering::Less => {
                    i += 1;
                    a[i - 1].clone()
                }
                Ordering::Greater => {
                    j += 1;
                    let (m, q) = &b[j - 1];
                    (m.clone(), if neg_o { -q.clone() } else { q.clone() })
                }
                Ordering::Equal => {
                    i += 1;
                    j += 1;
                    let s = if neg_o {
                        q_sub(&a[i - 1].1, &b[j - 1].1)
                    } else {
                        q_add(&a[i - 1].1, &b[j - 1].1)
                    };
                    if s.is_zero() {
                        continue;
                    }
                    (a[i - 1].0.clone(), s)
                }
            };
            // revision 5: the number of terms of one constant is capped
            if out.len() >= MAX_CONST_TERMS {
                limit("terms of one constant", MAX_CONST_TERMS as u64);
                return Konst::zero();
            }
            out.push(t);
        }
        if faulted() {
            return Konst::zero();
        }
        Konst::from_sorted(out)
    }
    pub fn add(&self, o: &Konst) -> Konst {
        self.merge2(o, false)
    }
    pub fn neg(&self) -> Konst {
        // revision 4 (review 4, B1): a negation copies every term
        if self.t.is_empty() {
            return Konst::zero();
        }
        if !charge(copy_units(self)) {
            return Konst::zero();
        }
        let out: Vec<Term> = self
            .t
            .iter()
            .map(|(m, q)| (m.clone(), -q.clone()))
            .collect();
        Konst::from_sorted(out)
    }
    pub fn sub(&self, o: &Konst) -> Konst {
        self.merge2(o, true)
    }
    /// revision 5: the sum of several signed constants by one merge (a chain a1 + a2 - a3 + ...
    /// along the left spine of the input): linear in the terms, where repeated `add` copies the
    /// growing accumulator at every step (quadratic).
    /// revision 6: a k-way merge with a binary heap of the k summands' cursors (no map): charged
    /// first copy_units of every summand and search_units(excess, k) for the heap's comparisons
    /// (about 2 log2 k per term, each walking a common prefix; the first SEARCH_FREE factors of a
    /// comparison are covered by COPY_TERM); the term cap before each new term.
    // revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
    #[inline(never)]
    pub fn sum_many(items: &[(bool, Konst)]) -> Konst {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let nz: Vec<&(bool, Konst)> = items.iter().filter(|(_, c)| !c.is_zero()).collect();
        match nz.len() {
            0 => return Konst::zero(),
            1 => {
                return if nz[0].0 {
                    nz[0].1.neg()
                } else {
                    nz[0].1.clone()
                }
            }
            2 if !nz[0].0 => return nz[0].1.merge2(&nz[1].1, nz[1].0),
            _ => {}
        }
        let k = nz.len() as u64;
        let copy: u64 = nz
            .iter()
            .map(|(_, c)| copy_units(c))
            .fold(0u64, |a, b| a.saturating_add(b));
        let excess: u64 = nz
            .iter()
            .map(|(_, c)| c.t.sz.excess)
            .fold(0u64, |a, b| a.saturating_add(b));
        let total: usize = nz.iter().map(|(_, c)| c.t.len()).sum();
        if !charge(
            copy.saturating_add(search_units(excess, k))
                .saturating_add(SER_TERM.saturating_mul(k)),
        ) {
            return Konst::zero();
        }
        let srcs: Vec<(bool, &[Term])> = nz.iter().map(|(n, c)| (*n, c.t.as_slice())).collect();
        let mut pos = vec![0usize; srcs.len()];
        let mut heap: BinaryHeap<Reverse<(&Mono, usize)>> = BinaryHeap::with_capacity(srcs.len());
        for (s, (_, v)) in srcs.iter().enumerate() {
            heap.push(Reverse((&v[0].0, s)));
        }
        let mut out: Vec<Term> = Vec::with_capacity(total.min(MAX_CONST_TERMS));
        let mut last_zero = false;
        while let Some(Reverse((m, s))) = heap.pop() {
            let (neg, v) = srcs[s];
            let q = &v[pos[s]].1;
            let same = matches!(out.last(), Some((lm, _)) if lm == m);
            if same {
                let lq = &mut out.last_mut().unwrap().1;
                let r = if neg { q_sub(lq, q) } else { q_add(lq, q) };
                last_zero = r.is_zero();
                *lq = r;
            } else {
                if last_zero {
                    out.pop();
                    last_zero = false;
                }
                if out.len() >= MAX_CONST_TERMS {
                    limit("terms of one constant", MAX_CONST_TERMS as u64);
                    return Konst::zero();
                }
                out.push((m.clone(), if neg { -q.clone() } else { q.clone() }));
            }
            pos[s] += 1;
            if pos[s] < v.len() {
                heap.push(Reverse((&v[pos[s]].0, s)));
            }
        }
        if last_zero {
            out.pop();
        }
        if faulted() {
            return Konst::zero();
        }
        Konst::from_sorted(out)
    }
    /// revision 5: the product of several constants (a chain a1 * a2 * ... along the left spine)
    /// by rounds of pairwise products, so a product of N single monomials copies O(N log N)
    /// factors, where a left-deep accumulation copies O(N^2)
    /// revision 6 (review 6, m2): a zero factor gives 0 before any round, so the outcome does
    /// not depend on where the zero sits in the chain
    // revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
    #[inline(never)]
    pub fn product_many(mut items: Vec<Konst>) -> Konst {
        if items.is_empty() {
            return Konst::one();
        }
        if items.iter().any(|c| c.is_zero()) {
            return Konst::zero();
        }
        while items.len() > 1 {
            if faulted() {
                return Konst::zero();
            }
            let mut next = Vec::with_capacity(items.len().div_ceil(2));
            let mut it = items.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => next.push(a.mul(&b)),
                    None => next.push(a),
                }
            }
            items = next;
        }
        items.pop().unwrap()
    }
    pub fn scale(&self, s: &Rational) -> Konst {
        if s.is_zero() || self.t.is_empty() {
            return Konst::zero();
        }
        // revision 5: scaling by 1 shares the map (an O(1) copy)
        if s.is_one() {
            return self.clone();
        }
        // revision 3: each term's product is charged by q_mul (product and normalization)
        // revision 5 (review 5, B2): the new terms copy every monomial, charged by copy_units
        if !charge(copy_units(self)) {
            return Konst::zero();
        }
        let out: Vec<Term> = self
            .t
            .iter()
            .map(|(m, q)| (m.clone(), q_mul(q, s)))
            .collect();
        if faulted() {
            return Konst::zero();
        }
        Konst::from_sorted(out)
    }
    /// product of two monomials: (product, negate, rational factor from radicals)
    /// exponent arithmetic is checked: an overflow sets a fault (Err(Unsupported) in the design)
    fn mono_mul(a: &Mono, b: &Mono) -> (Mono, bool, Option<Rational>) {
        let mut out: Mono = Vec::with_capacity((a.len() + b.len()).min(MAX_MONO_FACTORS));
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            // revision 5: the number of factors of one monomial is capped (checked before a push)
            if out.len() >= MAX_MONO_FACTORS {
                limit("factors of one monomial", MAX_MONO_FACTORS as u64);
                return (Vec::new(), false, None);
            }
            if j >= b.len() || (i < a.len() && a[i].0 < b[j].0) {
                out.push(a[i].clone());
                i += 1;
            } else if i >= a.len() || b[j].0 < a[i].0 {
                out.push(b[j].clone());
                j += 1;
            } else {
                match a[i].1.checked_add(b[j].1) {
                    Some(0) => {}
                    Some(e) => out.push((a[i].0.clone(), e)),
                    None => {
                        set_fault(SErr::Unsupported("a kernel exponent overflows i64".into()));
                        return (Vec::new(), false, None);
                    }
                }
                i += 1;
                j += 1;
            }
        }
        Konst::reduce_mono(out)
    }
    /// canonical reductions of one monomial: radicals n^(1/d) to exponents in [0, d) (when the
    /// folded factor is small), I^e to +-I^(e mod 4)
    /// revision 5: one pass, O(L) (revision 4 removed folded factors one by one, O(L^2))
    fn reduce_mono(mut out: Mono) -> (Mono, bool, Option<Rational>) {
        // revision 6: compacted in place (revision 5 built a second vector per pair)
        let mut neg = false;
        let mut scale_q: Option<Rational> = None;
        let mut w = 0;
        for r in 0..out.len() {
            let mut e = out[r].1;
            let mut keep = true;
            if let Some((n, d)) = &out[r].0 .0.root {
                let d = *d as i64;
                let (q, rr) = (e.div_euclid(d), e.rem_euclid(d));
                if q != 0 && q.unsigned_abs().saturating_mul(n.bit_length()) <= MAX_FOLD_BITS {
                    // revision 3 (review 3, m2): the fold n^(e div d) is charged by q_pow
                    let f = pow_rat_i(&Rational::from_integer(n.clone()), q);
                    scale_q = Some(scale_q.map(|s| q_mul(&s, &f)).unwrap_or(f));
                    if rr == 0 {
                        keep = false;
                    }
                    e = rr;
                }
            }
            if keep && out[r].0 .0.class == KClass::Imag {
                // a monomial has at most one factor I (kernels are interned)
                match e.rem_euclid(4) {
                    0 => keep = false,
                    1 => e = 1,
                    2 => {
                        neg = !neg;
                        keep = false;
                    }
                    _ => {
                        e = 1;
                        neg = !neg;
                    }
                }
            }
            if keep {
                out[r].1 = e;
                out.swap(w, r);
                w += 1;
            }
        }
        out.truncate(w);
        (out, neg, scale_q)
    }
    // revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
    #[inline(never)]
    pub fn mul(&self, o: &Konst) -> Konst {
        if self.t.is_empty() || o.t.is_empty() {
            return Konst::zero();
        }
        // fast path: rational * anything
        if let Some(q) = self.as_rational() {
            return o.scale(&q);
        }
        if let Some(q) = o.as_rational() {
            return self.scale(&q);
        }
        // a-priori charge of the monomial work of all pairs (revision 3: the rational products
        // and sums are charged by q_mul and q_add, by their own sizes)
        let (ta, tb) = (self.t.len() as u64, o.t.len() as u64);
        let la: u64 = self.t.sz.factors + ta;
        let lb: u64 = o.t.sz.factors + tb;
        // revision 5 (review 5, B2): each product monomial is built (MONO_FACTOR_UNITS per factor:
        // its memory and the merge) and searched in the accumulator (search_units)
        // revision 6 (review 6, B2): plus ACC_ENTRY per pair for its accumulator slot
        let pairs = ta.saturating_mul(tb);
        let acc_max = pairs.min(MAX_CONST_TERMS as u64);
        let mono = MONO_FACTOR_UNITS
            .saturating_mul(ta.saturating_mul(lb).saturating_add(tb.saturating_mul(la)));
        let search = search_units(
            tb.saturating_mul(self.t.sz.excess)
                .saturating_add(ta.saturating_mul(o.t.sz.excess)),
            acc_max,
        );
        if !charge(
            mono.saturating_add(search)
                .saturating_add(ACC_ENTRY.saturating_mul(pairs)),
        ) {
            return Konst::zero();
        }
        let mut acc: BTreeMap<Mono, Rational> = BTreeMap::new();
        for (ma, qa) in self.t.iter() {
            for (mb, qb) in o.t.iter() {
                let (m, neg, sq) = Konst::mono_mul(ma, mb);
                if faulted() {
                    return Konst::zero();
                }
                let mut q = q_mul(qa, qb);
                if let Some(sq) = sq {
                    q = q_mul(&q, &sq);
                }
                if neg {
                    q = -q;
                }
                // revision 5: the number of terms of one constant is capped (checked before a
                // new term is inserted)
                let full = acc.len() >= MAX_CONST_TERMS;
                match acc.entry(m) {
                    std::collections::btree_map::Entry::Occupied(mut p) => {
                        let s = q_add(p.get(), &q);
                        if s.is_zero() {
                            p.remove();
                        } else {
                            *p.get_mut() = s;
                        }
                    }
                    std::collections::btree_map::Entry::Vacant(v) => {
                        if full {
                            limit("terms of one constant", MAX_CONST_TERMS as u64);
                            return Konst::zero();
                        }
                        v.insert(q);
                    }
                }
            }
        }
        // the map's iteration order is the monomial order; the conversion is linear (an exact
        // size), and the map's nodes are freed as it is consumed
        Konst::from_sorted(acc.into_iter().collect())
    }

    /// Three-valued zero test (see the design, section "Zero test").
    pub fn zero_status(&self) -> ZS {
        if self.t.is_empty() {
            return ZS::Zero;
        }
        if self.t.len() == 1 {
            // revision 5: the class test reads the cached flag (O(1), no walk of the monomial)
            if !self.t.sz.any_opaque {
                return ZS::NonZero;
            }
            return self.numeric();
        }
        // revision 6: without indeterminates there is one group, the constant itself (no copy)
        if !self.t.sz.any_indet {
            return self.const_status();
        }
        match self.indet_groups() {
            None => self.numeric(),
            Some(groups) => {
                if groups.len() > 1 || groups.first().map(|(k, _)| !k.is_empty()).unwrap_or(false) {
                    // a Laurent polynomial in indeterminates with constant coefficients:
                    // nonzero iff some coefficient is nonzero
                    for (_, g) in groups.iter() {
                        if g.const_status() == ZS::NonZero {
                            return ZS::NonZero;
                        }
                    }
                    return ZS::Unknown;
                }
                self.const_status()
            }
        }
    }

    pub fn indet_groups_pub(&self) -> Option<Vec<(Mono, Konst)>> {
        if !self.t.sz.any_indet {
            return Some(vec![(Vec::new(), self.clone())]);
        }
        self.indet_groups()
    }
    /// group the terms by their indeterminate monomial; None when a non-indeterminate kernel
    /// depends on an indeterminate
    fn indet_groups(&self) -> Option<Vec<(Mono, Konst)>> {
        // revision 4 (review 4, B1): the grouping copies every term
        // revision 5 (review 5, B2): two searches per term (the group, the term in its group)
        // revision 6: one vector of (indeterminate part, rest, coefficient), sorted in place by
        // (indeterminate part, rest), then cut into groups; charged two copies (the triples and the
        // groups' constants) and the sort's comparisons. Revision 5 kept a BTreeMap of vectors, a
        // vector of 4 slots per group (about 600 bytes per one-term group for about 60 units).
        if self.t.sz.nonindet_dep {
            return None;
        }
        let tt = self.t.len() as u64;
        if !charge(
            2u64.saturating_mul(copy_units(self))
                .saturating_add(2u64.saturating_mul(search_units(self.t.sz.excess, tt))),
        ) {
            return None;
        }
        let mut v: Vec<(Mono, Term)> = Vec::with_capacity(self.t.len());
        for (m, q) in self.t.iter() {
            let ni = m.iter().filter(|(k, _)| k.0.class == KClass::Indet).count();
            let mut ind: Mono = Vec::with_capacity(ni);
            let mut rest: Mono = Vec::with_capacity(m.len() - ni);
            for (k, e) in m {
                if k.0.class == KClass::Indet {
                    ind.push((k.clone(), *e));
                } else {
                    if k.0.has_indet {
                        return None;
                    }
                    rest.push((k.clone(), *e));
                }
            }
            v.push((ind, (rest, q.clone())));
        }
        v.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1 .0.cmp(&b.1 .0)));
        let mut groups: Vec<(Mono, Konst)> = Vec::new();
        let mut it = v.into_iter().peekable();
        while let Some((ind, t0)) = it.next() {
            let mut g: Vec<Term> = vec![t0];
            while matches!(it.peek(), Some((i2, _)) if *i2 == ind) {
                g.push(it.next().unwrap().1);
            }
            groups.push((ind, Konst::from_sorted(g)));
        }
        Some(groups)
    }

    /// zero test for a constant free of indeterminates
    fn const_status(&self) -> ZS {
        if self.t.is_empty() {
            return ZS::Zero;
        }
        if self.t.len() == 1 {
            if !self.t.sz.any_opaque {
                return ZS::NonZero;
            }
            return self.numeric();
        }
        // A + B*I with A, B Laurent polynomials over Q in a single named constant (pi or e):
        // pi and e are transcendental and real, so this vanishes iff A = B = 0 as polynomials.
        // revision 6: the class is a cached flag (O(1))
        if self.t.sz.one_named_imag {
            ZS::NonZero
        } else {
            self.numeric()
        }
    }

    /// rigorous numeric fallback: c = A + B*I with A, B free of I; NonZero iff A and B both
    /// have real ball enclosures and one of them excludes 0
    fn numeric(&self) -> ZS {
        let (a, b) = self.split_imag();
        if crate::series_ball::proves_nonzero_ab(&a, &b) {
            ZS::NonZero
        } else {
            ZS::Unknown
        }
    }

    /// (A, B) with self = A + B*I, A and B free of the kernel I
    pub fn split_imag(&self) -> (Konst, Konst) {
        // revision 6: without the kernel I, A is the constant itself (no copy)
        if !self.t.sz.any_imag {
            return (self.clone(), Konst::zero());
        }
        // revision 4 (review 4, B1): the split copies every term
        // revision 5 (review 5, B2): plus one search per term
        if !charge(
            copy_units(self).saturating_add(search_units(self.t.sz.excess, self.t.len() as u64)),
        ) {
            return (Konst::zero(), Konst::zero());
        }
        // revision 6: A keeps the monomial order; B is sorted after I is removed from each
        // monomial (t log t comparisons, the search_units above); both vectors get their exact
        // capacity from a counting pass
        let nb = self
            .t
            .keys()
            .filter(|m| m.iter().any(|(k, _)| k.0.class == KClass::Imag))
            .count();
        let mut a: Vec<Term> = Vec::with_capacity(self.t.len() - nb);
        let mut b: Vec<Term> = Vec::with_capacity(nb);
        for (m, q) in self.t.iter() {
            if let Some(pos) = m.iter().position(|(k, _)| k.0.class == KClass::Imag) {
                let mut m2 = m.clone();
                m2.remove(pos);
                b.push((m2, q.clone()));
            } else {
                a.push((m.clone(), q.clone()));
            }
        }
        b.sort_unstable_by(|x, y| x.0.cmp(&y.0));
        (Konst::from_sorted(a), Konst::from_sorted(b))
    }

    /// Inverse. `strict`: the constant must be decided nonzero (a derived leading
    /// coefficient). Otherwise (a constant the input itself divides by) an Unknown
    /// constant is inverted as a kernel 1/P, under the convention that the input is defined.
    pub fn inverse(&self, strict: bool) -> Result<Konst, SErr> {
        self.inverse_src(strict, None)
    }
    /// revision 6 (review 6, m1): `src` is the input node whose value is self (a divisor the input
    /// names); the kernel 1/P0 then takes `src / c` as its expression (P = c P0), so its depth is
    /// the input's, not that of P0's normal form
    // revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
    #[inline(never)]
    pub fn inverse_src(&self, strict: bool, src: Option<&Expr>) -> Result<Konst, SErr> {
        check()?;
        let st = self.zero_status();
        if st == ZS::Zero {
            return Err(SErr::DivisionByZero("division by the constant 0".into()));
        }
        if strict && st == ZS::Unknown {
            return Err(undecided(self));
        }
        if self.t.len() == 1 {
            let (m, q) = self.t.iter().next().unwrap();
            // revision 5: the cached flag (the same test as mono_invertible, O(1))
            if !self.t.sz.any_opaque {
                // revision 5 (review 5, B2): the inverted monomial is a new vector
                if !charge(
                    rat_words(q)
                        .saturating_add(MONO_FACTOR_UNITS.saturating_mul(m.len() as u64 + 1)),
                ) {
                    check()?;
                }
                let qinv = q
                    .reciprocal()
                    .map_err(|_| SErr::DivisionByZero("0".into()))?;
                let mut acc = Konst::rat(qinv);
                let mut plain: Mono = Vec::with_capacity(m.len());
                for (k, e) in m {
                    if k.0.class == KClass::Inverse {
                        // (1/P)^e inverted is P^e; revision 4 (review 4, m1): under the caps of
                        // an exact power, else the kernel P^e (NonZero: P is nonzero under the
                        // convention that made the kernel 1/P)
                        let p = k.0.inv_of.as_ref().unwrap();
                        let pe = match exact_pow_if_small(p, *e as u64) {
                            Some(v) => v,
                            None => Konst::kernel(mk_kernel(
                                "^",
                                KClass::NonZero,
                                &[p, &Konst::int(*e)],
                                || {
                                    Expr::Binary(
                                        BinaryOp::Pow,
                                        p.to_expr_arc_raw(),
                                        Arc::new(Expr::from(*e)),
                                    )
                                },
                            )),
                        };
                        acc = acc.mul(&pe);
                    } else {
                        plain.push((
                            k.clone(),
                            e.checked_neg().ok_or(SErr::Unsupported(
                                "a kernel exponent overflows i64".into(),
                            ))?,
                        ));
                    }
                }
                let (pm, neg, sq) = Konst::reduce_mono(plain);
                let mut c1 = if neg { qi(-1) } else { qi(1) };
                if let Some(sq) = sq {
                    c1 = q_mul(&c1, &sq);
                }
                let r = acc.mul(&Konst::from_sorted(vec![(pm, c1)]));
                check()?;
                return Ok(r);
            }
        }
        // multi-term (or an opaque monomial): a kernel 1/P0 with P = c * P0, where P0 has
        // coprime integer coefficients and a positive leading coefficient
        let mut num_g = Integer::from(0);
        let mut den_l = Integer::from(1);
        for q in self.t.values() {
            num_g = i_gcd(&num_g, q.numerator());
            den_l = i_lcm(&den_l, q.denominator());
        }
        check()?;
        let (_, lead) = self.t.iter().next().unwrap();
        let mut c = q_new(num_g, den_l);
        check()?;
        if lead.numerator().signum() < 0 {
            c = -c;
        }
        let p0 = self.scale(&c.reciprocal().unwrap());
        let src_d = src.and_then(|r| crate::series_engine::input_depth(r).map(|d| (r, d)));
        let k = match src_d {
            Some((r, d)) => {
                let one = c.is_one();
                let cc = c.clone();
                let d = d.saturating_add(if one { 0 } else { 1 });
                let units = 16
                    + if one {
                        0
                    } else {
                        2 * EXPR_NODE + expr_units_rat(&c)
                    };
                let tr = crate::series_engine::input_tree(r)
                    .unwrap_or(u64::MAX)
                    .saturating_add(if one { 0 } else { 2 });
                mk_kernel_inner(
                    "inv",
                    KClass::Inverse,
                    &[&p0],
                    move || {
                        if one {
                            r.clone()
                        } else {
                            Expr::Binary(
                                BinaryOp::Div,
                                Arc::new(r.clone()),
                                Arc::new(rat_expr(&cc)),
                            )
                        }
                    },
                    Some(p0.clone()),
                    None,
                    Some(d),
                    Some(units),
                    Some(tr),
                    None,
                )
            }
            None => {
                // revision 7rt3: P0 = P / c renders as the hint of P divided by c when P has one
                if let (Some(h), false) = (self.hint(), c.is_one()) {
                    let e = Arc::new(Expr::Binary(BinaryOp::Div, h.0, Arc::new(rat_expr(&c))));
                    if charge(2 * EXPR_NODE + expr_units_rat(&c)) {
                        p0.set_hint(Hint {
                            e,
                            tree: h.1.saturating_add(2),
                            depth: h.2.saturating_add(1),
                        });
                    }
                }
                mk_kernel_full(
                    "inv",
                    KClass::Inverse,
                    &[&p0],
                    || p0.to_expr_raw(),
                    Some(p0.clone()),
                    None,
                )
            }
        };
        let r = Konst::kernel(k).scale(&c.reciprocal().unwrap());
        check()?;
        Ok(r)
    }

    /// c^n for n >= 0 by binary powering; callers bound the result size first
    pub fn pow_u(&self, n: u64) -> Konst {
        if n == 0 {
            return Konst::one();
        }
        if self.t.len() == 1 {
            // one monomial: multiply the exponents (checked), power the coefficient
            let (m, q) = self.t.iter().next().unwrap();
            // revision 5 (review 5, B2): the powered monomial is a new vector
            if !charge(MONO_FACTOR_UNITS.saturating_mul(m.len() as u64 + 1)) {
                return Konst::zero();
            }
            let mut mm: Mono = Vec::with_capacity(m.len());
            for (k, e) in m {
                match e.checked_mul(n as i64) {
                    Some(v) if n <= i64::MAX as u64 => mm.push((k.clone(), v)),
                    _ => {
                        set_fault(SErr::Unsupported("a kernel exponent overflows i64".into()));
                        return Konst::zero();
                    }
                }
            }
            let qn = if q.is_one() {
                qi(1)
            } else if q.numerator().signum() < 0
                && q.numerator().bit_length() == 1
                && q.denominator().is_one()
            {
                // -1, by its sign and magnitude (no copy of the coefficient, design §6 "port traps")
                if n.is_multiple_of(2) {
                    qi(1)
                } else {
                    qi(-1)
                }
            } else {
                // revision 3 (review 3, m2): superlinear power and normalization, by q_pow
                pow_rat_i(q, n as i64)
            };
            let (pm, neg, sq) = Konst::reduce_mono(mm);
            let mut c = if neg { -qn } else { qn };
            if let Some(sq) = sq {
                c = q_mul(&c, &sq);
            }
            return Konst::from_sorted(vec![(pm, c)]);
        }
        let mut base = self.clone();
        let mut acc = Konst::one();
        let mut n = n;
        while n > 0 {
            if faulted() {
                return Konst::zero();
            }
            if n & 1 == 1 {
                acc = acc.mul(&base);
            }
            n >>= 1;
            if n > 0 {
                base = base.mul(&base);
            }
        }
        acc
    }

    /// a-priori bit size of the coefficients of self^n (an upper bound), and the term-count
    /// bound C(n + t - 1, t - 1) of the multinomial expansion
    fn power_size(&self, n: u64) -> (u64, u128) {
        let t = self.t.len() as u128;
        // revision 5: the largest term size is cached (a coefficient +-1 does not grow)
        let b = self.t.sz.max_term_bits;
        let logt = if t > 1 {
            64 - (t as u64).leading_zeros() as u64 + 1
        } else {
            0
        };
        let bits = n.saturating_mul(b.saturating_add(logt));
        // C(n + t - 1, t - 1), saturating above the cap
        let mut c: u128 = 1;
        for i in 1..t {
            c = c.saturating_mul(n as u128 + i) / i;
            if c > MAX_EXACT_POWER_TERMS {
                return (bits, c);
            }
        }
        (bits, c)
    }

    /// revision 4 (review 4, B1): charged by expr_units before it builds anything
    pub fn to_expr(&self) -> Expr {
        if !charge(expr_units(self)) {
            return Expr::from(0);
        }
        self.to_expr_raw()
    }
    /// to_expr without the charge (kernel creation charges expr_units of its arguments itself)
    pub fn to_expr_raw(&self) -> Expr {
        let a = self.to_expr_arc_raw();
        Arc::try_unwrap(a).unwrap_or_else(|a| (*a).clone())
    }
    /// revision 6: the rendering as a shared node. A kernel factor with exponent 1 is the kernel's
    /// own Arc (an increment, no allocation); every node it allocates is counted by expr_units.
    pub fn to_expr_arc_raw(&self) -> Arc<Expr> {
        if self.t.is_empty() {
            return Arc::new(Expr::from(0));
        }
        // revision 7rt3: a hint is a shared node (an Arc increment)
        if let Some(h) = self.hint() {
            return h.0;
        }
        let terms: Vec<Arc<Expr>> = self.t.iter().map(|(m, q)| mono_expr(q, m)).collect();
        balanced_arc(terms, BinaryOp::Add).unwrap()
    }
}

/// revision 6: a balanced tree of Arcs (the shape rule of balanced_sum: at most 4 leaves stay
/// left-deep, longer runs split in halves); recursion ceil(log2 n) levels
pub fn balanced_arc(mut v: Vec<Arc<Expr>>, op: BinaryOp) -> Option<Arc<Expr>> {
    fn build(t: &mut [Option<Arc<Expr>>], op: BinaryOp) -> Arc<Expr> {
        let n = t.len();
        if n == 1 {
            return t[0].take().unwrap();
        }
        if n <= 4 {
            let mut acc = t[0].take().unwrap();
            for x in t[1..].iter_mut() {
                acc = Arc::new(Expr::Binary(op, acc, x.take().unwrap()));
            }
            return acc;
        }
        let mid = n.div_ceil(2);
        let (l, r) = t.split_at_mut(mid);
        let a = build(l, op);
        let b = build(r, op);
        Arc::new(Expr::Binary(op, a, b))
    }
    if v.is_empty() {
        return None;
    }
    let mut w: Vec<Option<Arc<Expr>>> = v.drain(..).map(Some).collect();
    Some(build(&mut w, op))
}

fn kpow(e: &Arc<Expr>, n: i64) -> Arc<Expr> {
    if n == 1 {
        e.clone()
    } else {
        Arc::new(Expr::Binary(
            BinaryOp::Pow,
            e.clone(),
            Arc::new(Expr::from(n)),
        ))
    }
}

/// units of an Expr node holding the rational q (its node and digit vectors)
pub fn expr_units_rat(q: &Rational) -> u64 {
    EXPR_NODE + EXPR_WORD * (iw(q.numerator()) + iw(q.denominator())) + 2 * EXPR_INT
}

fn mono_expr(q: &Rational, m: &Mono) -> Arc<Expr> {
    let mut num: Vec<Arc<Expr>> = Vec::new();
    let mut den: Vec<Arc<Expr>> = Vec::new();
    for (k, e) in m {
        if k.0.class == KClass::Inverse {
            den.push(kpow(&k.0.expr, *e));
        } else if *e > 0 {
            num.push(kpow(&k.0.expr, *e));
        } else {
            den.push(kpow(&k.0.expr, -*e));
        }
    }
    let numer = match balanced_arc(num, BinaryOp::Mul) {
        None => Arc::new(rat_expr(q)),
        Some(p) => {
            if q.is_one() {
                p
            } else if q.numerator().signum() < 0
                && q.numerator().bit_length() == 1
                && q.denominator().is_one()
            {
                Arc::new(Expr::Unary(UnaryOp::Neg, p))
            } else {
                Arc::new(Expr::Binary(BinaryOp::Mul, Arc::new(rat_expr(q)), p))
            }
        }
    };
    match balanced_arc(den, BinaryOp::Mul) {
        None => numer,
        Some(d) => Arc::new(Expr::Binary(BinaryOp::Div, numer, d)),
    }
}

// ---------------------------------------------------------------------------
// building constants from expressions free of the variable

fn named_class(s: &Symbol) -> Result<KClass, SErr> {
    match s.name() {
        "I" => Ok(KClass::Imag),
        "pi" | "e" => Ok(KClass::Named),
        "oo" => Err(SErr::BadPoint("infinity (oo) inside the expression".into())),
        _ => Ok(KClass::Indet),
    }
}

/// Revision 4 (review 4, B2): a kernel's key is (kind, class, the arguments' normal forms), with
/// every sub-kernel written as its interning id. It is linear in the normal forms, whatever the
/// size of the kernels' expression trees (they share subtrees, so a tree can be exponentially
/// larger than its DAG). Kernels are interned per expansion, so ids are stable and the key is
/// injective. Upper bound of the bytes `write_nf` writes; the walk stops once it passes `limit`.
fn nf_key_size(c: &Konst, _limit: u64) -> u64 {
    // revision 5: from the cached sizes, O(1): per term 10 bytes of delimiters and two signs, 36
    // per factor (an id and an exponent in hexadecimal and decimal), the hexadecimal digits
    let z = &c.t.sz;
    2u64.saturating_add(11u64.saturating_mul(c.t.len() as u64))
        .saturating_add(36u64.saturating_mul(z.factors))
        .saturating_add(z.bits / 4 + 2 * c.t.len() as u64)
}
fn hex_int(n: &Integer, out: &mut String) {
    use std::fmt::Write;
    if n.signum() < 0 {
        out.push('-');
    }
    let _ = write!(out, "{:x}", n.as_bigint().magnitude());
}
/// prefix code of a normal form: '{' then per term '[' ('k' id '^' exponent ';')* '#' num '/' den ']' then '}'
fn write_nf(c: &Konst, out: &mut String) {
    use std::fmt::Write;
    out.push('{');
    for (m, q) in c.t.iter() {
        out.push('[');
        for (k, e) in m {
            let _ = write!(out, "k{:x}^{};", k.0.id, e);
        }
        out.push('#');
        hex_int(q.numerator(), out);
        out.push('/');
        hex_int(q.denominator(), out);
        out.push(']');
    }
    out.push('}');
}
pub fn mk_kernel(
    kind: &str,
    class: KClass,
    args: &[&Konst],
    build: impl FnOnce() -> Expr,
) -> Kernel {
    mk_kernel_full(kind, class, args, build, None, None)
}
/// revision 5: a kernel made while converting an input node takes that node as its expression
/// (an O(1) clone of the node: its children are shared Arcs), so its text is the input's text and
/// its depth is the node's depth (at most the input depth); `unary` records (f, c) for f(c).
pub fn mk_kernel_src(
    kind: &str,
    class: KClass,
    args: &[&Konst],
    src: Option<&Expr>,
    unary: Option<(UnaryOp, Konst)>,
    build: impl FnOnce() -> Expr,
) -> Kernel {
    match src {
        Some(e) => {
            let d = crate::series_engine::input_depth(e);
            // revision 7: the clone of an input node is shallow, but it copies the node's own
            // payload (a symbol's or a function's name, a function's argument vector), charged
            // COPY_WORD per 8 bytes (a 100 KB name was 16 units)
            let tr = crate::series_engine::input_tree(e);
            mk_kernel_inner(
                kind,
                class,
                args,
                || e.clone(),
                None,
                None,
                d,
                Some(16 + node_payload_units(e)),
                tr,
                unary,
            )
        }
        None => mk_kernel_inner(
            kind, class, args, build, None, None, None, None, None, unary,
        ),
    }
}
/// revision 7: the units of the payload that a shallow clone of `e` copies (its children are
/// shared Arcs): a name's bytes and 8 bytes per argument pointer, COPY_WORD per 8 bytes
pub fn node_payload_units(e: &Expr) -> u64 {
    let bytes: u64 = match e {
        Expr::Symbol(s) => s.name().len() as u64,
        Expr::Function(name, args) => name.len() as u64 + 8 * args.len() as u64,
        Expr::Integer(n) => 8 * iw(n),
        Expr::Rational(q) => 8 * (iw(q.numerator()) + iw(q.denominator())),
        _ => 0,
    };
    COPY_WORD.saturating_mul(bytes.div_ceil(8))
}
/// revision 7: the kernel of an input symbol; the clone copies the name, charged by
/// node_payload_units (a separate function, so that konst_of's recursive frame does not grow)
#[inline(never)]
fn symbol_kernel(e: &Expr, s: &Symbol) -> Result<Konst, SErr> {
    let cls = named_class(s)?;
    Ok(Konst::kernel(mk_kernel_inner(
        &format!("${}:{}", s.name().len(), s.name()),
        cls,
        &[],
        || e.clone(),
        None,
        None,
        None,
        Some(16 + node_payload_units(e)),
        Some(1),
        None,
    )))
}
fn fault_kernel() -> Kernel {
    Kernel(Arc::new(KData {
        id: u64::MAX,
        expr: Arc::new(Expr::from(0)),
        class: KClass::Opaque,
        has_indet: false,
        inv_of: None,
        root: None,
        depth: 1,
        unary: None,
        tree: 1,
    }))
}
pub fn mk_kernel_full(
    kind: &str,
    class: KClass,
    args: &[&Konst],
    build: impl FnOnce() -> Expr,
    inv_of: Option<Konst>,
    root: Option<(Integer, u32)>,
) -> Kernel {
    mk_kernel_inner(
        kind, class, args, build, inv_of, root, None, None, None, None,
    )
}
#[allow(clippy::too_many_arguments)]
fn mk_kernel_inner(
    kind: &str,
    class: KClass,
    args: &[&Konst],
    build: impl FnOnce() -> Expr,
    inv_of: Option<Konst>,
    root: Option<(Integer, u32)>,
    src_depth: Option<u32>,
    src_units: Option<u64>,
    src_tree: Option<u64>,
    unary: Option<(UnaryOp, Konst)>,
) -> Kernel {
    use std::fmt::Write;
    // revision 5: the depth of the kernel's expression is capped, checked first (O(1)): the
    // input node's depth for a kernel of an input node, else a bound from the arguments
    // revision 7rt3: the arguments' effective depth (a hint's when they have one)
    let depth = src_depth.unwrap_or_else(|| {
        1u32.saturating_add(args.iter().map(|a| a.depth_eff()).max().unwrap_or(0))
    });
    // revision 7rt3 (red team 3 structure, H2): the tree of the expression, O(1) from the cached
    // sizes: the input node's tree, or the nodes the expression builds above its arguments'
    // renderings. Revision 7rt4 (red team 4 structure, m1): counted per kind (revision 7rt3 counted
    // 3 for every kind, so a chain of k computed kernels was counted about 3k for a tree of k, and
    // the tree cap refused results under it): 1 for an operator f(c) or a power c^alpha (its
    // exponent is an argument), 2 for a radical n^(1/d) (the Pow node and the exponent 1/d, which
    // is not an argument), 0 for an inverse (its expression is P0's rendering; the quotient node
    // is counted by the monomial), 1 for a float leaf (no arguments)
    let base: u64 = if kind.starts_with("root") {
        2
    } else if kind == "inv" {
        0
    } else {
        1
    };
    let tree = src_tree.unwrap_or_else(|| {
        args.iter()
            .fold(base, |acc, a| acc.saturating_add(a.tree()))
    });
    let allowed = KDEPTH_ALLOWED.with(|c| c.get());
    if depth > allowed {
        limit("depth of a kernel's expression", allowed as u64);
        return fault_kernel();
    }
    // the key is charged one unit per byte before it is written; the size walk stops at the budget
    let lim = remaining();
    let mut size = kind.len() as u64 + 12;
    for a in args {
        size = size.saturating_add(nf_key_size(a, lim));
        if size > lim {
            break;
        }
    }
    if !charge(size) {
        return fault_kernel();
    }
    // revision 6: the key's capacity is its charged size bound (no doubling slack)
    let mut key = String::with_capacity(size.min(lim) as usize);
    let _ = write!(key, "{}:{:?}:", kind, class);
    for a in args {
        write_nf(a, &mut key);
    }
    // has_indet from the normal forms (an OR over the kernels' flags), never by a tree walk
    let has_indet = class == KClass::Indet || args.iter().any(|a| a.t.sz.any_indet);
    intern(key, |id| {
        // a new kernel: its expression (sharing the sub-kernels' expressions) is charged first;
        // an input node is shared, not built (revision 5)
        let eu: u64 = match src_units {
            Some(u) => u,
            None => 16 + args.iter().map(|a| expr_units(a)).sum::<u64>(),
        };
        let expr = Arc::new(if charge(eu) { build() } else { Expr::from(0) });
        KData {
            id,
            expr,
            class,
            has_indet,
            inv_of,
            root,
            depth,
            unary,
            tree,
        }
    })
}

fn pure_indet_nonconstant(c: &Konst) -> bool {
    c.as_rational().is_none() && c.t.sz.only_indet
}

/// exact rational root: q^(1/n) if q >= 0 is a perfect n-th power
fn rat_root(q: &Rational, n: u32) -> Option<Rational> {
    if q.numerator().signum() < 0 {
        return None;
    }
    // revision 3 (review 3, B2): Integer::exact_root (I3), charged by i_exact_root; the parts
    // of a reduced fraction have coprime roots, so no normalization is needed beyond q_new
    let a = i_exact_root(q.numerator(), n)?;
    let b = i_exact_root(q.denominator(), n)?;
    Some(q_new(a, b))
}

/// the radical kernel n^(1/d) (n > 1 an integer that is not a perfect d-th power)
fn radical_kernel(n: Integer, d: u32) -> Konst {
    let nk = Konst::rat(Rational::from_integer(n.clone()));
    let kind = format!("root{}", d);
    let k = mk_kernel_full(
        &kind,
        KClass::NonZero,
        &[&nk],
        || {
            Expr::Integer(n.clone()).pow(Expr::Rational(
                Rational::new(Integer::from(1), Integer::from(d as i64)).unwrap(),
            ))
        },
        None,
        Some((n.clone(), d)),
    );
    Konst::kernel(k)
}

/// n = m^d * r with the d-th powers of the primes below 2^16 moved into m (m10: so sqrt(8) is
/// 2*sqrt(2)); a cofactor that is a perfect d-th power is moved too
fn dth_power_part(mut n: Integer, d: u32) -> (Integer, Integer) {
    let mut m = Integer::from(1);
    if n.bit_length() <= 4096 {
        // revision 3: (words of n) times the number of candidate primes p < 2^16 with p^d <= n;
        // revision 4 (review 4, m2): plus a base of 8 per candidate (p^d, a clone, a remainder)
        let cand = if n.bit_length() / d as u64 >= 16 {
            6542
        } else {
            (1u64 << (n.bit_length() / d as u64 + 1)).min(6542)
        };
        let _ = charge((iw(&n) + 8) * cand);
        for &p in small_primes() {
            {
                let pd = Integer::from(p as i64).pow(d);
                if pd.bit_length() > n.bit_length() {
                    break;
                }
                while (n.clone() % pd.clone()).is_zero() {
                    n = n / pd.clone();
                    m = m * Integer::from(p as i64);
                }
            }
        }
    }
    // revision 3 (review 3, B2): the perfect-power test of the cofactor, charged
    if let Some(r) = i_exact_root(&n, d) {
        return (i_mul(&m, &r), Integer::from(1));
    }
    (m, n)
}
/// revision 4 (review 4, m2): the 6542 primes below 2^16, sieved once (the port uses
/// rustmath_integers::prime_range)
fn small_primes() -> &'static [u64] {
    static P: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        let n = 65536usize;
        let mut c = vec![true; n];
        c[0] = false;
        c[1] = false;
        let mut i = 2;
        while i * i < n {
            if c[i] {
                let mut j = i * i;
                while j < n {
                    c[j] = false;
                    j += i;
                }
            }
            i += 1;
        }
        (0..n).filter(|&i| c[i]).map(|i| i as u64).collect()
    })
}
#[allow(dead_code)]
fn is_small_prime(p: u64) -> bool {
    if p < 2 {
        return false;
    }
    let mut i = 2;
    while i * i <= p {
        if p.is_multiple_of(i) {
            return false;
        }
        i += 1;
    }
    true
}

/// c^alpha for constants c, alpha
pub fn kpow_const(c: &Konst, alpha: &Konst) -> Result<Konst, SErr> {
    kpow_const_src(c, alpha, None)
}
/// revision 5: `src` is the input node c^alpha came from (its expression when kept as a kernel)
// revision 6: never inlined (no frame on the recursive path of from_expr holds its locals)
#[inline(never)]
pub fn kpow_const_src(c: &Konst, alpha: &Konst, src: Option<&Expr>) -> Result<Konst, SErr> {
    check()?;
    if let Some(a) = alpha.as_rational() {
        if a.denominator().is_one() {
            let n = a.numerator();
            if c.is_zero() {
                if n.signum() > 0 {
                    return Ok(Konst::zero());
                }
                if n.is_zero() {
                    return Ok(Konst::one());
                }
                return Err(SErr::DivisionByZero("0 to a negative power".into()));
            }
            if let Some(nn) = NumericConversion::to_i64(n) {
                let un = nn.unsigned_abs();
                // the base whose |n|-th power is taken: a single monomial (exponents multiply,
                // checked) or a multi-term constant (multinomial expansion, size-capped)
                // revision 6 (review 6, m1): the input node's base is the divisor's expression
                let base_src = match src {
                    Some(Expr::Binary(BinaryOp::Pow, l, _)) => Some(&**l),
                    _ => None,
                };
                let b = if nn < 0 {
                    c.inverse_src(false, base_src)?
                } else {
                    c.clone()
                };
                if let Some(r) = exact_pow_if_small(&b, un) {
                    check()?;
                    return Ok(r);
                }
            }
            // revision 7rt2 (red team 2 structure, m3): 1^n = 1 and (-1)^n = +-1 for any n
            // (revision 7rt1 kept the kernel `1 ^ n` when n did not fit in i64)
            if let Some(q) = c.as_rational() {
                if q.denominator().is_one() && q.numerator().bit_length() == 1 {
                    let odd = (n.clone() % Integer::from(2)) != Integer::from(0);
                    return Ok(if q.numerator().signum() < 0 && odd {
                        Konst::int(-1)
                    } else {
                        Konst::one()
                    });
                }
            }
            // too large: keep c^n as a kernel
            let st = c.zero_status();
            let cls = if st == ZS::NonZero {
                KClass::NonZero
            } else {
                KClass::Opaque
            };
            let ak = Konst::rat(a.clone());
            return Ok(Konst::kernel(mk_kernel_src(
                "^",
                cls,
                &[c, &ak],
                src,
                None,
                || Expr::Binary(BinaryOp::Pow, c.to_expr_arc_raw(), Arc::new(rat_expr(&a))),
            )));
        }
        // rational non-integer exponent p/q
        if let Some(b) = c.as_rational() {
            if b.is_zero() {
                if a.numerator().signum() > 0 {
                    return Ok(Konst::zero());
                }
                return Err(SErr::DivisionByZero("0 to a negative power".into()));
            }
            let qd = NumericConversion::to_i64(a.denominator());
            // revision 3 (review 3, B2): one charged perfect-power test per radical, in
            // dth_power_part (the separate rat_root attempt repeated it)
            if b.numerator().signum() > 0 {
                if let Some(d) = qd {
                    if (2..=64).contains(&d) {
                        // b^(p/d) = (m * r^(1/d))^p / den^p with num * den^(d-1) = m^d r
                        let d = d as u32;
                        let den = b.denominator().clone();
                        let n = i_mul(b.numerator(), &i_pow(&den, d - 1));
                        let (m, r) = dth_power_part(n, d);
                        check()?;
                        let base = q_new(m, den);
                        check()?;
                        if r == Integer::from(1) {
                            // b is a perfect d-th power: b^(p/d) = base^p (size-capped)
                            return kpow_const(
                                &Konst::rat(base),
                                &Konst::rat(Rational::from_integer(a.numerator().clone())),
                            );
                        }
                        let mut k = Konst::rat(base);
                        if r != Integer::from(1) {
                            k = k.mul(&radical_kernel(r, d));
                        }
                        let p = NumericConversion::to_i64(a.numerator()).unwrap_or(i64::MAX);
                        if p.unsigned_abs() <= MAX_EXACT_EXPONENT as u64 {
                            let kk = if p < 0 { k.inverse(false)? } else { k };
                            // revision 4 (review 4, m1): the bit cap of an exact power applies
                            // here too; above it the kernel b^(p/d) below
                            if let Some(r) = exact_pow_if_small(&kk, p.unsigned_abs()) {
                                check()?;
                                return Ok(r);
                            }
                        }
                    }
                }
            }
            let (bk, ak) = (Konst::rat(b.clone()), Konst::rat(a.clone()));
            return Ok(Konst::kernel(mk_kernel_src(
                "^",
                KClass::NonZero,
                &[&bk, &ak],
                src,
                None,
                || rat_expr(&b).pow(rat_expr(&a)),
            )));
        }
    }
    if c.as_rational().map(|q| q.is_one()).unwrap_or(false) {
        return Ok(Konst::one());
    }
    let st = c.zero_status();
    if st == ZS::Zero {
        // revision 8 (review 8, m2): 0^alpha by the rule of 0^r (section 2.5) on Re alpha, so the
        // constant gets the answer the series rule gives (`x + 0^pi` is x, as `x + 0^(pi + x)` is);
        // revision 7rt3 returned UndecidedZero(0^alpha) here
        zero_pow_sign(alpha)?;
        return Ok(Konst::zero());
    }
    let cls = if st == ZS::NonZero {
        KClass::NonZero
    } else {
        KClass::Opaque
    };
    Ok(Konst::kernel(mk_kernel_src(
        "^",
        cls,
        &[c, alpha],
        src,
        None,
        || Expr::Binary(BinaryOp::Pow, c.to_expr_arc_raw(), alpha.to_expr_arc_raw()),
    )))
}

/// b^n (n >= 1) when the a-priori size passes the caps of an exact power (section 2.2), else None
fn exact_pow_if_small(b: &Konst, un: u64) -> Option<Konst> {
    let (bits, terms) = b.power_size(un);
    // a single monomial whose exponents would overflow i64 stays a kernel
    let fits =
        un <= i64::MAX as u64 && (b.t.sz.max_abs_exp as i128) * (un as i128) <= i64::MAX as i128;
    let exact = if b.t.len() == 1 {
        bits <= MAX_EXACT_POWER_BITS && fits
    } else {
        un <= MAX_EXACT_EXPONENT as u64
            && terms <= MAX_EXACT_POWER_TERMS
            && bits <= MAX_EXACT_POWER_BITS / 16
    };
    if exact {
        Some(b.pow_u(un))
    } else {
        None
    }
}

/// revision 8 (review 8, m2): whether 0^r is 0 near a point where r(a) = c, decided as section 2.5
/// says (shared by `zero_pow` in ser.rs and by a constant 0^alpha): c = A + B I by `split_imag`, the
/// sign of A by rule 5's enclosures (`real_sign`); A > 0: Ok (the value is 0); A < 0:
/// `DivisionByZero`; A exactly 0 with B decided nonzero: `NotLaurent`; anything undecided:
/// `UndecidedZero` of A or B
pub fn zero_pow_sign(c: &Konst) -> Result<(), SErr> {
    let (a, b) = c.split_imag();
    check()?;
    let sa = crate::series_ball::real_sign(&a);
    check()?;
    match sa {
        Some(1) => Ok(()),
        Some(-1) => Err(SErr::DivisionByZero(
            "0 to a power with negative real part".into(),
        )),
        Some(0) => match b.zero_status() {
            ZS::NonZero => Err(SErr::NotLaurent(
                "0^r with Re r = 0 at the point has no Laurent expansion there".into(),
            )),
            _ => Err(undecided(&b)),
        },
        _ => Err(undecided(&a)),
    }
}

/// m1: is the constant alpha decided to be a non-integer?
/// NonZero: decided non-integer; Zero: an integer; Unknown: undecided (with the nearest integer)
pub fn non_integer_status(alpha: &Konst) -> (ZS, Option<Integer>) {
    if let Some(q) = alpha.as_rational() {
        return if q.denominator().is_one() {
            (ZS::Zero, Some(q.numerator().clone()))
        } else {
            (ZS::NonZero, None)
        };
    }
    // a nonzero indeterminate part: alpha - n is nonzero for every integer n (generic convention)
    if let Some(groups) = alpha.indet_groups_pub() {
        for (ind, g) in groups.iter() {
            if !ind.is_empty() && g.zero_status() == ZS::NonZero {
                return (ZS::NonZero, None);
            }
        }
    }
    let (a, b) = alpha.split_imag();
    crate::series_ball::non_integer(&a, &b)
}

/// f(c) for a constant c
fn as_q_pi(c: &Konst) -> Option<Rational> {
    if c.t.len() != 1 {
        return None;
    }
    let (m, q) = c.t.iter().next().unwrap();
    if m.len() == 1
        && m[0].1 == 1
        && matches!(&*m[0].0 .0.expr, Expr::Symbol(s) if s.name() == "pi")
    {
        return Some(q.clone());
    }
    None
}

fn sin_q_pi(q: &Rational) -> Option<Konst> {
    // exact sin(q pi) for q with denominator 1, 2, 3, 4, 6
    let den = NumericConversion::to_i64(q.denominator())?;
    if ![1, 2, 3, 4, 6].contains(&den) {
        return None;
    }
    // reduce q mod 2 to an integer t in [0, 24) of twelfths. S1 fix-up round 2 (review 2,
    // semantics minor 4): the numerator of 12 q is reduced mod 24 as an Integer (charged, one pass
    // over its words), so sin, cos and tan at q pi stay exact for every |12 q|, also past 2^63
    // (`sin(10^20*pi)` is 0, `tan((2^60+1/2)*pi)` a pole); the i64 conversion of 12 q lost them.
    let twelfths = q_mul(q, &qi(12));
    if !charge(OP_BASE.saturating_add(iw(twelfths.numerator()))) {
        return None;
    }
    let t = NumericConversion::to_i64(&twelfths.numerator().modulo(&Integer::from(24)))?;
    let half = Rational::new(Integer::from(1), Integer::from(2)).unwrap();
    let s2 = kpow_const(&Konst::int(2), &Konst::rat(half.clone()))
        .ok()?
        .scale(&half);
    let s3 = kpow_const(&Konst::int(3), &Konst::rat(half.clone()))
        .ok()?
        .scale(&half);
    let v = match t % 12 {
        0 => Konst::zero(),
        2 | 10 => Konst::rat(half.clone()),
        3 | 9 => s2,
        4 | 8 => s3,
        6 => Konst::one(),
        _ => return None,
    };
    let v = match t % 12 {
        0 | 6 => v,
        _ => v,
    };
    Some(if t >= 12 { v.neg() } else { v })
}

/// m3 (review 2): log of q * e^k * exp(r1)^j1 * ... for a rational q > 0, integers k, j_i and
/// rationals r_i is log(q) + k + j1 r1 + ... (every factor is a positive real, so the principal
/// log is additive). None when c has another form.
fn log_exact(c: &Konst) -> Result<Option<Konst>, SErr> {
    if c.t.len() != 1 {
        return Ok(None);
    }
    let (m, q) = c.t.iter().next().unwrap();
    if m.is_empty() || q.numerator().signum() <= 0 {
        return Ok(None);
    }
    let mut acc = Konst::zero();
    for (k, j) in m {
        // revision 5: exp(r) is recognised by its recorded (f, c), not by its expression (which
        // may be the input's own subtree, such as exp(1+1))
        let rec = k.0.unary.clone();
        let r = match (&*k.0.expr, rec) {
            (Expr::Symbol(s), _) if s.name() == "e" => qi(1),
            (_, Some((UnaryOp::Exp, c))) => match c.as_rational() {
                Some(r) => r,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        acc = acc.add(&Konst::rat(q_mul(&r, &qi(*j))));
    }
    if !q.is_one() {
        acc = acc.add(&kapply(UnaryOp::Log, &Konst::rat(q.clone()))?);
    }
    check()?;
    Ok(Some(acc))
}

/// S1 fix-up round 2 (review 2, semantics minor 2), the inverse of `log_exact`: exp of
/// r + n_1 log(q_1) + ... + n_k log(q_k), for a rational r, nonzero rationals q_i (each log(q_i) a
/// kernel recorded as log of that rational) and integers n_i, is exp(r) q_1^n_1 ... q_k^n_k: exp
/// turns sums into products, and exp(n Log q) = (exp(Log q))^n = q^n for an integer n on the
/// principal branch (for q < 0 too). So `x^x` and `3^x` at 3 have the constant term 27, not the
/// kernel exp(3 log(3)). Folded only when the product has at most `MAX_EXACT_POWER_BITS` bits (an
/// a-priori bound from the sizes, the limit of the exact constant powers); None otherwise, and for
/// every other shape (the kernel exp(c)).
fn exp_of_logs(c: &Konst) -> Result<Option<Konst>, SErr> {
    let mut r: Option<Rational> = None;
    let mut factors: Vec<(Rational, i64)> = Vec::new();
    let mut bits: u64 = 0;
    for (m, q) in c.t.iter() {
        if m.is_empty() {
            r = Some(q.clone());
            continue;
        }
        if m.len() != 1 || m[0].1 != 1 || !q.denominator().is_one() {
            return Ok(None);
        }
        let n = match NumericConversion::to_i64(q.numerator()) {
            Some(n) => n,
            None => return Ok(None),
        };
        let p = match &m[0].0 .0.unary {
            Some((UnaryOp::Log, a)) => match a.as_rational() {
                Some(p) if !p.is_zero() => p,
                _ => return Ok(None),
            },
            _ => return Ok(None),
        };
        bits = bits.saturating_add(n.unsigned_abs().saturating_mul(rat_bits(&p)));
        if bits > MAX_EXACT_POWER_BITS {
            return Ok(None);
        }
        factors.push((p, n));
    }
    if factors.is_empty() {
        return Ok(None);
    }
    let mut acc = qi(1);
    for (p, n) in factors {
        let b = if n < 0 {
            q_new(p.denominator().clone(), p.numerator().clone())
        } else {
            p
        };
        acc = q_mul(&acc, &q_pow(&b, n.unsigned_abs()));
    }
    check()?;
    let v = Konst::rat(acc);
    match r {
        Some(r) if !r.is_zero() => Ok(Some(kapply(UnaryOp::Exp, &Konst::rat(r))?.mul(&v))),
        _ => Ok(Some(v)),
    }
}

pub fn kapply(op: UnaryOp, c: &Konst) -> Result<Konst, SErr> {
    kapply_src(op, c, None)
}
/// S1 fix-up round 5: q when c is exactly q i for a rational q (one term, the kernel I to the first
/// power)
fn as_q_i(c: &Konst) -> Option<Rational> {
    match c.t.as_slice() {
        [(m, q)] if m.len() == 1 && m[0].0 .0.class == KClass::Imag && m[0].1 == 1 => {
            Some(q.clone())
        }
        _ => None,
    }
}
/// S1 fix-up round 5: the rational n/d
fn qnd(n: i64, d: i64) -> Rational {
    Rational::new(Integer::from(n), Integer::from(d)).expect("a nonzero denominator")
}
/// whether c is exactly i or -i: one term, the kernel I to the first power, coefficient +-1
fn is_plus_minus_i(c: &Konst) -> bool {
    match c.t.as_slice() {
        [(m, q)] => {
            m.len() == 1
                && m[0].0 .0.class == KClass::Imag
                && m[0].1 == 1
                && q.denominator().is_one()
                && q.numerator().bit_length() == 1
        }
        _ => false,
    }
}

thread_local! {
    /// S1 fix-up round 5 (review 5 semantics, B1): whether the expansion variable is named `pi`
    /// (bit 1) or `I` (bit 2), the names the engine gives the constants it makes itself. Set by
    /// `series_engine::expansion`, cleared with the session.
    static VAR_SHADOW: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}
/// S1 fix-up round 5: record the expansion variable's name (see `VAR_SHADOW`)
pub fn set_var_shadow(var: &Symbol) {
    let bits = match var.name() {
        "pi" => 1,
        "I" => 2,
        _ => 0,
    };
    VAR_SHADOW.with(|c| c.set(bits));
}
/// S1 fix-up round 5: whether the variable is named `pi`, so the engine must not write the
/// constant pi as the symbol `pi`
pub fn var_is_named_pi() -> bool {
    VAR_SHADOW.with(|c| c.get() & 1 != 0)
}
fn var_is_named_i() -> bool {
    VAR_SHADOW.with(|c| c.get() & 2 != 0)
}

/// S1 fix-up round 4 (review 4 semantics, m1): q pi, or q i pi, as a constant (the kernels pi and
/// I of the input's own symbols, interned under the same keys). S1 fix-up round 5 (review 5
/// semantics, B1): `None` when the variable has the name of one of these symbols (`pi`, or `I`
/// for q i pi): the input's symbol is then the variable, and a kernel named after it would render
/// as the variable (`taylor(acos(0*pi), pi, 0, 2)` was `(1/2)*pi`). The caller keeps the kernel
/// f(c) instead (`arccos(0)`, as HEAD and round 3 did).
fn pi_times(q: Rational, imaginary: bool) -> Result<Option<Konst>, SErr> {
    if var_is_named_pi() || (imaginary && var_is_named_i()) {
        return Ok(None);
    }
    let named = |name: &str| -> Result<Konst, SErr> {
        let s = Symbol::new(name);
        symbol_kernel(&Expr::Symbol(s.clone()), &s)
    };
    let v = named("pi")?.scale(&q);
    if imaginary {
        Ok(Some(v.mul(&named("I")?)))
    } else {
        Ok(Some(v))
    }
}

/// revision 5: `src` is the input node f(c) came from (its expression when f(c) is a kernel)
// revision 6: never inlined (no frame on the recursive path of from_expr holds its locals); the
// S1 port had moved this attribute onto `is_plus_minus_i`; restored in fix-up round 1
#[inline(never)]
pub fn kapply_src(op: UnaryOp, c: &Konst, src: Option<&Expr>) -> Result<Konst, SErr> {
    use UnaryOp::*;
    check()?;
    if op == Neg {
        return Ok(c.neg());
    }
    if op == Sqrt {
        // one kernel for sqrt(c) and c^(1/2)
        return kpow_const_src(
            c,
            &Konst::rat(Rational::new(Integer::from(1), Integer::from(2)).unwrap()),
            src,
        );
    }
    if op == Log {
        if let Some(v) = log_exact(c)? {
            return Ok(v);
        }
    }
    if op == Exp {
        if let Some(v) = exp_of_logs(c)? {
            return Ok(v);
        }
    }
    if let Some(q) = as_q_pi(c) {
        let shifted = q_add(
            &q,
            &Rational::new(Integer::from(1), Integer::from(2)).unwrap(),
        );
        match op {
            Sin => {
                if let Some(v) = sin_q_pi(&q) {
                    return Ok(v);
                }
            }
            Cos => {
                if let Some(v) = sin_q_pi(&shifted) {
                    return Ok(v);
                }
            }
            Tan => {
                if let (Some(s), Some(co)) = (sin_q_pi(&q), sin_q_pi(&shifted)) {
                    if co.is_zero() {
                        return Err(SErr::DivisionByZero(
                            "tan at an odd multiple of pi/2".into(),
                        ));
                    }
                    return Ok(s.mul(&co.inverse(true)?));
                }
            }
            _ => {}
        }
        if matches!(op, Sin | Cos | Tan) {
            // sin(q pi) = 0 iff q is an integer; cos(q pi) = 0 iff q - 1/2 is an integer;
            // both are decided above for denominators 1, 2, so the value is nonzero
            let ok = !q.denominator().is_one() && q.denominator() != &Integer::from(2);
            let cls = if ok { KClass::NonZero } else { KClass::Opaque };
            return Ok(Konst::kernel(mk_kernel_src(
                &format!("{:?}", op),
                cls,
                &[c],
                src,
                Some((op, c.clone())),
                || Expr::Unary(op, c.to_expr_arc_raw()),
            )));
        }
    }
    // S1 port, the owner's first rule with review 9 m2: arctan has poles at +-i, which the kernel I
    // decides exactly, so arctan(i) and arctan(-i) are refused as arctanh(+-1) is below (gp:
    // `atan(I)` is "domain error"); revision 7rt4 made them kernels
    if op == Arctan && is_plus_minus_i(c) {
        return Err(SErr::DivisionByZero("arctan at i or -i".into()));
    }
    // S1 fix-up round 4 (review 4 semantics, m1): arcsinh(+-i) = +-i pi/2 (arcsinh z =
    // log(z + sqrt(1 + z^2)): log(i) = i pi/2, log(-i) = -i pi/2; gp: `asinh(I)` =
    // 1.5707963267948966192*I), so a sum holding it has an enclosure and is decided
    if op == Arcsinh && is_plus_minus_i(c) {
        if let Some(p) = pi_times(qhalf(), false)? {
            return Ok(c.mul(&p));
        }
    }
    // S1 fix-up round 5 (the owner's first rule): arcsinh(+-i/2) = +-i pi/6 (arcsinh(i y) =
    // i arcsin(y) for |y| <= 1) and arctanh(+-i) = +-i pi/4 (arctanh(i y) = i arctan(y)); gp:
    // `asinh(I/2)` = 0.52359877559829887308*I, `atanh(I)` = 0.78539816339744830962*I. As kernels
    // they had no enclosure, so 1/(3 + arcsinh(i/2)) was undecided
    if let Some(q) = as_q_i(c) {
        let half = q.denominator() == &Integer::from(2) && q.numerator().bit_length() == 1;
        let unit = q.denominator().is_one() && q.numerator().bit_length() == 1;
        let scale = match op {
            Arcsinh if half => Some(qnd(1, 3)),
            Arctanh if unit => Some(qquarter()),
            _ => None,
        };
        if let Some(sc) = scale {
            if let Some(p) = pi_times(sc, false)? {
                return Ok(c.mul(&p));
            }
        }
    }
    if let Some(q) = c.as_rational() {
        let z = q.is_zero();
        let one = q.is_one();
        let minus_one = q.numerator().signum() < 0
            && q.numerator().bit_length() == 1
            && q.denominator().is_one();
        let exact = match op {
            Exp if z => Some(Konst::one()),
            Log if one => Some(Konst::zero()),
            Log if z => return Err(SErr::DivisionByZero("log(0)".into())),
            // revision 7rt4 (red team 4 numeric, m1): the poles of section 2.2 at a rational
            // argument; revision 8 made arctanh(1), arctanh(-1) and zeta(1) kernels, so
            // `(1)/(x + arctanh(1))` inverted a pole and the limit path gave x*arctanh(1) -> 0
            Arctanh
                if one
                    || (q.numerator().signum() < 0
                        && q.numerator().bit_length() == 1
                        && q.denominator().is_one()) =>
            {
                return Err(SErr::DivisionByZero("arctanh at 1 or -1".into()))
            }
            Zeta if one => return Err(SErr::DivisionByZero("zeta(1)".into())),
            Sin | Tan | Sinh | Tanh | Arcsin | Arctan | Arcsinh | Arctanh | Erf if z => {
                Some(Konst::zero())
            }
            Cos | Cosh if z => Some(Konst::one()),
            Arccos | Arccosh if one => Some(Konst::zero()),
            // S1 fix-up round 4 (review 4 semantics, m1): the exact values at the other branch
            // points and at 0, Laurent polynomials in pi and I, which the zero test decides (as
            // kernels they had no ball enclosure, so 1/(3 + arcsin(1)) was undecided). Principal
            // values (Abramowitz and Stegun 4.4.1-4.4.3, 4.6.15; arccosh z = log(z + sqrt(z - 1)
            // sqrt(z + 1)), so arccosh(-1) = log(-1) = i pi and arccosh(0) = log(i) = i pi/2);
            // gp: `asin(1)` = 1.5707963267948966192, `acos(-1)` = 3.1415926535897932385,
            // `acos(0)` = 1.5707963267948966192, `acosh(-1)` = 3.1415926535897932385*I,
            // `acosh(0)` = 1.5707963267948966192*I, `atan(1)` = 0.78539816339744830962
            // S1 fix-up round 5: `None` (the kernel f(c)) when the variable is named pi or I
            Arcsin if one => pi_times(qhalf(), false)?,
            Arcsin if minus_one => pi_times(-qhalf(), false)?,
            Arccos if minus_one => pi_times(qi(1), false)?,
            Arccos if z => pi_times(qhalf(), false)?,
            Arccosh if minus_one => pi_times(qi(1), true)?,
            Arccosh if z => pi_times(qhalf(), true)?,
            Arctan if one => pi_times(qquarter(), false)?,
            Arctan if minus_one => pi_times(-qquarter(), false)?,
            // S1 fix-up round 5 (the owner's first rule): the exact values at +-1/2 (Abramowitz and
            // Stegun 4.4: arcsin(1/2) = pi/6, arccos(1/2) = pi/3, arccos(-1/2) = 2 pi/3; arccosh x
            // = i arccos x for -1 <= x <= 1); gp: `asin(1/2)` = 0.52359877559829887308,
            // `acos(1/2)` = 1.0471975511965977462, `acos(-1/2)` = 2.0943951023931954923,
            // `acosh(1/2)` = 1.0471975511965977462*I, `acosh(-1/2)` = 2.0943951023931954923*I
            Arcsin if q == qhalf() => pi_times(qnd(1, 6), false)?,
            Arcsin if q == -qhalf() => pi_times(qnd(-1, 6), false)?,
            Arccos if q == qhalf() => pi_times(qnd(1, 3), false)?,
            Arccos if q == -qhalf() => pi_times(qnd(2, 3), false)?,
            Arccosh if q == qhalf() => pi_times(qnd(1, 3), true)?,
            Arccosh if q == -qhalf() => pi_times(qnd(2, 3), true)?,
            Sqrt => rat_root(&q, 2).map(Konst::rat),
            Abs => Some(Konst::rat(q.abs())),
            Sign => Some(Konst::int(q.numerator().signum() as i64)),
            Gamma | Factorial => {
                let arg = if op == Factorial {
                    q_add(&q, &qi(1))
                } else {
                    q.clone()
                };
                if arg.denominator().is_one() && arg.numerator().signum() <= 0 {
                    return Err(SErr::DivisionByZero("gamma at a pole".into()));
                }
                None
            }
            _ => None,
        };
        if let Some(k) = exact {
            return Ok(k);
        }
        // f(q) for rational q is a nonzero real/complex number for these f (pi is irrational;
        // log q = 0 only at q = 1; cos, cosh never vanish at rationals; Gamma has no zeros)
        let cls = match op {
            Zeta => KClass::Opaque,
            _ => KClass::NonZero,
        };
        let qk = Konst::rat(q.clone());
        return Ok(Konst::kernel(mk_kernel_src(
            &format!("{:?}", op),
            cls,
            &[&qk],
            src,
            Some((op, qk.clone())),
            || Expr::Unary(op, Arc::new(rat_expr(&q))),
        )));
    }
    let cls = match op {
        Exp => KClass::NonZero,
        Sqrt if c.zero_status() == ZS::NonZero => KClass::NonZero,
        _ if pure_indet_nonconstant(c) => KClass::NonZero,
        _ => KClass::Opaque,
    };
    Ok(Konst::kernel(mk_kernel_src(
        &format!("{:?}", op),
        cls,
        &[c],
        src,
        Some((op, c.clone())),
        || Expr::Unary(op, c.to_expr_arc_raw()),
    )))
}

// ------------------------------------------------------------------------------------------
// revision 7rt2 (red team 2 structure, H1): from_expr (`konst_of`) without recursion over the
// input. Revision 7rt1 read the `+ -` and `* /` chains and the towers by loops but recursed once
// per nested operand through a dispatcher (`konst_of`, `konst_of_inner`) and a chain frame, so a
// text that used every grammar level in each parenthesis (a sum, a product, a negation, a tower,
// a negated exponent) cost 15.5 to 18.9 KB of debug stack per parenthesis against the parser's
// 9.7 KB, and overflowed a 2 MB debug thread on text the parser had read there. Now one loop runs
// an explicit work stack: one record per pending node on the current path (a chain, a tower, a
// call or a `%` node holds its items in a boxed record; a unary node is one small record), and
// the value of a finished node is handed to the record on top. The order of the conversions, the
// charges (KONST_NODE + COPY_WORD per literal word per node visit, KONST_NODE per chain item and
// per tower spine node), the checks (one at each visit, one when each value is finished) and the
// source node of every kernel are those of the recursive reading. Native stack: O(1) in the input.
// ------------------------------------------------------------------------------------------

/// the items of a `+ -` (sum) or `* /` (product) chain along the left spine, in input order
fn chain_items(e: &Expr, sum: bool) -> Vec<(bool, &Expr)> {
    let mut items: Vec<(bool, &Expr)> = Vec::new();
    let mut cur = e;
    loop {
        match cur {
            Expr::Binary(op @ (BinaryOp::Add | BinaryOp::Sub), l, r) if sum => {
                items.push((*op == BinaryOp::Sub, r));
                cur = l;
            }
            Expr::Binary(op @ (BinaryOp::Mul | BinaryOp::Div), l, r) if !sum => {
                items.push((*op == BinaryOp::Div, r));
                cur = l;
            }
            _ => break,
        }
    }
    items.push((false, cur));
    items.reverse();
    items
}

struct KSum<'a> {
    items: Vec<(bool, &'a Expr)>,
    next: usize,
    ks: Vec<(bool, Konst)>,
}
struct KProd<'a> {
    items: Vec<(bool, &'a Expr)>,
    next: usize,
    ks: Vec<Konst>,
}
struct KFunc<'a> {
    e: &'a Expr,
    name: &'a str,
    args: &'a [Arc<Expr>],
    ks: Vec<Konst>,
}
/// a right-deep tower b_1^(b_2^(...^t)): (the Pow node, its base, the negations between its
/// exponent and the next Pow node) per level
struct KTower<'a> {
    levels: Vec<(&'a Expr, &'a Expr, u32)>,
    top: &'a Expr,
    bases: Vec<Konst>,
}
struct KMod<'a> {
    e: &'a Expr,
    a: Option<Konst>,
}
/// revision 7rt2: units of one boxed record of the work stack of `from_expr` or `expand` (a chain, a
/// tower, a call or a `%` node: at most 96 bytes in a glibc chunk, plus its 16-byte slot in the
/// stack, at most doubled), charged when the record is pushed; with the node visit's KONST_NODE,
/// 128 bytes for 48 units. A unary record is the 16-byte slot only (the visit's KONST_NODE).
pub const RECORD_UNITS: u64 = 32;
/// one record of the work stack (16 bytes; the chain records are boxed)
enum KFrame<'a> {
    Sum(Box<KSum<'a>>),
    Prod(Box<KProd<'a>>),
    Func(Box<KFunc<'a>>),
    Tower(Box<KTower<'a>>),
    Mod(Box<KMod<'a>>),
    Unary(UnaryOp, &'a Expr),
}
enum KStep<'a> {
    Visit(&'a Expr),
    Done(Konst),
}

/// collect a tower along its right spine (charging KONST_NODE per spine node below e)
fn tower_levels<'a>(e: &'a Expr) -> Result<KTower<'a>, SErr> {
    let mut levels: Vec<(&Expr, &Expr, u32)> = Vec::new();
    let mut cur = e;
    let top: &Expr;
    let mut spine_nodes: u64 = 0;
    loop {
        let (l, r) = match cur {
            Expr::Binary(BinaryOp::Pow, l, r) => (&**l, &**r),
            _ => unreachable!(),
        };
        let mut nx: &Expr = r;
        let mut negs = 0u32;
        while let Expr::Unary(UnaryOp::Neg, i) = nx {
            nx = i;
            negs += 1;
        }
        if matches!(nx, Expr::Binary(BinaryOp::Pow, _, _)) {
            levels.push((cur, l, negs));
            spine_nodes += negs as u64 + 1;
            cur = nx;
        } else {
            levels.push((cur, l, 0));
            top = r;
            break;
        }
    }
    // every spine node below e is a node visit (e itself was charged by its visit); the record
    if !charge(
        KONST_NODE
            .saturating_mul(spine_nodes)
            .saturating_add(RECORD_UNITS),
    ) {
        check()?;
    }
    Ok(KTower {
        levels,
        top,
        bases: Vec::new(),
    })
}

impl<'a> KFrame<'a> {
    /// hand the value of the last child (None when the record was just pushed) to this record;
    /// it names the next child to read, or its own value
    fn resume(&mut self, v: Option<Konst>) -> Result<KStep<'a>, SErr> {
        match self {
            KFrame::Sum(s) => {
                if let Some(v) = v {
                    s.ks.push((s.items[s.next - 1].0, v));
                }
                if s.next < s.items.len() {
                    s.next += 1;
                    return Ok(KStep::Visit(s.items[s.next - 1].1));
                }
                Ok(KStep::Done(Konst::sum_many(&s.ks)))
            }
            KFrame::Prod(s) => {
                if let Some(v) = v {
                    // a divisor is inverted as it is read (so a zero divisor is DivisionByZero
                    // before a later zero factor is seen)
                    let (div, it) = s.items[s.next - 1];
                    s.ks.push(if div {
                        v.inverse_src(false, Some(it))?
                    } else {
                        v
                    });
                }
                if s.next < s.items.len() {
                    s.next += 1;
                    return Ok(KStep::Visit(s.items[s.next - 1].1));
                }
                Ok(KStep::Done(Konst::product_many(std::mem::take(&mut s.ks))))
            }
            KFrame::Func(f) => {
                if let Some(v) = v {
                    f.ks.push(v);
                }
                if f.ks.len() < f.args.len() {
                    return Ok(KStep::Visit(&f.args[f.ks.len()]));
                }
                let refs: Vec<&Konst> = f.ks.iter().collect();
                let kind = format!("@{}:{}", f.name.len(), f.name);
                let (name, ks) = (f.name, &f.ks);
                Ok(KStep::Done(Konst::kernel(mk_kernel_src(
                    &kind,
                    KClass::Opaque,
                    &refs,
                    Some(f.e),
                    None,
                    || {
                        Expr::Function(
                            name.to_string(),
                            ks.iter().map(|k| k.to_expr_arc_raw()).collect(),
                        )
                    },
                ))))
            }
            KFrame::Tower(t) => {
                // the bases left to right, then the top exponent, then the powers folded from the
                // inside out, each with its own input node as the kernel's source
                if let Some(v) = v {
                    if t.bases.len() < t.levels.len() {
                        t.bases.push(v);
                    } else {
                        let mut v = v;
                        for i in (0..t.levels.len()).rev() {
                            v = kpow_const_src(&t.bases[i], &v, Some(t.levels[i].0))?;
                            if i > 0 {
                                for _ in 0..t.levels[i - 1].2 {
                                    v = v.neg();
                                }
                                check()?;
                            }
                        }
                        return Ok(KStep::Done(v));
                    }
                }
                if t.bases.len() < t.levels.len() {
                    return Ok(KStep::Visit(t.levels[t.bases.len()].1));
                }
                Ok(KStep::Visit(t.top))
            }
            KFrame::Mod(m) => match (v, m.a.take()) {
                (None, _) => match m.e {
                    Expr::Binary(_, l, _) => Ok(KStep::Visit(l)),
                    _ => unreachable!(),
                },
                (Some(a), None) => {
                    m.a = Some(a);
                    match m.e {
                        Expr::Binary(_, _, r) => Ok(KStep::Visit(r)),
                        _ => unreachable!(),
                    }
                }
                (Some(b), Some(a)) => Ok(KStep::Done(Konst::kernel(mk_kernel_src(
                    "%",
                    KClass::Opaque,
                    &[&a, &b],
                    Some(m.e),
                    None,
                    || Expr::Binary(BinaryOp::Mod, a.to_expr_arc_raw(), b.to_expr_arc_raw()),
                )))),
            },
            KFrame::Unary(op, e) => match (v, *e) {
                (None, Expr::Unary(_, i)) => Ok(KStep::Visit(i)),
                (Some(c), _) => Ok(KStep::Done(kapply_src(*op, &c, Some(e))?)),
                _ => unreachable!(),
            },
        }
    }
}

/// the visit of one node: a leaf's value, or the record that reads its children
// never inlined: its locals are not on the frame of the loop
#[inline(never)]
fn konst_visit<'a>(e: &'a Expr) -> Result<Result<Konst, KFrame<'a>>, SErr> {
    check()?;
    // revision 5 (review 5, m1): every node visit is charged (a DAG input is converted once per
    // occurrence), a literal by its words (it is copied); revision 6: COPY_WORD per word
    let leaf = match e {
        Expr::Integer(n) => COPY_WORD * iw(n),
        Expr::Rational(q) => COPY_WORD * rw(q),
        _ => 0,
    };
    if !charge(KONST_NODE + leaf) {
        check()?;
    }
    Ok(match e {
        Expr::Integer(n) => Ok(Konst::rat(Rational::from_integer(n.clone()))),
        Expr::Rational(q) => Ok(Konst::rat(q.clone())),
        Expr::Real(f) => {
            if !f.is_finite() {
                return Err(SErr::Unsupported(format!("non-finite float {}", f)));
            }
            if *f == 0.0 {
                return Ok(Ok(Konst::zero()));
            }
            let f = *f;
            Ok(Konst::kernel(mk_kernel(
                &format!("r{:016x}", f.to_bits()),
                KClass::NonZero,
                &[],
                || Expr::Real(f),
            )))
        }
        Expr::Symbol(s) => Ok(symbol_kernel(e, s)?),
        Expr::Binary(BinaryOp::Add | BinaryOp::Sub, _, _) => {
            let items = chain_items(e, true);
            if !charge(
                KONST_NODE
                    .saturating_mul(items.len() as u64)
                    .saturating_add(RECORD_UNITS),
            ) {
                check()?;
            }
            let n = items.len();
            Err(KFrame::Sum(Box::new(KSum {
                items,
                next: 0,
                ks: Vec::with_capacity(n),
            })))
        }
        Expr::Binary(BinaryOp::Mul | BinaryOp::Div, _, _) => {
            let items = chain_items(e, false);
            if !charge(
                KONST_NODE
                    .saturating_mul(items.len() as u64)
                    .saturating_add(RECORD_UNITS),
            ) {
                check()?;
            }
            let n = items.len();
            Err(KFrame::Prod(Box::new(KProd {
                items,
                next: 0,
                ks: Vec::with_capacity(n),
            })))
        }
        Expr::Binary(BinaryOp::Pow, _, _) => Err(KFrame::Tower(Box::new(tower_levels(e)?))),
        Expr::Binary(BinaryOp::Mod, _, _) => {
            if !charge(RECORD_UNITS) {
                check()?;
            }
            Err(KFrame::Mod(Box::new(KMod { e, a: None })))
        }
        Expr::Unary(op, _) => Err(KFrame::Unary(*op, e)),
        Expr::Function(name, args) => {
            if !charge(RECORD_UNITS) {
                check()?;
            }
            Err(KFrame::Func(Box::new(KFunc {
                e,
                name,
                args,
                ks: Vec::with_capacity(args.len()),
            })))
        }
    })
}

/// Constant expression (free of the variable) to a Konst (from_expr). No recursion over e.
pub fn konst_of(e: &Expr) -> Result<Konst, SErr> {
    let mut stack: Vec<KFrame> = Vec::new();
    let mut visit: Option<&Expr> = Some(e);
    loop {
        // the next value: a leaf, or a record that finished
        let mut val: Option<Konst> = None;
        if let Some(x) = visit.take() {
            match konst_visit(x)? {
                Ok(k) => val = Some(k),
                Err(frame) => stack.push(frame),
            }
        }
        if let Some(k) = &val {
            let _ = k;
            check()?;
        }
        // hand values up until a record asks for a child
        loop {
            let top = match stack.last_mut() {
                Some(t) => t,
                None => return Ok(val.expect("a value when the work stack is empty")),
            };
            match top.resume(val.take())? {
                KStep::Visit(c) => {
                    visit = Some(c);
                    break;
                }
                KStep::Done(k) => {
                    stack.pop();
                    check()?;
                    val = Some(k);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1 fix-up round 3 (review 3, B1): a float leaf counts at least the characters `Display`
    /// prints for it (`{}` of an f64 never uses an exponent: 5e-324 prints 326 characters).
    #[test]
    fn node_text_bounds_the_display_of_a_float() {
        for f in [
            5e-324,
            -5e-324,
            1e-308,
            f64::MIN_POSITIVE,
            -1e-300,
            1e300,
            f64::MAX,
            -f64::MAX,
            0.5,
            2.0,
            1e23,
        ] {
            let e = Expr::Real(f);
            let shown = format!("{e}").len() as u64;
            assert!(node_text(&e) >= shown, "{f:e}: {} < {shown}", node_text(&e));
        }
    }

    /// S1 fix-up round 3 (review 3, B2): `tree_text` bounds the leaf text of e's tree and of the
    /// tree of `simplify(e)`, which folds rational subtrees (a power of a 201-bit integer to the
    /// 512th is 1.0e5 bits, about 31 000 digits, from 15 characters of input).
    #[test]
    fn tree_text_bounds_the_text_of_simplify() {
        let leaf_text = |e: &Expr| -> u64 {
            // the characters of leaf text as Display prints them (a tree walk; small inputs)
            let mut t = 0u64;
            let mut st: Vec<&Expr> = vec![e];
            while let Some(n) = st.pop() {
                match n {
                    Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => {
                        t += format!("{n}").len() as u64
                    }
                    Expr::Function(name, args) => {
                        t += name.len() as u64;
                        st.extend(args.iter().map(|a| &**a));
                    }
                    Expr::Binary(_, l, r) => {
                        st.push(l);
                        st.push(r);
                    }
                    Expr::Unary(_, a) => st.push(a),
                }
            }
            t
        };
        for s in [
            "(2^200+1)^512",
            "sin((2^200+1)^512) + x*sin((2^200+1)^512)",
            "(3^100*x)/(7^50)",
            "((5^300*x)/7^40)/11^60",
            "1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31",
            "(1/3 + 1/5)^300 * (2/7 - 1/9)^200",
            "exp(0) + log(1) + x*0 + 0*sin(2^4000) + (2^3000)^0",
            "g(2^1000*3^1000, (10^50+1)^20) - (2^64)^(-64)",
            "-(-(2^4096))",
            "(2^1000 + x)^1 + x^(2^10 - 1023)",
        ] {
            let e = crate::parser::parse(s).expect("parse");
            let mut memo = TextMemo::default();
            let t = tree_text(&e, &mut memo);
            let a = leaf_text(&e);
            let b = leaf_text(&crate::simplify::simplify(&e));
            assert!(
                t >= a && t >= b,
                "{s}: tree_text {t}, leaf text {a}, of simplify {b}"
            );
        }
    }

    /// The printing weight of e's tree as `Display` writes it (a tree walk, for small trees): at
    /// each occurrence of an integer `print_weight` of its text, of a rational that of each part
    /// and 1 for the slash, a float's and a symbol's length, a call's name and parentheses.
    fn printed_weight(e: &Expr) -> u64 {
        let mut t = 0u64;
        let mut st: Vec<&Expr> = vec![e];
        while let Some(n) = st.pop() {
            t += match n {
                Expr::Integer(i) => print_weight(format!("{i}").len() as u64),
                Expr::Rational(q) => {
                    print_weight(format!("{}", q.numerator()).len() as u64)
                        + print_weight(format!("{}", q.denominator()).len() as u64)
                        + 1
                }
                Expr::Real(_) | Expr::Symbol(_) => format!("{n}").len() as u64,
                Expr::Function(name, args) => {
                    st.extend(args.iter().map(|a| &**a));
                    name.len() as u64 + 2
                }
                Expr::Binary(_, l, r) => {
                    st.push(l);
                    st.push(r);
                    0
                }
                Expr::Unary(_, a) => {
                    st.push(a);
                    0
                }
            };
        }
        t
    }

    /// S1 fix-up round 3 (review 3, B2): a random tree of depth at most d with long literals,
    /// rationals, floats, powers, the shapes of `simplify`'s rules and shared subtrees (`pool`)
    /// (round 4: shared by the text and the fold-work tests)
    fn random_tree(rng: &mut rand::rngs::StdRng, d: u32, pool: &mut Vec<Arc<Expr>>) -> Arc<Expr> {
        use rand::Rng;
        let r: f64 = rng.gen();
        if d == 0 || r < 0.22 {
            let leaf = match rng.gen_range(0..14) {
                0 => Expr::symbol("x"),
                1 => Expr::symbol("pi"),
                2 => Expr::from(0),
                3 => Expr::from(1),
                4 => Expr::from(rng.gen_range(-9..10)),
                5 => Expr::Rational(
                    Rational::new(
                        Integer::from(rng.gen_range(-50..50)),
                        Integer::from(rng.gen_range(1..60)),
                    )
                    .unwrap(),
                ),
                6 | 7 => {
                    let k = rng.gen_range(1..700u32);
                    Expr::Integer(Integer::from(2).pow(k) + Integer::from(rng.gen_range(-3..4)))
                }
                8 => Expr::Rational(
                    Rational::new(
                        Integer::from(3).pow(rng.gen_range(1..300u32)),
                        Integer::from(7).pow(rng.gen_range(1..300u32)),
                    )
                    .unwrap(),
                ),
                9 => Expr::Real([0.5, -1.5, 5e-324, 1e300][rng.gen_range(0..4)]),
                10 if !pool.is_empty() => return pool[rng.gen_range(0..pool.len())].clone(),
                _ => Expr::from(rng.gen_range(2..6)),
            };
            return Arc::new(leaf);
        }
        let e = if r < 0.72 {
            let op = [
                BinaryOp::Add,
                BinaryOp::Sub,
                BinaryOp::Mul,
                BinaryOp::Mul,
                BinaryOp::Div,
                BinaryOp::Div,
                BinaryOp::Pow,
                BinaryOp::Pow,
            ][rng.gen_range(0..8)];
            let l = random_tree(rng, d - 1, pool);
            let rr = if op == BinaryOp::Pow && rng.gen_bool(0.7) {
                Arc::new(match rng.gen_range(0..9) {
                    0 => Expr::from(0),
                    1 => Expr::from(1),
                    2 => Expr::from(-2),
                    3 => Expr::from(64),
                    4 => Expr::from(512),
                    5 => Expr::from(-300),
                    6 => Expr::from(4096),
                    7 => Expr::Binary(
                        BinaryOp::Add,
                        Arc::new(Expr::from(100)),
                        Arc::new(Expr::from(28)),
                    ),
                    _ => Expr::Rational(Rational::new(Integer::from(1), Integer::from(2)).unwrap()),
                })
            } else {
                random_tree(rng, d - 1, pool)
            };
            Expr::Binary(op, l, rr)
        } else if r < 0.93 {
            let op = [
                UnaryOp::Neg,
                UnaryOp::Neg,
                UnaryOp::Sin,
                UnaryOp::Exp,
                UnaryOp::Log,
                UnaryOp::Sqrt,
                UnaryOp::Arccos,
            ][rng.gen_range(0..7)];
            Expr::Unary(op, random_tree(rng, d - 1, pool))
        } else {
            let a = random_tree(rng, d - 1, pool);
            let b = random_tree(rng, d - 1, pool);
            Expr::Function("g".into(), vec![a, b])
        };
        let e = Arc::new(e);
        if rng.gen_bool(0.15) {
            pool.push(e.clone());
        }
        e
    }

    /// S1 fix-up round 3 (review 3, B2): on 3000 random trees (seeded) with shared subtrees, long literals,
    /// rationals, powers (also with exponents that fold to integers) and the shapes of the rules
    /// of `simplify` (0 * x, x^0, x^1, (c * rest) / d), `tree_text` is at least the printing weight
    /// of the tree and of `simplify`'s tree.
    #[test]
    fn tree_text_bounds_simplify_on_random_trees() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5131_0003);
        let (mut tested, mut grew) = (0u32, 0u32);
        for i in 0..3000 {
            let mut pool = Vec::new();
            let depth = rng.gen_range(1..7);
            let e = random_tree(&mut rng, depth, &mut pool);
            // keep the folds small enough for a debug test
            if max_fold_guess(&e) > 1 << 18 {
                continue;
            }
            let mut memo = TextMemo::default();
            let t = tree_text(&e, &mut memo);
            let a = printed_weight(&e);
            let b = printed_weight(&crate::simplify::simplify(&e));
            assert!(
                t >= a && t >= b,
                "tree {i}: {e}: tree_text {t}, weight {a}, of simplify {b}"
            );
            tested += 1;
            if b > a {
                grew += 1;
            }
        }
        // the filter keeps most trees, and simplify's text grows on many of them
        assert!(tested > 2500 && grew > 300, "{tested} trees, {grew} grew");
    }

    /// a rough size of the largest power `simplify` may fold in e (to skip slow debug cases)
    fn max_fold_guess(e: &Expr) -> u64 {
        let mut best = 0u64;
        let mut st: Vec<&Expr> = vec![e];
        while let Some(n) = st.pop() {
            match n {
                Expr::Binary(op, l, r) => {
                    if *op == BinaryOp::Pow {
                        let k = match &**r {
                            Expr::Integer(n) => NumericConversion::to_i64(n)
                                .map(|k| k.unsigned_abs())
                                .unwrap_or(u64::MAX),
                            _ => 4096,
                        };
                        best = best.max(text_bits(l).saturating_mul(k));
                    }
                    st.push(l);
                    st.push(r);
                }
                Expr::Unary(_, a) => st.push(a),
                Expr::Function(_, args) => st.extend(args.iter().map(|a| &**a)),
                _ => {}
            }
        }
        best
    }
    fn text_bits(e: &Expr) -> u64 {
        let mut b = 0u64;
        let mut st: Vec<&Expr> = vec![e];
        while let Some(n) = st.pop() {
            match n {
                Expr::Integer(i) => b += i.bit_length() + 1,
                Expr::Rational(q) => b += q.numerator().bit_length() + q.denominator().bit_length(),
                Expr::Binary(_, l, r) => {
                    st.push(l);
                    st.push(r);
                }
                Expr::Unary(_, a) => st.push(a),
                Expr::Function(_, args) => st.extend(args.iter().map(|a| &**a)),
                _ => {}
            }
        }
        b
    }

    /// S1 fix-up round 4 (review 4, B1): the work of the folds `simplify` makes on e, with the
    /// cost formulas of `TextBound` applied to the actual operands (each distinct node once, as
    /// `simplify` walks it): the rules of `simplify_binary` and `simplify_unary` on the simplified
    /// children decide which fold runs. The model (`TextMemo::work`) must be at least this.
    fn actual_fold_work(root: &Expr) -> u64 {
        use std::collections::HashMap;
        fn lit(e: &Expr) -> Option<Fold> {
            match e {
                Expr::Integer(i) => Some(Fold::of_integer(i)),
                Expr::Rational(q) => Some(Fold::of_rational(q)),
                _ => None,
            }
        }
        fn is_int(e: &Expr, n: i64) -> bool {
            matches!(e, Expr::Integer(i) if *i == Integer::from(n))
        }
        fn node_work(e: &Expr, kids: &[Arc<Expr>]) -> u64 {
            match e {
                Expr::Binary(op, _, _) => {
                    let (l, r) = (&*kids[0], &*kids[1]);
                    let (fl, fr) = (lit(l), lit(r));
                    let copies = |a: Option<Fold>, b: Option<Fold>| {
                        a.map(fold_copy_work).unwrap_or(0) + b.map(fold_copy_work).unwrap_or(0)
                    };
                    match op {
                        BinaryOp::Add if is_int(l, 0) || is_int(r, 0) => 0,
                        BinaryOp::Sub if is_int(r, 0) => 0,
                        BinaryOp::Mul
                            if is_int(l, 0) || is_int(l, 1) || is_int(r, 0) || is_int(r, 1) =>
                        {
                            0
                        }
                        BinaryOp::Div if is_int(r, 1) => 0,
                        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => match (fl, fr) {
                            (Some(a), Some(b)) => fold_binary_work(*op, a, b),
                            (a, b) => copies(a, b),
                        },
                        BinaryOp::Div => match (fl, fr) {
                            (Some(a), Some(b)) if !b.zero => fold_binary_work(BinaryOp::Div, a, b),
                            (a, Some(b)) if !b.zero => {
                                let mut w = copies(a, Some(b)) + fold_copy_work(b);
                                if let Expr::Binary(BinaryOp::Mul, x, y) = l {
                                    if let Some(c) = lit(x).or_else(|| lit(y)) {
                                        w += fold_binary_work(BinaryOp::Div, c, b);
                                    }
                                }
                                w
                            }
                            (a, b) => copies(a, b),
                        },
                        BinaryOp::Pow => match (r, fl) {
                            (Expr::Integer(n), Some(f)) => {
                                let k = match NumericConversion::to_i64(n) {
                                    Some(k) => k,
                                    None => return 0,
                                };
                                let ku = k.unsigned_abs();
                                let bits = match l {
                                    Expr::Integer(c) => c.bit_length() + 1,
                                    Expr::Rational(c) => {
                                        c.numerator().bit_length() + c.denominator().bit_length()
                                    }
                                    _ => unreachable!(),
                                };
                                if k == 0
                                    || k == 1
                                    || ku > crate::simplify::MAX_FOLD_EXPONENT as u64
                                    || bits.saturating_mul(ku)
                                        > crate::simplify::MAX_FOLD_POWER_BITS
                                    || (k < 0 && f.zero)
                                {
                                    0
                                } else {
                                    fold_pow_work(f, ku, k < 0)
                                }
                            }
                            _ => 0,
                        },
                        BinaryOp::Mod => 0,
                    }
                }
                Expr::Unary(UnaryOp::Neg, _) => match lit(&kids[0]) {
                    Some(f) => {
                        let (n, d, _) = fold_parts_words(f);
                        OP_BASE + result_units(n, d)
                    }
                    None => 0,
                },
                Expr::Function(name, args) => OP_BASE.saturating_add(
                    COPY_WORD
                        .saturating_mul((name.len() as u64 + 8 * args.len() as u64).div_ceil(8)),
                ),
                _ => 0,
            }
        }
        // post-order over the DAG: each distinct node (by address) once, its value memoized
        let mut val: HashMap<usize, Arc<Expr>> = HashMap::new();
        let mut total = 0u64;
        let mut st: Vec<(&Expr, bool)> = vec![(root, false)];
        let ad = |e: &Expr| std::ptr::from_ref(e).addr();
        while let Some((n, done)) = st.pop() {
            if val.contains_key(&ad(n)) {
                continue;
            }
            let children: Vec<&Expr> = match n {
                Expr::Binary(_, l, r) => vec![&**l, &**r],
                Expr::Unary(_, a) => vec![&**a],
                Expr::Function(_, args) => args.iter().map(|a| &**a).collect(),
                _ => vec![],
            };
            if !done && !children.is_empty() {
                st.push((n, true));
                for c in children.into_iter().rev() {
                    st.push((c, false));
                }
                continue;
            }
            let kids: Vec<Arc<Expr>> = children.iter().map(|c| val[&ad(c)].clone()).collect();
            total = total.saturating_add(node_work(n, &kids));
            let v = match n {
                Expr::Binary(op, _, _) => {
                    crate::simplify::simplify_binary(*op, kids[0].clone(), kids[1].clone())
                }
                Expr::Unary(op, _) => crate::simplify::simplify_unary(*op, kids[0].clone()),
                Expr::Function(name, _) => Arc::new(Expr::Function(name.clone(), kids)),
                leaf => Arc::new(leaf.clone()),
            };
            val.insert(ad(n), v);
        }
        total
    }

    /// The chains of review 4's B1 (with smaller literals, for a debug test) and the other shapes
    /// whose folds grow: a `/ d` chain, a `* d` chain, a `+ 1/k` chain (a growing denominator),
    /// (c * rest) / d chains with a symbol, negations, rational powers with integer and folded
    /// exponents.
    fn fold_chains() -> Vec<Expr> {
        let p = |s: &str| crate::parser::parse(s).expect("parse");
        let k = "(2^253+1)^16";
        let mut v = Vec::new();
        for d in ["3", "7", "(2^70+1)", "1/3"] {
            v.push(p(&format!("{k}{}", format!("/{d}").repeat(150))));
            v.push(p(&format!("sin(pi*{k}{})", format!("/{d}").repeat(150))));
            v.push(p(&format!("g({k}{})", format!("*{d}").repeat(150))));
        }
        let sum: String = (1..120).map(|j| format!("+1/{}", 2 * j + 1)).collect();
        v.push(p(&format!("{k}{sum}")));
        v.push(p(&format!("({k}{sum})^3")));
        v.push(p(&format!("-(-(-(-({k}))))")));
        v.push(p("((2^200+3)/(2^199+5))^40 + (3/7)^(-60) + (5/11)^(2^5+3)"));
        v.push(p(&format!("(x*{k}*3)/7/11/13")));
        v.push(p(&format!("(({k}*x)/3)/5 + (pi*{k})/(2^64+1)")));
        v
    }

    /// S1 fix-up round 4 (review 4, B1): `TextMemo::work` (the model of `tree_bound`) is at least
    /// the fold work of `simplify` with the same cost formulas on the actual operands
    /// (`actual_fold_work`): on the 3000 random trees of the text test (seeded; shared subtrees,
    /// long literals, rationals, powers, the shapes of the rules) and on `fold_chains`. Before the
    /// round, `tree_bound` had no fold work: a 24 KB chain of `/ 3` under a kernel took `simplify`
    /// 35 s after the engine accepted the result in 0.04 s.
    #[test]
    fn fold_work_bounds_simplify_on_random_trees() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5131_0004);
        let mut checked = 0u32;
        let mut cases: Vec<Expr> = fold_chains();
        for _ in 0..3000 {
            let mut pool = Vec::new();
            let depth = rng.gen_range(1..7);
            let e = random_tree(&mut rng, depth, &mut pool);
            if max_fold_guess(&e) > 1 << 18 {
                continue;
            }
            cases.push((*e).clone());
        }
        for (i, e) in cases.iter().enumerate() {
            let mut memo = TextMemo::default();
            tree_text(e, &mut memo);
            let actual = actual_fold_work(e);
            assert!(
                memo.work >= actual,
                "case {i}: {e}: model {} < actual {actual}",
                memo.work
            );
            checked += 1;
        }
        assert!(checked > 2500, "{checked}");
    }

    /// S1 fix-up round 5 (review 5, B1): the per-occurrence model (`TextBound::total` of the root)
    /// is at least the fold work of `simplify` on a copy of the tree with no shared node, which is
    /// what a consumer that rebuilds the tree (`substitute`, `expand`) hands to `simplify`; and the
    /// distinct model (`TextMemo::work`) of the copy equals its per-occurrence model. The same
    /// 3000 seeded random trees and chains as above (their pools share subtrees).
    #[test]
    fn total_fold_work_bounds_simplify_of_an_unshared_copy() {
        use rand::{Rng, SeedableRng};
        fn unshared(e: &Expr) -> Expr {
            match e {
                Expr::Binary(op, l, r) => {
                    Expr::Binary(*op, Arc::new(unshared(l)), Arc::new(unshared(r)))
                }
                Expr::Unary(op, a) => Expr::Unary(*op, Arc::new(unshared(a))),
                Expr::Function(n, args) => Expr::Function(
                    n.clone(),
                    args.iter().map(|a| Arc::new(unshared(a))).collect(),
                ),
                leaf => leaf.clone(),
            }
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5131_0004);
        let mut cases: Vec<Expr> = fold_chains();
        for _ in 0..3000 {
            let mut pool = Vec::new();
            let depth = rng.gen_range(1..7);
            let e = random_tree(&mut rng, depth, &mut pool);
            if max_fold_guess(&e) > 1 << 18 {
                continue;
            }
            cases.push((*e).clone());
        }
        let mut checked = 0u32;
        let mut shared = 0u32;
        for (i, e) in cases.iter().enumerate() {
            let mut memo = TextMemo::default();
            let b = tree_bound(e, &mut memo);
            let copy = unshared(e);
            let actual = actual_fold_work(&copy);
            assert!(
                b.total >= actual,
                "case {i}: {e}: per-occurrence model {} < actual work on the copy {actual}",
                b.total
            );
            assert!(
                b.total >= memo.work,
                "case {i}: total below the distinct work"
            );
            let mut memo2 = TextMemo::default();
            let b2 = tree_bound(&copy, &mut memo2);
            assert_eq!(b2.total, b.total, "case {i}: the copy's total differs");
            if b.total > memo.work {
                shared += 1;
            }
            checked += 1;
        }
        assert!(checked > 2500, "{checked}");
        assert!(shared > 50, "only {shared} cases share a folding subtree");
    }

    /// S1 fix-up round 4 (review 4, B1), the calibration of `MAX_RESULT_FOLD_WORK` (release):
    /// `simplify`'s time per unit of the model on the slow shapes (unbalanced quotients, balanced
    /// gcds of rational powers, sums with growing denominators, (c * rest) / d chains), each
    /// scaled to about 2^26 to 2^29 units. Run with
    /// `cargo test --release -p rustmath-symbolic --lib fold_work_calibration -- --ignored --nocapture`.
    #[test]
    #[ignore = "release measurement"]
    fn fold_work_calibration() {
        let p = |s: &str| crate::parser::parse(s).expect("parse");
        let k = "(2^254+1)^4096";
        let mut rows: Vec<(String, Expr)> = Vec::new();
        for n in [400usize, 800, 1600] {
            rows.push((format!("K/3 x {n}"), p(&format!("{k}{}", "/3".repeat(n)))));
            rows.push((
                format!("pi*K/3 x {n}"),
                p(&format!("pi*{k}{}", "/3".repeat(n))),
            ));
            rows.push((
                format!("K/(2^61-1) x {n}"),
                p(&format!("{k}{}", "/(2^61-1)".repeat(n))),
            ));
        }
        for n in [100usize, 300, 600] {
            let sum: String = (1..n).map(|j| format!("+1/{}", 2 * j + 1)).collect();
            rows.push((
                format!("K + sum 1/(2j+1), {n} terms"),
                p(&format!("{k}{sum}")),
            ));
        }
        for n in [1usize, 2, 4] {
            let pw: Vec<String> = (1..=n)
                .map(|j| format!("((2^254+{j})/(2^253+3))^2048"))
                .collect();
            rows.push((format!("{n} rational powers"), p(&pw.join("+"))));
        }
        // balanced gcds: a quotient of two 2^20-bit integers, a product and a sum of two
        // rational powers with 2^19-bit parts, one quotient by a 2^15-bit power
        let qa = "((2^254+1)/(2^253+3))^2048";
        let qb = "((2^254+5)/(2^253+7))^2048";
        rows.push(("K/K'".into(), p(&format!("{k}/(2^254+3)^4096"))));
        rows.push(("QA*QB".into(), p(&format!("{qa}*{qb}"))));
        rows.push(("QA+QB".into(), p(&format!("{qa}+{qb}"))));
        rows.push(("QA/QB".into(), p(&format!("{qa}/{qb}"))));
        rows.push(("K/3^20000".into(), p(&format!("{k}/3^20000"))));
        for n in [2000usize, 6000] {
            rows.push((format!("K*3 x {n}"), p(&format!("{k}{}", "*3".repeat(n)))));
            rows.push((format!("K+1 x {n}"), p(&format!("{k}{}", "+1".repeat(n)))));
        }
        let mut worst = 0f64;
        for (what, e) in rows {
            let mut memo = TextMemo::default();
            tree_text(&e, &mut memo);
            let t = std::time::Instant::now();
            let s = crate::simplify::simplify(&e);
            let secs = t.elapsed().as_secs_f64();
            drop(s);
            let ns = secs * 1e9 / memo.work.max(1) as f64;
            worst = worst.max(ns);
            eprintln!(
                "{what}: model {:.3e} units, simplify {secs:.3} s, {ns:.2} ns/unit",
                memo.work as f64
            );
        }
        let cap_secs = worst * MAX_RESULT_FOLD_WORK as f64 / 1e9;
        eprintln!("worst {worst:.2} ns/unit: MAX_RESULT_FOLD_WORK is about {cap_secs:.1} s");
        assert!(cap_secs < 10.0, "{cap_secs} s at the cap");
    }
}
