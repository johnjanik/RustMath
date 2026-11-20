//! The truncated Laurent series engine (S1 design §2.1, §2.4, §2.5, §2.6, §2.8): sparse series
//! sum c_e h^e + O(h^P) with h = x - a (or 1/x at infinity) over the constant field of
//! `series_const`, the operations (sums, sparse products, reciprocal, integer powers by binary
//! powering, Miller's recurrence and the binomial path, exp, log, sin and cos, the atan family by
//! integration), the expansion of an `Expr` as one loop over an explicit work stack (no recursion
//! over the input), and the driver's working-precision search.
//!
//! Specification: MAGMA Handbook ch. 49, "Power, Laurent and Puiseux Series" (§49.1.3, §49.1.4,
//! §49.5), digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`; the S1
//! design (frozen at revision 7rt4 with revision 8), ported from its prototype `proto13`.
//!
//! References: Knuth, TAOCP vol. 2, 3rd ed., §4.7 (Miller's recurrence for powers, exp and log
//! through their differential equations); R. P. Brent and H. T. Kung, "Fast algorithms for
//! manipulating formal power series", J. ACM 25(4), 1978, 581-595; S. C. Johnson, "Sparse
//! polynomial arithmetic", ACM SIGSAM Bulletin 8(3), 1974, 63-71; M. Monagan and R. Pearce,
//! "Sparse polynomial division using a heap", J. Symbolic Comput. 46(7), 2011, 807-822.

// Prototype of the truncated Laurent series engine (S1 design, revision 1).
// Series: sum of c_e h^e over a sparse sorted list + O(h^prec), h = x - a (or 1/x at infinity).
// Recurrences: Knuth TAOCP vol. 2, 3rd ed., 4.7 (power, exp, log via the differential
// equations), Brent and Kung 1978 sections 2-3.
// Revision 1: sparse terms (B3b), checked exponents (B2), work charged before each step and
// before every dense allocation (B3), Miller / binomial paths for large integer powers,
// decided non-integer exponents (m1), doubling NeedPrecision growth (m2), strict constant base (m5).

use crate::series_const::*;
use crate::{BinaryOp, Expr, Symbol, UnaryOp};
use rustmath_core::{NumericConversion, Ring};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// revision 7rt4 (red team 4 series, H1): the precision of a series, out of band. `Abs(p)`: every
/// term of exponent below p is stored or is zero, nothing is known at p and above. `Exact`: every
/// term of the value is present. Revision 8 encoded `Exact` as the exponent i64::MAX, which is also
/// a legal exponent and a legal finite precision, so `(x^(2^62)*x^(2^62-1))/x^(2^63-1)` dropped its
/// term at i64::MAX and stayed "exact" (a decided 0 for the value 1). Now every i64 is a legal
/// exponent and a legal finite precision, and `Exact` is a variant of its own. Ordered `Abs(a) <
/// Abs(b)` for a < b and every `Abs` below `Exact`, so the precision of a sum is the minimum.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Prec {
    Abs(i64),
    Exact,
}
pub use Prec::{Abs, Exact};
thread_local! {
    /// revision 7rt4: set when a precision of this attempt was clamped to Abs(i64::MAX); a precision
    /// loss after a clamp cannot be repaired by a larger P, so the driver stops (Unsupported)
    pub static PREC_CLAMPED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
impl Prec {
    /// whether a term of exponent e lies below the precision (is stored or zero)
    pub fn covers(self, e: i64) -> bool {
        match self {
            Exact => true,
            Abs(p) => e < p,
        }
    }
    /// the precision of a series multiplied by h^k: `Exact` stays; a finite sum that overflows
    /// upwards is clamped to Abs(i64::MAX), a weaker true claim (the terms at and above it are in the
    /// O-term); one that overflows downwards is the overflow error of section 2.1
    pub fn add(self, k: i64) -> Result<Prec, SErr> {
        match self {
            Exact => Ok(Exact),
            Abs(p) => match p.checked_add(k) {
                Some(v) => Ok(Abs(v)),
                None if k > 0 => {
                    PREC_CLAMPED.with(|c| c.set(true));
                    Ok(Abs(i64::MAX))
                }
                None => Err(ovf()),
            },
        }
    }
    /// min(self, P) as a finite precision (the working precision P bounds every recurrence)
    pub fn at_most(self, p: i64) -> i64 {
        match self {
            Exact => p,
            Abs(q) => q.min(p),
        }
    }
}
/// integer powers up to this exponent use binary powering (exact canonical coefficients)
pub const POW_BINARY_MAX: i64 = 64;
/// an exact series with at most this many terms is not truncated at the working precision
/// (so 1/(x^300*(1+x)) needs no precision search); larger exact series are
pub const EXACT_KEEP_TERMS: usize = 64;
/// exact * exact products with at most this many pairs are computed in full
pub const EXACT_KEEP_PAIRS: usize = 4096;
/// revision 7rt4 (red team 4 series, H2): isize::MAX / 64, the largest dense buffer `alloc` admits
/// without a budget (a buffer of 16-byte slots, two per step in sin_cos, fits in isize::MAX bytes)
pub const MAX_SERIES_SLOTS: u64 = (isize::MAX as u64) / 64;

#[derive(Clone)]
pub struct Ser {
    /// (exponent, coefficient), increasing exponents, no structurally zero coefficient,
    /// every exponent below prec
    pub t: Vec<(i64, Konst)>,
    pub prec: Prec,
}

fn ovf() -> SErr {
    SErr::Unsupported("a series exponent overflows i64".into())
}
fn cadd(a: i64, b: i64) -> Result<i64, SErr> {
    a.checked_add(b).ok_or_else(ovf)
}
/// precision arithmetic: Exact absorbs (revision 7rt4: out of band, `Prec::add`)
fn padd(a: i64, b: Prec) -> Result<Prec, SErr> {
    b.add(a)
}

pub struct Ctx {
    /// the series of the variable itself
    pub xser: Ser,
    pub p: i64,
    pub work: Work,
}

impl Ctx {
    /// charge a dense buffer of n slots before allocating it
    /// revision 7rt4 (red team 4 series, H2): the units saturate (`16 n` for n >= 2^60 was
    /// u64::MAX, and the base of `Work::charge` wrapped it to 15 units, so the budget admitted the
    /// buffer and `Vec::with_capacity` panicked in release), so a budget refuses the buffer before
    /// the cast and the allocation; without a budget, a buffer of more than MAX_SERIES_SLOTS slots
    /// is refused, so no allocation of a slot count from the input can panic
    fn alloc(&mut self, n: i64) -> Result<usize, SErr> {
        let n = n.max(0) as u64;
        self.work.charge(n.saturating_mul(16))?;
        if n > MAX_SERIES_SLOTS {
            return Err(SErr::Unsupported(
                "a series buffer of more than 2^57 slots".into(),
            ));
        }
        Ok(n as usize)
    }
}

impl Ser {
    pub fn exact_zero() -> Ser {
        Ser {
            t: vec![],
            prec: Exact,
        }
    }
    pub fn exact_const(k: Konst) -> Ser {
        if k.is_zero() {
            return Ser::exact_zero();
        }
        Ser {
            t: vec![(0, k)],
            prec: Exact,
        }
    }
    pub fn is_exact_zero(&self) -> bool {
        self.t.is_empty() && self.prec == Exact
    }
    /// first exponent with a structurally nonzero coefficient (prec when empty, 0 for exact 0)
    pub fn val(&self) -> i64 {
        match self.t.first() {
            Some((e, _)) => *e,
            None => match self.prec {
                Exact => 0,
                Abs(p) => p,
            },
        }
    }
    pub fn coef(&self, e: i64) -> Konst {
        match self.t.binary_search_by(|(x, _)| x.cmp(&e)) {
            Ok(i) => self.t[i].1.clone(),
            Err(_) => Konst::zero(),
        }
    }
    /// working truncation at p: an exact series stays exact when all its terms are below p or
    /// when it has at most EXACT_KEEP_TERMS terms
    /// Revision 2 (review 2, m1): the working truncation keeps the first term. When every term
    /// is at or above p, the first term e0 is kept with precision min(prec, e0 + 1), so a decided
    /// valuation above the working precision is not forgotten.
    pub fn trunc(mut self, p: i64) -> Ser {
        if self.prec == Exact && self.t.len() <= EXACT_KEEP_TERMS {
            return self;
        }
        match self.t.first() {
            Some(&(e0, _)) if e0 >= p => {
                self.t.truncate(1);
                // revision 7rt4: e0 = i64::MAX only in an exact series whose single term it is (the
                // terms increase), which keeps its precision
                if let Some(e1) = e0.checked_add(1) {
                    self.prec = self.prec.min(Abs(e1));
                }
                self
            }
            _ => self.trunc_hard(p),
        }
    }
    /// truncation of a result to absolute precision p
    pub fn trunc_hard(mut self, p: i64) -> Ser {
        if self.prec == Exact && self.t.last().map(|(e, _)| *e < p).unwrap_or(true) {
            return self;
        }
        if self.prec > Abs(p) {
            self.prec = Abs(p);
        }
        let pr = self.prec;
        self.t.retain(|(e, _)| pr.covers(*e));
        self
    }
    fn is_exact_monomial(&self) -> bool {
        self.prec == Exact && self.t.len() == 1
    }
    fn lead_status(&self) -> Option<ZS> {
        self.t.first().map(|(_, k)| k.zero_status())
    }
    /// multiply by h^k
    fn shift(&self, k: i64) -> Result<Ser, SErr> {
        // revision 4 (review 4, B1): a term copy is O(1) (shared constants), charged 4 units
        charge(
            SER_TERM
                .saturating_mul(self.t.len() as u64)
                .saturating_add(16),
        );
        // revision 7rt4: a term at or above a clamped precision (Prec::add) is in the O-term; a term
        // whose exponent overflows is an error only in an exact series (or downwards)
        let prec = padd(k, self.prec)?;
        let mut t = Vec::with_capacity(self.t.len());
        for (e, c) in &self.t {
            match e.checked_add(k) {
                Some(e2) if prec.covers(e2) => t.push((e2, c.clone())),
                Some(_) => break,
                None if k > 0 && prec != Exact => break,
                None => return Err(ovf()),
            }
        }
        Ok(Ser { t, prec })
    }
    /// sparse series from a dense buffer g[i] = coefficient of h^(val + i)
    fn from_dense(val: i64, g: Vec<Konst>, prec: Prec) -> Ser {
        let t = g
            .into_iter()
            .enumerate()
            .filter(|(_, k)| !k.is_zero())
            .map(|(i, k)| (val + i as i64, k))
            .collect();
        Ser { t, prec }
    }
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn add(a: &Ser, b: &Ser, cx: &mut Ctx) -> Result<Ser, SErr> {
    let prec = a.prec.min(b.prec);
    // revision 4 (review 4, B1): 4 units per term copied (an O(1) copy of a shared constant)
    cx.work.charge(SER_TERM * (a.t.len() + b.t.len()) as u64)?;
    let mut t: Vec<(i64, Konst)> = Vec::with_capacity(a.t.len() + b.t.len());
    let (mut i, mut j) = (0, 0);
    while i < a.t.len() || j < b.t.len() {
        let (e, k) = if j >= b.t.len() || (i < a.t.len() && a.t[i].0 < b.t[j].0) {
            i += 1;
            (a.t[i - 1].0, a.t[i - 1].1.clone())
        } else if i >= a.t.len() || b.t[j].0 < a.t[i].0 {
            j += 1;
            (b.t[j - 1].0, b.t[j - 1].1.clone())
        } else {
            i += 1;
            j += 1;
            (a.t[i - 1].0, a.t[i - 1].1.add(&b.t[j - 1].1))
        };
        if !prec.covers(e) {
            break;
        }
        if !k.is_zero() {
            t.push((e, k));
        }
    }
    check()?;
    Ok(Ser { t, prec }.trunc(cx.p))
}

/// revision 5: the sum of several signed series and a constant by one k-way merge (each
/// exponent's coefficients summed by Konst::sum_many)
/// revision 6 (review 6, B1 class): the merge is a binary heap of the k cursors, so no container
/// grows per exponent (revision 5 kept a BTreeMap of vectors, about 176 bytes per exponent).
/// Charged first: 16 per series, SER_TERM per term, and SEARCH_CMP (bits(k) + 1) per term for
/// the heap.
// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
fn sum_series(mut sers: Vec<(bool, Ser)>, c: Konst, cx: &mut Ctx) -> Result<Ser, SErr> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let prec = sers.iter().map(|(_, s)| s.prec).min().unwrap_or(Exact);
    if !c.is_zero() && prec.covers(0) {
        sers.push((false, Ser::exact_const(c)));
    }
    let terms: u64 = sers.iter().map(|(_, s)| s.t.len() as u64).sum();
    let k = sers.len() as u64;
    // revision 6: 24 per series (its slot in `sers`, 40 bytes, its cursor and its heap entry)
    cx.work
        .charge((24 * k).saturating_add(terms.saturating_mul(SER_TERM + SEARCH_CMP * bits_u(k))))?;
    let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::with_capacity(sers.len());
    let mut pos = vec![0usize; sers.len()];
    for (i, (_, s)) in sers.iter().enumerate() {
        if let Some((e, _)) = s.t.first() {
            heap.push(Reverse((*e, i)));
        }
    }
    let mut t: Vec<(i64, Konst)> = Vec::new();
    let mut cur: Vec<(bool, Konst)> = Vec::new();
    let mut cur_e: Option<i64> = None;
    // the heap yields exponents in increasing order; each exponent's coefficients are summed by
    // one sum_many when the next exponent starts
    while let Some(Reverse((e, i))) = heap.pop() {
        if !prec.covers(e) {
            break;
        }
        if cur_e != Some(e) {
            if let Some(ce) = cur_e {
                let k = Konst::sum_many(&cur);
                cur.clear();
                if !k.is_zero() {
                    t.push((ce, k));
                }
            }
            cur_e = Some(e);
        }
        let (ng, s) = &sers[i];
        cur.push((*ng, s.t[pos[i]].1.clone()));
        pos[i] += 1;
        if pos[i] < s.t.len() {
            heap.push(Reverse((s.t[pos[i]].0, i)));
        }
        if faulted() {
            break;
        }
    }
    if let Some(ce) = cur_e {
        let k = Konst::sum_many(&cur);
        if !k.is_zero() {
            t.push((ce, k));
        }
    }
    check()?;
    Ok(Ser { t, prec }.trunc(cx.p))
}

pub fn neg(a: &Ser) -> Ser {
    // revision 4 (review 4, B1): each coefficient's negation charges copy_units
    charge(SER_TERM.saturating_mul(a.t.len() as u64).saturating_add(16));
    Ser {
        t: a.t.iter().map(|(e, k)| (*e, k.neg())).collect(),
        prec: a.prec,
    }
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn scale(a: &Ser, k: &Konst) -> Result<Ser, SErr> {
    if k.is_zero() {
        return Ok(Ser::exact_zero());
    }
    // revision 6: the output slots, SER_TERM each (a coefficient 1 times k is k itself, uncharged)
    charge(SER_TERM.saturating_mul(a.t.len() as u64).saturating_add(16));
    let mut t = Vec::with_capacity(a.t.len());
    for (e, x) in &a.t {
        let p = x.mul(k);
        if !p.is_zero() {
            t.push((*e, p));
        }
    }
    check()?;
    Ok(Ser { t, prec: a.prec })
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn mul(a: &Ser, b: &Ser, cx: &mut Ctx) -> Result<Ser, SErr> {
    if a.is_exact_zero() || b.is_exact_zero() {
        return Ok(Ser::exact_zero());
    }
    let (va, vb) = (a.val(), b.val());
    let val = cadd(va, vb)?;
    let prec = padd(va, b.prec)?.min(padd(vb, a.prec)?);
    // small exact products are computed in full (then truncated by the working rule)
    // m1 (review 2): at least the first term val is computed, even when val >= P
    // revision 7rt2 (red team 2, S-B2): every exponent sum of a pair is at least val, which fits,
    // so a sum can only overflow upwards. An exact product whose top exponent does not fit in i64
    // is not kept in full: it takes the working cutoff, and a pair sum that overflows lies above
    // any cutoff (checked_add None means "beyond the cutoff" below)
    let top_fits = match (a.t.last(), b.t.last()) {
        (Some(x), Some(y)) => x.0.checked_add(y.0).is_some(),
        _ => true, // an operand without terms: no pair
    };
    // revision 7rt4 (red team 4 series, H1): the cutoff is a Prec; an `Exact` cutoff drops no pair
    // (every pair sum lies between val and the top sum, which fit), so an exact product keeps
    // `Exact` only when it computed every pair
    let cutoff = if prec == Exact && top_fits && a.t.len() * b.t.len() <= EXACT_KEEP_PAIRS {
        Exact
    } else {
        prec.min(Abs(cx.p.max(val.saturating_add(1))))
    };
    // a-priori charge of every contributing pair, before any coefficient work
    let mut w: u64 = 0;
    let mut pairs: u64 = 0;
    let mut dropped = false;
    // revision 4 (review 4, m5): each term's words once, not once per pair
    let wbs: Vec<u64> = b.t.iter().map(|(_, k)| k.words()).collect();
    // revision 5: the counting loop stops once the sum passes the remaining budget (each pair adds
    // at least 2), so it never runs longer than the work it would admit
    let rem = remaining();
    'pairs: for (ea, ka) in &a.t {
        let wa = ka.words();
        for (jb, (eb, _kb)) in b.t.iter().enumerate() {
            // revision 7rt2: an overflowing sum is above every cutoff
            let e = match ea.checked_add(*eb) {
                Some(e) if cutoff.covers(e) => e,
                _ => {
                    dropped = true;
                    break;
                }
            };
            let _ = e;
            pairs += 1;
            let wb = wbs[jb];
            w = w.saturating_add(if wa <= 1 || wb <= 1 {
                wa + wb
            } else {
                wa.saturating_mul(wb)
            });
            if w > rem {
                break 'pairs;
            }
        }
    }
    cx.work.charge(w)?;
    let out_prec = if prec == Exact && !dropped {
        Exact
    } else {
        cutoff
    };
    if pairs == 0 {
        return Ok(Ser {
            t: vec![],
            prec: out_prec,
        });
    }
    let hi = a.t.last().unwrap().0.saturating_add(b.t.last().unwrap().0);
    let hi = match cutoff {
        Abs(c) => hi.min(c.saturating_sub(1)),
        Exact => hi,
    };
    // revision 7rt2 (red team 2, S-B2): the span is checked; a span that does not fit in i64 takes
    // the heap merge, whose size is the pair count (revision 7rt1 subtracted without a check, so
    // a span above i64::MAX wrapped and the dense write indexed an empty buffer)
    let span: Option<i64> = hi.checked_sub(val).and_then(|d| d.checked_add(1));
    let dense = matches!(span, Some(sp) if sp <= (2 * pairs as i64).saturating_add(64));
    let t = if dense {
        let n = cx.alloc(span.unwrap())?;
        let mut c = vec![Konst::zero(); n];
        for (ea, ka) in &a.t {
            for (eb, kb) in &b.t {
                let e = match ea.checked_add(*eb) {
                    Some(e) if cutoff.covers(e) => e,
                    _ => break,
                };
                let i = (e - val) as usize; // val <= e <= hi, and hi - val + 1 = span fits
                c[i] = c[i].add(&ka.mul(kb));
            }
        }
        Ser::from_dense(val, c, out_prec).t
    } else {
        // revision 6 (review 6, B1): a heap merge of the rows of the shorter operand (Johnson
        // 1974): the products come out in increasing exponent, so the only containers are the
        // heap (one cursor per row) and the output. Revision 5 accumulated every pair in a
        // BTreeMap uncharged (about 40 bytes and a search per pair). Charged first: per pair
        // SEARCH_CMP (bits(rows) + 1) for the heap and SER_TERM for an output slot, 16 per row.
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let (s, l) = if a.t.len() <= b.t.len() {
            (a, b)
        } else {
            (b, a)
        };
        let rows = s.t.len() as u64;
        cx.work.charge(
            pairs
                .saturating_mul(SER_TERM + SEARCH_CMP * bits_u(rows))
                .saturating_add(16 * rows),
        )?;
        let mut heap: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::with_capacity(s.t.len());
        let mut pos = vec![0usize; s.t.len()];
        for (i, (es, _)) in s.t.iter().enumerate() {
            if let Some(e) = es.checked_add(l.t[0].0).filter(|e| cutoff.covers(*e)) {
                heap.push(Reverse((e, i)));
            }
        }
        let mut out: Vec<(i64, Konst)> = Vec::with_capacity(match span {
            Some(sp) => (pairs as usize).min(sp as usize),
            None => pairs as usize,
        });
        while let Some(Reverse((e, i))) = heap.pop() {
            let p = s.t[i].1.mul(&l.t[pos[i]].1);
            match out.last_mut() {
                Some((le, lk)) if *le == e => *lk = lk.add(&p),
                _ => out.push((e, p)),
            }
            pos[i] += 1;
            if pos[i] < l.t.len() {
                if let Some(e2) = s.t[i]
                    .0
                    .checked_add(l.t[pos[i]].0)
                    .filter(|e| cutoff.covers(*e))
                {
                    heap.push(Reverse((e2, i)));
                }
            }
            if faulted() {
                break;
            }
        }
        out.retain(|(_, k)| !k.is_zero());
        out
    };
    check()?;
    Ok(Ser { t, prec: out_prec }.trunc(cx.p))
}

/// 1/b; b's leading coefficient must be decided nonzero.
// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn inv(b: &Ser, cx: &mut Ctx) -> Result<Ser, SErr> {
    if b.t.is_empty() {
        if b.prec == Exact {
            return Err(SErr::DivisionByZero(
                "division by an expression that is identically 0".into(),
            ));
        }
        return Err(SErr::NeedPrecision);
    }
    let vb = b.val();
    let g0 = b.t[0].1.inverse(true)?;
    let val = vb.checked_neg().ok_or_else(ovf)?;
    if b.is_exact_monomial() {
        return Ok(Ser {
            t: vec![(val, g0)],
            prec: Exact,
        }
        .trunc(cx.p));
    }
    // revision 7rt4: pb - 2 vb in two checked steps (Prec::add), then at most P
    let prec = b.prec.add(val)?.add(val)?.at_most(cx.p);
    let n = cx.alloc(prec.saturating_sub(val))?;
    if n == 0 {
        return Ok(Ser {
            t: vec![],
            prec: Abs(prec),
        });
    }
    cx.work.charge(SER_TERM * b.t.len() as u64)?;
    // revision 7rt3 (red team 3 series, m2): the offset e - vb as an unsigned distance, and only
    // the offsets below n (the loop never reads one at or above n): revision 7rt2 subtracted in i64,
    // which overflows for `(1)/((1)/x^2 + x^9223372036854775806)` (a panic in debug)
    let bo: Vec<(usize, &Konst)> = b.t[1..]
        .iter()
        .map(|(e, k)| (e.abs_diff(vb), k))
        .take_while(|(d, _)| *d < n as u64)
        .map(|(d, k)| (d as usize, k))
        .collect();
    let mut g: Vec<Konst> = Vec::with_capacity(n);
    g.push(g0.clone());
    for k in 1..n {
        let mut w = g0.words();
        for &(j, bj) in &bo {
            if j > k {
                break;
            }
            w = w.saturating_add(bj.words().saturating_mul(g[k - j].words()));
        }
        cx.work.charge(w)?;
        let mut s = Konst::zero();
        for &(j, bj) in &bo {
            if j > k {
                break;
            }
            if g[k - j].is_zero() {
                continue;
            }
            s = s.add(&bj.mul(&g[k - j]));
        }
        g.push(s.mul(&g0).neg());
        check()?;
    }
    Ok(Ser::from_dense(val, g, Abs(prec)))
}

/// split b = b0 + u with u of valuation >= 1; b must have valuation >= 0
fn split0(b: &Ser) -> (Konst, Ser) {
    charge(SER_TERM.saturating_mul(b.t.len() as u64).saturating_add(16));
    let b0 = b.coef(0);
    let u = Ser {
        t: b.t.iter().filter(|(e, _)| *e != 0).cloned().collect(),
        prec: b.prec,
    };
    (b0, u)
}

/// revision 7rt3 (red team 3 series, H1): the rule of section 2.1 for an empty truncated operand.
/// `O(h^q)` has an undecided valuation of at least q, so it meets "valuation >= 0" only when
/// q >= 0, and no precondition on its leading coefficient; an operation whose precondition it does
/// not meet asks for precision, as inv, log and pow_const do. Revision 7rt2 returned Ok for every
/// empty series, so exp(O(h^-1)) was O(h^-1), and a factor x^10 turned it into a decided 0.
fn require_nonneg_val(b: &Ser, what: &str) -> Result<(), SErr> {
    if b.t.is_empty() {
        return if b.prec >= Abs(0) {
            Ok(())
        } else {
            Err(SErr::NeedPrecision)
        };
    }
    if b.val() < 0 {
        return match b.lead_status() {
            Some(ZS::Unknown) => Err(undecided(&b.t[0].1)),
            _ => Err(SErr::NotLaurent(format!(
                "{} of an argument with a pole at the point has no Laurent expansion there",
                what
            ))),
        };
    }
    Ok(())
}

fn qrec(k: i64) -> Rational {
    Rational::new(Integer::from(1), Integer::from(k)).unwrap()
}

/// offsets and coefficients of u (valuation >= 1) below n
fn offsets(u: &Ser, n: usize) -> Vec<(usize, Konst)> {
    charge(SER_TERM.saturating_mul(u.t.len() as u64).saturating_add(16));
    u.t.iter()
        .filter(|(e, _)| *e >= 1 && (*e as u64) < n as u64)
        .map(|(e, k)| (*e as usize, k.clone()))
        .collect()
}

/// exp(u) for u of valuation >= 1, as a series to absolute precision prec
fn exp_u(u: &Ser, prec: i64, cx: &mut Ctx) -> Result<Ser, SErr> {
    let n = cx.alloc(prec)?;
    let iu = offsets(u, n);
    let mut g: Vec<Konst> = Vec::with_capacity(n);
    for k in 0..n {
        if k == 0 {
            g.push(Konst::one());
            continue;
        }
        let mut w = 1u64;
        for (j, uj) in &iu {
            if *j > k {
                break;
            }
            w = w.saturating_add(uj.words().saturating_mul(g[k - j].words()));
        }
        cx.work.charge(w)?;
        let mut s = Konst::zero();
        for (j, uj) in &iu {
            if *j > k {
                break;
            }
            if g[k - j].is_zero() {
                continue;
            }
            s = s.add(&uj.mul(&g[k - j]).scale(&qi(*j as i64)));
        }
        g.push(s.scale(&qrec(k as i64)));
        check()?;
    }
    Ok(Ser::from_dense(0, g, Abs(prec)))
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn exp(b: &Ser, cx: &mut Ctx) -> Result<Ser, SErr> {
    require_nonneg_val(b, "exp")?;
    let prec = b.prec.at_most(cx.p);
    let (b0, u) = split0(b);
    let e = exp_u(&u, prec, cx)?;
    let k = kapply(UnaryOp::Exp, &b0)?;
    scale(&e, &k)
}

/// (S, C) = (sin u, cos u) or (sinh u, cosh u) for u of valuation >= 1
fn sincos_u(u: &Ser, prec: i64, hyper: bool, cx: &mut Ctx) -> Result<(Ser, Ser), SErr> {
    // revision 7rt2: saturating (a working precision above 2^62 is charged, not wrapped)
    let n = cx.alloc(prec.max(0).saturating_mul(2))? / 2;
    let iu: Vec<(usize, Konst)> = offsets(u, n)
        .into_iter()
        .map(|(j, k)| (j, k.scale(&qi(j as i64))))
        .collect();
    let mut s: Vec<Konst> = Vec::with_capacity(n);
    let mut c: Vec<Konst> = Vec::with_capacity(n);
    for k in 0..n {
        if k == 0 {
            s.push(Konst::zero());
            c.push(Konst::one());
            continue;
        }
        let mut w = 1u64;
        for (j, ju) in &iu {
            if *j > k {
                break;
            }
            w = w.saturating_add(
                ju.words()
                    .saturating_mul(c[k - j].words() + s[k - j].words()),
            );
        }
        cx.work.charge(w)?;
        let mut ss = Konst::zero();
        let mut cc = Konst::zero();
        for (j, ju) in &iu {
            if *j > k {
                break;
            }
            if !c[k - j].is_zero() {
                ss = ss.add(&ju.mul(&c[k - j]));
            }
            if !s[k - j].is_zero() {
                cc = cc.add(&ju.mul(&s[k - j]));
            }
        }
        let r = qrec(k as i64);
        s.push(ss.scale(&r));
        let cc = cc.scale(&r);
        c.push(if hyper { cc } else { cc.neg() });
        check()?;
    }
    Ok((
        Ser::from_dense(0, s, Abs(prec)),
        Ser::from_dense(0, c, Abs(prec)),
    ))
}

pub fn sin_cos(b: &Ser, hyper: bool, cx: &mut Ctx) -> Result<(Ser, Ser), SErr> {
    require_nonneg_val(b, if hyper { "sinh/cosh" } else { "sin/cos" })?;
    let prec = b.prec.at_most(cx.p);
    let (b0, u) = split0(b);
    let (s, c) = sincos_u(&u, prec, hyper, cx)?;
    let (sop, cop) = if hyper {
        (UnaryOp::Sinh, UnaryOp::Cosh)
    } else {
        (UnaryOp::Sin, UnaryOp::Cos)
    };
    let sb0 = kapply(sop, &b0)?;
    let cb0 = kapply(cop, &b0)?;
    // sin(b0+u) = sin b0 cos u + cos b0 sin u ; cos(b0+u) = cos b0 cos u - sin b0 sin u
    // sinh(b0+u) = sinh b0 cosh u + cosh b0 sinh u ; cosh(b0+u) = cosh b0 cosh u + sinh b0 sinh u
    let sin_b = add(&scale(&c, &sb0)?, &scale(&s, &cb0)?, cx)?;
    let t = scale(&s, &sb0)?;
    let cos_b = add(&scale(&c, &cb0)?, &(if hyper { t } else { neg(&t) }), cx)?;
    Ok((sin_b, cos_b))
}

/// log b; b must have valuation 0 with a nonzero constant term. S1 request C-1: `tb` is the
/// class of b's coefficients (`Tail`); an argument that leaves the cut (-oo, 0) is refused
/// (`cut_check`)
// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn log(b: &Ser, tb: Tail, cx: &mut Ctx) -> Result<Ser, SErr> {
    if b.t.is_empty() {
        if b.prec == Exact {
            return Err(SErr::DivisionByZero("log(0)".into()));
        }
        return Err(SErr::NeedPrecision);
    }
    if let Some(ZS::Unknown) = b.lead_status() {
        return Err(undecided(&b.t[0].1));
    }
    if b.val() != 0 {
        return Err(SErr::NotLaurent(
            "log has a logarithmic singularity at the point".into(),
        ));
    }
    cut_check("log", Cut::Log, b, tb)?;
    let prec = b.prec.at_most(cx.p);
    let b0 = b.t[0].1.clone();
    let w = scale(b, &b0.inverse(true)?)?;
    let n = cx.alloc(prec)?;
    let iw = offsets(&w, n);
    let mut l: Vec<Konst> = vec![Konst::zero(); n];
    for k in 1..n {
        // k L_k = k w_k - sum_{j=1}^{k-1} j L_j w_{k-j}
        // revision 7rt2 (red team 2, S-B1): the offsets are sorted, so the charge loop stops at the
        // first offset >= k, as the summing loop below and every other recurrence do, and w_k is
        // found by one binary search. Revision 7rt1 scanned every offset at every step while
        // charging only those below k, so a sparse argument whose offsets lie far above the low
        // steps did n * |offsets| work for about 17 units per step.
        let below = iw.partition_point(|(i, _)| *i < k);
        let wk = match iw.get(below) {
            Some((i, wi)) if *i == k => wi.clone(),
            _ => Konst::zero(),
        };
        let mut cost = 1u64 + bits_u(iw.len() as u64);
        for (i, wi) in &iw[..below] {
            cost = cost.saturating_add(l[k - i].words().saturating_mul(wi.words()));
        }
        cx.work.charge(cost)?;
        let mut s = wk.scale(&qi(k as i64));
        for (i, wi) in &iw {
            if *i >= k {
                break;
            }
            let j = k - i;
            if l[j].is_zero() {
                continue;
            }
            s = s.sub(&l[j].mul(wi).scale(&qi(j as i64)));
        }
        l[k] = s.scale(&qrec(k as i64));
        check()?;
    }
    let lb0 = kapply(UnaryOp::Log, &b0)?;
    let ls = Ser::from_dense(0, l, Abs(prec));
    add(&ls, &Ser::exact_const(lb0), cx)
}

/// Miller's recurrence (Knuth 4.7 eq. (9)) for (1 + w)^alpha, w of valuation >= 1, n slots
fn miller(w: &Ser, alpha: &Konst, n: usize, cx: &mut Ctx) -> Result<Vec<Konst>, SErr> {
    let iw = offsets(w, n);
    let ap1 = alpha.add(&Konst::one());
    let mut g: Vec<Konst> = Vec::with_capacity(n);
    for k in 0..n {
        if k == 0 {
            g.push(Konst::one());
            continue;
        }
        let mut cost = 1u64;
        for (j, wj) in &iw {
            if *j > k {
                break;
            }
            cost = cost.saturating_add(
                ap1.words()
                    .saturating_mul(wj.words())
                    .saturating_mul(g[k - j].words()),
            );
        }
        cx.work.charge(cost)?;
        // g_k = (1/k) sum_{j=1}^k ((alpha+1) j - k) w_j g_{k-j}
        let mut s = Konst::zero();
        for (j, wj) in &iw {
            if *j > k {
                break;
            }
            if g[k - j].is_zero() {
                continue;
            }
            let coef = ap1.scale(&qi(*j as i64)).sub(&Konst::int(k as i64));
            if coef.is_zero() {
                continue;
            }
            s = s.add(&coef.mul(wj).mul(&g[k - j]));
        }
        g.push(s.scale(&qrec(k as i64)));
        check()?;
    }
    Ok(g)
}

/// b^alpha for a constant non-integer alpha. S1 request C-1: `tb` is the class of b's coefficients
/// for a power of the input (sqrt, a constant exponent), checked against the cut (-oo, 0) of the
/// principal power (`cut_check`), and `None` for the integrands of the atan family, whose
/// functions check their own cuts first
pub fn pow_const(b: &Ser, alpha: &Konst, tb: Option<Tail>, cx: &mut Ctx) -> Result<Ser, SErr> {
    if b.t.is_empty() {
        if b.prec == Exact {
            return Ok(Ser::exact_const(kpow_const(&Konst::zero(), alpha)?));
        }
        return Err(SErr::NeedPrecision);
    }
    if let Some(ZS::Unknown) = b.lead_status() {
        return Err(undecided(&b.t[0].1));
    }
    if b.val() != 0 {
        // a branch point only if alpha is decided to be a non-integer (m1)
        let (st, nearest) = non_integer_status(alpha);
        return match (st, nearest) {
            (ZS::NonZero, _) => Err(SErr::NotLaurent(format!(
                "a non-integer power of an argument of valuation {} has a branch point there",
                b.val()
            ))),
            (ZS::Zero, Some(n)) => pow_int(b, &n, cx),
            (_, Some(n)) => Err(undecided(
                &alpha.sub(&Konst::rat(Rational::from_integer(n))),
            )),
            (_, None) => Err(undecided(alpha)),
        };
    }
    if let Some(tb) = tb {
        if let Err(e) = cut_check("a non-integer power", Cut::Log, b, tb) {
            // the cut matters only for a non-integer exponent: an integer one is single-valued
            match non_integer_status(alpha) {
                (ZS::NonZero, _) => return Err(e),
                (ZS::Zero, _) => {}
                (_, Some(n)) => {
                    return Err(undecided(
                        &alpha.sub(&Konst::rat(Rational::from_integer(n))),
                    ))
                }
                (_, None) => return Err(undecided(alpha)),
            }
        }
    }
    let prec = b.prec.at_most(cx.p);
    let b0 = b.t[0].1.clone();
    let w = scale(b, &b0.inverse(true)?)?;
    let n = cx.alloc(prec)?;
    let g = miller(&w, alpha, n, cx)?;
    let k0 = kpow_const(&b0, alpha)?;
    scale(&Ser::from_dense(0, g, Abs(prec)), &k0)
}

fn binom(n: &Integer, m: u64) -> Rational {
    let mut c = Rational::from_integer(Integer::from(1));
    for i in 0..m {
        // revision 3: charged (q_new normalizes, q_mul multiplies; C(n, m) grows with m)
        c = q_mul(
            &c,
            &q_new(
                n.clone() - Integer::from(i as i64),
                Integer::from((i + 1) as i64),
            ),
        );
    }
    c
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
#[inline(never)]
pub fn pow_int(b: &Ser, n: &Integer, cx: &mut Ctx) -> Result<Ser, SErr> {
    if n.is_zero() {
        return Ok(Ser::exact_const(Konst::one()));
    }
    if b.is_exact_zero() {
        if n.signum() > 0 {
            return Ok(Ser::exact_zero());
        }
        return Err(SErr::DivisionByZero("0 to a negative power".into()));
    }
    if n.signum() < 0 {
        let i = inv(b, cx)?;
        return pow_int(&i, &(-n.clone()), cx);
    }
    if b.t.is_empty() {
        // O(h^p)^n = O(h^(n p)) for p > 0, else unknown
        if let Abs(q) = b.prec {
            if q > 0 {
                let np = Integer::from(q) * n.clone();
                let p = if np >= Integer::from(cx.p) {
                    cx.p
                } else {
                    NumericConversion::to_i64(&np).unwrap()
                };
                return Ok(Ser {
                    t: vec![],
                    prec: Abs(p),
                });
            }
        }
        return Err(SErr::NeedPrecision);
    }
    let vb = b.val();
    let nv_big = Integer::from(vb) * n.clone();
    if b.is_exact_monomial() {
        // exact: c^n h^(n vb), kept exact (one term) whatever P is
        // revision 7rt2 (red team 2 structure, m3): an exponent n vb above i64::MAX (n > 0 here,
        // vb > 0) lies above every working precision, so the power is O(h^P); only a pole of
        // that order is an overflow error
        let nv = match NumericConversion::to_i64(&nv_big) {
            Some(nv) => nv,
            None if vb > 0 => {
                return Ok(Ser {
                    t: vec![],
                    prec: Abs(cx.p),
                })
            }
            None => return Err(ovf()),
        };
        // revision 7rt4 (red team 4 series, H1): nv = i64::MAX is a legal exponent of an exact term
        let k = kpow_const(&b.t[0].1, &Konst::rat(Rational::from_integer(n.clone())))?;
        return Ok(Ser {
            t: vec![(nv, k)],
            prec: Exact,
        }
        .trunc(cx.p));
    }
    // n * vb >= P: keep the leading term c0^n h^(n vb) + O(h^(n vb + 1)) (m1, review 2), or
    // O(h^P) when n vb does not fit in i64
    if vb > 0 && nv_big >= Integer::from(cx.p) {
        // revision 7rt4 (red team 4 series, H1): the precision n vb + 1 is a finite Prec (revision 8
        // returned i64::MAX = EXACT for n vb = i64::MAX - 1); at n vb = i64::MAX the term has no
        // finite precision above it, so the power is O(h^(n vb)), a true claim
        return match NumericConversion::to_i64(&nv_big) {
            Some(nv) => match nv.checked_add(1) {
                Some(p1) => {
                    let k = kpow_const(&b.t[0].1, &Konst::rat(Rational::from_integer(n.clone())))?;
                    Ok(Ser {
                        t: vec![(nv, k)],
                        prec: Abs(p1),
                    })
                }
                None => Ok(Ser {
                    t: vec![],
                    prec: Abs(nv),
                }),
            },
            None => Ok(Ser {
                t: vec![],
                prec: Abs(cx.p),
            }),
        };
    }
    let nv = NumericConversion::to_i64(&nv_big).ok_or_else(ovf)?;
    let nn = NumericConversion::to_i64(n);
    if let Some(small) = nn.filter(|v| *v <= POW_BINARY_MAX) {
        // binary powering: exact canonical coefficients
        let mut acc = Ser::exact_const(Konst::one());
        for i in (0..64 - (small as u64).leading_zeros()).rev() {
            acc = mul(&acc, &acc, cx)?;
            if (small >> i) & 1 == 1 {
                acc = mul(&acc, b, cx)?;
            }
        }
        return Ok(acc);
    }
    // large n: b = h^vb (c0 + u'), result h^(n vb) (c0 + u')^n with S slots
    let bs = b.shift(-vb)?;
    // revision 7rt2: saturating (nv may be a pole order near i64::MIN)
    let slots = bs.prec.at_most(cx.p.saturating_sub(nv));
    let s = cx.alloc(slots)?;
    let c0 = bs.t[0].1.clone();
    let k0 = kpow_const(&c0, &Konst::rat(Rational::from_integer(n.clone())))?;
    // revision 4 (review 4, B1): the slices bs.t[1..] below copy pointers, 4 units per term
    charge(
        SER_TERM
            .saturating_mul(bs.t.len() as u64)
            .saturating_add(16),
    );
    let g = if c0.zero_status() == ZS::NonZero {
        // Miller with alpha = n on w = bs / c0 - 1: cost independent of n
        let w = scale(
            &Ser {
                t: bs.t[1..].to_vec(),
                prec: bs.prec,
            },
            &c0.inverse(true)?,
        )?;
        let g = miller(&w, &Konst::rat(Rational::from_integer(n.clone())), s, cx)?;
        scale(&Ser::from_dense(0, g, Abs(slots)), &k0)?
    } else {
        // undecided c0: binomial theorem, no division: sum_m C(n, m) c0^(n-m) u^m
        let u = Ser {
            t: bs.t[1..].to_vec(),
            prec: bs.prec,
        };
        let mut acc = Ser::exact_const(k0).trunc(slots);
        let mut um = Ser::exact_const(Konst::one());
        // revision 7rt1 (red team 1, m3): the loop runs in the caller's context with the working
        // precision set to the slots, so it charges the caller's Work (revision 7 built a second
        // context with a dummy `Work { used: 0, max: 0 }`, harmless only with the prototype's
        // thread-local counter; a port with `&mut Work` must pass the same Work)
        let saved_p = cx.p;
        cx.p = slots;
        // revision 7rt2: the terms m > n are zero (C(n, m) = 0), and c0^(n-m) with n - m < 0 is
        // not needed (it would invert the undecided c0), so the loop stops at m = n
        let mmax: u64 = match NumericConversion::to_i64(n) {
            Some(nn) => (s as u64).min(nn as u64 + 1),
            None => s as u64,
        };
        let r = (|| -> Result<Ser, SErr> {
            for m in 1..mmax {
                um = mul(&um, &u, cx)?;
                if um.t.is_empty() {
                    break;
                }
                let e = Integer::from((m) as i64);
                let cm = kpow_const(&c0, &Konst::rat(Rational::from_integer(n.clone() - e)))?
                    .scale(&binom(n, m));
                acc = add(&acc, &scale(&um, &cm)?, cx)?;
            }
            // revision 7rt2 (red team 2, S-B3): the partial sum of (c0 + u')^n is truncated at
            // the slots and carries precision `slots`, as Miller's branch does
            // (from_dense(0, g, slots)). Revision 7rt1 returned `acc.trunc(slots)`, which keeps
            // an exact series of at most EXACT_KEEP_TERMS terms exact: a power of a base with an
            // undecided constant term claimed to be the whole polynomial, so after a pole shift
            // the driver saw no loss and the requested terms m >= slots were dropped as zeros.
            Ok(Ser {
                t: acc.t,
                prec: acc.prec.min(Abs(slots)),
            }
            .trunc_hard(slots))
        })();
        cx.p = saved_p;
        r?
    };
    g.shift(nv)
}

fn deriv(b: &Ser) -> Result<Ser, SErr> {
    // d/dh sum c_e h^e
    charge(SER_TERM.saturating_mul(b.t.len() as u64).saturating_add(16));
    // revision 7rt2: checked (an exponent i64::MIN has no derivative term in i64)
    let mut t = Vec::with_capacity(b.t.len());
    for (e, k) in b.t.iter().filter(|(e, _)| *e != 0) {
        t.push((cadd(*e, -1)?, k.scale(&qi(*e))));
    }
    check()?;
    Ok(Ser {
        t,
        prec: b.prec.add(-1)?,
    })
}

/// the preconditions of `integrate`: q must have valuation >= 0
fn integrate_check(q: &Ser) -> Result<(), SErr> {
    // revision 7rt3 (red team 3 series, H1): an empty integrand O(h^q) with q < 0 may hold an h^-1
    // term (a logarithm), so its integral is not known to be a Laurent series: ask for precision
    if q.t.is_empty() && q.prec < Abs(0) {
        return Err(SErr::NeedPrecision);
    }
    if !q.t.is_empty() && q.val() < 0 {
        return match q.lead_status() {
            Some(ZS::Unknown) => Err(undecided(&q.t[0].1)),
            _ => Err(SErr::NotLaurent(
                "the function has a logarithmic branch point there".into(),
            )),
        };
    }
    Ok(())
}

/// integral with constant term g0; q must have valuation >= 0
fn integrate(q: &Ser, g0: Konst, cx: &mut Ctx) -> Result<Ser, SErr> {
    integrate_check(q)?;
    let prec = q.prec.add(1)?.at_most(cx.p);
    cx.work
        .charge(SER_TERM.saturating_mul(q.t.len() as u64 + 1))?;
    let mut t = Vec::with_capacity(q.t.len() + 1);
    if !g0.is_zero() {
        t.push((0, g0));
    }
    for (e, k) in &q.t {
        // revision 7rt4: checked (an exact integrand may hold a term at i64::MAX, above prec)
        let e1 = match e.checked_add(1) {
            Some(v) if v < prec => v,
            _ => break,
        };
        t.push((e1, k.scale(&qrec(e1))));
    }
    check()?;
    Ok(Ser { t, prec: Abs(prec) })
}

// ------------------------------------------------------------------------------------------
// S1 request C-1 (docs/workstreams/m23_golay/REQUESTS.md, section 2): branch cuts for a real
// variable. The variable is real (h = x - a real, or 1/x at infinity). Where the constant term c0
// of the argument b of a branch function lies on an open cut of the function, the expansion
// through the principal value at c0 is the function of a real variable only when b stays on the
// cut: when every other coefficient of b is a real multiple of the cut's direction (real for the
// cuts on the real axis, purely imaginary for those of arctan and arcsinh). Otherwise b leaves the
// cut for x != a, and the series is the function on one side of the point or on neither
// (`log(-1 - I*x^2)` at 0 is off by 2 pi I on both sides, gp's formal series alike). The rule:
// the engine expands there only when it decides that b stays on the cut, from the classes of the
// coefficients (`Tail`, which covers the coefficients beyond the precision too) or, for an exact
// series, from every coefficient; a coefficient decided off the cut's direction is `NotLaurent`
// ("leaves a branch cut at the point"); a real constant whose zero test is undecided (the
// imaginary part of a coefficient, the distance of c0 from an end of the cut) is `UndecidedZero`;
// anything else undecided is `NotLaurent` saying that the engine cannot decide it. A bounded
// check never decides that b stays on the cut.
// ------------------------------------------------------------------------------------------

/// S1 C-1 minor items (semantics review m3): the largest constant (nodes of its tree) whose class
/// `Tail::scaled_enclosed` reads from an enclosure
pub const SCALE_ENCLOSE_TREE: u64 = 256;

/// S1 request C-1: what is known about the coefficients of a value's series of every exponent
/// other than 0 (all of them, beyond the precision too), for a real variable: `re`, they are all
/// real; `im`, they are all purely imaginary. Both hold for a constant (it has none). Built node by
/// node from the structural classes of the constants (`Konst::struct_real`, `struct_imag`): a
/// sufficient condition, not a decision procedure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tail {
    pub re: bool,
    pub im: bool,
}

impl Tail {
    pub const CONST: Tail = Tail { re: true, im: true };
    pub const NONE: Tail = Tail {
        re: false,
        im: false,
    };
    pub const REAL: Tail = Tail {
        re: true,
        im: false,
    };
    pub const IMAG: Tail = Tail {
        re: false,
        im: true,
    };
    /// the class of a sum of series of these classes
    pub fn and(self, o: Tail) -> Tail {
        Tail {
            re: self.re && o.re,
            im: self.im && o.im,
        }
    }
    /// the class of c times a series of this class, for a constant c
    pub fn scaled(self, c: &Konst) -> Tail {
        let (r, i) = (c.struct_real(), c.struct_imag());
        Tail {
            re: (r && self.re) || (i && self.im),
            im: (r && self.im) || (i && self.re),
        }
    }
    /// S1 C-1 minor items (semantics review m3): `scaled`, with the class of a constant whose
    /// structure shows none decided by one complex enclosure at 64 bits (real when its imaginary
    /// part is an exact 0, which only exact arithmetic gives: `log(-1)^2`), for a constant of at
    /// most `SCALE_ENCLOSE_TREE` nodes (charged as `cdecide`; a larger one keeps `scaled`)
    pub fn scaled_enclosed(self, c: &Konst) -> Tail {
        if !(self.re || self.im) || c.struct_real() || c.struct_imag() {
            return self.scaled(c);
        }
        if c.tree() > SCALE_ENCLOSE_TREE {
            return self.scaled(c);
        }
        let (r, i) = match crate::series_ball::cball_konst(c, 64) {
            Ok(b) if b.im.is_exact_zero() => (true, false),
            Ok(b) if b.re.is_exact_zero() => (false, true),
            _ => return self.scaled(c),
        };
        Tail {
            re: (r && self.re) || (i && self.im),
            im: (r && self.im) || (i && self.re),
        }
    }
    /// the full class of a product of two series of these full classes
    pub fn times(self, o: Tail) -> Tail {
        Tail {
            re: (self.re && o.re) || (self.im && o.im),
            im: (self.re && o.im) || (self.im && o.re),
        }
    }
}

/// S1 request C-1: the constant term of s is known (it lies below the precision)
fn known_constant_term(s: &Ser) -> bool {
    s.prec == Exact || s.prec > Abs(0)
}

/// S1 request C-1: the full class of s (its constant term too) from the class `t` of its other
/// coefficients
fn full_class(s: &Ser, t: Tail) -> Tail {
    if !known_constant_term(s) {
        return Tail::NONE;
    }
    let c0 = s.coef(0);
    Tail {
        re: t.re && c0.struct_real(),
        im: t.im && c0.struct_imag(),
    }
}

/// S1 request C-1: the open cuts of the branch functions (the principal branches of the module
/// conventions of `series.rs`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cut {
    /// log and the non-integer powers: (-oo, 0)
    Log,
    /// arcsin, arccos, arctanh: (-oo, -1) and (1, oo)
    AsinLike,
    /// arccosh: (-oo, -1) and (-1, 1)
    Acosh,
    /// arctan, arcsinh: (-I oo, -I) and (I, I oo)
    Imag,
}

/// where a constant term lies with respect to an open cut
enum Pos {
    Off,
    On,
    Undecided(SErr),
}

/// what the coefficients of an argument on a cut do
enum Verdict {
    Stays,
    Leaves,
    Unknown(SErr),
}

fn leaves_cut(name: &str) -> SErr {
    SErr::NotLaurent(format!(
        "{name}: the argument leaves a branch cut at the point (its constant term lies on the \
         cut, and a coefficient that is not a real multiple of the cut's direction takes it off; \
         the engine expands a branch function on its cut only for an argument that stays there)"
    ))
}

fn cannot_decide_cut(name: &str) -> SErr {
    SErr::NotLaurent(format!(
        "{name}: the engine cannot decide that the argument stays off or on a branch cut at the \
         point (where its constant term lies, or whether its coefficients keep it on the cut); it \
         expands a branch function on its cut only for an argument decided to stay there"
    ))
}

/// the sign of a real ball minus q, decided only when it excludes 0 or is exactly 0
fn rsign(b: &crate::series_ball::Ball, q: i64) -> Option<i32> {
    if q == 0 {
        b.sign()
    } else {
        b.sign_minus(q)
    }
}

/// S1 request C-1: the position of t (a real number, the real ball `x`; for the cuts of the
/// imaginary axis, c0 = t*I) with respect to the open cut: Some(true) on it, Some(false) off it or
/// at an end (a branch point, left to the branch-point rules), None undecided
fn real_position(cut: Cut, x: &crate::series_ball::Ball) -> Option<bool> {
    match cut {
        Cut::Log => match rsign(x, 0)? {
            -1 => Some(true),
            _ => Some(false),
        },
        Cut::AsinLike | Cut::Imag => {
            let s1 = rsign(x, 1);
            let s2 = rsign(x, -1);
            if s1 == Some(1) || s2 == Some(-1) {
                return Some(true);
            }
            if s1 == Some(0) || s2 == Some(0) || (s1 == Some(-1) && s2 == Some(1)) {
                return Some(false);
            }
            None
        }
        Cut::Acosh => match rsign(x, 1)? {
            -1 => match rsign(x, -1) {
                // -1 is a branch point inside the cut: a leaving argument is NotLaurent there
                // too, and one that stays meets the branch-point rule of the integrand
                Some(0) => Some(false),
                _ => Some(true),
            },
            _ => Some(false),
        },
    }
}

/// S1 request C-1: the constant whose zero test decides where t lies (t minus the end of the cut
/// that its ball could not separate it from), for the `UndecidedZero` report. S1 C-1 minor items:
/// a fault of the enclosures (the budget) is returned as the fault
fn position_undecided(cut: Cut, t: &Konst) -> Result<SErr, SErr> {
    let ends: &[i64] = match cut {
        Cut::Log => &[0],
        Cut::Acosh => &[1],
        Cut::AsinLike | Cut::Imag => &[1, -1],
    };
    for e in ends {
        let d = t.sub(&Konst::int(*e));
        let sep = crate::series_ball::cdecide(&d, |b| b.re.sign().filter(|s| *s != 0));
        check()?;
        if sep.is_none() {
            return Ok(undecided(&d));
        }
    }
    Ok(undecided(t))
}

/// S1 request C-1: where the constant term c0 of an argument lies with respect to the open cuts of
/// `cut`. A constant term that depends on a free parameter is off the cut for generic values of
/// the parameters when it is a nonconstant function of them (`Konst::generic_nonconstant`: a
/// nonconstant polynomial in the parameters, decided by its groups as the zero test decides, or,
/// since the C-1 minor items (regressions review m1), a Laurent polynomial in one kernel of an
/// analytic function of such a function, `sin(a)` or `log(a)`); otherwise undecided. The
/// parameters are generic complex numbers (the module conventions), so the exception set has no
/// interior but can hold every real value of a parameter (`log(-y^2 + I*x)` at 0).
fn position(name: &str, cut: Cut, c0: &Konst) -> Result<Pos, SErr> {
    if c0.t.sz.any_indet {
        let generic = c0.generic_nonconstant(true);
        check()?;
        return Ok(if generic {
            Pos::Off
        } else {
            Pos::Undecided(cannot_decide_cut(name))
        });
    }
    let real_axis = cut != Cut::Imag;
    // structural exits: a real c0 is off the imaginary axis's cuts (0 is not on them); a
    // nonzero imaginary c0 is off the real axis
    if !real_axis && c0.struct_real() {
        return Ok(Pos::Off);
    }
    if real_axis && c0.struct_imag() && !c0.is_zero() {
        return Ok(Pos::Off);
    }
    // the coordinate that must vanish for c0 to lie on the axis of the cut, and the other one,
    // which locates it on the axis
    let exact_axis = if real_axis {
        c0.struct_real()
    } else {
        c0.struct_imag()
    };
    let decided = crate::series_ball::cdecide(c0, |b| {
        let (off_axis, along) = if real_axis {
            (&b.im, &b.re)
        } else {
            (&b.re, &b.im)
        };
        let on_axis = exact_axis || off_axis.is_exact_zero();
        if !on_axis {
            if matches!(off_axis.sign(), Some(1) | Some(-1)) {
                return Some(false);
            }
            // undecided distance from the axis: off when its position along the axis is off
            // the cut (or at an end of it) whatever the other coordinate is
            return match real_position(cut, along) {
                Some(false) => Some(false),
                _ => None,
            };
        }
        real_position(cut, along)
    });
    check()?;
    Ok(match decided {
        Some(true) => Pos::On,
        Some(false) => Pos::Off,
        None => {
            let (a, bb) = c0.split_imag();
            check()?;
            let parts_real = a.struct_real() && bb.struct_real();
            if exact_axis {
                // on the axis: the position along it is undecided; reported as a real constant
                // when c0 is one (real axis) or is t*I with t written out (imaginary axis)
                let t = if real_axis {
                    Some(c0.clone())
                } else if a.is_zero() && bb.struct_real() {
                    Some(bb)
                } else {
                    None
                };
                match t {
                    Some(t) => Pos::Undecided(position_undecided(cut, &t)?),
                    // S1 C-1 minor items (regressions review m3): c0 = i t on the imaginary axis
                    // with t not written out (`sqrt(-1)`): its distance from the ends +-i of the
                    // cuts is the real constant 1 + c0^2 = 1 - t^2, whose zero test is the
                    // undecided one (before C-1, the integrand's 1 + b^2 named it)
                    None => Pos::Undecided(undecided(&c0.mul(c0).add(&Konst::one()))),
                }
            } else if parts_real {
                // the distance from the axis is undecided
                Pos::Undecided(undecided(if real_axis { &bb } else { &a }))
            } else {
                Pos::Undecided(cannot_decide_cut(name))
            }
        }
    })
}

/// S1 request C-1: whether the coefficient c is a real multiple of the cut's direction
/// (Some(true)), decided not to be (Some(false)), or undecided (Err: the report)
fn coefficient_direction(name: &str, cut: Cut, c: &Konst) -> Result<Result<bool, SErr>, SErr> {
    let real_axis = cut != Cut::Imag;
    if (real_axis && c.struct_real()) || (!real_axis && c.struct_imag()) {
        return Ok(Ok(true));
    }
    let d = crate::series_ball::cdecide(c, |b| {
        let off = if real_axis { &b.im } else { &b.re };
        match off.sign() {
            Some(0) => Some(true),
            Some(_) => Some(false),
            None => None,
        }
    });
    check()?;
    Ok(match d {
        Some(v) => Ok(v),
        None => {
            let (a, bb) = c.split_imag();
            check()?;
            if a.struct_real() && bb.struct_real() {
                Err(undecided(if real_axis { &bb } else { &a }))
            } else {
                Err(cannot_decide_cut(name))
            }
        }
    })
}

/// S1 request C-1: whether the argument b (constant term on a cut) stays on it. Every stored
/// coefficient other than the constant term is checked: one decided off the cut's direction
/// decides that b leaves; all decided in it decide that b stays only when b is exact (every
/// coefficient is stored). A truncated b whose stored coefficients stay is undecided (its terms
/// beyond the precision are not known; `Tail` covers them when it can).
fn coefficient_verdict(name: &str, cut: Cut, b: &Ser) -> Result<Verdict, SErr> {
    let mut first: Option<SErr> = None;
    for (e, c) in &b.t {
        if *e == 0 {
            continue;
        }
        match coefficient_direction(name, cut, c)? {
            Ok(true) => {}
            Ok(false) => return Ok(Verdict::Leaves),
            Err(err) => {
                if first.is_none() {
                    first = Some(err);
                }
            }
        }
    }
    Ok(match first {
        None if b.prec == Exact => Verdict::Stays,
        None => Verdict::Unknown(cannot_decide_cut(name)),
        Some(err) => Verdict::Unknown(err),
    })
}

/// S1 request C-1: the branch-cut rule at one application of a branch function `name` with the
/// open cuts `cut` to the argument b of class `tb` (b has valuation >= 0; for log and the powers,
/// valuation 0 with a decided nonzero constant term). Ok when b's class keeps it in the cut's
/// direction, when its constant term is decided off the cut, or when it is decided to stay on it;
/// otherwise the refusal (`NotLaurent` when it leaves the cut, `UndecidedZero` or `NotLaurent`
/// when that cannot be decided).
fn cut_check(name: &str, cut: Cut, b: &Ser, tb: Tail) -> Result<(), SErr> {
    let in_direction = if cut == Cut::Imag { tb.im } else { tb.re };
    if in_direction || b.t.is_empty() || !known_constant_term(b) {
        return Ok(());
    }
    let c0 = b.coef(0);
    match position(name, cut, &c0)? {
        Pos::Off => Ok(()),
        Pos::On => match coefficient_verdict(name, cut, b)? {
            Verdict::Stays => Ok(()),
            Verdict::Leaves => Err(leaves_cut(name)),
            Verdict::Unknown(e) => Err(e),
        },
        Pos::Undecided(e) => match coefficient_verdict(name, cut, b)? {
            Verdict::Stays => Ok(()),
            _ => Err(e),
        },
    }
}

/// arctan, arctanh, arcsin, arccos, arcsinh, arccosh and erf of a series: f(b0) plus the integral
/// of the integrand Q (b' (1 + b^2)^-1 for arctan, and so on).
///
/// S1 port, the owner's two rules (decision of 2026-11-15; review 9 B1 and B2). Deviations from
/// the frozen design, which formed the integrand first for every argument and applied f to b0
/// before it:
/// 1. An argument whose series is an exact constant c (`x^0`, `x/x` at 0, `sign(x^2+1)`, a
///    constant factor 0, `arcsin(x - x)`, ...) gives the exact constant f(c): no expansion and no
///    integrand. The integrand of arcsin, arccos, arccosh and arcsinh is singular at +-1 (+-i),
///    so the design's order refused arcsin(x^0) with `DivisionByZero("0 to a negative power")`
///    where its value is pi/2 (B1). A pole of f(c) itself is `DivisionByZero` from `kapply`
///    (arctanh(+-1), arctan(+-i)): there the input applies f at an exact pole.
/// 2. For a non-constant argument the integrand is formed and its valuation is checked (the
///    checks of `integrate`) before f is applied to b0. At a branch point b0 of f the integrand
///    has a pole (arctanh, arctan) or a non-integer power of a series of positive valuation
///    (arcsin, arccos, arcsinh, arccosh), so the result is the typed `NotLaurent` (or a request
///    for precision when the integrand's valuation is undecided), never `DivisionByZero` from the
///    pole of f(b0) (B2: `atanh(1 + x)` at 0).
///
/// S1 request C-1: `tb` is the class of b's coefficients (`Tail`). After the exact-constant rule
/// and the valuation check, an argument whose constant term lies on an open cut of f and that
/// leaves the cut is refused (`cut_check`). The result carries the class of its own coefficients:
/// those of the integrand Q, the product of b' (whose class is b's tail class) and the multiplier
/// (1 +- b^2)^-1, (1 -+ b^2)^(-1/2) or (b - 1)^(-1/2) (b + 1)^(-1/2), whose class is that of its
/// constant term when b's own coefficients are all real or all purely imaginary (then 1 +- b^2
/// has real coefficients). erf is entire: real for a real b, odd.
pub fn atan_like(op: UnaryOp, b: &Ser, tb: Tail, cx: &mut Ctx) -> Result<(Ser, Tail), SErr> {
    if op != UnaryOp::Erf {
        if let Some(c) = exact_constant(b) {
            return Ok((Ser::exact_const(kapply(op, &c)?).trunc(cx.p), Tail::CONST));
        }
    }
    let name = format!("{:?}", op).to_lowercase();
    require_nonneg_val(b, &name)?;
    match op {
        UnaryOp::Erf => {}
        UnaryOp::Arctan | UnaryOp::Arcsinh => cut_check(&name, Cut::Imag, b, tb)?,
        UnaryOp::Arccosh => cut_check(&name, Cut::Acosh, b, tb)?,
        _ => cut_check(&name, Cut::AsinLike, b, tb)?,
    }
    let fb = full_class(b, tb);
    // b's coefficients all real or all purely imaginary: 1 +- b^2 has real coefficients
    let fb_real = fb.re || fb.im;
    let mut mcls: Option<Tail> = None;
    let b0 = b.coef(0);
    let d = deriv(b)?;
    let one = Ser::exact_const(Konst::one());
    let b2 = mul(b, b, cx)?;
    let half = Konst::rat(Rational::new(Integer::from(-1), Integer::from(2)).unwrap());
    let q = match op {
        UnaryOp::Arctan => {
            mcls = fb_real.then_some(Tail::REAL);
            mul(&d, &inv(&add(&one, &b2, cx)?, cx)?, cx)?
        }
        UnaryOp::Arctanh => {
            mcls = fb_real.then_some(Tail::REAL);
            mul(&d, &inv(&add(&one, &neg(&b2), cx)?, cx)?, cx)?
        }
        UnaryOp::Arcsin | UnaryOp::Arccos => {
            let m = pow_const(&add(&one, &neg(&b2), cx)?, &half, None, cx)?;
            mcls = (fb_real && known_constant_term(&m)).then(|| Tail::REAL.scaled(&m.coef(0)));
            let q = mul(&d, &m, cx)?;
            if op == UnaryOp::Arccos {
                neg(&q)
            } else {
                q
            }
        }
        UnaryOp::Arcsinh => {
            let m = pow_const(&add(&one, &b2, cx)?, &half, None, cx)?;
            mcls = (fb_real && known_constant_term(&m)).then(|| Tail::REAL.scaled(&m.coef(0)));
            mul(&d, &m, cx)?
        }
        UnaryOp::Arccosh => {
            // principal branch: 1/(sqrt(b-1) sqrt(b+1))
            let m1 = pow_const(&add(b, &neg(&one), cx)?, &half, None, cx)?;
            let p1 = pow_const(&add(b, &one, cx)?, &half, None, cx)?;
            // b - 1 and b + 1 have real coefficients only when b has (a real b)
            mcls = (fb.re && known_constant_term(&m1) && known_constant_term(&p1)).then(|| {
                Tail::REAL
                    .scaled(&m1.coef(0))
                    .times(Tail::REAL.scaled(&p1.coef(0)))
            });
            mul(&d, &mul(&m1, &p1, cx)?, cx)?
        }
        UnaryOp::Erf => {
            // 2/sqrt(pi) exp(-b^2) b'. S1 fix-up round 5 (review 5 semantics, B1): when the
            // variable is named pi, the constant pi is written arccos(-1) (the kernel that
            // `kapply` keeps then), so that the result does not name the variable
            let pi = if crate::series_const::var_is_named_pi() {
                kapply(UnaryOp::Arccos, &Konst::int(-1))?
            } else {
                konst_of(&Expr::symbol("pi"))?
            };
            let c = kpow_const(&pi, &half)?.scale(&qi(2));
            scale(&mul(&d, &exp(&neg(&b2), cx)?, cx)?, &c)?
        }
        _ => unreachable!(),
    };
    // rule 2: the integrand's own refusals (a pole: the logarithmic branch point; an empty
    // truncated integrand of negative precision: more precision) come before f(b0)
    integrate_check(&q)?;
    let g0 = kapply(op, &b0)?;
    let cls = if op == UnaryOp::Erf {
        if fb.re {
            Tail::REAL
        } else if fb.im {
            Tail::IMAG
        } else {
            Tail::NONE
        }
    } else {
        // b' has the class of b's tail (its coefficients are e c_e)
        match mcls {
            Some(m) => Tail {
                re: tb.re,
                im: tb.im,
            }
            .times(m),
            None => Tail::NONE,
        }
    };
    Ok((integrate(&q, g0, cx)?, cls))
}

/// abs and sign assume a real variable and a real point (all coefficients rational)
fn abs_sign(op: UnaryOp, b: &Ser) -> Result<Ser, SErr> {
    if b.t.is_empty() {
        if b.prec == Exact {
            return Ok(Ser::exact_const(Konst::zero()));
        }
        return Err(SErr::NeedPrecision);
    }
    let all_rat = b.t.iter().all(|(_, k)| k.as_rational().is_some());
    if !all_rat {
        return Err(SErr::Unsupported(
            format!("{:?} of a series with non-rational coefficients", op).to_lowercase(),
        ));
    }
    if b.val() % 2 != 0 {
        return Err(SErr::NotLaurent(
            format!(
                "{:?} of an argument with a zero of odd order is not analytic there",
                op
            )
            .to_lowercase(),
        ));
    }
    let s = b.t[0].1.as_rational().unwrap().numerator().signum() as i64;
    Ok(match op {
        UnaryOp::Abs => scale(b, &Konst::int(s))?,
        _ => Ser::exact_const(Konst::int(s)),
    })
}

/// the constant of an exact series with no term other than h^0 (m1: route x^(x-x+2) to the
/// constant-exponent path)
fn exact_constant(s: &Ser) -> Option<Konst> {
    if s.prec != Exact {
        return None;
    }
    match s.t.as_slice() {
        [] => Some(Konst::zero()),
        [(0, k)] => Some(k.clone()),
        _ => None,
    }
}

// revision 6: never inlined, so no frame on the recursive path of expand holds its locals
/// S1 request C-1: with the class of the result's coefficients (`pow_int_class`, `pow_class`)
#[inline(never)]
fn pow_by_const(b: Ser, tb: Tail, alpha: Konst, cx: &mut Ctx) -> Result<(Ser, Tail), SErr> {
    if let Some(q) = alpha.as_rational() {
        if q.denominator().is_one() {
            let cls = pow_int_class(&b, tb, q.numerator());
            return Ok((pow_int(&b, q.numerator(), cx)?, cls));
        }
    }
    let v = pow_const(&b, &alpha, Some(tb), cx)?;
    let cls = pow_class(&b, tb, &alpha, &v);
    Ok((v, cls))
}

/// S1 request C-1: the class of b^n for an integer n: real for a real b, and for an imaginary b
/// real or imaginary by the parity of n
fn pow_int_class(b: &Ser, tb: Tail, n: &Integer) -> Tail {
    if n.is_zero() {
        return Tail::CONST;
    }
    let fb = full_class(b, tb);
    if fb.re {
        Tail::REAL
    } else if fb.im {
        if n.is_odd() {
            Tail::IMAG
        } else {
            Tail::REAL
        }
    } else {
        Tail::NONE
    }
}

/// S1 request C-1: the class of v = b^alpha (pow_const: v = c0^alpha (1 + w)^alpha with w = u/c0).
/// When b's coefficients are all real or all purely imaginary, w has real coefficients, so for a
/// real alpha v's coefficients have the class of its constant term c0^alpha
fn pow_class(b: &Ser, tb: Tail, alpha: &Konst, v: &Ser) -> Tail {
    let fb = full_class(b, tb);
    if (fb.re || fb.im) && alpha.struct_real() && known_constant_term(v) {
        Tail::REAL.scaled(&v.coef(0))
    } else {
        Tail::NONE
    }
}

/// S1 fix-up round 3 (review 3, resources m3): a function name as an error message names it, at
/// most 40 characters and its length (a 100 KB name made a 100 KB message; the bounded `Display`
/// of the other variants, `point_text` and the `UndecidedZero` text, do the same for values)
pub fn name_text(name: &str) -> String {
    const SHOWN: usize = 40;
    if name.len() <= 2 * SHOWN {
        return name.to_string();
    }
    let cut = name
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|i| *i <= SHOWN)
        .last()
        .unwrap_or(0);
    format!("{}... (a name of {} bytes)", &name[..cut], name.len())
}

// revision 4 (review 4, m3): dependence on the variable, computed once per node by a pre-pass
// (expand used to call contains_symbol at every node: quadratic on left-deep chains)
// revision 5 (review 5, m1): the map is checked before descending, so a DAG input (a REPL
// variable built by `e = e + e`) is walked once per distinct node; the pass also records each
// node's depth (the input depth sets the allowed kernel depth, section 2.15)
thread_local! {
    /// per distinct input node: (depends on the variable, depth, and since revision 7rt3 (red team 3
    /// structure, H2) its tree size and the occurrences of the variable in its tree, saturating)
    static DEP: std::cell::RefCell<std::collections::HashMap<usize, (bool, u32, u64, u64)>> = std::cell::RefCell::new(std::collections::HashMap::new());
    /// revision 5 (review 5, m2): whether the point is the exact zero, recorded by the driver
    pub static POINT_ZERO: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
fn addr(e: &Expr) -> usize {
    std::ptr::from_ref(e).addr()
}
fn dep_pass(e: &Expr, var: &Symbol) -> (bool, u32) {
    let r = dep_pass4(e, var);
    (r.0, r.1)
}
fn dep_pass4(e: &Expr, var: &Symbol) -> (bool, u32, u64, u64) {
    // revision 5: an explicit-stack post-order walk (no recursion, whatever the input depth); a node
    // already in the map is not descended into, so a DAG costs its distinct nodes
    fn children(e: &Expr) -> Vec<&Expr> {
        match e {
            Expr::Binary(_, l, r) => vec![&**l, &**r],
            Expr::Unary(_, i) => vec![&**i],
            Expr::Function(_, args) => args.iter().map(|a| &**a).collect(),
            _ => vec![],
        }
    }
    let known = |x: &Expr| DEP.with(|m| m.borrow().contains_key(&addr(x)));
    if known(e) {
        return DEP.with(|m| m.borrow()[&addr(e)]);
    }
    let mut stack: Vec<(&Expr, bool)> = vec![(e, false)];
    while let Some((x, expanded)) = stack.pop() {
        if known(x) {
            continue;
        }
        if !expanded {
            stack.push((x, true));
            for c in children(x) {
                if !known(c) {
                    stack.push((c, false));
                }
            }
            continue;
        }
        let v = match x {
            Expr::Symbol(s) => (s == var, 1, 1u64, if s == var { 1u64 } else { 0 }),
            _ => {
                let mut dep = false;
                let mut dp = 0;
                let (mut tr, mut xo) = (1u64, 0u64);
                for c in children(x) {
                    let cv = DEP.with(|m| m.borrow()[&addr(c)]);
                    dep |= cv.0;
                    dp = dp.max(cv.1);
                    tr = tr.saturating_add(cv.2);
                    xo = xo.saturating_add(cv.3);
                }
                (dep, if children(x).is_empty() { 1 } else { 1 + dp }, tr, xo)
            }
        };
        DEP.with(|m| m.borrow_mut().insert(addr(x), v));
    }
    DEP.with(|m| m.borrow()[&addr(e)])
}
/// revision 5: the depth of an input node (from the pre-pass), for kernels made from input nodes
pub fn input_depth(e: &Expr) -> Option<u32> {
    DEP.with(|m| m.borrow().get(&addr(e)).map(|v| v.1))
}
/// revision 7rt3: the tree size of an input node (from the pre-pass)
pub fn input_tree(e: &Expr) -> Option<u64> {
    DEP.with(|m| m.borrow().get(&addr(e)).map(|v| v.2))
}
fn input_xocc(e: &Expr) -> u64 {
    DEP.with(|m| m.borrow().get(&addr(e)).map(|v| v.3).unwrap_or(u64::MAX))
}
fn depends(e: &Expr) -> bool {
    DEP.with(|m| m.borrow().get(&addr(e)).expect("dependence pre-pass").0)
}

// ------------------------------------------------------------------------------------------
// revision 7rt3 (red team 3 structure, H2): the value at the point of a regular node.
// A node is regular when every node of its subtree has a series of decided valuation >= 0 with a
// positive precision, and every series divisor (the inverse in a `/`, the base of a negative power)
// has valuation 0. Then the subtree is analytic at the point, every division in it is by a
// constant that is nonzero there, and the node with the variable replaced by the point (its
// "substitution") is an expression whose value is the series' constant term b0. When b0's normal
// form renders as a tree much larger than the substitution (a power or a product of a sum that
// names a kernel twice: (1 + K)^2 = 1 + 2 K + K^2), b0 gets the substitution as its rendering hint,
// so a kernel made from b0 (exp(b0), 1/b0, b0^alpha, ...) and a result coefficient b0 render it,
// as HEAD's result renders the input with the point substituted. The normal form (the key, the zero
// test, the arithmetic) is unchanged. A removable singularity is never substituted: its subtree has
// a series divisor of valuation above 0 (`sin(x)/x` at 0), so it is not regular.
// ------------------------------------------------------------------------------------------
/// a node with the point substituted for the variable: (node, its tree size, its depth)
type Subst = (std::sync::Arc<Expr>, u64, u32);
thread_local! {
    /// the substitution of each distinct input node that depends on the variable: (node, tree, depth)
    static SUBST: std::cell::RefCell<std::collections::HashMap<usize, Subst>> = std::cell::RefCell::new(std::collections::HashMap::new());
    /// the point as one shared node, with its tree and depth (None at infinity)
    static SUBST_POINT: std::cell::RefCell<Option<(std::sync::Arc<Expr>, u64, u32)>> = const { std::cell::RefCell::new(None) };
}
/// the rendered tree of b0 must exceed the substitution's estimate by more than this before a hint
/// is set, so that ordinary results keep their expanded normal form (revision 7rt4: a margin;
/// revision 8 asked for twice the estimate plus this, which with over-counted kernels fired where the
/// exact trees did not differ)
pub const HINT_MIN: u64 = 64;
/// per new node of a substitution: the node (96 bytes) and its memo entry (at most about 80 bytes,
/// more while the table grows): 2 bytes per unit
pub const SUBST_NODE: u64 = 96;

/// a series whose constant term is known and whose valuation is decided >= 0
fn regular_val(s: &Ser) -> bool {
    if s.prec <= Abs(0) {
        return false;
    }
    match s.t.first() {
        None => true,
        Some((e, _)) => *e >= 0,
    }
}
fn input_depth_or(e: &Expr) -> u32 {
    input_depth(e).unwrap_or(u32::MAX)
}
/// the substitution of an input node: an explicit-stack post-order copy of the nodes that depend on
/// the variable (memoized per node, so a DAG costs its distinct nodes once per expansion); a child
/// free of the variable is the input's own Arc, the variable the point's one node
fn subst_at(n: &Expr) -> Result<(std::sync::Arc<Expr>, u64, u32), SErr> {
    use std::sync::Arc;
    let (pt, ptree, pdepth) = match SUBST_POINT.with(|c| c.borrow().clone()) {
        Some(v) => v,
        None => return Err(SErr::Unsupported("substitution at infinity".into())),
    };
    let known = |x: &Expr| SUBST.with(|m| m.borrow().contains_key(&addr(x)));
    let mut stack: Vec<(&Expr, bool)> = vec![(n, false)];
    while let Some((x, done)) = stack.pop() {
        if known(x) {
            continue;
        }
        if !done {
            stack.push((x, true));
            match x {
                Expr::Binary(_, l, r) => {
                    for c in [&**r, &**l] {
                        if depends(c) && !known(c) {
                            stack.push((c, false));
                        }
                    }
                }
                Expr::Unary(_, i) => {
                    if depends(i) && !known(i) {
                        stack.push((i, false));
                    }
                }
                Expr::Function(_, args) => {
                    for a in args.iter() {
                        if depends(a) && !known(a) {
                            stack.push((a, false));
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        let child = |c: &Arc<Expr>| -> (Arc<Expr>, u64, u32) {
            if depends(c) {
                SUBST.with(|m| m.borrow()[&addr(c)].clone())
            } else {
                (
                    c.clone(),
                    input_tree(c).unwrap_or(u64::MAX),
                    input_depth_or(c),
                )
            }
        };
        let v = match x {
            Expr::Symbol(_) => (pt.clone(), ptree, pdepth),
            Expr::Binary(op, l, r) => {
                cx_charge(SUBST_NODE)?;
                let (a, b) = (child(l), child(r));
                (
                    Arc::new(Expr::Binary(*op, a.0, b.0)),
                    1u64.saturating_add(a.1).saturating_add(b.1),
                    1 + a.2.max(b.2),
                )
            }
            Expr::Unary(op, i) => {
                cx_charge(SUBST_NODE)?;
                let a = child(i);
                (
                    Arc::new(Expr::Unary(*op, a.0)),
                    a.1.saturating_add(1),
                    a.2.saturating_add(1),
                )
            }
            Expr::Function(name, args) => {
                cx_charge(SUBST_NODE.saturating_add(node_payload_units(x)))?;
                let cs: Vec<_> = args.iter().map(child).collect();
                let tr = cs.iter().fold(1u64, |t, c| t.saturating_add(c.1));
                let dp = 1 + cs.iter().map(|c| c.2).max().unwrap_or(0);
                (
                    Arc::new(Expr::Function(
                        name.clone(),
                        cs.into_iter().map(|c| c.0).collect(),
                    )),
                    tr,
                    dp,
                )
            }
            _ => unreachable!("a leaf that depends on the variable is the variable"),
        };
        SUBST.with(|m| m.borrow_mut().insert(addr(x), v));
    }
    Ok(SUBST.with(|m| m.borrow()[&addr(n)].clone()))
}
fn cx_charge(w: u64) -> Result<(), SErr> {
    charge(w);
    check()
}
/// the tree of an input node's substitution, from the pre-pass (O(1))
fn subst_tree_estimate(n: &Expr) -> u64 {
    let ptree = SUBST_POINT
        .with(|c| c.borrow().as_ref().map(|v| v.1))
        .unwrap_or(u64::MAX);
    input_tree(n)
        .unwrap_or(u64::MAX)
        .saturating_add(input_xocc(n).saturating_mul(ptree.saturating_sub(1)))
}
/// a constant without kernels (a rational) needs no hint
fn is_rational_k(c: &Konst) -> bool {
    c.t.sz.factors == 0
}
/// regular node n with series s: hint its constant term when its normal form's tree is much larger
fn maybe_hint(n: &Expr, s: &Ser) -> Result<(), SErr> {
    if s.prec <= Abs(0) {
        return Ok(());
    }
    let b0 = match s.t.binary_search_by(|(e, _)| e.cmp(&0)) {
        Ok(i) => &s.t[i].1,
        Err(_) => return Ok(()),
    };
    if b0.has_hint() || is_rational_k(b0) {
        return Ok(());
    }
    let est = subst_tree_estimate(n);
    // revision 7rt4 (red team 4 structure, m1): with exact kernel counts the test needs a margin,
    // not a factor (revision 8's factor 2 fired only because computed kernels were over-counted)
    if b0.t.sz.tree <= est.saturating_add(HINT_MIN) {
        return Ok(());
    }
    let (e, tree, depth) = subst_at(n)?;
    // revision 8: a seeded subtree (below) can make the substitution larger than its estimate; a
    // hint is set only when it is smaller than the normal form
    if tree >= b0.t.sz.tree {
        return Ok(());
    }
    b0.set_hint(Hint { e, tree, depth });
    Ok(())
}
/// revision 8 (review 8, B1): a node whose series is an exact constant c (no term other than h^0,
/// exact precision) has the value c near the point whatever its subtree holds (`sign(x^2)` at 0,
/// `((1)/x)*0`, `(x-1)/(x-1)` at 1, `(x-x)^3`), so it is regular with the rendering of c as its
/// substitution: a seed of the substitution memo, which a later substitution reads and does not
/// descend below. The seed is the engine's own value, so a substitution that reaches it renders
/// that value. Seeded only at a finite point, and only when c's rendering is not much larger than
/// the node's substitution would be (the hint rule in reverse); charged `SUBST_NODE + expr_units(c)`.
fn seed_const(n: &Expr, c: &Konst) -> Result<bool, SErr> {
    if SUBST_POINT.with(|p| p.borrow().is_none()) || !depends(n) {
        return Ok(false);
    }
    if SUBST.with(|m| m.borrow().contains_key(&addr(n))) {
        return Ok(true);
    }
    let t = c.tree();
    if t > subst_tree_estimate(n)
        .saturating_mul(2)
        .saturating_add(HINT_MIN)
    {
        return Ok(false);
    }
    // the memo entry as a substituted node's (SUBST_NODE), and the rendering of c
    cx_charge(SUBST_NODE.saturating_add(expr_units(c)))?;
    let e = c.to_expr_arc_raw();
    SUBST.with(|m| m.borrow_mut().insert(addr(n), (e, t, c.depth_eff())));
    Ok(true)
}
/// revision 8: the flag of a finished node: regular by the rules above (then its constant term may
/// get a hint), or an exact constant seeded by `seed_const`
fn finish_flag(n: &Expr, v: &Ser, reg: bool) -> Result<bool, SErr> {
    if reg {
        maybe_hint(n, v)?;
        return Ok(true);
    }
    match exact_constant(v) {
        Some(c) => seed_const(n, &c),
        None => Ok(false),
    }
}
/// the substitution of a node, or the node itself (a shallow copy) when it is free of the variable
fn subst_or_node(n: &Expr) -> Result<(std::sync::Arc<Expr>, u64, u32), SErr> {
    if depends(n) {
        subst_at(n)
    } else {
        cx_charge(EXPR_NODE.saturating_add(node_payload_units(n)))?;
        Ok((
            std::sync::Arc::new(n.clone()),
            input_tree(n).unwrap_or(u64::MAX),
            input_depth_or(n),
        ))
    }
}
/// l^r = exp(r log l) with l and r regular: the constant term of r log l is r(a) log(l(a)); hint it
/// when its normal form is much larger (g0 log(f0) multiplies out a g0 that names a kernel twice)
fn hint_exp_arg(t: &Ser, l: &Expr, r: &Expr) -> Result<(), SErr> {
    if t.prec <= Abs(0) {
        return Ok(());
    }
    let t0 = match t.t.binary_search_by(|(e, _)| e.cmp(&0)) {
        Ok(i) => &t.t[i].1,
        Err(_) => return Ok(()),
    };
    if t0.has_hint() || is_rational_k(t0) {
        return Ok(());
    }
    let est = subst_tree_estimate(l)
        .saturating_add(subst_tree_estimate(r))
        .saturating_add(2);
    // revision 7rt4: a margin, as in maybe_hint
    if t0.t.sz.tree <= est.saturating_add(HINT_MIN) {
        return Ok(());
    }
    let (sl, tl, dl) = subst_or_node(l)?;
    let (sr, tr, dr) = subst_or_node(r)?;
    let tree = tl.saturating_add(tr).saturating_add(2);
    // revision 8: as in maybe_hint, only a hint smaller than the normal form
    if tree >= t0.t.sz.tree {
        return Ok(());
    }
    cx_charge(2 * SUBST_NODE)?;
    let lg = std::sync::Arc::new(Expr::Unary(UnaryOp::Log, sl));
    let e = std::sync::Arc::new(Expr::Binary(BinaryOp::Mul, sr, lg));
    t0.set_hint(Hint {
        e,
        tree,
        depth: 2 + dl.max(dr),
    });
    Ok(())
}

// ------------------------------------------------------------------------------------------
// revision 7rt2 (red team 2 structure, H1): `expand` without recursion over the input. One loop
// runs an explicit work stack, as `konst_of` does: one record per pending node on the current
// path that depends on the variable (a `+ -` or `* /` chain and a tower hold their items in a
// boxed record, a unary node is one small record), and the series of a finished node is handed
// to the record on top. A constant operand is converted by `konst_of`, which has its own work
// stack and never calls `expand`. The order of operations (the items of a chain left to right,
// constants converted and divisors inverted as they are read; a tower's top, then its bases from
// the inside out), the charges (KONST_NODE per node visit, KONST_NODE per chain item and per
// tower spine node) and the checks are those of revision 7rt1's recursive reading. The native
// stack of `expand` is O(1) in the input, so on any thread the parser is the only recursion over
// a text (section 2.16, fact 3).
// revision 7rt3: every value carries whether its node is regular (above), and each record names
// its node, so that a finished regular node can hint its constant term.
// ------------------------------------------------------------------------------------------

/// S1 request C-1: a value is its series, whether its node is regular, and the class of its
/// coefficients (`Tail`)
type SV = (Ser, bool, Tail);
struct ESum<'a> {
    node: &'a Expr,
    items: Vec<(bool, &'a Expr)>,
    next: usize,
    consts: Vec<(bool, Konst)>,
    sers: Vec<(bool, Ser)>,
    reg: bool,
    tail: Tail,
}
struct EProd<'a> {
    node: &'a Expr,
    items: Vec<(bool, &'a Expr)>,
    next: usize,
    consts: Vec<Konst>,
    acc: Option<Ser>,
    reg: bool,
    tail: Tail,
}
/// where a tower is: waiting for nothing (just pushed), for the series of a base to raise to a
/// constant exponent, for the series of the top exponent, or for the series of a base to raise
/// to a series exponent (with the exponent's regularity)
enum TSt {
    Init,
    BaseConst(Konst, bool),
    Top,
    BaseGen(Ser, bool, Tail),
}
struct ETower<'a> {
    node: &'a Expr,
    /// (base, exponent node, negations between this level and the next)
    levels: Vec<(&'a Expr, &'a Expr, u32)>,
    top: &'a Expr,
    i: usize,
    st: TSt,
}
/// one record of the work stack (24 bytes; the chain and tower records are boxed)
enum EFrame<'a> {
    Sum(Box<ESum<'a>>),
    Prod(Box<EProd<'a>>),
    Tower(Box<ETower<'a>>),
    Unary(UnaryOp, &'a Expr),
}
enum EStep<'a> {
    Visit(&'a Expr),
    Done(SV),
}

impl<'a> ETower<'a> {
    /// level i's exponent has the series g: l_i^g (revision 7rt1's pow_by_series)
    fn with_exponent(
        &mut self,
        g: Ser,
        reg_g: bool,
        tg: Tail,
        cx: &mut Ctx,
    ) -> Result<Option<EStep<'a>>, SErr> {
        let (l, r, _) = self.levels[self.i];
        if let Some(alpha) = exact_constant(&g) {
            // revision 8 (review 8, B1): the exponent's regularity carries into the level; an exponent
            // that is not regular but expands to an exact constant is seeded with that constant
            let reg_e = reg_g || seed_const(r, &alpha)?;
            self.st = TSt::BaseConst(alpha, reg_e);
            return Ok(Some(EStep::Visit(l)));
        }
        if !depends(l) {
            let hint = if reg_g && regular_val(&g) {
                Some((l, r))
            } else {
                None
            };
            let (v, tv) = pow_const_base(l, g, tg, hint, cx)?;
            let reg = reg_g && regular_val(&v) && !v.is_exact_zero();
            return self.level_done(v, reg, tv, cx);
        }
        self.st = TSt::BaseGen(g, reg_g, tg);
        Ok(Some(EStep::Visit(l)))
    }
    /// level i has the value v: the value of the tower, or level i - 1's exponent
    fn level_done(
        &mut self,
        v: Ser,
        reg: bool,
        tv: Tail,
        cx: &mut Ctx,
    ) -> Result<Option<EStep<'a>>, SErr> {
        if self.i == 0 {
            let reg = finish_flag(self.node, &v, reg)?;
            return Ok(Some(EStep::Done((v, reg, tv))));
        }
        self.i -= 1;
        let mut v = v;
        for _ in 0..self.levels[self.i].2 {
            v = neg(&v);
            check()?;
        }
        self.with_exponent(v, reg, tv, cx)
    }
    fn resume(&mut self, v: Option<SV>, cx: &mut Ctx) -> Result<EStep<'a>, SErr> {
        let st = std::mem::replace(&mut self.st, TSt::Init);
        let step = match (st, v) {
            (TSt::Init, None) => {
                // the top level is l_k^t (revision 7rt1's expand_pow)
                self.i = self.levels.len() - 1;
                if !depends(self.top) {
                    let alpha = konst_of(self.top)?;
                    self.st = TSt::BaseConst(alpha, true);
                    Some(EStep::Visit(self.levels[self.i].0))
                } else {
                    self.st = TSt::Top;
                    Some(EStep::Visit(self.top))
                }
            }
            (TSt::BaseConst(alpha, reg_e), Some((b, reg_b, tb))) => {
                let (v, tv) = pow_by_const(b, tb, alpha, cx)?;
                let reg = reg_e && reg_b && regular_val(&v) && !v.is_exact_zero();
                self.level_done(v, reg, tv, cx)?
            }
            (TSt::Top, Some((g, reg_g, tg))) => self.with_exponent(g, reg_g, tg, cx)?,
            (TSt::BaseGen(g, reg_g, tg), Some((f, reg_f, tf))) => {
                let (l, r, _) = self.levels[self.i];
                let ok = reg_f && reg_g && regular_val(&g) && regular_val(&f);
                let hint = if ok { Some((l, r)) } else { None };
                let (v, tv) = pow_general(f, tf, g, tg, hint, cx)?;
                let reg = ok && regular_val(&v) && !v.is_exact_zero();
                self.level_done(v, reg, tv, cx)?
            }
            _ => unreachable!("tower state"),
        };
        Ok(step.expect("a tower step"))
    }
}

impl<'a> EFrame<'a> {
    fn resume(&mut self, v: Option<SV>, cx: &mut Ctx) -> Result<EStep<'a>, SErr> {
        match self {
            EFrame::Sum(s) => {
                // the constant summands by one merge, the series by one k-way merge
                if let Some((v, r, t)) = v {
                    s.reg &= r;
                    s.tail = s.tail.and(t);
                    s.sers.push((s.items[s.next - 1].0, v));
                }
                while s.next < s.items.len() {
                    let (ng, it) = s.items[s.next];
                    s.next += 1;
                    if depends(it) {
                        return Ok(EStep::Visit(it));
                    }
                    let k = konst_of(it)?;
                    s.consts.push((ng, k));
                }
                let c = Konst::sum_many(&s.consts);
                check()?;
                let v = sum_series(std::mem::take(&mut s.sers), c, cx)?;
                let reg = finish_flag(s.node, &v, s.reg && regular_val(&v))?;
                Ok(EStep::Done((v, reg, s.tail)))
            }
            EFrame::Prod(s) => {
                // a constant divisor is inverted as it is read (`inverse_src(false, its node)`),
                // a series factor is multiplied in order (a series divisor inverted, strict)
                if let Some((b, r, tb)) = v {
                    let div = s.items[s.next - 1].0;
                    // revision 7rt3: a series divisor must not vanish at the point (valuation 0)
                    s.reg &= r && (!div || b.t.first().is_some_and(|(e, _)| *e == 0));
                    // S1 request C-1: a single factor keeps its class; a quotient or a product
                    // takes the product of the full classes (1/b has the class of b)
                    s.tail = match (&s.acc, div) {
                        (None, false) => tb,
                        (None, true) => full_class(&b, tb),
                        (Some(a), _) => full_class(a, s.tail).times(full_class(&b, tb)),
                    };
                    s.acc = Some(chain_step(s.acc.take(), b, div, cx)?);
                }
                while s.next < s.items.len() {
                    let (div, it) = s.items[s.next];
                    s.next += 1;
                    if depends(it) {
                        return Ok(EStep::Visit(it));
                    }
                    let k = konst_of(it)?;
                    s.consts.push(if div {
                        k.inverse_src(false, Some(it))?
                    } else {
                        k
                    });
                }
                let (v, tv) =
                    finish_product(s.acc.take(), std::mem::take(&mut s.consts), s.tail, cx)?;
                let reg = finish_flag(s.node, &v, s.reg && regular_val(&v))?;
                Ok(EStep::Done((v, reg, tv)))
            }
            EFrame::Tower(t) => t.resume(v, cx),
            EFrame::Unary(op, node) => {
                let (b, rb, tb) = v.expect("the operand's series");
                // sign is not continuous where its argument vanishes (sign(x^2) at 0 is 1)
                let sign_ok = *op != UnaryOp::Sign || b.t.first().is_some_and(|(e, _)| *e == 0);
                let (v, tv) = expand_unary(*op, b, tb, cx)?;
                let reg = finish_flag(node, &v, rb && sign_ok && regular_val(&v))?;
                Ok(EStep::Done((v, reg, tv)))
            }
        }
    }
}

/// a tower b_1^(b_2^(...^t)) along its right spine, while the exponent depends on the variable
/// (any number of negations between the levels, `x^-x^-x`); KONST_NODE per spine node below e
fn tower_record<'a>(e: &'a Expr, cx: &mut Ctx) -> Result<ETower<'a>, SErr> {
    let mut levels: Vec<(&Expr, &Expr, u32)> = Vec::new();
    let mut cur = e;
    let mut spine_nodes: u64 = 0;
    let top: &Expr;
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
        if depends(r) && matches!(nx, Expr::Binary(BinaryOp::Pow, _, _)) {
            levels.push((l, r, negs));
            spine_nodes += negs as u64 + 1;
            cur = nx;
        } else {
            levels.push((l, r, 0));
            top = r;
            break;
        }
    }
    cx.work.charge(
        KONST_NODE
            .saturating_mul(spine_nodes)
            .saturating_add(RECORD_UNITS),
    )?;
    Ok(ETower {
        node: e,
        levels,
        top,
        i: 0,
        st: TSt::Init,
    })
}

/// the items of a chain along the left spine, in input order (sum: `+ -`, else `* /`)
fn chain_items_e(e: &Expr, sum: bool) -> Vec<(bool, &Expr)> {
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

/// the visit of one node: its series (a constant, the variable) with its regularity, or the
/// record that reads its children
#[inline(never)]
fn expand_visit<'a>(e: &'a Expr, cx: &mut Ctx) -> Result<Result<SV, EFrame<'a>>, SErr> {
    check()?;
    // revision 5 (review 5, m1): every node visit is charged (a DAG input is expanded once per
    // occurrence, each occurrence charged; `x*0` returns without other charges)
    cx.work.charge(KONST_NODE)?;
    if !depends(e) {
        return Ok(Ok((expand_const(e, cx)?, true, Tail::CONST)));
    }
    Ok(match e {
        // the variable is regular at a finite point (its series x or a + h); S1 request C-1: its
        // coefficients other than the constant term are 1 (h real, or 1/h at infinity)
        Expr::Symbol(_) => Ok((
            expand_var(cx)?,
            cx.xser.t.first().is_some_and(|(e, _)| *e >= 0),
            Tail::REAL,
        )),
        Expr::Binary(BinaryOp::Add | BinaryOp::Sub, _, _) => {
            let items = chain_items_e(e, true);
            cx.work.charge(
                KONST_NODE
                    .saturating_mul(items.len() as u64)
                    .saturating_add(RECORD_UNITS),
            )?;
            // revision 6: exact capacities, 56 bytes per item in all
            let n = items.len();
            Err(EFrame::Sum(Box::new(ESum {
                node: e,
                items,
                next: 0,
                consts: Vec::with_capacity(n),
                sers: Vec::with_capacity(n),
                reg: true,
                tail: Tail::CONST,
            })))
        }
        // revision 7rt1 (red team 1, H3): `*` and `/` are one chain
        Expr::Binary(BinaryOp::Mul | BinaryOp::Div, _, _) => {
            let items = chain_items_e(e, false);
            cx.work.charge(
                KONST_NODE
                    .saturating_mul(items.len() as u64)
                    .saturating_add(RECORD_UNITS),
            )?;
            let n = items.len();
            Err(EFrame::Prod(Box::new(EProd {
                node: e,
                items,
                next: 0,
                consts: Vec::with_capacity(n),
                acc: None,
                reg: true,
                tail: Tail::CONST,
            })))
        }
        // revision 7rt1 (red team 1, H1): a power tower is read along its right spine
        Expr::Binary(BinaryOp::Pow, _, _) => Err(EFrame::Tower(Box::new(tower_record(e, cx)?))),
        Expr::Binary(BinaryOp::Mod, _, _) => {
            return Err(SErr::Unsupported(
                "mod of an expression in the variable".into(),
            ))
        }
        Expr::Unary(op, _) => Err(EFrame::Unary(*op, e)),
        Expr::Function(name, _) => {
            return Err(SErr::Unsupported(format!(
                "{}(...) of an expression in the variable",
                name_text(name)
            )))
        }
        _ => unreachable!(),
    })
}

/// the series of e (no recursion over e: an explicit work stack)
pub fn expand(e: &Expr, cx: &mut Ctx) -> Result<Ser, SErr> {
    let mut stack: Vec<EFrame> = Vec::new();
    let mut visit: Option<&Expr> = Some(e);
    loop {
        let mut val: Option<SV> = None;
        if let Some(x) = visit.take() {
            match expand_visit(x, cx)? {
                Ok(s) => val = Some(s),
                Err(frame) => {
                    // a unary record reads its operand next; the others say what they read
                    if let (EFrame::Unary(..), Expr::Unary(_, i)) = (&frame, x) {
                        stack.push(frame);
                        visit = Some(i);
                        continue;
                    }
                    stack.push(frame);
                }
            }
        }
        loop {
            let top = match stack.last_mut() {
                Some(t) => t,
                None => return Ok(val.expect("a series when the work stack is empty").0),
            };
            match top.resume(val.take(), cx)? {
                EStep::Visit(c) => {
                    visit = Some(c);
                    break;
                }
                EStep::Done(s) => {
                    stack.pop();
                    val = Some(s);
                }
            }
        }
    }
}

#[inline(never)]
fn expand_const(e: &Expr, cx: &mut Ctx) -> Result<Ser, SErr> {
    let k = konst_of(e)?;
    Ok(Ser::exact_const(k).trunc(cx.p))
}

#[inline(never)]
fn expand_var(cx: &mut Ctx) -> Result<Ser, SErr> {
    // revision 4 (review 4, B1): an O(1) copy of the variable's series (shared constants)
    cx.work.charge(8)?;
    Ok(cx.xser.clone().trunc(cx.p))
}

/// revision 7rt1: one series factor of a `*` `/` chain
#[inline(never)]
fn chain_step(acc: Option<Ser>, b: Ser, div: bool, cx: &mut Ctx) -> Result<Ser, SErr> {
    Ok(match (acc, div) {
        (None, false) => b,
        (None, true) => inv(&b, cx)?,
        (Some(a), false) => mul(&a, &b, cx)?,
        (Some(a), true) => div_series(a, b, cx)?,
    })
}

/// S1 request C-1: `tail` is the class of the series factors' product; the result's class is that
/// of c times it
#[inline(never)]
fn finish_product(
    acc: Option<Ser>,
    consts: Vec<Konst>,
    tail: Tail,
    _cx: &mut Ctx,
) -> Result<(Ser, Tail), SErr> {
    let c = Konst::product_many(consts);
    check()?;
    let a = acc.expect("a chain that depends on the variable has a series factor");
    if c.is_zero() {
        return Ok((Ser::exact_zero(), Tail::CONST));
    }
    if c.as_rational().map(|q| q.is_one()).unwrap_or(false) {
        return Ok((a, tail));
    }
    Ok((scale(&a, &c)?, tail.scaled_enclosed(&c)))
}

/// the series quotient
#[inline(never)]
fn div_series(a: Ser, b: Ser, cx: &mut Ctx) -> Result<Ser, SErr> {
    let i = inv(&b, cx)?;
    mul(&a, &i, cx)
}

/// f^g = exp(g log f) (revision 6: never inlined, as div_series)
/// revision 7rt3 (red team 3 structure, H1): the rule is chosen from the series, not from the
/// syntax: a base that depends on the variable but expands to an exact constant (`x*0`, `x - x`,
/// `(x+1)^2 - x^2 - 2*x - 1`) takes the rule of a constant base with that constant, as an exponent
/// that is an exact constant already takes the rule of a constant exponent (review 1, m1).
/// Revision 7rt2 took log of the exact zero, `DivisionByZero("log(0)")`, where 0^r = 0.
/// S1 request C-1: `tf` and `tg` are the classes of f and g; `log f` checks the cut of log
#[inline(never)]
fn pow_general(
    f: Ser,
    tf: Tail,
    g: Ser,
    tg: Tail,
    hint: Option<(&Expr, &Expr)>,
    cx: &mut Ctx,
) -> Result<(Ser, Tail), SErr> {
    if let Some(c) = exact_constant(&f) {
        let cls = const_base_class(&c, &g, tg);
        return Ok((pow_const_base_k(c, g, hint, cx)?, cls));
    }
    if let Some(e) = variable_exponent(&f, &g) {
        return Err(e);
    }
    // f^g = exp(g log f) is real for a real f with a positive constant term and a real g. S1 C-1
    // minor items (regressions review m2): positive by its structure (`struct_positive`: pi,
    // sqrt(2), e) or by its enclosure (1 + sin(1); charged), as `const_base_class` and the kernel
    // classes read it (C-1 accepted a positive rational only)
    let real = full_class(&f, tf).re && full_class(&g, tg).re && known_constant_term(&f) && {
        let f0 = f.coef(0);
        struct_positive(&f0) || crate::series_ball::real_sign(&f0) == Some(1)
    };
    let cls = if real { Tail::REAL } else { Tail::NONE };
    Ok((pow_general_log(f, tf, g, hint, cx)?, cls))
}

#[inline(never)]
fn pow_general_log(
    f: Ser,
    tf: Tail,
    g: Ser,
    hint: Option<(&Expr, &Expr)>,
    cx: &mut Ctx,
) -> Result<Ser, SErr> {
    let lf = log(&f, tf, cx)?;
    let t = mul(&g, &lf, cx)?;
    // revision 7rt3: the constant term of r log l is r(a) log(l(a)) for regular l and r
    if let Some((l, r)) = hint {
        hint_exp_arg(&t, l, r)?;
    }
    exp(&t, cx)
}

/// f^g with f of decided nonzero valuation v and g not an exact constant: f = c h^v (1 + u), so
/// f^g = exp(g log c + v g log h + g log(1 + u)). It is a Laurent series exactly when g is an
/// integer constant k (then it is f^k): a term g_e h^e of g with e != 0 gives the term
/// v g_e h^e log h, and a non-integer k a branch point. So, before the rule `exp(g log f)`
/// (whose `log f` is `NotLaurent`):
/// - a decided non-integer constant term, or a term e != 0 with a decided nonzero coefficient,
///   keeps that rule (`None`, then `NotLaurent`), which is right then;
/// - otherwise the first term e != 0 with an undecided coefficient is reported (`UndecidedZero`
///   of that coefficient), for an exact and a truncated g alike: the function is f^k when it is 0;
/// - a g with no stored term other than its constant term is `truncated_integer_exponent`'s.
///
/// S1 finish, fix-up round 1 (fin1 semantics B1): before, every g with a stored term other than
/// its constant term took `log f`, also when that term's coefficient is undecided and 0, so
/// `x^(1+(log(2)+log(3)-log(6))*x)` at 0 (which is x) and `(x-pi)^((1+x)/(1+x))` at pi (which is
/// x - pi) were `NotLaurent`, a decided answer.
fn variable_exponent(f: &Ser, g: &Ser) -> Option<SErr> {
    if f.t.is_empty() || f.val() == 0 || f.lead_status() != Some(ZS::NonZero) {
        return None;
    }
    if g.t.iter().all(|(e, _)| *e == 0) {
        return truncated_integer_exponent(f, g);
    }
    // the constant term is known when it lies below the precision
    if g.prec == Exact || g.prec > Abs(0) {
        if let Some((_, k)) = g.t.iter().find(|(e, _)| *e == 0) {
            let st = non_integer_status(k).0;
            if let Err(e) = check() {
                return Some(e);
            }
            if st == ZS::NonZero {
                return None;
            }
        }
    }
    let mut first_undecided: Option<&Konst> = None;
    for (e, c) in &g.t {
        if *e == 0 {
            continue;
        }
        let st = c.zero_status();
        if let Err(err) = check() {
            return Some(err);
        }
        match st {
            ZS::NonZero => return None,
            ZS::Unknown if first_undecided.is_none() => first_undecided = Some(c),
            _ => {}
        }
    }
    first_undecided.map(undecided)
}

/// S1 finish under the narrowed bar (review 6, semantics lens, m1): f^g with f of decided nonzero
/// valuation v and g a truncated series whose only stored term is its constant term k (or that
/// has no stored term). Then f^g = f^k exp((g - k) log f), with g - k = O(h^P): it is a Laurent
/// series when g is the constant k and k is an integer, and it has the logarithmic term
/// v (g - k) log h otherwise. Whether g - k is 0 is undecided (a truncated series is not an exact
/// one), so the answer asks for precision (the driver raises it, and never decides:
/// `PrecisionExhausted`). k must be an integer for f^k to be Laurent: a decided non-integer k is
/// `NotLaurent` whatever g - k is, and an undecided one is reported (`UndecidedZero` of k - n, n
/// the nearest integer), the module rule for an exponent. A g with other stored terms is
/// `variable_exponent`'s (fix-up round 1 of the finish: the same argument for its stored terms).
/// `None`: the rule `exp(g log f)` applies.
fn truncated_integer_exponent(f: &Ser, g: &Ser) -> Option<SErr> {
    if g.prec == Exact || f.t.is_empty() || f.val() == 0 || f.lead_status() != Some(ZS::NonZero) {
        return None;
    }
    match g.t.as_slice() {
        // O(h^P): its constant term is 0 (P > 0) or undecided (P <= 0)
        [] => Some(SErr::NeedPrecision),
        [(0, k)] => match non_integer_status(k) {
            (ZS::NonZero, _) => Some(SErr::NotLaurent(format!(
                "a non-integer power of an argument of valuation {} has a branch point there",
                f.val()
            ))),
            (ZS::Zero, _) => Some(SErr::NeedPrecision),
            (_, Some(n)) => Some(undecided(&k.sub(&Konst::rat(Rational::from_integer(n))))),
            (_, None) => Some(undecided(k)),
        },
        _ => None,
    }
}

/// S1 request C-1: with the class of the result (`const_base_class`)
#[inline(never)]
fn pow_const_base(
    l: &Expr,
    g: Ser,
    tg: Tail,
    hint: Option<(&Expr, &Expr)>,
    cx: &mut Ctx,
) -> Result<(Ser, Tail), SErr> {
    let lc = konst_of(l)?;
    let cls = const_base_class(&lc, &g, tg);
    Ok((pow_const_base_k(lc, g, hint, cx)?, cls))
}

/// S1 request C-1: the class of c^g = exp(g log c) for a constant c: real for a structurally
/// positive c and a real g
fn const_base_class(c: &Konst, g: &Ser, tg: Tail) -> Tail {
    if struct_positive(c) && full_class(g, tg).re {
        Tail::REAL
    } else {
        Tail::NONE
    }
}

/// constant base: l^r = exp(r log l); the engine introduces log l, so l must be decided (m5)
#[inline(never)]
fn pow_const_base_k(
    lc: Konst,
    g: Ser,
    hint: Option<(&Expr, &Expr)>,
    cx: &mut Ctx,
) -> Result<Ser, SErr> {
    match lc.zero_status() {
        ZS::Zero => zero_pow(&g),
        ZS::Unknown => Err(undecided(&lc)),
        ZS::NonZero => {
            let ll = Ser::exact_const(kapply(UnaryOp::Log, &lc)?);
            let t = mul(&g, &ll, cx)?;
            // revision 7rt3: as in pow_general
            if let Some((l, r)) = hint {
                hint_exp_arg(&t, l, r)?;
            }
            exp(&t, cx)
        }
    }
}

/// revision 7rt3 (red team 3 structure, H1): 0^r, decided with the engine's own tests on the series
/// of r (0^r = 0 where Re r > 0; it is undefined where Re r < 0, so it has a Laurent expansion at the
/// point only when Re r > 0 near the point, or r = 0 identically). Revision 7rt2 read `val`, a lower
/// bound when the lead is undecided, and accepted only a positive rational constant term, so
/// `x + 0^(2^x)` at 1 and `x + 0^(x + sqrt(2))` at 0 were a decided `NotLaurent`.
fn zero_pow(g: &Ser) -> Result<Ser, SErr> {
    if g.t.is_empty() {
        // an exact zero exponent is an exact constant and took the constant rule (0^0 = 1); an
        // empty truncated exponent has an undecided valuation (section 2.1)
        return if g.prec == Exact {
            Ok(Ser::exact_const(Konst::one()))
        } else {
            Err(SErr::NeedPrecision)
        };
    }
    let (e0, c) = (g.t[0].0, &g.t[0].1);
    if e0 != 0 {
        // a pole of r (e0 < 0), or r(a) = 0 with r not constant (e0 > 0): Re r takes both signs in
        // every punctured neighbourhood, when the lead is decided nonzero
        return match c.zero_status() {
            ZS::NonZero => Err(SErr::NotLaurent(if e0 < 0 {
                "0^r with a pole of r at the point has no Laurent expansion there".into()
            } else {
                "0^r with r vanishing at the point has no Laurent expansion there".into()
            })),
            _ => Err(undecided(c)),
        };
    }
    // r(a) = c: the sign of Re c by rule 5's enclosures (split_imag, then a real ball of A); the
    // same rule decides a constant 0^alpha (revision 8, `konst.rs::zero_pow_sign`)
    crate::series_const::zero_pow_sign(c)?;
    Ok(Ser::exact_zero())
}

/// S1 request C-1: `tb` is the class of b's coefficients; the result carries its own class
#[inline(never)]
fn expand_unary(op: UnaryOp, b: Ser, tb: Tail, cx: &mut Ctx) -> Result<(Ser, Tail), SErr> {
    use UnaryOp::*;
    let fb = full_class(&b, tb);
    // a real function of a real argument is real; an odd one of an imaginary argument imaginary
    // and an even one real (sin(i t) = i sinh t, cos(i t) = cosh t)
    let odd = if fb.re {
        Tail::REAL
    } else if fb.im {
        Tail::IMAG
    } else {
        Tail::NONE
    };
    let even = if fb.re || fb.im {
        Tail::REAL
    } else {
        Tail::NONE
    };
    let r = match op {
        Neg => Ok((neg(&b), tb)),
        Exp => {
            let v = exp(&b, cx)?;
            // exp(b0 + u) = exp(b0) exp(u): exp(u) - 1 is real for a real tail u
            let cls = if tb.re && known_constant_term(&v) {
                Tail::REAL.scaled(&v.coef(0))
            } else {
                Tail::NONE
            };
            Ok((v, cls))
        }
        Log => {
            let v = log(&b, tb, cx)?;
            // log(c0 + u) = log c0 + log(1 + u/c0), u/c0 real
            Ok((v, even))
        }
        Sqrt => {
            let half = Konst::rat(Rational::new(Integer::from(1), Integer::from(2)).unwrap());
            let v = pow_const(&b, &half, Some(tb), cx)?;
            let cls = pow_class(&b, tb, &half, &v);
            Ok((v, cls))
        }
        Sin => Ok((sin_cos(&b, false, cx)?.0, odd)),
        Cos => Ok((sin_cos(&b, false, cx)?.1, even)),
        Sinh => Ok((sin_cos(&b, true, cx)?.0, odd)),
        Cosh => Ok((sin_cos(&b, true, cx)?.1, even)),
        Tan | Tanh => {
            let (s, c) = sin_cos(&b, op == Tanh, cx)?;
            let ic = inv(&c, cx)?;
            Ok((mul(&s, &ic, cx)?, odd))
        }
        Arctan | Arcsin | Arccos | Arcsinh | Arccosh | Arctanh | Erf => atan_like(op, &b, tb, cx),
        // abs and sign need rational coefficients: abs keeps them, sign is a constant
        Abs => Ok((abs_sign(op, &b)?, Tail::REAL)),
        Sign => Ok((abs_sign(op, &b)?, Tail::CONST)),
        Gamma | Factorial | Zeta => Err(SErr::Unsupported(
            format!("{:?} of an expression in the variable", op).to_lowercase(),
        )),
    };
    check()?;
    r
}

pub enum Point<'a> {
    At(&'a Expr),
    Infinity,
}

/// One use of the engine: the thread-local state of an expansion (the work counter, the fault,
/// the kernel table, the hint table, the pre-pass and the substitution memos) lives from `begin`
/// to the guard's drop, which clears it (also on a panic). The prototype of the design emulated
/// the explicit `&mut Work` of §2.7 with this thread-local state; the port keeps it (see the
/// module header). The engine calls nothing that expands again, so a nested session is refused
/// rather than allowed to clobber the outer one.
pub struct Session(());
thread_local! {
    static ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
impl Session {
    pub fn begin() -> Result<Session, SErr> {
        if ACTIVE.with(|c| c.replace(true)) {
            return Err(SErr::Unsupported(
                "a series expansion inside another one".into(),
            ));
        }
        Ok(Session(()))
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        clear_hints();
        reset_work(u64::MAX);
        DEP.with(|m| *m.borrow_mut() = std::collections::HashMap::new());
        SUBST.with(|m| *m.borrow_mut() = std::collections::HashMap::new());
        SUBST_POINT.with(|c| *c.borrow_mut() = None);
        crate::series_const::set_var_shadow(&Symbol::new("x"));
        ACTIVE.with(|c| c.set(false));
    }
}

/// Which stored terms of a result the caller reads, so the driver charges, bounds and renders
/// only those (design §2.13: `series_expansion` every term, `expansion_first_term` the first one,
/// `expansion_reaches` none).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Render {
    All,
    First,
    None,
}

/// Laurent expansion of f in h = x - a (or h = 1/x) with all terms of exponent <= max_exp
pub fn expansion(
    f: &Expr,
    var: &Symbol,
    point: Point,
    max_exp: i64,
    budget: u64,
    render: Render,
) -> Result<Ser, SErr> {
    reset_work(budget);
    DEP.with(|m| m.borrow_mut().clear());
    SUBST.with(|m| m.borrow_mut().clear());
    SUBST_POINT.with(|c| *c.borrow_mut() = None);
    // S1 fix-up round 5 (review 5 semantics, B1): a variable named pi or I shadows the names of the
    // constants the engine makes itself
    crate::series_const::set_var_shadow(var);
    // revision 5 (review 5, m1): the pre-pass (memoized per node, linear in the DAG) gives the
    // dependence flags and the input depth; the point is tested with the same walk
    let (_, fdepth) = dep_pass(f, var);
    let mut in_depth = fdepth;
    if let Point::At(a) = &point {
        let (dep, pd) = dep_pass(a, var);
        if dep {
            return Err(SErr::BadPoint(
                "the expansion point contains the variable".into(),
            ));
        }
        in_depth = in_depth.max(pd);
    }
    KDEPTH_ALLOWED
        .with(|c| c.set(MAX_KERNEL_DEPTH.max(in_depth.saturating_mul(KERNEL_DEPTH_FACTOR))));
    // revision 7rt3: the point as one shared node for the substitutions (a shallow copy, charged)
    if let Point::At(a) = &point {
        charge(EXPR_NODE.saturating_add(node_payload_units(a)));
        check()?;
        let pv = (
            std::sync::Arc::new((*a).clone()),
            input_tree(a).unwrap_or(u64::MAX),
            input_depth(a).unwrap_or(u32::MAX),
        );
        SUBST_POINT.with(|c| *c.borrow_mut() = Some(pv));
    }
    let xser = match point {
        Point::Infinity => Ser {
            t: vec![(-1, Konst::one())],
            prec: Exact,
        },
        Point::At(a) => {
            let k = konst_of(a)?;
            // revision 5 (review 5, m2): whether the point is zero is recorded here, once
            POINT_ZERO.with(|c| c.set(k.is_zero()));
            if k.is_zero() {
                Ser {
                    t: vec![(1, Konst::one())],
                    prec: Exact,
                }
            } else {
                Ser {
                    t: vec![(0, k), (1, Konst::one())],
                    prec: Exact,
                }
            }
        }
    };
    let target = max_exp
        .checked_add(1)
        .ok_or(SErr::Unsupported("order too large".into()))?;
    // revision 7rt2: clamped before the cast (a saturated u64 cast to i64 was -1)
    let need_cap = target
        .unsigned_abs()
        .saturating_mul(4)
        .max(128)
        .min(i64::MAX as u64) as i64;
    let mut p = target;
    // revision 7rt3 (red team 3 series, m1): the precision of the previous loss-driven attempt
    let mut last_prec: Option<Prec> = None;
    for _attempt in 0..24 {
        // revision 7rt3: the hints of the previous attempt refer to its constants
        clear_hints();
        PREC_CLAMPED.with(|c| c.set(false));
        let mut cx = Ctx {
            xser: xser.clone(),
            p,
            work: Work { used: 0 },
        };
        let r = expand(f, &mut cx);
        match r {
            Ok(s) => {
                if s.prec >= Abs(target) {
                    let s = s.trunc_hard(target);
                    // revision 7rt2: the rendering negates exponents (h^(-e), and e -> -e at
                    // infinity); only the first term can be i64::MIN
                    if s.t.first().is_some_and(|(e, _)| *e == i64::MIN) {
                        return Err(SErr::Unsupported("a series exponent overflows i64".into()));
                    }
                    // revision 7rt3 (red team 3 structure, H2): the result's tree is bounded
                    // before it is rendered (its printing and dropping are linear in it, its
                    // re-parsing in its text, MAX_RESULT_TEXT, plus its numbers,
                    // MAX_RESULT_PARSE_WORK; the consumers that rebuild it are not bounded by S1,
                    // see MAX_RESULT_TEXT); S1 fix-up round 2 (review 2, B2):
                    // the leaf text of that tree is bounded after the rendering, in
                    // series.rs `public_expansion` (MAX_RESULT_TEXT; round 3: and the leaf text
                    // of the tree `simplify` makes of it)
                    // revision 7rt3 (red team 3 memory, m3): the limit path reads the first stored
                    // term only, so only that term is charged and bounded (and rendered)
                    // revision 7rt4 (red team 4 memory, m2): no term is rendered for a caller that
                    // reads none (revision 8 charged and bounded every term for limit_with_error)
                    let take = match render {
                        Render::None => 0,
                        Render::First => 1,
                        Render::All => s.t.len(),
                    };
                    let rs = if take < s.t.len() {
                        Ser {
                            t: s.t.iter().take(take).cloned().collect(),
                            prec: s.prec,
                        }
                    } else {
                        s.clone()
                    };
                    let tr = render_tree(&rs);
                    if tr > MAX_RESULT_TREE {
                        return Err(SErr::Limit("nodes of the result's tree", MAX_RESULT_TREE));
                    }
                    // revision 4 (review 4, B1): rendering is charged before the result is
                    // returned (expr_units of every coefficient)
                    // revision 6 (review 6, B3): and the nodes of each stored term beyond its
                    // coefficient (the power of h, the product, its share of the sum)
                    charge(render_units(&rs));
                    drop(rs);
                    check()?;
                    return Ok(s);
                }
                // revision 7rt4 (red team 4 series, H1): an attempt that clamped a precision at
                // i64::MAX and did not raise the result's precision over the previous loss-driven
                // attempt cannot be repaired by a larger P (the clamped precision does not grow with
                // P): the loss is past the representation of the exponents. A clamp in an attempt whose
                // precision still grows (a loss elsewhere) lets the search go on.
                if PREC_CLAMPED.with(|c| c.get()) && last_prec.is_some_and(|lp| s.prec <= lp) {
                    return Err(SErr::Unsupported("a series precision overflows i64".into()));
                }
                // decided precision loss: the exact amount is known
                // revision 7rt2: checked (s.prec may lie far below the target)
                // (s.prec < Abs(target), so it is finite)
                let sp = match s.prec {
                    Abs(q) => q,
                    Exact => unreachable!("an exact result meets every target"),
                };
                let step = target.checked_sub(sp).ok_or(SErr::PrecisionExhausted(p))?;
                // revision 7rt3 (red team 3 series, m1): when an attempt did not raise the
                // precision (a kept leading term whose precision n vb + 1 does not grow with P),
                // the extra precision at least doubles, so 24 attempts reach about 2^23 T and the
                // budget bounds their cost; revision 7rt2 stepped by T - prec and ran out of
                // attempts on `((x+x^2)^1000)/x^999` at orders 2 to 40
                let extra = p.saturating_sub(target).max(0);
                let grow = match last_prec {
                    Some(lp) if s.prec <= lp => step.max(extra),
                    _ => step,
                };
                last_prec = Some(s.prec);
                p = p.checked_add(grow).ok_or(SErr::PrecisionExhausted(p))?;
            }
            Err(SErr::NeedPrecision) => {
                // an undecided valuation: double the extra precision, up to the cap
                let extra = (p - target).max(0);
                let next = extra.saturating_mul(2).max(8);
                if next > need_cap {
                    return Err(SErr::PrecisionExhausted(p));
                }
                p = target
                    .checked_add(next)
                    .ok_or(SErr::PrecisionExhausted(p))?;
            }
            Err(e) => return Err(e),
        }
    }
    Err(SErr::PrecisionExhausted(p))
}

/// revision 6 (review 6, B3): units of rendering one stored term beyond its coefficient's
/// expr_units: the Integer exponent node with its digit vector, the Pow node, the product or
/// quotient node and its share of the balanced sum (4 nodes), and its slot
pub const RENDER_TERM: u64 = 4 * EXPR_NODE + 2 * EXPR_INT + SER_TERM;
/// the a-priori units of to_expr(s): h (x - a: 2 nodes; h is shared by every term), then per term
/// expr_units of its coefficient and RENDER_TERM
pub fn render_units(s: &Ser) -> u64 {
    s.t.iter().fold(2 * EXPR_NODE, |acc, (_, k)| {
        acc.saturating_add(expr_units(k))
            .saturating_add(RENDER_TERM)
    })
}
/// revision 7rt3: the tree size of to_expr(s) (O(n) from the cached sizes): per term the
/// coefficient's tree, at most 4 nodes (the power of h, its exponent, the product, its share of
/// the sum) and h, which every term other than the constant one names (x, or x - a: 2 nodes and
/// the point's tree)
pub fn render_tree(s: &Ser) -> u64 {
    let th: u64 = match SUBST_POINT.with(|c| c.borrow().as_ref().map(|v| v.1)) {
        Some(pt) if !POINT_ZERO.with(|c| c.get()) => pt.saturating_add(2),
        _ => 1,
    };
    s.t.iter().fold(1u64, |acc, (_, k)| {
        acc.saturating_add(k.tree())
            .saturating_add(4)
            .saturating_add(th)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// S1 C-1 minor items (semantics review m7): a fault of the cut check inside `pow_const` (the
    /// budget refused an enclosure of the constant term) is reported as the fault, also when the
    /// exponent's integrality is undecided, for (-1 + I*h)^(sin(1)^2 + cos(1)^2). It passes on
    /// the code before the minor items too: the fault is sticky, and `undecided` (series_const.rs)
    /// returns it before it builds the constant of `UndecidedZero(alpha - n)`. A guard.
    #[test]
    fn pow_const_reports_a_budget_fault_of_the_cut_check() {
        reset_work(u64::MAX / 4);
        let alpha =
            konst_of(&crate::parser::parse("sin(1)^2 + cos(1)^2").expect("parse")).expect("alpha");
        let i = konst_of(&crate::parser::parse("I").expect("parse")).expect("I");
        assert_eq!(
            non_integer_status(&alpha).0,
            ZS::Unknown,
            "alpha's integrality"
        );
        let b = Ser {
            t: vec![(0, Konst::int(-1)), (1, i)],
            prec: Exact,
        };
        let mut cx = Ctx {
            xser: Ser {
                t: vec![(1, Konst::one())],
                prec: Exact,
            },
            p: 4,
            work: Work { used: 0 },
        };
        // a budget of one unit: the first enclosure of the cut check is refused
        COST.with(|c| c.set(0));
        MAXCOST.with(|m| m.set(1));
        let r = pow_const(&b, &alpha, Some(Tail::NONE), &mut cx);
        assert!(matches!(r, Err(SErr::Budget(_))), "{:?}", r.err());
        reset_work(u64::MAX / 4);
    }
}
