//! Galois rings `GR(p^a, d) = Z_{p^a}[x]/(D)` (MAGMA Handbook ch. 48).
//!
//! MAGMA source: Chapter 48 — Galois Rings (`GaloisRing`, `GR`, `Eltseq`,
//! `Characteristic`, `#R`, `Degree`, `ResidueField`, `IsUnit`, `IsNilpotent`,
//! `IsZeroDivisor`, `Quotrem`, `GCD`, `LCM`, `XGCD`).
//!
//! A Galois ring `R = GR(p^a, d)` is the extension of `Z/p^a Z` by a monic `D`
//! that is irreducible mod `p`. For `a > 1` it is **commutative but not a field
//! and not a domain** (`p·p^{a-1} = 0`); it is a finite chain ring — a local
//! principal ideal ring with unique maximal ideal `(p)` and nilpotency index
//! `a`. Accordingly this type implements [`Ring`]/[`CommutativeRing`] plus the
//! non-domain markers [`LocalRing`]/[`PrincipalIdealRing`]/[`FiniteChainRing`]
//! and a [`DiscreteValuation`] (the `p`-adic valuation), and exposes the
//! Euclidean operations `quotrem`/`gcd`/`lcm`/`xgcd` **as inherent methods** —
//! it deliberately does **not** implement [`Field`] or `IntegralDomain`.

use std::sync::LazyLock;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use rustmath_core::{
    divisibility::{FiniteChainRing, LocalRing, PrincipalIdealRing},
    valuation::DiscreteValuation,
    CommutativeRing, EuclideanDomain, Field, MathError, Parent, Result, Ring,
};
use rustmath_integers::Integer;

use crate::finite_field::{FiniteField, FiniteFieldElement};

fn red(x: Integer, m: &Integer) -> Integer {
    let r = x % m.clone();
    if r.signum() < 0 {
        r + m.clone()
    } else {
        r
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GaloisRingData {
    p: Integer,
    a: u32,
    pa: Integer, // p^a
    d: usize,
    /// Defining monic polynomial `D`, little-endian, coeffs in `[0, p^a)`,
    /// length `d + 1`, `D[d] == 1`. Reduces to an irreducible mod `p`.
    modulus: Vec<Integer>,
    /// Residue field `R/(p) = GF(p^d)`.
    residue: FiniteField,
}

/// A Galois ring `GR(p^a, d)`, carried as a shared [`Parent`].
#[derive(Clone)]
pub struct GaloisRing {
    data: Arc<GaloisRingData>,
}

static GR_CACHE: LazyLock<
    rustmath_core::UniqueCache<(Integer, u32, usize, Vec<Integer>), GaloisRingData>,
> = LazyLock::new(rustmath_core::UniqueCache::new);

impl rustmath_core::UniqueRepresentation for GaloisRingData {
    type Key = (Integer, u32, usize, Vec<Integer>);
    fn key(&self) -> Self::Key {
        (self.p.clone(), self.a, self.d, self.modulus.clone())
    }
    fn cache() -> &'static rustmath_core::UniqueCache<Self::Key, Self> {
        &GR_CACHE
    }
}

impl GaloisRing {
    /// Construct the default `GR(p^a, d)` (MAGMA `GaloisRing(p, a, d)`): the
    /// defining polynomial is the one used for `GF(p^d)` (Conway when tabulated),
    /// lifted to `Z/p^a`.
    pub fn new(p: Integer, a: u32, d: usize) -> Result<Self> {
        if !p.is_prime() {
            return Err(MathError::InvalidArgument(format!("{p} is not prime")));
        }
        if a < 1 || d < 1 {
            return Err(MathError::InvalidArgument("need a >= 1 and d >= 1".into()));
        }
        let residue = FiniteField::new(p.clone(), d)?;
        let pa = p.pow(a);
        // D = residue-field defining polynomial lifted to Z/p^a (coeffs already < p).
        let modulus: Vec<Integer> = residue
            .defining_polynomial()
            .iter()
            .map(|c| red(c.clone(), &pa))
            .collect();
        Ok(Self::from_data(GaloisRingData {
            p,
            a,
            pa,
            d,
            modulus,
            residue,
        }))
    }

    /// Construct `GR(p^a, D)` from a caller-supplied monic `D` (little-endian
    /// over `Z`) that is irreducible mod `p` (MAGMA `GaloisRing(p, a, D)`). `D`'s
    /// degree fixes `d`; coefficients are reduced mod `p^a`.
    pub fn with_modulus(p: Integer, a: u32, modulus: Vec<Integer>) -> Result<Self> {
        if !p.is_prime() {
            return Err(MathError::InvalidArgument(format!("{p} is not prime")));
        }
        if a < 1 || modulus.len() < 2 {
            return Err(MathError::InvalidArgument(
                "need a >= 1 and deg D >= 1".into(),
            ));
        }
        let d = modulus.len() - 1;
        let pa = p.pow(a);
        let modp: Vec<Integer> = modulus.iter().map(|c| red(c.clone(), &p)).collect();
        if !modp[d].is_one() {
            return Err(MathError::InvalidArgument("D must be monic".into()));
        }
        if d >= 2 && !crate::poly_factor::is_irreducible_fp(&modp, &p) {
            return Err(MathError::InvalidArgument(
                "D must be irreducible mod p".into(),
            ));
        }
        let residue = FiniteField::with_modulus(p.clone(), modp)?;
        let modulus: Vec<Integer> = modulus.iter().map(|c| red(c.clone(), &pa)).collect();
        Ok(Self::from_data(GaloisRingData {
            p,
            a,
            pa,
            d,
            modulus,
            residue,
        }))
    }

    fn from_data(d: GaloisRingData) -> Self {
        use rustmath_core::UniqueRepresentation;
        let key = d.key();
        GaloisRing {
            data: GaloisRingData::get_unique(key, || d),
        }
    }

    /// The characteristic `p^a`.
    pub fn characteristic(&self) -> &Integer {
        &self.data.pa
    }

    /// The prime `p`.
    pub fn prime(&self) -> &Integer {
        &self.data.p
    }

    /// The exponent `a` (so the base ring is `Z/p^a`).
    pub fn exponent(&self) -> u32 {
        self.data.a
    }

    /// The degree `d` of the defining polynomial.
    pub fn degree(&self) -> usize {
        self.data.d
    }

    /// The cardinality `p^{a d}` as an [`Integer`].
    pub fn cardinality(&self) -> Integer {
        self.data.pa.pow(self.data.d as u32)
    }

    /// The residue field `R/(p) = GF(p^d)` (MAGMA `ResidueField`).
    pub fn residue_field(&self) -> FiniteField {
        self.data.residue.clone()
    }

    /// Whether this ring is a field (equivalently a domain), i.e. `a == 1`.
    pub fn is_field(&self) -> bool {
        self.data.a == 1
    }

    /// The canonical quotient `GR(p^b, d)` for `1 <= b <= a`: same defining
    /// polynomial, coefficients reduced mod `p^b`. Together with
    /// [`GaloisRingElement::project`] this is the canonical surjection
    /// `GR(p^a, d) -> GR(p^b, d)` (for `b = 1` it reaches the residue field
    /// viewed as `GR(p, d)`; for `b = a` it is the identity).
    pub fn reduce_precision(&self, b: u32) -> Result<GaloisRing> {
        if b < 1 || b > self.data.a {
            return Err(MathError::InvalidArgument(format!(
                "precision b = {b} out of range 1..={}",
                self.data.a
            )));
        }
        if b == self.data.a {
            return Ok(self.clone());
        }
        // `with_modulus` re-reduces the coefficients mod p^b; the mod-p
        // reduction (hence the residue field) is unchanged.
        GaloisRing::with_modulus(self.data.p.clone(), b, self.data.modulus.clone())
    }

    /// The Teichmüller lift of a residue-field element: the unique `t` in this
    /// ring with `t^{p^d} = t` and `t mod p = x`.
    pub fn teichmuller_lift(&self, x: &FiniteFieldElement) -> Result<GaloisRingElement> {
        if x.field() != &self.data.residue {
            return Err(MathError::InvalidArgument(
                "element does not lie in this Galois ring's residue field".into(),
            ));
        }
        Ok(self.element(x.eltseq().to_vec()).teichmuller())
    }

    // -- element constructors -------------------------------------------------

    /// The zero element.
    pub fn zero(&self) -> GaloisRingElement {
        GaloisRingElement {
            coeffs: vec![Integer::zero(); self.data.d],
            ring: Some(self.clone()),
        }
    }

    /// The one element.
    pub fn one(&self) -> GaloisRingElement {
        let mut c = vec![Integer::zero(); self.data.d];
        c[0] = Integer::one();
        GaloisRingElement {
            coeffs: c,
            ring: Some(self.clone()),
        }
    }

    /// The generator `w = R.1` (class of `x`).
    pub fn generator(&self) -> GaloisRingElement {
        let mut c = vec![Integer::zero(); self.data.d];
        if self.data.d >= 2 {
            c[1] = Integer::one();
        }
        GaloisRingElement {
            coeffs: c,
            ring: Some(self.clone()),
        }
    }

    /// Coerce an integer `k` (identified with `k mod p^a`).
    pub fn from_int(&self, k: Integer) -> GaloisRingElement {
        let mut c = vec![Integer::zero(); self.data.d];
        c[0] = red(k, &self.data.pa);
        GaloisRingElement {
            coeffs: c,
            ring: Some(self.clone()),
        }
    }

    /// Build an element from a coefficient sequence over the base ring
    /// (MAGMA `R ! [a0, ..., a_{d-1}]`), reducing mod `p^a` and mod `D`.
    pub fn element(&self, coeffs: Vec<Integer>) -> GaloisRingElement {
        let reduced = self.reduce(coeffs);
        GaloisRingElement {
            coeffs: reduced,
            ring: Some(self.clone()),
        }
    }

    /// The element `p^k` (a convenience for uniformizer powers).
    pub fn p_power(&self, k: u32) -> GaloisRingElement {
        self.from_int(self.data.p.pow(k))
    }

    fn reduce(&self, mut poly: Vec<Integer>) -> Vec<Integer> {
        let pa = &self.data.pa;
        let d = self.data.d;
        let modulus = &self.data.modulus;
        for c in poly.iter_mut() {
            *c = red(c.clone(), pa);
        }
        for k in (d..poly.len()).rev() {
            let lead = poly[k].clone();
            if lead.is_zero() {
                continue;
            }
            for i in 0..d {
                poly[k - d + i] = red(poly[k - d + i].clone() - lead.clone() * modulus[i].clone(), pa);
            }
            poly[k] = Integer::zero();
        }
        poly.truncate(d);
        poly.resize(d, Integer::zero());
        poly
    }
}

impl fmt::Debug for GaloisRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GR({}^{}, {})", self.data.p, self.data.a, self.data.d)
    }
}

impl fmt::Display for GaloisRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GR({}^{}, {})", self.data.p, self.data.a, self.data.d)
    }
}

impl PartialEq for GaloisRing {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl Eq for GaloisRing {}

impl Parent for GaloisRing {
    type Element = GaloisRingElement;
    /// Whether `element` is an element of this ring. `false` for an unbound element (from
    /// `Ring::zero()`/`Ring::one()`): it has no parent until it is bound (see
    /// [`GaloisRingElement`]).
    fn contains(&self, element: &Self::Element) -> bool {
        element.ring.as_ref().is_some_and(|r| r.data == self.data)
    }
    fn zero(&self) -> Option<Self::Element> {
        Some(GaloisRing::zero(self))
    }
    fn one(&self) -> Option<Self::Element> {
        Some(GaloisRing::one(self))
    }
    /// `#R = p^(a d)` when it fits in `usize`; `None` otherwise (no claim, see
    /// [`Parent::cardinality`]). The exact order is the inherent
    /// [`GaloisRing::cardinality`].
    fn cardinality(&self) -> Option<usize> {
        use rustmath_core::NumericConversion;
        self.cardinality().to_usize()
    }
    /// Galois rings are finite (MAGMA Ch. 48 `IsFinite`), also when `p^(a d) > usize::MAX`.
    fn is_finite(&self) -> Option<bool> {
        Some(true)
    }
    fn name(&self) -> String {
        format!("{self}")
    }
}

// ---------------------------------------------------------------------------
// Element
// ---------------------------------------------------------------------------

/// An element of a [`GaloisRing`] `GR(p^a, d)`, carrying its shared parent.
///
/// # Unbound elements (`Ring::zero`/`Ring::one`)
///
/// The static [`Ring::zero`]/[`Ring::one`] constructors cannot know `p`, `a` or `d`, so (as
/// for [`FiniteFieldElement`] and the modulus-0 sentinel of [`crate::IntegerMod`] and
/// [`crate::PrimeField`]) they return *unbound* elements: a single integer constant `c ∈ Z`
/// awaiting the canonical map `Z → GR(p^a, d)`, `c ↦ c·1` (MAGMA Ch. 48 §48.2.3: `R ! a` for
/// an integer `a` is "identified with `a mod p^a`"). That map is the unique ring homomorphism
/// from `Z`, so binding is canonical. Use [`GaloisRing::zero`]/[`GaloisRing::one`]/
/// [`GaloisRing::from_int`] when the parent is known. The precise algebra:
///
/// * **`+`, `-`, `*`**: an unbound operand is bound on contact into the other operand's ring
///   (via [`GaloisRing::from_int`]), so `R::zero() + x == x` and `R::zero() * x` is the zero
///   of `x`'s ring. Two unbound operands combine in `Z` (the result stays unbound). Two
///   elements of *different* Galois rings panic, as before.
/// * **`Neg`**: negates in `Z` for an unbound element.
/// * **`==`** coerces exactly like the arithmetic: an unbound operand is bound into the other
///   operand's ring before comparing, so `R::zero() == r.zero()` and `unbound(9) == r.one()`
///   in `GR(8, d)`. Two unbound elements compare in `Z` (so `unbound(8) != unbound(0)`,
///   although both bind to `0` in `GR(8, d)`); two bound elements are equal iff same ring and
///   same coefficients. Transitivity can fail only across rings: `unbound(0)` equals the zero
///   of every Galois ring while those zeros differ pairwise. There is no `Hash` impl.
/// * **`is_zero`/`is_one`** look at `c` in `Z`. Generic code that never meets a bound element
///   computes in `Z`: eight `R::one()`s add up to `unbound(8)`, which is not `is_zero()`,
///   although `8 = 0` in `GR(8, d)`.
/// * **containers** (polynomials, matrices, ...) bind coefficient by coefficient, so a
///   container built from `R::one()`/`R::zero()` (for instance with
///   `UnivariatePolynomial::var()` or `UnivariatePolynomial::constant(R::one())`) keeps an
///   unbound integer in every coefficient that never met a bound element, and its degree and
///   zero tests see those integers in `Z`. Over `GR(8, 2)` with generator `w`,
///   `(x + 1)^8 - x^8 + w` built from `var()` and `R::one()` keeps the coefficients
///   `[1 + w, 8, 28, 56, 70, 56, 28, 8]` (only the constant term met `w`), so it has degree 7
///   and differs from the same polynomial built from `r.zero()`/`r.one()`, which is
///   `(1 + w) + 4x^2 + 6x^4 + 4x^6` of degree 6. Build containers from
///   [`GaloisRing::zero`]/[`GaloisRing::one`] (bound elements) whenever degrees, zero tests or
///   equality matter; a polynomial already built can be bound with `scalar_mul(&r.one())`.
/// * **the `Ring` hooks** (`unit_status`, `nilpotent_status`, `unit_inverse`,
///   `multiplicative_order_status`) and the inherent [`is_unit`](Self::is_unit),
///   [`unit_inverse`](Self::unit_inverse), [`multiplicative_order`](Self::multiplicative_order),
///   [`is_idempotent`](Self::is_idempotent) decide only what holds in **every** Galois ring
///   (`crate::ring_hooks`): `±1` are units, `0` is nilpotent and not a unit, `1` has order 1.
///   Anything else (e.g. whether `2` is a unit, or the order of `-1`, which is `1` in
///   `GR(2, d)`) is `None`/`Undecided`/an `Err` on the hooks, and a panic on the `bool`- or
///   `Option`-valued inherent methods, never a guess.
/// * **binary inherent operations** ([`quotrem`](Self::quotrem), [`gcd`](Self::gcd),
///   [`lcm`](Self::lcm), [`xgcd`](Self::xgcd), ...): an unbound operand is bound into the
///   other operand's ring. With both operands unbound, the `Result`-valued ones return
///   `Err(MathError::InvalidArgument)` and `gcd`/`lcm` panic.
/// * operations that *need* the parent ([`ring`](Self::ring), [`valuation`](Self::valuation),
///   [`residue`](Self::residue), [`project`](Self::project),
///   [`teichmuller`](Self::teichmuller), [`frobenius`](Self::frobenius), the inherent
///   [`pow`](Self::pow), [`FiniteChainRing`] methods, ...) panic on an unbound element with a
///   message that says so, rather than fabricate an answer.
/// * [`eltseq`](Self::eltseq) of an unbound element is the one-element slice `[c]` (the
///   unreduced integer constant), **not** a length-`d` sequence over `Z/p^a`.
/// * [`DiscreteValuation::valuation`]`(&r, x)` knows the parent `r`, so it binds an unbound
///   `x` into `r` first; [`Parent::contains`] is `false` for an unbound element.
#[derive(Clone)]
pub struct GaloisRingElement {
    /// `[a0, ..., a_{d-1}]` in `[0, p^a)` for a bound element; for an unbound element, the
    /// single integer constant `c`.
    coeffs: Vec<Integer>,
    /// The parent ring; `None` is the unbound sentinel of `Ring::zero()`/`Ring::one()` (see
    /// the type docs).
    ring: Option<GaloisRing>,
}

impl GaloisRingElement {
    /// The parent ring. Panics for an unbound element (from `Ring::zero()`/`Ring::one()`, see
    /// the type docs): it has no parent until it meets a bound element.
    pub fn ring(&self) -> &GaloisRing {
        self.require_ring("ring()")
    }

    /// Whether the element is bound to a parent ring (`false` only for the unbound
    /// `Ring::zero()`/`Ring::one()` sentinel and what is computed from it alone, see the type
    /// docs).
    pub fn is_bound(&self) -> bool {
        self.ring.is_some()
    }

    /// The coefficient sequence `[a0, ..., a_{d-1}]` over `Z/p^a` (MAGMA `Eltseq`). For an
    /// unbound element (see the type docs) this is the single unreduced integer constant
    /// `[c]`, not a length-`d` sequence.
    pub fn eltseq(&self) -> &[Integer] {
        &self.coeffs
    }

    /// An unbound element holding the integer constant `c` (see the type docs).
    fn unbound(c: Integer) -> Self {
        GaloisRingElement {
            coeffs: vec![c],
            ring: None,
        }
    }

    /// The parent ring, or a panic naming `op` for an unbound element.
    fn require_ring(&self, op: &str) -> &GaloisRing {
        self.ring.as_ref().unwrap_or_else(|| {
            panic!(
                "GaloisRingElement::{op} needs a parent ring, but this element is unbound \
                 (it came from Ring::zero()/Ring::one()); bind it by combining it with an \
                 element of a GaloisRing, or construct it with GaloisRing::zero()/one()/from_int()"
            )
        })
    }

    /// The integer constant of an unbound element. Only for `self.ring.is_none()`.
    fn unbound_value(&self) -> &Integer {
        debug_assert!(self.ring.is_none() && self.coeffs.len() == 1);
        &self.coeffs[0]
    }

    /// Bind an unbound element into `r` by the canonical map `Z → GR(p^a, d)`.
    fn bind_into(&self, r: &GaloisRing) -> Self {
        r.from_int(self.unbound_value().clone())
    }

    /// A question about an unbound constant whose answer differs between Galois rings.
    fn undecided_unbound(&self, op: &str) -> ! {
        panic!(
            "GaloisRingElement::{op}: the answer for the unbound integer constant {} depends on \
             the Galois ring it binds to; bind it first (combine it with an element of a \
             GaloisRing)",
            self.coeffs[0]
        )
    }

    /// Coerce the operands of a binary operator: one unbound operand is bound into the other's
    /// ring; two bound operands must share their ring (else the historical panic). `Err` when
    /// both are unbound (the operation then runs in `Z`).
    fn coerce_pair(self, other: Self) -> std::result::Result<(Self, Self), (Self, Self)> {
        match (&self.ring, &other.ring) {
            (Some(a), Some(b)) => {
                assert!(a.data == b.data, "operands live in different Galois rings");
                Ok((self, other))
            }
            (Some(r), None) => {
                let bound = other.bind_into(r);
                Ok((self, bound))
            }
            (None, Some(r)) => {
                let bound = self.bind_into(r);
                Ok((bound, other))
            }
            (None, None) => Err((self, other)),
        }
    }

    /// [`coerce_pair`](Self::coerce_pair) for the binary inherent methods that borrow their
    /// operands: `None` when both are unbound.
    fn bind_pair(&self, other: &Self) -> Option<(Self, Self)> {
        match (&self.ring, &other.ring) {
            (Some(a), Some(b)) => {
                assert!(a.data == b.data, "operands live in different Galois rings");
                Some((self.clone(), other.clone()))
            }
            (Some(r), None) => Some((self.clone(), other.bind_into(r))),
            (None, Some(r)) => Some((self.bind_into(r), other.clone())),
            (None, None) => None,
        }
    }

    fn data(&self, op: &str) -> &GaloisRingData {
        &self.require_ring(op).data
    }
    fn pa(&self) -> &Integer {
        &self.data("pa()").pa
    }
    fn p(&self) -> &Integer {
        &self.data("p()").p
    }

    fn assert_same(&self, other: &Self) {
        assert!(
            self.data("arithmetic") == other.data("arithmetic"),
            "operands live in different Galois rings"
        );
    }

    /// The `p`-adic valuation `v(x)`: the largest power of `p` dividing every
    /// coefficient. By convention `v(0) = a` (the nilpotency index). Panics for an unbound
    /// element (`a` is unknown; see the type docs).
    pub fn valuation(&self) -> u32 {
        let a = self.data("valuation()").a;
        let p = self.p();
        let mut v = a;
        for c in &self.coeffs {
            if c.is_zero() {
                continue;
            }
            let cv = c.valuation(p).min(a);
            if cv < v {
                v = cv;
            }
        }
        v
    }

    /// Whether the element is a unit (MAGMA `IsUnit`): `v(x) == 0`.
    ///
    /// For an unbound constant `c`: `true` for `±1`, `false` for `0`, and a panic for any other
    /// `c` (`2` is a unit in `GR(27, 2)` and nilpotent in `GR(8, 2)`; see the type docs).
    pub fn is_unit(&self) -> bool {
        if self.ring.is_none() {
            return crate::ring_hooks::sentinel_unit_status(self.unbound_value())
                .unwrap_or_else(|| self.undecided_unbound("is_unit()"));
        }
        self.valuation() == 0
    }

    /// Whether the element is nilpotent (MAGMA `IsNilpotent`): equivalently a
    /// non-unit (in a finite chain ring every non-unit lies in the nilpotent
    /// maximal ideal). Unbound constants as for [`is_unit`](Self::is_unit).
    pub fn is_nilpotent(&self) -> bool {
        !self.is_unit()
    }

    /// Whether the element is a zero divisor (MAGMA `IsZeroDivisor`): non-zero
    /// and not a unit. Unbound constants as for [`is_unit`](Self::is_unit).
    pub fn is_zero_divisor(&self) -> bool {
        !self.is_zero() && !self.is_unit()
    }

    /// Whether `x^2 == x` (MAGMA `IsIdempotent`). For an unbound constant: `true` for `0` and
    /// `1`, a panic for any other `c` (`9` is `1` in `GR(8, d)`, idempotent; it is not
    /// idempotent in `GR(27, d)`).
    pub fn is_idempotent(&self) -> bool {
        if self.ring.is_none() {
            let c = self.unbound_value();
            if c.is_zero() || c.is_one() {
                return true;
            }
            self.undecided_unbound("is_idempotent()");
        }
        (self.clone() * self.clone()) == *self
    }

    /// Reduce to the residue field `GF(p^d)` (the map `R -> R/(p)`). Panics for an unbound
    /// element.
    pub fn residue(&self) -> FiniteFieldElement {
        let data = self.data("residue()");
        let p = &data.p;
        let coeffs: Vec<Integer> = self.coeffs.iter().map(|c| red(c.clone(), p)).collect();
        data.residue.element(coeffs)
    }

    /// The canonical projection of this element into `GR(p^b, d)` for
    /// `1 <= b <= a` (coefficients reduced mod `p^b`). This is the canonical
    /// surjective ring homomorphism `GR(p^a, d) -> GR(p^b, d)`; the parent of
    /// the result is [`GaloisRing::reduce_precision`]`(b)`. Panics for an unbound element.
    pub fn project(&self, b: u32) -> Result<GaloisRingElement> {
        let target = self.require_ring("project()").reduce_precision(b)?;
        Ok(target.element(self.coeffs.clone()))
    }

    /// The Teichmüller representative of this element's residue class: the
    /// unique `t` with `t^{p^d} = t` and `t ≡ self (mod p)`. Panics for an unbound element.
    ///
    /// Computed as the limit of `x ↦ x^{p^d}`: if `x ≡ t (mod p^k)` then
    /// `x^{p^d} ≡ t^{p^d} = t (mod p^{k+d})`, so the iteration reaches the
    /// fixed point within `ceil((a-1)/d) <= a - 1` steps.
    pub fn teichmuller(&self) -> GaloisRingElement {
        let data = self.data("teichmuller()");
        let q = data.p.pow(data.d as u32); // p^d
        let mut t = self.clone();
        for _ in 0..data.a {
            let next = t.pow(&q);
            if next == t {
                break;
            }
            t = next;
        }
        debug_assert!(t.pow(&q) == t, "Teichmüller iteration failed to converge");
        t
    }

    /// The Teichmüller digit expansion: the unique `[t_0, ..., t_{a-1}]` with
    /// every `t_i` a Teichmüller representative (`t_i^{p^d} = t_i`) and
    /// `self = t_0 + t_1 p + ... + t_{a-1} p^{a-1}`. Panics for an unbound element.
    pub fn teichmuller_digits(&self) -> Vec<GaloisRingElement> {
        let data = self.data("teichmuller_digits()");
        let a = data.a;
        let p = data.p.clone();
        let pa = data.pa.clone();
        let mut digits = Vec::with_capacity(a as usize);
        let mut y = self.clone();
        for _ in 0..a {
            let t = y.teichmuller();
            // y <- (y - t)/p; exact on canonical representatives since y ≡ t (mod p).
            let coeffs: Vec<Integer> = y
                .coeffs
                .iter()
                .zip(t.coeffs.iter())
                .map(|(yc, tc)| {
                    let diff = red(yc.clone() - tc.clone(), &pa);
                    let (q, r) = diff.div_rem(&p).expect("p != 0");
                    debug_assert!(r.is_zero(), "digit extraction not exact");
                    q
                })
                .collect();
            y = GaloisRingElement {
                coeffs,
                ring: self.ring.clone(),
            };
            digits.push(t);
        }
        digits
    }

    /// The Frobenius automorphism `φ` applied to this element: writing
    /// `self = Σ t_i p^i` in Teichmüller digits, `φ(self) = Σ t_i^p p^i`.
    /// Panics for an unbound element.
    ///
    /// `φ` is the unique ring automorphism lifting `x ↦ x^p` on the residue
    /// field; it generates `Gal(GR(p^a, d) / Z_{p^a}) ≅ Z/dZ`, and its fixed
    /// subring is exactly the base ring `Z/p^a = GR(p^a, 1)`.
    pub fn frobenius(&self) -> GaloisRingElement {
        let ring = self.require_ring("frobenius()");
        let p = ring.data.p.clone();
        let p_elt = ring.from_int(p.clone());
        let mut result = ring.zero();
        let mut pk = ring.one(); // p^i as a ring element
        for t in self.teichmuller_digits() {
            result = result + t.pow(&p) * pk.clone();
            pk = pk * p_elt.clone();
        }
        result
    }

    /// The `k`-th power `φ^k` of the Frobenius automorphism (reduced mod the
    /// order `d` of the Galois group). Panics for an unbound element.
    pub fn frobenius_power(&self, k: u32) -> GaloisRingElement {
        let k = k % (self.data("frobenius_power()").d as u32);
        let mut x = self.clone();
        for _ in 0..k {
            x = x.frobenius();
        }
        x
    }

    /// `self^exp` for a non-negative integer exponent. Panics for an unbound element (use
    /// [`Ring::pow`], which computes an unbound power in `Z`, for a `u32` exponent).
    pub fn pow(&self, exp: &Integer) -> GaloisRingElement {
        let mut result = self.require_ring("pow()").one();
        let mut base = self.clone();
        let mut e = exp.clone();
        let two = Integer::from(2);
        while e > Integer::zero() {
            if red(e.clone(), &two).is_one() {
                result = result * base.clone();
            }
            e = e / two.clone();
            if e > Integer::zero() {
                base = base.clone() * base.clone();
            }
        }
        result
    }

    /// Divide every coefficient by `p^v` exactly (valid when `v <= v(self)`),
    /// yielding the "unit part" `u` with `self = p^v · u`.
    fn unit_part(&self, v: u32) -> GaloisRingElement {
        let pv = self.p().pow(v);
        let pa = self.pa();
        let coeffs: Vec<Integer> = self
            .coeffs
            .iter()
            .map(|c| {
                let (q, _r) = c.div_rem(&pv).unwrap();
                red(q, pa)
            })
            .collect();
        GaloisRingElement {
            coeffs,
            ring: self.ring.clone(),
        }
    }

    /// The inverse of a **unit** (`Err(NotInvertible)` for a non-unit), computed by lifting
    /// the residue-field inverse with Newton iteration.
    ///
    /// For an unbound constant `c` (see the type docs): `±1` are their own inverses, `0`
    /// gives `Err(NotInvertible)`, and any other `c` gives `Err(InvalidArgument)` (whether it
    /// is a unit depends on the ring it binds to).
    pub fn unit_inverse(&self) -> Result<GaloisRingElement> {
        if self.ring.is_none() {
            return crate::ring_hooks::sentinel_unit_inverse(self.unbound_value(), self);
        }
        if !self.is_unit() {
            return Err(MathError::NotInvertible);
        }
        let ring = self.require_ring("unit_inverse()");
        let a = ring.data.a;
        // initial approximation: inverse in the residue field, lifted.
        let inv_bar = self.residue().inverse()?;
        let mut w = ring.element(inv_bar.eltseq().to_vec());
        // Newton: w <- w * (2 - self*w); p-adic precision doubles each step.
        let two = ring.from_int(Integer::from(2));
        let mut prec = 1u32;
        while prec < a {
            let uw = self.clone() * w.clone();
            let corr = two.clone() - uw;
            w = w * corr;
            prec *= 2;
        }
        Ok(w)
    }

    /// `Err(InvalidArgument)` for a binary inherent operation on two unbound operands.
    fn both_unbound(op: &str) -> MathError {
        MathError::InvalidArgument(format!(
            "GaloisRingElement::{op}: both operands are unbound integer constants (from \
             Ring::zero()/Ring::one()); the result depends on the Galois ring they bind to"
        ))
    }

    /// Euclidean division: return `(q, r)` with `self = q·other + r` and either
    /// `r = 0` or `v(r) < v(other)` (MAGMA `Quotrem`). One unbound operand is bound into the
    /// other's ring; two unbound operands give `Err(InvalidArgument)`.
    pub fn quotrem(
        &self,
        other: &GaloisRingElement,
    ) -> Result<(GaloisRingElement, GaloisRingElement)> {
        let (a, b) = self
            .bind_pair(other)
            .ok_or_else(|| Self::both_unbound("quotrem()"))?;
        a.quotrem_bound(&b)
    }

    fn quotrem_bound(
        &self,
        other: &GaloisRingElement,
    ) -> Result<(GaloisRingElement, GaloisRingElement)> {
        self.assert_same(other);
        let ring = self.require_ring("quotrem()");
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok((ring.zero(), ring.zero()));
        }
        let va = self.valuation();
        let vb = other.valuation();
        if va < vb {
            return Ok((ring.zero(), self.clone()));
        }
        let ua = self.unit_part(va);
        let ub = other.unit_part(vb);
        let ubinv = ub.unit_inverse()?;
        let pk = ring.from_int(self.p().pow(va - vb));
        let q = pk * ua * ubinv;
        let r = self.clone() - q.clone() * other.clone();
        Ok((q, r))
    }

    /// Euclidean quotient (`self div other`).
    pub fn div_euclid(&self, other: &GaloisRingElement) -> Result<GaloisRingElement> {
        Ok(self.quotrem(other)?.0)
    }

    /// Euclidean remainder (`self mod other`).
    pub fn rem_euclid(&self, other: &GaloisRingElement) -> Result<GaloisRingElement> {
        Ok(self.quotrem(other)?.1)
    }

    /// Greatest common divisor: the canonical associate `p^{min(v(a),v(b))}`
    /// (MAGMA `GCD`). One unbound operand is bound into the other's ring; panics if both are
    /// unbound.
    pub fn gcd(&self, other: &GaloisRingElement) -> GaloisRingElement {
        let (x, y) = self
            .bind_pair(other)
            .unwrap_or_else(|| panic!("{}", Self::both_unbound("gcd()")));
        let ring = x.require_ring("gcd()");
        let a = ring.data.a;
        let va = if x.is_zero() { a } else { x.valuation() };
        let vb = if y.is_zero() { a } else { y.valuation() };
        let m = va.min(vb);
        if m >= a {
            ring.zero()
        } else {
            ring.from_int(ring.data.p.pow(m))
        }
    }

    /// Least common multiple: the canonical associate `p^{max(v(a),v(b))}`
    /// (MAGMA `LCM`), or `0` if either operand is `0`. One unbound operand is bound into the
    /// other's ring; panics if both are unbound.
    pub fn lcm(&self, other: &GaloisRingElement) -> GaloisRingElement {
        let (x, y) = self
            .bind_pair(other)
            .unwrap_or_else(|| panic!("{}", Self::both_unbound("lcm()")));
        let ring = x.require_ring("lcm()");
        if x.is_zero() || y.is_zero() {
            return ring.zero();
        }
        let a = ring.data.a;
        let m = x.valuation().max(y.valuation());
        if m >= a {
            ring.zero()
        } else {
            ring.from_int(ring.data.p.pow(m))
        }
    }

    /// Extended GCD: return `(g, s, t)` with `g = s·self + t·other`
    /// (MAGMA `XGCD`), `g` the canonical associate `p^{min}`. One unbound operand is bound
    /// into the other's ring; two unbound operands give `Err(InvalidArgument)`.
    pub fn xgcd(
        &self,
        other: &GaloisRingElement,
    ) -> Result<(GaloisRingElement, GaloisRingElement, GaloisRingElement)> {
        let (x, y) = self
            .bind_pair(other)
            .ok_or_else(|| Self::both_unbound("xgcd()"))?;
        let ring = x.require_ring("xgcd()");
        let a = ring.data.a;
        let va = if x.is_zero() { a } else { x.valuation() };
        let vb = if y.is_zero() { a } else { y.valuation() };
        if va >= a && vb >= a {
            return Ok((ring.zero(), ring.zero(), ring.zero()));
        }
        if va <= vb {
            // g = p^va = ua^{-1} * x
            let ua = x.unit_part(va);
            let s = ua.unit_inverse()?;
            let g = s.clone() * x.clone();
            Ok((g, s, ring.zero()))
        } else {
            let ub = y.unit_part(vb);
            let t = ub.unit_inverse()?;
            let g = t.clone() * y.clone();
            Ok((g, ring.zero(), t))
        }
    }
}

impl fmt::Debug for GaloisRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl fmt::Display for GaloisRingElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut terms: Vec<String> = Vec::new();
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            match i {
                0 => terms.push(format!("{c}")),
                1 => terms.push(format!("{c}*w")),
                _ => terms.push(format!("{c}*w^{i}")),
            }
        }
        if terms.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "{}", terms.join(" + "))
        }
    }
}

impl PartialEq for GaloisRingElement {
    /// Coercing (binding) equality, matching the arithmetic (see the type docs):
    ///
    /// * **bound vs bound**: equal iff same ring and same coefficients (unchanged); different
    ///   rings are simply unequal, never a panic.
    /// * **unbound vs bound**: the unbound constant is bound into the other operand's ring
    ///   first, so `R::zero() == r.zero()` and `unbound(9) == r.one()` in `GR(8, d)`.
    ///   Symmetric.
    /// * **unbound vs unbound**: equality of the integer constants in `Z`.
    fn eq(&self, other: &Self) -> bool {
        match (&self.ring, &other.ring) {
            (Some(a), Some(b)) => a.data == b.data && self.coeffs == other.coeffs,
            (None, None) => self.coeffs == other.coeffs,
            (None, Some(r)) => self.bind_into(r).coeffs == other.coeffs,
            (Some(r), None) => other.bind_into(r).coeffs == self.coeffs,
        }
    }
}

/// `Eq` marker with a documented caveat: equality is reflexive and symmetric, but
/// transitivity can fail in one corner. The unbound sentinel equals the bound zero (resp. one)
/// of every Galois ring, while bound elements of different rings stay unequal. That corner's
/// arithmetic already panics, and `GaloisRingElement` implements no `Hash`.
impl Eq for GaloisRingElement {}

impl Add for GaloisRingElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (a, b) = match self.coerce_pair(other) {
            Ok(pair) => pair,
            // Both unbound: add in Z (see the type docs).
            Err((a, b)) => {
                return Self::unbound(a.unbound_value().clone() + b.unbound_value().clone())
            }
        };
        let pa = a.pa().clone();
        let coeffs = a
            .coeffs
            .iter()
            .zip(b.coeffs.iter())
            .map(|(x, y)| red(x.clone() + y.clone(), &pa))
            .collect();
        GaloisRingElement {
            coeffs,
            ring: a.ring,
        }
    }
}

impl Sub for GaloisRingElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let (a, b) = match self.coerce_pair(other) {
            Ok(pair) => pair,
            // Both unbound: subtract in Z (see the type docs).
            Err((a, b)) => {
                return Self::unbound(a.unbound_value().clone() - b.unbound_value().clone())
            }
        };
        let pa = a.pa().clone();
        let coeffs = a
            .coeffs
            .iter()
            .zip(b.coeffs.iter())
            .map(|(x, y)| red(x.clone() - y.clone(), &pa))
            .collect();
        GaloisRingElement {
            coeffs,
            ring: a.ring,
        }
    }
}

impl Mul for GaloisRingElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (a, b) = match self.coerce_pair(other) {
            Ok(pair) => pair,
            // Both unbound: multiply in Z (see the type docs).
            Err((a, b)) => {
                return Self::unbound(a.unbound_value().clone() * b.unbound_value().clone())
            }
        };
        let pa = a.pa().clone();
        let mut prod = vec![Integer::zero(); a.coeffs.len() + b.coeffs.len() - 1];
        for (i, ai) in a.coeffs.iter().enumerate() {
            if ai.is_zero() {
                continue;
            }
            for (j, bj) in b.coeffs.iter().enumerate() {
                prod[i + j] = red(prod[i + j].clone() + ai.clone() * bj.clone(), &pa);
            }
        }
        let ring = a.ring;
        let coeffs = ring
            .as_ref()
            .expect("coerce_pair returns bound operands")
            .reduce(prod);
        GaloisRingElement { coeffs, ring }
    }
}

impl Neg for GaloisRingElement {
    type Output = Self;
    fn neg(self) -> Self {
        if self.ring.is_none() {
            // Unbound: negate in Z (see the type docs).
            return Self::unbound(-self.unbound_value().clone());
        }
        let pa = self.pa().clone();
        let coeffs = self.coeffs.iter().map(|c| red(-c.clone(), &pa)).collect();
        GaloisRingElement {
            coeffs,
            ring: self.ring,
        }
    }
}

impl Ring for GaloisRingElement {
    /// The additive identity, as an *unbound* element (see the type docs): the canonical image
    /// of `0 ∈ Z` in any `GR(p^a, d)`, bound on first contact with a bound element. Prefer
    /// [`GaloisRing::zero`] when the parent is known.
    fn zero() -> Self {
        Self::unbound(Integer::zero())
    }
    /// The multiplicative identity, as an *unbound* element (see [`Ring::zero`] above and the
    /// type docs). Prefer [`GaloisRing::one`] when the parent is known.
    fn one() -> Self {
        Self::unbound(Integer::one())
    }
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
    fn is_one(&self) -> bool {
        self.coeffs[0].is_one() && self.coeffs[1..].iter().all(|c| c.is_zero())
    }

    // Element predicates on the generic `Ring` path (MAGMA Ch. 48 `IsUnit`: p-adic
    // valuation zero; `IsNilpotent`: every non-unit of a finite chain ring). For a bound
    // element they delegate to the inherent methods. An unbound element is an integer
    // constant awaiting the map Z -> GR(p^a, d), so only what holds in every Galois ring is
    // decided (`crate::ring_hooks`). None of them calls the static `zero()`/`one()`.

    fn unit_status(&self) -> Option<bool> {
        if self.ring.is_none() {
            return crate::ring_hooks::sentinel_unit_status(self.unbound_value());
        }
        Some(GaloisRingElement::is_unit(self))
    }

    fn nilpotent_status(&self) -> Option<bool> {
        if self.ring.is_none() {
            return crate::ring_hooks::sentinel_nilpotent_status(self.unbound_value());
        }
        Some(GaloisRingElement::is_nilpotent(self))
    }

    fn unit_inverse(&self) -> Result<Self> {
        // The inherent method handles unbound constants with the same sentinel table.
        GaloisRingElement::unit_inverse(self)
    }

    /// The multiplicative order of a unit (see [`GaloisRingElement::multiplicative_order`],
    /// which gives the exact `Integer` when it exceeds `u64` and this hook reports
    /// `FiniteExceedsU64`). Beyond the handbook: MAGMA Ch. 48 lists no `Order`.
    fn multiplicative_order_status(&self) -> rustmath_core::MultiplicativeOrder {
        use rustmath_core::MultiplicativeOrder;
        if self.ring.is_none() {
            return crate::ring_hooks::sentinel_order(self.unbound_value());
        }
        match self.order_by_residue_lifting() {
            UnitOrder::NotAUnit => MultiplicativeOrder::NotAUnit,
            UnitOrder::Finite(order) => crate::ring_hooks::order_from_integer(&order),
            // Unreachable by the proof on `order_by_residue_lifting`; never guess.
            UnitOrder::Inconsistent => MultiplicativeOrder::Undecided,
        }
    }
}

/// Outcome of [`GaloisRingElement::order_by_residue_lifting`].
enum UnitOrder {
    NotAUnit,
    Finite(Integer),
    /// The residue of a unit had no order, or the proven bound `j <= a - 1` failed: an
    /// internal inconsistency that the proof rules out, never an order.
    Inconsistent,
}

impl GaloisRingElement {
    /// The multiplicative order of this element (MAGMA `Order(x)` as for Z/mZ, Ch. 19, and
    /// GF(q), Ch. 21; MAGMA Ch. 48 itself lists no `Order` for Galois rings): `Some(n)`
    /// with `n` the least `n >= 1` such that `x^n = 1`, and `None` for a non-unit (MAGMA
    /// returns 0 there). Exact for every size; [`Ring::multiplicative_order_status`]
    /// reports `FiniteExceedsU64` when `n >= 2^64` and the exact value is this one.
    ///
    /// Example: in `GR(2^66, 1) = Z/2^66`, `3` has order `2^64` (PARI/GP
    /// `znorder(Mod(3, 2^66))`).
    ///
    /// For an unbound constant `c` (see the type docs): `Some(1)` for `1`, `None` for `0`.
    ///
    /// # Panics
    ///
    /// - For an unbound constant other than `0` and `1`: its order depends on the ring it
    ///   binds to (`-1` has order 1 in `GR(2, d)` and 2 elsewhere).
    /// - On an internal inconsistency that the proof on the algorithm rules out (the
    ///   residue of a unit having no order, or the `p`-power bound `j <= a - 1` failing).
    pub fn multiplicative_order(&self) -> Option<Integer> {
        if self.ring.is_none() {
            return match crate::ring_hooks::sentinel_order(self.unbound_value()) {
                rustmath_core::MultiplicativeOrder::Finite(k) => Some(Integer::from(k)),
                rustmath_core::MultiplicativeOrder::NotAUnit => None,
                _ => self.undecided_unbound("multiplicative_order()"),
            };
        }
        match self.order_by_residue_lifting() {
            UnitOrder::NotAUnit => None,
            UnitOrder::Finite(order) => Some(order),
            UnitOrder::Inconsistent => unreachable!(
                "GaloisRingElement::multiplicative_order: the order of {self} is not m * p^j \
                 with j <= a - 1, contradicting the structure of GR(p^a, d)^*"
            ),
        }
    }

    /// The order of `x` in `R = GR(p^a, d)`, shared by [`GaloisRingElement::multiplicative_order`]
    /// and the `Ring::multiplicative_order_status` hook.
    ///
    /// Algorithm: let `m` be the order of the residue `x mod p` in `GF(p^d)^*` (exact,
    /// order-from-factorisation of `p^d - 1`, Ch. 21). Then `ord(x) = m * p^j`, where `p^j`
    /// is the order of `x^m`, found by repeated `p`-th powering with `j <= a - 1`.
    ///
    /// Proof: reduction mod `p` is a ring homomorphism `R -> GF(p^d)` with kernel `pR`, so
    /// `m | ord(x)` and `ord(x) = m * ord(x^m)`. `x^m` lies in `1 + pR`, and for `y = 1 + p t`
    /// induction on the binomial expansion gives `y^(p^k) = 1 mod p^(k+1)`, so `1 + pR` is a
    /// `p`-group of exponent dividing `p^(a-1)`. References: B. R. McDonald, *Finite Rings
    /// with Identity* (1974), Ch. XVI; Z.-X. Wan, *Lectures on Finite Fields and Galois
    /// Rings* (2003), Ch. 14 (structure of `GR(p^a, d)^*`).
    ///
    /// Only for bound elements (the callers handle the unbound sentinel first).
    fn order_by_residue_lifting(&self) -> UnitOrder {
        let a = self.data("multiplicative_order()").a;
        if !GaloisRingElement::is_unit(self) {
            return UnitOrder::NotAUnit;
        }
        let m = match self.residue().multiplicative_order() {
            Some(m) => m,
            None => return UnitOrder::Inconsistent, // unreachable: the residue of a unit is nonzero
        };
        let p = self.p().clone();
        let mut y = GaloisRingElement::pow(self, &m);
        let mut order = m;
        // y = x^(m p^j) for j = 0, 1, ..., a - 1; the proof bounds j by a - 1.
        for _ in 0..a {
            if y.is_one() {
                return UnitOrder::Finite(order);
            }
            y = GaloisRingElement::pow(&y, &p);
            order = order * p.clone();
        }
        // Unreachable by the proof above; never return an unproven order.
        UnitOrder::Inconsistent
    }
}

impl CommutativeRing for GaloisRingElement {}

// Non-domain markers (NOT Field / IntegralDomain): GR(p^a, d) is a finite chain
// ring — a local principal ideal ring.
impl LocalRing for GaloisRingElement {
    /// The inherent [`GaloisRingElement::is_unit`]. For an unbound constant `c` (from
    /// `Ring::zero()`/`Ring::one()`, see the type docs) it is `true` for `±1` and `false` for
    /// `0`, and it **panics** for any other `c`, whose unit status depends on the ring (`2` is a
    /// unit in `GR(27, 2)` and nilpotent in `GR(8, 2)`). The `Ring::unit_status` hook answers
    /// `None` (undecided) there instead of panicking.
    fn is_unit(&self) -> bool {
        GaloisRingElement::is_unit(self)
    }
}

impl PrincipalIdealRing for GaloisRingElement {}

/// Both methods need the parent ring: they panic for an unbound element (see
/// [`GaloisRingElement`]).
impl FiniteChainRing for GaloisRingElement {
    fn nilpotency_index(&self) -> u64 {
        self.data("nilpotency_index()").a as u64
    }
    fn chain_valuation(&self) -> u64 {
        self.valuation() as u64
    }
}

/// The `p`-adic valuation makes `GR(p^a, d)` a Euclidean (chain) ring.
impl DiscreteValuation<GaloisRingElement> for GaloisRing {
    /// `v(x)`, with `v(0) = INFINITY`. The parent is known here, so an unbound `x` (from
    /// `Ring::zero()`/`Ring::one()`) is first bound into this ring: `unbound(4)` has valuation
    /// 2 and `unbound(8)` is `0` in `GR(8, d)`.
    fn valuation(&self, x: &GaloisRingElement) -> i64 {
        let bound;
        let x = if x.is_bound() {
            x
        } else {
            bound = x.bind_into(self);
            &bound
        };
        if x.is_zero() {
            return Self::INFINITY;
        }
        x.valuation() as i64
    }
    fn uniformizer(&self) -> GaloisRingElement {
        self.from_int(self.data.p.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gr_8_2_basic() {
        // GR(2^3, 2): H48E1. characteristic 8, cardinality 64, w^2 + w + 1 = 0.
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        assert_eq!(r.characteristic(), &Integer::from(8));
        assert_eq!(r.cardinality(), Integer::from(64));
        assert_eq!(r.degree(), 2);
        // w^2 = -(w+1) mod 8  => w^2 + w + 1 = 0
        let w = r.generator();
        let w2 = w.clone() * w.clone();
        let should_be_zero = w2 + w.clone() + r.one();
        assert!(should_be_zero.is_zero());
        // Eltseq round trip: R![1,2] -> [1,2]
        let e = r.element(vec![Integer::from(1), Integer::from(2)]);
        assert_eq!(e.eltseq(), &[Integer::from(1), Integer::from(2)]);
    }

    #[test]
    fn gr_is_not_a_field_for_a_gt_1() {
        let r = GaloisRing::new(Integer::from(3), 3, 2).unwrap();
        assert!(!r.is_field());
        // 3 and 9 are non-units (H48E2 remark).
        assert!(!r.from_int(Integer::from(3)).is_unit());
        assert!(!r.from_int(Integer::from(9)).is_unit());
        assert!(r.from_int(Integer::from(3)).is_zero_divisor());
        // p * p^{a-1} = 0: 3 * 9 = 27 = 0 mod 27
        let three = r.from_int(Integer::from(3));
        let nine = r.from_int(Integer::from(9));
        assert!((three * nine).is_zero());
    }

    #[test]
    fn valuation_and_units() {
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        // 1 is a unit, v = 0
        assert_eq!(r.one().valuation(), 0);
        assert!(r.one().is_unit());
        // 2 has valuation 1, 4 valuation 2
        assert_eq!(r.from_int(Integer::from(2)).valuation(), 1);
        assert_eq!(r.from_int(Integer::from(4)).valuation(), 2);
        // v(0) = a = 3 by convention
        assert_eq!(r.zero().valuation(), 3);
        // generator w is a unit
        assert!(r.generator().is_unit());
    }

    #[test]
    fn finite_chain_ring_markers() {
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        let two = r.from_int(Integer::from(2));
        assert_eq!(FiniteChainRing::nilpotency_index(&two), 3);
        assert_eq!(FiniteChainRing::chain_valuation(&two), 1);
        assert!(!LocalRing::is_unit(&two));
        assert!(LocalRing::is_unit(&r.one()));
    }

    #[test]
    fn unit_inverse_newton() {
        let r = GaloisRing::new(Integer::from(3), 3, 2).unwrap();
        // a unit: w + 1
        let u = r.element(vec![Integer::from(1), Integer::from(1)]);
        assert!(u.is_unit());
        let inv = u.unit_inverse().unwrap();
        assert!((u * inv).is_one());
        // a base-ring unit 5 mod 27
        let five = r.from_int(Integer::from(5));
        let fi = five.unit_inverse().unwrap();
        assert!((five * fi).is_one());
    }

    #[test]
    fn euclidean_quotrem_gcd_xgcd() {
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        // gcd(2, 4) = 2 (min valuation 1)
        let two = r.from_int(Integer::from(2));
        let four = r.from_int(Integer::from(4));
        let g = two.gcd(&four);
        assert_eq!(g.valuation(), 1);
        // xgcd: g = s*2 + t*4
        let (g2, s, t) = two.xgcd(&four).unwrap();
        let check = s * two.clone() + t * four.clone();
        assert_eq!(check, g2);
        assert_eq!(g2.valuation(), 1);
        // quotrem: 4 = q*2 + r with r = 0 (4 divisible by 2)
        let (q, rem) = four.quotrem(&two).unwrap();
        assert!(rem.is_zero());
        assert_eq!(q * two.clone(), four.clone());
        // dividing by a unit gives remainder 0
        let u = r.element(vec![Integer::from(1), Integer::from(1)]);
        let (_, rem2) = four.quotrem(&u).unwrap();
        assert!(rem2.is_zero());
    }

    #[test]
    fn residue_field_map() {
        let r = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        let rf = r.residue_field();
        assert_eq!(rf.order(), Integer::from(4));
        // residue of (2 + w) = w in GF(4) (2 -> 0 mod 2)
        let e = r.element(vec![Integer::from(2), Integer::from(1)]);
        let res = e.residue();
        assert_eq!(res, rf.generator());
    }

    #[test]
    fn valuation_via_trait() {
        let r = GaloisRing::new(Integer::from(5), 2, 1).unwrap(); // = Z/25
        // GR(p^a, 1) = Z/p^a
        assert_eq!(r.cardinality(), Integer::from(25));
        let five = r.from_int(Integer::from(5));
        assert_eq!(DiscreteValuation::valuation(&r, &five), 1);
        assert_eq!(DiscreteValuation::valuation(&r, &r.zero()), i64::MAX);
        assert_eq!(r.uniformizer(), five);
    }

    #[test]
    fn shared_parent() {
        let r1 = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        let r2 = GaloisRing::new(Integer::from(2), 3, 2).unwrap();
        assert!(Arc::ptr_eq(&r1.data, &r2.data));
    }

    /// All 16 elements of GR(4, 2) = Z4[x]/(x^2 + x + 1).
    fn all_gr_4_2(r: &GaloisRing) -> Vec<GaloisRingElement> {
        let mut v = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                v.push(r.element(vec![Integer::from(i), Integer::from(j)]));
            }
        }
        v
    }

    #[test]
    fn gr_4_2_structure() {
        // GR(4, 2) = Z4[x]/(x^2 + x + 1) — every count below verified
        // independently (exhaustive Python model, MAGMA ch48 semantics):
        // 16 elements, unit group of order p^{(a-1)d}(p^d - 1) = 4·3 = 12,
        // maximal ideal (2) of size 4, residue field GF(4).
        let r = GaloisRing::new(Integer::from(2), 2, 2).unwrap();
        assert_eq!(r.cardinality(), Integer::from(16));
        assert_eq!(r.characteristic(), &Integer::from(4));
        assert!(!r.is_field());
        // Conway polynomial for GF(4) is x^2 + x + 1; its lift defines the ring.
        assert!(r.residue_field().is_conway());
        assert_eq!(r.residue_field().order(), Integer::from(4));

        let all = all_gr_4_2(&r);
        assert_eq!(all.len(), 16);
        let units: Vec<_> = all.iter().filter(|e| e.is_unit()).collect();
        assert_eq!(units.len(), 12);
        // every unit really is invertible
        for u in &units {
            let inv = u.unit_inverse().unwrap();
            assert!(((*u).clone() * inv).is_one());
        }
        // the 4 non-units form the maximal ideal (2); with a = 2, m^2 = 0
        let m: Vec<_> = all.iter().filter(|e| !e.is_unit()).collect();
        assert_eq!(m.len(), 4);
        for x in &m {
            assert!(x.is_nilpotent());
            assert!(((*x).clone() * (*x).clone()).is_zero());
            // every non-unit is divisible by the uniformizer 2
            assert!(x.is_zero() || x.valuation() >= 1);
        }
        // zero divisors = non-zero non-units (3 of them)
        assert_eq!(all.iter().filter(|e| e.is_zero_divisor()).count(), 3);
    }

    #[test]
    fn teichmuller_representatives() {
        let r = GaloisRing::new(Integer::from(2), 2, 2).unwrap();
        let q = Integer::from(4); // p^d
        // In GR(4,2) with the Conway lift, w^3 = 1, so the Teichmüller set is
        // {0} ∪ μ_3 = {0, 1, w, w^2 = 3 + 3w} (verified independently).
        let w = r.generator();
        assert_eq!(w.pow(&Integer::from(3)), r.one());
        let all = all_gr_4_2(&r);
        let teich: Vec<_> = all.iter().filter(|e| e.pow(&q) == **e).collect();
        assert_eq!(teich.len(), 4);
        assert!(teich.iter().any(|t| **t == w));
        assert!(teich
            .iter()
            .any(|t| **t == r.element(vec![Integer::from(3), Integer::from(3)])));

        for e in &all {
            let t = e.teichmuller();
            // t is a genuine Teichmüller representative of e's residue class
            assert_eq!(t.pow(&q), t);
            assert_eq!(t.residue(), e.residue());
            // idempotence
            assert_eq!(t.teichmuller(), t);
        }
        // multiplicativity: τ(xy) = τ(x)·τ(y)
        for x in &all {
            for y in &all {
                assert_eq!(
                    (x.clone() * y.clone()).teichmuller(),
                    x.teichmuller() * y.teichmuller()
                );
            }
        }
        // lift from the residue field round-trips
        let rf = r.residue_field();
        let g = rf.generator();
        let t = r.teichmuller_lift(&g).unwrap();
        assert_eq!(t.residue(), g);
        assert_eq!(t.pow(&q), t);
        // an element of a different residue field is rejected
        let other = crate::finite_field::FiniteField::new(Integer::from(3), 2).unwrap();
        assert!(r.teichmuller_lift(&other.generator()).is_err());
    }

    #[test]
    fn teichmuller_digits_reconstruct() {
        // GR(3^3, 2): x = Σ t_i 3^i with every digit a fixed point of x ↦ x^9.
        let r = GaloisRing::new(Integer::from(3), 3, 2).unwrap();
        let q = Integer::from(9);
        let samples = [
            r.element(vec![Integer::from(14), Integer::from(22)]),
            r.element(vec![Integer::from(3), Integer::from(9)]),
            r.zero(),
            r.one(),
            r.generator(),
            r.from_int(Integer::from(26)),
        ];
        for x in &samples {
            let digits = x.teichmuller_digits();
            assert_eq!(digits.len(), 3);
            let mut acc = r.zero();
            for (i, t) in digits.iter().enumerate() {
                assert_eq!(t.pow(&q), *t, "digit {i} is not Teichmüller");
                acc = acc + t.clone() * r.from_int(Integer::from(3).pow(i as u32));
            }
            assert_eq!(acc, *x, "digits do not reconstruct {x}");
        }
    }

    #[test]
    fn frobenius_gr_4_2_fixes_exactly_z4() {
        // Verified independently (exhaustive Python model): φ is a ring
        // automorphism of GR(4,2) with φ² = id whose fixed subring is EXACTLY
        // the 4 constants GR(4,1) = Z4.
        let r = GaloisRing::new(Integer::from(2), 2, 2).unwrap();
        let all = all_gr_4_2(&r);
        for x in &all {
            let fx = x.frobenius();
            // φ has order dividing d = 2
            assert_eq!(fx.frobenius(), *x);
            // φ lifts the residue Frobenius: φ(x) ≡ x^p (mod p)
            assert_eq!(fx.residue(), x.pow(&Integer::from(2)).residue());
            // homomorphism on all 16×16 pairs
            for y in &all {
                assert_eq!(
                    (x.clone() + y.clone()).frobenius(),
                    x.frobenius() + y.frobenius()
                );
                assert_eq!(
                    (x.clone() * y.clone()).frobenius(),
                    x.frobenius() * y.frobenius()
                );
            }
        }
        // fixed subring is exactly Z4 · 1
        let fixed: Vec<_> = all.iter().filter(|x| x.frobenius() == **x).collect();
        assert_eq!(fixed.len(), 4);
        for x in &fixed {
            assert!(
                x.eltseq()[1].is_zero(),
                "Frobenius-fixed element {x} is not a constant"
            );
        }
        // and φ genuinely moves the generator (φ ≠ id)
        let w = r.generator();
        assert_ne!(w.frobenius(), w);
        assert_eq!(
            w.frobenius(),
            r.element(vec![Integer::from(3), Integer::from(3)]) // w^2, since w is Teichmüller
        );
    }

    #[test]
    fn frobenius_deeper_rings() {
        // GR(3^3, 2): φ² = id, homomorphism on sample pairs.
        let r = GaloisRing::new(Integer::from(3), 3, 2).unwrap();
        let samples = [
            r.element(vec![Integer::from(14), Integer::from(22)]),
            r.element(vec![Integer::from(5), Integer::from(1)]),
            r.generator(),
            r.from_int(Integer::from(10)),
            r.element(vec![Integer::from(3), Integer::from(18)]),
        ];
        for x in &samples {
            assert_eq!(x.frobenius().frobenius(), *x);
            assert_eq!(
                x.frobenius().residue(),
                x.pow(&Integer::from(3)).residue()
            );
            for y in &samples {
                assert_eq!(
                    (x.clone() + y.clone()).frobenius(),
                    x.frobenius() + y.frobenius()
                );
                assert_eq!(
                    (x.clone() * y.clone()).frobenius(),
                    x.frobenius() * y.frobenius()
                );
            }
        }
        // base-ring elements are fixed
        assert_eq!(r.from_int(Integer::from(10)).frobenius(), r.from_int(Integer::from(10)));

        // GR(2^3, 3): the Galois group is Z/3 — φ ≠ id, φ² ≠ id, φ³ = id.
        let r3 = GaloisRing::new(Integer::from(2), 3, 3).unwrap();
        let w = r3.generator();
        assert_ne!(w.frobenius(), w);
        assert_ne!(w.frobenius_power(2), w);
        assert_eq!(w.frobenius_power(3), w);
        assert_eq!(w.frobenius_power(3), w.frobenius_power(0));

        // GR(p^a, 1) = Z/p^a: Frobenius is the identity.
        let z25 = GaloisRing::new(Integer::from(5), 2, 1).unwrap();
        let x = z25.from_int(Integer::from(17));
        assert_eq!(x.frobenius(), x);
    }

    #[test]
    fn canonical_projection() {
        let r8 = GaloisRing::new(Integer::from(2), 3, 2).unwrap(); // GR(8, 2)
        let r4 = r8.reduce_precision(2).unwrap();
        assert_eq!(r4.characteristic(), &Integer::from(4));
        assert_eq!(r4.degree(), 2);
        // lands on the very same unique parent as direct construction
        let direct = GaloisRing::new(Integer::from(2), 2, 2).unwrap();
        assert!(Arc::ptr_eq(&r4.data, &direct.data));
        // b = a is the identity
        assert!(Arc::ptr_eq(&r8.reduce_precision(3).unwrap().data, &r8.data));
        // out-of-range precisions are rejected
        assert!(r8.reduce_precision(0).is_err());
        assert!(r8.reduce_precision(4).is_err());

        // projection is a ring homomorphism (checked on sample pairs)
        let samples = [
            r8.element(vec![Integer::from(5), Integer::from(6)]),
            r8.element(vec![Integer::from(7), Integer::from(3)]),
            r8.generator(),
            r8.from_int(Integer::from(6)),
            r8.one(),
        ];
        assert!(r8.one().project(2).unwrap().is_one());
        assert!(r8.zero().project(2).unwrap().is_zero());
        for x in &samples {
            assert_eq!(x.project(3).unwrap(), *x); // identity at full precision
            for y in &samples {
                assert_eq!(
                    (x.clone() + y.clone()).project(2).unwrap(),
                    x.project(2).unwrap() + y.project(2).unwrap()
                );
                assert_eq!(
                    (x.clone() * y.clone()).project(2).unwrap(),
                    x.project(2).unwrap() * y.project(2).unwrap()
                );
            }
        }
        // explicit value: (5 + 6w) mod 4 = 1 + 2w
        let x = r8.element(vec![Integer::from(5), Integer::from(6)]);
        assert_eq!(
            x.project(2).unwrap().eltseq(),
            &[Integer::from(1), Integer::from(2)]
        );
        // b = 1 agrees with the residue map coefficientwise
        let x1 = x.project(1).unwrap();
        assert_eq!(x1.eltseq(), x.residue().eltseq());
        // projection commutes with Frobenius (both are canonical)
        assert_eq!(
            x.frobenius().project(2).unwrap(),
            x.project(2).unwrap().frobenius()
        );
    }
}
