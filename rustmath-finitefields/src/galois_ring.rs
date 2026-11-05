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
            ring: self.clone(),
        }
    }

    /// The one element.
    pub fn one(&self) -> GaloisRingElement {
        let mut c = vec![Integer::zero(); self.data.d];
        c[0] = Integer::one();
        GaloisRingElement {
            coeffs: c,
            ring: self.clone(),
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
            ring: self.clone(),
        }
    }

    /// Coerce an integer `k` (identified with `k mod p^a`).
    pub fn from_int(&self, k: Integer) -> GaloisRingElement {
        let mut c = vec![Integer::zero(); self.data.d];
        c[0] = red(k, &self.data.pa);
        GaloisRingElement {
            coeffs: c,
            ring: self.clone(),
        }
    }

    /// Build an element from a coefficient sequence over the base ring
    /// (MAGMA `R ! [a0, ..., a_{d-1}]`), reducing mod `p^a` and mod `D`.
    pub fn element(&self, coeffs: Vec<Integer>) -> GaloisRingElement {
        let reduced = self.reduce(coeffs);
        GaloisRingElement {
            coeffs: reduced,
            ring: self.clone(),
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
    fn contains(&self, element: &Self::Element) -> bool {
        element.ring.data == self.data
    }
    fn zero(&self) -> Option<Self::Element> {
        Some(GaloisRing::zero(self))
    }
    fn one(&self) -> Option<Self::Element> {
        Some(GaloisRing::one(self))
    }
    fn cardinality(&self) -> Option<usize> {
        use rustmath_core::NumericConversion;
        self.cardinality().to_usize()
    }
    fn name(&self) -> String {
        format!("{self}")
    }
}

// ---------------------------------------------------------------------------
// Element
// ---------------------------------------------------------------------------

/// An element of a [`GaloisRing`] `GR(p^a, d)`, carrying its shared parent.
#[derive(Clone)]
pub struct GaloisRingElement {
    coeffs: Vec<Integer>, // length d, in [0, p^a)
    ring: GaloisRing,
}

impl GaloisRingElement {
    /// The parent ring.
    pub fn ring(&self) -> &GaloisRing {
        &self.ring
    }

    /// The coefficient sequence `[a0, ..., a_{d-1}]` over `Z/p^a` (MAGMA `Eltseq`).
    pub fn eltseq(&self) -> &[Integer] {
        &self.coeffs
    }

    fn pa(&self) -> &Integer {
        &self.ring.data.pa
    }
    fn p(&self) -> &Integer {
        &self.ring.data.p
    }

    fn assert_same(&self, other: &Self) {
        assert!(
            self.ring.data == other.ring.data,
            "operands live in different Galois rings"
        );
    }

    /// The `p`-adic valuation `v(x)`: the largest power of `p` dividing every
    /// coefficient. By convention `v(0) = a` (the nilpotency index).
    pub fn valuation(&self) -> u32 {
        let a = self.ring.data.a;
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
    pub fn is_unit(&self) -> bool {
        self.valuation() == 0
    }

    /// Whether the element is nilpotent (MAGMA `IsNilpotent`): equivalently a
    /// non-unit (in a finite chain ring every non-unit lies in the nilpotent
    /// maximal ideal).
    pub fn is_nilpotent(&self) -> bool {
        !self.is_unit()
    }

    /// Whether the element is a zero divisor (MAGMA `IsZeroDivisor`): non-zero
    /// and not a unit.
    pub fn is_zero_divisor(&self) -> bool {
        !self.is_zero() && !self.is_unit()
    }

    /// Whether `x^2 == x` (MAGMA `IsIdempotent`).
    pub fn is_idempotent(&self) -> bool {
        (self.clone() * self.clone()) == *self
    }

    /// Reduce to the residue field `GF(p^d)` (the map `R -> R/(p)`).
    pub fn residue(&self) -> FiniteFieldElement {
        let p = self.p();
        let coeffs: Vec<Integer> = self.coeffs.iter().map(|c| red(c.clone(), p)).collect();
        self.ring.data.residue.element(coeffs)
    }

    /// The canonical projection of this element into `GR(p^b, d)` for
    /// `1 <= b <= a` (coefficients reduced mod `p^b`). This is the canonical
    /// surjective ring homomorphism `GR(p^a, d) -> GR(p^b, d)`; the parent of
    /// the result is [`GaloisRing::reduce_precision`]`(b)`.
    pub fn project(&self, b: u32) -> Result<GaloisRingElement> {
        let target = self.ring.reduce_precision(b)?;
        Ok(target.element(self.coeffs.clone()))
    }

    /// The Teichmüller representative of this element's residue class: the
    /// unique `t` with `t^{p^d} = t` and `t ≡ self (mod p)`.
    ///
    /// Computed as the limit of `x ↦ x^{p^d}`: if `x ≡ t (mod p^k)` then
    /// `x^{p^d} ≡ t^{p^d} = t (mod p^{k+d})`, so the iteration reaches the
    /// fixed point within `ceil((a-1)/d) <= a - 1` steps.
    pub fn teichmuller(&self) -> GaloisRingElement {
        let q = self.p().pow(self.ring.data.d as u32); // p^d
        let mut t = self.clone();
        for _ in 0..self.ring.data.a {
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
    /// `self = t_0 + t_1 p + ... + t_{a-1} p^{a-1}`.
    pub fn teichmuller_digits(&self) -> Vec<GaloisRingElement> {
        let a = self.ring.data.a;
        let p = self.p().clone();
        let pa = self.pa().clone();
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
    ///
    /// `φ` is the unique ring automorphism lifting `x ↦ x^p` on the residue
    /// field; it generates `Gal(GR(p^a, d) / Z_{p^a}) ≅ Z/dZ`, and its fixed
    /// subring is exactly the base ring `Z/p^a = GR(p^a, 1)`.
    pub fn frobenius(&self) -> GaloisRingElement {
        let p = self.p().clone();
        let p_elt = self.ring.from_int(p.clone());
        let mut result = self.ring.zero();
        let mut pk = self.ring.one(); // p^i as a ring element
        for t in self.teichmuller_digits() {
            result = result + t.pow(&p) * pk.clone();
            pk = pk * p_elt.clone();
        }
        result
    }

    /// The `k`-th power `φ^k` of the Frobenius automorphism (reduced mod the
    /// order `d` of the Galois group).
    pub fn frobenius_power(&self, k: u32) -> GaloisRingElement {
        let k = k % (self.ring.data.d as u32);
        let mut x = self.clone();
        for _ in 0..k {
            x = x.frobenius();
        }
        x
    }

    /// `self^exp` for a non-negative integer exponent.
    pub fn pow(&self, exp: &Integer) -> GaloisRingElement {
        let mut result = self.ring.one();
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

    /// The inverse of a **unit** (undefined behaviour / error otherwise),
    /// computed by lifting the residue-field inverse with Newton iteration.
    pub fn unit_inverse(&self) -> Result<GaloisRingElement> {
        if !self.is_unit() {
            return Err(MathError::NotInvertible);
        }
        let a = self.ring.data.a;
        // initial approximation: inverse in the residue field, lifted.
        let inv_bar = self.residue().inverse()?;
        let mut w = self.ring.element(inv_bar.eltseq().to_vec());
        // Newton: w <- w * (2 - self*w); p-adic precision doubles each step.
        let two = self.ring.from_int(Integer::from(2));
        let mut prec = 1u32;
        while prec < a {
            let uw = self.clone() * w.clone();
            let corr = two.clone() - uw;
            w = w * corr;
            prec *= 2;
        }
        Ok(w)
    }

    /// Euclidean division: return `(q, r)` with `self = q·other + r` and either
    /// `r = 0` or `v(r) < v(other)` (MAGMA `Quotrem`).
    pub fn quotrem(&self, other: &GaloisRingElement) -> Result<(GaloisRingElement, GaloisRingElement)> {
        self.assert_same(other);
        if other.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        if self.is_zero() {
            return Ok((self.ring.zero(), self.ring.zero()));
        }
        let va = self.valuation();
        let vb = other.valuation();
        if va < vb {
            return Ok((self.ring.zero(), self.clone()));
        }
        let ua = self.unit_part(va);
        let ub = other.unit_part(vb);
        let ubinv = ub.unit_inverse()?;
        let pk = self.ring.from_int(self.p().pow(va - vb));
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
    /// (MAGMA `GCD`).
    pub fn gcd(&self, other: &GaloisRingElement) -> GaloisRingElement {
        self.assert_same(other);
        let a = self.ring.data.a;
        let va = if self.is_zero() { a } else { self.valuation() };
        let vb = if other.is_zero() { a } else { other.valuation() };
        let m = va.min(vb);
        if m >= a {
            self.ring.zero()
        } else {
            self.ring.from_int(self.p().pow(m))
        }
    }

    /// Least common multiple: the canonical associate `p^{max(v(a),v(b))}`
    /// (MAGMA `LCM`), or `0` if either operand is `0`.
    pub fn lcm(&self, other: &GaloisRingElement) -> GaloisRingElement {
        self.assert_same(other);
        if self.is_zero() || other.is_zero() {
            return self.ring.zero();
        }
        let a = self.ring.data.a;
        let m = self.valuation().max(other.valuation());
        if m >= a {
            self.ring.zero()
        } else {
            self.ring.from_int(self.p().pow(m))
        }
    }

    /// Extended GCD: return `(g, s, t)` with `g = s·self + t·other`
    /// (MAGMA `XGCD`), `g` the canonical associate `p^{min}`.
    pub fn xgcd(
        &self,
        other: &GaloisRingElement,
    ) -> Result<(GaloisRingElement, GaloisRingElement, GaloisRingElement)> {
        self.assert_same(other);
        let a = self.ring.data.a;
        let va = if self.is_zero() { a } else { self.valuation() };
        let vb = if other.is_zero() { a } else { other.valuation() };
        if va >= a && vb >= a {
            return Ok((self.ring.zero(), self.ring.zero(), self.ring.zero()));
        }
        if va <= vb {
            // g = p^va = ua^{-1} * self
            let ua = self.unit_part(va);
            let s = ua.unit_inverse()?;
            let g = s.clone() * self.clone();
            Ok((g, s, self.ring.zero()))
        } else {
            let ub = other.unit_part(vb);
            let t = ub.unit_inverse()?;
            let g = t.clone() * other.clone();
            Ok((g, self.ring.zero(), t))
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
    fn eq(&self, other: &Self) -> bool {
        self.ring.data == other.ring.data && self.coeffs == other.coeffs
    }
}
impl Eq for GaloisRingElement {}

impl Add for GaloisRingElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.assert_same(&other);
        let pa = self.pa().clone();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| red(a.clone() + b.clone(), &pa))
            .collect();
        GaloisRingElement {
            coeffs,
            ring: self.ring,
        }
    }
}

impl Sub for GaloisRingElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.assert_same(&other);
        let pa = self.pa().clone();
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(a, b)| red(a.clone() - b.clone(), &pa))
            .collect();
        GaloisRingElement {
            coeffs,
            ring: self.ring,
        }
    }
}

impl Mul for GaloisRingElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.assert_same(&other);
        let pa = self.pa().clone();
        let mut prod = vec![Integer::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, ai) in self.coeffs.iter().enumerate() {
            if ai.is_zero() {
                continue;
            }
            for (j, bj) in other.coeffs.iter().enumerate() {
                prod[i + j] = red(prod[i + j].clone() + ai.clone() * bj.clone(), &pa);
            }
        }
        let ring = self.ring;
        GaloisRingElement {
            coeffs: ring.reduce(prod),
            ring,
        }
    }
}

impl Neg for GaloisRingElement {
    type Output = Self;
    fn neg(self) -> Self {
        let pa = self.pa().clone();
        let coeffs = self.coeffs.iter().map(|c| red(-c.clone(), &pa)).collect();
        GaloisRingElement {
            coeffs,
            ring: self.ring,
        }
    }
}

impl Ring for GaloisRingElement {
    fn zero() -> Self {
        panic!("GaloisRingElement::zero() needs a parent ring; use GaloisRing::zero()");
    }
    fn one() -> Self {
        panic!("GaloisRingElement::one() needs a parent ring; use GaloisRing::one()");
    }
    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| c.is_zero())
    }
    fn is_one(&self) -> bool {
        self.coeffs[0].is_one() && self.coeffs[1..].iter().all(|c| c.is_zero())
    }
}

impl CommutativeRing for GaloisRingElement {}

// Non-domain markers (NOT Field / IntegralDomain): GR(p^a, d) is a finite chain
// ring — a local principal ideal ring.
impl LocalRing for GaloisRingElement {
    fn is_unit(&self) -> bool {
        GaloisRingElement::is_unit(self)
    }
}

impl PrincipalIdealRing for GaloisRingElement {}

impl FiniteChainRing for GaloisRingElement {
    fn nilpotency_index(&self) -> u64 {
        self.ring.data.a as u64
    }
    fn chain_valuation(&self) -> u64 {
        self.valuation() as u64
    }
}

/// The `p`-adic valuation makes `GR(p^a, d)` a Euclidean (chain) ring.
impl DiscreteValuation<GaloisRingElement> for GaloisRing {
    fn valuation(&self, x: &GaloisRingElement) -> i64 {
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
