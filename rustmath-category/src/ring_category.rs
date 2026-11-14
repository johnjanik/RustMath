//! Ring Category - Category of rings and ring homomorphisms
//!
//! This module implements the category of rings, where:
//! - Objects are rings
//! - Morphisms are ring homomorphisms
//!
//! It provides both the categorical structure and SageMath-style element/parent methods.

use crate::axioms::{Associativity, Commutativity, Distributivity, Identity, Unity, AxiomSet};
use crate::category::Category;
use rustmath_core::{Ring, CommutativeRing, Parent, Result};
use std::fmt;

/// The category of rings
///
/// Objects: Rings (structures with addition and multiplication)
/// Morphisms: Ring homomorphisms (structure-preserving maps)
///
/// # Axioms
/// - Additive group structure (associativity, commutativity, identity, inverse)
/// - Multiplicative associativity
/// - Distributivity: a(b + c) = ab + ac
/// - Unity: multiplicative identity 1
#[derive(Clone, Debug)]
pub struct RingCategory;

impl RingCategory {
    /// Create a new ring category
    pub fn new() -> Self {
        RingCategory
    }
}

impl Default for RingCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for RingCategory {
    fn name(&self) -> &str {
        "Rings"
    }

    fn axioms(&self) -> Vec<&str> {
        vec![
            "closure",
            "associativity",
            "commutativity_additive",
            "identity_additive",
            "inverse_additive",
            "associativity_multiplicative",
            "distributivity",
            "unity",
        ]
    }

    fn description(&self) -> String {
        "Category of rings and ring homomorphisms".to_string()
    }
}

impl fmt::Display for RingCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ring")
    }
}

/// Subcategory of commutative rings
///
/// Rings where multiplication is commutative: ab = ba
#[derive(Clone, Debug)]
pub struct CommutativeRingCategory;

impl CommutativeRingCategory {
    /// Create a new commutative ring category
    pub fn new() -> Self {
        CommutativeRingCategory
    }
}

impl Default for CommutativeRingCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for CommutativeRingCategory {
    fn name(&self) -> &str {
        "CommutativeRings"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(RingCategory)]
    }

    fn axioms(&self) -> Vec<&str> {
        let mut axioms = RingCategory.axioms();
        axioms.push("commutativity_multiplicative");
        axioms
    }

    fn description(&self) -> String {
        "Category of commutative rings".to_string()
    }
}

impl fmt::Display for CommutativeRingCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CommutativeRing")
    }
}

/// Subcategory of integral domains
///
/// Commutative rings with no zero divisors
#[derive(Clone, Debug)]
pub struct IntegralDomainCategory;

impl IntegralDomainCategory {
    /// Create a new integral domain category
    pub fn new() -> Self {
        IntegralDomainCategory
    }
}

impl Default for IntegralDomainCategory {
    fn default() -> Self {
        Self::new()
    }
}

impl Category for IntegralDomainCategory {
    fn name(&self) -> &str {
        "IntegralDomains"
    }

    fn super_categories(&self) -> Vec<Box<dyn Category>> {
        vec![Box::new(CommutativeRingCategory)]
    }

    fn axioms(&self) -> Vec<&str> {
        let mut axioms = CommutativeRingCategory.axioms();
        axioms.push("no_zero_divisors");
        axioms
    }

    fn description(&self) -> String {
        "Category of integral domains (commutative rings with no zero divisors)".to_string()
    }
}

/// The multiplicative order of a ring element (re-exported from `rustmath_core`).
pub use rustmath_core::MultiplicativeOrder;

/// Methods for ring elements (Sage `Rings.ElementMethods`; MAGMA Handbook Ch. 17 §17.5.5
/// `IsUnit`, `IsNilpotent`, `IsIdempotent`; Ch. 19 `Order`).
///
/// These are thin views of the three-valued predicates on [`Ring`]
/// (`unit_status`, `nilpotent_status`, `multiplicative_order_status`). Every concrete ring
/// decides them in its own `impl Ring` (Q, R, C: every nonzero element is a unit; Z/mZ,
/// GF(q), Galois rings: exact). `None` or [`MultiplicativeOrder::Undecided`] means that
/// element type cannot decide; no answer here comes from a bounded search.
///
/// Concrete types with an inherent method of the same name (`IntegerMod::is_unit`,
/// `FiniteFieldElement::multiplicative_order`, ...) resolve to the inherent method at a
/// concrete call site; generic code (`R: Ring`) reaches these.
pub trait RingElementMethods: Ring {
    /// Whether this element is a unit: `Some(true)`/`Some(false)` decided, `None` undecided.
    fn is_unit(&self) -> Option<bool> {
        self.unit_status()
    }

    /// Whether `x^n = 0` for some `n >= 1`: `Some(_)` decided, `None` undecided.
    fn is_nilpotent(&self) -> Option<bool> {
        self.nilpotent_status()
    }

    /// Check if this element is idempotent
    ///
    /// An element x is idempotent if x^2 = x
    fn is_idempotent(&self) -> bool
    where
        Self: Sized,
    {
        let x_squared = self.clone() * self.clone();
        x_squared == *self
    }

    /// The multiplicative order (see [`MultiplicativeOrder`]).
    fn multiplicative_order(&self) -> MultiplicativeOrder {
        self.multiplicative_order_status()
    }
}

/// Blanket implementation for all rings
impl<R: Ring> RingElementMethods for R {}

/// Primality of a word, a proof (`rustmath_integers::prime::is_prime_u64`: strong tests to
/// the twelve prime bases 2..37, deterministic below psi_12 > 2^64, J. Sorenson and
/// J. Webster, "Strong pseudoprimes to twelve prime bases", Math. Comp. 86 (2017)).
pub(crate) fn prime_u64(n: u64) -> bool {
    rustmath_integers::prime::is_prime_u64(n)
}

/// Whether the word `n` is `q^k` with `q` prime and `k >= 1`: the minimal root of `n`
/// (`Integer::is_perfect_power` extracts q-th roots for each prime q in increasing order,
/// so its base is not a perfect power) is prime.
pub(crate) fn is_prime_power_u64(n: u64) -> bool {
    use rustmath_integers::Integer;
    if n < 2 {
        return false;
    }
    let (b, _) = Integer::from(n)
        .is_perfect_power()
        .unwrap_or_else(|| (Integer::from(n), 1));
    u64::try_from(&b).is_ok_and(prime_u64)
}

/// MAGMA `IsPrime(n : Proof := true)`: `Some(b)` proven, `None` when the ECPP proof of a
/// probable prime is unresolved (`rustmath_integers::prime::try_is_prime`).
pub(crate) fn proven_primality(n: &rustmath_integers::Integer) -> Option<bool> {
    rustmath_integers::prime::try_is_prime(n).ok()
}

/// Whether a ring of characteristic `c` is provably not an integral domain: `c = 1` is the
/// zero ring; for composite `c = ab` (1 < a, b < c), `(a·1)(b·1) = c·1 = 0` with `a·1` and
/// `b·1` nonzero. `false` for 0 and for primes.
fn characteristic_rules_out_domain(c: u64) -> bool {
    c == 1 || (c >= 4 && !prime_u64(c))
}

/// `Some(c)` when the parent's own data prove `R = Z·1 ≅ Z/cZ`: a known characteristic `c`
/// equal to the cardinality (the additive subgroup `Z·1` already has `c` elements), or a
/// prime cardinality `p` with unknown characteristic (`1 != 0` has additive order dividing
/// `p`, Lagrange; the default [`RingParentMethods::characteristic`] already answers `p`).
fn residue_ring_modulus<P: RingParentMethods>(parent: &P) -> Option<u64> {
    let n = u64::try_from(parent.cardinality()?).ok()?;
    if n == 0 {
        return None;
    }
    match parent.characteristic() {
        Some(c) => (c == n).then_some(c),
        None => prime_u64(n).then_some(n),
    }
}

/// Methods for ring parent structures: SageMath `Rings.ParentMethods`; MAGMA Handbook Ch. 17
/// §17.4.2 `Characteristic` and the §17.4.3 predicates.
///
/// Three-valued like MAGMA's predicates ("true if known; false if known not; error if
/// unknown"): `Some(_)` is a proof, `None` makes no claim (MAGMA raises an error there).
/// `Parent::Element` need not be a `Ring`, so the defaults cannot read
/// `Ring::static_characteristic`; they decide only what another method of the same parent
/// proves:
/// * a parent with one element is the zero ring (characteristic 1, commutative, not a
///   domain);
/// * a prime cardinality p gives characteristic p (Lagrange);
/// * a known characteristic equal to the cardinality, or a prime cardinality, makes the
///   ring `Z·1 ≅ Z/cZ`: commutative, and a domain iff c is prime;
/// * a composite characteristic rules out a domain;
/// * a cardinality n >= 2 that is not a prime power rules out a domain (a finite domain is
///   a field, of prime-power order);
/// * a non-domain is not a field; a finite domain is a field.
///
/// A parent that knows more overrides.
///
/// Default dependency order (an override may call only methods earlier in it):
/// `Parent::cardinality`, `Parent::is_finite` → `characteristic` → `is_commutative`,
/// `is_integral_domain` → `is_field` (→ the [`CommutativeRingParentMethods`] predicates).
/// Because `is_integral_domain` comes before `is_field`, a parent that overrides `is_field`
/// with `Some(true)` should also override `is_integral_domain` (a field is a domain).
pub trait RingParentMethods: Parent {
    /// MAGMA `Characteristic(R)` (Ch. 17 §17.4.2): the least `m >= 1` with `m·1 = 0`, or 0.
    ///
    /// `Some(c)` is exact (`Some(1)`: the zero ring); `None` means unknown or `>= 2^64`.
    /// Default: `Some(1)` when `Parent::cardinality` is 1 (then `1 = 0`); `Some(p)` when it is
    /// a prime p (the ring has more than one element, so `1 != 0`, and the additive order of
    /// 1 divides p); otherwise `None`.
    fn characteristic(&self) -> Option<u64> {
        let n = u64::try_from(self.cardinality()?).ok()?;
        if n == 1 || prime_u64(n) {
            Some(n)
        } else {
            None
        }
    }

    /// MAGMA `IsCommutative(R)` (Ch. 17 §17.4.3).
    ///
    /// Default: `Some(true)` for the zero ring (characteristic 1 or one element) and for a
    /// parent proven to be `Z/cZ` (a known characteristic equal to the cardinality, or a
    /// prime cardinality); otherwise `None`.
    fn is_commutative(&self) -> Option<bool> {
        let zero_ring = self.characteristic() == Some(1) || self.cardinality() == Some(1);
        if zero_ring || residue_ring_modulus(self).is_some() {
            Some(true)
        } else {
            None
        }
    }

    /// MAGMA `IsDomain(R)` (Ch. 17 §17.4.3): commutative, `1 != 0`, no zero divisors.
    ///
    /// Default:
    /// * the zero ring (characteristic 1, or one element) is not one;
    /// * a composite characteristic `c = ab` is not one (`(a·1)(b·1) = 0`);
    /// * a parent proven to be `Z/cZ` (see [`Self::is_commutative`]) is one iff `c` is prime;
    /// * a cardinality `n >= 2` that is not a prime power is not one;
    /// * otherwise `None` (characteristic 0 or a prime does not decide it).
    fn is_integral_domain(&self) -> Option<bool> {
        if self.characteristic() == Some(1) || self.cardinality() == Some(1) {
            return Some(false);
        }
        if let Some(c) = self.characteristic() {
            if characteristic_rules_out_domain(c) {
                return Some(false);
            }
        }
        if let Some(c) = residue_ring_modulus(self) {
            return Some(prime_u64(c));
        }
        match self.cardinality().and_then(|n| u64::try_from(n).ok()) {
            Some(n) if n >= 2 && !is_prime_power_u64(n) => Some(false),
            _ => None,
        }
    }

    /// MAGMA `IsField(R)` (Ch. 17 §17.4.3).
    ///
    /// Default: a proven non-domain is not a field; a finite domain is one (`x ↦ ax` is
    /// injective for `a != 0`, hence onto); otherwise `None`.
    fn is_field(&self) -> Option<bool> {
        match self.is_integral_domain() {
            Some(false) => Some(false),
            Some(true) if self.is_finite() == Some(true) => Some(true),
            _ => None,
        }
    }

    // `is_finite` is the supertrait's three-valued `Parent::is_finite` (MAGMA Ch. 17
    // §17.4.3); a duplicate here would claim "infinite" whenever `cardinality` is `None`.

    /// Get the axiom set for this ring
    fn axiom_set(&self) -> AxiomSet {
        AxiomSet::ring()
    }
}

/// Subcategory methods for commutative rings (MAGMA Ch. 17 §17.4.3 `IsEuclideanDomain`,
/// `IsPID`, `IsUFD`), three-valued as in [`RingParentMethods`].
///
/// The defaults use field ⇒ Euclidean domain ⇒ PID ⇒ UFD, and that a non-domain is none of
/// them; otherwise `None`.
pub trait CommutativeRingParentMethods: RingParentMethods {
    /// MAGMA `IsPID(R)`: a Euclidean domain is one; a non-domain is not; otherwise `None`.
    fn is_pid(&self) -> Option<bool> {
        if self.is_euclidean() == Some(true) {
            Some(true)
        } else if self.is_integral_domain() == Some(false) {
            Some(false)
        } else {
            None
        }
    }

    /// MAGMA `IsUFD(R)`: a PID is one; a non-domain is not; otherwise `None`.
    fn is_ufd(&self) -> Option<bool> {
        if self.is_pid() == Some(true) {
            Some(true)
        } else if self.is_integral_domain() == Some(false) {
            Some(false)
        } else {
            None
        }
    }

    /// MAGMA `IsEuclideanDomain(R)`: a field is one; a non-domain is not; otherwise `None`.
    fn is_euclidean(&self) -> Option<bool> {
        if self.is_field() == Some(true) {
            Some(true)
        } else if self.is_integral_domain() == Some(false) {
            Some(false)
        } else {
            None
        }
    }

    /// Get the axiom set for commutative rings
    fn axiom_set(&self) -> AxiomSet {
        let mut set = AxiomSet::ring();
        set.add_axiom(Commutativity);
        set
    }
}

/// Ring with basis - extends ParentWithBasis for rings
///
/// Rings that have a distinguished basis (like polynomial rings)
pub trait RingWithBasis: Ring + Parent {
    /// Type of basis indices
    type BasisIndex: Clone;

    /// Get a basis element by index
    fn basis_element(&self, index: Self::BasisIndex) -> Self::Element;

    /// Get all basis indices (for finite dimensional rings)
    fn basis_indices(&self) -> Option<Vec<Self::BasisIndex>> {
        None // Default: infinite or not enumerable
    }

    /// Express an element in terms of the basis
    fn coordinates(&self, element: &Self::Element) -> Vec<(Self::BasisIndex, Self)>;
}

/// Standard ring constructions
pub mod constructions {
    use super::*;

    /// Quotient ring R/I for an ideal I
    ///
    /// Elements are cosets r + I
    pub struct QuotientRing<R: Ring> {
        base_ring: R,
        // In a full implementation, would store the ideal
    }

    impl<R: Ring> QuotientRing<R> {
        /// Create a quotient ring R/I
        pub fn new(base_ring: R) -> Self {
            QuotientRing { base_ring }
        }

        /// Get the base ring
        pub fn base_ring(&self) -> &R {
            &self.base_ring
        }
    }

    /// Product ring R × S
    ///
    /// Elements are pairs (r, s)
    pub struct ProductRing<R: Ring, S: Ring> {
        first: R,
        second: S,
    }

    impl<R: Ring, S: Ring> ProductRing<R, S> {
        /// Create a product ring R × S
        pub fn new(first: R, second: S) -> Self {
            ProductRing { first, second }
        }

        /// Get the first component ring
        pub fn first_ring(&self) -> &R {
            &self.first
        }

        /// Get the second component ring
        pub fn second_ring(&self) -> &S {
            &self.second
        }
    }

    /// Matrix ring M_n(R)
    ///
    /// Ring of n×n matrices over R
    pub struct MatrixRing<R: Ring> {
        base_ring: R,
        dimension: usize,
    }

    impl<R: Ring> MatrixRing<R> {
        /// Create a matrix ring M_n(R)
        pub fn new(base_ring: R, dimension: usize) -> Self {
            MatrixRing {
                base_ring,
                dimension,
            }
        }

        /// Get the base ring
        pub fn base_ring(&self) -> &R {
            &self.base_ring
        }

        /// Get the matrix dimension
        pub fn dimension(&self) -> usize {
            self.dimension
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_category_creation() {
        let cat = RingCategory::new();
        assert_eq!(cat.name(), "Rings");
    }

    #[test]
    fn test_ring_category_axioms() {
        let cat = RingCategory;
        let axioms = cat.axioms();
        assert!(axioms.contains(&"associativity"));
        assert!(axioms.contains(&"distributivity"));
        assert!(axioms.contains(&"unity"));
    }

    #[test]
    fn test_ring_category_display() {
        let cat = RingCategory;
        assert_eq!(format!("{}", cat), "Ring");
    }

    #[test]
    fn test_commutative_ring_category() {
        let cat = CommutativeRingCategory::new();
        assert_eq!(cat.name(), "CommutativeRings");

        let axioms = cat.axioms();
        assert!(axioms.contains(&"commutativity_multiplicative"));
    }

    #[test]
    fn test_commutative_ring_is_subcategory() {
        let comm_cat = CommutativeRingCategory;
        let ring_cat = RingCategory;

        assert!(comm_cat.is_subcategory_of(&ring_cat));
    }

    #[test]
    fn test_integral_domain_category() {
        let cat = IntegralDomainCategory::new();
        assert_eq!(cat.name(), "IntegralDomains");

        let axioms = cat.axioms();
        assert!(axioms.contains(&"no_zero_divisors"));
    }

    #[test]
    fn test_integral_domain_is_subcategory() {
        let domain_cat = IntegralDomainCategory;
        let comm_cat = CommutativeRingCategory;
        let ring_cat = RingCategory;

        assert!(domain_cat.is_subcategory_of(&comm_cat));
        assert!(domain_cat.is_subcategory_of(&ring_cat));
    }

    // Test RingElementMethods
    #[test]
    fn test_is_unit_for_integers() {
        assert_eq!(1i32.is_unit(), Some(true));
        assert_eq!((-1i32).is_unit(), Some(true));
        assert_eq!(2i32.is_unit(), Some(false));
        assert_eq!(0i32.is_unit(), Some(false));
    }

    #[test]
    fn test_is_nilpotent() {
        assert_eq!(0i32.is_nilpotent(), Some(true));
        assert_eq!(1i32.is_nilpotent(), Some(false));
        assert_eq!(2i32.is_nilpotent(), Some(false));
    }

    #[test]
    fn test_is_idempotent() {
        assert!(0i32.is_idempotent()); // 0² = 0
        assert!(1i32.is_idempotent()); // 1² = 1
        assert!(!2i32.is_idempotent()); // 2² = 4 ≠ 2
    }

    #[test]
    fn test_multiplicative_order() {
        assert_eq!(1i32.multiplicative_order(), MultiplicativeOrder::Finite(1));
        assert_eq!((-1i32).multiplicative_order(), MultiplicativeOrder::Finite(2));
        assert_eq!(2i32.multiplicative_order(), MultiplicativeOrder::NotAUnit); // Not a unit
    }

    // The defect's generic path (MAGMA Ch. 20 `IsUnit`: every nonzero rational is a unit;
    // GAP: IsUnit(Rationals, 2/3) -> true, IsUnit(Integers, 2) -> false,
    // Order(2/3) -> infinity).
    fn generic_is_unit<R: Ring>(x: &R) -> Option<bool> {
        RingElementMethods::is_unit(x)
    }

    #[test]
    fn test_generic_path_rationals_and_integers() {
        use rustmath_integers::Integer;
        use rustmath_rationals::Rational;
        let two_thirds = Rational::new(2, 3).unwrap();
        assert_eq!(generic_is_unit(&two_thirds), Some(true));
        assert_eq!(generic_is_unit(&Integer::from(2)), Some(false));
        assert_eq!(two_thirds.multiplicative_order(), MultiplicativeOrder::Infinite);
        assert_eq!(two_thirds.is_nilpotent(), Some(false));
        assert_eq!(Integer::from(-1).multiplicative_order(), MultiplicativeOrder::Finite(2));
    }

    // Test constructions
    #[test]
    fn test_quotient_ring_construction() {
        let qr = constructions::QuotientRing::new(0i32);
        assert_eq!(qr.base_ring(), &0);
    }

    #[test]
    fn test_product_ring_construction() {
        // Product of two distinct ring types R = i32, S = i64 (f64 is not a Ring).
        let pr = constructions::ProductRing::new(0i32, 0i64);
        assert_eq!(pr.first_ring(), &0i32);
        assert_eq!(pr.second_ring(), &0i64);
    }

    #[test]
    fn test_matrix_ring_construction() {
        let mr = constructions::MatrixRing::new(0i32, 3);
        assert_eq!(mr.dimension(), 3);
        assert_eq!(mr.base_ring(), &0);
    }

    #[test]
    fn test_ring_category_description() {
        let cat = RingCategory;
        assert!(cat.description().contains("ring"));
    }

    #[test]
    fn test_ring_category_default() {
        let cat1 = RingCategory::new();
        let cat2 = RingCategory::default();
        assert_eq!(cat1.name(), cat2.name());
    }

    // ---- Honest parent defaults (MAGMA Ch. 17 §17.4.2, §17.4.3) ----
    // Test parents with `Element = u8` and a `contains` that matches their cardinality.

    /// A parent that overrides nothing: what every new implementor inherits.
    #[derive(Debug, Clone)]
    struct CatBare;
    impl Parent for CatBare {
        type Element = u8;
        fn contains(&self, _: &u8) -> bool {
            true
        }
    }
    impl RingParentMethods for CatBare {}
    impl CommutativeRingParentMethods for CatBare {}

    /// The zero ring {0}, known from its cardinality only.
    #[derive(Debug, Clone)]
    struct CatZero;
    impl Parent for CatZero {
        type Element = u8;
        fn contains(&self, x: &u8) -> bool {
            *x == 0
        }
        fn cardinality(&self) -> Option<usize> {
            Some(1)
        }
    }
    impl RingParentMethods for CatZero {}
    impl CommutativeRingParentMethods for CatZero {}

    /// Q, with its predicates overridden.
    #[derive(Debug, Clone)]
    struct CatQ;
    impl Parent for CatQ {
        type Element = u8;
        fn contains(&self, _: &u8) -> bool {
            true
        }
    }
    impl RingParentMethods for CatQ {
        fn characteristic(&self) -> Option<u64> {
            Some(0)
        }
        fn is_commutative(&self) -> Option<bool> {
            Some(true)
        }
        fn is_integral_domain(&self) -> Option<bool> {
            Some(true)
        }
        fn is_field(&self) -> Option<bool> {
            Some(true)
        }
    }
    impl CommutativeRingParentMethods for CatQ {}

    /// Z/7 = {0, ..., 6}: cardinality and characteristic 7, known to be a domain.
    #[derive(Debug, Clone)]
    struct CatF7;
    impl Parent for CatF7 {
        type Element = u8;
        fn contains(&self, x: &u8) -> bool {
            *x < 7
        }
        fn cardinality(&self) -> Option<usize> {
            Some(7)
        }
    }
    impl RingParentMethods for CatF7 {
        fn characteristic(&self) -> Option<u64> {
            Some(7)
        }
        fn is_integral_domain(&self) -> Option<bool> {
            Some(true)
        }
    }
    impl CommutativeRingParentMethods for CatF7 {}

    // At a46a6a8 a parent with no overrides answered characteristic 0 and `false` for every
    // predicate; it now makes no claim.
    #[test]
    fn test_ring_parent_defaults_make_no_claim() {
        let r = CatBare;
        assert_eq!(
            (
                r.characteristic(),
                r.is_commutative(),
                r.is_integral_domain(),
                r.is_field(),
                r.is_pid(),
                r.is_ufd(),
                r.is_euclidean()
            ),
            (None, None, None, None, None, None, None)
        );
    }

    // GAP 4.12.1: [Size, Characteristic, IsField, IsCommutative, Zero = One](ZmodnZ(1))
    // -> [1, 1, false, true, true], and IsIntegralRing(ZmodnZ(1)) -> false: the zero ring
    // (1 = 0) is not a domain, which gives `is_integral_domain` = Some(false).
    // The Euclidean / PID / UFD answers come from the definitions, not from GAP. MAGMA's
    // `IsEuclideanDomain`, `IsPID` and `IsUFD` (Ch. 17 §17.4.3) are domain predicates, and a
    // non-domain is none of them. GAP 4.12.1 answers IsEuclideanRing(ZmodnZ(1)) -> true and
    // IsUniqueFactorizationRing(ZmodnZ(1)) -> true, but those are categories GAP attaches to
    // every ZmodnZ(n) when it builds the ring (lib/zmodnz.gd, InstallTrueMethod for
    // IsEuclideanRing); by the same route it answers IsIntegralRing(ZmodnZ(6)) -> true,
    // although 2 * 3 = 0. GAP's own manual (§56.6-1, §56.4-2) defines both as integral rings.
    // These answers are therefore GAP artifacts, not reference values, and must not be used
    // to "correct" this test. The Ch. 17 digest does not discuss the zero ring itself; it
    // defines `IsDomain` as "integral domain", and the usual convention that a domain has
    // 1 != 0 is the one `RingParentMethods::is_integral_domain` documents.
    #[test]
    fn test_zero_ring_parent_category() {
        let r = CatZero;
        assert_eq!(r.characteristic(), Some(1));
        assert_eq!(r.is_commutative(), Some(true));
        assert_eq!(
            (
                r.is_integral_domain(),
                r.is_field(),
                r.is_pid(),
                r.is_ufd(),
                r.is_euclidean()
            ),
            (
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            )
        );
    }

    // GAP: IsEuclideanRing(Rationals) -> true, IsUniqueFactorizationRing(Rationals) -> true
    // (GAP has no IsPrincipalIdealRing; a Euclidean domain is a PID).
    #[test]
    fn test_field_parent_category() {
        let q = CatQ;
        assert_eq!(
            (q.is_euclidean(), q.is_pid(), q.is_ufd()),
            (Some(true), Some(true), Some(true))
        );
    }

    // GAP: [Characteristic, Size, IsField, IsCommutative](ZmodnZ(7)) -> [7, 7, true, true].
    // A finite domain is a field; characteristic = cardinality gives Z·1 = Z/7, commutative.
    #[test]
    fn test_finite_domain_is_field_category() {
        let f7 = CatF7;
        assert_eq!(f7.is_commutative(), Some(true));
        assert_eq!(
            (f7.is_field(), f7.is_euclidean(), f7.is_pid(), f7.is_ufd()),
            (Some(true), Some(true), Some(true), Some(true))
        );
    }

    // ---- The composite-characteristic and cardinality rules (D.2) ----

    /// A parent known only by its cardinality.
    #[derive(Debug, Clone)]
    struct ByCard(usize);
    impl Parent for ByCard {
        type Element = u8;
        fn contains(&self, _: &u8) -> bool {
            true
        }
        fn cardinality(&self) -> Option<usize> {
            Some(self.0)
        }
    }
    impl RingParentMethods for ByCard {}
    impl CommutativeRingParentMethods for ByCard {}

    /// A parent known only by its characteristic.
    #[derive(Debug, Clone)]
    struct ByChar(u64);
    impl Parent for ByChar {
        type Element = u8;
        fn contains(&self, _: &u8) -> bool {
            true
        }
    }
    impl RingParentMethods for ByChar {
        fn characteristic(&self) -> Option<u64> {
            Some(self.0)
        }
    }
    impl CommutativeRingParentMethods for ByChar {}

    /// A parent known by its cardinality and characteristic.
    #[derive(Debug, Clone)]
    struct ByBoth(usize, u64);
    impl Parent for ByBoth {
        type Element = u8;
        fn contains(&self, _: &u8) -> bool {
            true
        }
        fn cardinality(&self) -> Option<usize> {
            Some(self.0)
        }
    }
    impl RingParentMethods for ByBoth {
        fn characteristic(&self) -> Option<u64> {
            Some(self.1)
        }
    }
    impl CommutativeRingParentMethods for ByBoth {}

    fn domain_tuple<P: CommutativeRingParentMethods>(r: &P) -> [Option<bool>; 5] {
        [r.is_integral_domain(), r.is_field(), r.is_pid(), r.is_ufd(), r.is_euclidean()]
    }

    // PARI: Mod(2,4)^2 -> Mod(0, 4); isprime(561) -> 0 (561 = 3*11*17, a Carmichael
    // number); isprime(2^64-59) -> 1. At c79a582 every answer here was None.
    #[test]
    fn test_composite_characteristic_is_not_a_domain() {
        assert_eq!(domain_tuple(&ByChar(4)), [Some(false); 5]);
        assert_eq!(domain_tuple(&ByChar(561)), [Some(false); 5]);
        assert_eq!(ByChar(18_446_744_073_709_551_557).is_integral_domain(), None);
        assert_eq!(ByChar(0).is_integral_domain(), None);
    }

    // GAP: IsField(ZmodnZ(7)) -> true, Characteristic(ZmodnZ(7)) -> 7; IsField(GF(4)) ->
    // true and IsField(ZmodnZ(4)) -> false (cardinality 4 decides nothing). At c79a582 the
    // domain was None for Z/6 and every answer was None for cardinality 7.
    #[test]
    fn test_residue_ring_by_cardinality() {
        let z6 = ByBoth(6, 6);
        assert_eq!(z6.is_commutative(), Some(true));
        assert_eq!(z6.is_integral_domain(), Some(false));
        let f7 = ByCard(7);
        assert_eq!(f7.characteristic(), Some(7));
        assert_eq!(f7.is_commutative(), Some(true));
        assert_eq!(domain_tuple(&f7), [Some(true); 5]);
        assert_eq!(ByCard(4).is_integral_domain(), None);
        assert_eq!(ByCard(4).characteristic(), None);
    }

    // GAP: IsPrimePowerInt(6) -> false, IsPrimePowerInt(12) -> false, IsField(ZmodnZ(6)) ->
    // false; IsPrimePowerInt(49) -> true (GF(49) is a field, Z/49 is not).
    #[test]
    fn test_non_prime_power_cardinality_is_not_a_domain() {
        assert_eq!(domain_tuple(&ByCard(6)), [Some(false); 5]);
        assert_eq!(domain_tuple(&ByCard(12)), [Some(false); 5]);
        assert_eq!(ByCard(49).is_integral_domain(), None);
        assert_eq!(ByCard(6).characteristic(), None);
    }
}
