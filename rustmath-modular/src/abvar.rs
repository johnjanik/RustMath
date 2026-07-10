//! Modular abelian varieties
//!
//! This module implements modular abelian varieties, which are abelian varieties
//! arising from modular forms.

use crate::arithgroup::{Gamma0, Gamma1};
use crate::hecke::{HeckeOperator, Newform};
use crate::modsym::ModularSymbolSpace;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Euler's totient phi(n) for small u64 arguments.
fn euler_phi(mut n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut p = 2u64;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

/// Genus of the modular curve X_0(N) (= dimension of J_0(N)).
///
/// Uses the exact formula
///   g = 1 + mu/12 - eps2/4 - eps3/3 - eps_inf/2
/// where mu = [SL2(Z):Gamma0(N)], eps_inf is the number of cusps, and
/// eps2, eps3 are the numbers of elliptic points of order 2 and 3, computed
/// as the residue counts #{x mod N : x^2 + 1 = 0} and
/// #{x mod N : x^2 + x + 1 = 0} respectively. Verified against SageMath.
fn genus_x0(level: u64) -> usize {
    if level == 1 {
        return 0;
    }
    let g0 = Gamma0::new(level);
    let mu = g0.compute_index() as i64;
    let cusps = g0.compute_cusp_count() as i64;
    let n = level as u128;
    // Elliptic points of order 2: solutions of x^2 + 1 == 0 (mod N).
    let eps2 = (0..level)
        .filter(|&x| ((x as u128) * (x as u128) + 1) % n == 0)
        .count() as i64;
    // Elliptic points of order 3: solutions of x^2 + x + 1 == 0 (mod N).
    let eps3 = (0..level)
        .filter(|&x| ((x as u128) * (x as u128) + (x as u128) + 1) % n == 0)
        .count() as i64;
    // g = 1 + (mu - 3*eps2 - 4*eps3 - 6*cusps) / 12; the numerator is always
    // divisible by 12 and the result is >= 0.
    let num = mu - 3 * eps2 - 4 * eps3 - 6 * cusps;
    debug_assert_eq!(num.rem_euclid(12), 0, "X0 genus numerator not divisible by 12");
    let g = 1 + num / 12;
    debug_assert!(g >= 0, "negative genus for X0({level})");
    g.max(0) as usize
}

/// Genus of the modular curve X_1(N) (= dimension of J_1(N)).
///
/// For N <= 4 the genus is 0. For N >= 5, Gamma1(N) has no elliptic points,
/// so
///   g = 1 + mu/24 - eps_inf/2
/// where mu = [SL2(Z):Gamma1(N)] and eps_inf = (1/2) sum_{d|N} phi(d) phi(N/d)
/// is the number of cusps. Verified against SageMath.
fn genus_x1(level: u64) -> usize {
    if level <= 4 {
        return 0;
    }
    let mu = Gamma1::new(level).compute_index() as i64;
    // 2 * (number of cusps) = sum_{d|N} phi(d) phi(N/d).
    let mut twice_cusps = 0i64;
    for d in 1..=level {
        if level % d == 0 {
            twice_cusps += (euler_phi(d) * euler_phi(level / d)) as i64;
        }
    }
    // g = 1 + mu/24 - cusps/2 = 1 + (mu - 6 * twice_cusps) / 24.
    let num = mu - 6 * twice_cusps;
    debug_assert_eq!(num.rem_euclid(24), 0, "X1 genus numerator not divisible by 24");
    let g = 1 + num / 24;
    debug_assert!(g >= 0, "negative genus for X1({level})");
    g.max(0) as usize
}

/// A modular abelian variety
#[derive(Debug, Clone)]
pub struct ModularAbelianVariety {
    /// Level
    level: u64,
    /// Dimension over Q
    dimension: usize,
    /// Conductor (same as level for most cases)
    conductor: u64,
}

impl ModularAbelianVariety {
    /// Create a new modular abelian variety
    pub fn new(level: u64, dimension: usize) -> Self {
        ModularAbelianVariety {
            level,
            dimension,
            conductor: level,
        }
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the conductor
    pub fn conductor(&self) -> u64 {
        self.conductor
    }

    /// Check if this abelian variety is simple.
    ///
    /// Implemented cases (rigorous): dimension 0 (the zero variety, not
    /// simple by convention, like a unit not being prime) and dimension 1
    /// (an abelian variety of dimension 1 is an elliptic curve and has no
    /// nonzero proper abelian subvarieties, since such a subvariety would
    /// need dimension strictly between 0 and 1).  For dimension >= 2 the
    /// answer genuinely depends on the isogeny decomposition (Hecke charpoly
    /// factorization on the newform factors), which is not yet implemented;
    /// note the old facade's `dimension == 1` stand-in was wrong precisely
    /// there (higher-dimensional simple factors exist).
    pub fn is_simple(&self) -> bool {
        match self.dimension {
            0 => false,
            1 => true,
            _ => unimplemented!(
                "ModularAbelianVariety::is_simple for dimension >= 2 requires \
                 the isogeny decomposition, which is not yet implemented"
            ),
        }
    }

    /// Decompose into simple factors (up to isogeny)
    pub fn decomposition(&self) -> Vec<ModularAbelianVariety> {
        // Isogeny decomposition into simple factors is not yet implemented.
        // Previously this returned `vec![self.clone()]` unconditionally.
        unimplemented!(
            "ModularAbelianVariety::decomposition not yet implemented (facade): previously returned self unconditionally"
        )
    }
}

/// The Jacobian J_0(N) of the modular curve X_0(N)
#[derive(Debug, Clone)]
pub struct J0 {
    /// Level N
    level: u64,
}

impl J0 {
    /// Create J_0(N).
    ///
    /// Construction is metadata-only and never panics; the dimension (the
    /// genus of X_0(N)) is computed lazily by [`J0::dimension`].
    pub fn new(level: u64) -> Self {
        J0 { level }
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get the dimension (the genus of X_0(N)), computed on demand.
    pub fn dimension(&self) -> usize {
        genus_x0(self.level)
    }

    /// Convert to ModularAbelianVariety
    pub fn to_abvar(&self) -> ModularAbelianVariety {
        ModularAbelianVariety::new(self.level, self.dimension())
    }

    /// Get the underlying modular curve
    pub fn modular_curve(&self) -> ModularCurve {
        ModularCurve::X0(self.level)
    }
}

/// The Jacobian J_1(N) of the modular curve X_1(N)
#[derive(Debug, Clone)]
pub struct J1 {
    /// Level N
    level: u64,
}

impl J1 {
    /// Create J_1(N).
    ///
    /// Construction is metadata-only and never panics; the dimension (the
    /// genus of X_1(N)) is computed lazily by [`J1::dimension`].
    pub fn new(level: u64) -> Self {
        J1 { level }
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get the dimension (the genus of X_1(N)), computed on demand.
    pub fn dimension(&self) -> usize {
        genus_x1(self.level)
    }

    /// Convert to ModularAbelianVariety
    pub fn to_abvar(&self) -> ModularAbelianVariety {
        ModularAbelianVariety::new(self.level, self.dimension())
    }
}

/// Modular curves
#[derive(Debug, Clone)]
pub enum ModularCurve {
    /// X_0(N)
    X0(u64),
    /// X_1(N)
    X1(u64),
    /// X(N)
    X(u64),
}

impl ModularCurve {
    /// Get the level
    pub fn level(&self) -> u64 {
        match self {
            ModularCurve::X0(n) | ModularCurve::X1(n) | ModularCurve::X(n) => *n,
        }
    }

    /// Get the genus
    pub fn genus(&self) -> usize {
        match self {
            ModularCurve::X0(n) => J0::new(*n).dimension(),
            ModularCurve::X1(n) => J1::new(*n).dimension(),
            ModularCurve::X(n) => {
                // Genus of X(N) for N > 1 requires a more complex formula
                // (involving the index of Gamma(N) and its elliptic points /
                // cusps); it is not yet implemented. X(1) is genuinely genus 0.
                if *n == 1 {
                    0
                } else {
                    unimplemented!(
                        "ModularCurve::genus for X(N), N > 1 not yet implemented (facade): previously returned the constant 1"
                    )
                }
            }
        }
    }

    /// Get the Jacobian
    pub fn jacobian(&self) -> ModularAbelianVariety {
        match self {
            ModularCurve::X0(n) => J0::new(*n).to_abvar(),
            ModularCurve::X1(n) => J1::new(*n).to_abvar(),
            ModularCurve::X(n) => ModularAbelianVariety::new(*n, self.genus()),
        }
    }
}

/// Abelian variety associated to a newform
#[derive(Debug, Clone)]
pub struct AbelianVarietyNewform {
    /// The newform
    newform: Newform,
    /// Dimension
    dimension: usize,
}

impl AbelianVarietyNewform {
    /// Create abelian variety from a newform.
    ///
    /// The dimension of A_f equals the degree over Q of the field generated
    /// by the Fourier coefficients of f.  Implemented case (rigorous):
    /// weight 2 at a level N with genus(X_0(N)) = 1 while every proper
    /// divisor M | N has genus(X_0(M)) = 0.  Then dim S_2(Gamma0(N)) = 1
    /// and the old subspace is zero, so f is the unique newform, its
    /// eigenvalues are rational (the 1-dimensional space is Hecke-stable),
    /// and A_f is an elliptic curve: dimension 1.  This covers e.g.
    /// N = 11, 14, 15, 17, 19, 20, 21, 24, 27, 32, 36, 49.
    ///
    /// The general case needs the newform decomposition of the cuspidal
    /// modular symbols space (Hecke charpoly factorization over Q), which
    /// is not yet implemented.
    pub fn new(newform: Newform) -> Self {
        if newform.weight() == 2 {
            let n = newform.level();
            if n >= 1 && genus_x0(n) == 1 {
                let old_part_zero =
                    (1..n).filter(|m| n % m == 0).all(|m| genus_x0(m) == 0);
                if old_part_zero {
                    return AbelianVarietyNewform {
                        newform,
                        dimension: 1,
                    };
                }
            }
        }
        unimplemented!(
            "AbelianVarietyNewform::new: the coefficient-field degree is only \
             computed for weight 2 at levels where S_2(Gamma0(N)) is \
             1-dimensional and entirely new; general newform decomposition is \
             not yet implemented"
        )
    }

    /// Get the newform
    pub fn newform(&self) -> &Newform {
        &self.newform
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.newform.level()
    }

    /// Convert to ModularAbelianVariety
    pub fn to_abvar(&self) -> ModularAbelianVariety {
        ModularAbelianVariety::new(self.level(), self.dimension)
    }
}

/// Homology of a modular abelian variety
#[derive(Debug, Clone)]
pub struct Homology {
    /// The abelian variety
    abvar: ModularAbelianVariety,
    /// Rank of the homology (as a Z-module)
    rank: usize,
}

impl Homology {
    /// Create homology of an abelian variety
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        // H_1(A, Z) has rank 2 * dim(A)
        let rank = 2 * abvar.dimension();
        Homology { abvar, rank }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the underlying abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

/// Torsion subgroup of an abelian variety
#[derive(Debug, Clone)]
pub struct TorsionSubgroup {
    /// The abelian variety
    abvar: ModularAbelianVariety,
    /// Order (if finite and known)
    order: Option<Integer>,
}

impl TorsionSubgroup {
    /// Create torsion subgroup
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        TorsionSubgroup { abvar, order: None }
    }

    /// Get the order (if known)
    pub fn order(&self) -> Option<&Integer> {
        self.order.as_ref()
    }

    /// Set the order
    pub fn set_order(&mut self, order: Integer) {
        self.order = Some(order);
    }

    /// Get the underlying abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

/// Cuspidal subgroup of J_0(N)
#[derive(Debug, Clone)]
pub struct CuspidalSubgroup {
    /// Level N
    level: u64,
    /// Generators
    generators: Vec<Vec<Integer>>,
}

impl CuspidalSubgroup {
    /// Create cuspidal subgroup of J_0(N)
    pub fn new(level: u64) -> Self {
        CuspidalSubgroup {
            level,
            generators: Vec::new(),
        }
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Add a generator
    pub fn add_generator(&mut self, gen: Vec<Integer>) {
        self.generators.push(gen);
    }

    /// Get generators
    pub fn generators(&self) -> &[Vec<Integer>] {
        &self.generators
    }
}

/// Endomorphism ring of an abelian variety
#[derive(Debug, Clone)]
pub struct EndomorphismRing {
    /// The abelian variety
    abvar: ModularAbelianVariety,
    /// Rank (dimension as a Z-module)
    rank: usize,
}

impl EndomorphismRing {
    /// Create endomorphism ring
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        // End(A) ⊗ Q has dimension at most 4 * dim(A)^2
        let rank = 4 * abvar.dimension() * abvar.dimension();
        EndomorphismRing { abvar, rank }
    }

    /// Get the rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the underlying abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_abelian_variety() {
        let a = ModularAbelianVariety::new(11, 1);
        assert_eq!(a.level(), 11);
        assert_eq!(a.dimension(), 1);
        // dimension 1 => simple (an elliptic curve has no nonzero proper
        // abelian subvarieties); realized in stage 2, previously a facade.
        assert!(a.is_simple());
    }

    #[test]
    fn test_j0() {
        let j0_11 = J0::new(11);
        assert_eq!(j0_11.level(), 11);
        // dim J_0(11) = genus of X_0(11) = 1 (an elliptic curve).
        assert_eq!(j0_11.dimension(), 1);
    }

    #[test]
    fn test_j1() {
        let j1_11 = J1::new(11);
        assert_eq!(j1_11.level(), 11);
        // dim J_1(11) = genus of X_1(11) = 1 (equal to dim J_0(11) here).
        assert_eq!(j1_11.dimension(), 1);
    }

    #[test]
    fn test_modular_curve() {
        let x0_11 = ModularCurve::X0(11);
        assert_eq!(x0_11.level(), 11);
        // genus of X_0(11) = 1.
        assert_eq!(x0_11.genus(), 1);

        let jac = x0_11.jacobian();
        assert_eq!(jac.level(), 11);
        assert_eq!(jac.dimension(), 1);
    }

    #[test]
    fn test_homology() {
        let a = ModularAbelianVariety::new(11, 1);
        let h = Homology::new(a);
        assert_eq!(h.rank(), 2); // 2 * dim = 2 * 1 = 2
    }

    #[test]
    fn test_torsion_subgroup() {
        let a = ModularAbelianVariety::new(11, 1);
        let mut tors = TorsionSubgroup::new(a);
        assert_eq!(tors.order(), None);

        tors.set_order(Integer::from(5));
        assert_eq!(tors.order(), Some(&Integer::from(5)));
    }

    #[test]
    fn test_cuspidal_subgroup() {
        let cusp = CuspidalSubgroup::new(11);
        assert_eq!(cusp.level(), 11);
        assert_eq!(cusp.generators().len(), 0);
    }

    #[test]
    fn test_endomorphism_ring() {
        let a = ModularAbelianVariety::new(11, 1);
        let end = EndomorphismRing::new(a);
        assert_eq!(end.rank(), 4); // 4 * 1 * 1 = 4
    }

    #[test]
    fn test_abvar_from_newform() {
        // Weight 2, level 11: genus(X_0(11)) = 1 and every proper divisor
        // has genus 0, so the unique newform is rational and A_f is an
        // elliptic curve.  Realized in stage 2, previously a facade.
        let f = Newform::new(2, 11);
        let a = AbelianVarietyNewform::new(f);
        assert_eq!(a.level(), 11);
        assert_eq!(a.dimension(), 1);
    }

    #[test]
    fn test_abvar_from_newform_more_genus_one_levels() {
        for n in [14u64, 15, 17, 19, 20, 21, 24, 27, 32, 36, 49] {
            let a = AbelianVarietyNewform::new(Newform::new(2, n));
            assert_eq!(a.level(), n);
            assert_eq!(a.dimension(), 1, "A_f dimension at level {n}");
        }
    }

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_abvar_from_newform_level_with_old_part_unimplemented() {
        // N = 22: genus(X_0(22)) = 2 with a nonzero old part from level 11;
        // the coefficient-field computation is honestly refused.
        let _ = AbelianVarietyNewform::new(Newform::new(2, 22));
    }
}

/// ModularAbelianVariety constructed from modular symbols
#[derive(Debug, Clone)]
pub struct ModularAbelianVarietyModsym {
    /// Base modular abelian variety
    base: ModularAbelianVariety,
    /// Associated modular symbol space
    modsym_space: Option<ModularSymbolSpace>,
}

impl ModularAbelianVarietyModsym {
    /// Create from a modular symbol space
    pub fn from_modsym_space(space: ModularSymbolSpace) -> Self {
        let level = space.level();
        let dimension = space.dimension();
        ModularAbelianVarietyModsym {
            base: ModularAbelianVariety::new(level, dimension),
            modsym_space: Some(space),
        }
    }

    /// Get the base abelian variety
    pub fn base(&self) -> &ModularAbelianVariety {
        &self.base
    }

    /// Get the modular symbol space
    pub fn modsym_space(&self) -> Option<&ModularSymbolSpace> {
        self.modsym_space.as_ref()
    }

    /// Get level
    pub fn level(&self) -> u64 {
        self.base.level()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.base.dimension()
    }
}

/// Check if an object is a ModularAbelianVariety
pub fn is_modular_abelian_variety(obj: &ModularAbelianVariety) -> bool {
    // In Rust, type checking is done at compile time
    // This function exists for API compatibility
    true
}

/// Factor a modular symbols space into new factors
pub fn factor_modsym_space_new_factors(space: &ModularSymbolSpace) -> Vec<ModularSymbolSpace> {
    // Decomposing into newform factors requires sophisticated algorithms
    // from modular forms theory (e.g. Hecke algebra eigenspace splitting)
    // that are not yet implemented. Previously this returned `vec![space.clone()]`
    // unconditionally, i.e. pretended the space was already a single factor.
    let _ = space;
    unimplemented!(
        "factor_modsym_space_new_factors not yet implemented (facade): previously returned the input space unchanged"
    )
}

/// Factor the new space of modular symbols
pub fn factor_new_space(level: u64, weight: i32) -> Vec<ModularSymbolSpace> {
    // Factoring the new subspace requires the newform decomposition, which
    // is not yet implemented. Previously this just wrapped a freshly
    // constructed space without doing any factoring.
    let _ = (level, weight);
    unimplemented!(
        "factor_new_space not yet implemented (facade): previously returned an unfactored space"
    )
}

/// Generate a random Hecke operator for testing
pub fn random_hecke_operator(level: u64, max_index: u64) -> HeckeOperator {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let n = rng.gen_range(1..=max_index);
    HeckeOperator::new(n, level)
}

/// Compute modular symbol lattices
pub fn modsym_lattices(space: &ModularSymbolSpace) -> Vec<Vec<Rational>> {
    // Computing period lattices requires numerical/algebraic integration of
    // modular symbols against a period map, which is not implemented here.
    // Previously this returned a single fake basis vector `[[1]]`.
    let _ = space;
    unimplemented!(
        "modsym_lattices not yet implemented (facade): previously returned a fake constant lattice [[1]]"
    )
}

/// Simple factorization of a modular symbols space
pub fn simple_factorization_of_modsym_space(space: &ModularSymbolSpace) -> Vec<ModularSymbolSpace> {
    // Decomposing into simple (irreducible) Hecke-stable factors is not yet
    // implemented. Previously this returned `vec![space.clone()]`
    // unconditionally, i.e. pretended the space was already simple.
    let _ = space;
    unimplemented!(
        "simple_factorization_of_modsym_space not yet implemented (facade): previously returned the input space unchanged"
    )
}

/// Compute the square root of a polynomial, or `None` if it cannot be computed.
///
/// A genuine polynomial square-root algorithm is not implemented, so this
/// returns `None` to mean "no square root computed" — a usable, honest
/// "no result" that callers treat as "could not obtain a square root",
/// rather than panicking.
pub fn sqrt_poly(coeffs: &[Rational]) -> Option<Vec<Rational>> {
    let _ = coeffs;
    None
}

/// Finite subgroup of an abelian variety
#[derive(Debug, Clone)]
pub struct FiniteSubgroup {
    /// Parent abelian variety
    abvar: ModularAbelianVariety,
    /// Generators as vectors
    generators: Vec<Vec<Integer>>,
}

impl FiniteSubgroup {
    /// Create a new finite subgroup
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        FiniteSubgroup {
            abvar,
            generators: Vec::new(),
        }
    }

    /// Add a generator
    pub fn add_generator(&mut self, gen: Vec<Integer>) {
        self.generators.push(gen);
    }

    /// Get generators
    pub fn generators(&self) -> &[Vec<Integer>] {
        &self.generators
    }

    /// Get the parent abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

/// Finite subgroup defined by a lattice
#[derive(Debug, Clone)]
pub struct FiniteSubgroupLattice {
    /// Base finite subgroup
    base: FiniteSubgroup,
    /// Lattice basis
    lattice_basis: Vec<Vec<Integer>>,
}

impl FiniteSubgroupLattice {
    /// Create from a lattice
    pub fn new(abvar: ModularAbelianVariety, lattice_basis: Vec<Vec<Integer>>) -> Self {
        FiniteSubgroupLattice {
            base: FiniteSubgroup::new(abvar),
            lattice_basis,
        }
    }

    /// Get lattice basis
    pub fn lattice_basis(&self) -> &[Vec<Integer>] {
        &self.lattice_basis
    }
}

/// Homomorphism space between abelian varieties
#[derive(Debug, Clone)]
pub struct Homspace {
    /// Domain abelian variety
    domain: ModularAbelianVariety,
    /// Codomain abelian variety
    codomain: ModularAbelianVariety,
}

impl Homspace {
    /// Create a homomorphism space.
    ///
    /// Construction is metadata-only and never panics; only
    /// [`Homspace::dimension`] (the genuinely unimplemented quantity) panics
    /// when actually requested.
    pub fn new(domain: ModularAbelianVariety, codomain: ModularAbelianVariety) -> Self {
        Homspace { domain, codomain }
    }

    /// Get domain
    pub fn domain(&self) -> &ModularAbelianVariety {
        &self.domain
    }

    /// Get codomain
    pub fn codomain(&self) -> &ModularAbelianVariety {
        &self.codomain
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        // The dimension of Hom(A, B) depends on the isogeny decomposition of
        // both varieties, which is not yet computed. Previously this was
        // hardcoded to 0, i.e. silently claimed no homomorphisms exist.
        unimplemented!(
            "Homspace::dimension not yet implemented (facade): depends on the isogeny decomposition of domain and codomain, which is not computed"
        )
    }
}

/// Endomorphism subring (when domain = codomain)
#[derive(Debug, Clone)]
pub struct EndomorphismSubring {
    /// Homomorphism space (domain = codomain)
    homspace: Homspace,
}

impl EndomorphismSubring {
    /// Create endomorphism subring
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        let homspace = Homspace::new(abvar.clone(), abvar);
        EndomorphismSubring { homspace }
    }

    /// Get the underlying homspace
    pub fn homspace(&self) -> &Homspace {
        &self.homspace
    }
}

/// L-series attached to a modular abelian variety
#[derive(Debug, Clone)]
pub struct Lseries {
    /// Parent abelian variety
    abvar: ModularAbelianVariety,
}

impl Lseries {
    /// Create L-series
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        Lseries { abvar }
    }

    /// Get the parent abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

/// Complex L-series
#[derive(Debug, Clone)]
pub struct LseriesComplex {
    /// Base L-series
    base: Lseries,
}

impl LseriesComplex {
    /// Create complex L-series
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        LseriesComplex {
            base: Lseries::new(abvar),
        }
    }

    /// Evaluate at a complex number
    pub fn evaluate(&self, _s: f64) -> f64 {
        // Computing L(s) for the L-series of a modular abelian variety
        // (e.g. via the L-functions of its newform factors) is not yet
        // implemented. Previously this unconditionally returned 0.0.
        unimplemented!(
            "LseriesComplex::evaluate not yet implemented (facade): previously returned the constant 0.0"
        )
    }
}

/// p-adic L-series
#[derive(Debug, Clone)]
pub struct LseriesPadic {
    /// Base L-series
    base: Lseries,
    /// Prime p
    p: u64,
}

impl LseriesPadic {
    /// Create p-adic L-series
    pub fn new(abvar: ModularAbelianVariety, p: u64) -> Self {
        LseriesPadic {
            base: Lseries::new(abvar),
            p,
        }
    }

    /// Get the prime
    pub fn prime(&self) -> u64 {
        self.p
    }
}

/// Morphism between modular abelian varieties
#[derive(Debug, Clone)]
pub struct Morphism {
    /// Domain
    domain: ModularAbelianVariety,
    /// Codomain
    codomain: ModularAbelianVariety,
    /// Matrix representation (on homology)
    matrix: Vec<Vec<Rational>>,
}

impl Morphism {
    /// Create a morphism
    pub fn new(
        domain: ModularAbelianVariety,
        codomain: ModularAbelianVariety,
        matrix: Vec<Vec<Rational>>,
    ) -> Self {
        Morphism {
            domain,
            codomain,
            matrix,
        }
    }

    /// Get domain
    pub fn domain(&self) -> &ModularAbelianVariety {
        &self.domain
    }

    /// Get codomain
    pub fn codomain(&self) -> &ModularAbelianVariety {
        &self.codomain
    }

    /// Get matrix
    pub fn matrix(&self) -> &[Vec<Rational>] {
        &self.matrix
    }
}

/// Degeneracy map between Jacobians
#[derive(Debug, Clone)]
pub struct DegeneracyMap {
    /// Base morphism
    morphism: Morphism,
    /// Degeneracy parameter
    param: u64,
}

impl DegeneracyMap {
    /// Create a degeneracy map
    pub fn new(domain: ModularAbelianVariety, codomain: ModularAbelianVariety, param: u64) -> Self {
        // The degeneracy map's matrix representation on homology is not
        // computed here. Previously this constructed a `Morphism` with an
        // empty matrix, silently pretending it was the zero-dimensional map.
        let _ = (domain, codomain, param);
        unimplemented!(
            "DegeneracyMap::new not yet implemented (facade): previously used an empty placeholder matrix"
        )
    }

    /// Get the morphism
    pub fn morphism(&self) -> &Morphism {
        &self.morphism
    }

    /// Get parameter
    pub fn param(&self) -> u64 {
        self.param
    }
}

/// Homology with additional structure
#[derive(Debug, Clone)]
pub struct HomologyOverBase {
    /// Base homology
    base: Homology,
    /// Base ring (Z, Q, Z/NZ, etc.)
    base_ring: String,
}

impl HomologyOverBase {
    /// Create homology over a specific base
    pub fn new(abvar: ModularAbelianVariety, base_ring: String) -> Self {
        HomologyOverBase {
            base: Homology::new(abvar),
            base_ring,
        }
    }

    /// Get base ring
    pub fn base_ring(&self) -> &str {
        &self.base_ring
    }

    /// Get the base homology
    pub fn base(&self) -> &Homology {
        &self.base
    }
}

/// Homology submodule
#[derive(Debug, Clone)]
pub struct HomologySubmodule {
    /// Ambient homology
    ambient: Homology,
    /// Generators of the submodule
    generators: Vec<Vec<Integer>>,
}

impl HomologySubmodule {
    /// Create a submodule
    pub fn new(ambient: Homology, generators: Vec<Vec<Integer>>) -> Self {
        HomologySubmodule {
            ambient,
            generators,
        }
    }

    /// Get ambient homology
    pub fn ambient(&self) -> &Homology {
        &self.ambient
    }

    /// Get generators
    pub fn generators(&self) -> &[Vec<Integer>] {
        &self.generators
    }
}

/// Rational cusp subgroup
#[derive(Debug, Clone)]
pub struct RationalCuspSubgroup {
    /// Base cuspidal subgroup
    base: CuspidalSubgroup,
}

impl RationalCuspSubgroup {
    /// Create rational cusp subgroup
    pub fn new(level: u64) -> Self {
        RationalCuspSubgroup {
            base: CuspidalSubgroup::new(level),
        }
    }

    /// Get the base
    pub fn base(&self) -> &CuspidalSubgroup {
        &self.base
    }
}

/// Rational cuspidal subgroup (full rational torsion from cusps)
#[derive(Debug, Clone)]
pub struct RationalCuspidalSubgroup {
    /// Base cuspidal subgroup
    base: CuspidalSubgroup,
}

impl RationalCuspidalSubgroup {
    /// Create rational cuspidal subgroup
    pub fn new(level: u64) -> Self {
        RationalCuspidalSubgroup {
            base: CuspidalSubgroup::new(level),
        }
    }

    /// Get the base
    pub fn base(&self) -> &CuspidalSubgroup {
        &self.base
    }
}

/// Check if a cusp is rational for Gamma0
pub fn is_rational_cusp_gamma0(numerator: i64, denominator: i64, level: u64) -> bool {
    use crate::cusps::Cusp;

    fn gcd_u64(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd_u64(b, a % b) }
    }

    let cusp = Cusp::from_i64(numerator, denominator);
    // A cusp p/q is rational for Gamma0(N) if gcd(q, N) = 1
    if let Some(q) = cusp.denominator() {
        let q_val = q.to_string().parse::<u64>().unwrap_or(level);
        gcd_u64(q_val, level) == 1
    } else {
        // Infinity is always rational
        true
    }
}

/// QQbar torsion subgroup (torsion over algebraic closure)
#[derive(Debug, Clone)]
pub struct QQbarTorsionSubgroup {
    /// Parent abelian variety
    abvar: ModularAbelianVariety,
}

impl QQbarTorsionSubgroup {
    /// Create QQbar torsion subgroup
    pub fn new(abvar: ModularAbelianVariety) -> Self {
        QQbarTorsionSubgroup { abvar }
    }

    /// Get parent abelian variety
    pub fn abvar(&self) -> &ModularAbelianVariety {
        &self.abvar
    }
}

/// Abelian variety constructor functions
pub mod constructor {
    use super::*;

    /// Generic abelian variety constructor
    pub fn abelian_variety(level: u64, dimension: usize) -> ModularAbelianVariety {
        ModularAbelianVariety::new(level, dimension)
    }

    /// JH - Jacobian of X_H(N)
    pub fn jh(level: u64, h_subgroup: Vec<u64>) -> ModularAbelianVariety {
        // The genus/dimension formula for X_H(N) (more complex than X_0(N)
        // or X_1(N), depending on the subgroup H) is not yet implemented.
        // Previously this always used the constant dimension 1.
        let _ = (level, h_subgroup);
        unimplemented!(
            "constructor::jh not yet implemented (facade): dimension previously hardcoded to 1"
        )
    }
}
