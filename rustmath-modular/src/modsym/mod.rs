//! Modular symbols
//!
//! Modular symbols provide a way to compute with modular forms using
//! homological algebra. They form a link between modular forms and
//! homology groups.
//!
//! Corresponds to `sage.modular.modsym` and the MAGMA handbook chapter
//! "Modular Symbols".
//!
//! Submodules:
//! - [`p1list`]: the projective line P^1(Z/NZ) with canonical
//!   normalization (indexes the Manin generators).
//! - [`gamma0`]: the Manin-symbol presentation of M_2(Gamma0(N)) over Q,
//!   with quotient basis, boundary map to the cusps, cuspidal subspace,
//!   and conversion {alpha, beta} -> basis coordinates (Manin trick).
//! - [`heilbronn`]: Merel's determinant-n matrix families for the Hecke
//!   action on Manin symbols.
//! - [`hecke`]: the Hecke operators T_n / U_p on [`ModularSymbolsGamma0`],
//!   their restriction to the cuspidal subspace, and exact characteristic
//!   polynomials (Eichler-Shimura).
//! - [`decomposition`]: the Hecke-eigenspace (newform-style) decomposition
//!   of the cuspidal subspace into Q-irreducible Hecke-stable summands,
//!   with honest algebraic eigenvalues (irreducible polynomials) for
//!   coefficient fields of degree > 1.
//! - [`involutions`]: the star involution (+/- eigenspace split, signed
//!   decompositions) and the Atkin-Lehner involutions W_Q for Q || N,
//!   including their per-summand signs.
//! - [`degeneracy`]: the degeneracy maps between levels M | N (lowering
//!   and raising/transfer) and the old/new splitting of the cuspidal
//!   subspace.
//! - [`winding`]: the winding element {0, oo}, Hecke-equivariant
//!   projections onto summands, and the EXACT L(f, 1) = 0 criterion
//!   (Manin--Birch) -- the certified-zero source for L-values.
//! - [`lseries`]: numeric L(f, 1) and L'(f, 1) for rational newforms over
//!   BigFloat, with rigorous tails and a documented rounding allowance --
//!   the certified-NONZERO source.

pub mod decomposition;
pub mod degeneracy;
pub mod gamma0;
pub mod hecke;
pub mod heilbronn;
pub mod involutions;
pub mod lseries;
pub mod p1list;
pub mod winding;

pub use decomposition::{
    CuspidalHeckeDecomposition, HeckeEigenvalue, HeckeSummand, SummandHeckeAction,
};
pub use degeneracy::{SummandProvenance, UlPiece, UlRefinement};
pub use gamma0::{cusps_equivalent_gamma0, ModularSymbolsGamma0};
pub use heilbronn::merel_matrices;
pub use involutions::{InvolutionAction, SummandInvolutions};
pub use lseries::{euler_gamma, exp_integral_e1, LValue, RationalNewformLSeries};
pub use p1list::P1List;

use crate::cusps::Cusp;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// A modular symbol {α → β}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularSymbol {
    /// Starting cusp
    alpha: Cusp,
    /// Ending cusp
    beta: Cusp,
}

impl ModularSymbol {
    /// Create a new modular symbol {α → β}
    pub fn new(alpha: Cusp, beta: Cusp) -> Self {
        ModularSymbol { alpha, beta }
    }

    /// Get the starting cusp
    pub fn alpha(&self) -> &Cusp {
        &self.alpha
    }

    /// Get the ending cusp
    pub fn beta(&self) -> &Cusp {
        &self.beta
    }

    /// Reverse the modular symbol: {α → β} becomes {β → α}
    pub fn reverse(&self) -> Self {
        ModularSymbol {
            alpha: self.beta.clone(),
            beta: self.alpha.clone(),
        }
    }

    /// Apply a matrix [[a,b],[c,d]] to the modular symbol
    /// {α → β} maps to {γα → γβ}
    pub fn apply_matrix(
        &self,
        a: &Integer,
        b: &Integer,
        c: &Integer,
        d: &Integer,
    ) -> ModularSymbol {
        ModularSymbol {
            alpha: self.alpha.apply_matrix(a, b, c, d),
            beta: self.beta.apply_matrix(a, b, c, d),
        }
    }
}

/// A formal linear combination of modular symbols
#[derive(Debug, Clone)]
pub struct ModularSymbolSpace {
    /// Weight of the modular symbols
    weight: i32,
    /// Level (for congruence subgroups)
    level: u64,
    /// Sign (+1 for +1 eigenspace, -1 for -1 eigenspace, 0 for both)
    sign: i8,
    /// Basis of modular symbols
    basis: Vec<ModularSymbol>,
    /// Dimension
    dimension: usize,
}

impl ModularSymbolSpace {
    /// Create a new modular symbol space
    pub fn new(weight: i32, level: u64, sign: i8) -> Self {
        assert!(sign == -1 || sign == 0 || sign == 1);
        ModularSymbolSpace {
            weight,
            level,
            sign,
            basis: Vec::new(),
            dimension: 0,
        }
    }

    /// Get the weight
    pub fn weight(&self) -> i32 {
        self.weight
    }

    /// Get the level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Get the sign
    pub fn sign(&self) -> i8 {
        self.sign
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the basis
    pub fn basis(&self) -> &[ModularSymbol] {
        &self.basis
    }

    /// Add a basis element
    pub fn add_basis_element(&mut self, symbol: ModularSymbol) {
        self.basis.push(symbol);
        self.dimension += 1;
    }

    /// The dimension of the space of modular symbols for Gamma0(N) of this
    /// weight and sign.
    ///
    /// PANICS for any weight other than 2; see
    /// [`Self::try_compute_dimension_gamma0`], which says why in full.  (This
    /// used to return an "approximate genus" `index/12 - 1` for weight 2 and a
    /// flat 0 for every other weight.)
    pub fn compute_dimension_gamma0(&self) -> usize {
        self.try_compute_dimension_gamma0()
            .expect("compute_dimension_gamma0")
    }

    /// The dimension of the space of modular symbols for Gamma0(N) of this
    /// weight and sign, or an honest error.
    ///
    /// # Weight 2
    ///
    /// For sign 0 the dimension is `2g + c - 1`, with `g` the genus of X_0(N)
    /// and `c` its number of cusps -- both taken EXACTLY from
    /// [`crate::dims::gamma0_invariants`] (2g from the cuspidal part, via
    /// Eichler-Shimura, and c - 1 from the boundary).  The tests check this
    /// against [`ModularSymbolsGamma0::dimension`], which computes the same
    /// number independently by reducing the Manin symbols, and against PARI/GP's
    /// `msdim(msinit(N, 2))`.
    ///
    /// For sign +/-1 there is NO such closed formula: the star involution's
    /// eigenspaces split the boundary part unevenly, and the naive guess
    /// `dim^+ = g + c - 1`, `dim^- = g` is simply false (PARI: at N = 9 the split
    /// of the 3-dimensional space is 2/1, not 3/0; likewise at N = 16, 18, 25,
    /// 27).  So the +/- dimensions are DELEGATED to
    /// [`ModularSymbolsGamma0::star_eigenspace_ambient`], which computes the
    /// eigenspace as an actual kernel.
    ///
    /// # Every other weight
    ///
    /// Refused.  This crate's modular symbols engine
    /// ([`ModularSymbolsGamma0`], and the Manin-symbol presentation behind it)
    /// is weight 2 only, and no dimension formula for weight != 2 is gated
    /// anywhere in the crate; returning a number here would be inventing one.
    pub fn try_compute_dimension_gamma0(&self) -> Result<usize, String> {
        if self.level == 0 {
            return Err("Gamma0(0) is not a group: the level must be >= 1".to_string());
        }
        if self.weight != 2 {
            return Err(format!(
                "modular symbols of weight {} for Gamma0({}) are not implemented: the \
                 Manin-symbol engine of this crate (ModularSymbolsGamma0) is weight 2 \
                 only, and no weight != 2 dimension formula is certified anywhere in \
                 the crate. Refusing rather than returning a number nothing computed.",
                self.weight, self.level
            ));
        }

        match self.sign {
            0 => {
                let inv = crate::dims::gamma0_invariants(self.level)?;
                // 2g (cuspidal, by Eichler-Shimura) + (c - 1) (boundary).
                let d = 2 * inv.genus + inv.cusps - 1;
                usize::try_from(d).map_err(|_| {
                    format!("dim of modular symbols for Gamma0({}) = {d} overflows usize", self.level)
                })
            }
            s => {
                let space = ModularSymbolsGamma0::new(self.level);
                Ok(space.star_eigenspace_ambient(s)?.len())
            }
        }
    }
}

/// Element of a modular symbol space (linear combination)
#[derive(Debug, Clone)]
pub struct ModularSymbolElement {
    /// Coefficients for each basis element
    coefficients: Vec<Rational>,
    /// Reference to the ambient space
    dimension: usize,
}

impl ModularSymbolElement {
    /// Create a new element
    pub fn new(dimension: usize) -> Self {
        ModularSymbolElement {
            coefficients: vec![Rational::zero(); dimension],
            dimension,
        }
    }

    /// Set coefficient
    pub fn set_coefficient(&mut self, index: usize, value: Rational) {
        if index < self.dimension {
            self.coefficients[index] = value;
        }
    }

    /// Get coefficient
    pub fn coefficient(&self, index: usize) -> Option<&Rational> {
        self.coefficients.get(index)
    }

    /// Add two elements
    pub fn add(&self, other: &ModularSymbolElement) -> Option<ModularSymbolElement> {
        if self.dimension != other.dimension {
            return None;
        }

        let mut result = ModularSymbolElement::new(self.dimension);
        for i in 0..self.dimension {
            result.coefficients[i] = &self.coefficients[i] + &other.coefficients[i];
        }
        Some(result)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: &Rational) -> ModularSymbolElement {
        let mut result = ModularSymbolElement::new(self.dimension);
        for i in 0..self.dimension {
            result.coefficients[i] = &self.coefficients[i] * scalar;
        }
        result
    }
}

/// Manin symbols (for computing modular symbols)
/// A Manin symbol is [P(X,Y), (c:d)] where P is a polynomial
#[derive(Debug, Clone)]
pub struct ManinSymbol {
    /// Coefficients of polynomial P(X,Y) of degree k-2
    polynomial_coeffs: Vec<Integer>,
    /// Cusp (c:d)
    c: Integer,
    d: Integer,
}

impl ManinSymbol {
    /// Create a new Manin symbol
    pub fn new(polynomial_coeffs: Vec<Integer>, c: Integer, d: Integer) -> Self {
        ManinSymbol {
            polynomial_coeffs,
            c,
            d,
        }
    }

    /// Get polynomial coefficients
    pub fn polynomial_coeffs(&self) -> &[Integer] {
        &self.polynomial_coeffs
    }

    /// Get cusp coordinates
    pub fn cusp_coords(&self) -> (&Integer, &Integer) {
        (&self.c, &self.d)
    }
}

/// Space of Manin symbols
pub struct ManinSymbolSpace {
    weight: i32,
    level: u64,
    symbols: Vec<ManinSymbol>,
}

impl ManinSymbolSpace {
    /// Create a new space of Manin symbols
    pub fn new(weight: i32, level: u64) -> Self {
        ManinSymbolSpace {
            weight,
            level,
            symbols: Vec::new(),
        }
    }

    /// Get weight
    pub fn weight(&self) -> i32 {
        self.weight
    }

    /// Get level
    pub fn level(&self) -> u64 {
        self.level
    }

    /// Add a Manin symbol
    pub fn add_symbol(&mut self, symbol: ManinSymbol) {
        self.symbols.push(symbol);
    }

    /// Get all symbols
    pub fn symbols(&self) -> &[ManinSymbol] {
        &self.symbols
    }
}

/// Boundary map from modular symbols to cusps
pub struct BoundaryMap {
    /// Domain dimension
    domain_dim: usize,
    /// Codomain dimension (number of cusps)
    codomain_dim: usize,
    /// Matrix representation
    matrix: Vec<Vec<Integer>>,
}

impl BoundaryMap {
    /// Create a new boundary map
    pub fn new(domain_dim: usize, codomain_dim: usize) -> Self {
        BoundaryMap {
            domain_dim,
            codomain_dim,
            matrix: vec![vec![Integer::zero(); domain_dim]; codomain_dim],
        }
    }

    /// Apply the boundary map to an element
    pub fn apply(&self, element: &ModularSymbolElement) -> Vec<Rational> {
        let mut result = vec![Rational::zero(); self.codomain_dim];
        for i in 0..self.codomain_dim {
            for j in 0..self.domain_dim {
                if let Some(coeff) = element.coefficient(j) {
                    result[i] = &result[i]
                        + &(Rational::from_integer(self.matrix[i][j].clone()) * coeff.clone());
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_symbol_creation() {
        let alpha = Cusp::infinity();
        let beta = Cusp::zero();
        let sym = ModularSymbol::new(alpha, beta);
        assert!(sym.alpha().is_infinity());
        assert_eq!(
            sym.beta().numerator(),
            Some(&Integer::zero())
        );
    }

    #[test]
    fn test_modular_symbol_reverse() {
        let sym = ModularSymbol::new(Cusp::infinity(), Cusp::zero());
        let rev = sym.reverse();
        assert_eq!(rev.alpha().numerator(), Some(&Integer::zero()));
        assert!(rev.beta().is_infinity());
    }

    #[test]
    fn test_modular_symbol_space() {
        let mut space = ModularSymbolSpace::new(2, 11, 0);
        assert_eq!(space.weight(), 2);
        assert_eq!(space.level(), 11);
        assert_eq!(space.dimension(), 0);

        let sym = ModularSymbol::new(Cusp::infinity(), Cusp::zero());
        space.add_basis_element(sym);
        assert_eq!(space.dimension(), 1);
    }

    /// GATE: the weight-2 sign-0 dimension `2g + c - 1`, taken from the exact
    /// Gamma0 invariants, agrees with [`ModularSymbolsGamma0::dimension`], which
    /// gets the same number by an entirely different route (reducing the Manin
    /// symbols modulo the 2- and 3-term relations).  Baked constants are from
    /// PARI/GP `msdim(msinit(N, 2))`.
    #[test]
    fn test_compute_dimension_gamma0_matches_the_manin_symbol_computation() {
        for n in 1..=40u64 {
            let space = ModularSymbolSpace::new(2, n, 0);
            assert_eq!(
                space.compute_dimension_gamma0(),
                ModularSymbolsGamma0::new(n).dimension(),
                "2g + c - 1 vs the Manin-symbol dimension at N = {n}"
            );
        }
        // PARI/GP: msdim(msinit(N, 2)) for N = 1..15
        let pari = [0usize, 1, 1, 2, 1, 3, 1, 3, 3, 3, 3, 5, 1, 5, 5];
        for (i, &d) in pari.iter().enumerate() {
            let n = (i + 1) as u64;
            assert_eq!(
                ModularSymbolSpace::new(2, n, 0).compute_dimension_gamma0(),
                d,
                "msdim(msinit({n}, 2))"
            );
        }
        // the old facade returned (index/12 - 1), e.g. 0 at N = 11 where the
        // true dimension is 3
        assert_eq!(ModularSymbolSpace::new(2, 11, 0).compute_dimension_gamma0(), 3);
    }

    /// GATE: the +/- star eigenspaces are computed, not guessed.  Their
    /// dimensions sum to the sign-0 dimension, and at N = 9 they are 2 and 1 --
    /// NOT the g + c - 1 = 3 and g = 0 that a naive closed formula would give.
    /// (PARI: msdim(msinit(9, 2, 1)) = 2, msdim(msinit(9, 2, -1)) = 1.)
    #[test]
    fn test_star_eigenspace_dimensions_have_no_closed_formula() {
        for n in [9u64, 11, 15, 16, 18, 25, 27] {
            let d0 = ModularSymbolSpace::new(2, n, 0).compute_dimension_gamma0();
            let dp = ModularSymbolSpace::new(2, n, 1).compute_dimension_gamma0();
            let dm = ModularSymbolSpace::new(2, n, -1).compute_dimension_gamma0();
            assert_eq!(dp + dm, d0, "dim^+ + dim^- = dim at N = {n}");

            let inv = crate::dims::gamma0_invariants(n).unwrap();
            let (g, c) = (inv.genus as usize, inv.cusps as usize);
            if n == 9 {
                // the split is 2/1 while the naive formula predicts 3/0
                assert_eq!((dp, dm), (2, 1));
                assert_eq!((g + c - 1, g), (3, 0));
                assert_ne!((dp, dm), (g + c - 1, g));
            }
        }
    }

    /// Honest refusal for weights the crate has no engine for, rather than a
    /// flat 0.
    #[test]
    fn test_nonweight2_modular_symbols_are_refused() {
        for k in [1i32, 3, 4, 6, 12] {
            let space = ModularSymbolSpace::new(k, 11, 0);
            assert!(
                space.try_compute_dimension_gamma0().is_err(),
                "weight {k} must be refused, not answered with 0"
            );
        }
        assert!(ModularSymbolSpace::new(2, 11, 0).try_compute_dimension_gamma0().is_ok());
    }

    #[test]
    #[should_panic(expected = "compute_dimension_gamma0")]
    fn test_weight4_panics_rather_than_returning_zero() {
        let _ = ModularSymbolSpace::new(4, 11, 0).compute_dimension_gamma0();
    }

    #[test]
    fn test_modular_symbol_element() {
        let mut elem = ModularSymbolElement::new(3);
        elem.set_coefficient(0, Rational::one());
        elem.set_coefficient(1, Rational::from_integer(Integer::from(2)));

        assert_eq!(elem.coefficient(0), Some(&Rational::one()));
        assert_eq!(
            elem.coefficient(1),
            Some(&Rational::from_integer(Integer::from(2)))
        );
    }

    #[test]
    fn test_modular_symbol_addition() {
        let mut elem1 = ModularSymbolElement::new(2);
        elem1.set_coefficient(0, Rational::one());

        let mut elem2 = ModularSymbolElement::new(2);
        elem2.set_coefficient(0, Rational::from_integer(Integer::from(2)));

        let sum = elem1.add(&elem2).unwrap();
        assert_eq!(
            sum.coefficient(0),
            Some(&Rational::from_integer(Integer::from(3)))
        );
    }

    #[test]
    fn test_manin_symbol() {
        let poly = vec![Integer::one(), Integer::from(2)];
        let sym = ManinSymbol::new(poly, Integer::one(), Integer::zero());
        assert_eq!(sym.polynomial_coeffs().len(), 2);
        assert_eq!(sym.cusp_coords(), (&Integer::one(), &Integer::zero()));
    }
}
