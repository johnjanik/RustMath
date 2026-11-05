//! The homogeneous Belyi ansatz (data model) and the direct encoder producing a
//! [`PolySystem`](rustmath_polynomials::poly_system::PolySystem).
//!
//! Ported from `dessin_engine/src/belyi_system.rs` (the ansatz data model, D1′)
//! and `dessin_engine/src/belyi_encode.rs` (the encoder, D1) in
//! `/home/john/inverse_galois/M23/dessin_engine`. The reference implementation's
//! private `MPoly`/`MPolySystem` are replaced by this crate's dependency
//! `rustmath_polynomials::multivariate::MultivariatePolynomial<Integer>` and
//! `rustmath_polynomials::poly_system::PolySystem`.
//!
//! For the genus-0 fast path, `φ = [P(Y,Z) : Q(Y,Z)]` with `P,Q` binary forms of
//! degree `n`, and the branch pattern is imposed factor-by-factor:
//! `P = ∏ Aᵢ^{mᵢ}` over `t=0`, `Q = ∏ Rⱼ^{kⱼ}` over `t=∞`, and
//! `P − Q = c·∏ Uₗ^{lₗ}` over `t=1`. The identity `P − Q − c·W ≡ 0` of degree-`n`
//! binary forms expands to `n+1` coefficient equations in the unknown factor
//! coefficients (and `c`).

use rustmath_integers::Integer;
use rustmath_polynomials::multivariate::{Monomial, MultivariatePolynomial};
use rustmath_polynomials::poly_system::PolySystem;
use std::collections::BTreeMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Data model (ported from belyi_system.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AnsatzError {
    #[error("fiber {fiber:?}: Σ deg·mult = {got}, expected degree {expected}")]
    DegreeSumMismatch {
        fiber: BranchValue,
        got: usize,
        expected: usize,
    },
    #[error("fewer than 3 pinned points: PGL2 freedom not killed")]
    UnderNormalized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchValue {
    Zero,
    One,
    Infinity,
}

/// One irreducible factor of a fiber: `name^multiplicity`, `name` of given degree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HomogeneousFactor {
    pub name: String,
    pub degree: usize,
    pub multiplicity: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchFiberPattern {
    pub branch_value: BranchValue,
    pub factors: Vec<HomogeneousFactor>,
}

impl BranchFiberPattern {
    /// `Σ deg·mult` — must equal the cover degree.
    pub fn weighted_degree(&self) -> usize {
        self.factors.iter().map(|f| f.degree * f.multiplicity).sum()
    }

    /// The cycle type this fiber imposes (a multiplicity-`m`, degree-`d` factor
    /// contributes `d` cycles of length `m`), descending.
    pub fn cycle_type(&self) -> Vec<usize> {
        let mut ct = Vec::new();
        for f in &self.factors {
            for _ in 0..f.degree {
                ct.push(f.multiplicity);
            }
        }
        ct.sort_unstable_by(|a, b| b.cmp(a));
        ct
    }
}

/// A pinned geometric point in `P¹_Y` (homogeneous `[y:z]`). `[1:0]` is infinity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedPoint {
    pub label: String,
    pub y: i64,
    pub z: i64,
}

#[derive(Debug, Clone)]
pub struct GenusZeroBelyiAnsatz {
    pub degree: usize,
    pub zero_fiber: BranchFiberPattern,
    pub one_fiber: BranchFiberPattern,
    pub infinity_fiber: BranchFiberPattern,
    /// At least three labelled points to kill PGL₂ (one may be `[1:0]`).
    pub pins: Vec<PinnedPoint>,
}

impl GenusZeroBelyiAnsatz {
    /// Check `Σ deg·mult = degree` on each fiber and that ≥ 3 points are pinned.
    pub fn validate(&self) -> Result<(), AnsatzError> {
        for fib in [&self.zero_fiber, &self.one_fiber, &self.infinity_fiber] {
            let got = fib.weighted_degree();
            if got != self.degree {
                return Err(AnsatzError::DegreeSumMismatch {
                    fiber: fib.branch_value,
                    got,
                    expected: self.degree,
                });
            }
        }
        if self.pins.len() < 3 {
            return Err(AnsatzError::UnderNormalized);
        }
        Ok(())
    }

    /// The three imposed cycle types `(over 0, over 1, over ∞)`.
    pub fn cycle_types(&self) -> [Vec<usize>; 3] {
        [
            self.zero_fiber.cycle_type(),
            self.one_fiber.cycle_type(),
            self.infinity_fiber.cycle_type(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Encoder (ported from belyi_encode.rs)
// ---------------------------------------------------------------------------

/// A factor coefficient: a fixed normalization constant, or a named unknown.
#[derive(Debug, Clone)]
pub enum Coeff {
    Fixed(i64),
    Unknown(String),
}

/// One irreducible factor `F^mult`, `F` a binary form of degree `coeffs.len()-1`
/// with coefficients ascending in the `Y`-power: `F = Σ_i coeffs[i] · Y^i Z^{d-i}`.
#[derive(Debug, Clone)]
pub struct FactorSpec {
    pub mult: u32,
    pub coeffs: Vec<Coeff>,
}

impl FactorSpec {
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }
}

/// The ansatz: the three fibers and the scalar `c`.
#[derive(Debug, Clone)]
pub struct BelyiAnsatzSystem {
    pub zero: Vec<FactorSpec>,
    pub inf: Vec<FactorSpec>,
    pub one: Vec<FactorSpec>,
    pub c: Coeff,
}

/// The encoded system plus the ordered names of its unknown variables.
pub struct Encoded {
    pub system: PolySystem,
    pub var_names: Vec<String>,
    pub degree: usize,
}

impl BelyiAnsatzSystem {
    fn total_degree(fibers: &[FactorSpec]) -> usize {
        fibers.iter().map(|f| f.degree() * f.mult as usize).sum()
    }

    /// Encode into the coefficient-identity system `P − Q − c·W = 0`.
    ///
    /// # Panics
    /// Panics if the three fibers do not share a common degree `n`.
    pub fn encode(&self) -> Encoded {
        let n = Self::total_degree(&self.zero);
        assert_eq!(Self::total_degree(&self.inf), n, "inf fiber degree");
        assert_eq!(Self::total_degree(&self.one), n, "one fiber degree");

        // Assign variable indices to every Unknown coefficient (and c), in order.
        let mut names: Vec<String> = Vec::new();
        for f in self.zero.iter().chain(&self.inf).chain(&self.one) {
            for c in &f.coeffs {
                if let Coeff::Unknown(nm) = c {
                    names.push(nm.clone());
                }
            }
        }
        if let Coeff::Unknown(nm) = &self.c {
            names.push(nm.clone());
        }
        let u = names.len();
        let idx_of = |nm: &str| names.iter().position(|x| x == nm).unwrap();

        // Full layout: unknowns 0..u, then Y = u, Z = u+1.
        let (yi, zi) = (u, u + 1);

        let mono = |pairs: &[(usize, u32)]| -> Monomial {
            let mut map = BTreeMap::new();
            for &(v, e) in pairs {
                if e > 0 {
                    map.insert(v, e);
                }
            }
            Monomial::from_exponents(map)
        };

        // a binary form factor as a MultivariatePolynomial over the full layout
        let build_form = |f: &FactorSpec| -> MultivariatePolynomial<Integer> {
            let d = f.degree();
            let mut acc = MultivariatePolynomial::<Integer>::zero();
            for (i, c) in f.coeffs.iter().enumerate() {
                match c {
                    Coeff::Fixed(v) => {
                        acc.add_term(mono(&[(yi, i as u32), (zi, (d - i) as u32)]), Integer::from(*v));
                    }
                    Coeff::Unknown(nm) => {
                        acc.add_term(
                            mono(&[(idx_of(nm), 1), (yi, i as u32), (zi, (d - i) as u32)]),
                            Integer::one(),
                        );
                    }
                }
            }
            acc
        };

        let poly_pow = |base: &MultivariatePolynomial<Integer>, e: u32| {
            let mut acc = MultivariatePolynomial::<Integer>::constant(Integer::one());
            for _ in 0..e {
                acc = acc * base.clone();
            }
            acc
        };

        let product = |fibers: &[FactorSpec]| -> MultivariatePolynomial<Integer> {
            let mut acc = MultivariatePolynomial::<Integer>::constant(Integer::one());
            for f in fibers {
                acc = acc * poly_pow(&build_form(f), f.mult);
            }
            acc
        };

        let p = product(&self.zero);
        let q = product(&self.inf);
        let w = product(&self.one);

        // c·W
        let cw = match &self.c {
            Coeff::Fixed(v) => w * MultivariatePolynomial::constant(Integer::from(*v)),
            Coeff::Unknown(nm) => {
                w * MultivariatePolynomial::<Integer>::variable(idx_of(nm))
            }
        };

        let d = p - q - cw;

        // Extract one equation per (Y,Z) bidegree: strip Y,Z, keep unknown exps.
        let mut by_bidegree: BTreeMap<(u32, u32), MultivariatePolynomial<Integer>> = BTreeMap::new();
        for (m, coeff) in d.terms() {
            let key = (m.exponent(yi), m.exponent(zi));
            let mut stripped = BTreeMap::new();
            for (&v, &e) in m.iter_exponents() {
                if v != yi && v != zi {
                    stripped.insert(v, e);
                }
            }
            by_bidegree
                .entry(key)
                .or_insert_with(MultivariatePolynomial::zero)
                .add_term(Monomial::from_exponents(stripped), coeff.clone());
        }
        let polys: Vec<MultivariatePolynomial<Integer>> = by_bidegree.into_values().collect();

        Encoded {
            system: PolySystem::new(u, polys),
            var_names: names,
            degree: n,
        }
    }
}

/// The normalized degree-24 `[2,12,5]` ansatz (encoder form):
///   `P = A²B` (A,B deg 8), `Q = R⁵S` (R,S deg 4), `P − Q = c·U¹²` (U deg 2),
/// with branch cycles `2⁸1⁸` (t=0), `12²` (t=1), `5⁴1⁴` (t=∞).
///
/// Normalization (kills the 7-dim PGL₂ × scaling symmetry, leaving a square
/// 25×25 system): scaling `a₈=b₈=r₃=u₂=1`; PGL₂ frame `a₀=0` (a double-zero at
/// `x=0`), `a₁=1` (x-scale), `r₄=0` (a quintuple-pole at `x=∞`).
///
/// Ported from `dessin_engine/src/belyi_encode.rs::ansatz_2_12_5`. Kept as a
/// fixture; the pinned solve route lives in [`crate::belyi::pinned`].
pub fn ansatz_2_12_5() -> BelyiAnsatzSystem {
    let u = |s: &str| Coeff::Unknown(s.into());
    let f = Coeff::Fixed;
    let a = FactorSpec {
        mult: 2,
        coeffs: vec![f(0), f(1), u("a2"), u("a3"), u("a4"), u("a5"), u("a6"), u("a7"), f(1)],
    };
    let b = FactorSpec {
        mult: 1,
        coeffs: vec![u("b0"), u("b1"), u("b2"), u("b3"), u("b4"), u("b5"), u("b6"), u("b7"), f(1)],
    };
    let r = FactorSpec {
        mult: 5,
        coeffs: vec![u("r0"), u("r1"), u("r2"), f(1), f(0)],
    };
    let s = FactorSpec {
        mult: 1,
        coeffs: vec![u("s0"), u("s1"), u("s2"), u("s3"), u("s4")],
    };
    let uu = FactorSpec {
        mult: 12,
        coeffs: vec![u("u0"), u("u1"), f(1)],
    };
    BelyiAnsatzSystem {
        zero: vec![a, b],
        inf: vec![r, s],
        one: vec![uu],
        c: u("c"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_rationals::Rational;

    fn fac(name: &str, degree: usize, multiplicity: usize) -> HomogeneousFactor {
        HomogeneousFactor {
            name: name.into(),
            degree,
            multiplicity,
        }
    }

    /// Data-model [2,12,5] ansatz fixture (ported from belyi_system.rs tests).
    fn ansatz_2_12_5_model() -> GenusZeroBelyiAnsatz {
        GenusZeroBelyiAnsatz {
            degree: 24,
            zero_fiber: BranchFiberPattern {
                branch_value: BranchValue::Zero,
                factors: vec![fac("A", 8, 2), fac("B", 8, 1)],
            },
            one_fiber: BranchFiberPattern {
                branch_value: BranchValue::One,
                factors: vec![fac("U", 2, 12)],
            },
            infinity_fiber: BranchFiberPattern {
                branch_value: BranchValue::Infinity,
                factors: vec![fac("R", 4, 5), fac("S", 4, 1)],
            },
            pins: vec![
                PinnedPoint { label: "p0".into(), y: 0, z: 1 },
                PinnedPoint { label: "p1".into(), y: 1, z: 1 },
                PinnedPoint { label: "pinf".into(), y: 1, z: 0 },
            ],
        }
    }

    #[test]
    fn ansatz_2_12_5_validates_and_has_right_cycle_types() {
        let a = ansatz_2_12_5_model();
        assert!(a.validate().is_ok());
        let [c0, c1, cinf] = a.cycle_types();
        assert_eq!(c0, vec![2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(c1, vec![12, 12]);
        assert_eq!(cinf, vec![5, 5, 5, 5, 1, 1, 1, 1]);
    }

    #[test]
    fn rejects_bad_degree_sum() {
        let mut a = ansatz_2_12_5_model();
        a.zero_fiber.factors[1].degree = 7; // 16 + 7 = 23 != 24
        assert!(matches!(a.validate(), Err(AnsatzError::DegreeSumMismatch { .. })));
    }

    fn unk(s: &str) -> Coeff {
        Coeff::Unknown(s.into())
    }

    /// Degree-3 dessin φ = (3Y²Z − 2Y³)/Z³, branch (2,1)(2,1)(3);
    /// known solution b0=3, b1=−2, v0=1/2, c=−2 (ported from belyi_encode.rs).
    fn deg3_dessin() -> BelyiAnsatzSystem {
        BelyiAnsatzSystem {
            zero: vec![
                FactorSpec { mult: 2, coeffs: vec![Coeff::Fixed(0), Coeff::Fixed(1)] },
                FactorSpec { mult: 1, coeffs: vec![unk("b0"), unk("b1")] },
            ],
            inf: vec![FactorSpec { mult: 3, coeffs: vec![Coeff::Fixed(1), Coeff::Fixed(0)] }],
            one: vec![
                FactorSpec { mult: 2, coeffs: vec![Coeff::Fixed(-1), Coeff::Fixed(1)] },
                FactorSpec { mult: 1, coeffs: vec![unk("v0"), Coeff::Fixed(1)] },
            ],
            c: unk("c"),
        }
    }

    #[test]
    fn encodes_degree3_system_shape() {
        let enc = deg3_dessin().encode();
        assert_eq!(enc.degree, 3);
        assert_eq!(enc.var_names, vec!["b0", "b1", "v0", "c"]);
        assert_eq!(enc.system.num_equations(), 4);
        assert_eq!(enc.system.num_variables(), 4);
        // the known solution is an exact zero
        let sol = [
            Rational::new(3, 1).unwrap(),
            Rational::new(-2, 1).unwrap(),
            Rational::new(1, 2).unwrap(),
            Rational::new(-2, 1).unwrap(),
        ];
        assert!(enc.system.is_exact_solution(&sol));
    }

    #[test]
    fn encodes_degree24_2_12_5_system() {
        let enc = ansatz_2_12_5().encode();
        assert_eq!(enc.degree, 24);
        assert_eq!(enc.var_names.len(), 25); // square 25x25 system
        assert_eq!(enc.system.num_variables(), 25);
        assert_eq!(enc.system.num_equations(), 25);
    }

    // independent binary-form arithmetic (coeffs ascending in Y-degree)
    fn bmul(a: &[i64], b: &[i64]) -> Vec<i64> {
        let mut c = vec![0i64; a.len() + b.len() - 1];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                c[i + j] += ai * bj;
            }
        }
        c
    }
    fn bpow(a: &[i64], e: u32) -> Vec<i64> {
        let mut acc = vec![1i64];
        for _ in 0..e {
            acc = bmul(&acc, a);
        }
        acc
    }

    #[test]
    fn degree24_encoding_matches_independent_binary_forms() {
        // Round-trip: a random unknown assignment must make every encoder equation
        // equal the corresponding coefficient of P - Q - c*U^12 computed by hand.
        let enc = ansatz_2_12_5().encode();
        // encoder variable order: a2..a7 (6), b0..b7 (8), r0,r1,r2 (3),
        // s0..s4 (5), u0,u1 (2), c (1) = 25
        let vals: Vec<i64> = vec![
            2, -1, 3, 1, -2, 4, // a2..a7
            1, 2, -1, 3, 1, -1, 2, 1, // b0..b7
            -1, 2, 1, // r0..r2
            3, -2, 1, 2, -1, // s0..s4
            1, -3, // u0,u1
            2, // c
        ];
        assert_eq!(vals.len(), 25);

        let a = vec![0, 1, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], 1];
        let b = vec![vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], vals[12], vals[13], 1];
        let r = vec![vals[14], vals[15], vals[16], 1, 0];
        let s = vec![vals[17], vals[18], vals[19], vals[20], vals[21]];
        let uu = vec![vals[22], vals[23], 1];
        let c = vals[24];

        let p = bmul(&bpow(&a, 2), &b);
        let q = bmul(&bpow(&r, 5), &s);
        let cu: Vec<i64> = bpow(&uu, 12).iter().map(|x| x * c).collect();
        // d[k] = coeff of Y^k in P - Q - c*U^12
        let d: Vec<i64> = (0..25)
            .map(|k| {
                p.get(k).copied().unwrap_or(0)
                    - q.get(k).copied().unwrap_or(0)
                    - cu.get(k).copied().unwrap_or(0)
            })
            .collect();

        // encoder equations are keyed by (Y-exp, Z-exp) ascending in BTreeMap order:
        // Y-exp ascending 0..24 (Z-exp = 24 - Y-exp), so the k-th poly is coeff of Y^k.
        let assignment: Vec<Rational> = vals.iter().map(|&v| Rational::from_i64(v)).collect();
        let residual = enc.system.evaluate(&assignment);
        assert_eq!(residual.len(), 25);
        for (k, val) in residual.iter().enumerate() {
            assert_eq!(*val, Rational::from_i64(d[k]), "mismatch at Y-degree {k}");
        }
    }

    #[test]
    fn var_names_are_unique() {
        use std::collections::BTreeSet;
        let enc = ansatz_2_12_5().encode();
        let set: BTreeSet<&String> = enc.var_names.iter().collect();
        assert_eq!(set.len(), enc.var_names.len());
    }
}
