//! The factorized Belyi residual for the `[2,12,5]` passport — the object Newton
//! refines once a packing (or any seed) gives approximate preimages.
//!
//! Rather than solving for the 25 coefficients directly (Bézout ≈ 8.2e18), we
//! parametrize by the **roots** of the ramification factors and enforce the Belyi
//! identity as a polynomial residual:
//!
//! ```text
//!   A(x)² B(x) − λ R(x)⁵ S(x) − c U(x)¹² = 0
//! ```
//!
//! with `A` = 8 double zeros over 0 (`2⁸`), `B` = 8 simple zeros over 0 (`1⁸`),
//! `R` = 4 order-5 poles (`5⁴`), `S` = 4 simple poles (`1⁴`), `U` = 2 white points
//! over 1 (`12²`), and scalars `λ, c`. Each factor is degree ≤ 24, and the three
//! terms are each degree 24, so the residual is a degree-≤24 polynomial that must
//! vanish identically (25 complex coefficients).
//!
//! Adapted from the P2 review note (`inverse_galois/P2_note.md`) to RustMath's
//! `num_complex` usage.

use num_complex::Complex64;

/// A univariate complex polynomial, low-to-high coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct PolyC {
    pub c: Vec<Complex64>,
}

fn trim_near_zero(mut c: Vec<Complex64>) -> Vec<Complex64> {
    while c.len() > 1 && c.last().unwrap().norm() < 1e-30 {
        c.pop();
    }
    c
}

impl PolyC {
    pub fn zero() -> Self {
        Self {
            c: vec![Complex64::new(0.0, 0.0)],
        }
    }

    pub fn one() -> Self {
        Self {
            c: vec![Complex64::new(1.0, 0.0)],
        }
    }

    /// The monic polynomial `∏ (x − r)`.
    pub fn from_roots_monic(roots: &[Complex64]) -> Self {
        let mut p = PolyC::one();
        for &r in roots {
            p = p.mul(&PolyC {
                c: vec![-r, Complex64::new(1.0, 0.0)],
            });
        }
        p
    }

    pub fn degree(&self) -> usize {
        self.c.len().saturating_sub(1)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let mut out = vec![Complex64::new(0.0, 0.0); self.c.len() + rhs.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            for (j, b) in rhs.c.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        PolyC {
            c: trim_near_zero(out),
        }
    }

    /// `self + scale · rhs`.
    pub fn add_scaled(&self, scale: Complex64, rhs: &Self) -> Self {
        let n = self.c.len().max(rhs.c.len());
        let mut out = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            if i < self.c.len() {
                out[i] += self.c[i];
            }
            if i < rhs.c.len() {
                out[i] += scale * rhs.c[i];
            }
        }
        PolyC {
            c: trim_near_zero(out),
        }
    }

    pub fn pow(&self, n: usize) -> Self {
        assert!(n >= 1);
        let mut out = self.clone();
        for _ in 1..n {
            out = out.mul(self);
        }
        out
    }

    /// Coefficients padded (or truncated) to length `n`.
    pub fn pad_to(&self, n: usize) -> Vec<Complex64> {
        let mut out = self.c.clone();
        out.resize(n, Complex64::new(0.0, 0.0));
        out
    }
}

/// The `[2,12,5]` factors as root lists plus the two scalars.
#[derive(Debug, Clone)]
pub struct FactorizedRoots {
    pub roots_a: Vec<Complex64>, // 8 double zeros (2^8)
    pub roots_b: Vec<Complex64>, // 8 simple zeros / leaves (1^8)
    pub roots_r: Vec<Complex64>, // 4 order-5 poles (5^4)
    pub roots_s: Vec<Complex64>, // 4 simple poles (1^4)
    pub roots_u: Vec<Complex64>, // 2 white points (12^2)
    pub lambda: Complex64,
    pub c: Complex64,
}

impl FactorizedRoots {
    pub fn to_polys(&self) -> FactorizedPolys {
        FactorizedPolys {
            a: PolyC::from_roots_monic(&self.roots_a),
            b: PolyC::from_roots_monic(&self.roots_b),
            r: PolyC::from_roots_monic(&self.roots_r),
            s: PolyC::from_roots_monic(&self.roots_s),
            u: PolyC::from_roots_monic(&self.roots_u),
            lambda: self.lambda,
            c: self.c,
        }
    }

    pub fn residual_coefficients(&self) -> Vec<Complex64> {
        self.to_polys().residual_coefficients()
    }

    /// The residual as an interleaved real vector `[re, im, re, im, …]` — the form
    /// a real Levenberg–Marquardt / Newton step consumes.
    pub fn residual_real_vector(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(50);
        for z in self.residual_coefficients() {
            out.push(z.re);
            out.push(z.im);
        }
        out
    }

    pub fn residual_norm(&self) -> f64 {
        self.residual_coefficients()
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}

/// The `[2,12,5]` factors as polynomials.
#[derive(Debug, Clone)]
pub struct FactorizedPolys {
    pub a: PolyC,
    pub b: PolyC,
    pub r: PolyC,
    pub s: PolyC,
    pub u: PolyC,
    pub lambda: Complex64,
    pub c: Complex64,
}

impl FactorizedPolys {
    /// `A²B`, the zero-divisor part (degree 24).
    pub fn p_zero(&self) -> PolyC {
        self.a.pow(2).mul(&self.b)
    }

    /// `R⁵S`, the pole part (degree 24).
    pub fn p_inf(&self) -> PolyC {
        self.r.pow(5).mul(&self.s)
    }

    /// `U¹²`, the one-part (degree 24).
    pub fn p_one(&self) -> PolyC {
        self.u.pow(12)
    }

    /// `A²B − λ R⁵S − c U¹²`.
    pub fn residual_poly(&self) -> PolyC {
        self.p_zero()
            .add_scaled(-self.lambda, &self.p_inf())
            .add_scaled(-self.c, &self.p_one())
    }

    /// The residual coefficients padded to 25 (degree ≤ 24).
    pub fn residual_coefficients(&self) -> Vec<Complex64> {
        self.residual_poly().pad_to(25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn from_roots_gives_expected_coefficients() {
        // (x-1)(x-2) = x² - 3x + 2.
        let p = PolyC::from_roots_monic(&[c(1.0, 0.0), c(2.0, 0.0)]);
        assert_eq!(p.c.len(), 3);
        assert!((p.c[0] - c(2.0, 0.0)).norm() < 1e-12);
        assert!((p.c[1] - c(-3.0, 0.0)).norm() < 1e-12);
        assert!((p.c[2] - c(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn pow_squares_a_binomial() {
        // (x-1)² = x² - 2x + 1.
        let p = PolyC::from_roots_monic(&[c(1.0, 0.0)]).pow(2);
        assert!((p.c[0] - c(1.0, 0.0)).norm() < 1e-12);
        assert!((p.c[1] - c(-2.0, 0.0)).norm() < 1e-12);
        assert!((p.c[2] - c(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn constant_residual_is_one_minus_lambda_minus_c() {
        // All factors empty ⇒ each term is the constant 1 ⇒ residual = 1 − λ − c.
        let polys = FactorizedPolys {
            a: PolyC::one(),
            b: PolyC::one(),
            r: PolyC::one(),
            s: PolyC::one(),
            u: PolyC::one(),
            lambda: c(0.3, 0.0),
            c: c(0.5, 0.0),
        };
        let res = polys.residual_poly();
        assert_eq!(res.degree(), 0);
        assert!((res.c[0] - c(0.2, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn residual_terms_have_degree_24() {
        // Passport degrees: A²B, R⁵S, U¹² are all degree 24.
        let roots = FactorizedRoots {
            roots_a: (0..8).map(|k| c(k as f64, 0.0)).collect(),
            roots_b: (0..8).map(|k| c(0.0, k as f64 + 1.0)).collect(),
            roots_r: (0..4).map(|k| c(-(k as f64) - 1.0, 0.0)).collect(),
            roots_s: (0..4).map(|k| c(0.0, -(k as f64) - 1.0)).collect(),
            roots_u: vec![c(0.5, 0.5), c(-0.5, -0.5)],
            lambda: c(1.0, 0.0),
            c: c(1.0, 0.0),
        };
        let polys = roots.to_polys();
        assert_eq!(polys.p_zero().degree(), 24);
        assert_eq!(polys.p_inf().degree(), 24);
        assert_eq!(polys.p_one().degree(), 24);
        assert_eq!(roots.residual_coefficients().len(), 25);
        // A generic (non-Belyi) configuration has nonzero residual.
        assert!(roots.residual_norm() > 1e-6);
    }
}
