//! High-precision (rug/MPFR) port of the KMSV §2 embedding Δ(a,b,c) ↪ PSL₂(ℝ).
//!
//! Mirrors [`super::triangle_group`] but with `rug::Float` matrix entries at a chosen
//! bit precision, so the whole §4 power-series pipeline can reach the ~10⁻³⁰ accuracy
//! the method needs (f64's ρ^{-N} dynamic range caps it at ~10⁻³). Entries are real
//! (Δ ⊂ PSL₂(ℝ)); `apply` acts on `rug::Complex` points of ℍ.

use rug::float::Constant;
use rug::{Complex, Float};

/// A real Möbius transformation with `rug::Float` entries at precision `prec` bits.
#[derive(Clone, Debug)]
pub struct MobiusHp {
    pub a: Float,
    pub b: Float,
    pub c: Float,
    pub d: Float,
    pub prec: u32,
}

impl MobiusHp {
    pub fn new(a: Float, b: Float, c: Float, d: Float, prec: u32) -> Self {
        MobiusHp { a, b, c, d, prec }
    }

    pub fn identity(prec: u32) -> Self {
        MobiusHp {
            a: Float::with_val(prec, 1.0),
            b: Float::with_val(prec, 0.0),
            c: Float::with_val(prec, 0.0),
            d: Float::with_val(prec, 1.0),
            prec,
        }
    }

    /// Matrix product self·o.
    pub fn mul(&self, o: &MobiusHp) -> MobiusHp {
        let p = self.prec;
        MobiusHp {
            a: Float::with_val(p, &self.a * &o.a) + Float::with_val(p, &self.b * &o.c),
            b: Float::with_val(p, &self.a * &o.b) + Float::with_val(p, &self.b * &o.d),
            c: Float::with_val(p, &self.c * &o.a) + Float::with_val(p, &self.d * &o.c),
            d: Float::with_val(p, &self.c * &o.b) + Float::with_val(p, &self.d * &o.d),
            prec: p,
        }
    }

    /// Inverse (adjugate over determinant).
    pub fn inverse(&self) -> MobiusHp {
        let p = self.prec;
        let det = Float::with_val(p, &self.a * &self.d) - Float::with_val(p, &self.b * &self.c);
        MobiusHp {
            a: Float::with_val(p, &self.d / &det),
            b: -Float::with_val(p, &self.b / &det),
            c: -Float::with_val(p, &self.c / &det),
            d: Float::with_val(p, &self.a / &det),
            prec: p,
        }
    }

    /// z ↦ (az + b)/(cz + d).
    pub fn apply(&self, z: &Complex) -> Complex {
        let p = self.prec;
        let num = Complex::with_val(p, z * &self.a) + Complex::with_val(p, (&self.b, 0.0));
        let den = Complex::with_val(p, z * &self.c) + Complex::with_val(p, (&self.d, 0.0));
        Complex::with_val(p, &num / &den)
    }

    pub fn pow_u(&self, n: u32) -> MobiusHp {
        let mut r = MobiusHp::identity(self.prec);
        for _ in 0..n {
            r = r.mul(self);
        }
        r
    }

    pub fn pow_signed(&self, n: i32) -> MobiusHp {
        if n >= 0 {
            self.pow_u(n as u32)
        } else {
            self.inverse().pow_u((-n) as u32)
        }
    }
}

/// The high-precision embedding data of Δ(a,b,c).
#[derive(Clone, Debug)]
pub struct TriangleGroupHp {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub prec: u32,
    pub mu: Float,
    pub z_a: Complex,
    pub z_b: Complex,
    pub z_c: Complex,
    pub delta_a: MobiusHp,
    pub delta_b: MobiusHp,
    pub delta_c: MobiusHp,
}

impl TriangleGroupHp {
    pub fn new(a: u32, b: u32, c: u32, prec: u32) -> Self {
        let f = |v: f64| Float::with_val(prec, v);
        let pi = Float::with_val(prec, Constant::Pi);
        let pa = Float::with_val(prec, &pi / f(a as f64));
        let pb = Float::with_val(prec, &pi / f(b as f64));
        let pc = Float::with_val(prec, &pi / f(c as f64));
        let (cos_a, cos_b, cos_c) = (pa.clone().cos(), pb.clone().cos(), pc.clone().cos());
        let (sin_a, sin_b) = (pa.clone().sin(), pb.clone().sin());

        // Λ = (cos π/a cos π/b + cos π/c)/(sin π/a sin π/b);  μ = Λ + √(Λ²−1).
        let num = Float::with_val(prec, &cos_a * &cos_b) + &cos_c;
        let den = Float::with_val(prec, &sin_a * &sin_b);
        let lambda = Float::with_val(prec, &num / &den);
        let lam2m1 = Float::with_val(prec, &lambda * &lambda) - f(1.0);
        let mu = Float::with_val(prec, &lambda + lam2m1.sqrt());

        let z_a = Complex::with_val(prec, (0.0, 1.0));
        let z_b = Complex::with_val(prec, (0.0, &mu)); // μ i

        // z_c per (2.4): re = (μ²−1)/denom, denom = 2(cot π/a + μ cot π/b);
        // im = √(1/sin²π/a − (re − cot π/a)²).
        let cot_a = Float::with_val(prec, &cos_a / &sin_a);
        let cot_b = Float::with_val(prec, &cos_b / &sin_b);
        let denom_c = f(2.0) * (Float::with_val(prec, &cot_a + Float::with_val(prec, &mu * &cot_b)));
        let mu2m1 = Float::with_val(prec, &mu * &mu) - f(1.0);
        let re_c = Float::with_val(prec, &mu2m1 / &denom_c);
        let inv_sin2 = f(1.0) / Float::with_val(prec, &sin_a * &sin_a);
        let shift = Float::with_val(prec, &re_c - &cot_a);
        let inside = Float::with_val(prec, &inv_sin2 - Float::with_val(prec, &shift * &shift));
        let z_c = Complex::with_val(prec, (&re_c, inside.sqrt()));

        // δ_a = [[cos π/a, sin π/a],[−sin π/a, cos π/a]]  (rotation about z_a = i).
        let delta_a = MobiusHp::new(
            cos_a.clone(),
            sin_a.clone(),
            Float::with_val(prec, -(&sin_a)),
            cos_a.clone(),
            prec,
        );
        // δ_b = diag(√μ,1/√μ)·R(π/b)·diag(1/√μ,√μ) = [[cos π/b, μ sin π/b],[−sin π/b/μ, cos π/b]].
        let delta_b = MobiusHp::new(
            cos_b.clone(),
            Float::with_val(prec, &mu * &sin_b),
            -Float::with_val(prec, &sin_b / &mu),
            cos_b.clone(),
            prec,
        );
        // δ_c = (δ_b δ_a)⁻¹.
        let delta_c = delta_b.mul(&delta_a).inverse();

        TriangleGroupHp {
            a,
            b,
            c,
            prec,
            mu,
            z_a,
            z_b,
            z_c,
            delta_a,
            delta_b,
            delta_c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREC: u32 = 256;

    // A high-precision Möbius power that should be ±I: off-diagonal ≈ 0 to ~10⁻⁷⁰.
    fn assert_scalar(m: &MobiusHp, tag: &str) {
        let b = m.b.clone().abs().to_f64();
        let c = m.c.clone().abs().to_f64();
        assert!(b < 1e-70 && c < 1e-70, "{tag}: off-diagonal not ~0 (b={b:.2e}, c={c:.2e})");
        // diagonal is ±1
        let a = m.a.clone().to_f64();
        let d = m.d.clone().to_f64();
        assert!((a.abs() - 1.0).abs() < 1e-12 && (d.abs() - 1.0).abs() < 1e-12, "{tag}: diag not ±1");
    }

    fn check(a: u32, b: u32, c: u32) {
        let tg = TriangleGroupHp::new(a, b, c, PREC);
        assert_scalar(&tg.delta_a.pow_u(a), "δ_a^a");
        assert_scalar(&tg.delta_b.pow_u(b), "δ_b^b");
        assert_scalar(&tg.delta_c.pow_u(c), "δ_c^c");
        assert_scalar(&tg.delta_c.mul(&tg.delta_b).mul(&tg.delta_a), "δ_c δ_b δ_a");
        // fixed points: δ_a fixes z_a, δ_b fixes z_b.
        let fa = tg.delta_a.apply(&tg.z_a);
        let db = (Complex::with_val(PREC, &fa - &tg.z_a)).abs().real().to_f64();
        assert!(db < 1e-70, "δ_a should fix z_a (residual {db:.2e})");
        let fb = tg.delta_b.apply(&tg.z_b);
        let dbb = (Complex::with_val(PREC, &fb - &tg.z_b)).abs().real().to_f64();
        assert!(dbb < 1e-70, "δ_b should fix z_b (residual {dbb:.2e})");
    }

    #[test]
    fn hp_embedding_2_12_5() {
        check(2, 12, 5);
        // μ agrees with the f64 embedding to f64 accuracy.
        let tg = TriangleGroupHp::new(2, 12, 5, PREC);
        let tg64 = super::super::triangle_group::TriangleGroup::new(2, 12, 5);
        assert!((tg.mu.to_f64() - tg64.mu).abs() < 1e-12, "μ mismatch vs f64");
    }

    #[test]
    fn hp_embedding_5_3_3() {
        check(5, 3, 3);
    }

    #[test]
    fn hp_embedding_2_3_7() {
        check(2, 3, 7);
    }
}
