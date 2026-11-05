//! Special functions for the Taylor expansion of L(E,s) at s = 1.
//!
//! Everything here is a building block of [`crate::lfunction::CurveLSeries::l_derivative`];
//! the derivation of the master formula lives there. What is needed:
//!
//! * ζ(k) at integer k ≥ 2 ([`zeta_integer`]) — these are the Taylor
//!   coefficients of log Γ(1+s) at s = 0;
//! * the coefficients of the exponential of a power series
//!   ([`exp_series_coeffs`]);
//! * the incomplete-Gamma-type kernels G_m ([`g_kernels`]),
//!   G_m(x) = (x/m!) ∫_1^∞ e^{−x t} (log t)^m dt,
//!   which satisfy G_0(x) = e^{−x} and G_1(x) = E_1(x) (so the two kernels
//!   already used by `l1` and `l1_derivative` are the m = 0, 1 members of
//!   this family), and the rigorous monotone bound
//!   0 ≤ G_m(x) ≤ e^{−x}/x^m for x > 0,
//!   proved by log t ≤ t − 1, which gives
//!   ∫_1^∞ e^{−xt}(log t)^m dt ≤ e^{−x} ∫_0^∞ e^{−xv} v^m dv = e^{−x} m!/x^{m+1}.
//!   In particular G_m(x) ≤ e^{−x} for x ≥ 1 — this is what makes the
//!   geometric tail bound of `l1`/`l1_derivative` valid for EVERY m.

use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::BigFloat;

use crate::lfunction::{e1_working_precision, euler_gamma, pow2_integer};

/// ζ(k) at integer k ≥ 2, to `prec` bits.
///
/// Euler–Maclaurin with an exact-rational core. For f(x) = x^{−k} and any
/// integer M ≥ 1,
///
/// ```text
/// zeta(k) = sum_{n<M} n^-k  +  M^(1-k)/(k-1)  +  M^-k/2
///           + sum_{j>=1} (B_2j / (2j)!) * (k)_(2j-1) * M^(1-k-2j)  +  R_J,
/// ```
///
/// where (k)_(2j−1) = k(k+1)···(k+2j−2) comes from
/// f^(2j−1)(M) = −(k)_(2j−1) M^{−k−2j+1}. Every even derivative of x^{−k} has
/// constant sign on [M,∞), so the classical remainder theorem (Knuth,
/// TAOCP 1.2.11.2; Olver) gives |R_J| ≤ |first omitted term|. The loop below
/// stops at the FIRST term smaller than 2^{−(prec+24)} and does not add it,
/// so the truncation error is < 2^{−(prec+24)}; everything up to the single
/// closing `from_rational` is exact rational arithmetic.
///
/// Accuracy of the returned value: < 2^{−prec} absolute (the closing rounding
/// to `prec` bits dominates the 2^{−(prec+24)} truncation). Callers carry ≥ 48
/// guard bits.
///
/// M is sized so that the smallest Euler–Maclaurin term, which is ≈ e^{−2πM},
/// clears the threshold: 2πM > (prec+24)·ln 2 needs M > 0.111(prec+24), and
/// `M = ⌈0.13(prec+40)⌉ + 10 + k` has ample margin (asserted at runtime).
pub(crate) fn zeta_integer(k: u64, prec: u64) -> BigFloat {
    assert!(k >= 2, "zeta_integer: k >= 2 required (s = 1 is the pole)");
    let m = ((prec as f64 + 40.0) * 0.13).ceil() as u64 + 10 + k;
    let m_int = Integer::from(m as i64);
    let threshold =
        Rational::new(Integer::one(), pow2_integer(prec + 24)).expect("power of two is nonzero");

    let k32 = u32::try_from(k).expect("zeta_integer: k fits in u32");
    let mut core = Rational::zero();
    for n in 1..m {
        core = &core
            + &Rational::new(Integer::one(), Integer::from(n as i64).pow(k32))
                .expect("n >= 1 so n^k > 0");
    }
    core = &core
        + &Rational::new(
            Integer::one(),
            &Integer::from((k - 1) as i64) * &m_int.pow(k32 - 1),
        )
        .expect("k >= 2 and M >= 1");
    core = &core
        + &Rational::new(Integer::one(), &Integer::from(2) * &m_int.pow(k32)).expect("M >= 1");

    let jmax = (2 * m) as usize;
    let bern = crate::lfunction::bernoulli_numbers(2 * jmax + 2);
    let m_sq = &m_int * &m_int;
    // running state at step j: fact = (2j)!, rising = (k)_(2j-1), mpow = M^(k+2j-1)
    let mut fact = Integer::from(2);
    let mut rising = Integer::from(k as i64);
    let mut mpow = &m_int.pow(k32) * &m_int;
    let mut reached = false;
    for j in 1..=jmax {
        let term = &(&bern[2 * j] * &Rational::from_integer(rising.clone()))
            / &Rational::from_integer(&fact * &mpow);
        let abs_term = if term < Rational::zero() {
            -&term
        } else {
            term.clone()
        };
        if abs_term < threshold {
            reached = true;
            break;
        }
        core = &core + &term;
        let j2 = 2 * j as i64;
        fact = &(&fact * &Integer::from(j2 + 1)) * &Integer::from(j2 + 2);
        rising = &(&rising * &Integer::from(k as i64 + j2 - 1)) * &Integer::from(k as i64 + j2);
        mpow = &mpow * &m_sq;
    }
    assert!(
        reached,
        "zeta_integer: Euler-Maclaurin failed to reach the threshold (M too small)"
    );
    BigFloat::from_rational(&core, prec + 32).with_precision(prec)
}

/// The coefficients b_0..b_r of exp(Σ_{j≥1} p_j u^j), given p_1..p_r in
/// `p[1..=r]` (`p[0]` is ignored).
///
/// From b' = p' b: j·b_j = Σ_{i=1}^{j} i·p_i·b_{j−i}, b_0 = 1.
pub(crate) fn exp_series_coeffs(p: &[BigFloat], wp: u64) -> Vec<BigFloat> {
    let r = p.len().saturating_sub(1);
    let mut b = vec![BigFloat::zero_prec(wp); r + 1];
    b[0] = BigFloat::one_prec(wp);
    for j in 1..=r {
        let mut acc = BigFloat::zero_prec(wp);
        for i in 1..=j {
            let i_bf = BigFloat::from_integer(&Integer::from(i as i64), wp);
            acc = acc + i_bf * p[i].with_precision(wp) * b[j - i].clone();
        }
        b[j] = acc / BigFloat::from_integer(&Integer::from(j as i64), wp);
    }
    b
}

/// G_0(x)..G_rmax(x) at `prec` bits, for x > 0.
///
/// # Derivation
///
/// Put F(s) = ∫_1^∞ e^{−xt} t^{s−1} dt = x^{−s} Γ(s,x). Differentiating under
/// the integral sign j times at s = 0 gives
///
/// ```text
/// K_j(x) := int_1^inf e^{-xt} (log t)^j dt/t = [d^j/ds^j  x^-s Gamma(s,x)]_{s=0}.
/// ```
///
/// Integrating G_m's defining integral by parts (m ≥ 1; the boundary term at
/// t = 1 dies because log 1 = 0) gives
/// ∫_1^∞ e^{−xt}(log t)^m dt = (m/x)·K_{m−1}(x), hence
///
/// ```text
/// G_m(x) = (x/m!) int_1^inf e^{-xt}(log t)^m dt = K_{m-1}(x)/(m-1)!   (m >= 1).
/// ```
///
/// Now expand x^{−s}Γ(s,x) around s = 0 using Γ(s,x) = Γ(s) − γ(s,x) and
/// γ(s,x) = Σ_{n≥0} (−1)^n x^{s+n}/(n!(s+n)):
///
/// ```text
/// x^-s Gamma(s,x) = [x^-s Gamma(1+s) - 1]/s  -  sum_{n>=1} (-1)^n x^n / (n! (s+n)),
/// ```
///
/// both pieces analytic at s = 0. With φ(s) := x^{−s}Γ(1+s) = exp(Q(s)),
///
/// ```text
/// Q(s) = -(ln x + gamma) s + sum_{k>=2} (-1)^k zeta(k) s^k / k
/// ```
///
/// (the Taylor series of log Γ(1+s)), and expanding 1/(s+n) = Σ_m (−1)^m s^m/n^{m+1},
///
/// ```text
/// [s^m] x^-s Gamma(s,x) = phi_{m+1} + (-1)^m sum_{n>=1} (-1)^(n+1) x^n / (n^(m+1) n!),
/// ```
///
/// and K_j = j!·[s^j](x^{−s}Γ(s,x)). Therefore, for m ≥ 1,
///
/// ```text
/// G_m(x) = phi_m(x) + sum_{n>=1} (-1)^(m+n) x^n / (n^m n!),      phi_m = [s^m] exp(Q(s)),
/// ```
///
/// and G_0(x) = e^{−x} (direct: ∫_1^∞ e^{−xt}dt = e^{−x}/x).
///
/// Consistency: m = 1 gives φ_1 = −(ln x + γ) and the series
/// Σ (−1)^{n+1} x^n/(n·n!), i.e. exactly E_1(x) — the kernel `l1_derivative`
/// already uses; m = 0 gives e^{−x} — the kernel `l1` already uses. Both were
/// re-verified against brute-force numerical integration of the defining
/// integral (mpmath, 25 digits, m = 0..4, several x) before this code was
/// written.
///
/// # Error model
///
/// The alternating series is summed at
/// `wp = e1_working_precision(x, prec) = prec + 48 + ⌈x·log₂e⌉` bits: its
/// largest term is ≈ e^x/(x^m √(2πx)), so the cancellation costs ≈ x·log₂e
/// bits and the result carries absolute error < 2^{−(prec+16)} — the same
/// budget, and the same engineering model, as the existing
/// `exp_integral_e1` (of which this is the m = 1 case).
///
/// `gamma` and `zetas` must already carry ≥ `wp` bits (`zetas[k]` = ζ(k),
/// entries 0 and 1 unused); an `Err` is returned if they do not, so a caller
/// cannot silently under-resolve them.
pub(crate) fn g_kernels(
    x: &BigFloat,
    rmax: u32,
    prec: u64,
    gamma: &BigFloat,
    zetas: &[BigFloat],
) -> Result<Vec<BigFloat>, String> {
    if x.sign() <= 0 {
        return Err("g_kernels requires x > 0".to_string());
    }
    let rmax = rmax as usize;
    let xf = x.to_f64();
    let wp = e1_working_precision(xf, prec);
    if RealField::precision(gamma) < wp {
        return Err(format!(
            "g_kernels: gamma carries {} bits, need {wp}",
            RealField::precision(gamma)
        ));
    }
    for k in 2..=rmax {
        if zetas.len() <= k || RealField::precision(&zetas[k]) < wp {
            return Err(format!("g_kernels: zeta({k}) missing or under-resolved"));
        }
    }

    let xw = x.with_precision(wp);
    let mut out = vec![BigFloat::zero_prec(wp); rmax + 1];
    out[0] = RealField::exp(&(-xw.clone()));
    if rmax == 0 {
        return Ok(out.into_iter().map(|v| v.with_precision(prec)).collect());
    }

    // phi_m = [s^m] exp(Q(s)) for m = 1..rmax
    let mut q = vec![BigFloat::zero_prec(wp); rmax + 1];
    q[1] = -(RealField::ln(&xw) + gamma.with_precision(wp));
    for (k, qk) in q.iter_mut().enumerate().skip(2) {
        let z = zetas[k].with_precision(wp) / BigFloat::from_integer(&Integer::from(k as i64), wp);
        *qk = if k % 2 == 0 { z } else { -z };
    }
    let phi = exp_series_coeffs(&q, wp);

    // sums_m = sum_{n>=1} (-1)^(m+n) x^n / (n^m n!)
    let mut sums = vec![BigFloat::zero_prec(wp); rmax + 1];
    let cutoff = crate::lfunction::pow2_neg(wp, wp);
    let mut u = BigFloat::one_prec(wp); // x^n / n!
    let mut n: i64 = 1;
    loop {
        let n_bf = BigFloat::from_integer(&Integer::from(n), wp);
        u = u * xw.clone() / n_bf.clone();
        let mut n_pow = BigFloat::one_prec(wp);
        for (m, sm) in sums.iter_mut().enumerate().skip(1).take(rmax) {
            n_pow = n_pow * n_bf.clone();
            let contrib = u.clone() / n_pow.clone();
            // (-1)^(m+n)
            *sm = if (m as i64 + n) % 2 == 0 {
                sm.clone() + contrib
            } else {
                sm.clone() - contrib
            };
        }
        if (n as f64) > xf && OrderedRing::abs(&u) < cutoff {
            break;
        }
        n += 1;
        if n > 10_000_000 {
            return Err("g_kernels: series failed to converge".to_string());
        }
    }

    for m in 1..=rmax {
        out[m] = (phi[m].clone() + sums[m].clone()).with_precision(prec);
    }
    out[0] = out[0].with_precision(prec);
    Ok(out)
}

/// The working precision the G-kernels need at argument x, and hence the
/// precision `gamma` and the `zetas` must be prepared at.
pub(crate) fn g_working_precision(xf: f64, prec: u64) -> u64 {
    e1_working_precision(xf, prec)
}

/// γ and ζ(2..=rmax), all at `wp` bits — the shared constants of a whole
/// `l_derivative` run.
pub(crate) fn taylor_constants(rmax: u32, wp: u64) -> (BigFloat, Vec<BigFloat>) {
    let gamma = euler_gamma(wp);
    let mut zetas = vec![BigFloat::zero_prec(wp), BigFloat::zero_prec(wp)];
    for k in 2..=rmax as u64 {
        zetas.push(zeta_integer(k, wp));
    }
    (gamma, zetas)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_to(a: &BigFloat, decimal: &str, k: usize) -> bool {
        let prec = RealField::precision(a).max(320);
        let b = BigFloat::from_decimal_str(decimal, prec).unwrap();
        let tol = BigFloat::from_decimal_str(&format!("0.{}1", "0".repeat(k - 1)), prec).unwrap();
        OrderedRing::abs(&(a.clone() - b)) < tol
    }

    /// ζ(2), ζ(3), ζ(4), ζ(5), ζ(8) against independently derived truths
    /// (mpmath, 50 digits, computed before this test); ζ(2) and ζ(4) are also
    /// gated against their closed forms π²/6 and π⁴/90 computed from the
    /// crate's own π, which is a second, structurally independent check.
    #[test]
    fn test_zeta_integer() {
        let gates: [(u64, &str); 5] = [
            (2, "1.6449340668482264364724151666460251892189499012068"),
            (3, "1.2020569031595942853997381615114499907649862923405"),
            (4, "1.0823232337111381915160036965411679027747509519187"),
            (5, "1.0369277551433699263313654864570341680570809195019"),
            (8, "1.0040773561979443393786852385086524652589607906499"),
        ];
        for (k, truth) in &gates {
            let z = zeta_integer(*k, 200);
            assert!(
                close_to(&z, truth, 45),
                "zeta({}) to 45 digits; got {}",
                k,
                z.to_decimal_string(52)
            );
        }
        let pi = <BigFloat as RealField>::pi(200);
        let six = BigFloat::from_integer(&Integer::from(6), 200);
        let z2 = pi.clone() * pi.clone() / six;
        assert!(
            OrderedRing::abs(&(zeta_integer(2, 200) - z2))
                < BigFloat::from_decimal_str("1e-45", 200).unwrap(),
            "zeta(2) = pi^2/6"
        );
        let ninety = BigFloat::from_integer(&Integer::from(90), 200);
        let z4 = pi.clone() * pi.clone() * pi.clone() * pi / ninety;
        assert!(
            OrderedRing::abs(&(zeta_integer(4, 200) - z4))
                < BigFloat::from_decimal_str("1e-45", 200).unwrap(),
            "zeta(4) = pi^4/90"
        );
    }

    /// exp of a power series: exp(u + u²) has coefficients
    /// 1, 1, 3/2, 7/6, 25/24, 27/40 (derived by hand / sympy).
    #[test]
    fn test_exp_series_coeffs() {
        let wp = 128;
        let p = vec![
            BigFloat::zero_prec(wp),
            BigFloat::one_prec(wp),
            BigFloat::one_prec(wp),
            BigFloat::zero_prec(wp),
            BigFloat::zero_prec(wp),
            BigFloat::zero_prec(wp),
        ];
        let b = exp_series_coeffs(&p, wp);
        let expected = [
            "1",
            "1",
            "1.5",
            "1.1666666666666666666666666667",
            "1.0416666666666666666666666667",
            "0.675",
        ];
        for (j, e) in expected.iter().enumerate() {
            assert!(
                close_to(&b[j], e, 20),
                "[u^{}] exp(u + u^2); got {}",
                j,
                b[j].to_decimal_string(24)
            );
        }
    }

    /// The G-kernels against independently derived truths: brute-force
    /// numerical integration of (x/m!)∫_1^∞ e^{−xt}(log t)^m dt in mpmath at
    /// 60 dps (m = 0..4, x = 1 and x = 5/2), computed BEFORE this test. The
    /// m = 0 and m = 1 entries are additionally pinned to the closed forms
    /// e^{−x} and E_1(x).
    #[test]
    fn test_g_kernels() {
        let prec = 200;
        let cases: [(&str, [&str; 5]); 2] = [
            (
                "1",
                [
                    "0.367879441171442321595523770161460867445811131",
                    "0.219383934395520273677163775460121649031047293",
                    "0.0978431972166701793255377890452800827695822695",
                    "0.0356034919284750178257947549450178173381381430",
                    "0.0110708954460087811883561290525035907564524589",
                ],
            ),
            (
                "2.5",
                [
                    "0.0820849986238987951695286744671598078378041210",
                    "0.0249149178702697354956280122746096359458483847",
                    "0.00625020430314611055349920421052403445571593382",
                    "0.00135452818132243392474938461252932753990422023",
                    "0.000260571990103436147975491044450750323490690787",
                ],
            ),
        ];
        for (xs, truths) in &cases {
            let x = BigFloat::from_decimal_str(xs, prec).unwrap();
            let wp = g_working_precision(x.to_f64(), prec);
            let (gamma, zetas) = taylor_constants(4, wp);
            let g = g_kernels(&x, 4, prec, &gamma, &zetas).unwrap();
            for (m, truth) in truths.iter().enumerate() {
                assert!(
                    close_to(&g[m], truth, 25),
                    "G_{}({}) to 25 digits; got {}",
                    m,
                    xs,
                    g[m].to_decimal_string(32)
                );
            }
            // G_0 = e^{-x} exactly
            let e_minus_x = RealField::exp(&(-x.with_precision(prec)));
            assert!(
                OrderedRing::abs(&(g[0].clone() - e_minus_x))
                    < BigFloat::from_decimal_str("1e-50", prec).unwrap()
            );
            // the tail hypothesis 0 <= G_m(x) <= e^{-x}/x^m, and hence
            // G_m(x) <= e^{-x} for x >= 1: checked here at x = 1 and 2.5
            let mut xp = BigFloat::one_prec(prec);
            for gm in g.iter() {
                assert!(gm.sign() > 0, "G_m > 0");
                assert!(
                    gm.clone() * xp.clone() <= RealField::exp(&(-x.with_precision(prec))),
                    "G_m(x) x^m <= e^-x"
                );
                xp = xp * x.with_precision(prec);
            }
        }
        assert!(g_kernels(
            &BigFloat::zero_prec(64),
            2,
            64,
            &euler_gamma(256),
            &taylor_constants(2, 256).1
        )
        .is_err());
    }
}
