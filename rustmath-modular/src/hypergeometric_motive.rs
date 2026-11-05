//! Hypergeometric motives
//!
//! NOT IMPLEMENTED beyond the data of the parameter multisets.  Every arithmetic
//! quantity below (conductor, Euler factors, wild ramification) used to be a
//! placeholder returning a plausible constant -- 1, `vec![1.0]`, `false` -- and
//! now refuses instead.  See the individual methods for what each would need.

/// A hypergeometric motive H(α, β) where α, β are multisets of rationals
#[derive(Clone, Debug)]
pub struct HypergeometricMotive {
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

impl HypergeometricMotive {
    pub fn new(alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        Self { alpha, beta }
    }

    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// The conductor of the motive.
    ///
    /// NOT IMPLEMENTED.  The conductor of H(alpha, beta) is a product of local
    /// conductors: the tame primes contribute through the Hodge/zigzag data of
    /// the two multisets, and the wild primes (those dividing the denominators of
    /// alpha or beta) contribute exponents that need the local monodromy at p.
    /// None of that is computed here.
    ///
    /// This used to return a hard-coded 1, i.e. "conductor 1" for every motive.
    pub fn conductor(&self) -> u64 {
        unimplemented!(
            "HypergeometricMotive::conductor: not implemented. It needs the local \
             conductor exponents at the tame primes (from the Hodge data of alpha and \
             beta) and at the wild primes (from the local monodromy at p); none of that \
             is computed here. Previously returned a hard-coded 1."
        )
    }

    /// The Euler factor at the prime p.
    ///
    /// NOT IMPLEMENTED.  This needs the hypergeometric trace formula: the Gauss
    /// sums / Jacobi sums over F_q attached to (alpha, beta) via Greene's finite
    /// hypergeometric function, assembled into the local L-polynomial. Nothing of
    /// the kind is computed here.
    ///
    /// This used to return `vec![1.0]`, i.e. a trivial Euler factor at every
    /// prime.
    pub fn euler_factor(&self, p: u64) -> Vec<f64> {
        unimplemented!(
            "HypergeometricMotive::euler_factor({p}): not implemented. It needs the \
             hypergeometric trace formula over F_p (Gauss/Jacobi sums for the finite \
             hypergeometric function attached to alpha and beta), which is not computed \
             here. Previously returned vec![1.0], a trivial Euler factor at every prime."
        )
    }

    /// Whether the motive is wildly ramified at p.
    ///
    /// NOT IMPLEMENTED.  Wild ramification at p is governed by whether p divides
    /// the denominators of the parameters in alpha and beta and by the resulting
    /// local monodromy; this is not determined here.
    ///
    /// This used to return `false` unconditionally, i.e. "never wildly ramified".
    pub fn is_wildly_ramified(&self, p: u64) -> bool {
        unimplemented!(
            "HypergeometricMotive::is_wildly_ramified({p}): not implemented. It depends \
             on p dividing the denominators of alpha/beta and on the local monodromy at \
             p, neither of which is computed here. Previously returned false \
             unconditionally."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn motive() -> HypergeometricMotive {
        HypergeometricMotive::new(vec![0.5, 0.5], vec![0.0, 0.0])
    }

    #[test]
    fn test_parameters_are_kept() {
        let h = motive();
        assert_eq!(h.alpha(), &[0.5, 0.5]);
        assert_eq!(h.beta(), &[0.0, 0.0]);
    }

    /// Each arithmetic quantity is refused, not faked: the conductor was a
    /// hard-coded 1, the Euler factor a trivial vec![1.0], and the wild
    /// ramification a flat false.
    #[test]
    #[should_panic(expected = "HypergeometricMotive::conductor")]
    fn test_conductor_is_refused_not_faked() {
        let _ = motive().conductor();
    }

    #[test]
    #[should_panic(expected = "HypergeometricMotive::euler_factor")]
    fn test_euler_factor_is_refused_not_faked() {
        let _ = motive().euler_factor(7);
    }

    #[test]
    #[should_panic(expected = "HypergeometricMotive::is_wildly_ramified")]
    fn test_wild_ramification_is_refused_not_faked() {
        let _ = motive().is_wildly_ramified(2);
    }
}
