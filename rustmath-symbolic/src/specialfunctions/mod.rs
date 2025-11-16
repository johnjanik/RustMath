//! Special Functions Module
//!
//! This module provides implementations of various special functions
//! used in mathematics, physics, and engineering.
//!
//! Corresponds to sage.functions.*
//!
//! # Submodules
//!
//! - `generalized`: Generalized functions (Dirac delta, Heaviside, etc.)
//! - `min_max`: Symbolic minimum and maximum functions
//! - `airy`: Airy functions and their derivatives
//! - `other`: Various mathematical utility functions (abs, ceil, floor, factorial, etc.)
//! - `prime_pi`: Prime counting function and related functions
//! - `hypergeometric`: Hypergeometric functions
//! - `orthogonal_polys`: Orthogonal polynomials (Chebyshev, Legendre, Hermite, etc.)
//! - `jacobi`: Jacobi elliptic functions
//! - `special`: Elliptic integrals and spherical harmonics
//! - `transcendental`: Zeta functions and related transcendental functions
//! - `wigner`: Wigner symbols and Clebsch-Gordan coefficients
//!
//! # Examples
//!
//! ```
//! use rustmath_symbolic::specialfunctions::generalized::*;
//! use rustmath_symbolic::specialfunctions::min_max::*;
//! use rustmath_symbolic::specialfunctions::other::*;
//! use rustmath_symbolic::Expr;
//!
//! let h = heaviside(&Expr::from(5));
//! let m = min_symbolic(&[Expr::from(1), Expr::from(2)]);
//! let f = factorial(&Expr::from(5));
//! ```

pub mod airy;
pub mod generalized;
pub mod hypergeometric;
pub mod jacobi;
pub mod min_max;
pub mod orthogonal_polys;
pub mod other;
pub mod prime_pi;
pub mod special;
pub mod transcendental;
pub mod wigner;

pub use airy::{airy_ai, airy_ai_prime, airy_bi, airy_bi_prime};
pub use generalized::{dirac_delta, heaviside, kronecker_delta, signum, unit_step};
pub use hypergeometric::{hypergeometric, hypergeometric_m, hypergeometric_u};
pub use jacobi::{inverse_jacobi, jacobi_am, jacobi_cn, jacobi_dn, jacobi_sn};
pub use min_max::{max2, max_symbolic, min2, min_symbolic};
pub use orthogonal_polys::{
    chebyshev_t, chebyshev_u, gen_laguerre, hermite, jacobi_p, laguerre, legendre_p, legendre_q,
    ultraspherical,
};
pub use other::{
    abs_symbolic, binomial, ceil, conjugate, factorial, floor, frac, imag_part, real_part,
};
pub use prime_pi::{legendre_phi, prime_pi};
pub use special::{
    elliptic_e, elliptic_ec, elliptic_eu, elliptic_f, elliptic_j, elliptic_kc, elliptic_pi,
    spherical_harmonic,
};
pub use transcendental::{
    dickman_rho, hurwitz_zeta, stieltjes, zeta, zeta_deriv, zeta_symmetric,
};
pub use wigner::{clebsch_gordan, gaunt, racah, wigner_3j, wigner_6j, wigner_9j};


