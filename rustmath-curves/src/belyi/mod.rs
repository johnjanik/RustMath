//! Belyi maps and dessins d'enfants — the construction layer for covers of `P¹`.
//!
//! This submodule folds the proven `dessin_engine` Belyi/cover machinery into
//! `rustmath-curves` (semantic home: covers of curves), retyped onto RustMath
//! foundation types (`rustmath_integers::Integer`, `rustmath_rationals::Rational`)
//! and this crate's `rustmath_polynomials` system types
//! ([`PolySystem`](rustmath_polynomials::poly_system::PolySystem),
//! [`MultivariatePolynomial`](rustmath_polynomials::multivariate::MultivariatePolynomial)).
//!
//! Source of the port: `/home/john/inverse_galois/M23/dessin_engine`. This is the
//! **construct + verify** half:
//!
//! * [`monodromy`] — [`Permutation`](monodromy::Permutation),
//!   [`BelyiTriple`](monodromy::BelyiTriple), and the Riemann–Hurwitz genus
//!   [`genus_from_branch_cycles`](monodromy::genus_from_branch_cycles).
//! * [`encode`] — the homogeneous ansatz data model and the direct encoder
//!   producing a `PolySystem`.
//! * [`portal`] — a general genus-0 ansatz from cycle types.
//! * [`pinned`] — the λ-pinned degree-24 `[2,12,5]` system `A²B − λR⁵S = c·x¹²`
//!   plus the parameter-homotopy contract (`ψ`, `p*`) — newly authored from the
//!   math spec.
//! * [`bad_locus`] — the conservative `Z_C` predicate.
//! * [`verify`] — the exact cover verifier and its hard constructibility gate.
//!
//! The descent / decide half (conic bridge, descent, pipeline) is added later by
//! a separate agent.

pub mod audit;
pub mod bad_locus;
pub mod bridge;
pub mod descent;
pub mod encode;
pub mod monodromy;
pub mod pinned;
pub mod pipeline;
pub mod portal;
pub mod verify;
