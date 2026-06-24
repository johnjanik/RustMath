//! # rustmath-galois — generic Stauduhar Galois-group identification
//!
//! This crate identifies the Galois group of an irreducible degree-`n` integer
//! polynomial as a transitive-group label `nTt`, **natively** (no OSCAR/MAGMA
//! oracle). It implements the classical *Stauduhar descent*: label the roots
//! numerically, then walk down the subgroup lattice of `S_n`, at each step
//! testing a relative resolvent for a simple rational root and descending into
//! the corresponding maximal subgroup.
//!
//! ## Layout
//!
//! * [`perm`] — generic (arbitrary-degree) permutations and the small amount of
//!   permutation-group machinery the descent needs (closure, cosets, parity,
//!   conjugation, named groups).
//! * [`subgroups`] — the subgroup lattice and **maximal-subgroup** computation
//!   for small `|G|` (used by the small-degree descent).
//! * [`resolvent_eval`] — numeric construction of the **relative resolvent**
//!   `R_{G,H}` at the labeled roots and the rational-root (Stauduhar) criterion,
//!   built on `rustmath_polynomials::root_label` arbitrary-precision complex
//!   arithmetic.
//! * [`labels`] — small-degree (`n = 3, 4, 5`) transitive-group tables and a
//!   conjugacy-invariant identifier mapping a stabilised group to its `nTt`.
//! * [`descent`] — the small-degree Stauduhar driver: [`descent::galois_group`].
//! * [`deg24`] — the degree-24 imprimitive narrowing: it consumes the degree-24
//!   transitive atlas (`rustmath_groups::transitive24`) and the native cycle-type
//!   / resolvent machinery to narrow the candidate class as far as the resolvents
//!   reach.
//!
//! ## Reuse
//!
//! The numeric kernel (`complex_roots`, `BigComplex`, `round_to_integer_if_close`,
//! `rational_reconstruction`) and the exact resolvent builders
//! (`subset_sum_resolvent`, `resolvent_orbit_signature`) come from
//! `rustmath-polynomials`; the degree-24 atlas and orbit-signature separators
//! come from `rustmath-groups`. This crate adds the generic descent driver that
//! ties them together.

pub mod deg24;
pub mod descent;
pub mod labels;
pub mod perm;
pub mod resolvent_eval;
pub mod subgroups;

pub use descent::{galois_group, Config, GaloisResult};
pub use labels::{identify, NamedGroup};
