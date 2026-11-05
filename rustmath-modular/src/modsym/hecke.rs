//! # Hecke operators on weight-2 modular symbols for Gamma0(N), over Q
//!
//! The Hecke operators T_n acting on the Manin-symbol presentation of
//! M_2(Gamma0(N)) from [`super::gamma0`], computed exactly over Q:
//!
//! * primes p: via Merel's determinant-p matrix family
//!   ([`super::heilbronn::merel_matrices`]), T_p [(c:d)] = sum_h [(c:d) h].
//!   For p | N the terms falling outside P^1(Z/NZ) are omitted, which
//!   computes the operator U_p (verified against the p-coset double-coset
//!   definition; see the module docs of `heilbronn`).
//! * prime powers, p not dividing N: T_{p^{k+1}} = T_p T_{p^k} - p T_{p^{k-1}}
//!   (weight 2, trivial character, so the diamond operator is the identity).
//! * prime powers, p | N: T_{p^k} = U_p^k.
//! * general n: T_{mn} = T_m T_n for coprime m, n.
//!
//! The restriction to the cuspidal subspace and its characteristic
//! polynomial (Samuelson-Berkowitz, exact over Q) realize the
//! Eichler-Shimura relation: for a genus-one level N with elliptic curve E,
//! charpoly(T_p | S-part) = (x - a_p)^2 with a_p = p + 1 - #E(F_p).
//!
//! Corresponds to `sage.modular.modsym.ambient` (`hecke_matrix`,
//! `_compute_hecke_matrix_prime`) and the MAGMA handbook chapter "Modular
//! Symbols" (`HeckeOperator`).  References: Cremona, *Algorithms for Modular
//! Elliptic Curves*, sections 2.4 and 2.9; Stein, *Modular Forms: A
//! Computational Approach*, chapter 8; Merel, *Universal Fourier expansions
//! of modular forms*.
//!
//! Every expected value asserted in the tests below was recomputed
//! independently in python before implementation: conductors via the
//! discriminant/c4 criterion (all five curve models are semistable with
//! Delta supported exactly at the primes of N), a_p by direct point
//! counting over F_p (nonsingular points at multiplicative primes), and the
//! composite recursion against classical eta-product q-expansions whose
//! prime coefficients were themselves matched against the point counts.

use super::gamma0::ModularSymbolsGamma0;
use super::heilbronn::merel_matrices;
use crate::hecke::HeckeModule;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::{charpoly_berkowitz, Matrix};
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

/// Rational from a small signed integer.
fn rat(k: i64) -> Rational {
    Rational::from_integer(Integer::from(k))
}

/// Deterministic primality test by trial division (inputs here are small).
fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Prime factorization n = prod p^e as (p, e) pairs, p increasing.
fn factor_u64(mut n: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            let mut e = 0u32;
            while n.is_multiple_of(d) {
                n /= d;
                e += 1;
            }
            out.push((d, e));
        }
        d += 1;
    }
    if n > 1 {
        out.push((n, 1));
    }
    out
}

impl ModularSymbolsGamma0 {
    /// Image of the i-th Manin generator under sum_h [(c:d) h] over the
    /// given determinant-n matrices, projected to basis coordinates.
    /// Terms landing outside P^1(Z/NZ) (possible only when gcd(n, N) > 1)
    /// are omitted, per the U_p convention.
    fn hecke_image_of_generator(
        &self,
        mats: &[[[i64; 2]; 2]],
        i: usize,
    ) -> Vec<Rational> {
        let mut acc = vec![Rational::zero(); self.dimension()];
        for h in mats {
            if let Some(j) = self.p1().apply_right(i, *h) {
                for (t, v) in acc.iter_mut().zip(self.manin_generator_coords(j)) {
                    *t = &*t + v;
                }
            }
        }
        acc
    }

    /// The matrix of T_p (p prime, p not dividing N) or U_p (p | N) on the
    /// quotient basis, acting on coordinate column vectors: column k is the
    /// image of the k-th basis element.
    ///
    /// Computed by Merel's determinant-p family; see [`merel_matrices`] for
    /// the defining property and its brute-force verification.
    pub fn hecke_matrix_prime(&self, p: u64) -> Matrix<Rational> {
        assert!(
            is_prime_u64(p),
            "hecke_matrix_prime requires a prime; use hecke_matrix for composite n"
        );
        let mats = merel_matrices(p);
        let dim = self.dimension();
        let cols: Vec<Vec<Rational>> = self
            .basis_manin_indices()
            .iter()
            .map(|&g| self.hecke_image_of_generator(&mats, g))
            .collect();
        let mut flat = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for col in &cols {
                flat.push(col[i].clone());
            }
        }
        Matrix::from_vec(dim, dim, flat).expect("dim x dim Hecke matrix")
    }

    /// The matrix of the Hecke operator T_n, n >= 1, on the quotient basis
    /// (acting on coordinate column vectors).
    ///
    /// Composite indices use the verified Hecke-algebra recursion for
    /// weight 2 and trivial character: T_{mn} = T_m T_n for coprime m, n;
    /// T_{p^{k+1}} = T_p T_{p^k} - p T_{p^{k-1}} for p not dividing N;
    /// T_{p^k} = U_p^k for p | N.
    pub fn hecke_matrix(&self, n: u64) -> Matrix<Rational> {
        assert!(n >= 1, "Hecke operators are indexed by n >= 1");
        let dim = self.dimension();
        let mut result: Matrix<Rational> = Matrix::identity(dim);
        for (p, e) in factor_u64(n) {
            let tp = self.hecke_matrix_prime(p);
            let tpe = if self.level().is_multiple_of(p) {
                // U_p^e
                let mut acc = tp.clone();
                for _ in 1..e {
                    acc = (acc * tp.clone()).expect("square matrices of equal size");
                }
                acc
            } else {
                // T_{p^{k+1}} = T_p T_{p^k} - p T_{p^{k-1}}
                let mut prev: Matrix<Rational> = Matrix::identity(dim);
                let mut cur = tp.clone();
                for _ in 1..e {
                    let prod = (tp.clone() * cur.clone())
                        .expect("square matrices of equal size");
                    let next = (prod - prev.scalar_mul(&rat(p as i64)))
                        .expect("square matrices of equal size");
                    prev = cur;
                    cur = next;
                }
                cur
            };
            result = (result * tpe).expect("square matrices of equal size");
        }
        result
    }

    /// The matrix of T_n restricted to the cuspidal subspace, in the
    /// coordinates of [`Self::cuspidal_basis`].
    ///
    /// Panics if T_n does not preserve the cuspidal subspace, which is
    /// mathematically impossible (Hecke operators commute with the boundary
    /// map); the check is kept as a hard internal-consistency assertion.
    pub fn hecke_matrix_cuspidal(&self, n: u64) -> Matrix<Rational> {
        let s = self.cuspidal_dimension();
        if s == 0 {
            return Matrix::zeros(0, 0);
        }
        let dim = self.dimension();
        let t = self.hecke_matrix(n);
        let cb = self.cuspidal_basis();
        // tc[j] = T * (j-th cuspidal basis vector)
        let mut tc = vec![vec![Rational::zero(); dim]; s];
        for (v, trow) in cb.iter().zip(tc.iter_mut()) {
            for (i, out) in trow.iter_mut().enumerate() {
                let mut sum = Rational::zero();
                for (k, vk) in v.iter().enumerate() {
                    if vk.is_zero() {
                        continue;
                    }
                    let a = t.get(i, k).expect("entry in range");
                    sum = &sum + &(a * vk);
                }
                *out = sum;
            }
        }
        // Solve C X = T C via rref of the augmented matrix [C | TC]: the
        // columns of C are linearly independent, so the pivots are exactly
        // 0..s iff every T C column lies in the span of C.
        let mut flat = Vec::with_capacity(dim * 2 * s);
        for i in 0..dim {
            for v in cb.iter() {
                flat.push(v[i].clone());
            }
            for v in tc.iter() {
                flat.push(v[i].clone());
            }
        }
        let aug = Matrix::from_vec(dim, 2 * s, flat).expect("augmented matrix shape");
        let rref = aug
            .reduced_row_echelon_form()
            .expect("exact rref over Q cannot fail");
        assert_eq!(
            rref.pivots,
            (0..s).collect::<Vec<_>>(),
            "T_{n} must preserve the cuspidal subspace"
        );
        let mut xflat = Vec::with_capacity(s * s);
        for i in 0..s {
            for j in 0..s {
                xflat.push(rref.matrix.get(i, s + j).expect("entry in range").clone());
            }
        }
        Matrix::from_vec(s, s, xflat).expect("s x s restricted matrix")
    }

    /// The exact characteristic polynomial of T_n on the cuspidal subspace,
    /// via the division-free Samuelson-Berkowitz algorithm.  Monic, with
    /// coefficients in increasing degree order.
    pub fn hecke_charpoly_cuspidal(&self, n: u64) -> UnivariatePolynomial<Rational> {
        charpoly_berkowitz(&self.hecke_matrix_cuspidal(n))
            .expect("cuspidal Hecke matrix is square")
    }
}

/// The Manin-symbol space is a Hecke module: T_n acts through
/// [`ModularSymbolsGamma0::hecke_matrix`].
impl HeckeModule for ModularSymbolsGamma0 {
    fn dimension(&self) -> usize {
        ModularSymbolsGamma0::dimension(self)
    }

    fn apply_hecke(&self, n: u64, element: &[Rational]) -> Vec<Rational> {
        let dim = ModularSymbolsGamma0::dimension(self);
        assert_eq!(element.len(), dim, "coordinate length");
        let t = self.hecke_matrix(n);
        let mut out = vec![Rational::zero(); dim];
        for (i, o) in out.iter_mut().enumerate() {
            for (k, x) in element.iter().enumerate() {
                if x.is_zero() {
                    continue;
                }
                *o = &*o + &(t.get(i, k).expect("entry in range") * x);
            }
        }
        out
    }

    fn level(&self) -> u64 {
        ModularSymbolsGamma0::level(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cusps::Cusp;

    /// prod over (root, multiplicity) of (x - root)^multiplicity, monic.
    fn poly_with_roots(roots: &[(i64, usize)]) -> UnivariatePolynomial<Rational> {
        let mut acc = UnivariatePolynomial::new(vec![rat(1)]);
        for &(r, m) in roots {
            for _ in 0..m {
                acc = acc * UnivariatePolynomial::new(vec![rat(-r), rat(1)]);
            }
        }
        acc
    }

    #[test]
    fn test_t1_is_identity() {
        for n in [1u64, 11, 14, 37] {
            let m = ModularSymbolsGamma0::new(n);
            assert_eq!(m.hecke_matrix(1), Matrix::identity(m.dimension()));
        }
    }

    #[test]
    fn test_hecke_well_defined_on_quotient() {
        // The Merel sum must kill the Manin relations x + xS and
        // x + xT + xT^2 for EVERY generator, else T_p would not descend to
        // the quotient.  Verified in python first; re-asserted here.
        for n in [11u64, 14, 15, 37] {
            let m = ModularSymbolsGamma0::new(n);
            let p1 = m.p1();
            for p in [2u64, 3, 5] {
                let mats = merel_matrices(p);
                for i in 0..m.num_generators() {
                    let si = p1.apply_s(i);
                    let ti = p1.apply_t(i);
                    let tti = p1.apply_t(ti);
                    let hi = m.hecke_image_of_generator(&mats, i);
                    let hs = m.hecke_image_of_generator(&mats, si);
                    assert!(
                        hi.iter().zip(hs.iter()).all(|(a, b)| (a + b).is_zero()),
                        "S-relation broken by T_{p} at level {n}, generator {i}"
                    );
                    let ht = m.hecke_image_of_generator(&mats, ti);
                    let htt = m.hecke_image_of_generator(&mats, tti);
                    assert!(
                        hi.iter()
                            .zip(ht.iter())
                            .zip(htt.iter())
                            .all(|((a, b), c)| (&(a + b) + c).is_zero()),
                        "T-relation broken by T_{p} at level {n}, generator {i}"
                    );
                }
            }
        }
    }

    /// gamma(cusp) for an integer matrix acting as a Mobius map.
    fn apply_mat(g: [[i64; 2]; 2], cusp: &Cusp) -> Cusp {
        let (p, q) = match cusp {
            Cusp::Infinity => (Integer::one(), Integer::zero()),
            Cusp::Rational(p, q) => (p.clone(), q.clone()),
        };
        let a = Integer::from(g[0][0]);
        let b = Integer::from(g[0][1]);
        let c = Integer::from(g[1][0]);
        let d = Integer::from(g[1][1]);
        Cusp::new(&(&a * &p) + &(&b * &q), &(&c * &p) + &(&d * &q))
    }

    #[test]
    fn test_hecke_matrix_matches_double_coset_path_definition() {
        // Independent in-Rust cross-check of the defining property:
        // T_p {a, b} = sum_{r<p} {(a+r)/p, (b+r)/p} + {pa, pb}, the last
        // term omitted for p | N (U_p).  The right-hand side goes through
        // the continued-fraction Manin trick, sharing nothing with the
        // Merel-family computation.
        let pairs = [
            (Cusp::zero(), Cusp::infinity()),
            (Cusp::from_i64(1, 2), Cusp::from_i64(2, 3)),
            (Cusp::from_i64(-1, 3), Cusp::infinity()),
        ];
        for (n, p) in [
            (11u64, 2u64),
            (11, 3),
            (15, 2),
            (37, 2),
            (14, 2), // U_2
            (15, 3), // U_3
        ] {
            let m = ModularSymbolsGamma0::new(n);
            let pi = p as i64;
            let mut deltas: Vec<[[i64; 2]; 2]> =
                (0..pi).map(|r| [[1, r], [0, pi]]).collect();
            if n % p != 0 {
                deltas.push([[pi, 0], [0, 1]]);
            }
            for (alpha, beta) in &pairs {
                let coords = m.modular_symbol(alpha, beta);
                let lhs = m.apply_hecke(p, &coords);
                let mut rhs = vec![Rational::zero(); m.dimension()];
                for d in &deltas {
                    let img =
                        m.modular_symbol(&apply_mat(*d, alpha), &apply_mat(*d, beta));
                    for (t, v) in rhs.iter_mut().zip(img.iter()) {
                        *t = &*t + v;
                    }
                }
                assert_eq!(
                    lhs, rhs,
                    "T_{p} via Merel != double-coset definition at level {n}"
                );
            }
        }
    }

    #[test]
    fn test_eichler_shimura_level_11() {
        // E = 11a1: y^2 + y = x^3 - x^2 - 10x - 20.
        // Conductor 11 verified independently: c4 = 496, Delta = -161051 =
        // -11^5 (support {11}), 11 does not divide c4, so the model is
        // minimal at 11 with multiplicative reduction: N = 11.
        // a_p = p + 1 - #E(F_p) by DIRECT POINT COUNTING in python over all
        // (x, y) in F_p^2 plus infinity; a_11 = 11 - #E_ns(F_11) at the
        // multiplicative prime.
        let m = ModularSymbolsGamma0::new(11);
        assert_eq!(m.cuspidal_dimension(), 2);
        let ap = [
            (2i64, -2i64),
            (3, -1),
            (5, 1),
            (7, -2),
            (13, 4),
            (17, -2),
            (19, 0),
            (23, -1),
        ];
        for (p, a) in ap {
            assert_eq!(
                m.hecke_charpoly_cuspidal(p as u64),
                poly_with_roots(&[(a, 2)]),
                "charpoly(T_{p}) != (x - {a})^2 at level 11"
            );
        }
        // U_11: a_11 = 1 (split multiplicative; 11 - #E_ns(F_11) = 1).
        assert_eq!(m.hecke_charpoly_cuspidal(11), poly_with_roots(&[(1, 2)]));
    }

    #[test]
    fn test_eichler_shimura_level_14() {
        // E = 14a: y^2 + xy + y = x^3 + 4x - 6.
        // Conductor 14 verified independently: c4 = -215 = -5*43,
        // Delta = -21952 = -2^6 * 7^3 (support {2, 7}), gcd(c4, 14) = 1, so
        // the model is minimal at 2 and 7 with multiplicative reduction and
        // good reduction elsewhere: N = 14.  a_p by direct point counting.
        let m = ModularSymbolsGamma0::new(14);
        assert_eq!(m.cuspidal_dimension(), 2);
        for (p, a) in [(3i64, -2i64), (5, 0), (11, 0), (13, -4)] {
            assert_eq!(
                m.hecke_charpoly_cuspidal(p as u64),
                poly_with_roots(&[(a, 2)]),
                "charpoly(T_{p}) != (x - {a})^2 at level 14"
            );
        }
        // U_2, U_7 at the multiplicative primes: a_2 = 2 - #E_ns(F_2) = -1,
        // a_7 = 7 - #E_ns(F_7) = 1.
        assert_eq!(m.hecke_charpoly_cuspidal(2), poly_with_roots(&[(-1, 2)]));
        assert_eq!(m.hecke_charpoly_cuspidal(7), poly_with_roots(&[(1, 2)]));
    }

    #[test]
    fn test_eichler_shimura_level_15() {
        // E = 15a: y^2 + xy + y = x^3 + x^2 - 10x - 10.
        // Conductor 15 verified independently: c4 = 481 = 13*37,
        // Delta = 50625 = 3^4 * 5^4 (support {3, 5}), gcd(c4, 15) = 1:
        // multiplicative at 3 and 5, good elsewhere: N = 15.
        let m = ModularSymbolsGamma0::new(15);
        assert_eq!(m.cuspidal_dimension(), 2);
        for (p, a) in [(2i64, -1i64), (7, 0), (11, -4), (13, -2)] {
            assert_eq!(
                m.hecke_charpoly_cuspidal(p as u64),
                poly_with_roots(&[(a, 2)]),
                "charpoly(T_{p}) != (x - {a})^2 at level 15"
            );
        }
        // U_3: a_3 = 3 - #E_ns(F_3) = -1; U_5: a_5 = 5 - #E_ns(F_5) = 1.
        assert_eq!(m.hecke_charpoly_cuspidal(3), poly_with_roots(&[(-1, 2)]));
        assert_eq!(m.hecke_charpoly_cuspidal(5), poly_with_roots(&[(1, 2)]));
    }

    #[test]
    fn test_level_37_genus_2_two_newforms() {
        // S_2(Gamma0(37)) has two rational newforms; both isogeny classes
        // were confirmed independently: 37a (y^2 + y = x^3 - x, c4 = 48,
        // Delta = 37) and 37b (y^2 + y = x^3 + x^2 - 23x - 50, c4 = 1120,
        // Delta = 50653 = 37^3), both semistable with Delta supported at 37
        // and c4 coprime to 37, hence both of conductor 37.  Point counts:
        // a_2 = -2 / 0, a_3 = -3 / 1, a_5 = -2 / 0 for 37a / 37b.  The
        // classes are distinct (different a_2), so on the 4-dimensional
        // cuspidal space charpoly(T_p) = (x - a_p(37a))^2 (x - a_p(37b))^2.
        let m = ModularSymbolsGamma0::new(37);
        assert_eq!(m.cuspidal_dimension(), 4);
        assert_eq!(
            m.hecke_charpoly_cuspidal(2),
            poly_with_roots(&[(-2, 2), (0, 2)]),
            "charpoly(T_2) != (x + 2)^2 x^2 at level 37"
        );
        assert_eq!(
            m.hecke_charpoly_cuspidal(3),
            poly_with_roots(&[(-3, 2), (1, 2)])
        );
        assert_eq!(
            m.hecke_charpoly_cuspidal(5),
            poly_with_roots(&[(-2, 2), (0, 2)])
        );
    }

    #[test]
    fn test_ambient_charpoly_includes_eisenstein_eigenvalue() {
        // On the full space M_2 the Eisenstein part contributes the
        // eigenvalue p + 1 (verified in python via the exact ambient
        // matrices): N = 11: charpoly(T_2) = (x - 3)(x + 2)^2,
        // charpoly(T_3) = (x - 4)(x + 1)^2; N = 37:
        // charpoly(T_2) = x^2 (x - 3)(x + 2)^2.
        let m11 = ModularSymbolsGamma0::new(11);
        assert_eq!(
            charpoly_berkowitz(&m11.hecke_matrix(2)).unwrap(),
            poly_with_roots(&[(3, 1), (-2, 2)])
        );
        assert_eq!(
            charpoly_berkowitz(&m11.hecke_matrix(3)).unwrap(),
            poly_with_roots(&[(4, 1), (-1, 2)])
        );
        let m37 = ModularSymbolsGamma0::new(37);
        assert_eq!(
            charpoly_berkowitz(&m37.hecke_matrix(2)).unwrap(),
            poly_with_roots(&[(0, 2), (3, 1), (-2, 2)])
        );
    }

    #[test]
    fn test_hecke_commutativity() {
        // T_p T_q = T_q T_p as exact ambient matrices, two prime pairs per
        // level, including a U_p at level 15.
        for (n, p, q) in [
            (11u64, 2u64, 3u64),
            (11, 2, 5),
            (15, 2, 7),
            (15, 2, 3), // T_2 with U_3
            (37, 2, 3),
            (37, 3, 5),
        ] {
            let m = ModularSymbolsGamma0::new(n);
            let tp = m.hecke_matrix(p);
            let tq = m.hecke_matrix(q);
            let pq = (tp.clone() * tq.clone()).unwrap();
            let qp = (tq * tp).unwrap();
            assert_eq!(pq, qp, "T_{p} T_{q} != T_{q} T_{p} at level {n}");
        }
    }

    #[test]
    fn test_hecke_preserves_cuspidal_subspace() {
        // The boundary of T_n(v) vanishes for every cuspidal basis vector v.
        // (hecke_matrix_cuspidal additionally hard-asserts invariance when
        // solving for the restriction.)
        for n in [11u64, 14, 15, 37] {
            let m = ModularSymbolsGamma0::new(n);
            for hn in [2u64, 3, 6] {
                for v in m.cuspidal_basis() {
                    let image = m.apply_hecke(hn, v);
                    assert!(
                        m.is_cuspidal(&image),
                        "T_{hn} broke cuspidality at level {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_composite_recursion_matches_merel_direct() {
        // For gcd(n, N) = 1 Merel's family is valid for composite n too;
        // computing T_n directly from the determinant-n family must agree
        // with the recursion used by hecke_matrix (T_4 = T_2^2 - 2,
        // T_6 = T_2 T_3, T_9 = T_3^2 - 3).  Verified in python first.
        for (level, n) in [(11u64, 4u64), (11, 6), (11, 9), (14, 9), (15, 4)] {
            let m = ModularSymbolsGamma0::new(level);
            let dim = m.dimension();
            let mats = merel_matrices(n);
            let cols: Vec<Vec<Rational>> = m
                .basis_manin_indices()
                .iter()
                .map(|&g| m.hecke_image_of_generator(&mats, g))
                .collect();
            let mut flat = Vec::with_capacity(dim * dim);
            for i in 0..dim {
                for col in &cols {
                    flat.push(col[i].clone());
                }
            }
            let direct = Matrix::from_vec(dim, dim, flat).unwrap();
            assert_eq!(
                direct,
                m.hecke_matrix(n),
                "Merel-direct T_{n} != recursion at level {level}"
            );
        }
    }

    #[test]
    fn test_composite_eigenvalues_level_11() {
        // 11a eigenvalues for composite indices, verified against the
        // eta-product q-expansion q prod (1-q^k)^2 (1-q^{11k})^2 whose prime
        // coefficients match every point-count a_p (p <= 13):
        // a_4 = a_2^2 - 2 = 2, a_6 = a_2 a_3 = 2, a_9 = a_3^2 - 3 = -2,
        // a_12 = a_4 a_3 = -2, a_8 = a_2 a_4 - 2 a_2 = 0.
        let m = ModularSymbolsGamma0::new(11);
        for (n, a) in [(4u64, 2i64), (6, 2), (9, -2), (12, -2), (8, 0)] {
            assert_eq!(
                m.hecke_charpoly_cuspidal(n),
                poly_with_roots(&[(a, 2)]),
                "a_{n} != {a} at level 11"
            );
        }
    }

    #[test]
    fn test_composite_eigenvalues_with_up_powers_level_14() {
        // 14a: a_2 = -1 (U_2), so a_4 = a_2^2 = 1, a_8 = a_2^3 = -1 (powers
        // of U_p at p | N), a_6 = a_2 a_3 = 2, a_12 = a_4 a_3 = -2; all
        // match the eta product for 14a (q-expansion
        // [1, -1, -2, 1, 0, 2, 1, -1, ...], primes matched to point counts).
        let m = ModularSymbolsGamma0::new(14);
        for (n, a) in [(4u64, 1i64), (8, -1), (6, 2), (12, -2)] {
            assert_eq!(
                m.hecke_charpoly_cuspidal(n),
                poly_with_roots(&[(a, 2)]),
                "a_{n} != {a} at level 14"
            );
        }
    }

    #[test]
    fn test_genus_zero_level_has_trivial_cuspidal_charpoly() {
        // N = 3: no cusp forms; the restricted matrix is 0 x 0 and its
        // characteristic polynomial is the constant 1.
        let m = ModularSymbolsGamma0::new(3);
        assert_eq!(m.cuspidal_dimension(), 0);
        assert_eq!(
            m.hecke_charpoly_cuspidal(2),
            UnivariatePolynomial::new(vec![rat(1)])
        );
    }

    #[test]
    fn test_hecke_module_trait_scalar_action_on_cuspidal() {
        // At level 11 every T_n acts as the scalar a_n on the cuspidal
        // subspace (verified in python: the restricted matrix is literally
        // a_n * I), so cuspidal basis vectors are exact eigenvectors.
        let m = ModularSymbolsGamma0::new(11);
        for (n, a) in [(2u64, -2i64), (3, -1), (5, 1)] {
            for v in m.cuspidal_basis() {
                let image = m.apply_hecke(n, v);
                let expected: Vec<Rational> =
                    v.iter().map(|x| &rat(a) * x).collect();
                assert_eq!(image, expected, "T_{n} v != {a} v at level 11");
            }
        }
        // HeckeModule::level and dimension route to the real space.
        assert_eq!(HeckeModule::level(&m), 11);
        assert_eq!(HeckeModule::dimension(&m), 3);
    }
}
