//! Cross-checks of the word-size kernel `SmallFq` (MAGMA Handbook ch. 21, digest
//! `rustmath-lava/docs/Chapter_21_Finite_Fields.md`) against the canonical `FiniteField` of this
//! crate (products, inverses, Frobenius, norm, squares), and of `count_distinct_roots` against
//! `fp_factor` (K = 1), `poly_factor::roots` over `FiniteField` (K = 2, 3, 4) and brute force.

use rustmath_core::Field;
use rustmath_finitefields::small_field::{count_distinct_roots_any_degree, SMALL_FIELD_MAX_DEGREE};
use rustmath_finitefields::{
    count_distinct_roots, roots, FiniteField, FiniteFieldElement, FqPoly, SmallFq, SmallFqElt,
};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;

/// SplitMix64 (Steele, Lea, Flood, OOPSLA 2014): a fixed, seeded stream for reproducible tests.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn to_ff<const K: usize>(ff: &FiniteField, a: &SmallFqElt<K>) -> FiniteFieldElement {
    ff.element(a.iter().map(|&c| Integer::from(c)).collect())
}

fn from_ff<const K: usize>(a: &FiniteFieldElement) -> SmallFqElt<K> {
    let mut r = [0u64; K];
    for (i, c) in a.eltseq().iter().enumerate() {
        if i < K {
            r[i] = c.to_i64().unwrap() as u64;
        }
    }
    r
}

fn arithmetic_agrees<const K: usize>(p: u64, n: usize, seed: u64) {
    let fq = SmallFq::<K>::new(p).unwrap();
    let modulus: Vec<Integer> = fq.modulus().iter().map(|&c| Integer::from(c)).collect();
    let ff = FiniteField::with_modulus(Integer::from(p), modulus).unwrap();
    assert_eq!(fq.order() as u128, (p as u128).pow(K as u32));
    let mut rng = Rng(seed);
    for _ in 0..n {
        let a = fq.from_code(rng.below(fq.order()));
        let b = fq.from_code(rng.below(fq.order()));
        let (fa, fb) = (to_ff(&ff, &a), to_ff(&ff, &b));
        assert_eq!(
            from_ff::<K>(&(fa.clone() * fb.clone())),
            fq.mul(&a, &b),
            "mul p={p} K={K}"
        );
        assert_eq!(
            from_ff::<K>(&(fa.clone() + fb.clone())),
            fq.add(&a, &b),
            "add"
        );
        assert_eq!(
            from_ff::<K>(&(fa.clone() - fb.clone())),
            fq.sub(&a, &b),
            "sub"
        );
        assert_eq!(
            from_ff::<K>(&fa.frobenius()),
            fq.frobenius(&a),
            "frobenius p={p} K={K}"
        );
        assert_eq!(fa.norm(), Integer::from(fq.norm(&a)), "norm p={p} K={K}");
        assert_eq!(fa.is_square(), fq.is_square(&a), "is_square p={p} K={K}");
        match fq.inverse(&a) {
            None => assert!(fq.is_zero(&a)),
            Some(ai) => {
                assert_eq!(fq.mul(&a, &ai), fq.one());
                let inv = fa.inverse().unwrap();
                assert_eq!(from_ff::<K>(&inv), ai, "inverse p={p} K={K}");
            }
        }
    }
}

#[test]
fn arithmetic_matches_finite_field() {
    for &p in &[5u64, 7, 31, 127, 65521] {
        arithmetic_agrees::<1>(p, 200, 11 + p);
        arithmetic_agrees::<2>(p, 200, 12 + p);
        arithmetic_agrees::<3>(p, 200, 13 + p);
        if p < 65521 {
            arithmetic_agrees::<4>(p, 200, 14 + p);
        }
    }
    // 65521^4 = 1.84e19 > 2^63: refused (FieldTooLarge in the curve counter).
    assert!(SmallFq::<4>::new(65521).is_err());
    assert_eq!(SMALL_FIELD_MAX_DEGREE, 8);
    // K = 8 with a small p.
    arithmetic_agrees::<8>(3, 50, 99);
}

/// The product of `(x - r)` over `roots` (with repetitions) and of `extra` (little-endian).
fn product<const K: usize>(
    fq: &SmallFq<K>,
    linear: &[SmallFqElt<K>],
    extra: &[Vec<SmallFqElt<K>>],
) -> Vec<SmallFqElt<K>> {
    let mut f = vec![fq.one()];
    let mul = |a: &[SmallFqElt<K>], b: &[SmallFqElt<K>]| -> Vec<SmallFqElt<K>> {
        let mut r = vec![fq.zero(); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                r[i + j] = fq.add(&r[i + j], &fq.mul(x, y));
            }
        }
        r
    };
    for r in linear {
        f = mul(&f, &[fq.neg(r), fq.one()]);
    }
    for e in extra {
        f = mul(&f, e);
    }
    f
}

fn brute_force<const K: usize>(fq: &SmallFq<K>, f: &[SmallFqElt<K>]) -> usize {
    (0..fq.order())
        .filter(|&n| {
            let x = fq.from_code(n);
            let mut acc = fq.zero();
            for c in f.iter().rev() {
                acc = fq.add(&fq.mul(&acc, &x), c);
            }
            fq.is_zero(&acc)
        })
        .count()
}

fn roots_agree<const K: usize>(p: u64, n: usize, seed: u64) {
    let fq = SmallFq::<K>::new(p).unwrap();
    let modulus: Vec<Integer> = fq.modulus().iter().map(|&c| Integer::from(c)).collect();
    let ff = FiniteField::with_modulus(Integer::from(p), modulus).unwrap();
    let mut rng = Rng(seed);
    for case in 0..n {
        // 0..3 linear factors drawn from a small pool (so repeats occur), and 0..2 random
        // monic factors of degree 1..3; total degree 1..=6 (after trimming).
        let pool: Vec<SmallFqElt<K>> = (0..3)
            .map(|_| fq.from_code(rng.below(fq.order())))
            .collect();
        let nl = rng.below(4) as usize;
        let linear: Vec<SmallFqElt<K>> = (0..nl).map(|_| pool[rng.below(3) as usize]).collect();
        let mut extra = Vec::new();
        let mut deg = nl;
        for _ in 0..rng.below(3) {
            let d = 1 + rng.below(3) as usize;
            if deg + d > 6 {
                break;
            }
            let mut g: Vec<SmallFqElt<K>> = (0..d)
                .map(|_| fq.from_code(rng.below(fq.order())))
                .collect();
            g.push(fq.one());
            extra.push(g);
            deg += d;
        }
        if deg == 0 {
            continue;
        }
        // a nonzero leading scalar
        let lead = fq.from_code(1 + rng.below(fq.order() - 1));
        let f: Vec<SmallFqElt<K>> = product(&fq, &linear, &extra)
            .iter()
            .map(|c| fq.mul(c, &lead))
            .collect();
        let got = count_distinct_roots(&fq, &f).unwrap();
        let got_vec = count_distinct_roots_any_degree(&fq, &f).unwrap();
        assert_eq!(
            got, got_vec,
            "fixed-size and vector paths, p={p} K={K} case {case}"
        );
        let expected = if K == 1 {
            let fi: Vec<i64> = f.iter().map(|c| c[0] as i64).collect();
            fp_factor::factor_with_multiplicity(&fi, p as i64)
                .iter()
                .filter(|(g, _)| fp_factor::degree(g) == 1)
                .count()
        } else {
            let poly = FqPoly::new(ff.clone(), f.iter().map(|c| to_ff(&ff, c)).collect());
            roots(&poly).len()
        };
        assert_eq!(got, expected, "root count p={p} K={K} case {case} f={f:?}");
        if fq.order() <= 400 {
            assert_eq!(
                got,
                brute_force(&fq, &f),
                "brute force p={p} K={K} case {case}"
            );
        }
    }
}

#[test]
fn distinct_roots_match_factorization() {
    for &p in &[5u64, 7, 31, 127] {
        roots_agree::<1>(p, 100, 21 + p);
    }
    for &p in &[5u64, 7, 31] {
        roots_agree::<2>(p, 100, 22 + p);
    }
    roots_agree::<3>(5, 100, 23);
    roots_agree::<3>(7, 60, 24);
    roots_agree::<4>(5, 60, 25);
    roots_agree::<4>(7, 30, 26);
}

/// Splitting fields: `x^q - x` itself has q roots; a product of all linear factors of a
/// subfield `GF(p^d)` (the polynomial `x^(p^d) - x`) has `p^d` roots in `GF(p^K)` for `d | K`.
#[test]
fn subfield_polynomials() {
    let fq = SmallFq::<2>::new(5).unwrap();
    // x^5 - x over GF(25): the 5 elements of F_5 (vector path, degree 5).
    let mut f = vec![fq.zero(); 6];
    f[1] = fq.from_fp(4);
    f[5] = fq.one();
    assert_eq!(count_distinct_roots(&fq, &f), Some(5));
    // x^25 - x over GF(25): 25 roots.
    let mut g = vec![fq.zero(); 26];
    g[1] = fq.from_fp(4);
    g[25] = fq.one();
    assert_eq!(count_distinct_roots(&fq, &g), Some(25));
    // over GF(5^4): x^25 - x has the 25 elements of GF(25).
    let f4 = SmallFq::<4>::new(5).unwrap();
    let mut h = vec![f4.zero(); 26];
    h[1] = f4.from_fp(4);
    h[25] = f4.one();
    assert_eq!(count_distinct_roots(&f4, &h), Some(25));
    // and x^5 - x over GF(5^3) (vector path, 3 does not matter): 5 roots.
    let f3 = SmallFq::<3>::new(5).unwrap();
    let mut k = vec![f3.zero(); 6];
    k[1] = f3.from_fp(4);
    k[5] = f3.one();
    assert_eq!(count_distinct_roots(&f3, &k), Some(5));
}

/// Degree 23 polynomials (the T-fibration of the M23 curve): vector path against
/// `poly_factor::roots` over `FiniteField`.
#[test]
fn degree_23_vector_path() {
    let fq = SmallFq::<2>::new(7).unwrap();
    let modulus: Vec<Integer> = fq.modulus().iter().map(|&c| Integer::from(c)).collect();
    let ff = FiniteField::with_modulus(Integer::from(7), modulus).unwrap();
    let mut rng = Rng(2323);
    for _ in 0..4 {
        let pool: Vec<SmallFqElt<2>> = (0..4).map(|_| fq.from_code(rng.below(49))).collect();
        let linear: Vec<SmallFqElt<2>> = (0..9).map(|_| pool[rng.below(4) as usize]).collect();
        let mut g: Vec<SmallFqElt<2>> = (0..14).map(|_| fq.from_code(rng.below(49))).collect();
        g.push(fq.one());
        let f = product(&fq, &linear, &[g]);
        assert_eq!(f.len(), 24);
        let got = count_distinct_roots(&fq, &f).unwrap();
        let poly = FqPoly::new(ff.clone(), f.iter().map(|c| to_ff(&ff, c)).collect());
        assert_eq!(got, roots(&poly).len());
        assert_eq!(got, brute_force(&fq, &f));
    }
}
