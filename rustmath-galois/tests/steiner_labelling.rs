//! The certified Steiner labelling of M23 polynomials (`rustmath_galois::steiner_labelling`):
//! paper `golay_m23.tex` of the M23 repository (`johnjanik/M23` at `d05475b`), Section 5; MAGMA
//! ch. 38 (`IsInt`, `GaloisRoot`), ch. 47, ch. 48.
//!
//! Fast tests (normal suite, debug) run on Le Lay's CC0 certified M23 polynomial
//! (`tests/data/m23/lelay_specialization_t2_Z.json`, monic associate) and on small inputs. Their
//! reference values have two sources each: the owner's `arith.py` run as an oracle with
//! python-flint 0.9.0 (`tests/data/m23/cc0_t2_steiner_reference.json`, `make_cc0_reference.py`)
//! and PARI/GP 2.15.4 (design SL `gp/steiner.gp`, by `polrootspadic`; its review's `rev.gp`, by
//! Newton in `Mod(Mod(., p^N), f)`; implementation report SL `gp/check_sl.gp`).
//!
//! Opt-in tests:
//! - `full_` (`#[ignore]`, release): more CC0 primes.
//! - `env_` (`#[ignore]`, release): read HJLPPZ's f1, f2 at run time from the checkout named by
//!   `M23ISGALOIS` (no licence there, so they are never copied here), and compare with the
//!   owner's data `tests/data/m23/golay_resolvents_no_f.json` (f removed) and
//!   `frobenius_signs.json`. Without `M23ISGALOIS` they panic; they never pass vacuously.
//!
//! ```text
//! cargo test -p rustmath-galois --test steiner_labelling
//! cargo test --release -p rustmath-galois --test steiner_labelling -- --ignored full_
//! M23ISGALOIS=<checkout of shaowuz/m23isgalois> \
//!   cargo test --release -p rustmath-galois --test steiner_labelling -- --ignored env_
//! ```

use rustmath_coding::cyclic_golay::{heptads, support};
use rustmath_finitefields::FrobeniusRootsError;
use rustmath_galois::integrality::{symmetric_residue, IntegralityError};
use rustmath_galois::steiner_labelling::{
    duad_resolvent, first_inert_prime, frobenius_sign_given_t5, monic_associate, precision_for,
    steiner_labelling, t5_candidates, Certification, FrobeniusSign, GaloisHypothesis, GolaySign,
    LabellingOptions, LabellingStatus, RootBound, SteinerError, SteinerLabelling,
};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------

fn int(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap_or_else(|| panic!("not a decimal integer: {s}"))
}

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&c| Integer::from(c)).collect()
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/m23")
}

fn read_json(name: &str) -> Value {
    let path = data_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON {}: {e}", path.display()))
}

/// A JSON integer given as a number or a decimal string.
fn jint(v: &Value) -> Integer {
    match v {
        Value::String(s) => int(s),
        Value::Number(n) => int(&n.to_string()),
        other => panic!("not an integer: {other}"),
    }
}

fn jints(v: &Value) -> Vec<Integer> {
    v.as_array().expect("array").iter().map(jint).collect()
}

fn opts(p: u64) -> LabellingOptions {
    LabellingOptions {
        prime: Some(p),
        ..LabellingOptions::default()
    }
}

fn digits(n: &Integer) -> usize {
    n.abs().to_string().len()
}

/// Le Lay's certified specialization (CC0), raw coefficients.
fn cc0_raw() -> Vec<Integer> {
    jints(&read_json("lelay_specialization_t2_Z.json")["coefficients_low_to_high"])
}

/// Its monic associate `a^22 f(x/a)`.
fn cc0() -> Vec<Integer> {
    monic_associate(&cc0_raw()).unwrap().0
}

fn cc0_ref() -> Value {
    read_json("cc0_t2_steiner_reference.json")
}

/// The CC0 labelling with the default options (first inert prime 29, N = 270), computed once
/// and shared by the fast tests (about 2.5 s in debug).
fn cc0_at_29() -> &'static SteinerLabelling {
    static L: OnceLock<SteinerLabelling> = OnceLock::new();
    L.get_or_init(|| steiner_labelling(&cc0(), &LabellingOptions::default()).unwrap())
}

/// `x^23 - x - 1` (Galois group S23: H. Osada, J. Number Theory 25 (1987) 230-238).
fn s23() -> Vec<Integer> {
    let mut g = vec![Integer::zero(); 24];
    g[0] = Integer::from(-1);
    g[1] = Integer::from(-1);
    g[23] = Integer::one();
    g
}

/// The degree 23 subfield of Q(zeta_47) (Galois group C23): PARI/GP `Vecrev(polsubcyclo(47, 23))`
/// (gp/check_sl.gp).
fn c23() -> Vec<Integer> {
    ints(&[
        -1, -12, 66, 286, -715, -2002, 3003, 6435, -6435, -11440, 8008, 12376, -6188, -8568, 3060,
        3876, -969, -1140, 190, 210, -21, -22, 1, 1,
    ])
}

fn residues(l: &SteinerLabelling) -> (Integer, Integer) {
    (
        l.t5_residue(GolaySign::Plus).clone(),
        l.t5_residue(GolaySign::Minus).clone(),
    )
}

fn factor_degrees(f: &[Integer], p: i64) -> Vec<usize> {
    let pi = Integer::from(p);
    let fp: Vec<i64> = f.iter().map(|c| c.modulo(&pi).to_i64().unwrap()).collect();
    let mut d: Vec<usize> = fp_factor::factor_with_multiplicity(&fp, p)
        .into_iter()
        .map(|(g, e)| {
            assert_eq!(e, 1);
            fp_factor::trim(&g).len() - 1
        })
        .collect();
    d.sort_unstable();
    d
}

/// Power sums `P_0..P_k` of the roots of a monic `f` (Newton's identities over `Z`).
fn power_sums(f: &[Integer], k: usize) -> Vec<Integer> {
    let n = f.len() - 1;
    let a = |j: usize| -> Integer {
        if j <= n {
            f[n - j].clone()
        } else {
            Integer::zero()
        }
    };
    let mut p = vec![Integer::from(n as i64)];
    for m in 1..=k {
        let mut s = -(Integer::from(m as i64) * a(m));
        for j in 1..m {
            s = s - a(j) * p[m - j].clone();
        }
        p.push(s);
    }
    p
}

/// The set partitions of `0..k`.
fn set_partitions(k: usize) -> Vec<Vec<Vec<usize>>> {
    let mut out: Vec<Vec<Vec<usize>>> = vec![vec![]];
    for i in 0..k {
        let mut next = Vec::new();
        for part in out {
            for b in 0..part.len() {
                let mut q = part.clone();
                q[b].push(i);
                next.push(q);
            }
            let mut q = part.clone();
            q.push(vec![i]);
            next.push(q);
        }
        out = next;
    }
    out
}

/// The monomial symmetric function `m_lambda` of the roots, from power sums by Moebius
/// inversion over set partitions (augmented monomials divided by the part multiplicities).
fn monomial(lambda: &[usize], p: &[Integer]) -> Integer {
    let mut aug = Integer::zero();
    for part in set_partitions(lambda.len()) {
        let mut term = Integer::one();
        for b in &part {
            let s: usize = b.iter().map(|&i| lambda[i]).sum();
            let mu = (1..b.len() as i64).product::<i64>() * if b.len() % 2 == 0 { -1 } else { 1 };
            term = term * Integer::from(mu) * p[s].clone();
        }
        aug = aug + term;
    }
    let mut den = 1i64;
    let mut i = 0;
    while i < lambda.len() {
        let j = (i..lambda.len())
            .take_while(|&j| lambda[j] == lambda[i])
            .count();
        den *= (1..=j as i64).product::<i64>();
        i += j;
    }
    let (q, r) = (aug.clone() / Integer::from(den), aug % Integer::from(den));
    assert!(r.is_zero());
    q
}

/// `T5 - 120 J5` from the 4-design property (design SL, 2.7): independent of all p-adic code.
fn t5_minus_120_j5(f: &[Integer]) -> Integer {
    let p = power_sums(f, 5);
    let terms: [(i64, &[usize]); 6] = [
        (77, &[5]),
        (105, &[4, 1]),
        (210, &[3, 2]),
        (100, &[3, 1, 1]),
        (150, &[2, 2, 1]),
        (60, &[2, 1, 1, 1]),
    ];
    terms.iter().fold(Integer::zero(), |acc, (c, l)| {
        acc + Integer::from(*c) * monomial(l, &p)
    })
}

/// The direct sum over all 253 heptads of `s_H^5`, a base-ring value (symmetric residue).
fn t5_direct(roots: &rustmath_finitefields::FrobeniusRoots, sign: GolaySign) -> Integer {
    let r = roots.roots();
    let ring = roots.ring();
    let mut tot = ring.zero();
    for &h in heptads(sign) {
        let s = support(h)
            .into_iter()
            .fold(ring.zero(), |acc, i| acc + r[i].clone());
        let s2 = s.clone() * s.clone();
        tot = tot + s2.clone() * s2 * s;
    }
    symmetric_residue(
        &tot.base_ring_value().expect("T5 is fixed by the Frobenius"),
        roots.modulus(),
    )
}

// ------------------------------------------------------------------------------------------
// Fast tests: bounds, small inputs, degenerate cases
// ------------------------------------------------------------------------------------------

#[test]
fn bounds_and_precision() {
    // arith.py root_bound and PARI/GP fbest (review 1, rev.gp) agree on best.
    let b = RootBound::fujiwara(&s23()).unwrap();
    assert_eq!(b.best(), &Integer::from(256));
    assert_eq!(b.t5_bound(), Integer::from(136069473));
    let c = RootBound::fujiwara(&c23()).unwrap();
    assert_eq!(c.best(), &Integer::from(1201));
    assert_eq!(c.t5_bound(), int("309225934046"));
    // x^2 - 4: mu = 2, best = 512, M = 4 = 2 mu (Fujiwara is sharp up to the factor 2).
    let q = RootBound::fujiwara(&ints(&[-4, 0, 1])).unwrap();
    assert_eq!(q.best(), &Integer::from(512));
    assert_eq!(q.power_bound(1, 1, 1), Integer::from(5));
    assert_eq!(q.binomial_bound(1, 2), Integer::from(26));
    assert!(matches!(
        RootBound::fujiwara(&ints(&[1])),
        Err(SteinerError::WrongDegree(0))
    ));
    assert!(matches!(
        RootBound::fujiwara(&ints(&[1, 2])),
        Err(SteinerError::NotMonic)
    ));
    // N(B, e): least N with p^N > 2 B 10^e.
    assert_eq!(precision_for(&Integer::from(29), &Integer::from(14), 0), 1);
    assert_eq!(precision_for(&Integer::from(29), &Integer::from(15), 0), 2);
    assert_eq!(
        precision_for(&Integer::from(23), &Integer::from(136069473), 0),
        7
    );
    assert_eq!(
        precision_for(&Integer::from(23), &Integer::from(136069473), 30),
        29
    );
    assert_eq!(
        precision_for(&Integer::from(157), &Integer::from(136069473), 30),
        18
    );
}

#[test]
fn monic_associate_scales_the_roots() {
    // 2x^2 - 3x + 1 = (2x - 1)(x - 1): a = 2, g = x^2 - 3x + 2 = (x - 1)(x - 2).
    let (g, a) = monic_associate(&ints(&[1, -3, 2, 0])).unwrap();
    assert_eq!(a, Integer::from(2));
    assert_eq!(g, ints(&[2, -3, 1]));
    // Already monic: unchanged.
    let (g, a) = monic_associate(&ints(&[5, 1])).unwrap();
    assert_eq!((g, a), (ints(&[5, 1]), Integer::one()));
    // -x + 2 (root 2): a = -1, g = x + 2 (root -2 = a * 2).
    let (g, a) = monic_associate(&ints(&[2, -1])).unwrap();
    assert_eq!((g, a), (ints(&[2, 1]), Integer::from(-1)));
    assert!(matches!(
        monic_associate(&ints(&[0, 0])),
        Err(SteinerError::ZeroLeadingCoefficient)
    ));
}

#[test]
fn c23_is_unresolved_at_every_precision() {
    // gp/check_sl.gp (rev.gp), design SL gp/c23.gp and arith.py agree: at 2 and 5 the residues
    // are T5+ = -98317, T5- = -30637; at 3 they are swapped.
    let f = c23();
    for (p, plus, minus) in [(2u64, -98317i64, -30637i64), (3, -30637, -98317)] {
        let l = steiner_labelling(
            &f,
            &LabellingOptions {
                prime: Some(p),
                retries: 2,
                ..LabellingOptions::default()
            },
        )
        .unwrap();
        assert_eq!(l.status(), &LabellingStatus::Unresolved, "p = {p}");
        assert_eq!(l.attempts().len(), 3, "three precisions tried");
        assert_eq!(residues(&l), (Integer::from(plus), Integer::from(minus)));
        assert_eq!(l.sign(), None);
        assert_eq!(l.t5(), None);
        assert!(matches!(
            l.golay_invariants(),
            Err(SteinerError::NotDecided(LabellingStatus::Unresolved))
        ));
        assert!(matches!(
            l.heptad_resolvent(),
            Err(SteinerError::NotDecided(_))
        ));
        assert!(matches!(
            l.frobenius_sign(&Integer::from(5), 1),
            Err(SteinerError::NotDecided(_))
        ));
    }
    // A precision with p^N <= 2 B5 is refused: 2^39 <= 2 * 309225934046 < 2^40.
    let refused = steiner_labelling(
        &f,
        &LabellingOptions {
            prime: Some(2),
            precision: Some(39),
            ..LabellingOptions::default()
        },
    );
    assert!(matches!(
        refused,
        Err(SteinerError::Integrality(
            IntegralityError::PrecisionTooLow { .. }
        ))
    ));
    let l = steiner_labelling(
        &f,
        &LabellingOptions {
            prime: Some(2),
            precision: Some(40),
            ..LabellingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(l.status(), &LabellingStatus::Unresolved);
    // Frobenius signs given T5 (review 1, m3): T5+ - T5- = 67680 = 120 * 564, so at q = 2, 3, 5
    // with N = 1 both candidates match -98317; t5 = 0 at q = 2 matches neither (both candidates
    // are odd); at q = 5, N = 2 the residues 8 and 13 (gp/check_sl.gp) separate them.
    let t = Integer::from(-98317);
    for q in [2i64, 3, 5] {
        assert_eq!(
            frobenius_sign_given_t5(&f, &Integer::from(q), &t, 1).unwrap(),
            FrobeniusSign::Unresolved {
                matching: vec![GolaySign::Plus, GolaySign::Minus]
            },
            "q = {q}"
        );
    }
    assert_eq!(
        frobenius_sign_given_t5(&f, &Integer::from(2), &Integer::zero(), 1).unwrap(),
        FrobeniusSign::Unresolved { matching: vec![] }
    );
    assert_eq!(
        frobenius_sign_given_t5(&f, &Integer::from(5), &t, 2).unwrap(),
        FrobeniusSign::Decided(GolaySign::Plus)
    );
}

#[test]
fn s23_refutes_the_hypothesis() {
    // x^23 - x - 1 at its inert prime 23, default margin: N = 29 (gp/check_sl.gp). Both
    // candidates are the reductions of the PARI/GP and arith.py values at N = 30 (design SL, 7.1)
    // and lie outside B5 = 136069473: Gal(f) is in no conjugate of M23.
    let l = steiner_labelling(&s23(), &opts(23)).unwrap();
    assert_eq!(l.precision(), 29);
    assert_eq!(l.status(), &LabellingStatus::NotInM23);
    assert_eq!(l.t5_bound(), &Integer::from(136069473));
    let m = Integer::from(23).pow(29);
    assert_eq!(
        residues(&l),
        (
            symmetric_residue(&int("-29593503946453640472557674395096324685942"), &m),
            symmetric_residue(&int("26819164821679513709525347975140930751221"), &m)
        )
    );
    assert_eq!(
        residues(&l),
        (
            int("1317082484481734755437783990304108704688"),
            int("-1000362966162323995670564571719459300346")
        )
    );
    assert!(matches!(
        l.golay_invariants(),
        Err(SteinerError::NotDecided(LabellingStatus::NotInM23))
    ));
}

#[test]
fn false_hypothesis_can_decide() {
    // Review 1 of design SL, 1.4 (PARI/GP run_wrong.gp, arith.py with extra_digits = 0,
    // gp/check_sl.gp): x^23 - x - 1 (group S23) at 157 with the minimal rigorous N = 4 gives
    // residues 182056093, 128168772 against B5 = 136069473, so the rule says Decided(-). The
    // decision is relative to the hypothesis Gal(f) = M23, which is false here.
    let f = s23();
    let l = steiner_labelling(
        &f,
        &LabellingOptions {
            prime: Some(157),
            precision: Some(4),
            ..LabellingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        l.status(),
        &LabellingStatus::Decided {
            sign: GolaySign::Minus
        }
    );
    assert_eq!(l.hypothesis(), GaloisHypothesis::M23);
    assert_eq!(
        residues(&l),
        (Integer::from(182056093), Integer::from(128168772))
    );
    // With the default margin (N = 18) neither passes: the hypothesis is refuted.
    let l = steiner_labelling(&f, &opts(157)).unwrap();
    assert_eq!(l.precision(), 18);
    assert_eq!(l.status(), &LabellingStatus::NotInM23);
    assert_eq!(
        residues(&l),
        (
            int("1174593018291759664548746434547509919577"),
            int("-433949046558353058032350371950976022510")
        )
    );
}

#[test]
fn duad_resolvent_three_ways() {
    // x^23 - x - 1 at 23: the orbit-polynomial route, the direct product over the 253 duads
    // (mod 23^3) and the power-sum route of rustmath-polynomials (exact over Q) agree.
    let f = s23();
    let p = Integer::from(23);
    let d = duad_resolvent(&f, &p).unwrap();
    assert_eq!(d.certification(), &Certification::Symmetric);
    assert_eq!(d.coefficients().len(), 254);
    assert_eq!(
        d.bound(),
        &RootBound::fujiwara(&f).unwrap().duad_resolvent_bound()
    );
    assert_eq!(
        d.coefficients(),
        rustmath_polynomials::resolvent::subset_sum_resolvent(&f, 2).as_slice()
    );
    let roots = rustmath_finitefields::FrobeniusRoots::new(&f, &p, 3).unwrap();
    let r = roots.roots();
    let mut values = Vec::new();
    for i in 0..23 {
        for j in i + 1..23 {
            values.push(r[i].clone() + r[j].clone());
        }
    }
    let direct = rustmath_finitefields::orbit_polynomial(&values).unwrap();
    let m = roots.modulus();
    let reduced: Vec<Integer> = d.coefficients().iter().map(|c| c.modulo(m)).collect();
    assert_eq!(reduced, direct);
}

#[test]
fn refusals() {
    let g = cc0();
    // Not monic: the raw CC0 polynomial (monic_associate is the documented way).
    assert!(matches!(
        steiner_labelling(&cc0_raw(), &opts(29)),
        Err(SteinerError::NotMonic)
    ));
    // Degree 22.
    let mut d22 = vec![Integer::zero(); 23];
    d22[0] = Integer::from(-1);
    d22[1] = Integer::from(-1);
    d22[22] = Integer::one();
    assert!(matches!(
        steiner_labelling(&d22, &opts(29)),
        Err(SteinerError::WrongDegree(22))
    ));
    // Composite prime.
    assert!(matches!(
        steiner_labelling(&g, &opts(25)),
        Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(_)))
    ));
    // p < 2 is refused before the precision is computed (review 1 of SL, m1: it panicked).
    for p in [0u64, 1] {
        assert!(matches!(
            steiner_labelling(&g, &opts(p)),
            Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(_)))
        ));
    }
    for p in [-23i64, 0, 1, 25] {
        assert!(matches!(
            duad_resolvent(&s23(), &Integer::from(p)),
            Err(SteinerError::Frobenius(FrobeniusRootsError::Prime(_)))
        ));
    }
    // x^23 + 1 = (x + 1)^23 mod 23: not squarefree.
    let mut x23p1 = vec![Integer::zero(); 24];
    x23p1[0] = Integer::one();
    x23p1[23] = Integer::one();
    assert!(matches!(
        steiner_labelling(&x23p1, &opts(23)),
        Err(SteinerError::Frobenius(
            FrobeniusRootsError::NotSquarefreeModP
        ))
    ));
    // CC0 at 11: factor degrees 2, 7, 14 (Le Lay's even_frobenius certificate).
    assert_eq!(
        steiner_labelling(&g, &opts(11)).unwrap_err(),
        SteinerError::Frobenius(FrobeniusRootsError::NotInertModP {
            factor_degrees: Some(vec![2, 7, 14])
        })
    );
    // A product of two polynomials is never inert: the bounded search says so, and no more.
    let a: Vec<Integer> = {
        let mut v = vec![Integer::zero(); 12];
        v[0] = Integer::from(-1);
        v[1] = Integer::from(-1);
        v[11] = Integer::one();
        v
    };
    let b: Vec<Integer> = {
        let mut v = vec![Integer::zero(); 13];
        v[0] = Integer::from(-1);
        v[1] = Integer::from(-1);
        v[12] = Integer::one();
        v
    };
    let prod = rustmath_polynomials::zx::mul(&a, &b);
    assert_eq!(first_inert_prime(&prod, 3, 200), None);
    assert_eq!(
        steiner_labelling(
            &prod,
            &LabellingOptions {
                search_limit: 200,
                ..LabellingOptions::default()
            }
        )
        .unwrap_err(),
        SteinerError::NoInertPrime { below: 200 }
    );
    // Zero precision is a refused precision.
    assert!(matches!(
        steiner_labelling(
            &g,
            &LabellingOptions {
                prime: Some(29),
                precision: Some(0),
                ..LabellingOptions::default()
            }
        ),
        Err(SteinerError::Integrality(_))
    ));
    // t5_candidates needs 23 roots.
    let roots =
        rustmath_finitefields::FrobeniusRoots::new(&ints(&[1, 1, 0, 1]), &Integer::from(5), 3)
            .unwrap();
    assert!(matches!(
        t5_candidates(&roots),
        Err(SteinerError::WrongDegree(3))
    ));
}

// ------------------------------------------------------------------------------------------
// Fast tests: Le Lay's CC0 certified M23 polynomial
// ------------------------------------------------------------------------------------------

#[test]
fn cc0_certificate_facts() {
    let v = read_json("lelay_specialization_t2_Z.json");
    assert_eq!(v["status"], "CERTIFIED_M23_SPECIALIZATION");
    let f = cc0_raw();
    assert_eq!(f.len(), 24);
    let content = f.iter().fold(Integer::zero(), |g, c| g.gcd(c));
    assert_eq!(content, Integer::one());
    assert_eq!(
        f[23],
        int("239024099105221865598815800284190403153851130391180859371046699194977683")
    );
    // The two modular facts of the certificate, recomputed.
    let cert = &v["modular_certificates"];
    assert_eq!(factor_degrees(&f, 29), vec![23]);
    assert_eq!(
        cert["irreducibility"]["factor_degrees"],
        serde_json::json!([23])
    );
    assert_eq!(factor_degrees(&f, 11), vec![2, 7, 14]);
    assert_eq!(
        cert["even_frobenius"]["factor_degrees"],
        serde_json::json!([2, 7, 14])
    );
    // The monic associate has the same factorization pattern (a is a unit mod 29 and 11), and
    // 29 is its first inert prime (arith.py inert_primes).
    let g = cc0();
    assert_eq!(factor_degrees(&g, 11), vec![2, 7, 14]);
    assert_eq!(first_inert_prime(&g, 3, 10_000), Some(29));
    let r = cc0_ref();
    assert_eq!(
        r["source"]["sha256"],
        "f775aa4c004541f25d121fac999fca9c62e493100f99982cc8253fbfa4445862"
    );
    assert_eq!(f[23], jint(&r["leading_coefficient"]));
}

#[test]
fn cc0_labelling_at_29() {
    let g = cc0();
    let r = cc0_ref();
    let at = &r["labelling"]["29"];
    let l = cc0_at_29();
    assert_eq!(l.prime(), &Integer::from(29));
    assert_eq!(l.root_bound().best(), &jint(&r["fujiwara_best"]));
    assert_eq!(l.t5_bound(), &jint(&r["t5_bound"]));
    assert_eq!(digits(l.t5_bound()), 364);
    assert_eq!(l.precision(), 270);
    assert_eq!(at["N"], 270);
    assert_eq!(l.attempts(), &[270]);
    assert_eq!(
        l.status(),
        &LabellingStatus::Decided {
            sign: GolaySign::Minus
        }
    );
    assert_eq!(at["sign"], -1);
    assert_eq!(l.t5(), Some(&jint(&r["T5"])));
    assert_eq!(digits(l.t5().unwrap()), 356);
    assert_eq!(l.rejected_t5(), Some(&jint(&at["T5_plus"])));
    assert_eq!(digits(l.rejected_t5().unwrap()), 395);
    // The minimal rigorous precision N = 250 decides too, with the same T5; its rejected residue
    // is the reduction of the one at 270.
    let m = steiner_labelling(
        &g,
        &LabellingOptions {
            prime: Some(29),
            precision: Some(250),
            ..LabellingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(precision_for(&Integer::from(29), l.t5_bound(), 0), 250);
    assert_eq!(m.sign(), Some(GolaySign::Minus));
    assert_eq!(m.t5(), l.t5());
    let r250 = &r["labelling"]["29_minimal"];
    assert_eq!(r250["N"], 250);
    assert_eq!(m.rejected_t5(), Some(&jint(&r250["T5_plus"])));
    assert_eq!(
        m.rejected_t5().unwrap(),
        &symmetric_residue(l.rejected_t5().unwrap(), &Integer::from(29).pow(250))
    );
}

#[test]
fn cc0_labelling_at_67() {
    // The second inert prime: the labelling there is unrelated to the one at 29 (sign +), and
    // T5 is the same integer (arith.py; gp/check_sl.gp for both residues).
    let g = cc0();
    let r = cc0_ref();
    let at = &r["labelling"]["67"];
    let l = steiner_labelling(&g, &opts(67)).unwrap();
    assert_eq!(l.precision(), 216);
    assert_eq!(at["N"], 216);
    assert_eq!(l.sign(), Some(GolaySign::Plus));
    assert_eq!(l.t5(), Some(&jint(&r["T5"])));
    assert_eq!(l.rejected_t5(), Some(&jint(&at["T5_minus"])));
    assert_eq!(digits(l.rejected_t5().unwrap()), 394);
}

#[test]
fn cc0_golay_invariants() {
    let g = cc0();
    let r = cc0_ref();
    let l = cc0_at_29();
    let j = l.golay_invariants().unwrap();
    assert_eq!(j.j5.value(), &jint(&r["J5"]));
    assert_eq!(j.j7.value(), &jint(&r["J7"]));
    assert_eq!((digits(j.j5.value()), digits(j.j7.value())), (353, 491));
    assert_eq!(j.j5.bound(), &jint(&r["j5_bound"]));
    assert_eq!(j.j7.bound(), &jint(&r["j7_bound"]));
    assert_eq!(j.j5.precision(), 344);
    assert_eq!(j.j7.prime(), &Integer::from(29));
    assert_eq!(
        j.j5.certification(),
        &Certification::Labelling {
            hypothesis: GaloisHypothesis::M23,
            sign: GolaySign::Minus,
            prime: Integer::from(29),
            precision: 270
        }
    );
    // T5 = 120 J5 + (a symmetric function of the roots), design SL 2.7: an identity independent
    // of all p-adic code.
    assert_eq!(
        l.t5().unwrap().clone(),
        Integer::from(120) * j.j5.value().clone() + t5_minus_120_j5(&g)
    );
}

#[test]
fn cc0_frobenius_signs() {
    // Both references and the f1 rows of the owner's frobenius_signs.json (the CC0 polynomial and
    // f1 define the same field, HJLPPZ Example 1.2): Frob_73 and Frob_109 have sign -.
    let r = cc0_ref();
    let signs = read_json("frobenius_signs.json");
    let l = cc0_at_29();
    for q in [73u64, 109] {
        let s = l.frobenius_sign(&Integer::from(q), 1).unwrap();
        assert_eq!(s, FrobeniusSign::Decided(GolaySign::Minus), "q = {q}");
        assert_eq!(r["frobenius_signs"][q.to_string()], -1);
        assert_eq!(signs["f1"][q.to_string()], -1);
    }
}

#[test]
fn trace_route_equals_direct_sums() {
    // On the CC0 roots at a small precision, the direct sums over the 253 heptads (fixed by the
    // Frobenius, so constants) equal the trace route of t5_candidates.
    let g = cc0();
    let roots = rustmath_finitefields::FrobeniusRoots::new(&g, &Integer::from(29), 40).unwrap();
    let cands = t5_candidates(&roots).unwrap();
    for (s, v) in cands {
        assert_eq!(t5_direct(&roots, s), v, "sign {s}");
    }
}

// ------------------------------------------------------------------------------------------
// Opt-in, release: more CC0 primes
// ------------------------------------------------------------------------------------------

#[test]
#[ignore = "release; the twelve inert primes of the CC0 polynomial below 900"]
fn full_cc0_frobenius_signs() {
    let g = cc0();
    let r = cc0_ref();
    let signs = read_json("frobenius_signs.json");
    let l = steiner_labelling(&g, &opts(29)).unwrap();
    let primes: Vec<u64> = r["inert_primes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_u64().unwrap())
        .collect();
    assert_eq!(primes.len(), 12);
    let mut found = Vec::new();
    let mut p = 3;
    while found.len() < 12 {
        p = first_inert_prime(&g, p, 1000).expect("twelve inert primes below 1000");
        found.push(p);
        p += 1;
    }
    assert_eq!(found, primes);
    for q in primes {
        let want = r["frobenius_signs"][q.to_string()].as_i64().unwrap();
        assert_eq!(
            signs["f1"][q.to_string()].as_i64().unwrap(),
            want,
            "f1 row {q}"
        );
        let got = l.frobenius_sign(&Integer::from(q), 1).unwrap();
        assert_eq!(
            got,
            FrobeniusSign::Decided(GolaySign::from_i8(want as i8).unwrap()),
            "q = {q}"
        );
    }
}

#[test]
#[ignore = "release; the CC0 invariants at the second prime 67"]
fn full_cc0_invariants_at_67() {
    let g = cc0();
    let r = cc0_ref();
    assert_eq!(r["J_at_67_equal"], true);
    let l = steiner_labelling(&g, &opts(67)).unwrap();
    let j = l.golay_invariants().unwrap();
    assert_eq!(j.j5.value(), &jint(&r["J5"]));
    assert_eq!(j.j7.value(), &jint(&r["J7"]));
}

// ------------------------------------------------------------------------------------------
// Opt-in: HJLPPZ's f1 and f2, read at run time from $M23ISGALOIS
// ------------------------------------------------------------------------------------------

/// The checkout of `shaowuz/m23isgalois`; panics without `M23ISGALOIS` (never vacuous).
fn require_m23isgalois() -> PathBuf {
    std::env::var_os("M23ISGALOIS")
        .map(PathBuf::from)
        .expect("set M23ISGALOIS to a checkout of shaowuz/m23isgalois")
}

/// Read `<name> += (<int>)*x^<i>;` from `polynomial_<name>.gp` and check the file's own
/// fingerprint line `<name>_expect = [degree, nonzero terms, total digits, maximal digits,
/// content, f(3) mod 2^61 - 1]`.
fn read_f(name: &str) -> Vec<Integer> {
    let path = require_m23isgalois().join(format!("polynomial_{name}.gp"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut f = vec![Integer::zero(); 24];
    let mut expect: Option<Vec<Integer>> = None;
    let prefix = format!("{name} += (");
    let fp = format!("{name}_expect = [");
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&prefix) {
            let (c, tail) = rest.split_once(")*x^").expect("term");
            let i: usize = tail.trim_end_matches(';').parse().expect("exponent");
            f[i] = f[i].clone() + int(c);
        } else if let Some(rest) = line.strip_prefix(&fp) {
            let body = rest.trim_end_matches("];");
            expect = Some(body.split(',').map(|s| int(s.trim())).collect());
        }
    }
    let expect = expect.expect("fingerprint line");
    let nz: Vec<&Integer> = f.iter().filter(|c| !c.is_zero()).collect();
    let total: usize = nz.iter().map(|c| digits(c)).sum();
    let max = nz.iter().map(|c| digits(c)).max().unwrap();
    let content = f.iter().fold(Integer::zero(), |g, c| g.gcd(c));
    let m61 = Integer::from((1u64 << 61) - 1);
    let at3 = f
        .iter()
        .rev()
        .fold(Integer::zero(), |acc, c| acc * Integer::from(3) + c.clone())
        .modulo(&m61);
    let got = vec![
        Integer::from(23),
        Integer::from(nz.len() as i64),
        Integer::from(total as i64),
        Integer::from(max as i64),
        content,
        at3,
    ];
    assert_eq!(got, expect, "{name}: fingerprint");
    f
}

fn golay_ref(name: &str) -> Value {
    read_json("golay_resolvents_no_f.json")[name].clone()
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_six_integers_and_signs() {
    // golay_resolvents.json (owner, FLINT) for T5, J5, J7 and the sign; the rejected residues
    // and the bounds from arith.py rerun here (python-flint 0.9.0) and PARI/GP (design SL
    // run_f1.gp, run_f2.gp; review 1 rev.gp).
    for (name, best, b5, bj5, bj7, n, rejected) in [
        (
            "f2",
            2157,
            "5778453176946",
            "7220058114",
            "97634149933",
            30,
            "-29622039743232607524836197957123800868867866",
        ),
        (
            "f1",
            3473,
            "62529476039478",
            "78129291179",
            "2738950710192",
            31,
            "-883496073094568376895161675988763338141335547",
        ),
    ] {
        let f = read_f(name);
        let d = golay_ref(name);
        assert_eq!(d["labelling_prime"], 29);
        let l = steiner_labelling(&f, &LabellingOptions::default()).unwrap();
        assert_eq!(l.prime(), &Integer::from(29));
        assert_eq!(l.root_bound().best(), &Integer::from(best));
        assert_eq!(l.t5_bound(), &int(b5));
        assert_eq!(l.root_bound().j5_bound(), int(bj5));
        assert_eq!(l.root_bound().j7_bound(), int(bj7));
        assert_eq!(l.precision(), n, "{name}");
        assert_eq!(l.sign().map(GolaySign::to_i8), Some(-1), "{name}");
        assert_eq!(d["sign_at_prime"], -1);
        assert_eq!(l.t5(), Some(&jint(&d["T5"])), "{name}");
        assert_eq!(l.rejected_t5(), Some(&int(rejected)), "{name}");
        assert!(digits(l.rejected_t5().unwrap()) > 40);
        let j = l.golay_invariants().unwrap();
        assert_eq!(j.j5.value(), &jint(&d["J5"]), "{name}");
        assert_eq!(j.j7.value(), &jint(&d["J7"]), "{name}");
        eprintln!(
            "{name}: p = 29, N = {n}, sign -, T5 = {}, J5 = {}, J7 = {}, rejected T5+ ({} digits)",
            l.t5().unwrap(),
            j.j5.value(),
            j.j7.value(),
            digits(l.rejected_t5().unwrap())
        );
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_f2_minimal_precision() {
    // The symmetric reductions of the reference residues at N = 30 and 31 (arith.py, PARI/GP,
    // review 1 Section 2).
    let f2 = read_f("f2");
    let at = |n: u32| {
        steiner_labelling(
            &f2,
            &LabellingOptions {
                prime: Some(29),
                precision: Some(n),
                ..LabellingOptions::default()
            },
        )
    };
    assert!(matches!(
        at(8),
        Err(SteinerError::Integrality(
            IntegralityError::PrecisionTooLow { .. }
        ))
    ));
    let l9 = at(9).unwrap();
    assert_eq!(l9.status(), &LabellingStatus::Unresolved);
    assert_eq!(residues(&l9), (int("-3209424281433"), int("-1059840")));
    let l10 = at(10).unwrap();
    assert_eq!(l10.sign(), Some(GolaySign::Minus));
    assert_eq!(l10.rejected_t5(), Some(&int("-206309467943599")));
    // One retry separates them at N = 9 + 7 = 16 (10 more digits).
    let lr = steiner_labelling(
        &f2,
        &LabellingOptions {
            prime: Some(29),
            precision: Some(9),
            retries: 1,
            ..LabellingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(lr.attempts(), &[9, 16]);
    assert_eq!(lr.sign(), Some(GolaySign::Minus));
    assert_eq!(lr.t5(), Some(&int("-1059840")));
    let f1 = read_f("f1");
    let l = steiner_labelling(
        &f1,
        &LabellingOptions {
            prime: Some(29),
            precision: Some(10),
            ..LabellingOptions::default()
        },
    )
    .unwrap();
    assert_eq!(l.sign(), Some(GolaySign::Minus));
    assert_eq!(l.rejected_t5(), Some(&int("-114831914638610")));
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_heptad_resolvents() {
    for (name, n) in [("f2", 360), ("f1", 395)] {
        let f = read_f(name);
        let l = steiner_labelling(&f, &LabellingOptions::default()).unwrap();
        let t = std::time::Instant::now();
        let h = l.heptad_resolvent().unwrap();
        eprintln!(
            "{name}: heptad resolvent at N = {} in {:.2} s",
            h.precision(),
            t.elapsed().as_secs_f64()
        );
        assert_eq!(h.precision(), n);
        assert_eq!(
            h.coefficients(),
            jints(&golay_ref(name)["R253_heptad"]).as_slice(),
            "{name}"
        );
        assert!(matches!(h.certification(), Certification::Labelling { .. }));
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_duad_resolvents() {
    // Three references: the owner's data, the p-adic orbit route, the power-sum route of
    // rustmath-polynomials (exact over Q).
    for (name, n) in [("f2", 267), ("f1", 302)] {
        let f = read_f(name);
        let want = jints(&golay_ref(name)["R253_duad"]);
        let l = steiner_labelling(&f, &LabellingOptions::default()).unwrap();
        let t = std::time::Instant::now();
        let d = l.duad_resolvent().unwrap();
        eprintln!(
            "{name}: duad resolvent at N = {} in {:.2} s",
            d.precision(),
            t.elapsed().as_secs_f64()
        );
        assert_eq!(d.precision(), n);
        assert_eq!(d.coefficients(), want.as_slice(), "{name}");
        assert_eq!(d.certification(), &Certification::Symmetric);
        assert_eq!(duad_resolvent(&f, &Integer::from(29)).unwrap(), d);
        let t = std::time::Instant::now();
        let s = rustmath_polynomials::resolvent::subset_sum_resolvent(&f, 2);
        eprintln!(
            "{name}: subset_sum_resolvent in {:.2} s",
            t.elapsed().as_secs_f64()
        );
        assert_eq!(s, want, "{name}");
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_second_prime() {
    // verified_at_second_prime of golay_resolvents.json: f1 at 73, f2 at 223. The labelling
    // there is unrelated to the one at 29; the six integers and the four resolvents agree.
    for (name, q) in [("f1", 73u64), ("f2", 223)] {
        let f = read_f(name);
        let d = golay_ref(name);
        assert_eq!(d["verified_at_second_prime"], serde_json::json!([q, true]));
        let l = steiner_labelling(&f, &opts(q)).unwrap();
        assert_eq!(l.sign(), Some(GolaySign::Minus), "{name} at {q}");
        assert_eq!(l.t5(), Some(&jint(&d["T5"])));
        let j = l.golay_invariants().unwrap();
        assert_eq!(j.j5.value(), &jint(&d["J5"]));
        assert_eq!(j.j7.value(), &jint(&d["J7"]));
        let h = l.heptad_resolvent().unwrap();
        assert_eq!(h.coefficients(), jints(&d["R253_heptad"]).as_slice());
        let du = l.duad_resolvent().unwrap();
        assert_eq!(du.coefficients(), jints(&d["R253_duad"]).as_slice());
        eprintln!(
            "{name} at {q}: N = {}, N_J = {}, N_heptad = {}, N_duad = {}",
            l.precision(),
            j.j5.precision(),
            h.precision(),
            du.precision()
        );
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_frobenius_signs() {
    // All 80 rows of frobenius_signs.json (PARI/GP run_signs.gp agrees), at N = 1.
    let signs = read_json("frobenius_signs.json");
    for name in ["f1", "f2"] {
        let f = read_f(name);
        let l = steiner_labelling(&f, &opts(29)).unwrap();
        let rows = signs[name].as_object().unwrap();
        assert_eq!(rows.len(), 40);
        for (q, s) in rows {
            let got = l.frobenius_sign(&int(q), 1).unwrap();
            let want = GolaySign::from_i8(s.as_i64().unwrap() as i8).unwrap();
            assert_eq!(got, FrobeniusSign::Decided(want), "{name} at {q}");
        }
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_t5_j5_identity() {
    for name in ["f1", "f2"] {
        let f = read_f(name);
        let d = golay_ref(name);
        assert_eq!(
            jint(&d["T5"]),
            Integer::from(120) * jint(&d["J5"]) + t5_minus_120_j5(&f),
            "{name}"
        );
    }
}

#[test]
#[ignore = "needs M23ISGALOIS; release"]
fn env_stability_cross_check() {
    // A cross-check, not a certificate: J5, J7 by direct sums over the 253 heptads, and the
    // heptad resolvent by orbit polynomials rebuilt here, at seven more p-adic digits, agree with
    // the certified values of f2.
    let f = read_f("f2");
    let l = steiner_labelling(&f, &opts(29)).unwrap();
    let sign = l.sign().unwrap();
    let j = l.golay_invariants().unwrap();
    let roots = l.roots().with_precision(j.j5.precision() + 7).unwrap();
    let ring = roots.ring();
    let r = roots.roots();
    let (mut e5, mut e7) = (ring.zero(), ring.zero());
    for &h in heptads(sign) {
        let pts = support(h);
        let mut e = vec![ring.one()];
        e.extend((0..7).map(|_| ring.zero()));
        for &i in &pts {
            for k in (1..=7).rev() {
                e[k] = e[k].clone() + e[k - 1].clone() * r[i].clone();
            }
        }
        e5 = e5 + e[5].clone();
        e7 = e7 + e[7].clone();
    }
    let m = roots.modulus();
    assert_eq!(
        &symmetric_residue(&e5.base_ring_value().unwrap(), m),
        j.j5.value()
    );
    assert_eq!(
        &symmetric_residue(&e7.base_ring_value().unwrap(), m),
        j.j7.value()
    );
    let h = l.heptad_resolvent().unwrap();
    let roots = l.roots().with_precision(h.precision() + 7).unwrap();
    let m = roots.modulus().clone();
    let mut total = vec![Integer::one()];
    for &rep in rustmath_coding::cyclic_golay::heptad_orbit_representatives(sign) {
        let values: Vec<_> = (0..23)
            .map(|k| {
                support(rustmath_coding::cyclic_golay::rotate(rep, k))
                    .into_iter()
                    .fold(roots.ring().zero(), |acc, i| acc + roots.roots()[i].clone())
            })
            .collect();
        let o = rustmath_finitefields::orbit_polynomial(&values).unwrap();
        total = rustmath_polynomials::zx::mul(&total, &o)
            .into_iter()
            .map(|c| c.modulo(&m))
            .collect();
    }
    let again: Vec<Integer> = total.iter().map(|c| symmetric_residue(c, &m)).collect();
    assert_eq!(again.as_slice(), h.coefficients());
}
