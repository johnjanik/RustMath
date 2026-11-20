//! The Frobenius module `Jac[2]` from the Frobenius polynomial (`rustmath_curves::weil`,
//! `TwoTorsionAnalysis`; design C57, `docs/workstreams/m23_golay/REQUESTS.md` C-5; MAGMA ch. 114
//! `LPolynomial(C, m)`, `ClassGroupAbelianInvariants`, digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`).
//!
//! Reference data in `tests/data/two_torsion/` (README there: tool versions, seeds, commands):
//! - `hyperelliptic_jac2.txt`: 3098 curves `y^2 = f(x)` of genus 1 to 4 over `F_p`, `3 <= p <= 47`,
//!   with `P` from PARI/GP `hyperellcharpoly` and the exact Frobenius module `Jac[2]` (the even
//!   subsets of the branch points modulo the whole set, Frobenius acting by the cycle type of
//!   `f mod p`): its order and `dim ker (F - 1)^j` (`hyper_truth.gp`).
//! - `gl42_classes.txt`: the 14 classes of `GL_4(2)` (GAP 4.12.1, `gl42_classes.g`), equal to
//!   the `classes` of the owner's `data/a8_frobenius_predictions.json` (johnjanik/M23 at
//!   `e0d25b8`).

mod common;

use common::m23::{canonical_form, cycle_type_on_points, matrix_order, GL42_CLASSES};
use rustmath_curves::weil::{f2_poly_string, FrobeniusPolynomial, TwoTorsionOrder};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;
use std::path::Path;

fn data(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/two_torsion")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The integers of a bracketed list `[a, b, c]`.
fn list(s: &str) -> Vec<i64> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse().expect("integer"))
        .collect()
}

/// One line of `hyperelliptic_jac2.txt`: `p g [P] [cycle type] order [kernel dims]`.
struct Curve {
    p: i64,
    g: usize,
    poly: Vec<i64>,
    order: u64,
    kers: Vec<usize>,
}

fn curves() -> Vec<Curve> {
    data("hyperelliptic_jac2.txt")
        .lines()
        .map(|line| {
            let open: Vec<usize> = line.match_indices('[').map(|m| m.0).collect();
            let close: Vec<usize> = line.match_indices(']').map(|m| m.0).collect();
            assert_eq!((open.len(), close.len()), (3, 3), "{line}");
            let head: Vec<&str> = line[..open[0]].split_whitespace().collect();
            let order = line[close[1] + 1..open[2]].trim().parse().expect("order");
            Curve {
                p: head[0].parse().expect("p"),
                g: head[1].parse().expect("g"),
                poly: list(&line[open[0]..=close[0]]),
                order,
                kers: list(&line[open[2]..=close[2]])
                    .into_iter()
                    .map(|k| k as usize)
                    .collect(),
            }
        })
        .collect()
}

#[test]
fn sound_on_hyperelliptic_jacobians() {
    // Soundness (design C57 2.6): the true type of Jac[2] is a candidate, so the true order and
    // the true kernel dimensions at the eigenvalue 1 occur together in one candidate, and every
    // decided order is the true one. The number decided (1468 of 3098, the design's prototype
    // and its review) pins the strength of the rule.
    let cs = curves();
    assert_eq!(cs.len(), 3098);
    let mut decided = 0;
    for c in &cs {
        let coeffs: Vec<Integer> = c.poly.iter().map(|&x| Integer::from(x)).collect();
        let fp = FrobeniusPolynomial::from_coefficients(&Integer::from(c.p), coeffs).unwrap();
        assert_eq!(fp.genus(), c.g);
        let a = fp.two_torsion_analysis().unwrap();
        let one = a
            .factors
            .iter()
            .position(|f| fp_factor::trim(&f.f) == vec![1, 1]);
        let kers_of = |t: &rustmath_curves::weil::ModuleType| -> Vec<usize> {
            (1..=2 * c.g)
                .map(|j| match one {
                    Some(i) => t.0[i].iter().map(|&x| (x as usize).min(j)).sum(),
                    None => 0,
                })
                .collect()
        };
        assert!(
            a.candidates
                .iter()
                .any(|t| a.order_of(t) == c.order && kers_of(t) == c.kers),
            "p = {}, P = {:?}: the true type is not a candidate",
            c.p,
            c.poly
        );
        match fp.two_torsion_frobenius_order() {
            TwoTorsionOrder::DeterminedByP(n)
            | TwoTorsionOrder::DeterminedByTwoRank { order: n, .. } => {
                assert_eq!(n, c.order, "p = {}, P = {:?}", c.p, c.poly);
                decided += 1;
            }
            TwoTorsionOrder::Unresolved { candidates, .. } => {
                assert!(
                    candidates.contains(&c.order),
                    "p = {}, P = {:?}",
                    c.p,
                    c.poly
                );
                assert!(candidates.len() >= 2);
            }
        }
    }
    assert_eq!(decided, 1468);
}

#[test]
fn gl42_classes() {
    // The table of the M23 layer equals GAP's (every column), the classes add up to
    // |GL_4(2)| = 20160, and the rational canonical form of each class gives, by brute force on
    // the 15 points of PG(3, 2), the class's order, fixed dimension and cycle type.
    let rows: Vec<Vec<String>> = data("gl42_classes.txt")
        .lines()
        .map(|l| l.split('|').map(|s| s.trim().to_string()).collect())
        .collect();
    assert_eq!(rows.len(), 14);
    assert_eq!(GL42_CLASSES.len(), 14);
    let mut total = 0;
    for r in &rows {
        let cp: Vec<i64> = r[0].split(',').map(|x| x.parse().unwrap()).collect();
        let order: u64 = r[1].parse().unwrap();
        let size: u64 = r[2].parse().unwrap();
        let kers: Vec<usize> = r[3].split(',').map(|x| x.parse().unwrap()).collect();
        let c = GL42_CLASSES
            .iter()
            .find(|c| {
                c.charpoly == f2_poly_string(&cp) && c.order == order && c.dim_fixed == kers[0]
            })
            .unwrap_or_else(|| panic!("no class {r:?}"));
        assert_eq!(c.class_size, size);
        assert_eq!(c.cycle_type_15, r[4]);
        assert_eq!(r[4], r[5], "points and hyperplanes");
        assert_eq!(c.cycle_type_8, r[6]);
        total += size;
    }
    assert_eq!(total, 20160);
    // Every module type of degree 4 over F_2 with nonzero constant term: its canonical form is
    // one of the 14 classes, and two types never give the same class.
    let mut seen = Vec::new();
    for bits in 0u32..16 {
        let cp: Vec<i64> = (0..5)
            .map(|i| if i == 4 || (bits >> i) & 1 == 1 { 1 } else { 0 })
            .collect();
        if cp[0] == 0 {
            continue;
        }
        for parts in common::m23::module_types(&cp) {
            let m = canonical_form(&parts);
            let order = matrix_order(&m, 4);
            let fixed = (0u32..16)
                .filter(|&v| v != 0 && common::m23::apply(&m, v) == v)
                .count();
            let dim_fixed = (fixed + 1).trailing_zeros() as usize;
            let c = GL42_CLASSES
                .iter()
                .position(|c| {
                    c.charpoly == f2_poly_string(&cp)
                        && c.order == order
                        && c.dim_fixed == dim_fixed
                })
                .unwrap_or_else(|| panic!("no class for {parts:?}"));
            assert_eq!(
                cycle_type_on_points(&m, 4),
                GL42_CLASSES[c].cycle_type_15,
                "{parts:?}"
            );
            assert!(!seen.contains(&c), "two types give class {c}");
            seen.push(c);
        }
    }
    assert_eq!(seen.len(), 14);
}
