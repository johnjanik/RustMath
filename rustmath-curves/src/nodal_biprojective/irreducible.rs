//! Absolute irreducibility of `F mod p` for a curve in `P^1 x P^1`: a hypothesis of the
//! reduction lemma behind every Decided status of [`certify`](fn@super::certify).
//!
//! Specification: MAGMA Handbook ch. 114 *Algebraic Curves* (digest
//! `rustmath-lava/docs/Chapter_114_Algebraic_Curves.md`), 114.8.1 `IsAbsolutelyIrreducible`;
//! ch. 23 *Univariate Polynomial Rings* (digest `Chapter_23_Univariate_Polynomial_Rings.md`) for
//! the factorization of the fibres over `F_q`. MAGMA decides absolute irreducibility by an
//! absolute factorization. This module gives a sufficient criterion and says `NotShown` when it
//! does not apply: it never claims reducibility except for a line component, which it exhibits.
//!
//! Reference: the owner's paper (johnjanik/M23 at `2d11038`, `paper/golay_m23.tex`), Lemma 9.1
//! (`\label{lem:reduction}`): if `F` is primitive of bidegree `(a, b)`, `F mod p` has bidegree
//! `(a, b)` and is absolutely irreducible, and the delta invariants of its singular points over
//! `F_p-bar` add up to `(a - 1)(b - 1) - g`, then the curve has good reduction at `p` and its
//! reduction is the normalization of `F mod p = 0`. The paper's criterion: "absolute
//! irreducibility follows from a fibre `F(T, v0)` that is irreducible of degree 4, because the
//! degrees 4 and 23 are coprime".
//!
//! Criterion (F-9 design, Section 2). Let `F` have bidegree `(a, b)`, `a, b >= 1`, over `F_p`.
//!
//! 1. Line components. If the coefficients of `F` as a form in `X` have a common factor (a form
//!    in `Y` alone; `Y = infinity` when no coefficient has degree `b`), or the same with `X` and
//!    `Y` exchanged, then `F mod p` contains a line of a ruling, or drops its bidegree. This is
//!    refused as `DegenerateModel(FibreComponent)`, as `singular_locus` refuses a horizontal
//!    line.
//! 2. A fibre over a closed point `v0` of degree `d` of the base line that is irreducible of full
//!    degree over `F_(p^d)` shows that `F` is irreducible over `F_(p^d)`, hence over `F_p`.
//!    Proof: a factor of a bihomogeneous form is bihomogeneous; a factorization `F = G H` over
//!    `F_(p^d)` restricts to the fibre as `G(X, v0) H(X, v0)`, binary forms of degrees
//!    `a1 + a2 = a`, both nonzero; irreducibility forces `a1 = 0`, so `G` is a form in `Y`
//!    alone dividing every coefficient (over `F_p` too, since a gcd does not depend on the
//!    field), which step 1 excludes.
//! 3. Over `F_p-bar` an `F_p`-irreducible `F` is `c G_0 ... G_(m-1)` with `G_i = sigma^i G_0`
//!    pairwise non-associate (`sigma` the Frobenius; one orbit since the product over an orbit
//!    is defined over `F_p`; multiplicity one since `F_p` is perfect), all of bidegree
//!    `(a/m, b/m)`, so `m | gcd(a, b)`. Over `F_(p^d)` the factors fall into `gcd(m, d)` orbits
//!    of `sigma^d`, so the fibre of step 2 also gives `gcd(m, d) = 1`. If every prime factor of
//!    `gcd(a, b)` divides `d` (`gcd(a, b) = 1` in particular), then `m = 1`. Otherwise a smooth
//!    `F_p`-rational point `P` gives `m = 1`: `P` lies on some `G_j`, hence on all of them
//!    (`G_(j+1)(P) = sigma(G_j(P))` since `P` is fixed by `sigma`), and a point on two
//!    components of `F = G_0 R` has `grad F(P) = R(P) grad G_0(P) + G_0(P) grad R(P) = 0`.
//!
//! The search is bounded: the closed points of degree `<= MAX_BASE_DEGREE` of each base line, at
//! most `MAX_FIBRES_PER_DEGREE` of them per ruling and degree (so for `p >= 64` the rational
//! points tried are `0..63` and infinity is not tried). The pass for a smooth point is complete
//! over `P^1 x P^1 (F_p)`. A search that finds nothing returns `NotShown`.

use super::residue::{eval_fp_poly, factor_degrees, rdeg, PolyField};
use super::{CountError, Degeneracy, Fibration, FormModP};
use rustmath_polynomials::fp_factor as fp;
use std::fmt;

/// The largest residue degree of the base points tried.
pub const MAX_BASE_DEGREE: usize = 4;
/// The number of base points tried per ruling and residue degree.
pub const MAX_FIBRES_PER_DEGREE: usize = 64;

/// A closed point of the base line of a fibration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BasePoint {
    /// A point of `F_p`.
    Rational(u64),
    /// The point at infinity.
    Infinity,
    /// The root of `g`, monic irreducible over `F_p` of degree `>= 2` (ascending coefficients).
    Closed(Vec<u64>),
}

/// A fibre that is irreducible of full degree over the residue field of its base point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrreducibleFibre {
    /// `FibresInX`: the fibre over a point of the `Y`-line, a form of degree `dx` in `X`;
    /// `FibresInY`: over a point of the `X`-line, of degree `dy` in `Y`.
    pub fibration: Fibration,
    pub base: BasePoint,
}

impl IrreducibleFibre {
    /// The degree of the base point over `F_p`.
    pub fn residue_degree(&self) -> usize {
        match &self.base {
            BasePoint::Closed(g) => g.len() - 1,
            _ => 1,
        }
    }
}

impl fmt::Display for IrreducibleFibre {
    /// `x-fibre over y = 6`, `x-fibre over y = inf`, `y-fibre over x = 2`,
    /// `x-fibre over the point g(y) = [2, 0, 1] of degree 2`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (base, fibre) = match self.fibration {
            Fibration::FibresInX => ("y", "x"),
            Fibration::FibresInY => ("x", "y"),
        };
        match &self.base {
            BasePoint::Rational(v) => write!(f, "{fibre}-fibre over {base} = {v}"),
            BasePoint::Infinity => write!(f, "{fibre}-fibre over {base} = inf"),
            BasePoint::Closed(g) => write!(
                f,
                "{fibre}-fibre over the point g({base}) = {g:?} of degree {}",
                g.len() - 1
            ),
        }
    }
}

/// Why irreducibility over `F_p` gives absolute irreducibility (module header, step 3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AbsoluteWitness {
    /// Every prime factor of `gcd = gcd(dx, dy)` divides the residue degree of the fibre's base
    /// point; in particular `gcd = 1`.
    Bidegree { gcd: usize },
    /// A smooth `F_p`-rational point `(x, y)` (`None` for infinity).
    SmoothRationalPoint { x: Option<u64>, y: Option<u64> },
}

/// Evidence that `F mod p` is absolutely irreducible: a fibre that shows irreducibility over
/// `F_p` (step 2) and the reason it is absolute (step 3). Only this module builds it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct IrreducibilityEvidence {
    pub fibre: IrreducibleFibre,
    pub absolute: AbsoluteWitness,
}

impl fmt::Display for IrreducibilityEvidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.absolute {
            AbsoluteWitness::Bidegree { gcd: 1 } => write!(
                f,
                "{} irreducible of full degree; gcd(dx, dy) = 1",
                self.fibre
            ),
            AbsoluteWitness::Bidegree { gcd } => write!(
                f,
                "{} irreducible of full degree; gcd(dx, dy) = {gcd}, settled by the residue degree {}",
                self.fibre,
                self.fibre.residue_degree()
            ),
            AbsoluteWitness::SmoothRationalPoint { x, y } => {
                let s = |v: &Option<u64>| v.map_or("inf".to_string(), |c| c.to_string());
                write!(
                    f,
                    "{} irreducible of full degree; smooth F_p-point ({}, {})",
                    self.fibre,
                    s(x),
                    s(y)
                )
            }
        }
    }
}

/// Why absolute irreducibility is not shown. Neither case says that `F mod p` is reducible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrreducibilityGap {
    /// No fibre over the closed points searched (degree `<= MAX_BASE_DEGREE`, the first
    /// `MAX_FIBRES_PER_DEGREE` per ruling and degree) is irreducible of full degree.
    NoIrreducibleFibre,
    /// `F` is irreducible over `F_p` (by `fibre`), `gcd(dx, dy) = gcd > 1`, there is no smooth
    /// `F_p`-rational point, and no irreducible fibre was found over a point of a degree that
    /// settles `gcd` (step 3).
    NoSmoothRationalPoint { fibre: IrreducibleFibre, gcd: usize },
}

impl fmt::Display for IrreducibilityGap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrreducibilityGap::NoIrreducibleFibre => write!(
                f,
                "no irreducible fibre among the first {MAX_FIBRES_PER_DEGREE} closed points of \
                 each degree <= {MAX_BASE_DEGREE} in either ruling"
            ),
            IrreducibilityGap::NoSmoothRationalPoint { fibre, gcd } => write!(
                f,
                "irreducible over F_p by the {fibre}; gcd(dx, dy) = {gcd} and no smooth F_p-point"
            ),
        }
    }
}

/// The result of [`absolute_irreducibility`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Irreducibility {
    Absolute(IrreducibilityEvidence),
    NotShown(IrreducibilityGap),
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `n` without the prime factors it shares with `d`.
fn coprime_part(mut n: usize, d: usize) -> usize {
    loop {
        let g = gcd_usize(n, d);
        if g == 1 {
            return n;
        }
        n /= g;
    }
}

/// `Err(Shape)` unless both entries of the bidegree are positive.
pub(crate) fn check_bidegree(f: &FormModP) -> Result<(), CountError> {
    let (dx, dy) = f.bidegree();
    if dx == 0 || dy == 0 {
        return Err(CountError::Shape(format!(
            "bidegree ({dx}, {dy}): both degrees must be positive"
        )));
    }
    Ok(())
}

/// A line of a ruling: the place of the first irreducible factor of the gcd `g` (a place in
/// the sense of `singular.rs`, `g` monic irreducible).
fn line_place(var: &str, g: &[i64], p: i64) -> CountError {
    let first = fp::factor_with_multiplicity(g, p)
        .into_iter()
        .next()
        .map(|(h, _)| h)
        .expect("a gcd of positive degree has a factor");
    CountError::DegenerateModel(Degeneracy::FibreComponent {
        place: format!("{var}-place g({var}) = {first:?} (ascending)"),
    })
}

/// Step 1: refuse a factor in one variable alone (a line of a ruling, including the lines at
/// infinity). Lemma 9.1's "F mod p keeps its bidegree" is the absence of the lines at infinity.
pub(crate) fn check_line_components(f: &FormModP) -> Result<(), CountError> {
    let p = f.p() as i64;
    let (dx, dy) = f.bidegree();
    let c = f.coeffs();
    // c[i] is the polynomial in Y of the coefficient of X^i.
    let mut g: Vec<i64> = vec![0];
    for row in c {
        g = fp::gcd(&g, &row.iter().map(|&v| v as i64).collect::<Vec<_>>(), p);
    }
    if fp::degree(&g) > 0 {
        return Err(line_place("y", &g, p));
    }
    if c.iter().all(|row| row[dy] == 0) {
        return Err(CountError::DegenerateModel(Degeneracy::FibreComponent {
            place: "y = infinity".into(),
        }));
    }
    // the rows of the transpose are the polynomials in X of the coefficients of Y^j
    let mut h: Vec<i64> = vec![0];
    for col in f.transpose().coeffs() {
        h = fp::gcd(&h, &col.iter().map(|&v| v as i64).collect::<Vec<_>>(), p);
    }
    if fp::degree(&h) > 0 {
        return Err(line_place("x", &h, p));
    }
    if c[dx].iter().all(|&v| v == 0) {
        return Err(CountError::DegenerateModel(Degeneracy::FibreComponent {
            place: "x = infinity".into(),
        }));
    }
    Ok(())
}

/// The closed points of degree `d` of the line in a fixed order: for `d = 1` the points
/// `0, 1, ..., p - 1` and infinity; for `d >= 2` the monic irreducible `g` of degree `d` in the
/// order of their codes `sum g_i p^i`, `i < d`.
fn line_points_of_degree(p: i64, d: usize) -> Box<dyn Iterator<Item = BasePoint>> {
    if d == 1 {
        return Box::new(
            (0..p as u64)
                .map(BasePoint::Rational)
                .chain(std::iter::once(BasePoint::Infinity)),
        );
    }
    let pd = (p as u128).saturating_pow(d as u32);
    Box::new((0..pd).filter_map(move |mut code| {
        let mut g = vec![0i64; d + 1];
        for c in g.iter_mut().take(d) {
            *c = (code % p as u128) as i64;
            code /= p as u128;
        }
        g[d] = 1;
        let fs = fp::factor_with_multiplicity(&g, p);
        (fs.len() == 1 && fs[0].1 == 1 && fp::degree(&fs[0].0) as usize == d)
            .then(|| BasePoint::Closed(g.iter().map(|&c| c as u64).collect()))
    }))
}

/// Is the fibre of `cols` over the root of `g` (monic irreducible over `F_p`) irreducible of
/// full degree `n = cols.len() - 1` over `K = F_p[t]/(g)`? `cols[i]` is the coefficient of the
/// `i`-th power of the fibre variable, a polynomial in the base variable. By (F4) of the F-9
/// design: irreducible iff squarefree with one block of degree `n` in the distinct-degree
/// factorization (`factor_degrees`).
fn fibre_irreducible(cols: &[Vec<i64>], p: i64, g: &[i64]) -> bool {
    let k = PolyField::new(p, g);
    let v0 = k.generator();
    let n = cols.len() - 1;
    let fib: Vec<Vec<i64>> = cols.iter().map(|c| eval_fp_poly(&k, c, &v0)).collect();
    if rdeg(&k, &fib) != n as isize {
        return false;
    }
    factor_degrees(&k, &fib) == Some(vec![n])
}

/// The first irreducible fibre over a closed point of degree `d` in this ruling, among the first
/// `MAX_FIBRES_PER_DEGREE` points of [`line_points_of_degree`].
fn search_fibre(f: &FormModP, fib: Fibration, d: usize) -> Option<IrreducibleFibre> {
    let p = f.p() as i64;
    let g = match fib {
        Fibration::FibresInX => f.clone(),
        Fibration::FibresInY => f.transpose(),
    };
    let cols: Vec<Vec<i64>> = g
        .coeffs()
        .iter()
        .map(|r| r.iter().map(|&v| v as i64).collect())
        .collect();
    // Rows are rectangular and never trimmed (form.rs), so the reversed columns at 0 are the
    // coefficients of the top power of the base variable: the fibre over infinity.
    let rev: Vec<Vec<i64>> = cols
        .iter()
        .map(|c| c.iter().rev().cloned().collect())
        .collect();
    for base in line_points_of_degree(p, d).take(MAX_FIBRES_PER_DEGREE) {
        let ok = match &base {
            BasePoint::Rational(v) => fibre_irreducible(&cols, p, &[(p - *v as i64) % p, 1]),
            BasePoint::Infinity => fibre_irreducible(&rev, p, &[0, 1]),
            BasePoint::Closed(g) => {
                fibre_irreducible(&cols, p, &g.iter().map(|&c| c as i64).collect::<Vec<_>>())
            }
        };
        if ok {
            return Some(IrreducibleFibre {
                fibration: fib,
                base,
            });
        }
    }
    None
}

fn eval2(c: &[Vec<u64>], x: u64, y: u64, p: u64) -> u64 {
    let mut acc = 0u64;
    for row in c.iter().rev() {
        let mut inner = 0u64;
        for &v in row.iter().rev() {
            inner = (inner * y + v) % p;
        }
        acc = (acc * x + inner) % p;
    }
    acc
}

/// `(F, F_X, F_Y)` at `(x, y)` in a chart (`p < 2^16` and degrees `<= 4096`: no overflow).
fn value_and_gradient(c: &[Vec<u64>], x: u64, y: u64, p: u64) -> (u64, u64, u64) {
    let dxc: Vec<Vec<u64>> = (1..c.len())
        .map(|i| c[i].iter().map(|&v| v * i as u64 % p).collect())
        .collect();
    let dyc: Vec<Vec<u64>> = c
        .iter()
        .map(|r| {
            (1..r.len())
                .map(|j| r[j] * j as u64 % p)
                .collect::<Vec<_>>()
        })
        .map(|r| if r.is_empty() { vec![0] } else { r })
        .collect();
    let fx = if dxc.is_empty() {
        0
    } else {
        eval2(&dxc, x, y, p)
    };
    (eval2(c, x, y, p), fx, eval2(&dyc, x, y, p))
}

/// A smooth `F_p`-rational point of `V(F)` (`None` for a coordinate at infinity), if there is
/// one. The pass is exhaustive over `P^1 x P^1 (F_p)`: the four charts, and in each finite
/// fibre the roots in `F_p` in ascending order. So `None` is a fact, not a bounded search.
pub(crate) fn smooth_rational_point(f: &FormModP) -> Option<(Option<u64>, Option<u64>)> {
    let p = f.p();
    let charts = [
        (f.clone(), false, false),
        (f.reverse_x(), true, false),
        (f.reverse_y(), false, true),
        (f.reverse_x().reverse_y(), true, true),
    ];
    for (c, xinf, yinf) in &charts {
        let ys: Vec<u64> = if *yinf { vec![0] } else { (0..p).collect() };
        for &y in &ys {
            // the roots in F_p of the x-fibre (only x = 0 in the charts at x = infinity)
            let xs: Vec<u64> = if *xinf {
                vec![0]
            } else {
                let fib: Vec<i64> = c
                    .coeffs()
                    .iter()
                    .map(|r| r.iter().rev().fold(0u64, |acc, &v| (acc * y + v) % p) as i64)
                    .collect();
                if fp::is_zero(&fib) {
                    (0..p).collect()
                } else if fp::degree(&fib) <= 0 {
                    vec![]
                } else {
                    let mut roots: Vec<u64> = fp::factor_with_multiplicity(&fib, p as i64)
                        .into_iter()
                        .filter(|(g, _)| fp::degree(g) == 1)
                        .map(|(g, _)| ((p as i64 - g[0]) % p as i64) as u64)
                        .collect();
                    roots.sort_unstable();
                    roots
                }
            };
            for x in xs {
                let (v, fx, fy) = value_and_gradient(c.coeffs(), x, y, p);
                if v == 0 && (fx != 0 || fy != 0) {
                    return Some((
                        if *xinf { None } else { Some(x) },
                        if *yinf { None } else { Some(y) },
                    ));
                }
            }
        }
    }
    None
}

/// Steps 2 and 3 of the module header for a form that passed step 1.
pub(crate) fn irreducibility_search(f: &FormModP) -> Irreducibility {
    let (dx, dy) = f.bidegree();
    let n = gcd_usize(dx, dy);
    let mut first: Option<IrreducibleFibre> = None;
    for d in 1..=MAX_BASE_DEGREE {
        let settles = coprime_part(n, d) == 1;
        if first.is_some() && !settles {
            continue;
        }
        for fib in [Fibration::FibresInX, Fibration::FibresInY] {
            if let Some(w) = search_fibre(f, fib, d) {
                if settles {
                    return Irreducibility::Absolute(IrreducibilityEvidence {
                        fibre: w,
                        absolute: AbsoluteWitness::Bidegree { gcd: n },
                    });
                }
                if first.is_none() {
                    // F is irreducible over F_p (step 2): Claim C applies.
                    if let Some((x, y)) = smooth_rational_point(f) {
                        return Irreducibility::Absolute(IrreducibilityEvidence {
                            fibre: w,
                            absolute: AbsoluteWitness::SmoothRationalPoint { x, y },
                        });
                    }
                    first = Some(w);
                }
                break;
            }
        }
    }
    Irreducibility::NotShown(match first {
        None => IrreducibilityGap::NoIrreducibleFibre,
        Some(fibre) => IrreducibilityGap::NoSmoothRationalPoint { fibre, gcd: n },
    })
}

/// Whether `F mod p` is absolutely irreducible, by the criterion of the module header.
/// `Absolute` is a proof; `NotShown` claims nothing.
///
/// Errors: `Shape` for a bidegree with a zero entry; `DegenerateModel(FibreComponent)` for a
/// factor in one variable alone (a line of a ruling, including the lines at infinity).
///
/// Cost: two gcd chains over `F_p`, then at most `2 MAX_BASE_DEGREE MAX_FIBRES_PER_DEGREE` fibre
/// tests (a distinct-degree factorization of a polynomial of degree `dx` or `dy` over
/// `F_(p^d)`), and, when `gcd(dx, dy) > 1`, one pass over the `p + 1` rational fibres for a
/// smooth point. The search stops at the first witness. For the M23 forms (F-9 design, 2.8,
/// debug build): one to 17 quartic fibres over `F_p`, about 1 ms, or four fibres over `F_(p^2)`
/// at `p = 5`, about 20 ms. The worst case is a form that exhausts the search with large `dy`
/// (the `Y`-fibres have degree `dy`, at most 4096).
pub fn absolute_irreducibility(f: &FormModP) -> Result<Irreducibility, CountError> {
    check_bidegree(f)?;
    check_line_components(f)?;
    Ok(irreducibility_search(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_form(p: u64, rows: &[&[u64]]) -> FormModP {
        FormModP::new(p, rows.iter().map(|r| r.to_vec()).collect()).unwrap()
    }

    fn component_place(r: Result<(), CountError>) -> String {
        match r {
            Err(CountError::DegenerateModel(Degeneracy::FibreComponent { place })) => place,
            r => panic!("expected a fibre component, got {r:?}"),
        }
    }

    #[test]
    fn coprime_parts() {
        assert_eq!(coprime_part(1, 3), 1);
        assert_eq!(coprime_part(4, 2), 1);
        assert_eq!(coprime_part(6, 2), 3);
        assert_eq!(coprime_part(4, 3), 4);
        assert_eq!(coprime_part(12, 6), 1);
    }

    #[test]
    fn closed_points_by_gauss() {
        // Gauss: the number of monic irreducibles of degree d over F_q is
        // (1/d) sum_{e | d} mu(e) q^(d/e); with infinity for d = 1. PARI (F-9 design,
        // `wpf/gp/unit.gp`): 5, 10, 40, 150 over F_5 and 78 over F_13 in degree 2.
        let count = |p: i64, d: usize| line_points_of_degree(p, d).count();
        assert_eq!(count(5, 1), 6);
        assert_eq!(count(5, 2), 10);
        assert_eq!(count(5, 3), 40);
        assert_eq!(count(5, 4), 150);
        assert_eq!(count(13, 2), 78);
        // the order: codes ascending, so x^2 + 2 is the first of degree 2 over F_5
        assert_eq!(
            line_points_of_degree(5, 2).next(),
            Some(BasePoint::Closed(vec![2, 0, 1]))
        );
    }

    #[test]
    fn line_components_are_refused() {
        // F-9 design 3.3 (PARI `wpf/gp/regress.gp`), p = 13.
        // (iii) (x - 3) G: the vertical line x = 3, g(x) = x + 10.
        let f3 = test_form(
            13,
            &[
                &[5, 10, 11],
                &[4, 5, 12],
                &[1, 0, 2],
                &[7, 1, 10],
                &[2, 0, 1],
            ],
        );
        assert_eq!(
            component_place(check_line_components(&f3)),
            "x-place g(x) = [10, 1] (ascending)"
        );
        // (iii') G of bidegree (3, 2) read as (4, 2): the line x = infinity.
        let f4 = test_form(
            13,
            &[&[7, 1, 5], &[1, 3, 2], &[0, 1, 0], &[2, 0, 1], &[0, 0, 0]],
        );
        assert_eq!(component_place(check_line_components(&f4)), "x = infinity");
        // (iii'') (x^4 + 2x + 1) y + (x^3 + 5) read as (4, 2): the line y = infinity.
        let f5 = test_form(
            13,
            &[&[5, 1, 0], &[0, 2, 0], &[0, 0, 0], &[1, 0, 0], &[0, 1, 0]],
        );
        assert_eq!(component_place(check_line_components(&f5)), "y = infinity");
        // (x^2 + 2)(y + x + 1) over F_5 (x^2 + 2 irreducible mod 5): a vertical quadratic
        // factor, reported by the place x^2 + 2. Coefficients of x^i y^j:
        // x^3 + x^2 + 2x + 2 + (x^2 + 2) y.
        let fq = test_form(5, &[&[2, 2], &[2, 0], &[1, 1], &[1, 0]]);
        assert_eq!(
            component_place(check_line_components(&fq)),
            "x-place g(x) = [2, 0, 1] (ascending)"
        );
        // A reducible gcd reports its first irreducible factor in the order of fp_factor (by
        // degree, then by the coefficients from the top): (y - 1)(y - 2)(x + y) as a (1, 3)
        // form over F_7, (y^2 + 4y + 2) y + (y^2 + 4y + 2) x; y - 2 = [5, 1] comes first.
        let fr = test_form(7, &[&[0, 2, 4, 1], &[2, 4, 1, 0]]);
        assert_eq!(
            component_place(check_line_components(&fr)),
            "y-place g(y) = [5, 1] (ascending)"
        );
        // (v) x^2 + 11, bidegree (2, 0): a shape error, before any line test.
        let fv = test_form(13, &[&[11], &[0], &[1]]);
        assert!(matches!(
            absolute_irreducibility(&fv),
            Err(CountError::Shape(_))
        ));
    }

    #[test]
    fn display_strings() {
        let f6 = IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Rational(6),
        };
        assert_eq!(f6.to_string(), "x-fibre over y = 6");
        let fi = IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Infinity,
        };
        assert_eq!(fi.to_string(), "x-fibre over y = inf");
        let fy = IrreducibleFibre {
            fibration: Fibration::FibresInY,
            base: BasePoint::Rational(2),
        };
        assert_eq!(fy.to_string(), "y-fibre over x = 2");
        let fc = IrreducibleFibre {
            fibration: Fibration::FibresInX,
            base: BasePoint::Closed(vec![2, 0, 1]),
        };
        assert_eq!(
            fc.to_string(),
            "x-fibre over the point g(y) = [2, 0, 1] of degree 2"
        );
        assert_eq!(fc.residue_degree(), 2);
        let e1 = IrreducibilityEvidence {
            fibre: f6.clone(),
            absolute: AbsoluteWitness::Bidegree { gcd: 1 },
        };
        assert_eq!(
            e1.to_string(),
            "x-fibre over y = 6 irreducible of full degree; gcd(dx, dy) = 1"
        );
        let e2 = IrreducibilityEvidence {
            fibre: fc,
            absolute: AbsoluteWitness::Bidegree { gcd: 2 },
        };
        assert_eq!(
            e2.to_string(),
            "x-fibre over the point g(y) = [2, 0, 1] of degree 2 irreducible of full degree; \
             gcd(dx, dy) = 2, settled by the residue degree 2"
        );
        let e3 = IrreducibilityEvidence {
            fibre: f6.clone(),
            absolute: AbsoluteWitness::SmoothRationalPoint {
                x: Some(5),
                y: None,
            },
        };
        assert_eq!(
            e3.to_string(),
            "x-fibre over y = 6 irreducible of full degree; smooth F_p-point (5, inf)"
        );
        assert_eq!(
            IrreducibilityGap::NoIrreducibleFibre.to_string(),
            "no irreducible fibre among the first 64 closed points of each degree <= 4 in \
             either ruling"
        );
        assert_eq!(
            IrreducibilityGap::NoSmoothRationalPoint { fibre: f6, gcd: 4 }.to_string(),
            "irreducible over F_p by the x-fibre over y = 6; gcd(dx, dy) = 4 and no smooth \
             F_p-point"
        );
    }

    #[test]
    fn smooth_points_and_fibres_at_infinity() {
        // y = x^2 + 1 over F_7 as a (2, 1) form: every point is smooth, and the first in the
        // order of the pass is (0, 1).
        let f = test_form(7, &[&[1, 6], &[0, 0], &[1, 0]]);
        assert_eq!(smooth_rational_point(&f), Some((Some(0), Some(1))));
        // F = (x^2 + 1) y + (x^2 + 2) over F_7. The x-fibre over y = 0 is x^2 + 2 and the one
        // over y = infinity is x^2 + 1 (the coefficient of y); -2 = 5 and -1 = 6 are nonsquares
        // mod 7, so both are irreducible. Over y = 3 the fibre is 4x^2 + 5, x^2 = 4: reducible.
        let g = test_form(7, &[&[2, 1], &[0, 0], &[1, 1]]);
        let cols: Vec<Vec<i64>> = g
            .coeffs()
            .iter()
            .map(|r| r.iter().map(|&v| v as i64).collect())
            .collect();
        let rev: Vec<Vec<i64>> = cols
            .iter()
            .map(|c| c.iter().rev().cloned().collect())
            .collect();
        assert!(fibre_irreducible(&cols, 7, &[0, 1]));
        assert!(fibre_irreducible(&rev, 7, &[0, 1]));
        assert!(!fibre_irreducible(&cols, 7, &[4, 1]));
    }
}
