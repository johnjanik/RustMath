//! The singular locus of `V(F mod p)` in `P^1 x P^1`, point by point, in residue fields.
//!
//! Lemma (WP-F design, Lemma 4.3). Every singular point of `X0_p` over `F_p-bar` lies over a
//! root of the formal discriminant `D(Y)` of `F` as a binary form of degree `dx` in `X` (a binary
//! form of degree 2 to 4, whose formal discriminant is a binary form of degree `(2dx - 2) dy` in
//! `Y`). Proof: at a singular point both `F` and `F_X` vanish, so the fibre has a multiple root
//! there (at `X = infinity` in the other chart), and a binary form has a multiple root in `P^1`
//! exactly when its formal discriminant vanishes. If `D = 0` identically every fibre has a
//! multiple root; the model is refused.
//!
//! Procedure: factor `D` over `F_p` (`fp_factor::factor_with_multiplicity`) and analyse every
//! irreducible factor `g` (whatever its multiplicity, so no tameness assumption is needed), and
//! `Y = infinity` when `deg D` is below the formal degree. In `K = F_p[Y]/(g)` with `y0 = Y mod g`
//! the singular points with `X` affine are the roots of `G = gcd(f, f_X, F_Y(X, y0))` in
//! `K[X]`, `deg G <= 3`; the point at `X = infinity` is singular iff `c_dx(y0) = c_(dx-1)(y0) =
//! c_dx'(y0) = 0`. A node is recognised by the Hessian discriminant (MAGMA `IsNode`), anything
//! else goes to the local classifier.

use super::local::{classify, LocalType};
use super::residue::{eval_fp_poly, rdeg, rgcd, Poly, PolyField, QuadExt, ResField};
use super::{CountError, Degeneracy, FormModP};
use rustmath_polynomials::fp_factor as fp;

/// A place of the `Y`-line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YPlace {
    /// `g(Y) = 0`, `g` monic irreducible over `F_p` (ascending coefficients).
    Finite(Vec<u64>),
    /// `Y = infinity`.
    Infinity,
}

/// A closed singular point of `V(F mod p)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingularPoint {
    /// The degree of its residue field over `F_p`: the point has this many geometric points.
    pub residue_degree: usize,
    /// The place of the `Y`-line below it.
    pub y_place: YPlace,
    /// True if the point lies on `X = infinity`.
    pub x_at_infinity: bool,
    /// For a point with residue field `F_p`: `(x, y)`, `None` for infinity.
    pub coordinates: Option<(Option<u64>, Option<u64>)>,
    /// For a point with `X` finite: the characteristic polynomial over `F_p` of its
    /// `X`-coordinate in the residue field (monic, ascending, degree `residue_degree`): the
    /// product of `X - x_i` over its geometric points.
    pub x_charpoly: Option<Vec<u64>>,
    /// The analytic type.
    pub local: LocalType,
    /// The delta invariant of each geometric point (`None` if unclassified).
    pub delta: Option<usize>,
    /// Sizes of the Frobenius orbits (over the residue field) of the branches (`None` if
    /// unclassified): a node `[1, 1]` (rational branches) or `[2]` (conjugate), `A_k` with `k`
    /// even `[1]`, an ordinary m-fold point the degrees of the factors of its tangent cone.
    pub branch_orbits: Option<Vec<usize>>,
}

impl SingularPoint {
    /// The number of `F_(p^k)`-rational branches above one geometric point, for
    /// `residue_degree | k` (`None` if unclassified).
    pub fn rational_branches(&self, k: usize) -> Option<usize> {
        let d = self.residue_degree;
        if !k.is_multiple_of(d) {
            return Some(0);
        }
        let e = k / d;
        self.branch_orbits
            .as_ref()
            .map(|o| o.iter().filter(|&&s| e.is_multiple_of(s)).sum())
    }

    /// A one-line description.
    pub fn describe(&self) -> String {
        let loc = match &self.coordinates {
            Some((x, y)) => {
                let s = |v: &Option<u64>| v.map(|c| c.to_string()).unwrap_or_else(|| "inf".into());
                format!("(x, y) = ({}, {})", s(x), s(y))
            }
            None => {
                let yp = match &self.y_place {
                    YPlace::Finite(g) => format!("g(y) of degree {}", g.len() - 1),
                    YPlace::Infinity => "y = inf".into(),
                };
                format!(
                    "degree {} point over {}{}",
                    self.residue_degree,
                    yp,
                    if self.x_at_infinity { ", x = inf" } else { "" }
                )
            }
        };
        let br = match &self.branch_orbits {
            Some(o) => format!(" branch orbits {o:?}"),
            None => String::new(),
        };
        format!("{} at {loc}{br}", self.local)
    }
}

/// The singular points of `V(F mod p)` and the discriminant data they were found from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingularLocus {
    pub p: u64,
    pub points: Vec<SingularPoint>,
    /// `deg D` as a polynomial in `Y`.
    pub disc_degree: usize,
    /// `(2 dx - 2) dy`; the multiplicity of `Y = infinity` is the difference.
    pub disc_formal_degree: usize,
    /// `(degree, multiplicity)` of the irreducible factors of `D`, with `Y = infinity` last as
    /// `(1, multiplicity)` when it is a root.
    pub disc_factor_degrees: Vec<(usize, u32)>,
}

impl SingularLocus {
    /// The number of nodes over `F_p-bar`.
    pub fn node_count(&self) -> usize {
        self.points
            .iter()
            .filter(|s| s.local == LocalType::Node)
            .map(|s| s.residue_degree)
            .sum()
    }

    /// The closed points that are not nodes.
    pub fn non_nodal(&self) -> Vec<&SingularPoint> {
        self.points
            .iter()
            .filter(|s| s.local != LocalType::Node)
            .collect()
    }

    /// The sum of the delta invariants over `F_p-bar`, if every point is classified.
    pub fn delta_sum(&self) -> Option<usize> {
        self.points
            .iter()
            .map(|s| s.delta.map(|d| d * s.residue_degree))
            .sum()
    }
}

fn quartic_like_disc(c: &[Poly], p: i64) -> Poly {
    let m = |xs: &[&Poly], k: i64| -> Poly {
        let mut r: Poly = vec![k.rem_euclid(p)];
        for x in xs {
            r = fp::mul(&r, x, p);
        }
        r
    };
    let sum = |terms: Vec<Poly>| -> Poly {
        let mut s: Poly = vec![0];
        for t in &terms {
            s = fp::add(&s, t, p);
        }
        s
    };
    match c.len() - 1 {
        2 => {
            let (a, b, cc) = (&c[2], &c[1], &c[0]);
            sum(vec![m(&[b, b], 1), m(&[a, cc], -4)])
        }
        3 => {
            let (a, b, cc, d) = (&c[3], &c[2], &c[1], &c[0]);
            sum(vec![
                m(&[b, b, cc, cc], 1),
                m(&[a, cc, cc, cc], -4),
                m(&[b, b, b, d], -4),
                m(&[a, a, d, d], -27),
                m(&[a, b, cc, d], 18),
            ])
        }
        _ => {
            let (a, b, cc, d, e) = (&c[4], &c[3], &c[2], &c[1], &c[0]);
            sum(vec![
                m(&[a, a, a, e, e, e], 256),
                m(&[a, a, b, d, e, e], -192),
                m(&[a, a, cc, cc, e, e], -128),
                m(&[a, a, cc, d, d, e], 144),
                m(&[a, a, d, d, d, d], -27),
                m(&[a, b, b, cc, e, e], 144),
                m(&[a, b, b, d, d, e], -6),
                m(&[a, b, cc, cc, d, e], -80),
                m(&[a, b, cc, d, d, d], 18),
                m(&[a, cc, cc, cc, cc, e], 16),
                m(&[a, cc, cc, cc, d, d], -4),
                m(&[b, b, b, b, e, e], -27),
                m(&[b, b, b, cc, d, e], 18),
                m(&[b, b, b, d, d, d], -4),
                m(&[b, b, cc, cc, cc, e], -4),
                m(&[b, b, cc, cc, d, d], 1),
            ])
        }
    }
}

/// The formal discriminant `D(Y)` of `F` as a binary form in `X` (degree `dx` in `2..=4`), in
/// `F_p[Y]`. The formulas are the standard ones (the quartic: 16 terms).
pub(crate) fn discriminant(f: &FormModP) -> Result<Poly, CountError> {
    let (dx, _) = f.bidegree();
    if !(2..=4).contains(&dx) {
        return Err(CountError::FibreDegree(dx));
    }
    let p = f.p() as i64;
    let cols: Vec<Poly> = f
        .coeffs()
        .iter()
        .map(|r| fp::trim(&r.iter().map(|&x| x as i64).collect::<Vec<_>>()))
        .collect();
    Ok(fp::trim(&quartic_like_disc(&cols, p)))
}

struct PlaceCtx<'a> {
    /// Columns of the chart: `cols[i]` = polynomial in `Y` of `X^i`, coefficients in `[0, p)`.
    cols: &'a [Vec<i64>],
    place: YPlace,
}

fn place_name(place: &YPlace) -> String {
    match place {
        YPlace::Finite(g) => format!("y-place g(y) = {g:?} (ascending)"),
        YPlace::Infinity => "y = infinity".into(),
    }
}

/// Node test by the Hessian discriminant at a known singular point, then the classifier.
fn analyse_point<F: ResField>(
    field: &F,
    cols: &[Vec<i64>],
    x0: &F::E,
    y0: &F::E,
    place: &YPlace,
    x_at_infinity: bool,
    coordinates: Option<(Option<u64>, Option<u64>)>,
) -> SingularPoint {
    let p = field.p();
    // c_i(y0), c_i'(y0), c_i''(y0)
    let dcol = |c: &[i64]| fp::derivative_of(c, p);
    let mut fxx = field.zero();
    let mut fxy = field.zero();
    let mut fyy = field.zero();
    let mut xp = vec![field.one()];
    for i in 1..cols.len() {
        xp.push(field.mul(&xp[i - 1], x0));
    }
    for (i, c) in cols.iter().enumerate() {
        let c1 = dcol(c);
        let c2 = dcol(&c1);
        if i >= 2 {
            let v = eval_fp_poly(field, c, y0);
            fxx = field.add(
                &fxx,
                &field.scale(&field.mul(&v, &xp[i - 2]), (i * (i - 1)) as i64),
            );
        }
        if i >= 1 {
            let v = eval_fp_poly(field, &c1, y0);
            fxy = field.add(&fxy, &field.scale(&field.mul(&v, &xp[i - 1]), i as i64));
        }
        let v = eval_fp_poly(field, &c2, y0);
        fyy = field.add(&fyy, &field.mul(&v, &xp[i]));
    }
    let h = field.sub(&field.mul(&fxy, &fxy), &field.mul(&fxx, &fyy));
    let (local, branch_orbits) = if !field.is_zero(&h) {
        let o = if field.is_square(&h) {
            vec![1, 1]
        } else {
            vec![2]
        };
        (LocalType::Node, Some(o))
    } else {
        let d = classify(field, cols, x0, y0);
        (d.local, d.branch_orbits)
    };
    let delta = local.delta();
    let x_charpoly = if x_at_infinity {
        None
    } else {
        Some(field.charpoly(x0).iter().map(|&c| c as u64).collect())
    };
    SingularPoint {
        residue_degree: field.degree(),
        y_place: place.clone(),
        x_at_infinity,
        coordinates,
        x_charpoly,
        local,
        delta,
        branch_orbits,
    }
}

fn unclassified_point(d: usize, place: &YPlace, why: String) -> SingularPoint {
    SingularPoint {
        residue_degree: d,
        y_place: place.clone(),
        x_at_infinity: false,
        coordinates: None,
        x_charpoly: None,
        local: LocalType::Unclassified(why),
        delta: None,
        branch_orbits: None,
    }
}

/// All singular points over the place `g` (monic irreducible) of the `Y`-line of the chart.
fn analyse_place(
    ctx: &PlaceCtx<'_>,
    g: &[i64],
    p: i64,
    out: &mut Vec<SingularPoint>,
) -> Result<(), CountError> {
    let cols = ctx.cols;
    let k = PolyField::new(p, g);
    let d = k.degree();
    let y0 = k.generator();
    let dx = cols.len() - 1;
    let yc = |kk: &PolyField, a: &Poly| -> Option<u64> {
        if matches!(ctx.place, YPlace::Infinity) {
            None
        } else {
            kk.as_fp(a).map(|v| v as u64)
        }
    };
    let ycoord = if d == 1 { yc(&k, &y0) } else { None };
    let f: Vec<Poly> = cols.iter().map(|c| eval_fp_poly(&k, c, &y0)).collect();
    if f.iter().all(|c| k.is_zero(c)) {
        return Err(CountError::DegenerateModel(Degeneracy::FibreComponent {
            place: place_name(&ctx.place),
        }));
    }
    let dv: Vec<Poly> = cols
        .iter()
        .map(|c| eval_fp_poly(&k, &fp::derivative_of(c, p), &y0))
        .collect();
    let ft: Vec<Poly> = (1..=dx).map(|i| k.scale(&f[i], i as i64)).collect();
    let g1 = rgcd(&k, &f, &ft);
    let gs = rgcd(&k, &g1, &dv);
    let dg = rdeg(&k, &gs);
    let one_point = |tau: Poly, out: &mut Vec<SingularPoint>| {
        let coords = if d == 1 {
            Some((k.as_fp(&tau).map(|v| v as u64), ycoord))
        } else {
            None
        };
        out.push(analyse_point(
            &k, cols, &tau, &y0, &ctx.place, false, coords,
        ));
    };
    let inv = |a: i64| fp::mod_inv(a.rem_euclid(p), p).expect("p odd prime");
    match dg {
        d0 if d0 <= 0 => {}
        1 => one_point(k.neg(&gs[0]), out),
        2 => {
            let (c0, c1) = (gs[0].clone(), gs[1].clone());
            let disc = k.sub(&k.mul(&c1, &c1), &k.scale(&c0, 4));
            let half = inv(2);
            if k.is_zero(&disc) {
                one_point(k.scale(&k.neg(&c1), half), out);
            } else if k.is_square(&disc) {
                let s = k.sqrt(&disc).expect("square");
                one_point(k.scale(&k.sub(&s, &c1), half), out);
                one_point(k.scale(&k.sub(&k.neg(&s), &c1), half), out);
            } else {
                // One closed point of degree 2d with residue field K[X]/(G).
                let e = QuadExt::new(k.clone(), c1, c0);
                let x0 = e.generator();
                let y0e = e.embed(y0.clone());
                out.push(analyse_point(&e, cols, &x0, &y0e, &ctx.place, false, None));
            }
        }
        3 => {
            // deg G = 3 only for f = c (X - tau)^4 (G = (X - tau)^3); check it.
            // (X - tau)^3 = X^3 - 3 tau X^2 + 3 tau^2 X - tau^3
            let tau = if p != 3 {
                Some(k.scale(&k.neg(&gs[2]), inv(3)))
            } else {
                None
            };
            let ok = tau.as_ref().is_some_and(|tau| {
                let t2 = k.mul(tau, tau);
                let t3 = k.mul(&t2, tau);
                k.is_zero(&k.sub(&gs[1], &k.scale(&t2, 3))) && k.is_zero(&k.add(&gs[0], &t3))
            });
            if ok {
                one_point(tau.expect("checked"), out);
            } else {
                out.push(unclassified_point(
                    d,
                    &ctx.place,
                    "gcd(f, f_X, F_Y) of degree 3 is not a cube (X - tau)^3".into(),
                ));
            }
        }
        _ => out.push(unclassified_point(
            d,
            &ctx.place,
            format!("gcd of degree {dg} in one fibre"),
        )),
    }
    // X = infinity: chart U = 1/X.
    if k.is_zero(&f[dx]) && k.is_zero(&f[dx - 1]) && k.is_zero(&dv[dx]) {
        let rcols: Vec<Vec<i64>> = cols.iter().rev().cloned().collect();
        let coords = if d == 1 { Some((None, ycoord)) } else { None };
        out.push(analyse_point(
            &k,
            &rcols,
            &k.zero(),
            &y0,
            &ctx.place,
            true,
            coords,
        ));
    }
    Ok(())
}

/// The singular locus of `V(F mod p)` for a form with `X`-degree in `2..=4` (Lemma 4.3 above).
///
/// Errors: `FibreDegree` (`dx` not in `2..=4`), `DegenerateModel(DiscriminantVanishes)`,
/// `DegenerateModel(FibreComponent)`.
pub fn singular_locus(f: &FormModP) -> Result<SingularLocus, CountError> {
    let disc = discriminant(f)?;
    let p = f.p() as i64;
    if fp::is_zero(&disc) {
        return Err(CountError::DegenerateModel(
            Degeneracy::DiscriminantVanishes,
        ));
    }
    let (dx, dy) = f.bidegree();
    let formal = (2 * dx - 2) * dy;
    let dd = fp::degree(&disc) as usize;
    let cols: Vec<Vec<i64>> = f
        .coeffs()
        .iter()
        .map(|r| r.iter().map(|&x| x as i64).collect())
        .collect();
    let mut points = Vec::new();
    let mut fdeg = Vec::new();
    if dd > 0 {
        for (g, e) in fp::factor_with_multiplicity(&disc, p) {
            fdeg.push((fp::degree(&g) as usize, e));
            let gu: Vec<u64> = g.iter().map(|&c| c as u64).collect();
            let ctx = PlaceCtx {
                cols: &cols,
                place: YPlace::Finite(gu),
            };
            analyse_place(&ctx, &g, p, &mut points)?;
        }
    }
    if dd < formal {
        let rev: Vec<Vec<i64>> = cols
            .iter()
            .map(|c| c.iter().rev().cloned().collect())
            .collect();
        fdeg.push((1, (formal - dd) as u32));
        let ctx = PlaceCtx {
            cols: &rev,
            place: YPlace::Infinity,
        };
        analyse_place(&ctx, &[0, 1], p, &mut points)?;
    }
    Ok(SingularLocus {
        p: f.p(),
        points,
        disc_degree: dd,
        disc_formal_degree: formal,
        disc_factor_degrees: fdeg,
    })
}
