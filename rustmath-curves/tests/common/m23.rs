//! Loaders and the per-prime driver for the M23 curve X of HJLPPZ (WP-F,
//! `docs/workstreams/m23_golay/HANDOFF.md` Section 3). Shared by `tests/m23_zeta.rs` and
//! `examples/m23_zeta.rs` (included there with `#[path]`). Not library API.
//!
//! Models (the README in `tests/data/m23/` gives sources, licences and checksums):
//! - `G`: Le Lay's CC0 bidegree (4,23) equation `data/optimal_23_4_Z.json` (vendored). It is
//!   HJLPPZ's `F` in another coordinate: `F(T, (967W - 3)/(81 - 2267W)) (81 - 2267W)^23 = c G(T, W)`
//!   with `c > 0` (checked by the opt-in test `env_f_equals_g_by_mobius`).
//! - `G7 = G(T, 7V + 3)` and `G13 = G(T, (11V + 130)/(V + 13)) (V + 13)^23`, made primitive:
//!   further integral models of the same curve, good at 7 and 13 where `G` degenerates (the
//!   determinant of the map to `F` is `71526 = 2 3 7 13 131`).
//! - `F` and `calF` (the covers over the sextic field `L`) are read from a checkout of
//!   `shaowuz/m23isgalois` named by the environment variable `M23ISGALOIS`. That repository has
//!   no licence, so nothing from it is copied into RustMath.
//!
//! The genus of X and of the six covers over `L` is 4: [HJLPPZ, Prop. 3.10] (62 nodes on the
//! (4,23) model), equivalently Riemann-Hurwitz for the branch data `(2A, 23A, 23B)`:
//! `2g - 2 = -2*23 + (23 - 15) + 22 + 22 = 6`.

#![allow(dead_code)]

use rustmath_curves::nodal_biprojective::{
    certify, normalization_point_counts, singular_locus, BihomogeneousForm, BranchRule, CountError,
    CountOptions, FormModP, NormalizationCounts, ReductionStatus, Unresolved,
};
use rustmath_curves::weil::{f2_poly_string, FrobeniusPolynomial, TwoTorsionOrder};
use rustmath_integers::Integer;
use std::path::{Path, PathBuf};

/// The genus of X and of the covers over L (HJLPPZ Prop. 3.10; Riemann-Hurwitz).
pub const GENUS: usize = 4;

/// `s^6 - s^5 - 3s^4 - 3s^3 + s^2 + 5s + 4`, ascending (HJLPPZ Section 3.7).
pub const L_POLY: [i64; 7] = [4, 5, 1, -3, -3, -1, 1];

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/m23")
}

fn read_json(name: &str) -> serde_json::Value {
    let path = data_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON {}: {e}", path.display()))
}

/// Le Lay's `G(T, W)`: `coefficients_T_then_W[i][j]` = coefficient of `T^i W^j`.
pub fn model_g() -> BihomogeneousForm {
    let v = read_json("lelay_optimal_23_4_Z.json");
    let rows: Vec<Vec<String>> = v["coefficients_T_then_W"]
        .as_array()
        .expect("array")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row")
                .iter()
                .map(|x| x.as_str().expect("decimal string").to_string())
                .collect()
        })
        .collect();
    BihomogeneousForm::from_decimal_rows(&rows).expect("G parses")
}

/// `G7 = G(T, 7V + 3)` (primitive).
pub fn model_g7(g: &BihomogeneousForm) -> BihomogeneousForm {
    g.mobius_y(7, 3, 0, 1).expect("G7")
}

/// `G13 = G(T, (11V + 130)/(V + 13)) (V + 13)^23` (primitive).
pub fn model_g13(g: &BihomogeneousForm) -> BihomogeneousForm {
    g.mobius_y(11, 130, 1, 13).expect("G13")
}

/// The list of CC0 models tried at each prime, in order.
pub fn cc0_models() -> Vec<(&'static str, BihomogeneousForm)> {
    let g = model_g();
    let g7 = model_g7(&g);
    let g13 = model_g13(&g);
    vec![("G", g), ("G7", g7), ("G13", g13)]
}

/// The checkout of `shaowuz/m23isgalois`, if `M23ISGALOIS` is set.
pub fn m23isgalois() -> Option<PathBuf> {
    std::env::var_os("M23ISGALOIS").map(PathBuf::from)
}

/// For `#[ignore]` tests that need the external data: never pass vacuously.
pub fn require_m23isgalois() -> PathBuf {
    m23isgalois().expect("set M23ISGALOIS to a checkout of shaowuz/m23isgalois")
}

/// Parse an integer with optional sign ending at byte `end` (exclusive) of `s`.
fn int_before(s: &[u8], end: usize) -> (Integer, usize) {
    let mut i = end;
    while i > 0 && s[i - 1].is_ascii_digit() {
        i -= 1;
    }
    let mut start = i;
    if i > 0 && (s[i - 1] == b'-' || s[i - 1] == b'+') {
        start = i - 1;
    }
    let txt = std::str::from_utf8(&s[start..end]).expect("ascii");
    (
        Integer::from_decimal_str(txt.trim_start_matches('+')).expect("integer"),
        start,
    )
}

fn uint_after(s: &[u8], start: usize) -> (usize, usize) {
    let mut i = start;
    while i < s.len() && s[i].is_ascii_digit() {
        i += 1;
    }
    let txt = std::str::from_utf8(&s[start..i]).expect("ascii");
    (txt.parse().expect("exponent"), i)
}

/// `F[i][j]` from `polynomial_F.m`: terms `<int>*T^<i>*V^<j>` (as `m23golay/data.py`).
pub fn model_f(dir: &Path) -> BihomogeneousForm {
    let text = std::fs::read_to_string(dir.join("polynomial_F.m"))
        .unwrap_or_else(|e| panic!("polynomial_F.m in {}: {e}", dir.display()));
    let body: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let s = body.as_bytes();
    let mut coeffs = vec![vec![Integer::zero(); 24]; 5];
    let mut n = 0;
    let pat = b"*T^";
    let mut k = 0;
    while k + 3 <= s.len() {
        if &s[k..k + 3] == pat {
            let (c, _) = int_before(s, k);
            let (i, after) = uint_after(s, k + 3);
            assert_eq!(&s[after..after + 3], b"*V^", "term shape");
            let (j, after2) = uint_after(s, after + 3);
            coeffs[i][j] = coeffs[i][j].clone() + c;
            n += 1;
            k = after2;
        } else {
            k += 1;
        }
    }
    assert_eq!(n, 120, "polynomial_F.m: 120 terms");
    BihomogeneousForm::new(coeffs).expect("F")
}

/// `calF[(i, j)]` = power-basis coefficients `[c_0..c_5]` of the coefficient of `T^i V^j`
/// (`polynomial_calF.m`, as `m23golay/data.py`).
pub fn model_calf(dir: &Path) -> Vec<Vec<[Integer; 6]>> {
    let text = std::fs::read_to_string(dir.join("polynomial_calF.m"))
        .unwrap_or_else(|e| panic!("polynomial_calF.m in {}: {e}", dir.display()));
    let start = text.find("calF :=").expect("calF :=") + 7;
    let end = text[start..].find("assert").expect("assert") + start;
    let body: String = text[start..end]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let s = body.as_bytes();
    let zero = || std::array::from_fn::<Integer, 6, _>(|_| Integer::zero());
    let mut out: Vec<Vec<[Integer; 6]>> =
        (0..5).map(|_| (0..24).map(|_| zero()).collect()).collect();
    let mut k = 0;
    let mut n = 0;
    while k < s.len() {
        if s[k] != b'(' {
            k += 1;
            continue;
        }
        // an outer group "((c0)*s^0+(c1)*s^1+...)*T^i*V^j"
        let mut depth = 0;
        let mut e = k;
        loop {
            match s[e] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            e += 1;
        }
        let inner = &s[k + 1..e];
        assert_eq!(&s[e + 1..e + 4], b"*T^", "term shape");
        let (i, a) = uint_after(s, e + 4);
        assert_eq!(&s[a..a + 3], b"*V^");
        let (j, b) = uint_after(s, a + 3);
        let mut cs = zero();
        let mut m = 0;
        while m < inner.len() {
            assert_eq!(inner[m], b'(');
            let close = m + inner[m..].iter().position(|&c| c == b')').expect(")");
            let txt = std::str::from_utf8(&inner[m + 1..close]).expect("ascii");
            let c = Integer::from_decimal_str(txt).expect("coefficient");
            assert_eq!(&inner[close + 1..close + 4], b"*s^");
            let (ex, after) = uint_after(inner, close + 4);
            cs[ex] = c;
            m = after;
            if m < inner.len() && inner[m] == b'+' {
                m += 1;
            }
        }
        out[i][j] = cs;
        n += 1;
        k = b;
    }
    assert_eq!(n, 120, "polynomial_calF.m: 120 terms");
    out
}

fn mod_p(c: &Integer, p: u64) -> u64 {
    let pi = Integer::from(p);
    let r = (c % &pi).to_i64().expect("residue");
    r.rem_euclid(p as i64) as u64
}

/// The reduction of `calF` at the degree-one prime `(p, s - s0)` of `L`.
pub fn calf_mod(calf: &[Vec<[Integer; 6]>], p: u64, s0: u64) -> Result<FormModP, CountError> {
    let mut pw = [1u64; 6];
    for e in 1..6 {
        pw[e] = pw[e - 1] * s0 % p;
    }
    let cols: Vec<Vec<u64>> = calf
        .iter()
        .map(|row| {
            row.iter()
                .map(|cs| {
                    cs.iter()
                        .zip(pw.iter())
                        .fold(0u64, |acc, (c, &w)| (acc + mod_p(c, p) * w) % p)
                })
                .collect()
        })
        .collect();
    FormModP::new(p, cols)
}

/// The roots `s0` in `F_p` of the defining polynomial of `L` (the degree-one primes `(p, s - s0)`).
pub fn degree_one_primes(p: u64) -> Vec<u64> {
    (0..p)
        .filter(|&s| {
            let mut acc = 0i64;
            for &c in L_POLY.iter().rev() {
                acc = (acc * s as i64 + c).rem_euclid(p as i64);
            }
            acc == 0
        })
        .collect()
}

/// One row of `zeta_X.json` / `zeta_L.json`.
#[derive(Clone, Debug)]
pub struct RefRow {
    pub p: u64,
    pub s0: Option<u64>,
    pub ok: bool,
    pub counts: Option<Vec<u64>>,
    pub p_coeffs: Option<Vec<i64>>,
    pub q_candidates: Vec<String>,
}

fn ref_rows(v: &serde_json::Value) -> Vec<RefRow> {
    v.as_array()
        .expect("rows")
        .iter()
        .map(|r| RefRow {
            p: r["p"].as_u64().expect("p"),
            s0: r.get("s0").and_then(|x| x.as_u64()),
            ok: r["status"].as_str() == Some("ok"),
            counts: r
                .get("counts")
                .and_then(|c| c.as_array())
                .map(|c| c.iter().map(|x| x.as_u64().expect("count")).collect()),
            p_coeffs: r
                .get("P")
                .and_then(|c| c.as_array())
                .map(|c| c.iter().map(|x| x.as_i64().expect("coeff")).collect()),
            q_candidates: r
                .get("Q candidates")
                .and_then(|c| c.as_array())
                .map(|c| {
                    c.iter()
                        .map(|x| x.as_str().expect("Q").to_string())
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect()
}

/// `zeta_X.json` (johnjanik/M23 at d05475b): 19 rows, p = 5 (skipped, "not a Weil
/// polynomial", counts 8, 36, 170, 560) and p = 7..79.
pub fn zeta_x() -> Vec<RefRow> {
    ref_rows(&read_json("zeta_X.json"))
}

/// `zeta_L.json` rows and `check_53` (johnjanik/M23 at d05475b).
pub fn zeta_l() -> (Vec<RefRow>, Vec<u64>) {
    let v = read_json("zeta_L.json");
    let check: Vec<u64> = v["check_53"]
        .as_array()
        .expect("check_53")
        .iter()
        .map(|x| x.as_u64().expect("count"))
        .collect();
    (ref_rows(&v["results"]), check)
}

/// `h62(T)` ascending (`h62.json`): `disc_V F = e T^8 (T^2+23)^22 h62^2`.
pub fn h62() -> Vec<Integer> {
    read_json("h62.json")["h62"]
        .as_array()
        .expect("h62")
        .iter()
        .map(|x| Integer::from_decimal_str(x.as_str().expect("decimal")).expect("integer"))
        .collect()
}

/// The rank of a status for model selection: Nodal > DeltaSumCertified > Unclassified >
/// DeltaMismatch (a refusal ranks below all of them).
pub fn status_rank(s: &ReductionStatus) -> u8 {
    match s {
        ReductionStatus::Decided(_) => 4,
        ReductionStatus::Unresolved(Unresolved::DeltaSumCertified { .. }) => 3,
        ReductionStatus::Unresolved(Unresolved::Unclassified(_)) => 2,
        ReductionStatus::Unresolved(Unresolved::DeltaMismatch { .. }) => 1,
    }
}

/// The result of one row.
pub struct Row {
    pub p: u64,
    pub s0: Option<u64>,
    pub model: String,
    pub result: Result<NormalizationCounts, CountError>,
}

/// Panics on `Err(CountError::GenusInconsistent)`. A delta-sum below `p_a - g` contradicts the
/// genus 4 of X and of the covers ([`GENUS`]), on which every certificate rests. So it is a
/// fault in the genus input or in the code, not a refusal of one model.
pub fn assert_genus_consistent<T>(result: &Result<T, CountError>, what: &str) {
    if let Err(e @ CountError::GenusInconsistent { .. }) = result {
        panic!(
            "{what}: {e}. This contradicts the genus hypothesis g = {GENUS} of every \
             certificate; it is not a refusal"
        );
    }
}

/// The certificate of each model at p (cheap), and the index of the best one (None if all are
/// refused). The first model wins among equal ranks.
///
/// Panics if a model gives `CountError::GenusInconsistent` (see [`assert_genus_consistent`]).
pub fn select_model(
    forms: &[(String, FormModP)],
) -> (Vec<Result<ReductionStatus, CountError>>, Option<usize>) {
    let statuses: Vec<Result<ReductionStatus, CountError>> = forms
        .iter()
        .map(|(name, f)| {
            let status =
                singular_locus(f).and_then(|locus| certify(&locus, f.arithmetic_genus(), GENUS));
            assert_genus_consistent(&status, &format!("model {name} at p = {}", f.p()));
            status
        })
        .collect();
    let mut best: Option<(usize, u8)> = None;
    for (i, s) in statuses.iter().enumerate() {
        if let Ok(st) = s {
            let r = status_rank(st);
            if best.is_none_or(|(_, b)| r > b) {
                best = Some((i, r));
            }
        }
    }
    (statuses, best.map(|b| b.0))
}

/// X at p from the CC0 models `[G, G7, G13]`: the first model with the best status. When two
/// models both have a status that gives counts (Decided or DeltaSumCertified), their counts at
/// `k <= min(kmax, 2)` are checked to agree (they must: the models are isomorphic over Q and the
/// counts are those of the reduction of X).
pub fn x_row(
    models: &[(&'static str, BihomogeneousForm)],
    p: u64,
    kmax: u32,
    opts: &CountOptions,
) -> Row {
    let mut forms = Vec::new();
    let mut first_err = None;
    for (name, g) in models {
        match g.reduce(p) {
            Ok(f) => forms.push((name.to_string(), f)),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e)
                }
            }
        }
    }
    let (statuses, best) = select_model(&forms);
    let Some(b) = best else {
        let err = statuses
            .into_iter()
            .find_map(|s| s.err())
            .or(first_err)
            .unwrap_or(CountError::ZeroForm);
        return Row {
            p,
            s0: None,
            model: "all".into(),
            result: Err(err),
        };
    };
    let result = normalization_point_counts(&forms[b].1, GENUS, kmax, opts);
    // Only exact counts are invariants of X; the reference rule may differ between models.
    if let (Ok(r), BranchRule::Exact) = (&result, opts.branch_rule) {
        let k = kmax.min(2);
        for (i, s) in statuses.iter().enumerate() {
            if i == b || !matches!(s, Ok(st) if status_rank(st) >= 3) {
                continue;
            }
            let other = normalization_point_counts(&forms[i].1, GENUS, k, opts)
                .expect("second model counts");
            if !r.counts.is_empty() && !other.counts.is_empty() {
                assert_eq!(
                    &r.counts[..k as usize],
                    &other.counts[..],
                    "models {} and {} disagree at p = {p}",
                    forms[b].0,
                    forms[i].0
                );
            }
        }
    }
    Row {
        p,
        s0: None,
        model: forms[b].0.clone(),
        result,
    }
}

/// `chi(p) = (-23/p)`.
pub fn chi(p: u64) -> i64 {
    Integer::from(-23).kronecker_symbol(&Integer::from(p)) as i64
}

/// The derived checks of one row with counts.
#[derive(Clone, Debug)]
pub struct Checks {
    pub frob: FrobeniusPolynomial,
    pub weil: bool,
    pub mod23: (i64, i64),
    pub p_mod2: String,
    pub q: Vec<String>,
    pub two_torsion: TwoTorsionOrder,
    pub predicted: Option<bool>,
}

pub fn checks(r: &NormalizationCounts) -> Option<Result<Checks, String>> {
    let fp = match r.frobenius_polynomial()? {
        Ok(f) => f,
        Err(e) => return Some(Err(e.to_string())),
    };
    let p = r.p;
    let c = chi(p);
    let m23 = Integer::from(23);
    let e1 = fp.eval_mod(&Integer::from(c), &m23).to_i64().unwrap_or(-1);
    let e2 = fp
        .eval_mod(&Integer::from(c * p as i64), &m23)
        .to_i64()
        .unwrap_or(-1);
    let p_mod2 = fp
        .factor_mod(2)
        .iter()
        .map(|(f, e)| {
            if *e == 1 {
                format!("({})", f2_poly_string(f))
            } else {
                format!("({})^{e}", f2_poly_string(f))
            }
        })
        .collect::<Vec<_>>()
        .join("");
    let q = fp
        .reciprocal_halves_mod2()
        .iter()
        .map(|c| f2_poly_string(c))
        .collect();
    Some(Ok(Checks {
        weil: fp.is_weil(),
        mod23: (e1, e2),
        p_mod2,
        q,
        two_torsion: fp.two_torsion_frobenius_order(),
        predicted: r.predicted_counts_agree(),
        frob: fp,
    }))
}

/// A printable line for a row.
pub fn format_row(row: &Row) -> String {
    let head = match row.s0 {
        Some(s0) => format!("p = {:3} s0 = {:3}", row.p, s0),
        None => format!("p = {:3}", row.p),
    };
    match &row.result {
        Err(e) => format!("{head} refused: {e}"),
        Ok(r) => {
            let rule = match r.branch_rule {
                BranchRule::Exact => "exact branches",
                BranchRule::AssumedUnibranch => "assumed unibranch",
            };
            let mut s = format!(
                "{head} model {} N = {:?} ({rule}) status: {}",
                row.model,
                r.counts,
                r.status.label()
            );
            if let Some(c) = checks(r) {
                match c {
                    Err(e) => s += &format!(" | P: {e}"),
                    Ok(c) => {
                        let a: Vec<String> = c.frob.coefficients()[1..=GENUS]
                            .iter()
                            .map(|x| x.to_string())
                            .collect();
                        let tt = match (&c.two_torsion, r.status.is_decided()) {
                            (TwoTorsionOrder::DeterminedByP(n), true) => format!("{n} (Decided)"),
                            (TwoTorsionOrder::DeterminedByP(n), false) => {
                                format!("{n} (conditional on the counts)")
                            }
                            (
                                TwoTorsionOrder::Unresolved {
                                    semisimple_order,
                                    divides,
                                },
                                _,
                            ) => {
                                format!("Unresolved (multiple of {semisimple_order}, divides {divides})")
                            }
                        };
                        s += &format!(
                            " | a = ({}) Weil {} | P(chi), P(chi p) mod 23 = {:?} | P mod 2 = {} | Q = {} | Frob order on J[2]: {tt}",
                            a.join(", "),
                            c.weil,
                            c.mod23,
                            c.p_mod2,
                            c.q.join(" or "),
                        );
                        if let Some(pr) = c.predicted {
                            s += &format!(" | counts beyond g predicted: {pr}");
                        }
                        if r.branch_rule == BranchRule::AssumedUnibranch && c.weil {
                            s += " | Unresolved: assumed unibranch; Weil test passed";
                        }
                    }
                }
            }
            s
        }
    }
}
