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
//!
//! C-5 and C-7 (`docs/workstreams/m23_golay/REQUESTS.md`; design C57): [`Jac2`] is the analysis
//! of Frobenius on `Jac[2]` of a row (`rustmath_curves::weil::TwoTorsionAnalysis`) with the
//! classes of `F` on the Golay Lagrangian `W` in `GL_4(2) = A_8` ([`GL42_CLASSES`]), and
//! [`row_json`] / [`table_json`] write the rows in the keys of the owner's `data/zeta_X.json`.
//! The W columns rest on the trusted hypothesis [`H_W`]; the order on `Jac[2]` does not.

#![allow(dead_code)]

use rustmath_curves::nodal_biprojective::{
    certify, normalization_point_counts, BihomogeneousForm, BranchRule, Certificate, CountError,
    CountOptions, FormModP, LocalType, NormalizationCounts, ReductionStatus, SingularLocus,
    Unresolved,
};
use rustmath_curves::weil::{
    f2_poly_string, FrobeniusPolynomial, RankBound, TwoTorsionAnalysis, TwoTorsionError,
    TwoTorsionOrder,
};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
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
///
/// The hypotheses of Lemma 9.1 (owner's paper, `\label{lem:reduction}`) that `certify` cannot
/// check here, and why they hold. Primitivity: HJLPPZ state in the header of
/// `polynomial_calF.m` that the coefficients of calF generate the unit ideal of `O_L`. The
/// prime: PARI gives `[O_L : Z[s]] = 1` (F-9 design, `wpf/gp/lfield.gp`), so `(p, s - s0)` is a
/// prime of `O_L` with residue field `F_p` and this is the reduction there. Lemma 9.1 is stated
/// over Q, but its proof uses only that the base is a discrete valuation ring with perfect
/// residue field. The ramified primes of L are 2, 11 and 23, and the only rows there,
/// (11, 2), (23, 5) and (23, 15), are refused; so every counted row is at an unramified prime.
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
    /// `certificate.singular_points` of the owner's classifier (`m23golay/curve.py` at
    /// `2d11038`): the number of geometric singular points of each type, with the labels
    /// `A_1`, `A_k`, `ordinary m-fold`.
    pub singular_points: Option<BTreeMap<String, usize>>,
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
            singular_points: r
                .get("certificate")
                .and_then(|c| c.get("singular_points"))
                .and_then(|c| c.as_object())
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| (k.clone(), v.as_u64().expect("number") as usize))
                        .collect()
                }),
        })
        .collect()
}

/// `zeta_X.json` (johnjanik/M23 at `2d11038`): 19 rows, p = 5..79 without 23, all "ok", each
/// with the owner's certificate. The row at 5 is the corrected one (F-8): at `d05475b` it was
/// skipped with the counts 8, 36, 170, 560 ("not a Weil polynomial").
pub fn zeta_x() -> Vec<RefRow> {
    ref_rows(&read_json("zeta_X.json"))
}

/// `zeta_L.json` rows and `check_53` (johnjanik/M23 at `2d11038`): 9 rows, four refused
/// ((5, 3), (11, 2), (23, 5), (23, 15)), and (31, 12) corrected (F-8; at `d05475b` it was
/// 44, 1014, 30266, 923726).
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

/// The rank of a status for model selection: Decided nodal > Decided delta-sum >
/// IrreducibilityNotShown > Unclassified > DeltaMismatch (a refusal ranks below all of them).
pub fn status_rank(s: &ReductionStatus) -> u8 {
    match s {
        ReductionStatus::Decided(Certificate::Nodal { .. }) => 5,
        ReductionStatus::Decided(Certificate::DeltaSum { .. }) => 4,
        ReductionStatus::Unresolved(Unresolved::IrreducibilityNotShown(_)) => 3,
        ReductionStatus::Unresolved(Unresolved::Unclassified(_)) => 2,
        ReductionStatus::Unresolved(Unresolved::DeltaMismatch { .. }) => 1,
    }
}

/// The singular points of a locus in the owner's labels (`m23golay/curve.py` at `2d11038`,
/// `resolve_point`): `A_1` for a node, `A_k`, `ordinary m-fold`, counted over `F_p-bar` (each
/// closed point with its residue degree).
pub fn owner_types(locus: &SingularLocus) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for s in &locus.points {
        let key = match &s.local {
            LocalType::Node => "A_1".to_string(),
            LocalType::A(k) => format!("A_{k}"),
            LocalType::OrdinaryMultiple(m) => format!("ordinary {m}-fold"),
            LocalType::Unclassified(_) => "unclassified".to_string(),
        };
        *out.entry(key).or_insert(0) += s.residue_degree;
    }
    out
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

/// The result of `certify` for one model.
pub type ModelStatus = Result<(ReductionStatus, SingularLocus), CountError>;

/// The status and singular locus of each model at p (cheap: `certify`), and the index of the
/// best one (None if all are refused). The first model wins among equal ranks.
///
/// Panics if a model gives `CountError::GenusInconsistent` (see [`assert_genus_consistent`]).
pub fn select_model(forms: &[(String, FormModP)]) -> (Vec<ModelStatus>, Option<usize>) {
    let statuses: Vec<ModelStatus> = forms
        .iter()
        .map(|(name, f)| {
            let status = certify(f, GENUS);
            assert_genus_consistent(&status, &format!("model {name} at p = {}", f.p()));
            status
        })
        .collect();
    let mut best: Option<(usize, u8)> = None;
    for (i, s) in statuses.iter().enumerate() {
        if let Ok((st, _)) = s {
            let r = status_rank(st);
            if best.is_none_or(|(_, b)| r > b) {
                best = Some((i, r));
            }
        }
    }
    (statuses, best.map(|b| b.0))
}

/// X at p from the CC0 models `[G, G7, G13]`: the first model with the best status. The models
/// are reduced by `reduce_primitive` (Lemma 9.1 needs a primitive form). When two models are
/// both Decided, their counts at `k <= min(kmax, 2)` are checked to agree (they must: both are
/// the counts of the reduction of X).
pub fn x_row(
    models: &[(&'static str, BihomogeneousForm)],
    p: u64,
    kmax: u32,
    opts: &CountOptions,
) -> Row {
    let mut forms = Vec::new();
    let mut first_err = None;
    for (name, g) in models {
        match g.reduce_primitive(p) {
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
    // The counts of a Decided model are those of the reduction of X, so two Decided models agree.
    if let Ok(r) = &result {
        let k = kmax.min(2);
        for (i, s) in statuses.iter().enumerate() {
            if i == b || !r.status.is_decided() || !matches!(s, Ok((st, _)) if st.is_decided()) {
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
    /// The analysis of `Jac[2]` (C-5), with the status of the row.
    pub jac2: Jac2,
}

/// The derived checks of a row with counts (the order on `Jac[2]` is Decided only on a Decided
/// row).
pub fn checks(r: &NormalizationCounts) -> Option<Result<Checks, String>> {
    let mut c = checks_of_counts(r.p, &r.counts)?;
    if let Ok(c) = &mut c {
        c.jac2.row_decided = r.status.is_decided();
    }
    Some(c)
}

/// The derived checks of counts `N_1..N_k` of a genus-4 curve at p, `k >= 4` (`None` below):
/// those of [`checks`], for counts that do not come from a [`NormalizationCounts`] (the reference
/// rule of [`NormalizationCounts::counts_under`]).
pub fn checks_of_counts(p: u64, counts: &[u64]) -> Option<Result<Checks, String>> {
    if counts.len() < GENUS {
        return None;
    }
    let fp = match FrobeniusPolynomial::from_counts_u64(p, &counts[..GENUS]) {
        Ok(f) => f,
        Err(e) => return Some(Err(e.to_string())),
    };
    let predicted = (counts.len() > GENUS).then(|| {
        let c: Vec<Integer> = counts.iter().map(|&n| Integer::from(n)).collect();
        fp.predicts(&c)
    });
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
    let weil = fp.is_weil();
    // One analysis per row (C57 review 1, m3): `two_torsion` is the order of `jac2`, which is
    // `two_torsion_frobenius_order()`.
    let jac2 = jac2_of(&fp, weil, false);
    Some(Ok(Checks {
        weil,
        mod23: (e1, e2),
        p_mod2,
        q,
        two_torsion: jac2.order.clone(),
        predicted,
        jac2,
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
            if let Some(c) = r.status.certificate() {
                s += &format!(
                    " | irreducibility: {} | types {:?}",
                    c.irreducibility(),
                    owner_types(&r.locus)
                );
            }
            if let Some(c) = checks(r) {
                match c {
                    Err(e) => s += &format!(" | P: {e}"),
                    Ok(c) => {
                        let a: Vec<String> = c.frob.coefficients()[1..=GENUS]
                            .iter()
                            .map(|x| x.to_string())
                            .collect();
                        s += &format!(
                            " | a = ({}) Weil {} | P(chi), P(chi p) mod 23 = {:?} | P mod 2 = {} | Q = {}",
                            a.join(", "),
                            c.weil,
                            c.mod23,
                            c.p_mod2,
                            c.q.join(" or "),
                        );
                        s += &format!(
                            " | Frob order on J[2]: {} | J[2] at 1: {} | W at 1: {}",
                            c.jac2.order_text(),
                            c.jac2.j_text(),
                            c.jac2.w_text(),
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

// ------------------------------------------------------------------------------------------
// C-5: Frobenius on Jac[2] and on the Golay Lagrangian W (design C57).
// ------------------------------------------------------------------------------------------

/// The trusted hypothesis of the W columns (design C57 2.5): `Jac[2]` has an `F`-stable
/// Lagrangian `W` on which `F` has characteristic polynomial `Q` or `Q^rev`.
pub const H_W: &str = "W: Jac[2] has an F-stable Lagrangian subspace on which F has \
characteristic polynomial Q or Q^rev (HJLPPZ Thm. 4.8 and Cor. 4.4; Gal = M23, branch data \
(2A, 23A, 23B)); used by the W columns only";

/// The genus input of every certificate.
pub const H_GENUS: &str = "genus 4 (HJLPPZ Prop. 3.10)";

/// A conjugacy class of `GL_4(2) = A_8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gl42Class {
    /// The characteristic polynomial (as `f2_poly_string`).
    pub charpoly: &'static str,
    pub order: u64,
    /// `dim ker(g - 1)`.
    pub dim_fixed: usize,
    pub class_size: u64,
    /// The cycle type on the 15 points of `PG(3, 2)` (equal to the one on the hyperplanes).
    pub cycle_type_15: &'static str,
    /// The cycle type on 8 letters through `GL_4(2) = A_8`.
    pub cycle_type_8: &'static str,
}

const fn cl(
    charpoly: &'static str,
    order: u64,
    dim_fixed: usize,
    class_size: u64,
    cycle_type_15: &'static str,
    cycle_type_8: &'static str,
) -> Gl42Class {
    Gl42Class {
        charpoly,
        order,
        dim_fixed,
        class_size,
        cycle_type_15,
        cycle_type_8,
    }
}

/// The 14 classes of `GL_4(2)`: GAP 4.12.1 (`tests/data/two_torsion/gl42_classes.g`,
/// `gl42_classes.txt`), equal in every column to the `classes` of the owner's
/// `data/a8_frobenius_predictions.json` (johnjanik/M23 at `e0d25b8`) and to the table of
/// REQUESTS 1.2. The key `(charpoly, order, dim_fixed)` is unique. The 15-point type is also
/// computed here by brute force ([`cycle_type_on_points`]); the 8-letter type is only looked up
/// (a class invariant computed by GAP and by the owner's script).
pub const GL42_CLASSES: [Gl42Class; 14] = [
    cl("x^4+1", 1, 4, 1, "1^15", "1^8"),
    cl("x^4+1", 2, 3, 105, "1^7 2^4", "2^4"),
    cl("x^4+1", 2, 2, 210, "1^3 2^6", "1^4 2^2"),
    cl("x^4+1", 4, 2, 1260, "1^3 2^2 4^2", "4^2"),
    cl("x^4+1", 4, 1, 2520, "1 2 4^3", "1^2 2 4"),
    cl("x^4+x^2+1", 3, 0, 112, "3^5", "1^5 3"),
    cl("x^4+x^2+1", 6, 0, 1680, "3 6^2", "1 2^2 3"),
    cl("x^4+x^3+1", 15, 0, 1344, "15", "3 5"),
    cl("x^4+x+1", 15, 0, 1344, "15", "3 5"),
    cl("x^4+x^3+x^2+x+1", 5, 0, 1344, "5^3", "1^3 5"),
    cl("x^4+x^3+x+1", 3, 2, 1120, "1^3 3^4", "1^2 3^2"),
    cl("x^4+x^3+x+1", 6, 1, 3360, "1 2 3^2 6", "2 6"),
    cl("x^4+x^2+x+1", 7, 1, 2880, "1 7^2", "1 7"),
    cl("x^4+x^3+x^2+1", 7, 1, 2880, "1 7^2", "1 7"),
];

/// The class with this key, if any.
pub fn gl42_class(charpoly: &str, order: u64, dim_fixed: usize) -> Option<&'static Gl42Class> {
    GL42_CLASSES
        .iter()
        .find(|c| c.charpoly == charpoly && c.order == order && c.dim_fixed == dim_fixed)
}

/// `M v` over `F_2`: `m[j]` is the image of the basis vector `e_j`, vectors are bit masks.
pub fn apply(m: &[u32], v: u32) -> u32 {
    m.iter()
        .enumerate()
        .filter(|(j, _)| (v >> j) & 1 == 1)
        .fold(0, |acc, (_, c)| acc ^ c)
}

/// The rational canonical form of the module `sum_f sum_i F_2[x]/(f^(mu_i))`: one companion
/// block of `f^a` for each part `a` (`parts`: `(f ascending bits, mu_f)`).
pub fn canonical_form(parts: &[(Vec<i64>, Vec<u32>)]) -> Vec<u32> {
    let mut m: Vec<u32> = Vec::new();
    for (f, mu) in parts {
        for &a in mu {
            let mut h = vec![1i64];
            for _ in 0..a {
                h = fp_factor::mul(&h, f, 2);
            }
            let h = fp_factor::trim(&h);
            let n = h.len() - 1;
            let start = m.len();
            for k in 0..n {
                m.push(if k + 1 < n {
                    1u32 << (start + k + 1)
                } else {
                    (0..n).fold(0u32, |acc, i| acc | (((h[i] & 1) as u32) << (start + i)))
                });
            }
        }
    }
    m
}

/// The order of an invertible matrix over `F_2` of size `dim`.
pub fn matrix_order(m: &[u32], dim: usize) -> u64 {
    let ident: Vec<u32> = (0..dim).map(|i| 1u32 << i).collect();
    let mut pw = m.to_vec();
    let mut n = 1u64;
    while pw != ident {
        pw = pw.iter().map(|&c| apply(m, c)).collect();
        n += 1;
        assert!(n <= 1 << dim, "matrix_order: not invertible");
    }
    n
}

/// A cycle type `1 7^2` (lengths ascending, multiplicities as exponents).
fn cycle_string(lengths: &[usize]) -> String {
    let mut c: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in lengths {
        *c.entry(l).or_insert(0) += 1;
    }
    c.iter()
        .map(|(l, m)| {
            if *m == 1 {
                l.to_string()
            } else {
                format!("{l}^{m}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The cycle type of `m` on the `2^dim - 1` points of `PG(dim - 1, 2)` (brute force).
pub fn cycle_type_on_points(m: &[u32], dim: usize) -> String {
    let n = 1usize << dim;
    let mut seen = vec![false; n];
    let mut lengths = Vec::new();
    for v in 1..n {
        if seen[v] {
            continue;
        }
        let mut len = 0;
        let mut x = v;
        while !seen[x] {
            seen[x] = true;
            len += 1;
            x = apply(m, x as u32) as usize;
        }
        lengths.push(len);
    }
    cycle_string(&lengths)
}

/// Every module type with characteristic polynomial `cp` (ascending bits, `cp(0) = 1`): one
/// partition of the multiplicity of each irreducible factor.
pub fn module_types(cp: &[i64]) -> Vec<Vec<(Vec<i64>, Vec<u32>)>> {
    let mut out: Vec<Vec<(Vec<i64>, Vec<u32>)>> = vec![Vec::new()];
    for (f, e) in fp_factor::factor_with_multiplicity(cp, 2) {
        let parts: Vec<Vec<u32>> = rustmath_combinatorics::partitions(e as usize)
            .iter()
            .map(|p| p.parts().iter().map(|&x| x as u32).collect())
            .collect();
        let mut next = Vec::new();
        for prefix in &out {
            for mu in &parts {
                let mut v = prefix.clone();
                v.push((fp_factor::trim(&f), mu.clone()));
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// The bit mask of a polynomial over `F_2` (ascending bits).
fn mask(f: &[i64]) -> u64 {
    f.iter()
        .enumerate()
        .fold(0, |acc, (i, &c)| acc | (((c & 1) as u64) << i))
}

/// The `Q` candidates in the owner's order (bit mask descending, as `zeta_X.json`).
pub fn q_candidates(fp: &FrobeniusPolynomial) -> Vec<Vec<i64>> {
    let mut q = fp.reciprocal_halves_mod2();
    q.sort_by_key(|f| std::cmp::Reverse(mask(f)));
    q
}

/// The factors of `P mod 2` in the owner's order (degree, then bit mask, ascending).
pub fn p_mod2_owner(fp: &FrobeniusPolynomial) -> Vec<(Vec<i64>, u32)> {
    let mut f = fp.factor_mod(2);
    f.sort_by_key(|(g, _)| (fp_factor::degree(g), mask(g)));
    f
}

/// A partition as `[2, 1, 1]`, or `none`.
fn part_text(p: &[u32]) -> String {
    if p.is_empty() {
        "none".to_string()
    } else {
        format!("{p:?}")
    }
}

/// One class of `F` on `W`: a `Q` and a type of `W` compatible with some candidate of `Jac[2]`.
#[derive(Clone, Debug)]
pub struct WClass {
    /// The characteristic polynomial of `F` on `W` (one of the `Q` candidates).
    pub q: String,
    /// `(f, mu_f)` for the factors of `Q`.
    pub w_module: Vec<(String, Vec<u32>)>,
    /// `mu_(x+1)`, empty when `x + 1` does not divide `Q`.
    pub w_type_one: Vec<u32>,
    pub order: u64,
    pub fixed_dimension: usize,
    /// Genus 4: the cycle type on the 15 points (computed), and the class of `GL_4(2)`.
    pub cycle_type_15: Option<String>,
    pub class: Option<Gl42Class>,
    /// The types of `Jac[2]` at 1 of the candidates this type is compatible with.
    pub j_types_one: Vec<Vec<u32>>,
}

/// The analysis of `Jac[2]` of one row (C-5).
#[derive(Clone, Debug)]
pub struct Jac2 {
    pub analysis: Result<TwoTorsionAnalysis, TwoTorsionError>,
    pub order: TwoTorsionOrder,
    /// The exact Weil test of `P`.
    pub weil: bool,
    /// The row is Decided (its counts are certified by Lemma 9.1 of the owner's paper, F-9): a
    /// decided order is then a statement about the curve.
    pub row_decided: bool,
    /// One entry per (`Q`, type of `W`), `Q` in the owner's order.
    pub w: Vec<WClass>,
    /// Why the W columns are missing: an error of `lagrangian_types`, or no compatible type (a
    /// contradiction with [`H_W`]).
    pub w_error: Option<String>,
}

/// The analysis of `Jac[2]` from `P` (the row status is set by the caller).
///
/// Panics if the 15-point cycle type computed from the canonical form differs from the table's
/// (a fault in the code or the table, not a property of the row).
pub fn jac2_of(fp: &FrobeniusPolynomial, weil: bool, row_decided: bool) -> Jac2 {
    let analysis = fp.two_torsion_analysis();
    // two_torsion_frobenius_order() is analysis.order() when the analysis is made; only its
    // fallback (the bound of P mod 2 alone) needs the second call.
    let order = match &analysis {
        Ok(a) => a.order(),
        Err(_) => fp.two_torsion_frobenius_order(),
    };
    let mut w = Vec::new();
    let mut w_error = None;
    if let Ok(a) = &analysis {
        let one = a.factors.iter().position(|f| f.is_x_plus_1());
        let qs = q_candidates(fp);
        for q in &qs {
            match a.lagrangian_types(q) {
                Err(e) => w_error = Some(e.to_string()),
                Ok(types) => {
                    for t in types {
                        let mut j1: BTreeSet<Vec<u32>> = BTreeSet::new();
                        if let Some(i) = one {
                            for &c in &t.compatible {
                                j1.insert(a.candidates[c].0[i].clone());
                            }
                        }
                        let (mut c15, mut class) = (None, None);
                        if fp.genus() == 4 {
                            let m = canonical_form(&t.parts);
                            assert_eq!(matrix_order(&m, 4), t.order(), "order on W");
                            let c = cycle_type_on_points(&m, 4);
                            let k = gl42_class(&f2_poly_string(q), t.order(), t.fixed_dimension())
                                .unwrap_or_else(|| panic!("no class of GL_4(2) for {:?}", t.parts));
                            assert_eq!(c, k.cycle_type_15, "15-point type of {:?}", t.parts);
                            c15 = Some(c);
                            class = Some(*k);
                        }
                        w.push(WClass {
                            q: f2_poly_string(q),
                            w_module: t
                                .parts
                                .iter()
                                .map(|(f, mu)| (f2_poly_string(f), mu.clone()))
                                .collect(),
                            w_type_one: t.eigenvalue_one(),
                            order: t.order(),
                            fixed_dimension: t.fixed_dimension(),
                            cycle_type_15: c15,
                            class,
                            j_types_one: j1.into_iter().rev().collect(),
                        });
                    }
                }
            }
        }
        if w.is_empty() && w_error.is_none() && !qs.is_empty() {
            w_error = Some(
                "no type of W is compatible with a candidate of Jac[2]: this contradicts H_W"
                    .to_string(),
            );
        }
    }
    Jac2 {
        analysis,
        order,
        weil,
        row_decided,
        w,
        w_error,
    }
}

/// A rank bound with `F_p` for `F_q` (the base field of every M23 row is `F_p`).
pub fn bound_label(b: &RankBound) -> String {
    b.label().replace("F_q", "F_p")
}

impl Jac2 {
    /// The order on `Jac[2]` when every candidate has the same one.
    pub fn decided_order(&self) -> Option<u64> {
        match &self.order {
            TwoTorsionOrder::DeterminedByP(n) => Some(*n),
            TwoTorsionOrder::DeterminedByTwoRank { order, .. } => Some(*order),
            TwoTorsionOrder::Unresolved { .. } => None,
        }
    }

    /// The status word for a quantity with `n` possible values.
    fn status_of(&self, n: usize) -> String {
        if !self.weil {
            "not applicable (P is not a Weil polynomial)".to_string()
        } else if let Err(e) = &self.analysis {
            format!("error: {e}")
        } else if n == 1 {
            if self.row_decided {
                "Decided".to_string()
            } else {
                "conditional on the counts".to_string()
            }
        } else {
            "Unresolved".to_string()
        }
    }

    /// `Decided`, `conditional on the counts` (decided for `P`, the row is not Decided),
    /// `Unresolved`, `not applicable (P is not a Weil polynomial)` or `error: ...`.
    pub fn order_status(&self) -> String {
        let n = if self.decided_order().is_some() { 1 } else { 2 };
        self.status_of(n)
    }

    /// `P mod 2 squarefree`, or the decisive bounds joined by `; `, or empty.
    pub fn reason(&self) -> String {
        match &self.order {
            TwoTorsionOrder::DeterminedByP(_) => "P mod 2 squarefree".to_string(),
            TwoTorsionOrder::DeterminedByTwoRank { bounds, .. } => bounds
                .iter()
                .map(bound_label)
                .collect::<Vec<_>>()
                .join("; "),
            TwoTorsionOrder::Unresolved { .. } => String::new(),
        }
    }

    fn orders(&self) -> Vec<u64> {
        match &self.order {
            TwoTorsionOrder::DeterminedByP(n) => vec![*n],
            TwoTorsionOrder::DeterminedByTwoRank { order, .. } => vec![*order],
            TwoTorsionOrder::Unresolved { candidates, .. } => candidates.clone(),
        }
    }

    /// What the analysis tried (REQUESTS Section 4: `Unresolved` with what was tried; C57
    /// review 2, n1): `tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | 14`, then the
    /// bounds skipped because only `v2 >= 128` was seen, if any. None without an analysis.
    pub fn tried(&self) -> Option<String> {
        let a = self.analysis.as_ref().ok()?;
        let mut s = format!(
            "tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | {}",
            a.max_order
        );
        if !a.skipped_bounds.is_empty() {
            let skipped: Vec<String> = a
                .skipped_bounds
                .iter()
                .map(|&(k, twisted)| {
                    let field = if k == 1 {
                        "F_p".to_string()
                    } else {
                        format!("F_p^{k}")
                    };
                    format!("#Jac{}({field})", if twisted { "'" } else { "" })
                })
                .collect();
            s += &format!("; skipped, only v2 >= 128 seen: {}", skipped.join(", "));
        }
        Some(s)
    }

    /// The text column of the order: `14 (Decided; v2 #Jac(F_p) = 1)`, `Unresolved: 7 or 14
    /// (tried: v2 #Jac(F_p^k) and v2 #Jac'(F_p^k) for every k | 14)`.
    pub fn order_text(&self) -> String {
        let status = self.order_status();
        if !self.weil || self.analysis.is_err() {
            return status;
        }
        match self.decided_order() {
            Some(n) => format!("{n} ({status}; {})", self.reason()),
            None => {
                let orders = self
                    .orders()
                    .iter()
                    .map(|o| o.to_string())
                    .collect::<Vec<_>>()
                    .join(" or ");
                match self.tried() {
                    Some(t) => format!("Unresolved: {orders} ({t})"),
                    None => format!("Unresolved: {orders}"),
                }
            }
        }
    }

    /// The types of `Jac[2]` at the eigenvalue 1 (None when `x + 1` does not divide `P mod 2`).
    pub fn j_types_one(&self) -> Option<Vec<Vec<u32>>> {
        self.analysis.as_ref().ok()?.eigenvalue_one_types()
    }

    /// `[2] or [1, 1]`, `none`.
    pub fn j_text(&self) -> String {
        match self.j_types_one() {
            None => "none".to_string(),
            Some(ts) => ts
                .iter()
                .map(|t| part_text(t))
                .collect::<Vec<_>>()
                .join(" or "),
        }
    }

    /// The types of `W` at the eigenvalue 1, distinct, reverse lexicographic (`[]` for none).
    pub fn w_types_one(&self) -> Vec<Vec<u32>> {
        let s: BTreeSet<Vec<u32>> = self.w.iter().map(|c| c.w_type_one.clone()).collect();
        s.into_iter().rev().collect()
    }

    /// The pairs (cycle type on 8 letters, on 15 points) of the classes of `F` on `W`, distinct,
    /// sorted (genus 4 only).
    pub fn w_cycle_types(&self) -> Vec<(String, String)> {
        let s: BTreeSet<(String, String)> = self
            .w
            .iter()
            .filter_map(|c| Some((c.class?.cycle_type_8.to_string(), c.cycle_type_15.clone()?)))
            .collect();
        s.into_iter().collect()
    }

    pub fn w_type_status(&self) -> String {
        if self.w.is_empty() {
            return self
                .w_error
                .clone()
                .map_or("none".to_string(), |e| format!("error: {e}"));
        }
        self.status_of(self.w_types_one().len())
    }

    pub fn w_cycle_status(&self) -> String {
        if self.w.is_empty() {
            return self
                .w_error
                .clone()
                .map_or("none".to_string(), |e| format!("error: {e}"));
        }
        self.status_of(self.w_cycle_types().len())
    }

    /// `[1] (Decided); cycle types 1 7 (8 letters), 1 7^2 (15 points) (Decided; the class
    /// depends on Q or Q^rev)`. The status is that of the cycle types; when the two `Q` differ,
    /// the class of `GL_4(2)` (two classes of order 7, or of order 15) is not decided.
    pub fn w_text(&self) -> String {
        if self.w.is_empty() {
            return self.w_error.clone().unwrap_or_else(|| "none".to_string());
        }
        let types = self
            .w_types_one()
            .iter()
            .map(|t| part_text(t))
            .collect::<Vec<_>>()
            .join(" or ");
        // Both lists in the order of the pairs (a class determines both cycle types).
        let ct = self.w_cycle_types();
        let mut c8: Vec<&str> = ct.iter().map(|c| c.0.as_str()).collect();
        c8.dedup();
        let mut c15: Vec<&str> = ct.iter().map(|c| c.1.as_str()).collect();
        c15.dedup();
        let mut s = if self.w_types_one() == vec![Vec::<u32>::new()] {
            "none".to_string()
        } else {
            format!("{types} ({})", self.w_type_status())
        };
        if !ct.is_empty() {
            let qs: BTreeSet<&str> = self.w.iter().map(|c| c.q.as_str()).collect();
            s += &format!(
                "; cycle types {} (8 letters), {} (15 points) ({}{})",
                c8.join(" or "),
                c15.join(" or "),
                self.w_cycle_status(),
                if qs.len() > 1 {
                    "; the class depends on Q or Q^rev"
                } else {
                    ""
                }
            );
        }
        s
    }

    /// The `jac2` object of the JSON rows (design C57 8.2).
    pub fn json(&self) -> Value {
        let a = match &self.analysis {
            Err(e) => {
                return json!({
                    "error": e.to_string(),
                    "order_status": self.order_status(),
                    "order_candidates": self.orders(),
                })
            }
            Ok(a) => a,
        };
        let bound = |b: &RankBound| json!({"k": b.k, "twisted": b.twisted, "v2": b.v2});
        let decisive: Vec<Value> = match &self.order {
            TwoTorsionOrder::DeterminedByTwoRank { bounds, .. } => {
                bounds.iter().map(bound).collect()
            }
            _ => Vec::new(),
        };
        let qs: BTreeSet<&str> = self.w.iter().map(|c| c.q.as_str()).collect();
        let w: Vec<Value> = self
            .w
            .iter()
            .map(|c| {
                json!({
                    "Q": c.q,
                    "W_type_eigenvalue_1": c.w_type_one,
                    "W_module": c.w_module.iter().map(|(f, mu)| json!([f, mu])).collect::<Vec<_>>(),
                    "order_on_W": c.order,
                    "fixed_dimension": c.fixed_dimension,
                    "cycle_type_8": c.class.map(|k| k.cycle_type_8),
                    "cycle_type_15": c.cycle_type_15,
                    "class_size": c.class.map(|k| k.class_size),
                    "J_types_eigenvalue_1": c.j_types_one,
                })
            })
            .collect();
        let j1 = a.eigenvalue_one_types();
        let skipped: Vec<Value> = a
            .skipped_bounds
            .iter()
            .map(|(k, t)| json!({"k": k, "twisted": t}))
            .collect();
        let cycle_types: Vec<Value> = self
            .w_cycle_types()
            .iter()
            .map(|(a, b)| json!([a, b]))
            .collect();
        json!({
            "order": self.decided_order(),
            "order_candidates": self.orders(),
            "order_status": self.order_status(),
            "reason": self.reason(),
            "decisive_bounds": decisive,
            "bounds": a.bounds.iter().map(bound).collect::<Vec<_>>(),
            "skipped_bounds": skipped,
            "semisimple_order": a.semisimple_order,
            "max_order": a.max_order,
            "candidate_count": a.candidates.len(),
            "symplectic_removed": a.symplectic_removed,
            "J_type_eigenvalue_1": j1,
            "J_type_eigenvalue_1_status": self.status_of(j1.as_ref().map_or(1, |v| v.len())),
            "W": w,
            "W_type_eigenvalue_1": self.w_types_one(),
            "W_type_eigenvalue_1_status": self.w_type_status(),
            "W_cycle_types": cycle_types,
            "W_cycle_types_status": self.w_cycle_status(),
            "W_Q_status": if qs.len() > 1 {
                "Q and Q^rev differ; which one is the characteristic polynomial on W is not decided"
            } else {
                "Q = Q^rev"
            },
            "W_error": self.w_error,
            "W_hypothesis": H_W,
        })
    }
}

// ------------------------------------------------------------------------------------------
// C-7: the JSON rows (design C57 Section 8).
// ------------------------------------------------------------------------------------------

/// An integer as a JSON number when it fits in an `i64`, else as a decimal string.
fn int_json(x: &Integer) -> Value {
    x.to_i64()
        .map_or_else(|| Value::String(x.to_string()), |v| json!(v))
}

/// One row in the keys of the owner's `data/zeta_X.json` (and `s0` for L rows), with RustMath's
/// keys (`model`, `rustmath_status`, `irreducibility`, `jac2`, ...) and the legacy keys `a`,
/// `P_mod_2`, `Q` of `data/rustmath_wpf_tables.json` (design C57 8.2).
pub fn row_json(row: &Row, time_s: Option<f64>) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("p".into(), json!(row.p));
    if let Some(s0) = row.s0 {
        o.insert("s0".into(), json!(s0));
    }
    o.insert("model".into(), json!(row.model));
    match &row.result {
        Err(e) => {
            o.insert("status".into(), json!("refused"));
            o.insert("reason".into(), json!(e.to_string()));
            o.insert("rustmath_status".into(), json!(format!("refused: {e}")));
            o.insert("trusted_hypotheses".into(), json!([H_GENUS]));
        }
        Ok(r) => {
            let counted = !r.counts.is_empty();
            o.insert(
                "status".into(),
                json!(if counted { "ok" } else { "not counted" }),
            );
            o.insert(
                "certificate".into(),
                json!({
                    "certified": r.status.is_decided(),
                    "singular_points": owner_types(&r.locus),
                    "delta": r.locus.delta_sum(),
                }),
            );
            let g = r.counts.len().min(GENUS);
            o.insert("counts".into(), json!(&r.counts[..g]));
            if r.counts.len() > GENUS {
                o.insert("counts_beyond_g".into(), json!(&r.counts[GENUS..]));
                o.insert(
                    "predicted_counts_agree".into(),
                    json!(r.predicted_counts_agree()),
                );
            }
            o.insert("rustmath_status".into(), json!(r.status.label()));
            o.insert("decided".into(), json!(r.status.is_decided()));
            o.insert(
                "branch_rule".into(),
                json!(match r.branch_rule {
                    BranchRule::Exact => "exact branches",
                    BranchRule::AssumedUnibranch => "assumed unibranch",
                }),
            );
            o.insert(
                "irreducibility".into(),
                json!(r
                    .status
                    .certificate()
                    .map(|c| c.irreducibility().to_string())),
            );
            o.insert("P".into(), Value::Null);
            o.insert("jac2".into(), Value::Null);
            match checks(r) {
                None => {}
                Some(Err(e)) => {
                    o.insert("P_error".into(), json!(e));
                }
                Some(Ok(c)) => {
                    let fp = &c.frob;
                    o.insert(
                        "P".into(),
                        Value::Array(fp.coefficients().iter().map(int_json).collect()),
                    );
                    o.insert("chi_minus23".into(), json!(chi(row.p)));
                    o.insert("P(chi) mod 23".into(), json!(c.mod23.0));
                    o.insert("P(chi p) mod 23".into(), json!(c.mod23.1));
                    o.insert(
                        "P mod 2".into(),
                        Value::Array(
                            p_mod2_owner(fp)
                                .iter()
                                .map(|(f, e)| json!([f2_poly_string(f), e]))
                                .collect(),
                        ),
                    );
                    let qs: Vec<String> =
                        q_candidates(fp).iter().map(|q| f2_poly_string(q)).collect();
                    o.insert("golay_split".into(), json!(!qs.is_empty()));
                    o.insert(
                        "a".into(),
                        Value::Array(fp.coefficients()[1..=GENUS].iter().map(int_json).collect()),
                    );
                    o.insert("P_mod_2".into(), json!(c.p_mod2));
                    if let Some(q) = qs.first() {
                        o.insert("Q".into(), json!(q));
                    }
                    o.insert("Q candidates".into(), json!(qs));
                    o.insert("weil".into(), json!(c.weil));
                    o.insert("jac2".into(), c.jac2.json());
                }
            }
            o.insert("trusted_hypotheses".into(), json!([H_GENUS, H_W]));
        }
    }
    if let Some(t) = time_s {
        o.insert("time_s".into(), json!((t * 10.0).round() / 10.0));
    }
    Value::Object(o)
}

/// The provenance of a run (C-7: the commit and the command in the file).
#[derive(Clone, Debug)]
pub struct Provenance {
    /// `X` or `L`.
    pub cover: String,
    pub models: Vec<String>,
    pub argv: Vec<String>,
    pub pmin: u64,
    pub pmax: u64,
    pub kmax: u32,
    pub threads: Option<usize>,
    pub unix_time: u64,
    /// `git rev-parse HEAD` of this checkout when the run started.
    pub commit_at_run: Option<String>,
    /// `git status --porcelain --untracked-files=no` was empty.
    pub tree_clean_at_run: Option<bool>,
    pub commit_time: Option<i64>,
    pub executable_mtime: Option<i64>,
    /// The checkout of `shaowuz/m23isgalois` read at run time, and its commit.
    pub m23isgalois: Option<(String, Option<String>)>,
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

impl Provenance {
    /// Collect the provenance at the start of a run.
    pub fn collect(
        cover: &str,
        models: Vec<String>,
        params: (u64, u64, u32),
        threads: Option<usize>,
        m23isgalois: Option<&Path>,
    ) -> Provenance {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let unix_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let executable_mtime = std::env::current_exe()
            .ok()
            .and_then(|p| std::fs::metadata(p).ok())
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64);
        Provenance {
            cover: cover.to_string(),
            models,
            argv: std::env::args().collect(),
            pmin: params.0,
            pmax: params.1,
            kmax: params.2,
            threads,
            unix_time,
            commit_at_run: git(here, &["rev-parse", "HEAD"]),
            tree_clean_at_run: git(here, &["status", "--porcelain", "--untracked-files=no"])
                .map(|s| s.is_empty()),
            commit_time: git(here, &["log", "-1", "--format=%ct"]).and_then(|s| s.parse().ok()),
            executable_mtime,
            m23isgalois: m23isgalois
                .map(|d| (d.display().to_string(), git(d, &["rev-parse", "HEAD"]))),
        }
    }
}

/// The JSON file of a run (design C57 8.1): provenance, trusted hypotheses, and the rows under
/// the key `X` or `L` (the owner's scripts read `json.load(...)["X"]`).
pub fn table_json(prov: &Provenance, rows: &[Value], complete: bool) -> Value {
    let args: Vec<&str> = prov.argv.iter().skip(1).map(|s| s.as_str()).collect();
    let release = !cfg!(debug_assertions);
    let command = format!(
        "cargo run{} -p rustmath-curves --example m23_zeta -- {}",
        if release { " --release" } else { "" },
        args.join(" ")
    );
    let mut o = serde_json::Map::new();
    o.insert("format".into(), json!("rustmath-m23-zeta/1"));
    o.insert(
        "source".into(),
        json!("RustMath, rustmath-curves example m23_zeta"),
    );
    o.insert("rustmath_commit_at_run".into(), json!(prov.commit_at_run));
    o.insert(
        "rustmath_tree_clean_at_run".into(),
        json!(prov.tree_clean_at_run),
    );
    o.insert("rustmath_commit_time".into(), json!(prov.commit_time));
    o.insert("executable_mtime".into(), json!(prov.executable_mtime));
    o.insert(
        "provenance_note".into(),
        json!(
            "rustmath_commit_at_run is the commit of the source tree when the run started (git \
             rev-parse HEAD); the executable is the one cargo built from that tree for this run \
             (compare executable_mtime with rustmath_commit_time and the tree state). command is \
             reconstructed from argv; argv is exact."
        ),
    );
    o.insert("command".into(), json!(command));
    o.insert("argv".into(), json!(prov.argv));
    o.insert(
        "profile".into(),
        json!(if release { "release" } else { "debug" }),
    );
    o.insert("unix_time".into(), json!(prov.unix_time));
    o.insert("cover".into(), json!(prov.cover));
    o.insert("models".into(), json!(prov.models));
    o.insert(
        "m23isgalois".into(),
        match &prov.m23isgalois {
            None => Value::Null,
            Some((p, c)) => json!({"path": p, "commit": c}),
        },
    );
    o.insert("genus".into(), json!(GENUS));
    o.insert("pmin".into(), json!(prov.pmin));
    o.insert("pmax".into(), json!(prov.pmax));
    o.insert("kmax".into(), json!(prov.kmax));
    o.insert("threads".into(), json!(prov.threads));
    o.insert("trusted_hypotheses".into(), json!([H_GENUS, H_W]));
    o.insert("complete".into(), json!(complete));
    o.insert(prov.cover.clone(), Value::Array(rows.to_vec()));
    Value::Object(o)
}
