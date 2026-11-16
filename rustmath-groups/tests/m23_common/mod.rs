//! Readers for the M23 test data (WP-P), shared by the tests (`mod m23_common;`) and by the
//! example `m23_class_vectors` (`#[path = "../tests/m23_common/mod.rs"] mod m23_common;`).
//!
//! The owner's JSON files are vendored in `tests/data/m23/` (README there: sources, sha256,
//! licences). The HJLPPZ file and Le Lay's certificate are never vendored: the `env_` tests
//! read them at run time from `$M23ISGALOIS` and `$LELAY_M23`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use rustmath_groups::m23::group::from_cycles_1based;
use rustmath_groups::m23::{M23Class, Perm23};

/// The vendored data directory `tests/data/m23`.
pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/m23")
}

/// Reads and parses a JSON file; the error names the path.
pub fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// A vendored file of `tests/data/m23`, parsed; panics with the path on failure.
pub fn vendored(name: &str) -> serde_json::Value {
    read_json(&data_dir().join(name)).unwrap_or_else(|e| panic!("{e}"))
}

fn field<'a>(
    v: &'a serde_json::Value,
    key: &str,
    what: &str,
) -> Result<&'a serde_json::Value, String> {
    v.get(key).ok_or_else(|| format!("{what}: no field {key}"))
}

fn as_u64(v: &serde_json::Value, key: &str, what: &str) -> Result<u64, String> {
    field(v, key, what)?
        .as_u64()
        .ok_or_else(|| format!("{what}: {key} is not an unsigned integer"))
}

fn classes_of(v: &serde_json::Value, what: &str) -> Result<[M23Class; 3], String> {
    let arr = field(v, "classes", what)?
        .as_array()
        .ok_or_else(|| format!("{what}: classes is not a list"))?;
    if arr.len() != 3 {
        return Err(format!("{what}: {} classes", arr.len()));
    }
    let mut out = [M23Class::C1A; 3];
    for (o, c) in out.iter_mut().zip(arr) {
        let s = c
            .as_str()
            .ok_or_else(|| format!("{what}: class is not a string"))?;
        *o = M23Class::from_name(s).ok_or_else(|| format!("{what}: unknown class {s}"))?;
    }
    Ok(out)
}

/// One row of `m23_three_point_nielsen.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct NielsenRow {
    /// `"rational"` or `"conjugate_pair"`.
    pub kind: String,
    /// The class vector, in the file's order.
    pub classes: [M23Class; 3],
    /// The genus of the degree 23 cover.
    pub genus23: i64,
    /// The structure constant as a string (`"43/2"`).
    pub structure_constant: String,
    /// The number of Nielsen classes of generating triples.
    pub size: u64,
    /// The number of classes fixed by complex conjugation.
    pub real: u64,
}

/// The 57 rows of the owner's `data/m23_three_point_nielsen.json`.
pub fn read_nielsen_rows(path: &Path) -> Result<Vec<NielsenRow>, String> {
    let j = read_json(path)?;
    let what = path.display().to_string();
    let rows = field(&j, "passports", &what)?
        .as_array()
        .ok_or_else(|| format!("{what}: passports is not a list"))?;
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let w = format!("{what}, row {i}");
        out.push(NielsenRow {
            kind: field(r, "type", &w)?
                .as_str()
                .ok_or_else(|| format!("{w}: type"))?
                .to_string(),
            classes: classes_of(r, &w)?,
            genus23: field(r, "genus23", &w)?
                .as_i64()
                .ok_or_else(|| format!("{w}: genus23"))?,
            structure_constant: field(r, "structure_constant", &w)?
                .as_str()
                .ok_or_else(|| format!("{w}: structure_constant"))?
                .to_string(),
            size: as_u64(r, "size", &w)?,
            real: as_u64(r, "real", &w)?,
        });
    }
    if out.is_empty() {
        return Err(format!("{what}: no rows"));
    }
    Ok(out)
}

/// The rows `(type, classes, structure constant)` of the owner's
/// `data/m23_three_point_passports.json` (computed there by GAP from the same table).
pub fn read_passport_rows(path: &Path) -> Result<Vec<(String, [M23Class; 3], String)>, String> {
    let j = read_json(path)?;
    let what = path.display().to_string();
    let rows = field(&j, "passports", &what)?
        .as_array()
        .ok_or_else(|| format!("{what}: passports is not a list"))?;
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let w = format!("{what}, row {i}");
        out.push((
            field(r, "type", &w)?
                .as_str()
                .ok_or_else(|| format!("{w}: type"))?
                .to_string(),
            classes_of(r, &w)?,
            field(r, "structure_constant", &w)?
                .as_str()
                .ok_or_else(|| format!("{w}: structure_constant"))?
                .to_string(),
        ));
    }
    Ok(out)
}

/// The seven rows of `group_facts.json`, key `nielsen` (indices, dimensions, dictionaries).
pub fn read_group_facts_nielsen(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let j = read_json(path)?;
    let rows = field(&j, "nielsen", &path.display().to_string())?
        .as_array()
        .ok_or_else(|| format!("{}: nielsen is not a list", path.display()))?
        .clone();
    if rows.len() != 7 {
        return Err(format!("{}: {} nielsen rows", path.display(), rows.len()));
    }
    Ok(rows)
}

/// The owner's cycle strings `"(6 14)(7 20)..."` (1-based, space separated).
pub fn owner_cycles(s: &str) -> Result<Perm23, String> {
    let t = s.trim();
    if t.is_empty() || t == "()" {
        return Ok(rustmath_groups::m23::group::identity23());
    }
    if !t.starts_with('(') || !t.ends_with(')') {
        return Err(format!("not a cycle string: {s}"));
    }
    let mut cycles: Vec<Vec<u8>> = Vec::new();
    for c in t[1..t.len() - 1].split(")(") {
        let mut cyc = Vec::new();
        for tok in c.split_whitespace() {
            cyc.push(tok.parse::<u8>().map_err(|e| format!("{s}: {tok}: {e}"))?);
        }
        cycles.push(cyc);
    }
    from_cycles_1based(&cycles).map_err(|e| format!("{s}: {e}"))
}

/// The vendored `janik_table.json`: the seven rows `(g2, g3)` of Janik's Table 1, each with
/// first entry the published `g1` (1-based cycles in the file).
pub fn read_janik_table(path: &Path) -> Result<Vec<(Perm23, Perm23)>, String> {
    let j = read_json(path)?;
    let what = path.display().to_string();
    let rows = field(&j, "janik_table_g2_g3", &what)?
        .as_array()
        .ok_or_else(|| format!("{what}: janik_table_g2_g3 is not a list"))?;
    let to_perm = |v: &serde_json::Value| -> Result<Perm23, String> {
        let cyc: Vec<Vec<u8>> = v
            .as_array()
            .ok_or_else(|| format!("{what}: a permutation is not a list"))?
            .iter()
            .map(|c| {
                c.as_array()
                    .ok_or_else(|| format!("{what}: a cycle is not a list"))?
                    .iter()
                    .map(|x| {
                        x.as_u64()
                            .filter(|&x| x <= 255)
                            .map(|x| x as u8)
                            .ok_or_else(|| format!("{what}: a point is not a small integer"))
                    })
                    .collect::<Result<Vec<u8>, String>>()
            })
            .collect::<Result<_, String>>()?;
        from_cycles_1based(&cyc).map_err(|e| format!("{what}: {e}"))
    };
    let mut out = Vec::new();
    for r in rows {
        let pair = r
            .as_array()
            .ok_or_else(|| format!("{what}: a row is not a list"))?;
        if pair.len() != 2 {
            return Err(format!("{what}: a row has {} entries", pair.len()));
        }
        out.push((to_perm(&pair[0])?, to_perm(&pair[1])?));
    }
    if out.len() != 7 {
        return Err(format!("{what}: {} rows instead of 7", out.len()));
    }
    Ok(out)
}

/// The permutations `S!(...)` of a Magma file, in order (1-based cycles with commas): the
/// format of HJLPPZ's `seven_nielsen_class_representatives.m`.
pub fn parse_magma_perms(text: &str) -> Result<Vec<Perm23>, String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(k) = rest.find("S!(") {
        rest = &rest[k + 2..];
        let b = rest.as_bytes();
        let mut i = 0;
        let mut cycles: Vec<Vec<u8>> = Vec::new();
        while i < b.len() && b[i] == b'(' {
            let j = i + rest[i..].find(')').ok_or("unterminated cycle")?;
            let mut cyc = Vec::new();
            for t in rest[i + 1..j].split(',') {
                let t = t.trim();
                if !t.is_empty() {
                    cyc.push(t.parse::<u8>().map_err(|e| format!("{t}: {e}"))?);
                }
            }
            cycles.push(cyc);
            i = j + 1;
        }
        out.push(from_cycles_1based(&cycles).map_err(|e| e.to_string())?);
        rest = &rest[i..];
    }
    Ok(out)
}

/// GAP source: the text between `name :=` at the start of a line and the next `;;`.
pub fn gap_assignment<'a>(text: &'a str, name: &str) -> Result<&'a str, String> {
    let key = format!("{name} :=");
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.trim_start().starts_with(&key) {
            let start = offset + line.find(&key).expect("found") + key.len();
            let end = text[start..]
                .find(";;")
                .ok_or_else(|| format!("no ;; after {name}"))?;
            return Ok(&text[start..start + end]);
        }
        offset += line.len();
    }
    Err(format!("no assignment to {name}"))
}

/// GAP permutations in cycle notation: one permutation, or a list `[p, q, ...]` (commas outside
/// cycles separate the entries). Whitespace and line breaks are ignored. `Err` on nested or
/// unbalanced cycles, foreign characters, points outside 1..23 or repeated points.
pub fn gap_perms(s: &str) -> Result<Vec<Perm23>, String> {
    let mut out = Vec::new();
    let mut cycles: Vec<Vec<u8>> = Vec::new();
    let mut cur: Option<Vec<u8>> = None;
    let mut num = String::new();
    fn flush(cur: &mut Option<Vec<u8>>, num: &mut String) -> Result<(), String> {
        if !num.is_empty() {
            let v: u8 = num.parse().map_err(|e| format!("{num}: {e}"))?;
            cur.as_mut().ok_or("number outside a cycle")?.push(v);
            num.clear();
        }
        Ok(())
    }
    for ch in s.chars() {
        match ch {
            '(' => {
                if cur.is_some() {
                    return Err("nested (".into());
                }
                cur = Some(Vec::new());
            }
            ')' => {
                flush(&mut cur, &mut num)?;
                let c = cur.take().ok_or("unbalanced )")?;
                if c.len() > 1 {
                    cycles.push(c);
                }
            }
            ',' => {
                if cur.is_some() {
                    flush(&mut cur, &mut num)?;
                } else {
                    out.push(from_cycles_1based(&cycles).map_err(|e| e.to_string())?);
                    cycles.clear();
                }
            }
            '0'..='9' => {
                if cur.is_none() {
                    return Err("digit outside a cycle".into());
                }
                num.push(ch);
            }
            '[' | ']' => {}
            c if c.is_whitespace() => {
                if cur.is_some() {
                    flush(&mut cur, &mut num)?;
                }
            }
            other => return Err(format!("unexpected character {other:?}")),
        }
    }
    if cur.is_some() {
        return Err("unterminated cycle".into());
    }
    out.push(from_cycles_1based(&cycles).map_err(|e| e.to_string())?);
    Ok(out)
}

/// The peak resident set size of this process (`VmHWM` of `/proc/self/status`), in MB.
pub fn vm_hwm_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("VmHWM:")
                    .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            })
        })
        .map_or(0, |kb| kb / 1024)
}
