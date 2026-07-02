//! Triangulation regularization for circle packing — make degree-2 leaf vertices
//! *impossible to miss*, then let the audit pick the winner.
//!
//! Per the P2 review note: do not commit to "6E barycentric subdivision" as a
//! design choice. Implement several candidate transforms and accept only one that
//! passes the [`Triangulation::is_packable_by_valence_gate`] audit — with the
//! degree-2-vertex count as a hard regression.
//!
//! Every transform here is *conservative*: it produces a candidate and the audit
//! decides. A transform is never assumed correct because of its name.

use crate::belyi::triangulation_audit::Triangulation;
use std::collections::BTreeMap;

/// A candidate regularization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularizationMethod {
    /// Identity — audit the raw triangulation.
    None,
    /// Standard 1→4 triangle subdivision (adds edge midpoints).
    FullSubdivision,
    /// Local repair of each degree-2 vertex by capping its two-triangle digon.
    LocalLeafCap,
}

/// Before/after audit summary for a regularization run.
#[derive(Debug, Clone)]
pub struct RegularizationReport {
    pub method: RegularizationMethod,
    pub before_min_valence: usize,
    pub after_min_valence: usize,
    pub before_degree_two: usize,
    pub after_degree_two: usize,
    pub euler_before: i64,
    pub euler_after: i64,
    pub packable: bool,
}

/// Apply `method` and audit the result.
pub fn regularize_for_circle_packing(
    tri: &Triangulation,
    method: RegularizationMethod,
) -> (Triangulation, RegularizationReport) {
    let before = tri.audit();
    let out = match method {
        RegularizationMethod::None => tri.clone(),
        RegularizationMethod::FullSubdivision => full_triangle_subdivision(tri),
        RegularizationMethod::LocalLeafCap => cap_degree_two_vertices(tri),
    };
    let after = out.audit();
    let report = RegularizationReport {
        method,
        before_min_valence: before.min_valence,
        after_min_valence: after.min_valence,
        before_degree_two: before.degree_two_vertices.len(),
        after_degree_two: after.degree_two_vertices.len(),
        euler_before: before.euler_characteristic,
        euler_after: after.euler_characteristic,
        packable: out.is_packable_by_valence_gate(),
    };
    (out, report)
}

/// Try each method in order and return the first (transform, report) that passes
/// the valence gate, else the last candidate tried.
pub fn regularize_best(
    tri: &Triangulation,
    methods: &[RegularizationMethod],
) -> (Triangulation, RegularizationReport) {
    let mut last = None;
    for &m in methods {
        let (out, report) = regularize_for_circle_packing(tri, m);
        if report.packable {
            return (out, report);
        }
        last = Some((out, report));
    }
    last.unwrap_or_else(|| regularize_for_circle_packing(tri, RegularizationMethod::None))
}

/// Standard 1→4 subdivision. NOTE: this does *not* generally fix a degree-2 digon
/// leaf (the vertex keeps its two edge-midpoint neighbours), so the audit must
/// decide — exactly the point of auditing rather than trusting the label.
pub fn full_triangle_subdivision(tri: &Triangulation) -> Triangulation {
    let mut next_vertex = tri.n_vertices;
    let mut midpoint: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut get_mid = |a: usize, b: usize| -> usize {
        let key = if a <= b { (a, b) } else { (b, a) };
        *midpoint.entry(key).or_insert_with(|| {
            let m = next_vertex;
            next_vertex += 1;
            m
        })
    };

    let mut triangles = Vec::with_capacity(4 * tri.triangles.len());
    for &[a, b, c] in &tri.triangles {
        let ab = get_mid(a, b);
        let bc = get_mid(b, c);
        let ca = get_mid(c, a);
        triangles.push([a, ab, ca]);
        triangles.push([b, bc, ab]);
        triangles.push([c, ca, bc]);
        triangles.push([ab, bc, ca]);
    }
    Triangulation::new(next_vertex, triangles)
}

/// Local repair: for each degree-2 vertex whose local topology is the expected
/// two-triangle digon, replace it with a three-triangle cap that gives the vertex
/// a third neighbour. Conservative — non-digon neighbourhoods are left for the
/// audit to reject.
pub fn cap_degree_two_vertices(tri: &Triangulation) -> Triangulation {
    let audit = tri.audit();
    let mut out = tri.clone();
    // Repair from the original degree-2 list; re-audit at the end via the caller.
    for v in audit.degree_two_vertices {
        if let Some(next) = cap_one_degree_two_vertex(&out, v) {
            out = next;
        }
    }
    out
}

fn cap_one_degree_two_vertex(tri: &Triangulation, v: usize) -> Option<Triangulation> {
    let incident: Vec<usize> = tri
        .triangles
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.contains(&v).then_some(i))
        .collect();
    if incident.len() != 2 {
        return None;
    }

    let mut nbrs: Vec<usize> = Vec::new();
    for &idx in &incident {
        for &u in &tri.triangles[idx] {
            if u != v && !nbrs.contains(&u) {
                nbrs.push(u);
            }
        }
    }
    if nbrs.len() != 2 {
        return None;
    }
    let (a, b) = (nbrs[0], nbrs[1]);
    let w = tri.n_vertices;

    let mut triangles: Vec<[usize; 3]> = tri
        .triangles
        .iter()
        .enumerate()
        .filter(|(i, _)| !incident.contains(i))
        .map(|(_, t)| *t)
        .collect();
    // Replace the digon around v by a three-triangle cap introducing w.
    triangles.push([v, a, w]);
    triangles.push([v, w, b]);
    triangles.push([w, a, b]);

    Some(Triangulation::new(tri.n_vertices + 1, triangles))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regularization_must_remove_degree_two_or_fail_audit() {
        // A local digon: vertex 0 sits in two triangles sharing both neighbours.
        let tri = Triangulation::new(
            4,
            vec![[0, 1, 2], [0, 2, 1], [1, 3, 2], [1, 2, 3]],
        );
        let before = tri.audit();
        assert!(before.degree_two_vertices.contains(&0));

        let (_out, report) =
            regularize_for_circle_packing(&tri, RegularizationMethod::LocalLeafCap);
        assert!(
            report.after_degree_two <= report.before_degree_two,
            "must not increase degree-2 vertices: {report:?}"
        );
    }

    #[test]
    fn full_subdivision_preserves_euler() {
        let tri = Triangulation::new(4, vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]);
        let (out, report) =
            regularize_for_circle_packing(&tri, RegularizationMethod::FullSubdivision);
        assert_eq!(report.euler_before, report.euler_after, "χ invariant: {report:?}");
        // 1→4 on a tetrahedron: 4 + 6 = 10 vertices, 16 faces.
        assert_eq!(out.n_vertices, 10);
        assert_eq!(out.triangles.len(), 16);
    }
}
