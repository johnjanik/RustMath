//! The exact t-design test `t_design_index` / `is_steiner_system` (MAGMA ch. 147 `IsDesign`,
//! `IsSteiner`).
//!
//! References: the Fano plane is the 2-(7,3,1) design; adding an eighth point to its lines and
//! taking the complements gives the Steiner quadruple system S(3,4,8) (the extended Hamming code
//! words of weight 4); the six faces of the cube are a 1-(8,4,3) design, as in the handbook
//! example H147E9 (`IsDesign(I,1)` returns `true 3`). The S(4,7,23) of the binary Golay code is
//! checked in `rustmath-coding/tests/cyclic_golay.rs`.

use rustmath_combinatorics::{
    is_steiner_system, t_design_index, DesignCheckError, DESIGN_CHECK_LIMIT,
};

fn fano() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2],
        vec![0, 3, 4],
        vec![0, 5, 6],
        vec![1, 3, 5],
        vec![1, 4, 6],
        vec![2, 3, 6],
        vec![2, 4, 5],
    ]
}

#[test]
fn fano_plane_is_a_2_design() {
    assert_eq!(t_design_index(7, 2, &fano()), Ok(Some(1)));
    assert_eq!(t_design_index(7, 1, &fano()), Ok(Some(3)));
    assert_eq!(t_design_index(7, 0, &fano()), Ok(Some(7)));
    assert_eq!(is_steiner_system(7, 2, &fano()), Ok(true));
    // Two lines meet in one point, so a 3-subset lies in at most one line: not a 3-design.
    assert_eq!(t_design_index(7, 3, &fano()), Ok(None));
    // Without a line some pairs are covered and some are not.
    let mut minus = fano();
    minus.pop();
    assert_eq!(t_design_index(7, 2, &minus), Ok(None));
    assert_eq!(is_steiner_system(7, 2, &minus), Ok(false));
    // A repeated line doubles the count of its pairs only.
    let mut twice = fano();
    twice.push(fano()[0].clone());
    assert_eq!(t_design_index(7, 2, &twice), Ok(None));
}

#[test]
fn steiner_quadruple_system_of_order_8() {
    let mut blocks: Vec<Vec<usize>> = fano()
        .into_iter()
        .map(|mut l| {
            l.push(7);
            l
        })
        .collect();
    for l in fano() {
        blocks.push((0..7).filter(|x| !l.contains(x)).collect());
    }
    assert_eq!(blocks.len(), 14);
    assert_eq!(t_design_index(8, 3, &blocks), Ok(Some(1)));
    assert_eq!(is_steiner_system(8, 3, &blocks), Ok(true));
    // A 3-(8,4,1) design is a 2-(8,4,3) design and a 1-(8,4,7) design.
    assert_eq!(t_design_index(8, 2, &blocks), Ok(Some(3)));
    assert_eq!(t_design_index(8, 1, &blocks), Ok(Some(7)));
    assert_eq!(is_steiner_system(8, 2, &blocks), Ok(false));
}

#[test]
fn cube_faces_are_a_1_design() {
    // Vertices 0..7 as bit vectors (x, y, z); a face fixes one coordinate.
    let mut faces = Vec::new();
    for bit in 0..3 {
        for val in 0..2 {
            faces.push(
                (0..8usize)
                    .filter(|v| (v >> bit) & 1 == val)
                    .collect::<Vec<_>>(),
            );
        }
    }
    assert_eq!(t_design_index(8, 1, &faces), Ok(Some(3)));
    assert_eq!(t_design_index(8, 2, &faces), Ok(None));
    assert_eq!(is_steiner_system(8, 1, &faces), Ok(false));
}

/// All `k`-subsets of `v` points (the complete design), in colex order.
fn k_subsets(v: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for last in k - 1..v {
        for mut s in k_subsets(last, k - 1) {
            s.push(last);
            out.push(s);
        }
    }
    out
}

/// The counters are indexed by the colex rank of a `t`-subset (review 1 of SL, m2). One block of
/// all `v` points meets every `t`-subset once, so a wrong rank (a collision or a gap) gives
/// `None`. The complete design of all `k`-subsets is a `t`-design with
/// `lambda = C(v - t, k - t)` (T. Beth, D. Jungnickel, H. Lenz, *Design Theory*, I.1).
#[test]
fn colex_ranks_cover_every_subset_once() {
    for (v, t) in [
        (1, 1),
        (5, 0),
        (5, 2),
        (9, 4),
        (12, 6),
        (12, 11),
        (12, 12),
        (30, 3),
    ] {
        let all: Vec<usize> = (0..v).rev().collect(); // unsorted on purpose
        assert_eq!(
            t_design_index(v, t, std::slice::from_ref(&all)),
            Ok(Some(1)),
            "v = {v}, t = {t}"
        );
        assert_eq!(
            t_design_index(v, t, &[all.clone(), all]),
            Ok(Some(2)),
            "v = {v}, t = {t}"
        );
    }
    let blocks = k_subsets(9, 4);
    assert_eq!(blocks.len(), 126);
    assert_eq!(t_design_index(9, 2, &blocks), Ok(Some(21))); // C(7, 2)
    assert_eq!(t_design_index(9, 3, &blocks), Ok(Some(6))); // C(6, 1)
    assert_eq!(t_design_index(9, 4, &blocks), Ok(Some(1)));
    assert_eq!(t_design_index(9, 5, &blocks), Ok(Some(0)));
    // One block fewer: covered unequally (the dense path, not the early exit).
    assert_eq!(t_design_index(9, 2, &blocks[1..]), Ok(None));
}

#[test]
fn small_and_degenerate_cases() {
    // No block meets a 2-subset: every pair lies in 0 blocks.
    assert_eq!(t_design_index(5, 2, &[vec![0], vec![3]]), Ok(Some(0)));
    assert_eq!(t_design_index(5, 2, &[]), Ok(Some(0)));
    assert_eq!(is_steiner_system(5, 2, &[]), Ok(false));
    // All 2-subsets of 4 points, each once: the trivial S(2,2,4).
    let pairs: Vec<Vec<usize>> = (0..4)
        .flat_map(|a| (a + 1..4).map(move |b| vec![a, b]))
        .collect();
    assert_eq!(is_steiner_system(4, 2, &pairs), Ok(true));
    // Mixed block sizes are never a Steiner system.
    let mixed = vec![vec![0, 1, 2, 3], vec![0, 1]];
    assert_eq!(is_steiner_system(4, 2, &mixed), Ok(false));
}

#[test]
fn refusals() {
    assert_eq!(
        t_design_index(7, 2, &[vec![0, 1, 7]]),
        Err(DesignCheckError::PointOutOfRange { block: 0 })
    );
    assert_eq!(
        t_design_index(7, 2, &[vec![0, 1, 2], vec![3, 3, 4]]),
        Err(DesignCheckError::RepeatedPoint { block: 1 })
    );
    assert_eq!(
        t_design_index(3, 4, &[vec![0, 1, 2]]),
        Err(DesignCheckError::TExceedsPoints { t: 4, v: 3 })
    );
    // C(200, 5) = 2535650040 t-subsets of one block: refused before any enumeration.
    let block: Vec<usize> = (0..200).collect();
    assert_eq!(
        t_design_index(200, 5, &[block]),
        Err(DesignCheckError::TooLarge {
            subsets: 2_535_650_040,
            limit: DESIGN_CHECK_LIMIT
        })
    );
    assert!(DesignCheckError::RepeatedPoint { block: 1 }
        .to_string()
        .contains("block 1"));
}
