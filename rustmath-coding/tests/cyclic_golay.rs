//! The two cyclic binary Golay codes of length 23 and their Steiner systems (MAGMA ch. 152
//! `CyclicCode`, H152E5; ch. 147 `IsSteiner`).
//!
//! References: `m23golay/golay.py` `verify_basic_facts` of the owner's M23 repository (`d05475b`)
//! and Section 2 of its paper `golay_m23.tex` (weight distribution, `(z+1) g+ g- = z^23 - 1`,
//! `g- = reverse(g+)`, the replication numbers 77, 21, 5, the intersection numbers 14168 and
//! 17710, `i -> -i` exchanges the systems). The 11 orbit representatives and the heptads through
//! `{0, 1, 2}` were computed by golay.py and checked by an independent script
//! (`golay_check.py`, review 1 of design SL).

use rustmath_coding::cyclic_golay::{
    codewords, heptad_orbit_representatives, heptads, mask_of, negate, rotate, scale, support,
    weight_distribution, GolaySign, LENGTH,
};
use rustmath_coding::BinaryGolayCode;
use rustmath_combinatorics::{is_steiner_system, t_design_index};
use std::collections::{BTreeMap, HashSet};

/// Product in F_2[z] (masks).
fn f2_mul(a: u64, b: u64) -> u64 {
    let mut out = 0u64;
    for i in 0..64 {
        if (b >> i) & 1 == 1 {
            out ^= a << i;
        }
    }
    out
}

#[test]
fn generator_polynomials() {
    let gp = GolaySign::Plus.generator_polynomial() as u64;
    let gm = GolaySign::Minus.generator_polynomial() as u64;
    assert_eq!((gp, gm), (0xAE3, 0xC75));
    // (z + 1) g+ g- = z^23 + 1 over F_2.
    assert_eq!(f2_mul(f2_mul(gp, gm), 0b11), (1 << 23) | 1);
    // g- is the reciprocal of g+ (degree 11).
    let rev = (0..12).fold(0u64, |acc, i| acc | ((gp >> i) & 1) << (11 - i));
    assert_eq!(rev, gm);
    assert_eq!(GolaySign::Plus.opposite(), GolaySign::Minus);
    assert_eq!(GolaySign::Minus.to_i8(), -1);
    assert_eq!(GolaySign::from_i8(1), Some(GolaySign::Plus));
    assert_eq!(GolaySign::from_i8(0), None);
}

#[test]
fn weight_distributions() {
    let mut want = [0u64; 24];
    for (w, c) in [
        (0, 1),
        (7, 253),
        (8, 506),
        (11, 1288),
        (12, 1288),
        (15, 506),
        (16, 253),
        (23, 1),
    ] {
        want[w] = c;
    }
    for s in GolaySign::ALL {
        assert_eq!(weight_distribution(s), want, "sign {s}");
        let words = codewords(s);
        assert_eq!(words.iter().collect::<HashSet<_>>().len(), 4096);
        assert!(words.iter().all(|w| w >> LENGTH == 0));
    }
}

fn blocks(s: GolaySign) -> Vec<Vec<usize>> {
    heptads(s).iter().map(|&h| support(h)).collect()
}

#[test]
fn both_systems_are_s_4_7_23() {
    for s in GolaySign::ALL {
        let b = blocks(s);
        assert_eq!(b.len(), 253);
        assert_eq!(is_steiner_system(23, 4, &b), Ok(true), "sign {s}");
        // A 4-(23,7,1) design is a 3-(23,7,5), a 2-(23,7,21) and a 1-(23,7,77) design.
        assert_eq!(t_design_index(23, 3, &b), Ok(Some(5)));
        assert_eq!(t_design_index(23, 2, &b), Ok(Some(21)));
        assert_eq!(t_design_index(23, 1, &b), Ok(Some(77)));
        assert_eq!(t_design_index(23, 5, &b), Ok(None));
    }
    assert_ne!(heptads(GolaySign::Plus), heptads(GolaySign::Minus));
    let plus: HashSet<u32> = heptads(GolaySign::Plus).iter().copied().collect();
    assert!(
        heptads(GolaySign::Minus).iter().all(|h| !plus.contains(h)),
        "no common heptad"
    );
}

#[test]
fn symmetries() {
    for s in GolaySign::ALL {
        let hs = heptads(s);
        let set: HashSet<u32> = hs.iter().copied().collect();
        // Shift and i -> 2i (2 is a square mod 23) preserve each system.
        assert!(hs.iter().all(|&h| set.contains(&rotate(h, 1))));
        assert!(hs.iter().all(|&h| set.contains(&scale(h, 2))));
        // i -> -i exchanges them.
        let other: HashSet<u32> = heptads(s.opposite()).iter().copied().collect();
        assert!(hs.iter().all(|&h| other.contains(&negate(h))));
        assert!(hs.iter().all(|&h| !set.contains(&scale(h, 5)))); // 5 is a non-square
    }
    assert_eq!(rotate(1, 23), 1);
    assert_eq!(rotate(1 << 22, 1), 1);
    assert_eq!(negate(0b10), 1 << 22);
}

#[test]
fn intersections_and_orbits() {
    for s in GolaySign::ALL {
        let hs = heptads(s);
        let mut inter: BTreeMap<u32, u64> = BTreeMap::new();
        for (i, a) in hs.iter().enumerate() {
            for b in &hs[i + 1..] {
                *inter.entry((a & b).count_ones()).or_insert(0) += 1;
            }
        }
        assert_eq!(inter, BTreeMap::from([(1, 14168), (3, 17710)]), "sign {s}");
        let reps = heptad_orbit_representatives(s);
        assert_eq!(reps.len(), 11);
        // Each orbit has 23 distinct heptads and the orbits cover the system.
        let mut covered = HashSet::new();
        for &r in reps {
            let orbit: HashSet<u32> = (0..23).map(|k| rotate(r, k)).collect();
            assert_eq!(orbit.len(), 23);
            covered.extend(orbit);
        }
        assert_eq!(covered.len(), 253);
    }
    assert_eq!(
        heptad_orbit_representatives(GolaySign::Plus),
        &[
            0xae3, 0x34a9, 0x483d, 0xb115, 0xcd81, 0x14345, 0x18427, 0x21c13, 0x22259, 0x32921,
            0x51231
        ]
    );
    assert_eq!(
        heptad_orbit_representatives(GolaySign::Minus),
        &[
            0xc75, 0x254b, 0x5e09, 0x81b3, 0xa88d, 0x14585, 0x1c843, 0x21253, 0x26911, 0x320e1,
            0x46245
        ]
    );
}

#[test]
fn heptads_through_0_1_2() {
    let through = |s: GolaySign| -> Vec<Vec<usize>> {
        let t = mask_of(&[0, 1, 2]);
        let mut v: Vec<Vec<usize>> = heptads(s)
            .iter()
            .filter(|&&h| h & t == t)
            .map(|&h| support(h))
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        through(GolaySign::Plus),
        vec![
            vec![0, 1, 2, 3, 9, 12, 21],
            vec![0, 1, 2, 4, 6, 18, 19],
            vec![0, 1, 2, 5, 10, 15, 16],
            vec![0, 1, 2, 7, 13, 14, 17],
            vec![0, 1, 2, 8, 11, 20, 22],
        ]
    );
    assert_eq!(
        through(GolaySign::Minus),
        vec![
            vec![0, 1, 2, 3, 5, 14, 17],
            vec![0, 1, 2, 4, 13, 16, 22],
            vec![0, 1, 2, 6, 7, 19, 21],
            vec![0, 1, 2, 8, 11, 12, 18],
            vec![0, 1, 2, 9, 10, 15, 20],
        ]
    );
}

#[test]
fn binary_golay_code_is_g_minus() {
    let words: HashSet<u32> = codewords(GolaySign::Minus).into_iter().collect();
    let code = BinaryGolayCode::new();
    let rows = code.generator_matrix();
    assert_eq!(rows.len(), 12);
    for row in rows {
        let m = row
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &b)| acc | ((b as u32 & 1) << i));
        assert!(words.contains(&m), "row {m:#x} is not in G-");
    }
}
