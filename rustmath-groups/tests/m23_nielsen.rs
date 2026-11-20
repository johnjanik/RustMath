//! The Nielsen classes `(C, 23A, 23B)` of M23, complex conjugation, real structures, ovals and
//! parabolic cohomology over `F_2` (WP-P, HANDOFF 4.2 step 7 and 4.3), and the three
//! dictionaries of the acceptance of 4.2. MAGMA ch. 58 (permutation groups) and ch. 68
//! (cohomology).
//!
//! References (vendored in `tests/data/m23/`, README there): the owner's `group_facts.json`
//! (key `nielsen`) and `passports_C_23A_23B.json`; `janik_table.json`, the owner's
//! transcription of Table 1 of his manuscript [Janik26]. Read at run time only, never
//! vendored: HJLPPZ's `seven_nielsen_class_representatives.m` from `$M23ISGALOIS` (no licence)
//! and Le Lay's `verification/certify_geometric_reflection_obstruction.g` from `$LELAY_M23`
//! (CC BY-NC-SA 4.0).
//!
//! ```text
//! cargo test -p rustmath-groups --test m23_nielsen
//! cargo test --release -p rustmath-groups --test m23_nielsen -- --ignored --exact full_c_23a_23b
//! M23ISGALOIS=<checkout of shaowuz/m23isgalois> \
//! LELAY_M23=<checkout of research-reflexivity/frontier-math-m23-cover-investigation> \
//!   cargo test -p rustmath-groups --test m23_nielsen -- --ignored env_
//! ```

mod m23_common;

use std::time::Instant;

use m23_common::{
    data_dir, gap_assignment, gap_perms, owner_cycles, parse_magma_perms, read_group_facts_nielsen,
    read_janik_table, vendored,
};
use rustmath_finitefields::PrimeField;
use rustmath_groups::m23::classes::ORBIT_CAP;
use rustmath_groups::m23::group::{from_cycles_1based, identity23, inverse, rmul};
use rustmath_groups::m23::nielsen::complex_conjugate;
use rustmath_groups::m23::{
    standard_modules, AtlasTable, ClassIdentifier, ClassLists, F2Error, F2Module, F2Span, M23Class,
    NielsenClass23, NielsenError, Triple, TripleCohomology, M23,
};
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;

fn setup() -> (M23, ClassIdentifier) {
    let g = M23::published().unwrap();
    let id = ClassIdentifier::new(&g).unwrap();
    (g, id)
}

fn lists_2a(g: &M23, id: &ClassIdentifier) -> ClassLists {
    ClassLists::by_orbit(g, id, &[M23Class::C2A], ORBIT_CAP).unwrap()
}

/// One entry of the `real` list of passports_C_23A_23B.json, computed.
fn real_entry(
    ni: &NielsenClass23,
    mods: &[F2Module; 3],
    i: usize,
    genus: usize,
) -> serde_json::Value {
    let t = ni.triple(i);
    let c0 = ni
        .real_structure(i)
        .unwrap()
        .expect("a fixed class has a real structure");
    let comps = ni.real_components(i, &c0).unwrap();
    let j2r = TripleCohomology::new(&mods[2], &t)
        .unwrap()
        .conjugation_fixed_dim(&c0)
        .unwrap();
    let wr = TripleCohomology::new(&mods[0], &t)
        .unwrap()
        .conjugation_fixed_dim(&c0)
        .unwrap();
    assert_eq!(
        j2r,
        genus + comps - 1,
        "dim Jac(X)(R)[2] = g + r - 1 at class {i}"
    );
    serde_json::json!({
        "index": i,
        "components": comps,
        "dim_J2_real": j2r,
        "dim_W_real": wr,
        "real_points_above_0": ni.real_points_above_0(i, &c0),
    })
}

// group_facts.json, key nielsen: the seven classes of (2A, 23A, 23B), complex conjugation
// [1, 0, 2, 3, 6, 5, 4], dim W = dim W^dual = 4, dim Jac[2] = 8, dim H^1_orb(G0) = 7, the real
// data of the three fixed classes; Janik's seven rows map to janik_orbit [3, 5, 1, 6, 2, 4, 0];
// passports_C_23A_23B.json "2A": representatives_g1 and the real entries.
#[test]
fn nielsen_2a_23a_23b_and_janik() {
    let t0 = Instant::now();
    let (g, id) = setup();
    let lists = lists_2a(&g, &id);
    let ni = NielsenClass23::new(&g, &id, &lists, M23Class::C2A).unwrap();
    assert_eq!(ni.len(), 7);
    assert_eq!(ni.num_candidates(), 161);
    let rows = read_group_facts_nielsen(&data_dir().join("group_facts.json")).unwrap();
    let conj = ni.complex_conjugation().unwrap();
    assert_eq!(conj, vec![1, 0, 2, 3, 6, 5, 4]);
    let mods = standard_modules(&g).unwrap();
    assert_eq!([mods[0].dim(), mods[1].dim(), mods[2].dim()], [11, 11, 22]);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["index"].as_u64(), Some(i as u64));
        assert_eq!(row["conjugate"].as_u64(), Some(conj[i] as u64), "class {i}");
        assert_eq!(row["real"].as_bool(), Some(conj[i] == i), "class {i}");
        let t = ni.triple(i);
        let h = [
            TripleCohomology::new(&mods[0], &t).unwrap(),
            TripleCohomology::new(&mods[1], &t).unwrap(),
            TripleCohomology::new(&mods[2], &t).unwrap(),
        ];
        assert_eq!(h[0].dim_parabolic() as u64, row["dim_W"].as_u64().unwrap());
        assert_eq!(
            h[1].dim_parabolic() as u64,
            row["dim_Wdual"].as_u64().unwrap()
        );
        assert_eq!(h[2].dim_parabolic() as u64, row["dim_J2"].as_u64().unwrap());
        assert_eq!(
            h[0].dim_orbifold() as u64,
            row["dim_H1_orbifold_G0"].as_u64().unwrap()
        );
        match ni.real_structure(i).unwrap() {
            Some(c0) => {
                assert_eq!(
                    ni.real_components(i, &c0).unwrap() as u64,
                    row["real_components"].as_u64().unwrap()
                );
                assert_eq!(
                    h[2].conjugation_fixed_dim(&c0).unwrap() as u64,
                    row["dim_J2_real"].as_u64().unwrap()
                );
                assert_eq!(
                    h[0].conjugation_fixed_dim(&c0).unwrap() as u64,
                    row["dim_W_real"].as_u64().unwrap()
                );
                assert_eq!(
                    ni.real_points_above_0(i, &c0) as u64,
                    row["fix_c0_cap_fix_g1"].as_u64().unwrap()
                );
            }
            None => assert_eq!(row["real"].as_bool(), Some(false)),
        }
    }
    // Janik's Table 1: the direct reading g1 g2 g3 = 1 with g1 the published g1
    let janik = read_janik_table(&data_dir().join("janik_table.json")).unwrap();
    let mut jk = Vec::new();
    for (r, (g2, g3)) in janik.iter().enumerate() {
        assert_eq!(rmul(&rmul(&g.g1(), g2), g3), identity23(), "Janik row {r}");
        let idx = ni.index_of(&(g.g1(), *g2, *g3)).unwrap();
        assert_eq!(rows[idx]["janik_orbit"].as_u64(), Some(r as u64));
        jk.push(idx);
    }
    assert_eq!(jk, vec![3, 5, 1, 6, 2, 4, 0]);
    // passports_C_23A_23B.json, "2A"
    let p = vendored("passports_C_23A_23B.json");
    let reps: Vec<_> = p["2A"]["representatives_g1"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| owner_cycles(v.as_str().unwrap()).unwrap())
        .collect();
    assert_eq!(reps, ni.representatives());
    let real: Vec<serde_json::Value> = (0..7)
        .filter(|&i| conj[i] == i)
        .map(|i| real_entry(&ni, &mods, i, 4))
        .collect();
    assert_eq!(serde_json::Value::Array(real), p["2A"]["real"]);
    println!("seven classes, cohomology and Janik: {:?}", t0.elapsed());
}

/// The dimension of the parabolic cohomology by `Matrix<PrimeField>` ranks over GF(2), an
/// independent route: `Z_par` is the kernel of `(c1, c2) -> (c1 K1, c2 K2, c3 K3)` with
/// `c3 = c1 M2 M3 + c2 M3` and `K_i` a basis of the right kernel of `A_i = 1 + M_i` (so
/// `c` is in the row space of `A_i` iff `c K_i = 0`); `B` is the row space of `[A1 | A2]`.
fn dim_parabolic_gf2(m: &F2Module, t: &Triple) -> usize {
    let d = m.dim();
    let two = Integer::from(2);
    let f = |b: bool| PrimeField::new(Integer::from(b as i64), two.clone()).unwrap();
    let to_matrix = |rows: &[u128]| -> Matrix<PrimeField> {
        let data: Vec<PrimeField> = rows
            .iter()
            .flat_map(|&r| (0..d).map(move |j| (r >> j) & 1 == 1))
            .map(f)
            .collect();
        Matrix::from_vec(d, d, data).unwrap()
    };
    let mats: Vec<Matrix<PrimeField>> = [t.0, t.1, t.2]
        .iter()
        .map(|p| to_matrix(&m.matrix(p).unwrap()))
        .collect();
    let id: Vec<u128> = (0..d).map(|i| 1u128 << i).collect();
    let a: Vec<Matrix<PrimeField>> = [t.0, t.1, t.2]
        .iter()
        .map(|p| {
            let rows: Vec<u128> = m
                .matrix(p)
                .unwrap()
                .iter()
                .zip(&id)
                .map(|(r, e)| r ^ e)
                .collect();
            to_matrix(&rows)
        })
        .collect();
    let kernels: Vec<Matrix<PrimeField>> = a
        .iter()
        .map(|ai| {
            let k = ai.kernel().unwrap();
            let cols = k.len();
            let mut data = Vec::with_capacity(d * cols);
            for i in 0..d {
                for col in &k {
                    data.push(col[i].clone());
                }
            }
            Matrix::from_vec(d, cols, data).unwrap()
        })
        .collect();
    let g23 = mats[1].mul(&mats[2]).unwrap();
    let blocks = [
        kernels[0].clone(),
        g23.mul(&kernels[2]).unwrap(),
        kernels[1].clone(),
        mats[2].mul(&kernels[2]).unwrap(),
    ];
    let (k1, k2, k3) = (kernels[0].cols(), kernels[1].cols(), kernels[2].cols());
    let width = k1 + k2 + k3;
    let zero = f(false);
    let mut data = vec![zero.clone(); 2 * d * width];
    for r in 0..d {
        for c in 0..k1 {
            data[r * width + c] = blocks[0].get(r, c).unwrap().clone();
        }
        for c in 0..k3 {
            data[r * width + k1 + k2 + c] = blocks[1].get(r, c).unwrap().clone();
        }
        for c in 0..k2 {
            data[(d + r) * width + k1 + c] = blocks[2].get(r, c).unwrap().clone();
        }
        for c in 0..k3 {
            data[(d + r) * width + k1 + k2 + c] = blocks[3].get(r, c).unwrap().clone();
        }
    }
    let l = Matrix::from_vec(2 * d, width, data).unwrap();
    let dim_z = 2 * d - l.rank().unwrap();
    let mut bdata = Vec::with_capacity(d * 2 * d);
    for r in 0..d {
        for c in 0..d {
            bdata.push(a[0].get(r, c).unwrap().clone());
        }
        for c in 0..d {
            bdata.push(a[1].get(r, c).unwrap().clone());
        }
    }
    let dim_b = Matrix::from_vec(d, 2 * d, bdata).unwrap().rank().unwrap();
    dim_z - dim_b
}

// The three dimensions of the rational class (index 5) of (2A, 23A, 23B) by Matrix<PrimeField>
// over GF(2): 4, 4, 8 (group_facts.json).
#[test]
fn cohomology_cross_check_matrix_gf2() {
    let (g, id) = setup();
    let lists = lists_2a(&g, &id);
    let ni = NielsenClass23::new(&g, &id, &lists, M23Class::C2A).unwrap();
    let mods = standard_modules(&g).unwrap();
    let t = ni.triple(5);
    let t0 = Instant::now();
    let dims: Vec<usize> = mods.iter().map(|m| dim_parabolic_gf2(m, &t)).collect();
    println!("Matrix<PrimeField> cross-check: {:?}", t0.elapsed());
    assert_eq!(dims, vec![4, 4, 8]);
    let fast: Vec<usize> = mods
        .iter()
        .map(|m| TripleCohomology::new(m, &t).unwrap().dim_parabolic())
        .collect();
    assert_eq!(dims, fast);
}

// The GAP reader of Le Lay's file: malformed input is an Err (design Section 7).
#[test]
fn gap_reader_rejects() {
    assert!(gap_perms("(1,2)(3,24)").is_err());
    assert!(gap_perms("(1,2)(2,3)").is_err());
    assert!(gap_perms("(1,(2,3))").is_err());
    assert!(gap_perms("(1,2").is_err());
    assert!(gap_perms("1,2)").is_err());
    assert!(gap_perms("(1,x)").is_err());
    assert!(gap_assignment("z := (1,2);;", "y").is_err());
    assert!(gap_assignment("y := (1,2)", "y").is_err());
    let p = gap_perms("[ (1,2)(3,4),\n (5,6,7) ]").unwrap();
    assert_eq!(p.len(), 2);
    assert_eq!(
        gap_perms("(1,2,\n  3)").unwrap()[0],
        from_cycles_1based(&[[1u8, 2, 3]]).unwrap()
    );
    assert_eq!(
        gap_assignment("x := 1;;\n  y := (1,2);;", "y").unwrap(),
        " (1,2)"
    );
    assert!(parse_magma_perms("S!(1,2)(3,99)").is_err());
    assert_eq!(parse_magma_perms("S!(1,2)(3,4); S!(5,6)").unwrap().len(), 2);
    assert!(owner_cycles("(1 2)(3 x)").is_err());
}

// The cheap Err cases of design Section 9 for the Nielsen and F_2 layers.
#[test]
fn degenerate_nielsen_cases() {
    let (g, id) = setup();
    let lists = lists_2a(&g, &id);
    let ni = NielsenClass23::new(&g, &id, &lists, M23Class::C2A).unwrap();
    for c in [M23Class::C1A, M23Class::C7A, M23Class::C23A] {
        assert_eq!(
            NielsenClass23::new(&g, &id, &lists, c).err(),
            Some(NielsenError::NotARationalClass(c))
        );
    }
    assert_eq!(
        NielsenClass23::new(&g, &id, &lists, M23Class::C4A).err(),
        Some(NielsenError::ClassListMissing(M23Class::C4A))
    );
    let (a, b, c) = ni.triple(0);
    assert_eq!(ni.index_of(&(a, b, b)), Err(NielsenError::ProductNotOne));
    // second entry in 23B
    let g3 = g.g3();
    let t = (g.g1(), g3, inverse(&rmul(&g.g1(), &g3)));
    assert_eq!(ni.index_of(&t), Err(NielsenError::SecondEntryNotIn23A));
    // a triple of (3A, 23A, 23B) is not in the 2A class
    let lists3 = ClassLists::by_orbit(&g, &id, &[M23Class::C3A], ORBIT_CAP).unwrap();
    let n3 = NielsenClass23::new(&g, &id, &lists3, M23Class::C3A).unwrap();
    assert_eq!(n3.len(), 134);
    assert_eq!(ni.index_of(&n3.triple(0)), Err(NielsenError::NotInClass));
    assert_eq!(ni.index_of(&(a, b, c)), Ok(0));
    // a non-fixed class has no real structure; a wrong c0 is refused by the oval graph
    assert_eq!(ni.real_structure(0).unwrap(), None);
    assert!(matches!(
        ni.real_components(2, &g.g1()),
        Err(NielsenError::OvalGraph(_))
    ));
    // the F_2 layer
    let mut s = F2Span::new();
    for i in 0..128u32 {
        assert_eq!(s.add(1u128 << i), Ok(true));
    }
    assert_eq!(s.add(1), Err(F2Error::CapacityExceeded));
    assert_eq!(s.dim(), 128);
    assert_eq!(
        F2Module::subquotient(&[0b11], &[0b100]).err(),
        Some(F2Error::NotInModule)
    );
    let mods = standard_modules(&g).unwrap();
    let transposition = from_cycles_1based(&[[1u8, 2]]).unwrap();
    assert_eq!(
        mods[0].matrix(&transposition).err(),
        Some(F2Error::NotInModule)
    );
    // review 1: a transposition preserves P0 but not G0, so it does not act on Q = P0 / G0
    assert_eq!(mods[2].matrix(&transposition).map(|m| m.len()), Ok(22));
    assert_eq!(
        mods[1].matrix(&transposition).err(),
        Some(F2Error::NotInModule)
    );
    assert_eq!(
        TripleCohomology::new(&mods[1], &(transposition, transposition, identity23())).err(),
        Some(F2Error::NotInModule)
    );
    assert_eq!(
        TripleCohomology::new(&mods[0], &(a, b, b)).err(),
        Some(F2Error::ProductNotOne)
    );
    // review 1: c0 = 1 is not a real structure of class 5 (c0 g2 c0 = g2 != g3^-1); the
    // computed real structure is accepted
    let h5 = TripleCohomology::new(&mods[2], &ni.triple(5)).unwrap();
    assert_eq!(
        h5.conjugation_fixed_dim(&identity23()),
        Err(F2Error::NotARealStructure)
    );
    let c5 = ni.real_structure(5).unwrap().expect("class 5 is real");
    assert!(h5.conjugation_fixed_dim(&c5).is_ok());
    // complex conjugation of a triple is a triple
    let cc = complex_conjugate(&ni.triple(1));
    assert_eq!(rmul(&rmul(&cc.0, &cc.1), &cc.2), identity23());
}

/// The six classes (C, 23A, 23B) against passports_C_23A_23B.json entry by entry: sizes, the
/// representatives, real_classes and the real entries; dim W = dim W^dual = g and
/// dim Jac[2] = 2g on all 6,127 triples; g + r - 1 on every real cover; size = n.
/// Design: 1.3 s, 77 MB (release).
#[test]
#[ignore]
fn full_c_23a_23b() {
    let t0 = Instant::now();
    let (g, id) = setup();
    let rational = [
        M23Class::C2A,
        M23Class::C3A,
        M23Class::C4A,
        M23Class::C5A,
        M23Class::C6A,
        M23Class::C8A,
    ];
    let lists = ClassLists::by_enumeration(&g, &id, &rational).unwrap();
    let p = vendored("passports_C_23A_23B.json");
    let mods = standard_modules(&g).unwrap();
    let t = AtlasTable::m23();
    let mut total = 0;
    for c in rational {
        let tc = Instant::now();
        let ni = NielsenClass23::new(&g, &id, &lists, c).unwrap();
        let r = &p[c.name()];
        let genus = r["genus"].as_u64().unwrap() as usize;
        assert_eq!(ni.len() as u64, r["size"].as_u64().unwrap(), "{c}");
        let n = t
            .structure_constant([c, M23Class::C23A, M23Class::C23B])
            .unwrap();
        assert_eq!(
            n.to_string(),
            ni.len().to_string(),
            "every triple generates ({c})"
        );
        let reps: Vec<_> = r["representatives_g1"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| owner_cycles(v.as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(reps, ni.representatives(), "representatives_g1 of {c}");
        let conj = ni.complex_conjugation().unwrap();
        let mut real = Vec::new();
        for (i, &ci) in conj.iter().enumerate() {
            let tr = ni.triple(i);
            let dims: Vec<usize> = mods
                .iter()
                .map(|m| TripleCohomology::new(m, &tr).unwrap().dim_parabolic())
                .collect();
            assert_eq!(dims, vec![genus, genus, 2 * genus], "{c} class {i}");
            if ci == i {
                real.push(real_entry(&ni, &mods, i, genus));
            }
        }
        assert_eq!(
            real.len() as u64,
            r["real_classes"].as_u64().unwrap(),
            "{c}"
        );
        assert_eq!(serde_json::Value::Array(real.clone()), r["real"], "{c}");
        total += ni.len();
        println!(
            "({c},23A,23B): size {} real {} genus {genus}: {:?}",
            ni.len(),
            real.len(),
            tc.elapsed()
        );
    }
    assert_eq!(total, 6127);
    println!(
        "full_c_23a_23b total {:?}, VmHWM {} MB",
        t0.elapsed(),
        m23_common::vm_hwm_mb()
    );
}

/// The seven HJLPPZ triples, read at run time from $M23ISGALOIS (never vendored: no licence).
/// Magma right action, `g3*g2*g1 = 1`; the file's fixed `g2` is the published `g3`; the
/// RustMath triple is (file g1, file g3, file g2). Rows 1..7 map to [5, 4, 0, 2, 1, 3, 6]
/// (group_facts.json, hjlppz_file_row).
#[test]
#[ignore]
fn env_hjlppz_dictionary() {
    let dir =
        std::env::var("M23ISGALOIS").expect("set M23ISGALOIS to a checkout of shaowuz/m23isgalois");
    let t0 = Instant::now();
    let (g, id) = setup();
    let lists = lists_2a(&g, &id);
    let ni = NielsenClass23::new(&g, &id, &lists, M23Class::C2A).unwrap();
    let rows = read_group_facts_nielsen(&data_dir().join("group_facts.json")).unwrap();
    let path = format!("{dir}/seven_nielsen_class_representatives.m");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let perms = parse_magma_perms(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(perms.len(), 1 + 7 * 2, "{path}");
    let y = perms[0];
    assert_eq!(y, g.g3(), "the file's fixed entry is the published g3");
    let mut hj = Vec::new();
    for r in 0..7 {
        let (x, z) = (perms[1 + 2 * r], perms[2 + 2 * r]);
        let idx = ni.index_of(&(x, z, y)).unwrap();
        assert_eq!(rows[idx]["hjlppz_file_row"].as_u64(), Some(r as u64 + 1));
        hj.push(idx);
    }
    let mut sorted = hj.clone();
    sorted.sort();
    assert_eq!(sorted, (0..7).collect::<Vec<_>>(), "bijective");
    assert_eq!(hj, vec![5, 4, 0, 2, 1, 3, 6]);
    println!("HJLPPZ rows -> {hj:?}: {:?}", t0.elapsed());
}

/// Le Lay's seven representatives, read at run time from
/// $LELAY_M23/verification/certify_geometric_reflection_obstruction.g (CC BY-NC-SA 4.0, never
/// vendored): `y` is the published g3, each `x_k` an involution of M23; `z = (x y)^-1` and the
/// braid `(x, y z y^-1, y)`; classes [0, 1, 2, 4, 6, 5, 3] (HANDOFF Section 7, lelay_id); his
/// real classes 3, 6, 7 are the fixed classes 2, 5, 3.
#[test]
#[ignore]
fn env_lelay_dictionary() {
    let dir = std::env::var("LELAY_M23").expect(
        "set LELAY_M23 to a checkout of research-reflexivity/frontier-math-m23-cover-investigation",
    );
    let t0 = Instant::now();
    let (g, id) = setup();
    let lists = lists_2a(&g, &id);
    let ni = NielsenClass23::new(&g, &id, &lists, M23Class::C2A).unwrap();
    let rows = read_group_facts_nielsen(&data_dir().join("group_facts.json")).unwrap();
    let path = format!("{dir}/verification/certify_geometric_reflection_obstruction.g");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let y = gap_perms(gap_assignment(&text, "y").unwrap_or_else(|e| panic!("{path}: {e}")))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(y.len(), 1);
    let y = y[0];
    assert_eq!(y, g.g3(), "Le Lay's y is the published g3");
    let xs = gap_perms(
        gap_assignment(&text, "representatives").unwrap_or_else(|e| panic!("{path}: {e}")),
    )
    .unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(xs.len(), 7);
    let mut ll = Vec::new();
    for (k, x) in xs.iter().enumerate() {
        assert_eq!(
            id.class_of_checked(&g, x).unwrap(),
            M23Class::C2A,
            "x_{} is an involution of M23",
            k + 1
        );
        let z = inverse(&rmul(x, &y));
        let idx = ni
            .index_of(&(*x, rmul(&rmul(&y, &z), &inverse(&y)), y))
            .unwrap();
        assert_eq!(rows[idx]["lelay_id"].as_u64(), Some(k as u64 + 1));
        ll.push(idx);
    }
    assert_eq!(ll, vec![0, 1, 2, 4, 6, 5, 3], "HANDOFF Section 7");
    let conj = ni.complex_conjugation().unwrap();
    let mut lr = vec![ll[2], ll[5], ll[6]];
    lr.sort();
    assert_eq!(lr, (0..7).filter(|&i| conj[i] == i).collect::<Vec<_>>());
    println!("Le Lay x_1..x_7 -> {ll:?}: {:?}", t0.elapsed());
}
