//! The ATLAS character table of M23 as data (WP-P, HANDOFF 4.2; MAGMA ch. 91), its match with
//! RustMath's classes, the structure constants, and the T-215 inner products on a non-abelian
//! table.
//!
//! References: GAP 4.12.1 with CTblLib 1.3.7, `CharacterTable("M23")`, written by
//! `data/m23/m23_atlas_table.g` (the embedded file); the owner's
//! `data/m23_three_point_passports.json` (structure constants computed there by GAP from the
//! same table), vendored in `tests/data/m23/`.

mod m23_common;

use std::time::Instant;

use rustmath_groups::m23::{AtlasNumber, AtlasTable, ClassIdentifier, M23Class, M23};
use rustmath_rationals::Rational;

/// GAP 4.12.1: `SizesCentralizers(CharacterTable("M23"))`.
const GAP_CENTRALIZERS: [u64; 17] = [
    10200960, 2688, 180, 32, 15, 12, 14, 14, 8, 11, 11, 14, 14, 15, 15, 23, 23,
];

#[test]
fn atlas_file_is_the_gap_output() {
    let t = AtlasTable::m23();
    assert_eq!(
        t.source(),
        "GAP 4.12.1, CTblLib 1.3.7, CharacterTable(\"M23\")"
    );
    assert_eq!(t.num_irreducibles(), 17);
    for c in M23Class::ALL {
        assert_eq!(t.centralizer_order(c), GAP_CENTRALIZERS[c.index()], "{c}");
        assert_eq!(t.element_order(c), c.element_order(), "{c}");
        assert_eq!(t.is_rational_class(c), c.is_rational(), "{c}");
        // the power map -1 is the inverse class
        assert_eq!(t.power_map(-1, c), Some(c.inverse()), "{c}");
    }
    assert_eq!(t.power_map_keys(), vec![-1, 2, 3, 5, 7, 11, 23]);
    assert_eq!(t.power_map(4, M23Class::C2A), None);
    // degrees of the irreducibles (ATLAS p. 71): 1, 22, 45, 45, 230, 231, 231, 231, 253, 770,
    // 770, 896, 896, 990, 990, 1035, 2024
    let mut degrees: Vec<i64> = (0..17)
        .map(|i| {
            t.value(i, M23Class::C1A)
                .rational_part()
                .unwrap()
                .to_string()
                .parse()
                .unwrap()
        })
        .collect();
    degrees.sort();
    assert_eq!(
        degrees,
        vec![1, 22, 45, 45, 230, 231, 231, 231, 253, 770, 770, 896, 896, 990, 990, 1035, 2024]
    );
}

#[test]
fn atlas_orthogonality_and_57_constants() {
    let t = AtlasTable::m23();
    let t0 = Instant::now();
    t.check_orthogonality().unwrap();
    println!("row and column orthogonality, exact: {:?}", t0.elapsed());
    let rows = m23_common::read_passport_rows(
        &m23_common::data_dir().join("m23_three_point_passports.json"),
    )
    .unwrap();
    assert_eq!(rows.len(), 57);
    for (_, classes, n) in &rows {
        let got = t.structure_constant(*classes).unwrap();
        assert_eq!(got.to_string(), *n, "{classes:?}");
        // n is symmetric in the three classes
        let rev = t
            .structure_constant([classes[2], classes[1], classes[0]])
            .unwrap();
        assert_eq!(rev, got);
    }
    // (2A, 23A, 23B): n = 7. MAGMA's class-algebra coefficient a_{2A, 23A, 23A} is
    // n |C(23B)| = 161 (review 2, n4).
    let n = t
        .structure_constant([M23Class::C2A, M23Class::C23A, M23Class::C23B])
        .unwrap();
    assert_eq!(n, Rational::from(7i64));
    assert_eq!(&n * &Rational::from(23i64), Rational::from(161i64));
}

// Element orders, class sizes and centralizer orders (computed by RustMath), the power maps
// 2, 3, 5, 7, 11, 23, -1 and the permutation character of GAP's table, on RustMath's classes.
#[test]
fn class_data_match_gap() {
    let g = M23::published().unwrap();
    let id = ClassIdentifier::new(&g).unwrap();
    let t0 = Instant::now();
    AtlasTable::m23().check_against(&g, &id).unwrap();
    println!("check_against: {:?}", t0.elapsed());
    // the permutation character is the number of fixed points, also on the generators
    let t = AtlasTable::m23();
    assert_eq!(t.permutation_character_23(M23Class::C2A), 7);
    assert_eq!(t.permutation_character_23(M23Class::C23A), 0);
}

#[test]
fn atlas_numbers() {
    // b7 = (-1 + sqrt(-7)) / 2 satisfies b^2 + b + 2 = 0
    let b7 = AtlasNumber::from_atlas(0, 1, 7).unwrap();
    let two = AtlasNumber::from_rational(Rational::from(2i64));
    let sum = &(&(&b7 * &b7) + &b7) + &two;
    assert_eq!(sum, AtlasNumber::zero());
    // conj(b7) = -1 - b7, and b7 conj(b7) = 2
    let one = AtlasNumber::from_rational(Rational::from(1i64));
    assert_eq!(b7.conj(), &(-&one) - &b7);
    assert_eq!(
        (&b7 * &b7.conj()).rational_part(),
        Some(Rational::from(2i64))
    );
    assert_eq!(b7.rational_part(), None);
    let v = b7.to_complex_f64();
    assert!((v.real() + 0.5).abs() < 1e-12 && (v.imag() - 7f64.sqrt() / 2.0).abs() < 1e-12);
    // b23 b15 is in the degree 4 part
    let p =
        &AtlasNumber::from_atlas(0, 1, 23).unwrap() * &AtlasNumber::from_atlas(0, 1, 15).unwrap();
    assert_eq!(p.rational_part(), None);
    assert!(AtlasNumber::from_atlas(1, 1, 5).is_err());
    assert!(AtlasNumber::from_atlas(1, 1, 1).is_err());
}

#[test]
fn atlas_parse_rejects_bad_files() {
    let good = include_str!("../data/m23/m23_atlas_table.json");
    assert!(AtlasTable::parse(good).is_ok());
    // two columns swapped in the class names
    let bad = good.replacen("\"7a\",\"7b\"", "\"7b\",\"7a\"", 1);
    assert!(AtlasTable::parse(&bad).is_err());
    // a value outside the four fields
    let bad = good.replacen("[0,1,7]", "[0,1,5]", 1);
    assert!(AtlasTable::parse(&bad).is_err());
    // review 1: no silent casts. A negative centralizer, and d = 2^32 + 7 (which 'as u32'
    // turned into 7), are refused with the field named.
    let bad = good.replacen(
        "\"centralizer_orders\":[10200960,",
        "\"centralizer_orders\":[-1,",
        1,
    );
    assert_ne!(bad, good);
    let e = AtlasTable::parse(&bad).err().unwrap();
    assert!(e.contains("centralizer_orders"), "{e}");
    let bad = good.replacen("[0,1,7]", "[0,1,4294967303]", 1);
    let e = AtlasTable::parse(&bad).err().unwrap();
    assert!(e.starts_with("d:"), "{e}");
    // a centralizer of i64::MAX at 2A parses; check_orthogonality refuses it, and
    // |C| |C(c)| = 3795 (2^63 - 1) overflows u64: an Err, not a debug overflow panic
    let bad = good.replacen(
        "\"centralizer_orders\":[10200960,2688,",
        "\"centralizer_orders\":[10200960,9223372036854775807,",
        1,
    );
    assert_ne!(bad, good);
    let t = AtlasTable::parse(&bad).unwrap();
    let e = t.check_orthogonality().err().unwrap();
    assert!(e.starts_with("|C| |C(c)| != |G| at"), "{e}");
    assert!(AtlasTable::parse("{}").is_err());
    assert!(AtlasTable::parse("not json").is_err());
}

// T-215 on a non-abelian table: the 17 irreducibles of M23 as an f64 CharacterTable with class
// sizes are orthonormal for the weighted inner product (to 1e-9).
#[test]
fn t215_m23_float_table_is_orthonormal() {
    let ft = AtlasTable::m23().to_float_table();
    assert_eq!(ft.order(), 10_200_960);
    let chars = ft.irreducible_characters();
    assert_eq!(chars.len(), 17);
    for (i, a) in chars.iter().enumerate() {
        assert!(a.is_irreducible(), "chi_{i}");
        for (j, b) in chars.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            let ip = a.inner_product(b);
            assert!(
                (ip.real() - want).abs() < 1e-9 && ip.imag().abs() < 1e-9,
                "<chi_{i}, chi_{j}> = {ip:?}"
            );
            let ip = ft.inner_product(a, b).unwrap();
            assert!((ip.real() - want).abs() < 1e-9 && ip.imag().abs() < 1e-9);
        }
    }
    assert!(ft.verify_column_orthogonality());
}
