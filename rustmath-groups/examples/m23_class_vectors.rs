//! Prints the 57 three-point class vectors of M23 with their counts and statuses (WP-P,
//! HANDOFF 4.1). MAGMA ch. 58 and 91.
//!
//! ```text
//! cargo run --release -p rustmath-groups --example m23_class_vectors
//! cargo run --release -p rustmath-groups --example m23_class_vectors -- --check rustmath-groups/tests/data/m23
//! ```
//!
//! With `--check <dir>` it reads `<dir>/m23_three_point_nielsen.json` (the owner's reference)
//! and prints `reproduced` or `MISMATCH` on each row; the exit code is 1 on any mismatch.

#[path = "../tests/m23_common/mod.rs"]
mod m23_common;

use std::path::Path;
use std::time::Instant;

use rustmath_groups::m23::{
    count_class_vector, enumerate_class_vectors, AtlasTable, ClassIdentifier, ClassLists,
    ClassVector, CountStatus, M23Class, VectorKind, M23,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let check_dir = args
        .iter()
        .position(|a| a == "--check")
        .map(|i| args.get(i + 1).expect("--check needs a directory").clone());
    let reference = check_dir.as_ref().map(|d| {
        m23_common::read_nielsen_rows(&Path::new(d).join("m23_three_point_nielsen.json"))
            .unwrap_or_else(|e| panic!("{e}"))
    });

    let t0 = Instant::now();
    let g = M23::published().expect("M23 on the published generators");
    let id = ClassIdentifier::new(&g).expect("class identifier");
    let t = AtlasTable::m23();
    let rational = [
        M23Class::C2A,
        M23Class::C3A,
        M23Class::C4A,
        M23Class::C5A,
        M23Class::C6A,
        M23Class::C8A,
    ];
    let lists = ClassLists::by_enumeration(&g, &id, &rational).expect("class lists");
    let mut vectors = enumerate_class_vectors(t, &id).expect("class vectors");
    // the order of HANDOFF 4.1: conjugate pairs, then rational, each by size
    let mut counts = Vec::new();
    for v in vectors.drain(..) {
        let tr = Instant::now();
        let c = count_class_vector(&g, &id, t, &lists, &v).expect("count");
        counts.push((c, tr.elapsed()));
    }
    counts.sort_by_key(|(c, _)| (c.vector.kind != VectorKind::ConjugatePair, c.size));
    println!("source of the table: {}", t.source());
    println!(
        "{:<15} {:<16} {:>5} {:>9} {:>9} {:>6} {:>6} {:>5}  {:<8} {:>9}  trusted",
        "type",
        "class vector",
        "genus",
        "n",
        "survivors",
        "|C(b)|",
        "size",
        "real",
        "status",
        "time"
    );
    let mut mismatches = 0;
    let mut reproduced = 0;
    for (c, el) in &counts {
        let status = match c.status {
            CountStatus::Decided => "Decided",
        };
        let trusted: Vec<&str> = c.trusted.iter().map(|r| r.label()).collect();
        let mut line = format!(
            "{:<15} {:<16} {:>5} {:>9} {:>9} {:>6} {:>6} {:>5}  {:<8} {:>9.2?}  {}",
            c.vector.kind.name(),
            c.vector.to_string(),
            c.genus23,
            c.structure_constant.to_string(),
            c.survivors,
            c.centralizer_order,
            c.size,
            c.real,
            status,
            el,
            trusted.join("; ")
        );
        if let Some(rows) = &reference {
            let want = rows.iter().find(|r| {
                ClassVector {
                    kind: if r.kind == "rational" {
                        VectorKind::Rational
                    } else {
                        VectorKind::ConjugatePair
                    },
                    classes: r.classes,
                } == c.vector
            });
            let ok = want.is_some_and(|r| {
                r.size == c.size
                    && r.real == c.real
                    && r.genus23 == c.genus23
                    && r.structure_constant == c.structure_constant.to_string()
            });
            if ok {
                reproduced += 1;
                line.push_str("  reproduced");
            } else {
                mismatches += 1;
                line.push_str("  MISMATCH");
            }
        }
        println!("{line}");
    }
    println!("{} class vectors in {:?}", counts.len(), t0.elapsed());
    if let Some(rows) = &reference {
        println!(
            "{reproduced} reproduced, {mismatches} mismatches, {} reference rows",
            rows.len()
        );
        if mismatches > 0 || reproduced != rows.len() {
            std::process::exit(1);
        }
    }
}
