//! Demonstration of the Cremona Elliptic Curve Database interface
//!
//! Run with: cargo run --example cremona_demo

use rustmath_databases::cremona::{CremonaDatabase, CurveLabel, EllipticCurve, WeierstrassEquation, Point};

fn main() {
    println!("=== Cremona Elliptic Curve Database Demo ===\n");

    let db = CremonaDatabase::new();

    // Example 1: Looking up curves by label
    println!("1. Looking Up Curves by Label:");
    println!("   Cremona curves are labeled as <conductor><class><number>\n");

    let labels = vec!["11a1", "37a1", "389a1", "5077a1"];

    for label in labels {
        if let Some(curve) = db.lookup_curve(label) {
            println!("   Curve {}:", label);
            println!("      Conductor: {}", curve.conductor);
            println!("      Equation: {}", curve.equation);
            println!("      Rank: {}", curve.rank);
            println!("      Torsion: {}", curve.torsion_order);
            if !curve.generators.is_empty() {
                println!("      Generators: {:?}", curve.generators);
            }
            println!();
        }
    }

    // Example 2: Parsing curve labels
    println!("2. Parsing Curve Labels:");
    println!("   Understanding the label structure\n");

    let label_examples = vec!["11a1", "37b2", "389a1", "5077a1"];

    for label_str in label_examples {
        if let Ok(label) = CurveLabel::parse(label_str) {
            println!("   Label: {}", label);
            println!("      Conductor: {}", label.conductor);
            println!("      Isogeny Class: {}", label.isogeny_class);
            println!("      Curve Number: {}", label.curve_number);
            println!();
        }
    }

    // Example 3: Curves of a given conductor
    println!("3. All Curves of Conductor 11:");
    println!("   Finding all curves with the same conductor\n");

    let conductor_11 = db.curves_of_conductor(11);
    println!("   Found {} curves of conductor 11:", conductor_11.len());

    for curve in conductor_11 {
        println!("      {} - {}", curve.label, curve.equation);
    }

    // Example 4: Isogeny classes
    println!("\n4. Isogeny Classes:");
    println!("   Curves in the same isogeny class are connected by isogenies\n");

    let isogeny_class = db.curves_in_isogeny_class(11, "a");
    println!("   Isogeny class 11a has {} curves:", isogeny_class.len());

    for curve in isogeny_class {
        println!("      {}", curve.label);
    }

    // Example 5: Curves by rank
    println!("\n5. Curves by Rank:");
    println!("   Finding curves with specific ranks\n");

    for rank in 0..=3 {
        let curves = db.curves_of_rank(rank);
        println!("   Rank {}: {} curves in database", rank, curves.len());
        if !curves.is_empty() {
            print!("      Examples: ");
            for (i, curve) in curves.iter().take(5).enumerate() {
                if i > 0 { print!(", "); }
                print!("{}", curve.label);
            }
            println!();
        }
    }

    // Example 6: Torsion subgroups
    println!("\n6. Torsion Subgroups:");
    println!("   Finding curves with specific torsion orders\n");

    let torsion_orders = vec![1, 2, 3, 4, 5, 6, 8];

    for torsion in torsion_orders {
        let curves = db.curves_with_torsion(torsion);
        if !curves.is_empty() {
            println!("   Torsion order {}: {} curves", torsion, curves.len());
            if let Some(example) = curves.first() {
                println!("      Example: {} ({})", example.label, example.equation);
            }
        }
    }

    // Example 7: High rank curves
    println!("\n7. High Rank Curves:");
    println!("   Examples of curves with rank ≥ 2\n");

    let high_rank = db.curves_of_rank(2);
    for curve in high_rank.iter().take(3) {
        println!("   Curve {} (rank {}):", curve.label, curve.rank);
        println!("      Equation: {}", curve.equation);
        println!("      Generators: {} points", curve.generators.len());
        for (i, gen) in curve.generators.iter().enumerate() {
            println!("         P{} = {}", i + 1, gen);
        }
        println!();
    }

    // Example 8: Weierstrass equations
    println!("8. Weierstrass Equation Format:");
    println!("   y^2 + a1*xy + a3*y = x^3 + a2*x^2 + a4*x + a6\n");

    let curve_11a1 = db.lookup_curve("11a1").unwrap();
    let eq = &curve_11a1.equation;
    println!("   Curve 11a1 coefficients: [{}, {}, {}, {}, {}]",
             eq.a1, eq.a2, eq.a3, eq.a4, eq.a6);
    println!("   Displayed as: {}", eq);

    // Example 9: Database statistics
    println!("\n9. Database Statistics:");
    println!("   Overview of the database contents\n");

    let total_curves = db.curve_count();
    let conductors = db.all_conductors();

    println!("   Total curves: {}", total_curves);
    println!("   Conductors: {} different values", conductors.len());
    println!("   Smallest conductor: {}", conductors.first().unwrap_or(&0));
    println!("   Largest conductor: {}", conductors.last().unwrap_or(&0));

    // Example 10: Famous curves
    println!("\n10. Famous Elliptic Curves:");
    println!("    Some well-known curves in number theory\n");

    let famous = vec![
        ("11a1", "Smallest conductor curve"),
        ("37a1", "First rank 1 curve"),
        ("389a1", "Rank 2 example"),
        ("5077a1", "Famous rank 3 curve (Selmer)"),
    ];

    for (label, description) in famous {
        if let Some(curve) = db.lookup_curve(label) {
            println!("    {} - {}", label, description);
            println!("       {}", curve.equation);
            println!("       Rank: {}, Torsion: {}", curve.rank, curve.torsion_order);
            println!();
        }
    }

    // Example 11: Working with points
    println!("11. Points on Elliptic Curves:");
    println!("    Generators of the Mordell-Weil group\n");

    let curve_37a1 = db.lookup_curve("37a1").unwrap();
    println!("    Curve 37a1 has rank {}", curve_37a1.rank);

    for (i, point) in curve_37a1.generators.iter().enumerate() {
        println!("       Generator {}: {}", i + 1, point);
    }

    // Example 12: Curve labels and naming
    println!("\n12. Understanding Curve Labels:");
    println!("    The Cremona labeling system\n");

    println!("    Format: <N><C><i>");
    println!("       N = conductor (positive integer)");
    println!("       C = isogeny class ('a', 'b', 'c', ...)");
    println!("       i = curve number in class (1, 2, 3, ...)");
    println!();
    println!("    Examples:");
    println!("       11a1 → conductor 11, class a, first curve");
    println!("       37b2 → conductor 37, class b, second curve");
    println!("       389a1 → conductor 389, class a, first curve");

    println!("\n=== Demo Complete ===");
}
