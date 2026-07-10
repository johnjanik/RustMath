//! Golden tests: every fixture under topiary/test/{input,expected}/ must
//! round-trip through lava-core::format_str unchanged. This is the regression
//! net keeping lava and the upstream topiary CLI in agreement.

use lava_core::{FormatOptions, format_str};
use std::fs;
use std::path::PathBuf;

fn corpus_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("topiary-tests")
}

#[tokio::test(flavor = "current_thread")]
async fn all_fixtures_match_expected() {
    let inputs = corpus_root().join("input");
    let expected_dir = corpus_root().join("expected");

    let mut entries: Vec<_> = fs::read_dir(&inputs)
        .expect("topiary/test/input missing")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "m"))
        .collect();
    entries.sort_by_key(|e| e.path());
    assert!(!entries.is_empty(), "no fixtures found");

    let mut failures: Vec<String> = Vec::new();

    for entry in entries {
        let name = entry.file_name();
        let input = fs::read_to_string(entry.path()).unwrap();
        let expected_path = expected_dir.join(&name);
        let Ok(expected) = fs::read_to_string(&expected_path) else {
            failures.push(format!("{}: missing expected file", name.to_string_lossy()));
            continue;
        };
        match format_str(&input, &FormatOptions::default()) {
            Ok((actual, _)) if actual == expected => {}
            Ok((actual, _)) => {
                failures.push(format!(
                    "{}: output differs\n--- expected\n{}\n--- actual\n{}",
                    name.to_string_lossy(),
                    expected,
                    actual
                ));
            }
            Err(e) => {
                failures.push(format!("{}: format errored: {e}", name.to_string_lossy()));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} fixture(s) failed:\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}
