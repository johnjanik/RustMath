use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("lava-core")
        .join("tests")
        .join("topiary-tests")
        .join(name)
}

#[test]
fn highlight_stdin_to_stdout() {
    Command::cargo_bin("lava")
        .unwrap()
        .args(["highlight"])
        .write_stdin("function f() return 1; end function;")
        .assert()
        .success();
}


#[test]
fn format_stdin_to_stdout() {
    let input = fs::read_to_string(fixture("input/assert_statement.m")).unwrap();
    let expected = fs::read_to_string(fixture("expected/assert_statement.m")).unwrap();
    Command::cargo_bin("lava")
        .unwrap()
        .args(["format"])
        .write_stdin(input)
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn format_single_file_to_stdout() {
    let expected = fs::read_to_string(fixture("expected/assert_statement.m")).unwrap();
    Command::cargo_bin("lava")
        .unwrap()
        .arg("format")
        .arg(fixture("input/assert_statement.m"))
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn format_multi_file_no_w_errors() {
    Command::cargo_bin("lava")
        .unwrap()
        .arg("format")
        .arg(fixture("input/assert_statement.m"))
        .arg(fixture("input/comprehensions.m"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("refusing to concatenate"));
}

#[test]
fn format_w_rewrites_in_place() {
    let tmp = tempfile::tempdir().unwrap();
    let target = tmp.path().join("a.m");
    fs::copy(fixture("input/assert_statement.m"), &target).unwrap();
    Command::cargo_bin("lava")
        .unwrap()
        .args(["format", "-w"])
        .arg(&target)
        .assert()
        .success();
    let actual = fs::read_to_string(&target).unwrap();
    let expected = fs::read_to_string(fixture("expected/assert_statement.m")).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn format_check_clean_file_exits_0() {
    Command::cargo_bin("lava")
        .unwrap()
        .args(["format", "--check"])
        .arg(fixture("expected/assert_statement.m"))
        .assert()
        .success();
}

#[test]
fn format_check_dirty_file_exits_1() {
    Command::cargo_bin("lava")
        .unwrap()
        .args(["format", "--check"])
        .arg(fixture("input/assert_statement.m"))
        .assert()
        .code(1)
        .stderr(predicate::str::contains("would be reformatted"));
}

#[test]
fn fmt_alias_works() {
    Command::cargo_bin("lava")
        .unwrap()
        .args(["fmt", "--help"])
        .assert()
        .success();
}
