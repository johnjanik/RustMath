use crate::cli::TestArgs;
use anyhow::{Context, Result, anyhow};
use futures::stream::{self, StreamExt};
use lava_core::{TestCase, discover_tests, generate_test_command};
use nu_ansi_term::Color;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::process::Command;

/// Helper for colorized output with TTY detection.
struct Colors {
    enabled: bool,
}

impl Colors {
    fn new() -> Self {
        Self {
            enabled: std::io::stderr().is_terminal(),
        }
    }

    fn green(&self, s: &str) -> String {
        if self.enabled {
            Color::Green.paint(s).to_string()
        } else {
            s.to_string()
        }
    }

    fn red(&self, s: &str) -> String {
        if self.enabled {
            Color::Red.paint(s).to_string()
        } else {
            s.to_string()
        }
    }

    fn gray(&self, s: &str) -> String {
        if self.enabled {
            Color::Fixed(245).paint(s).to_string()
        } else {
            s.to_string()
        }
    }
}

/// Result of running a single test.
#[derive(Debug)]
struct TestResult {
    file: PathBuf,
    name: String,
    outcome: TestOutcome,
    duration: Duration,
}

#[derive(Debug)]
enum TestOutcome {
    Passed,
    Failed(String),
    Ignored,
    Error(String),
}

/// A test to run, combining file path and test case info.
#[derive(Debug, Clone)]
struct TestToRun {
    file: PathBuf,
    test: TestCase,
}

pub async fn run(args: TestArgs) -> Result<i32> {
    let colors = Colors::new();
    let magma_path = find_magma(&args.magma)?;
    let test_files = discover_test_files(&args.paths)?;

    if test_files.is_empty() {
        eprintln!("No test files found.");
        return Ok(0);
    }

    // Discover all tests from all files.
    let mut all_tests: Vec<TestToRun> = Vec::new();
    for file in &test_files {
        let source = std::fs::read_to_string(file)
            .with_context(|| format!("reading {}", file.display()))?;
        let tests = discover_tests(&source)
            .map_err(|e| anyhow!("discovering tests in {}: {}", file.display(), e))?;
        for test in tests {
            all_tests.push(TestToRun {
                file: file.clone(),
                test,
            });
        }
    }

    if all_tests.is_empty() {
        eprintln!("No test procedures found.");
        return Ok(0);
    }

    let total = all_tests.len();
    let ignored_count = all_tests.iter().filter(|t| t.test.ignored).count();

    eprintln!("running {} tests", total);

    // Run tests in parallel using tokio.
    let results: Vec<TestResult> = stream::iter(all_tests)
        .map(|test_to_run| {
            let magma = magma_path.clone();
            async move { run_single_test(&magma, test_to_run).await }
        })
        .buffer_unordered(num_cpus())
        .collect()
        .await;

    // Print results.
    let mut passed = 0;
    let mut failed = 0;
    let mut failed_tests: Vec<&TestResult> = Vec::new();

    for result in &results {
        let (status, should_print) = match &result.outcome {
            TestOutcome::Passed => {
                passed += 1;
                (colors.green("ok"), true)
            }
            TestOutcome::Failed(_) => {
                failed += 1;
                failed_tests.push(result);
                (colors.red("FAILED"), true)
            }
            TestOutcome::Ignored => (colors.gray("ignored"), args.include_ignored),
            TestOutcome::Error(_) => {
                failed += 1;
                failed_tests.push(result);
                (colors.red("ERROR"), true)
            }
        };

        if should_print {
            eprintln!(
                "test {}::{} ... {} ({:.2?})",
                result.file.display(),
                result.name,
                status,
                result.duration
            );
        }
    }

    eprintln!();

    // Print failure details.
    if !failed_tests.is_empty() {
        eprintln!("failures:");
        eprintln!();
        for result in &failed_tests {
            eprintln!("---- {}::{} ----", result.file.display(), result.name);
            match &result.outcome {
                TestOutcome::Failed(msg) => eprintln!("{}", msg),
                TestOutcome::Error(msg) => eprintln!("error: {}", msg),
                _ => {}
            }
            eprintln!();
        }
    }

    // Summary.
    let result_str = if failed > 0 {
        colors.red("FAILED")
    } else {
        colors.green("ok")
    };
    eprintln!(
        "test result: {}. {} passed; {} failed; {} ignored",
        result_str, passed, failed, ignored_count
    );

    if failed > 0 {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Run a single test and return its result.
async fn run_single_test(magma_path: &Path, test_to_run: TestToRun) -> TestResult {
    let TestToRun { file, test } = test_to_run;

    if test.ignored {
        return TestResult {
            file,
            name: test.name,
            outcome: TestOutcome::Ignored,
            duration: Duration::ZERO,
        };
    }

    let start = Instant::now();

    // Generate and run the magma command.
    let cmd_str = generate_test_command(&file, &test.name);

    let output = match Command::new(magma_path)
        .arg("-b")
        .arg("-e")
        .arg(&cmd_str)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
    {
        Ok(output) => output,
        Err(e) => {
            return TestResult {
                file,
                name: test.name,
                outcome: TestOutcome::Error(format!("failed to run magma: {}", e)),
                duration: start.elapsed(),
            };
        }
    };

    let duration = start.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let outcome = parse_test_output(&stdout, &stderr);

    TestResult {
        file,
        name: test.name,
        outcome,
        duration,
    }
}

/// Find the magma binary.
fn find_magma(override_path: &Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        if path.exists() {
            return Ok(path.clone());
        }
        anyhow::bail!("magma binary not found at {}", path.display());
    }

    // Try to find magma in PATH.
    which::which("magma")
        .map_err(|_| anyhow!("magma not found in PATH. Use --magma to specify the path."))
}

/// Discover test files from the given paths.
fn discover_test_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    use ignore::WalkBuilder;

    // If no paths given, look for test/ or tests/ directories.
    let roots: Vec<PathBuf> = if paths.is_empty() {
        let test_dir = PathBuf::from("test");
        let tests_dir = PathBuf::from("tests");
        if test_dir.exists() {
            vec![test_dir]
        } else if tests_dir.exists() {
            vec![tests_dir]
        } else {
            return Ok(Vec::new());
        }
    } else {
        paths.to_vec()
    };

    let mut files: Vec<PathBuf> = Vec::new();

    for root in roots {
        if root.is_file() {
            if is_magma_file(&root) {
                files.push(root);
            }
            continue;
        }

        let mut wb = WalkBuilder::new(&root);
        wb.require_git(false);
        wb.add_custom_ignore_filename(".lavaignore");

        for result in wb.build() {
            let entry = result.with_context(|| format!("walking {}", root.display()))?;
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let p = entry.path();
            if is_magma_file(p) {
                files.push(p.to_path_buf());
            }
        }
    }

    files.sort();
    files.dedup();
    Ok(files)
}

fn is_magma_file(path: &Path) -> bool {
    path.extension()
        .map(|e| e == "m" || e == "magma")
        .unwrap_or(false)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Parse Magma output to determine test outcome.
///
/// Looks for `LAVA_TEST_PASS` or `LAVA_TEST_FAIL:` markers in the output.
/// For failures, extracts the full multi-line error message after the marker.
fn parse_test_output(stdout: &str, stderr: &str) -> TestOutcome {
    if stdout.contains("LAVA_TEST_PASS") {
        TestOutcome::Passed
    } else if stdout.contains("LAVA_TEST_FAIL:") {
        // Extract everything after LAVA_TEST_FAIL: marker.
        // Magma prints each argument on separate lines, so the error message
        // spans multiple lines after the marker.
        let error_msg = stdout
            .split_once("LAVA_TEST_FAIL:")
            .map(|(_, rest)| rest.trim().to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        TestOutcome::Failed(error_msg)
    } else {
        // No marker found - likely a crash or syntax error.
        let combined = format!(
            "stdout:\n{}\nstderr:\n{}",
            stdout.trim(),
            stderr.trim()
        );
        TestOutcome::Error(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_output_pass() {
        let stdout = "Loading \"test.m\"\nLAVA_TEST_PASS\n";
        let stderr = "";
        match parse_test_output(stdout, stderr) {
            TestOutcome::Passed => {}
            other => panic!("Expected Passed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_output_fail_single_line_error() {
        let stdout = "Loading \"test.m\"\nLAVA_TEST_FAIL: Runtime error: oops\n";
        let stderr = "";
        match parse_test_output(stdout, stderr) {
            TestOutcome::Failed(msg) => {
                assert!(msg.contains("Runtime error: oops"), "msg was: {}", msg);
            }
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_output_fail_multiline_assertion_error() {
        // This is the actual output format from Magma for assertion failures.
        // The error is printed on multiple lines after LAVA_TEST_FAIL:
        let stdout = r#"Loading "test/test.m"
LAVA_TEST_FAIL: 
test_fail(
)
In file "test/test.m", line 7, column 5:
>>     assert 1 eq 2;
       ^
Runtime error in assert: Assertion failed
"#;
        let stderr = "";
        match parse_test_output(stdout, stderr) {
            TestOutcome::Failed(msg) => {
                // Should capture the full multi-line error
                assert!(msg.contains("test_fail"), "should contain procedure name: {}", msg);
                assert!(msg.contains("assert 1 eq 2"), "should contain source line: {}", msg);
                assert!(msg.contains("Assertion failed"), "should contain error message: {}", msg);
            }
            other => panic!("Expected Failed, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_output_error_no_marker() {
        // When Magma crashes or has a syntax error before our markers
        let stdout = "Syntax error at line 1";
        let stderr = "some stderr output";
        match parse_test_output(stdout, stderr) {
            TestOutcome::Error(msg) => {
                assert!(msg.contains("Syntax error"), "should contain stdout: {}", msg);
                assert!(msg.contains("some stderr"), "should contain stderr: {}", msg);
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }
}
