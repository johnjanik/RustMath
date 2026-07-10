//! Test discovery and execution for Magma projects.
//!
//! This module provides functionality to discover test procedures in Magma files
//! and execute them using the Magma interpreter.

use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

/// A discovered test procedure.
#[derive(Debug, Clone)]
pub struct TestCase {
    /// Name of the procedure.
    pub name: String,
    /// Whether this test is marked as ignored (via `// ignore` comment).
    pub ignored: bool,
    /// Line number where the procedure is defined (1-indexed).
    pub line: usize,
}

/// Discover all top-level test procedures in a Magma source file.
///
/// A test procedure is any top-level `procedure` definition. Procedures
/// preceded by a `// ignore` comment (case-insensitive) are marked as ignored.
/// Nested procedures (defined inside other procedures) are not discovered.
pub fn discover_tests(source: &str) -> Result<Vec<TestCase>, DiscoveryError> {
    let mut parser = Parser::new();
    let language = tree_sitter_magma::LANGUAGE.into();
    parser
        .set_language(&language)
        .map_err(|e| DiscoveryError::LanguageError(e.to_string()))?;

    let tree = parser
        .parse(source, None)
        .ok_or(DiscoveryError::ParseError)?;

    // Query for top-level procedure definitions.
    // The tree structure is: (program (expression_statement (procedure_definition ...)))
    // We want procedures at the top level (direct child of program via expression_statement).
    let query_str = r#"
        (program
            (expression_statement
                (procedure_definition
                    name: (identifier) @name) @procedure))
    "#;

    let query =
        Query::new(&language, query_str).map_err(|e| DiscoveryError::QueryError(e.to_string()))?;

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());

    let name_idx = query
        .capture_index_for_name("name")
        .ok_or_else(|| DiscoveryError::QueryError("Missing 'name' capture".to_string()))?;
    let procedure_idx = query
        .capture_index_for_name("procedure")
        .ok_or_else(|| DiscoveryError::QueryError("Missing 'procedure' capture".to_string()))?;

    let mut tests = Vec::new();

    while let Some(m) = matches.next() {
        let mut name_node = None;
        let mut procedure_node = None;

        for capture in m.captures {
            if capture.index == name_idx {
                name_node = Some(capture.node);
            } else if capture.index == procedure_idx {
                procedure_node = Some(capture.node);
            }
        }

        if let (Some(name), Some(proc)) = (name_node, procedure_node) {
            let proc_name = name
                .utf8_text(source.as_bytes())
                .map_err(|e| DiscoveryError::Utf8Error(e.to_string()))?
                .to_string();

            let line = proc.start_position().row + 1; // 1-indexed

            // Check for `// ignore` comment before this procedure.
            let ignored = is_ignored(source, proc);

            tests.push(TestCase {
                name: proc_name,
                ignored,
                line,
            });
        }
    }

    Ok(tests)
}

/// Check if a node is preceded by an `// ignore` comment.
fn is_ignored(source: &str, node: tree_sitter::Node) -> bool {
    let start_byte = node.start_byte();
    if start_byte == 0 {
        return false;
    }

    // Look at the text before this node to find a preceding comment.
    let before = &source[..start_byte];

    // Find the last non-whitespace content before this node.
    // If it's a comment containing "ignore", mark as ignored.
    for line in before.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("//") {
            let comment_text = trimmed.trim_start_matches('/').trim();
            if comment_text.eq_ignore_ascii_case("ignore") {
                return true;
            }
        }
        // If we hit non-comment content, stop looking.
        break;
    }

    false
}

/// Generate the Magma command to run a test with try/catch error handling.
///
/// The command loads the file and runs the test procedure, wrapping it in
/// try/catch to capture any errors.
pub fn generate_test_command(file_path: &Path, test_name: &str) -> String {
    // Magma's try/catch syntax:
    // try <statement>; catch e <statement>; end try;
    //
    // We want to:
    // 1. Load the file
    // 2. Call the test procedure
    // 3. Print "PASS" if successful
    // 4. Print "FAIL: <error>" if it throws
    let file_str = file_path.display();
    format!(
        r#"load "{file_str}"; try {test_name}(); print "LAVA_TEST_PASS"; catch e print "LAVA_TEST_FAIL:", e; end try; quit;"#
    )
}

/// Errors that can occur during test discovery.
#[derive(Debug, thiserror::Error)]
pub enum DiscoveryError {
    #[error("Failed to set tree-sitter language: {0}")]
    LanguageError(String),

    #[error("Failed to parse source file")]
    ParseError,

    #[error("Tree-sitter query error: {0}")]
    QueryError(String),

    #[error("UTF-8 decoding error: {0}")]
    Utf8Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_simple_procedure() {
        let source = r#"
procedure test_foo()
    x := 1;
end procedure;
"#;
        let tests = discover_tests(source).unwrap();
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "test_foo");
        assert!(!tests[0].ignored);
    }

    #[test]
    fn test_discover_multiple_procedures() {
        let source = r#"
procedure test_one()
    x := 1;
end procedure;

procedure test_two()
    y := 2;
end procedure;
"#;
        let tests = discover_tests(source).unwrap();
        assert_eq!(tests.len(), 2);
        assert_eq!(tests[0].name, "test_one");
        assert_eq!(tests[1].name, "test_two");
    }

    #[test]
    fn test_ignored_procedure() {
        let source = r#"
// ignore
procedure test_skip()
    x := 1;
end procedure;

procedure test_run()
    y := 2;
end procedure;
"#;
        let tests = discover_tests(source).unwrap();
        assert_eq!(tests.len(), 2);
        assert!(tests[0].ignored);
        assert!(!tests[1].ignored);
    }

    #[test]
    fn test_ignore_case_insensitive() {
        let source = r#"
// IGNORE
procedure test_skip()
    x := 1;
end procedure;
"#;
        let tests = discover_tests(source).unwrap();
        assert_eq!(tests.len(), 1);
        assert!(tests[0].ignored);
    }

    #[test]
    fn test_functions_not_included() {
        let source = r#"
function helper(x)
    return x + 1;
end function;

procedure test_foo()
    y := helper(1);
end procedure;
"#;
        let tests = discover_tests(source).unwrap();
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "test_foo");
    }

    #[test]
    fn test_generate_command() {
        let path = Path::new("/path/to/test.m");
        let cmd = generate_test_command(path, "test_foo");
        assert!(cmd.contains("load \"/path/to/test.m\""));
        assert!(cmd.contains("test_foo()"));
        assert!(cmd.contains("try"));
        assert!(cmd.contains("catch"));
        assert!(cmd.contains("LAVA_TEST_PASS"));
        assert!(cmd.contains("LAVA_TEST_FAIL"));
    }
}
