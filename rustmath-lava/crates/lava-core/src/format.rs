//! Synchronous public API around topiary-core's formatter.

use crate::error::{Error, QuerySource, Result};
use crate::queries;
use std::borrow::Cow;
use std::path::PathBuf;
use topiary_core::{Language, Operation, TopiaryQuery, formatter};

/// Options for `format_str`.
#[derive(Debug, Default, Clone)]
pub struct FormatOptions {
    /// Path to a custom `magma.scm`. If `None`, the resolution order in
    /// [`crate::queries::resolve`] applies, starting from `cwd`.
    pub query_override: Option<PathBuf>,
    /// CWD used as the starting point for the walk-up resolution.
    /// Defaults to the process CWD.
    pub cwd: Option<PathBuf>,
    /// If true, the formatter does not require a clean parse.
    pub tolerate_parse_errors: bool,
    /// If true, skip the idempotence check (faster; for trusted callers).
    pub skip_idempotence: bool,
}

/// Format a Magma source string.
///
/// Returns the formatted source plus the resolved query source so callers
/// can log/display where the query came from.
pub fn format_str(source: &str, opts: &FormatOptions) -> Result<(String, QuerySource)> {
    let cwd: PathBuf = match &opts.cwd {
        Some(p) => p.clone(),
        None => std::env::current_dir().map_err(|source| Error::Io {
            path: PathBuf::from("."),
            source,
        })?,
    };

    let (query_str, query_src): (Cow<'static, str>, QuerySource) =
        queries::resolve(&cwd, opts.query_override.as_deref()).map_err(|source| Error::Io {
            path: opts
                .query_override
                .clone()
                .unwrap_or_else(|| PathBuf::from("<query>")),
            source,
        })?;

    let grammar = topiary_tree_sitter_facade::Language::from(tree_sitter_magma::LANGUAGE);
    let query = TopiaryQuery::new(&grammar, &query_str).map_err(|e| Error::Query {
        from: query_src.clone(),
        message: e.to_string(),
    })?;

    let language = Language {
        name: "magma".to_owned(),
        query,
        grammar,
        indent: Some("    ".to_owned()),
    };

    let mut input_buf = source.as_bytes();
    let mut output: Vec<u8> = Vec::new();

    formatter(
        &mut input_buf,
        &mut output,
        &language,
        Operation::Format {
            skip_idempotence: opts.skip_idempotence,
            tolerate_parsing_errors: opts.tolerate_parse_errors,
        },
    )?;

    let formatted = String::from_utf8(output).map_err(|_| Error::Parse {
        path: "<output>".into(),
        message: "formatter produced invalid UTF-8".into(),
    })?;

    Ok((formatted, query_src))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("topiary-tests")
            .join(name)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn formats_a_known_fixture() {
        let input = std::fs::read_to_string(fixture("input/assert_statement.m"))
            .expect("fixture missing");
        let expected = std::fs::read_to_string(fixture("expected/assert_statement.m"))
            .expect("fixture missing");
        let (actual, _src) = format_str(&input, &FormatOptions::default()).expect("format failed");
        pretty_assertions::assert_eq!(actual, expected);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn round_trip_is_stable() {
        let input = std::fs::read_to_string(fixture("input/assert_statement.m")).unwrap();
        let (once, _) = format_str(&input, &FormatOptions::default()).unwrap();
        let (twice, _) = format_str(&once, &FormatOptions::default()).unwrap();
        pretty_assertions::assert_eq!(once, twice, "format is not idempotent");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parse_error_surfaces_as_error() {
        let input = "function f(\n  // unclosed";
        let result = format_str(input, &FormatOptions::default());
        assert!(matches!(
            result,
            Err(Error::Topiary(_)) | Err(Error::Parse { .. })
        ));
    }
}
