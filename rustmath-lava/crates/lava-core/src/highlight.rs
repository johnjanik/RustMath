use std::io::Write;
use std::path::PathBuf;
use tree_sitter_highlight::{HighlightConfiguration, HighlightEvent, Highlighter, HtmlRenderer};
use crate::Result;
use crate::error::Error;

/// Target output format for syntax highlighting.
#[derive(Debug, Default, Clone, Copy)]
pub enum HighlightTarget {
    /// ANSI-escaped terminal output (default).
    #[default]
    Terminal,
    /// HTML `<span class="...">` output.
    Html,
}

/// Options for `highlight_str`.
#[derive(Debug, Default, Clone)]
pub struct HighlightOptions {
    /// Path to a custom highlight query file. If `None`, the embedded
    /// default queries are used.
    pub query_override: Option<PathBuf>,
    pub target: HighlightTarget,
}

const HIGHLIGHT_NAMES: &[&str] = &[
    "attribute",
    "attribute.declaration",
    "boolean",
    "comment",
    "function",
    "keyword",
    "keyword.conditional",
    "keyword.conditional.ternary",
    "keyword.directive",
    "keyword.operator",
    "keyword.repeat",
    "keyword.return",
    "number",
    "number.float",
    "operator",
    "punctuation",
    "punctuation.bracket",
    "punctuation.delimiter",
    "punctuation.documentation",
    "punctuation.special",
    "string",
    "string.special",
    "type",
    "variable",
    "variable.builtin",
];

// hardcoded terminal colors - MAYBE: make this user-configurable
fn style_for_highlight(class: &str) -> nu_ansi_term::Style {
    use nu_ansi_term::{Color, Style};
    match class {
        "attribute" => Style::new().fg(Color::Cyan),
        "boolean" => Style::new().fg(Color::Blue),
        "comment" => Style::new().fg(Color::Fixed(245)), // light gray
        "function" | "function.builtin" => Style::new().fg(Color::Blue),
        "keyword" => Style::new().fg(Color::Magenta),
	"number" => Style::new().fg(Color::Yellow),
        "operator" => Style::new().fg(Color::Cyan),
        "string" | "punctuation" | "string.special" => Style::new().fg(Color::Green),
        "type" => Style::new().fg(Color::Yellow),
        "variable" => Style::new(),
        "variable.builtin" => Style::new().fg(Color::Cyan),
        _ => Style::new(),
    }
}

pub fn highlight_str(source: &str, opts: &HighlightOptions) -> Result<String> {
    let magma_language = tree_sitter_magma::LANGUAGE.into();
    let mut highlighter = Highlighter::new();

    let (highlight_query, locals_query): (String, String) = match &opts.query_override {
        Some(path) => {
            let hq = std::fs::read_to_string(path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            // For a single file we treat it as the highlight query and use
            // an empty locals/injections query.
            (hq, String::new())
        }
        None => (
            include_str!("../queries/highlights.scm").to_string(),
            include_str!("../queries/locals.scm").to_string(),
        ),
    };

    let mut magma_config = HighlightConfiguration::new(
        magma_language,
        "magma",
        &highlight_query,
        "",
        &locals_query,
    ).unwrap();
    magma_config.configure(HIGHLIGHT_NAMES);

    let highlights = highlighter.highlight(
        &magma_config,
        source.as_bytes(),
        None,
        |_| None
    ).unwrap();

    match opts.target {
        HighlightTarget::Html => render_html(source, highlights),
        HighlightTarget::Terminal => render_terminal(source, highlights),
    }
}

fn render_html<'a>(
    source: &str,
    highlights: impl Iterator<Item = std::result::Result<HighlightEvent, tree_sitter_highlight::Error>>,
) -> Result<String> {
    let mut renderer = HtmlRenderer::new();
    renderer.render(
        highlights,
        source.as_bytes(),
        &|highlight, output| {
            let name = HIGHLIGHT_NAMES[highlight.0];
            write!(output, r#"class="{}""#, name).unwrap();
        },
    ).unwrap();

    let html = renderer.lines().collect::<Vec<_>>().join("");
    Ok(html)
}

fn render_terminal<'a>(
    source: &str,
    highlights: impl Iterator<Item = std::result::Result<HighlightEvent, tree_sitter_highlight::Error>>,
) -> Result<String> {
    let mut output = String::new();
    let mut stack: Vec<nu_ansi_term::Style> = Vec::new();

    for event in highlights {
        match event.unwrap() {
            HighlightEvent::Source {start, end} => {
                let text = &source[start..end];
                match stack.last() {
                    Some(style) => output.push_str(&style.paint(text).to_string()),
                    None => output.push_str(text),
                }
            },
            HighlightEvent::HighlightStart(s) => {
                let class = HIGHLIGHT_NAMES[s.0];
                stack.push(style_for_highlight(class));
            },
            HighlightEvent::HighlightEnd => {
                stack.pop();
            },
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highlight_function_html() {
        let input = "function foo(x) return x +1; end function;";
        let opts = HighlightOptions {
            target: HighlightTarget::Html,
            ..Default::default()
        };
        let result = highlight_str(input, &opts).unwrap();
        assert!(result.contains("function"));
        assert!(result.contains("foo"));
        assert!(result.contains("<span class=\""));
        assert!(result.contains("</span>"));
    }

    #[test]
    fn test_highlight_function_terminal() {
        let input = "function foo(x) return x +1; end function;";
        let opts = HighlightOptions {
            target: HighlightTarget::Terminal,
            ..Default::default()
        };
        let result = highlight_str(input, &opts).unwrap();
        assert!(result.contains("function"));
        assert!(result.contains("foo"));
        // ANSI escape codes should be present for highlighted spans.
        assert!(result.contains('\x1b'));
    }
}
