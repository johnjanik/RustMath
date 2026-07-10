use crate::cli::HighlightArgs;
use anyhow::{Context, Result, anyhow};
use lava_core::{HighlightOptions, HighlightTarget, highlight_str};
use std::io::{Read, Write};

pub async fn run(args: HighlightArgs) -> Result<i32> {
    let target = if args.html {
        HighlightTarget::Html
    } else {
        HighlightTarget::Terminal
    };

    let opts = HighlightOptions {
        query_override: args.query.clone(),
        target,
    };

    // Reading from stdin: zero paths, or a single `-`.
    let read_stdin = args.paths.is_empty()
        || (args.paths.len() == 1 && args.paths[0].as_os_str() == "-");
    if read_stdin {
        let mut source = String::new();
        std::io::stdin()
            .read_to_string(&mut source)
            .context("reading stdin")?;
        let highlighted = highlight_one(&source, "<stdin>", &opts)?;
        std::io::stdout()
            .write_all(highlighted.as_bytes())
            .context("writing stdout")?;
        return Ok(0);
    }

    if args.paths.len() > 1 {
        return Err(anyhow!(
            "refusing to highlight multiple files to stdout; pass one path at a time"
        ));
    }

    let path = &args.paths[0];
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let highlighted = highlight_one(&source, &path.display().to_string(), &opts)?;
    std::io::stdout()
        .write_all(highlighted.as_bytes())
        .context("writing stdout")?;
    Ok(0)
}

fn highlight_one(source: &str, label: &str, opts: &HighlightOptions) -> Result<String> {
    highlight_str(source, opts)
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| format!("highlighting {label}"))
}
