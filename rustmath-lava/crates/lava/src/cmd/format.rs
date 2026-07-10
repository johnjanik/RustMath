use crate::cli::FormatArgs;
use anyhow::{Context, Result, anyhow};
use lava_core::{FormatOptions, format_str};
use std::io::{Read, Write};
use std::path::Path;

pub async fn run(args: FormatArgs) -> Result<i32> {
    let opts = FormatOptions {
        query_override: args.query.clone(),
        cwd: None,
        tolerate_parse_errors: args.tolerate_parse_errors,
        skip_idempotence: false,
    };

    let expanded_paths: Vec<std::path::PathBuf> = if args.recursive {
        expand_recursive(&args.paths)?
    } else {
        args.paths.clone()
    };

    if args.check {
        if expanded_paths.is_empty() {
            anyhow::bail!("--check requires at least one path");
        }
        let mut changed: Vec<&Path> = Vec::new();
        for path in &expanded_paths {
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let formatted = format_one(&source, &path.display().to_string(), &opts)?;
            if formatted != source {
                changed.push(path);
                if args.diff {
                    let diff = similar::TextDiff::from_lines(&source, &formatted);
                    eprintln!("--- {}\n+++ {} (formatted)", path.display(), path.display());
                    for hunk in diff.unified_diff().context_radius(3).iter_hunks() {
                        eprint!("{hunk}");
                    }
                }
            }
        }
        if changed.is_empty() {
            return Ok(0);
        }
        for p in &changed {
            eprintln!("{} would be reformatted", p.display());
        }
        return Ok(1);
    }

    // Reading from stdin: zero paths, or a single `-`.
    let read_stdin = expanded_paths.is_empty()
        || (expanded_paths.len() == 1 && expanded_paths[0].as_os_str() == "-");
    if read_stdin {
        if args.write {
            anyhow::bail!("-w cannot be used with stdin");
        }
        let mut source = String::new();
        std::io::stdin()
            .read_to_string(&mut source)
            .context("reading stdin")?;
        let formatted = format_one(&source, "<stdin>", &opts)?;
        std::io::stdout()
            .write_all(formatted.as_bytes())
            .context("writing stdout")?;
        return Ok(0);
    }

    if !args.write && expanded_paths.len() > 1 {
        return Err(anyhow!(
            "refusing to concatenate multiple files to stdout; pass -w or one path at a time"
        ));
    }

    if args.write {
        for path in &expanded_paths {
            let source = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let formatted = format_one(&source, &path.display().to_string(), &opts)?;
            atomic_write(path, &formatted)
                .with_context(|| format!("writing {}", path.display()))?;
        }
        return Ok(0);
    }

    // Single file → stdout.
    let path = &expanded_paths[0];
    let source =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let formatted = format_one(&source, &path.display().to_string(), &opts)?;
    std::io::stdout()
        .write_all(formatted.as_bytes())
        .context("writing stdout")?;
    Ok(0)
}

fn format_one(source: &str, label: &str, opts: &FormatOptions) -> Result<String> {
    let (formatted, src) = format_str(source, opts)
        .map_err(|e| anyhow!("{}", e))
        .with_context(|| format!("formatting {label}"))?;
    tracing::debug!(?src, "format_one done");
    Ok(formatted)
}

fn atomic_write(target: &Path, contents: &str) -> std::io::Result<()> {
    let parent = target.parent().unwrap_or(Path::new("."));
    let mut tmp = tempfile::Builder::new()
        .prefix(".lava-")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    tmp.write_all(contents.as_bytes())?;
    tmp.flush()?;
    tmp.persist(target).map_err(|e| e.error)?;
    Ok(())
}

fn expand_recursive(roots: &[std::path::PathBuf]) -> Result<Vec<std::path::PathBuf>> {
    use ignore::WalkBuilder;
    if roots.is_empty() {
        anyhow::bail!("--recursive requires at least one path");
    }
    let mut out: Vec<std::path::PathBuf> = Vec::new();
    for root in roots {
        if root.is_file() {
            out.push(root.clone());
            continue;
        }
        let mut wb = WalkBuilder::new(root);
        wb.require_git(false);
        wb.add_custom_ignore_filename(".lavaignore");
        for result in wb.build() {
            let entry = result.with_context(|| format!("walking {}", root.display()))?;
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let p = entry.path();
            let is_magma = p
                .extension()
                .map(|e| e == "m" || e == "magma")
                .unwrap_or(false);
            if is_magma {
                out.push(p.to_path_buf());
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}
