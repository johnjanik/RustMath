//! Query string resolution: explicit override → CWD walk-up → XDG → embedded.

use crate::error::QuerySource;
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Embedded canonical Magma query file.
pub const EMBEDDED: &str = include_str!("../queries/topiary.scm");

/// Resolve which query string to use, plus its source for diagnostics.
///
/// Resolution order:
///   1. `override_path` (if `Some`)
///   2. `./.lava/magma.scm` walking up to the filesystem root
///   3. `$XDG_CONFIG_HOME/lava/magma.scm` (or the `dirs::config_dir()` fallback)
///   4. embedded fallback
pub fn resolve(
    cwd: &Path,
    override_path: Option<&Path>,
) -> std::io::Result<(Cow<'static, str>, QuerySource)> {

    if let Some(p) = override_path {
        let bytes = std::fs::read_to_string(p)?;
        tracing::debug!(path = %p.display(), "loaded query from --query");
        return Ok((Cow::Owned(bytes), QuerySource::Path(p.to_path_buf())));
    }

    if let Some(found) = walk_up(cwd, ".lava/magma.scm") {
        let bytes = std::fs::read_to_string(&found)?;
        tracing::debug!(path = %found.display(), "loaded query from CWD walk-up");
        return Ok((Cow::Owned(bytes), QuerySource::Path(found)));
    }

    if let Some(config) = dirs::config_dir() {
        let candidate = config.join("lava").join("magma.scm");
        if candidate.is_file() {
            let bytes = std::fs::read_to_string(&candidate)?;
            tracing::debug!(path = %candidate.display(), "loaded query from XDG");
            return Ok((Cow::Owned(bytes), QuerySource::Path(candidate)));
        }
    }
    tracing::debug!("using embedded query");
    Ok((Cow::Borrowed(EMBEDDED), QuerySource::Embedded))
}

fn walk_up(start: &Path, rel: &str) -> Option<PathBuf> {
    let mut dir: Option<&Path> = Some(start);
    while let Some(d) = dir {
        let candidate = d.join(rel);
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn falls_back_to_embedded_when_nothing_found() {
        let tmp = TempDir::new().unwrap();
        let (q, src) = resolve(tmp.path(), None).unwrap();
        assert_eq!(q, EMBEDDED);
        assert!(matches!(src, QuerySource::Embedded));
    }

    #[test]
    fn explicit_override_wins() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("custom.scm");
        std::fs::write(&p, "; custom override\n").unwrap();
        let (q, src) = resolve(tmp.path(), Some(&p)).unwrap();
        assert_eq!(&*q, "; custom override\n");
        assert!(matches!(src, QuerySource::Path(_)));
    }

    #[test]
    fn cwd_walk_up_finds_dot_lava() {
        let tmp = TempDir::new().unwrap();
        let dot_lava = tmp.path().join(".lava");
        std::fs::create_dir(&dot_lava).unwrap();
        std::fs::write(dot_lava.join("magma.scm"), "; from cwd\n").unwrap();
        let nested = tmp.path().join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        let (q, src) = resolve(&nested, None).unwrap();
        assert_eq!(&*q, "; from cwd\n");
        assert!(matches!(src, QuerySource::Path(_)));
    }
}
