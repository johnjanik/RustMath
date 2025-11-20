//! GAP Saved Workspace Management
//!
//! This module provides functionality for managing GAP workspaces, which allow
//! saving and loading GAP session state to disk.
//!
//! # Overview
//!
//! GAP supports saving its current state (all defined variables, loaded packages,
//! etc.) to a workspace file. This can significantly speed up startup time when
//! the same packages and data are needed repeatedly.
//!
//! # Usage
//!
//! ```rust,ignore
//! use rustmath_interfaces::saved_workspace::*;
//!
//! // Get the workspace path
//! let ws_path = workspace()?;
//! println!("Workspace: {}", ws_path.display());
//!
//! // Check when workspace was last updated
//! if let Some(ts) = timestamp()? {
//!     println!("Last modified: {:?}", ts);
//! }
//! ```
//!
//! # Note
//!
//! Since RustMath uses process-based GAP communication rather than libGAP,
//! workspace functionality is more limited. Workspaces are primarily useful
//! for GAP itself, not for the Rust interface state.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Error type for workspace operations
#[derive(Debug)]
pub enum WorkspaceError {
    /// IO error
    Io(io::Error),
    /// Workspace not found
    NotFound,
    /// Invalid workspace path
    InvalidPath(String),
}

impl std::fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkspaceError::Io(e) => write!(f, "IO error: {}", e),
            WorkspaceError::NotFound => write!(f, "Workspace not found"),
            WorkspaceError::InvalidPath(p) => write!(f, "Invalid workspace path: {}", p),
        }
    }
}

impl std::error::Error for WorkspaceError {}

impl From<io::Error> for WorkspaceError {
    fn from(e: io::Error) -> Self {
        WorkspaceError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, WorkspaceError>;

/// Get the path to the GAP workspace file
///
/// This returns the default location for GAP workspace files. The actual
/// location may vary depending on the GAP installation and configuration.
///
/// # Returns
///
/// The path to the workspace file, or an error if it cannot be determined.
///
/// # Example
///
/// ```rust,ignore
/// let ws = workspace()?;
/// println!("Workspace: {}", ws.display());
/// ```
pub fn workspace() -> Result<PathBuf> {
    // Try common workspace locations
    let possible_paths = vec![
        // User-specific workspace
        dirs::cache_dir().map(|d| d.join("gap/workspace")),
        // System-wide workspace
        Some(PathBuf::from("/usr/share/gap/workspace")),
        Some(PathBuf::from("/usr/local/share/gap/workspace")),
        // Home directory
        dirs::home_dir().map(|d| d.join(".gap/workspace")),
    ];

    for path_opt in possible_paths {
        if let Some(path) = path_opt {
            if path.exists() {
                return Ok(path);
            }
        }
    }

    // If no existing workspace found, return default location
    if let Some(cache_dir) = dirs::cache_dir() {
        let workspace_path = cache_dir.join("rustmath/gap_workspace");
        return Ok(workspace_path);
    }

    Err(WorkspaceError::NotFound)
}

/// Get the timestamp of when the workspace was last modified
///
/// This can be used to check if a workspace is up-to-date or needs rebuilding.
///
/// # Returns
///
/// The last modification time of the workspace, or None if the workspace
/// doesn't exist.
///
/// # Example
///
/// ```rust,ignore
/// if let Some(ts) = timestamp()? {
///     println!("Workspace last modified: {:?}", ts);
/// }
/// ```
pub fn timestamp() -> Result<Option<SystemTime>> {
    let ws_path = workspace()?;

    if !ws_path.exists() {
        return Ok(None);
    }

    let metadata = fs::metadata(&ws_path)?;
    let modified = metadata.modified()?;

    Ok(Some(modified))
}

/// Check if a workspace exists
///
/// # Example
///
/// ```rust
/// use rustmath_interfaces::saved_workspace::workspace_exists;
///
/// if workspace_exists() {
///     println!("Workspace is available");
/// }
/// ```
pub fn workspace_exists() -> bool {
    workspace().map(|p| p.exists()).unwrap_or(false)
}

/// Create a directory for the workspace if it doesn't exist
///
/// # Arguments
///
/// * `path` - The path where the workspace should be created
///
/// # Returns
///
/// Ok if successful, or an error if the directory cannot be created
pub fn ensure_workspace_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// Get workspace size in bytes
///
/// # Returns
///
/// The size of the workspace file, or None if it doesn't exist
pub fn workspace_size() -> Result<Option<u64>> {
    let ws_path = workspace()?;

    if !ws_path.exists() {
        return Ok(None);
    }

    let metadata = fs::metadata(&ws_path)?;
    Ok(Some(metadata.len()))
}

/// Delete the workspace file
///
/// This removes the saved workspace, which may be useful for troubleshooting
/// or when upgrading GAP versions.
///
/// # Returns
///
/// Ok if the workspace was deleted or didn't exist, error otherwise
pub fn delete_workspace() -> Result<()> {
    let ws_path = workspace()?;

    if ws_path.exists() {
        fs::remove_file(&ws_path)?;
    }

    Ok(())
}

/// Information about a workspace
#[derive(Debug, Clone)]
pub struct WorkspaceInfo {
    /// Path to the workspace file
    pub path: PathBuf,
    /// Size in bytes
    pub size: Option<u64>,
    /// Last modification time
    pub modified: Option<SystemTime>,
    /// Whether the workspace exists
    pub exists: bool,
}

/// Get information about the current workspace
///
/// # Example
///
/// ```rust,ignore
/// let info = workspace_info()?;
/// println!("Workspace: {}", info.path.display());
/// println!("Size: {} bytes", info.size.unwrap_or(0));
/// println!("Exists: {}", info.exists);
/// ```
pub fn workspace_info() -> Result<WorkspaceInfo> {
    let path = workspace()?;
    let exists = path.exists();

    let size = if exists {
        fs::metadata(&path).ok().map(|m| m.len())
    } else {
        None
    };

    let modified = if exists {
        fs::metadata(&path).ok().and_then(|m| m.modified().ok())
    } else {
        None
    };

    Ok(WorkspaceInfo {
        path,
        size,
        modified,
        exists,
    })
}

/// Save the current GAP state to a workspace file
///
/// Note: This requires GAP to support workspace saving, which depends on
/// how GAP was compiled and configured.
///
/// # Arguments
///
/// * `gap` - The GAP interface to save
/// * `path` - Optional custom path for the workspace
///
/// # Example
///
/// ```rust,ignore
/// use rustmath_interfaces::gap::GapInterface;
/// use rustmath_interfaces::saved_workspace::save_workspace;
///
/// let gap = GapInterface::new()?;
/// // ... do work ...
/// save_workspace(&gap, None)?;
/// ```
pub fn save_workspace(gap: &crate::gap::GapInterface, path: Option<&Path>) -> Result<()> {
    let ws_path = if let Some(p) = path {
        p.to_path_buf()
    } else {
        workspace()?
    };

    ensure_workspace_dir(&ws_path)?;

    // GAP command to save workspace
    let path_str = ws_path.to_string_lossy();
    gap.execute(&format!("SaveWorkspace(\"{}\");", path_str))
        .map_err(|e| WorkspaceError::InvalidPath(format!("Failed to save workspace: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_path() {
        // Should return a path without error
        let result = workspace();
        assert!(result.is_ok());
    }

    #[test]
    fn test_workspace_info() {
        // Should return workspace info
        let result = workspace_info();
        assert!(result.is_ok());

        if let Ok(info) = result {
            println!("Workspace: {}", info.path.display());
            println!("Exists: {}", info.exists);
            if let Some(size) = info.size {
                println!("Size: {} bytes", size);
            }
        }
    }

    #[test]
    fn test_workspace_exists() {
        // Should not panic
        let _exists = workspace_exists();
    }

    #[test]
    fn test_timestamp() {
        // Should not panic
        let result = timestamp();
        assert!(result.is_ok());
    }
}
