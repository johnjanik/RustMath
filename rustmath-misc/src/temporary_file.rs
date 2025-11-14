//! Temporary file management

use std::path::PathBuf;
use std::io;

/// A temporary file that is automatically deleted
#[derive(Debug)]
pub struct TemporaryFile {
    path: PathBuf,
}

impl TemporaryFile {
    /// Create a new temporary file
    pub fn new() -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!("rustmath_{}", rand::random::<u64>()));
        Ok(Self { path })
    }

    /// Get the path to the temporary file
    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
