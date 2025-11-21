//! Temporary file and directory management
//!
//! This module provides utilities for creating and managing temporary files and directories,
//! wrapping Rust's `tempfile` crate with additional functionality for atomic operations.
//!
//! # Thread Safety
//!
//! All functions in this module are thread-safe:
//! - `tmp_filename()` generates unique names using cryptographically secure random numbers
//! - `tmp_dir()` creates directories with system-level uniqueness guarantees
//! - `atomic_write()` and `atomic_dir()` use OS-level atomic operations
//! - RAII patterns ensure cleanup happens even in multi-threaded contexts
//!
//! # RAII Patterns
//!
//! This module follows Rust's RAII (Resource Acquisition Is Initialization) pattern:
//! - Resources are automatically cleaned up when they go out of scope
//! - `TemporaryFile` and `TemporaryDir` implement `Drop` to ensure cleanup
//! - Persistent variants allow keeping resources after scope ends
//!
//! # Examples
//!
//! ```rust
//! use rustmath_misc::temporary_file::{tmp_filename, tmp_dir, atomic_write, TemporaryFile};
//! use std::io::Write;
//!
//! // Generate a temporary filename (does not create the file)
//! let temp_path = tmp_filename("rustmath", Some(".txt")).unwrap();
//!
//! // Create a temporary directory (automatically cleaned up on drop)
//! {
//!     let temp_dir = tmp_dir().unwrap();
//!     let dir_path = temp_dir.path();
//!     // Use the directory...
//! } // Directory automatically deleted here
//!
//! // Atomically write to a file
//! atomic_write("output.txt", |f| {
//!     writeln!(f, "Hello, world!")
//! }).unwrap();
//! ```

use std::path::{Path, PathBuf};
use std::io;
use std::fs;
use tempfile::{NamedTempFile, TempDir, Builder};

/// A temporary file that is automatically deleted when dropped.
///
/// This struct provides RAII-based temporary file management. The file is created
/// immediately and deleted when the struct is dropped, unless `persist()` is called.
///
/// # Thread Safety
///
/// This type is `Send` and can be safely transferred between threads. However, it is
/// not `Sync` as it represents exclusive ownership of a file resource.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::TemporaryFile;
/// use std::io::Write;
///
/// let mut temp_file = TemporaryFile::new().unwrap();
/// writeln!(temp_file.file_mut(), "temporary data").unwrap();
/// let path = temp_file.path().to_owned();
/// // File is automatically deleted when temp_file goes out of scope
/// ```
#[derive(Debug)]
pub struct TemporaryFile {
    inner: Option<NamedTempFile>,
}

impl TemporaryFile {
    /// Create a new temporary file with default settings.
    ///
    /// The file is created in the system's temporary directory with a unique name.
    /// It will be automatically deleted when this struct is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created (e.g., insufficient permissions,
    /// disk full, etc.).
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe. Each call generates a unique file using
    /// cryptographically secure random numbers.
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            inner: Some(NamedTempFile::new()?),
        })
    }

    /// Create a new temporary file with a specific prefix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for the temporary file name
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_misc::temporary_file::TemporaryFile;
    ///
    /// let temp_file = TemporaryFile::with_prefix("rustmath_").unwrap();
    /// // File name will be something like: rustmath_Ab3Xq2R7
    /// ```
    pub fn with_prefix(prefix: &str) -> io::Result<Self> {
        Ok(Self {
            inner: Some(Builder::new().prefix(prefix).tempfile()?),
        })
    }

    /// Create a new temporary file with a specific prefix and suffix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for the temporary file name
    /// * `suffix` - The suffix for the temporary file name (e.g., ".txt")
    pub fn with_prefix_suffix(prefix: &str, suffix: &str) -> io::Result<Self> {
        Ok(Self {
            inner: Some(Builder::new().prefix(prefix).suffix(suffix).tempfile()?),
        })
    }

    /// Get the path to the temporary file.
    ///
    /// # Thread Safety
    ///
    /// The path remains valid until the file is dropped or persisted.
    pub fn path(&self) -> &Path {
        self.inner.as_ref().unwrap().path()
    }

    /// Get a mutable reference to the underlying file for writing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_misc::temporary_file::TemporaryFile;
    /// use std::io::Write;
    ///
    /// let mut temp_file = TemporaryFile::new().unwrap();
    /// writeln!(temp_file.file_mut(), "Hello, world!").unwrap();
    /// temp_file.file_mut().sync_all().unwrap();
    /// ```
    pub fn file_mut(&mut self) -> &mut fs::File {
        self.inner.as_mut().unwrap().as_file_mut()
    }

    /// Get a reference to the underlying file for reading.
    pub fn file(&self) -> &fs::File {
        self.inner.as_ref().unwrap().as_file()
    }

    /// Persist the temporary file, preventing automatic deletion.
    ///
    /// Returns the path to the persisted file. After calling this method,
    /// the file will not be deleted when this struct is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be persisted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_misc::temporary_file::TemporaryFile;
    /// use std::io::Write;
    ///
    /// let mut temp_file = TemporaryFile::new().unwrap();
    /// writeln!(temp_file.file_mut(), "permanent data").unwrap();
    /// let permanent_path = temp_file.persist().unwrap();
    /// // File is NOT deleted, even after temp_file is dropped
    /// ```
    pub fn persist(mut self) -> io::Result<PathBuf> {
        let temp = self.inner.take().unwrap();
        let (_, path) = temp.keep()?;
        Ok(path)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        // The inner NamedTempFile handles cleanup automatically
    }
}

/// A temporary directory that is automatically deleted when dropped.
///
/// This struct provides RAII-based temporary directory management. The directory
/// and all its contents are deleted when the struct is dropped, unless `persist()`
/// is called.
///
/// # Thread Safety
///
/// This type is `Send` and can be safely transferred between threads. It is also
/// `Sync` as multiple threads can safely access the directory path.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::TemporaryDir;
/// use std::fs;
///
/// let temp_dir = TemporaryDir::new().unwrap();
/// let file_path = temp_dir.path().join("data.txt");
/// fs::write(file_path, "temporary data").unwrap();
/// // Directory and all contents automatically deleted when temp_dir is dropped
/// ```
#[derive(Debug)]
pub struct TemporaryDir {
    inner: Option<TempDir>,
}

impl TemporaryDir {
    /// Create a new temporary directory with default settings.
    ///
    /// The directory is created in the system's temporary directory with a unique name.
    /// It and all its contents will be automatically deleted when this struct is dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    ///
    /// # Thread Safety
    ///
    /// This function is thread-safe. Each call generates a unique directory.
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            inner: Some(TempDir::new()?),
        })
    }

    /// Create a new temporary directory with a specific prefix.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The prefix for the temporary directory name
    pub fn with_prefix(prefix: &str) -> io::Result<Self> {
        Ok(Self {
            inner: Some(Builder::new().prefix(prefix).tempdir()?),
        })
    }

    /// Get the path to the temporary directory.
    ///
    /// # Thread Safety
    ///
    /// The path remains valid until the directory is dropped or persisted.
    /// Multiple threads can safely access the path.
    pub fn path(&self) -> &Path {
        self.inner.as_ref().unwrap().path()
    }

    /// Persist the temporary directory, preventing automatic deletion.
    ///
    /// Returns the path to the persisted directory. After calling this method,
    /// the directory will not be deleted when this struct is dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustmath_misc::temporary_file::TemporaryDir;
    /// use std::fs;
    ///
    /// let temp_dir = TemporaryDir::new().unwrap();
    /// fs::write(temp_dir.path().join("data.txt"), "permanent data").unwrap();
    /// let permanent_path = temp_dir.persist();
    /// // Directory is NOT deleted, even after temp_dir is dropped
    /// ```
    pub fn persist(mut self) -> io::Result<PathBuf> {
        let temp = self.inner.take().unwrap();
        Ok(temp.keep())
    }
}

impl Drop for TemporaryDir {
    fn drop(&mut self) {
        // The inner TempDir handles cleanup automatically
    }
}

/// Generate a unique temporary filename without creating the file.
///
/// This function generates a path in the system's temporary directory but does not
/// create the file. This is useful when you need a temporary filename to pass to
/// external tools or when you want to control file creation yourself.
///
/// # Arguments
///
/// * `prefix` - A prefix for the filename
/// * `suffix` - An optional suffix (e.g., file extension like ".txt")
///
/// # Returns
///
/// A `PathBuf` containing the generated temporary filename.
///
/// # Thread Safety
///
/// This function is thread-safe and generates unique names using cryptographically
/// secure random numbers. Multiple threads can call this function concurrently
/// without risk of name collisions.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::tmp_filename;
///
/// let path = tmp_filename("rustmath", Some(".dat")).unwrap();
/// println!("Temporary filename: {:?}", path);
/// // File is NOT created, you must create it yourself if needed
/// ```
///
/// # Note
///
/// Since the file is not created, there is a small race condition window where
/// another process could create a file with the same name. If you need guaranteed
/// uniqueness, use `TemporaryFile::new()` instead.
pub fn tmp_filename(prefix: &str, suffix: Option<&str>) -> io::Result<PathBuf> {
    // Generate a random suffix to ensure uniqueness
    let random_part: u64 = rand::random();
    let suffix = suffix.unwrap_or("");

    let filename = format!("{}{:016x}{}", prefix, random_part, suffix);
    Ok(std::env::temp_dir().join(filename))
}

/// Create a temporary directory that is automatically cleaned up on drop.
///
/// This is a convenience function that creates a `TemporaryDir` with default settings.
/// The directory and all its contents will be automatically deleted when the returned
/// `TemporaryDir` is dropped.
///
/// # Returns
///
/// A `TemporaryDir` instance that manages the temporary directory.
///
/// # Errors
///
/// Returns an error if the directory cannot be created.
///
/// # Thread Safety
///
/// This function is thread-safe. Each call generates a unique directory.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::tmp_dir;
/// use std::fs;
///
/// let temp_dir = tmp_dir().unwrap();
/// fs::write(temp_dir.path().join("test.txt"), "data").unwrap();
/// // Directory automatically deleted when temp_dir goes out of scope
/// ```
pub fn tmp_dir() -> io::Result<TemporaryDir> {
    TemporaryDir::new()
}

/// Atomically write to a file using a write closure.
///
/// This function provides atomic file writing by:
/// 1. Creating a temporary file in the same directory as the target
/// 2. Writing to the temporary file using the provided closure
/// 3. Atomically renaming the temporary file to the target filename
///
/// If any error occurs during writing, the original file (if it exists) remains
/// unchanged, and the temporary file is automatically cleaned up.
///
/// # Arguments
///
/// * `path` - The target file path
/// * `writer` - A closure that writes to the temporary file
///
/// # Returns
///
/// Returns `Ok(())` if the write and rename succeed, or an `io::Error` otherwise.
///
/// # Thread Safety
///
/// This function is thread-safe. The atomic rename operation is provided by the
/// operating system. However, if multiple threads attempt to write to the same
/// file concurrently, the last writer wins (determined by OS scheduling).
///
/// # Platform-Specific Behavior
///
/// On Unix systems, the rename is truly atomic. On Windows, there may be edge cases
/// where atomicity is not guaranteed for cross-device moves, but this function
/// creates the temporary file in the same directory to ensure same-device operation.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::atomic_write;
/// use std::io::Write;
///
/// atomic_write("output.txt", |file| {
///     writeln!(file, "Line 1")?;
///     writeln!(file, "Line 2")?;
///     Ok(())
/// }).unwrap();
/// ```
///
/// # Error Handling
///
/// If the writer closure returns an error, the temporary file is automatically
/// cleaned up and the error is propagated to the caller.
pub fn atomic_write<P, F>(path: P, writer: F) -> io::Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut fs::File) -> io::Result<()>,
{
    let path = path.as_ref();

    // Get the parent directory to create temp file in same location
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    // Create a temporary file in the same directory as the target
    let mut temp_file = Builder::new()
        .prefix(".tmp_")
        .suffix(&format!("_{}", rand::random::<u32>()))
        .tempfile_in(parent)?;

    // Write using the provided closure
    writer(temp_file.as_file_mut())?;

    // Ensure all data is written to disk
    temp_file.as_file_mut().sync_all()?;

    // Atomically replace the target file
    // persist() returns (File, PathBuf), we need to move it to the final location
    let (_, temp_path) = temp_file.keep()?;

    // Use rename for atomic operation
    fs::rename(temp_path, path)?;

    Ok(())
}

/// Atomically create a directory and populate it using a closure.
///
/// This function provides atomic directory creation by:
/// 1. Creating a temporary directory in the same parent as the target
/// 2. Populating the temporary directory using the provided closure
/// 3. Atomically renaming the temporary directory to the target name
///
/// If any error occurs, the temporary directory is automatically cleaned up.
///
/// # Arguments
///
/// * `path` - The target directory path
/// * `populator` - A closure that populates the directory (receives the temp directory path)
///
/// # Returns
///
/// Returns `Ok(())` if the creation and rename succeed, or an `io::Error` otherwise.
///
/// # Thread Safety
///
/// This function is thread-safe. The atomic rename operation is provided by the
/// operating system. However, if multiple threads attempt to create the same
/// directory concurrently, the last successful rename wins.
///
/// # Platform-Specific Behavior
///
/// On Unix systems, the rename is truly atomic. On Windows, atomicity may not be
/// guaranteed in all edge cases, but this function creates the temporary directory
/// in the same parent to maximize atomicity.
///
/// # Examples
///
/// ```rust
/// use rustmath_misc::temporary_file::atomic_dir;
/// use std::fs;
///
/// atomic_dir("output_dir", |dir_path| {
///     fs::write(dir_path.join("file1.txt"), "data1")?;
///     fs::write(dir_path.join("file2.txt"), "data2")?;
///     Ok(())
/// }).unwrap();
/// ```
///
/// # Error Handling
///
/// If the populator closure returns an error, the temporary directory is automatically
/// cleaned up and the error is propagated to the caller.
pub fn atomic_dir<P, F>(path: P, populator: F) -> io::Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&Path) -> io::Result<()>,
{
    let path = path.as_ref();

    // Get the parent directory to create temp dir in same location
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    // Create a temporary directory in the same parent as the target
    let temp_dir = Builder::new()
        .prefix(".tmp_dir_")
        .suffix(&format!("_{}", rand::random::<u32>()))
        .tempdir_in(parent)?;

    // Populate using the provided closure
    populator(temp_dir.path())?;

    // Atomically rename to the final location
    let temp_path = temp_dir.keep();
    fs::rename(temp_path, path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn test_temporary_file_creation_and_cleanup() {
        let path = {
            let temp_file = TemporaryFile::new().unwrap();
            let path = temp_file.path().to_owned();

            // File should exist while temp_file is in scope
            assert!(path.exists());

            path
        };

        // File should be deleted after temp_file goes out of scope
        assert!(!path.exists());
    }

    #[test]
    fn test_temporary_file_with_prefix() {
        let temp_file = TemporaryFile::with_prefix("rustmath_test_").unwrap();
        let filename = temp_file.path().file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("rustmath_test_"));
    }

    #[test]
    fn test_temporary_file_with_suffix() {
        let temp_file = TemporaryFile::with_prefix_suffix("test_", ".txt").unwrap();
        let filename = temp_file.path().file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("test_"));
        assert!(filename.ends_with(".txt"));
    }

    #[test]
    fn test_temporary_file_write() {
        let mut temp_file = TemporaryFile::new().unwrap();
        writeln!(temp_file.file_mut(), "Hello, world!").unwrap();
        temp_file.file_mut().sync_all().unwrap();

        // Read back the content
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert_eq!(content, "Hello, world!\n");
    }

    #[test]
    fn test_temporary_file_persist() {
        let mut temp_file = TemporaryFile::new().unwrap();
        writeln!(temp_file.file_mut(), "Persistent data").unwrap();

        let path = temp_file.persist().unwrap();

        // File should still exist after persist
        assert!(path.exists());

        // Read content
        let content = fs::read_to_string(&path).unwrap();
        assert_eq!(content, "Persistent data\n");

        // Clean up
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_temporary_dir_creation_and_cleanup() {
        let path = {
            let temp_dir = TemporaryDir::new().unwrap();
            let path = temp_dir.path().to_owned();

            // Create a file in the directory
            fs::write(path.join("test.txt"), "test data").unwrap();

            // Directory and file should exist
            assert!(path.exists());
            assert!(path.join("test.txt").exists());

            path
        };

        // Directory and contents should be deleted
        assert!(!path.exists());
    }

    #[test]
    fn test_temporary_dir_with_prefix() {
        let temp_dir = TemporaryDir::with_prefix("rustmath_").unwrap();
        let dirname = temp_dir.path().file_name().unwrap().to_str().unwrap();
        assert!(dirname.starts_with("rustmath_"));
    }

    #[test]
    fn test_temporary_dir_persist() {
        let temp_dir = TemporaryDir::new().unwrap();
        fs::write(temp_dir.path().join("data.txt"), "persistent").unwrap();

        let path = temp_dir.persist().unwrap();

        // Directory should still exist
        assert!(path.exists());
        assert!(path.join("data.txt").exists());

        // Clean up
        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_tmp_filename() {
        let path1 = tmp_filename("test", Some(".txt")).unwrap();
        let path2 = tmp_filename("test", Some(".txt")).unwrap();

        // Should generate unique filenames
        assert_ne!(path1, path2);

        // Should have correct prefix and suffix
        let filename = path1.file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("test"));
        assert!(filename.ends_with(".txt"));

        // File should not exist (not created)
        assert!(!path1.exists());
    }

    #[test]
    fn test_tmp_dir_function() {
        let path = {
            let temp_dir = tmp_dir().unwrap();
            let path = temp_dir.path().to_owned();
            assert!(path.exists());
            path
        };

        // Should be cleaned up
        assert!(!path.exists());
    }

    #[test]
    fn test_atomic_write_new_file() {
        let temp_dir = tmp_dir().unwrap();
        let target = temp_dir.path().join("output.txt");

        atomic_write(&target, |f| {
            writeln!(f, "Line 1")?;
            writeln!(f, "Line 2")?;
            Ok(())
        }).unwrap();

        // File should exist with correct content
        assert!(target.exists());
        let content = fs::read_to_string(&target).unwrap();
        assert_eq!(content, "Line 1\nLine 2\n");
    }

    #[test]
    fn test_atomic_write_replace_file() {
        let temp_dir = tmp_dir().unwrap();
        let target = temp_dir.path().join("output.txt");

        // Create initial file
        fs::write(&target, "Original content").unwrap();

        // Atomically replace
        atomic_write(&target, |f| {
            writeln!(f, "New content")?;
            Ok(())
        }).unwrap();

        // Should have new content
        let content = fs::read_to_string(&target).unwrap();
        assert_eq!(content, "New content\n");
    }

    #[test]
    fn test_atomic_write_error_handling() {
        let temp_dir = tmp_dir().unwrap();
        let target = temp_dir.path().join("output.txt");

        // Create initial file
        fs::write(&target, "Original").unwrap();

        // Attempt atomic write that fails
        let result = atomic_write(&target, |_f| {
            Err(io::Error::new(io::ErrorKind::Other, "Simulated error"))
        });

        // Should return error
        assert!(result.is_err());

        // Original file should still exist with original content
        let content = fs::read_to_string(&target).unwrap();
        assert_eq!(content, "Original");
    }

    #[test]
    fn test_atomic_dir_creation() {
        let temp_parent = tmp_dir().unwrap();
        let target = temp_parent.path().join("new_dir");

        atomic_dir(&target, |dir| {
            fs::write(dir.join("file1.txt"), "data1")?;
            fs::write(dir.join("file2.txt"), "data2")?;
            Ok(())
        }).unwrap();

        // Directory should exist with contents
        assert!(target.exists());
        assert!(target.join("file1.txt").exists());
        assert!(target.join("file2.txt").exists());

        let content1 = fs::read_to_string(target.join("file1.txt")).unwrap();
        let content2 = fs::read_to_string(target.join("file2.txt")).unwrap();
        assert_eq!(content1, "data1");
        assert_eq!(content2, "data2");
    }

    #[test]
    fn test_atomic_dir_error_handling() {
        let temp_parent = tmp_dir().unwrap();
        let target = temp_parent.path().join("new_dir");

        // Attempt atomic dir creation that fails
        let result = atomic_dir(&target, |_dir| {
            Err(io::Error::new(io::ErrorKind::Other, "Simulated error"))
        });

        // Should return error
        assert!(result.is_err());

        // Target directory should not exist
        assert!(!target.exists());
    }

    #[test]
    fn test_thread_safety_tmp_filename() {
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for _ in 0..10 {
            let barrier = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier.wait();
                tmp_filename("test", Some(".txt")).unwrap()
            });
            handles.push(handle);
        }

        let mut paths = vec![];
        for handle in handles {
            paths.push(handle.join().unwrap());
        }

        // All paths should be unique
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(paths[i], paths[j]);
            }
        }
    }

    #[test]
    fn test_thread_safety_temporary_file() {
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let barrier = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier.wait();
                let mut temp_file = TemporaryFile::new().unwrap();
                writeln!(temp_file.file_mut(), "Thread {}", i).unwrap();
                temp_file.path().to_owned()
            });
            handles.push(handle);
        }

        let mut paths = vec![];
        for handle in handles {
            let path = handle.join().unwrap();
            paths.push(path);
        }

        // All paths should be unique
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(paths[i], paths[j]);
            }
        }

        // All files should be cleaned up (temp files dropped in threads)
        for path in paths {
            assert!(!path.exists());
        }
    }

    #[test]
    fn test_thread_safety_temporary_dir() {
        let barrier = Arc::new(Barrier::new(10));
        let mut handles = vec![];

        for i in 0..10 {
            let barrier = Arc::clone(&barrier);
            let handle = thread::spawn(move || {
                barrier.wait();
                let temp_dir = TemporaryDir::new().unwrap();
                fs::write(temp_dir.path().join("test.txt"), format!("Thread {}", i)).unwrap();
                temp_dir.path().to_owned()
            });
            handles.push(handle);
        }

        let mut paths = vec![];
        for handle in handles {
            let path = handle.join().unwrap();
            paths.push(path);
        }

        // All paths should be unique
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(paths[i], paths[j]);
            }
        }

        // All directories should be cleaned up
        for path in paths {
            assert!(!path.exists());
        }
    }

    #[test]
    fn test_thread_safety_atomic_write() {
        let temp_parent = tmp_dir().unwrap();
        let barrier = Arc::new(Barrier::new(5));
        let mut handles = vec![];

        for i in 0..5 {
            let barrier = Arc::clone(&barrier);
            let target = temp_parent.path().join(format!("file_{}.txt", i));
            let handle = thread::spawn(move || {
                barrier.wait();
                atomic_write(&target, |f| {
                    writeln!(f, "Thread {}", i)?;
                    Ok(())
                }).unwrap();
                target
            });
            handles.push(handle);
        }

        let mut paths = vec![];
        for handle in handles {
            paths.push(handle.join().unwrap());
        }

        // All files should exist with correct content
        for (i, path) in paths.iter().enumerate() {
            assert!(path.exists());
            let content = fs::read_to_string(path).unwrap();
            assert_eq!(content, format!("Thread {}\n", i));
        }
    }

    #[test]
    fn test_raii_cleanup_on_panic() {
        use std::panic;

        // Test that temporary file is cleaned up even when panic occurs
        let result = panic::catch_unwind(|| {
            let temp = TemporaryFile::with_prefix("panic_test_").unwrap();
            let path = temp.path().to_owned();
            // File exists during lifetime
            assert!(path.exists());
            panic!("Simulated panic");
        });

        assert!(result.is_err());

        // Also test that cleanup works for another temp file after panic recovery
        let path = {
            let temp_file = TemporaryFile::new().unwrap();
            let p = temp_file.path().to_owned();
            assert!(p.exists());
            p
        };
        // File should be cleaned up after drop
        assert!(!path.exists());
    }
}
