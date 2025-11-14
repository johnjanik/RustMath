//! Verbose output control

use std::sync::atomic::{AtomicI32, Ordering};

static VERBOSE_LEVEL: AtomicI32 = AtomicI32::new(0);

/// Set the verbosity level
pub fn set_verbose(level: i32) {
    VERBOSE_LEVEL.store(level, Ordering::Relaxed);
}

/// Get the current verbosity level
pub fn get_verbose() -> i32 {
    VERBOSE_LEVEL.load(Ordering::Relaxed)
}

/// Check if verbose output is enabled
pub fn is_verbose() -> bool {
    get_verbose() > 0
}
