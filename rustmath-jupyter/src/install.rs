//! Kernel installation utilities
//!
//! Handles registering and unregistering the RustMath kernel with Jupyter.

use serde_json::json;
use std::env;
use std::fs;
use std::path::PathBuf;

/// Get the Jupyter kernels directory for the current user
pub fn get_kernel_dir() -> Option<PathBuf> {
    // Try user-specific location first
    if let Some(data_dir) = dirs::data_local_dir() {
        return Some(data_dir.join("jupyter").join("kernels").join("rustmath"));
    }

    // Fallback to home directory
    if let Some(home) = dirs::home_dir() {
        return Some(home.join(".local").join("share").join("jupyter").join("kernels").join("rustmath"));
    }

    None
}

/// Get the path to the kernel executable
pub fn get_kernel_executable() -> PathBuf {
    // Try to find the installed binary first
    if let Ok(exe) = env::current_exe() {
        return exe;
    }

    // Fallback to "rustmath-kernel" in PATH
    PathBuf::from("rustmath-kernel")
}

/// Generate the kernel.json content
pub fn generate_kernel_json(kernel_path: &PathBuf) -> serde_json::Value {
    json!({
        "argv": [
            kernel_path.to_string_lossy(),
            "start",
            "-f",
            "{connection_file}"
        ],
        "display_name": "RustMath",
        "language": "rustmath",
        "metadata": {
            "debugger": false
        }
    })
}

/// Install the kernel
pub fn install_kernel(user: bool) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let kernel_dir = if user {
        get_kernel_dir().ok_or("Could not determine kernel directory")?
    } else {
        // System-wide installation
        PathBuf::from("/usr/local/share/jupyter/kernels/rustmath")
    };

    // Create kernel directory
    fs::create_dir_all(&kernel_dir)?;

    // Get kernel executable path
    let kernel_exe = get_kernel_executable();

    // Generate and write kernel.json
    let kernel_json = generate_kernel_json(&kernel_exe);
    let kernel_json_path = kernel_dir.join("kernel.json");
    fs::write(&kernel_json_path, serde_json::to_string_pretty(&kernel_json)?)?;

    println!("Installed kernel spec to: {}", kernel_dir.display());
    println!("Kernel executable: {}", kernel_exe.display());

    Ok(kernel_dir)
}

/// Uninstall the kernel
pub fn uninstall_kernel(user: bool) -> Result<(), Box<dyn std::error::Error>> {
    let kernel_dir = if user {
        get_kernel_dir().ok_or("Could not determine kernel directory")?
    } else {
        PathBuf::from("/usr/local/share/jupyter/kernels/rustmath")
    };

    if kernel_dir.exists() {
        fs::remove_dir_all(&kernel_dir)?;
        println!("Removed kernel spec from: {}", kernel_dir.display());
    } else {
        println!("Kernel spec not found at: {}", kernel_dir.display());
    }

    Ok(())
}

/// Check if the kernel is installed
pub fn check_installation() -> bool {
    if let Some(kernel_dir) = get_kernel_dir() {
        kernel_dir.join("kernel.json").exists()
    } else {
        false
    }
}

/// Print installation status
pub fn print_status() {
    println!("RustMath Jupyter Kernel Status");
    println!("==============================");

    if let Some(kernel_dir) = get_kernel_dir() {
        let kernel_json = kernel_dir.join("kernel.json");
        if kernel_json.exists() {
            println!("Status: Installed");
            println!("Location: {}", kernel_dir.display());

            if let Ok(content) = fs::read_to_string(&kernel_json) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(argv) = json.get("argv").and_then(|v| v.as_array()) {
                        if let Some(exe) = argv.first().and_then(|v| v.as_str()) {
                            println!("Executable: {}", exe);
                            let exe_path = PathBuf::from(exe);
                            if exe_path.exists() {
                                println!("Executable exists: Yes");
                            } else {
                                println!("Executable exists: No (kernel may not work)");
                            }
                        }
                    }
                }
            }
        } else {
            println!("Status: Not installed");
            println!("Expected location: {}", kernel_dir.display());
        }
    } else {
        println!("Status: Unknown (could not determine kernel directory)");
    }

    println!();
    println!("To install: rustmath-kernel install");
    println!("To uninstall: rustmath-kernel uninstall");
}
