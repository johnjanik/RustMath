//! RustMath Jupyter Kernel CLI
//!
//! Command-line interface for starting, installing, and managing the kernel.

use clap::{Parser, Subcommand};
use rustmath_jupyter::kernel::RustMathKernel;
use rustmath_jupyter::install;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "rustmath-kernel")]
#[command(author = "RustMath Contributors")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "RustMath Jupyter Kernel - Fast symbolic mathematics in Rust")]
#[command(long_about = r#"
RustMath Jupyter Kernel

A native Rust implementation of the Jupyter kernel protocol for RustMath,
providing fast symbolic mathematics computation in Jupyter notebooks.

EXAMPLES:
    # Install the kernel for the current user
    rustmath-kernel install

    # Start the kernel with a connection file (called by Jupyter)
    rustmath-kernel start -f /path/to/connection.json

    # Check installation status
    rustmath-kernel status

    # Uninstall the kernel
    rustmath-kernel uninstall
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the kernel with a connection file
    Start {
        /// Path to the Jupyter connection file
        #[arg(short = 'f', long = "connection-file")]
        connection_file: PathBuf,
    },

    /// Install the kernel specification
    Install {
        /// Install system-wide (requires root)
        #[arg(long)]
        system: bool,
    },

    /// Uninstall the kernel specification
    Uninstall {
        /// Uninstall system-wide installation
        #[arg(long)]
        system: bool,
    },

    /// Show installation status
    Status,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Start { connection_file } => {
            eprintln!("Starting RustMath kernel with connection file: {}", connection_file.display());

            let mut kernel = RustMathKernel::from_connection_file(&connection_file).await?;
            kernel.run().await?;
        }

        Commands::Install { system } => {
            println!("Installing RustMath Jupyter kernel...");
            install::install_kernel(!system)?;
            println!();
            println!("Installation complete!");
            println!();
            println!("You can now select 'RustMath' as a kernel in Jupyter Lab.");
            println!("Try running: jupyter lab");
        }

        Commands::Uninstall { system } => {
            println!("Uninstalling RustMath Jupyter kernel...");
            install::uninstall_kernel(!system)?;
        }

        Commands::Status => {
            install::print_status();
        }
    }

    Ok(())
}
