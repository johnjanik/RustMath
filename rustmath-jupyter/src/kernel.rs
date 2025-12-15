//! Main kernel implementation
//!
//! Orchestrates the Jupyter protocol handling and REPL evaluation.

use crate::connection::{ConnectionInfo, KernelSockets};
use crate::handlers::{
    create_idle_status, create_starting_status, handle_complete, handle_execute,
    handle_interrupt, handle_is_complete, handle_kernel_info, handle_shutdown,
    publish_status,
};
use crate::protocol::JupyterMessage;
use crate::repl::ReplContext;
use std::path::Path;
use uuid::Uuid;

/// The RustMath Jupyter Kernel
pub struct RustMathKernel {
    sockets: KernelSockets,
    context: ReplContext,
    session: String,
    running: bool,
}

impl RustMathKernel {
    /// Create a new kernel from a connection file
    pub async fn from_connection_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let connection = ConnectionInfo::from_file(path)?;
        let sockets = KernelSockets::new(connection).await?;
        let session = Uuid::new_v4().to_string();

        Ok(Self {
            sockets,
            context: ReplContext::new(),
            session,
            running: true,
        })
    }

    /// Run the kernel main loop
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Publish starting status
        let starting_msg = create_starting_status(&self.session);
        self.sockets.publish(&starting_msg).await?;

        // Publish idle status to indicate kernel is ready
        let idle_msg = create_idle_status(&self.session);
        self.sockets.publish(&idle_msg).await?;

        eprintln!("RustMath kernel started");

        while self.running {
            // Handle messages sequentially to avoid borrow issues
            // In a production kernel, we'd use channels or split the socket struct

            // Try to receive from shell socket (main requests)
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                self.sockets.recv_shell()
            ).await {
                Ok(Ok(msg)) => {
                    if let Err(e) = self.handle_shell_message(msg).await {
                        eprintln!("Error handling shell message: {}", e);
                    }
                    continue;
                }
                Ok(Err(e)) => {
                    eprintln!("Error receiving shell message: {}", e);
                }
                Err(_) => {} // Timeout, continue to check other sockets
            }

            // Try to receive from control socket
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                self.sockets.recv_control()
            ).await {
                Ok(Ok(msg)) => {
                    if let Err(e) = self.handle_control_message(msg).await {
                        eprintln!("Error handling control message: {}", e);
                    }
                    continue;
                }
                Ok(Err(e)) => {
                    eprintln!("Error receiving control message: {}", e);
                }
                Err(_) => {} // Timeout
            }

            // Handle heartbeat
            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                self.sockets.handle_heartbeat()
            ).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    eprintln!("Heartbeat error: {}", e);
                }
                Err(_) => {} // Timeout
            }
        }

        eprintln!("RustMath kernel shutting down");
        Ok(())
    }

    async fn handle_shell_message(&mut self, msg: JupyterMessage) -> Result<(), Box<dyn std::error::Error>> {
        let msg_type = &msg.header.msg_type;
        eprintln!("Received shell message: {}", msg_type);

        match msg_type.as_str() {
            "kernel_info_request" => {
                publish_status(&mut self.sockets, &msg, "busy").await?;
                let reply = handle_kernel_info(&msg);
                self.sockets.send_shell(&reply).await?;
                publish_status(&mut self.sockets, &msg, "idle").await?;
            }

            "execute_request" => {
                let reply = handle_execute(&msg, &mut self.context, &mut self.sockets).await?;
                self.sockets.send_shell(&reply).await?;
            }

            "complete_request" => {
                let reply = handle_complete(&msg, &self.context);
                self.sockets.send_shell(&reply).await?;
            }

            "is_complete_request" => {
                let reply = handle_is_complete(&msg);
                self.sockets.send_shell(&reply).await?;
            }

            _ => {
                eprintln!("Unknown shell message type: {}", msg_type);
            }
        }

        Ok(())
    }

    async fn handle_control_message(&mut self, msg: JupyterMessage) -> Result<(), Box<dyn std::error::Error>> {
        let msg_type = &msg.header.msg_type;
        eprintln!("Received control message: {}", msg_type);

        match msg_type.as_str() {
            "kernel_info_request" => {
                publish_status(&mut self.sockets, &msg, "busy").await?;
                let reply = handle_kernel_info(&msg);
                self.sockets.send_control(&reply).await?;
                publish_status(&mut self.sockets, &msg, "idle").await?;
            }

            "shutdown_request" => {
                let (reply, restart) = handle_shutdown(&msg);
                self.sockets.send_control(&reply).await?;
                self.running = false;

                if restart {
                    eprintln!("Kernel restart requested");
                }
            }

            "interrupt_request" => {
                let reply = handle_interrupt(&msg);
                self.sockets.send_control(&reply).await?;
            }

            _ => {
                eprintln!("Unknown control message type: {}", msg_type);
            }
        }

        Ok(())
    }
}
