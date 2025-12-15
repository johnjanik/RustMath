//! Jupyter connection file parsing and ZMQ socket management
//!
//! Handles the connection file that Jupyter provides to kernels,
//! and manages the five ZMQ sockets required by the protocol.

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::fs;
use std::path::Path;
use zeromq::{PubSocket, RepSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::protocol::{Header, JupyterMessage, DELIMITER};

type HmacSha256 = Hmac<Sha256>;

/// Connection information from Jupyter's connection file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub shell_port: u16,
    pub iopub_port: u16,
    pub stdin_port: u16,
    pub control_port: u16,
    pub hb_port: u16,
    pub ip: String,
    pub key: String,
    pub transport: String,
    pub signature_scheme: String,
    pub kernel_name: String,
}

impl ConnectionInfo {
    /// Load connection info from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let info: ConnectionInfo = serde_json::from_str(&content)?;
        Ok(info)
    }

    /// Build a ZMQ address for a given port
    pub fn address(&self, port: u16) -> String {
        format!("{}://{}:{}", self.transport, self.ip, port)
    }
}

/// Manages all ZMQ sockets for kernel communication
pub struct KernelSockets {
    pub shell: RouterSocket,
    pub iopub: PubSocket,
    pub stdin: RouterSocket,
    pub control: RouterSocket,
    pub heartbeat: RepSocket,
    pub connection: ConnectionInfo,
    hmac_key: Option<Vec<u8>>,
}

impl KernelSockets {
    /// Create and bind all kernel sockets
    pub async fn new(connection: ConnectionInfo) -> Result<Self, Box<dyn std::error::Error>> {
        let mut shell = RouterSocket::new();
        let mut iopub = PubSocket::new();
        let mut stdin = RouterSocket::new();
        let mut control = RouterSocket::new();
        let mut heartbeat = RepSocket::new();

        shell.bind(&connection.address(connection.shell_port)).await?;
        iopub.bind(&connection.address(connection.iopub_port)).await?;
        stdin.bind(&connection.address(connection.stdin_port)).await?;
        control.bind(&connection.address(connection.control_port)).await?;
        heartbeat.bind(&connection.address(connection.hb_port)).await?;

        let hmac_key = if !connection.key.is_empty() {
            Some(connection.key.as_bytes().to_vec())
        } else {
            None
        };

        Ok(Self {
            shell,
            iopub,
            stdin,
            control,
            heartbeat,
            connection,
            hmac_key,
        })
    }

    /// Compute HMAC signature for message parts
    fn compute_signature(&self, header: &[u8], parent: &[u8], metadata: &[u8], content: &[u8]) -> String {
        match &self.hmac_key {
            Some(key) => {
                let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
                mac.update(header);
                mac.update(parent);
                mac.update(metadata);
                mac.update(content);
                hex::encode(mac.finalize().into_bytes())
            }
            None => String::new(),
        }
    }

    /// Verify HMAC signature of received message
    fn verify_signature(&self, signature: &str, header: &[u8], parent: &[u8], metadata: &[u8], content: &[u8]) -> bool {
        if self.hmac_key.is_none() {
            return true;
        }
        let computed = self.compute_signature(header, parent, metadata, content);
        computed == signature
    }

    /// Serialize a JupyterMessage to ZMQ multipart message
    pub fn serialize_message(&self, msg: &JupyterMessage) -> Result<ZmqMessage, Box<dyn std::error::Error>> {
        let header = serde_json::to_vec(&msg.header)?;
        let parent = match &msg.parent_header {
            Some(p) => serde_json::to_vec(p)?,
            None => b"{}".to_vec(),
        };
        let metadata = serde_json::to_vec(&msg.metadata)?;
        let content = serde_json::to_vec(&msg.content)?;

        let signature = self.compute_signature(&header, &parent, &metadata, &content);

        // Build frames in order: identities, delimiter, signature, header, parent, metadata, content
        let mut frames: Vec<bytes::Bytes> = Vec::new();

        // Add identities (required for ROUTER socket replies)
        eprintln!("DEBUG serialize: {} identities, msg_type={}", msg.identities.len(), msg.header.msg_type);
        for ident in &msg.identities {
            frames.push(ident.clone().into());
        }

        // Add delimiter
        frames.push(DELIMITER.to_vec().into());

        // Add signature
        frames.push(signature.into_bytes().into());

        // Add message parts
        frames.push(header.into());
        frames.push(parent.into());
        frames.push(metadata.into());
        frames.push(content.into());

        eprintln!("DEBUG serialize: total {} frames", frames.len());

        // Convert to ZmqMessage (requires at least one frame, which we always have: delimiter)
        let zmq_msg = ZmqMessage::try_from(frames)
            .map_err(|_| "Failed to create ZMQ message")?;

        Ok(zmq_msg)
    }

    /// Deserialize a ZMQ multipart message to JupyterMessage
    pub fn deserialize_message(&self, msg: ZmqMessage) -> Result<JupyterMessage, Box<dyn std::error::Error>> {
        let parts: Vec<Vec<u8>> = msg.iter().map(|f| f.to_vec()).collect();

        eprintln!("DEBUG deserialize: {} total parts", parts.len());

        // Find delimiter position
        let delim_pos = parts.iter()
            .position(|p| p.as_slice() == DELIMITER)
            .ok_or("No delimiter found in message")?;

        // Extract identities (before delimiter)
        let identities: Vec<Vec<u8>> = parts[..delim_pos].to_vec();
        eprintln!("DEBUG deserialize: {} identities (delim at pos {})", identities.len(), delim_pos);

        // Parts after delimiter: signature, header, parent_header, metadata, content
        if parts.len() < delim_pos + 6 {
            return Err("Message too short".into());
        }

        let signature = String::from_utf8(parts[delim_pos + 1].clone())?;
        let header_bytes = &parts[delim_pos + 2];
        let parent_bytes = &parts[delim_pos + 3];
        let metadata_bytes = &parts[delim_pos + 4];
        let content_bytes = &parts[delim_pos + 5];

        // Verify signature
        if !self.verify_signature(&signature, header_bytes, parent_bytes, metadata_bytes, content_bytes) {
            return Err("Invalid message signature".into());
        }

        let header: Header = serde_json::from_slice(header_bytes)?;
        let parent_header: Option<Header> = {
            let parent_str = String::from_utf8_lossy(parent_bytes);
            if parent_str == "{}" {
                None
            } else {
                Some(serde_json::from_slice(parent_bytes)?)
            }
        };
        let metadata = serde_json::from_slice(metadata_bytes)?;
        let content = serde_json::from_slice(content_bytes)?;

        Ok(JupyterMessage {
            header,
            parent_header,
            metadata,
            content,
            identities,
        })
    }

    /// Receive a message from the shell socket
    pub async fn recv_shell(&mut self) -> Result<JupyterMessage, Box<dyn std::error::Error>> {
        let msg = self.shell.recv().await?;
        self.deserialize_message(msg)
    }

    /// Send a message on the shell socket
    pub async fn send_shell(&mut self, msg: &JupyterMessage) -> Result<(), Box<dyn std::error::Error>> {
        let zmq_msg = self.serialize_message(msg)?;
        self.shell.send(zmq_msg).await?;
        Ok(())
    }

    /// Receive a message from the control socket
    pub async fn recv_control(&mut self) -> Result<JupyterMessage, Box<dyn std::error::Error>> {
        let msg = self.control.recv().await?;
        self.deserialize_message(msg)
    }

    /// Send a message on the control socket
    pub async fn send_control(&mut self, msg: &JupyterMessage) -> Result<(), Box<dyn std::error::Error>> {
        let zmq_msg = self.serialize_message(msg)?;
        self.control.send(zmq_msg).await?;
        Ok(())
    }

    /// Publish a message on the IOPub socket
    /// IOPub uses PUB/SUB pattern and needs a topic as the first frame
    pub async fn publish(&mut self, msg: &JupyterMessage) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("DEBUG publish: msg_type={}", msg.header.msg_type);

        let header = serde_json::to_vec(&msg.header)?;
        let parent = match &msg.parent_header {
            Some(p) => serde_json::to_vec(p)?,
            None => b"{}".to_vec(),
        };
        let metadata = serde_json::to_vec(&msg.metadata)?;
        let content = serde_json::to_vec(&msg.content)?;

        let signature = self.compute_signature(&header, &parent, &metadata, &content);

        // For IOPub, the first frame is the topic (msg_type) for PUB/SUB filtering
        let mut frames: Vec<bytes::Bytes> = Vec::new();

        // Topic frame - use msg_type as topic (Jupyter clients subscribe to "" to get all)
        let topic = msg.header.msg_type.as_bytes().to_vec();
        frames.push(topic.into());

        // Add delimiter
        frames.push(DELIMITER.to_vec().into());

        // Add signature
        frames.push(signature.into_bytes().into());

        // Add message parts
        frames.push(header.into());
        frames.push(parent.into());
        frames.push(metadata.into());
        frames.push(content.into());

        let frame_count = frames.len();
        let zmq_msg = ZmqMessage::try_from(frames)
            .map_err(|_| "Failed to create ZMQ message")?;

        self.iopub.send(zmq_msg).await?;
        eprintln!("DEBUG publish: sent {} frames", frame_count);
        Ok(())
    }

    /// Handle heartbeat (simple echo)
    pub async fn handle_heartbeat(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let msg = self.heartbeat.recv().await?;
        self.heartbeat.send(msg).await?;
        Ok(())
    }
}
