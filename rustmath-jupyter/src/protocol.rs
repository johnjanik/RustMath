//! Jupyter wire protocol message types
//!
//! Implements the Jupyter messaging protocol v5.4
//! See: https://jupyter-client.readthedocs.io/en/stable/messaging.html

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Protocol version
pub const PROTOCOL_VERSION: &str = "5.4";

/// Delimiter between ZMQ identities and message content
pub const DELIMITER: &[u8] = b"<IDS|MSG>";

/// Message header containing routing and type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub msg_id: String,
    pub session: String,
    pub username: String,
    pub date: DateTime<Utc>,
    pub msg_type: String,
    pub version: String,
}

impl Header {
    pub fn new(msg_type: &str, session: &str) -> Self {
        Self {
            msg_id: Uuid::new_v4().to_string(),
            session: session.to_string(),
            username: "rustmath".to_string(),
            date: Utc::now(),
            msg_type: msg_type.to_string(),
            version: PROTOCOL_VERSION.to_string(),
        }
    }
}

/// Complete Jupyter message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterMessage {
    pub header: Header,
    pub parent_header: Option<Header>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub content: serde_json::Value,
    #[serde(skip)]
    pub identities: Vec<Vec<u8>>,
}

impl JupyterMessage {
    pub fn new(msg_type: &str, session: &str, content: serde_json::Value) -> Self {
        Self {
            header: Header::new(msg_type, session),
            parent_header: None,
            metadata: HashMap::new(),
            content,
            identities: Vec::new(),
        }
    }

    pub fn reply(&self, msg_type: &str, content: serde_json::Value) -> Self {
        Self {
            header: Header::new(msg_type, &self.header.session),
            parent_header: Some(self.header.clone()),
            metadata: HashMap::new(),
            content,
            identities: self.identities.clone(),
        }
    }
}

// ============================================================================
// Request/Reply message content types
// ============================================================================

/// kernel_info_request content (empty)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfoRequest {}

/// kernel_info_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfoReply {
    pub status: String,
    pub protocol_version: String,
    pub implementation: String,
    pub implementation_version: String,
    pub language_info: LanguageInfo,
    pub banner: String,
    pub help_links: Vec<HelpLink>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub name: String,
    pub version: String,
    pub mimetype: String,
    pub file_extension: String,
    pub pygments_lexer: String,
    pub codemirror_mode: String,
    pub nbconvert_exporter: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelpLink {
    pub text: String,
    pub url: String,
}

/// execute_request content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub code: String,
    #[serde(default)]
    pub silent: bool,
    #[serde(default = "default_true")]
    pub store_history: bool,
    #[serde(default)]
    pub user_expressions: HashMap<String, String>,
    #[serde(default = "default_true")]
    pub allow_stdin: bool,
    #[serde(default)]
    pub stop_on_error: bool,
}

fn default_true() -> bool {
    true
}

/// execute_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteReply {
    pub status: String,
    pub execution_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_expressions: Option<HashMap<String, serde_json::Value>>,
    // Error fields (when status = "error")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evalue: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceback: Option<Vec<String>>,
}

/// complete_request content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteRequest {
    pub code: String,
    pub cursor_pos: usize,
}

/// complete_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteReply {
    pub status: String,
    pub matches: Vec<String>,
    pub cursor_start: usize,
    pub cursor_end: usize,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// is_complete_request content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsCompleteRequest {
    pub code: String,
}

/// is_complete_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsCompleteReply {
    pub status: String, // "complete", "incomplete", "invalid", "unknown"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indent: Option<String>,
}

/// shutdown_request content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownRequest {
    pub restart: bool,
}

/// shutdown_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownReply {
    pub status: String,
    pub restart: bool,
}

/// interrupt_request content (empty)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptRequest {}

/// interrupt_reply content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptReply {
    pub status: String,
}

// ============================================================================
// IOPub message content types
// ============================================================================

/// Kernel status message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusContent {
    pub execution_state: String, // "busy", "idle", "starting"
}

/// Stream output (stdout/stderr)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamContent {
    pub name: String, // "stdout" or "stderr"
    pub text: String,
}

/// Execute input broadcast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteInputContent {
    pub code: String,
    pub execution_count: u64,
}

/// Execute result (display hook output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteResultContent {
    pub execution_count: u64,
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Display data (rich output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayDataContent {
    pub data: HashMap<String, String>,
    pub metadata: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transient: Option<HashMap<String, serde_json::Value>>,
}

/// Error output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContent {
    pub ename: String,
    pub evalue: String,
    pub traceback: Vec<String>,
}

/// Clear output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearOutputContent {
    #[serde(default)]
    pub wait: bool,
}

// ============================================================================
// Stdin message types
// ============================================================================

/// Input request (kernel to frontend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRequest {
    pub prompt: String,
    #[serde(default)]
    pub password: bool,
}

/// Input reply (frontend to kernel)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputReply {
    pub value: String,
}
