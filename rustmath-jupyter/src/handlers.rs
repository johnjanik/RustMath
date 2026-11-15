//! Jupyter message handlers
//!
//! Implements handlers for all required and optional Jupyter messages.

use crate::connection::KernelSockets;
use crate::protocol::*;
use crate::repl::{EvalError, EvalResult};
use std::collections::HashMap;

/// Handle kernel_info_request
pub fn handle_kernel_info(msg: &JupyterMessage) -> JupyterMessage {
    let content = KernelInfoReply {
        status: "ok".to_string(),
        protocol_version: PROTOCOL_VERSION.to_string(),
        implementation: "rustmath".to_string(),
        implementation_version: env!("CARGO_PKG_VERSION").to_string(),
        language_info: LanguageInfo {
            name: "rustmath".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            mimetype: "text/x-rustmath".to_string(),
            file_extension: ".rm".to_string(),
            pygments_lexer: "python".to_string(), // Use Python lexer as fallback
            codemirror_mode: "python".to_string(),
            nbconvert_exporter: "python".to_string(),
        },
        banner: format!(
            r#"
╭─────────────────────────────────────────────────────────────╮
│  RustMath Kernel v{}                                    │
│  Fast Symbolic Mathematics in Rust                          │
│                                                             │
│  Type 'help' for available commands                         │
╰─────────────────────────────────────────────────────────────╯
"#,
            env!("CARGO_PKG_VERSION")
        ),
        help_links: vec![
            HelpLink {
                text: "RustMath Documentation".to_string(),
                url: "https://github.com/johnjanik/RustMath".to_string(),
            },
            HelpLink {
                text: "Jupyter Documentation".to_string(),
                url: "https://jupyter.org/documentation".to_string(),
            },
        ],
    };

    msg.reply("kernel_info_reply", serde_json::to_value(content).unwrap())
}

/// The result of one evaluation on the worker thread (S13).
#[derive(Debug)]
pub struct EvalOutcome {
    pub result: Result<EvalResult, EvalError>,
    pub stdout: String,
    pub stderr: String,
}

/// First half of execute_request: publish busy and execute_input. Returns the parsed request.
pub async fn begin_execute(
    msg: &JupyterMessage,
    execution_count: u64,
    sockets: &mut KernelSockets,
) -> Result<ExecuteRequest, Box<dyn std::error::Error>> {
    let request: ExecuteRequest = serde_json::from_value(msg.content.clone())?;
    publish_status(sockets, msg, "busy").await?;
    if !request.silent {
        let input_content = ExecuteInputContent {
            code: request.code.clone(),
            execution_count,
        };
        let input_msg = msg.reply("execute_input", serde_json::to_value(input_content)?);
        sockets.publish(&input_msg).await?;
    }
    Ok(request)
}

/// Second half of execute_request: publish the outputs and idle, and return execute_reply.
pub async fn finish_execute(
    msg: &JupyterMessage,
    request: &ExecuteRequest,
    execution_count: u64,
    outcome: EvalOutcome,
    sockets: &mut KernelSockets,
) -> Result<JupyterMessage, Box<dyn std::error::Error>> {
    for (name, text) in [("stdout", outcome.stdout), ("stderr", outcome.stderr)] {
        if !text.is_empty() && !request.silent {
            let stream_content = StreamContent { name: name.to_string(), text };
            let stream_msg = msg.reply("stream", serde_json::to_value(stream_content)?);
            sockets.publish(&stream_msg).await?;
        }
    }
    let reply = match outcome.result {
        Ok(eval_result) => {
            if eval_result.has_output && !request.silent {
                let result_content = ExecuteResultContent {
                    execution_count,
                    data: eval_result.to_data(),
                    metadata: HashMap::new(),
                };
                let result_msg = msg.reply("execute_result", serde_json::to_value(result_content)?);
                sockets.publish(&result_msg).await?;
            }
            ExecuteReply {
                status: "ok".to_string(),
                execution_count,
                payload: Some(Vec::new()),
                user_expressions: Some(HashMap::new()),
                ename: None,
                evalue: None,
                traceback: None,
            }
        }
        Err(eval_error) => {
            if !request.silent {
                let error_content = ErrorContent {
                    ename: eval_error.name.clone(),
                    evalue: eval_error.message.clone(),
                    traceback: eval_error.traceback.clone(),
                };
                let error_msg = msg.reply("error", serde_json::to_value(error_content)?);
                sockets.publish(&error_msg).await?;
            }
            ExecuteReply {
                status: "error".to_string(),
                execution_count,
                payload: None,
                user_expressions: None,
                ename: Some(eval_error.name),
                evalue: Some(eval_error.message),
                traceback: Some(eval_error.traceback),
            }
        }
    };
    publish_status(sockets, msg, "idle").await?;
    Ok(msg.reply("execute_reply", serde_json::to_value(reply)?))
}

/// execute_reply with status "aborted", for requests dropped after an error when
/// `stop_on_error` is set (Jupyter messaging protocol, "Execute").
pub fn aborted_reply(msg: &JupyterMessage, execution_count: u64) -> JupyterMessage {
    let content = serde_json::json!({ "status": "aborted", "execution_count": execution_count });
    msg.reply("execute_reply", content)
}

/// Publish a stream message on IOPub.
pub async fn publish_stream(
    sockets: &mut KernelSockets,
    parent: &JupyterMessage,
    name: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = StreamContent { name: name.to_string(), text: text.to_string() };
    let msg = parent.reply("stream", serde_json::to_value(content)?);
    sockets.publish(&msg).await
}

/// Handle complete_request (tab completion)
pub fn handle_complete(msg: &JupyterMessage) -> JupyterMessage {
    let request: CompleteRequest = serde_json::from_value(msg.content.clone()).unwrap_or(CompleteRequest {
        code: String::new(),
        cursor_pos: 0,
    });

    // Extract the word being completed
    let code = &request.code;
    let cursor = request.cursor_pos.min(code.len());

    // Find start of current word
    let start = code[..cursor]
        .rfind(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);

    let prefix = &code[start..cursor];

    // Built-in completions
    let builtins = [
        "Integer", "Rational", "Complex", "Symbol",
        "factorial", "gcd", "lcm", "is_prime", "factor",
        "diff", "derivative", "help", "vars", "print",
    ];

    let matches: Vec<String> = builtins
        .iter()
        .filter(|name| name.starts_with(prefix))
        .map(|s| s.to_string())
        .collect();

    let content = CompleteReply {
        status: "ok".to_string(),
        matches,
        cursor_start: start,
        cursor_end: cursor,
        metadata: HashMap::new(),
    };

    msg.reply("complete_reply", serde_json::to_value(content).unwrap())
}

/// Handle is_complete_request
pub fn handle_is_complete(msg: &JupyterMessage) -> JupyterMessage {
    let request: IsCompleteRequest = serde_json::from_value(msg.content.clone()).unwrap_or(IsCompleteRequest {
        code: String::new(),
    });

    // Simple heuristic: check for balanced brackets
    let code = &request.code;
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut brace_depth = 0i32;

    for ch in code.chars() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            _ => {}
        }
    }

    let (status, indent) = if paren_depth < 0 || bracket_depth < 0 || brace_depth < 0 {
        ("invalid", None)
    } else if paren_depth > 0 || bracket_depth > 0 || brace_depth > 0 {
        ("incomplete", Some("  ".to_string()))
    } else {
        ("complete", None)
    };

    let content = IsCompleteReply {
        status: status.to_string(),
        indent,
    };

    msg.reply("is_complete_reply", serde_json::to_value(content).unwrap())
}

/// Handle shutdown_request
pub fn handle_shutdown(msg: &JupyterMessage) -> (JupyterMessage, bool) {
    let request: ShutdownRequest = serde_json::from_value(msg.content.clone()).unwrap_or(ShutdownRequest {
        restart: false,
    });

    let content = ShutdownReply {
        status: "ok".to_string(),
        restart: request.restart,
    };

    let reply = msg.reply("shutdown_reply", serde_json::to_value(content).unwrap());
    (reply, request.restart)
}

/// Handle interrupt_request
pub fn handle_interrupt(msg: &JupyterMessage) -> JupyterMessage {
    let content = InterruptReply {
        status: "ok".to_string(),
    };

    msg.reply("interrupt_reply", serde_json::to_value(content).unwrap())
}

/// Publish kernel status on IOPub
pub async fn publish_status(
    sockets: &mut KernelSockets,
    parent: &JupyterMessage,
    state: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = StatusContent {
        execution_state: state.to_string(),
    };
    let msg = parent.reply("status", serde_json::to_value(content)?);
    sockets.publish(&msg).await
}

/// Create a status message for startup
pub fn create_starting_status(session: &str) -> JupyterMessage {
    let content = StatusContent {
        execution_state: "starting".to_string(),
    };
    JupyterMessage::new("status", session, serde_json::to_value(content).unwrap())
}

/// Create an idle status message (for initial ready state)
pub fn create_idle_status(session: &str) -> JupyterMessage {
    let content = StatusContent {
        execution_state: "idle".to_string(),
    };
    JupyterMessage::new("status", session, serde_json::to_value(content).unwrap())
}
