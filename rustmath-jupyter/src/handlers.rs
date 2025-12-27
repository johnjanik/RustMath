//! Jupyter message handlers
//!
//! Implements handlers for all required and optional Jupyter messages.

use crate::connection::KernelSockets;
use crate::protocol::*;
use crate::repl::ReplContext;
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

/// Handle execute_request
pub async fn handle_execute(
    msg: &JupyterMessage,
    ctx: &mut ReplContext,
    sockets: &mut KernelSockets,
) -> Result<JupyterMessage, Box<dyn std::error::Error>> {
    let request: ExecuteRequest = serde_json::from_value(msg.content.clone())?;

    // Increment execution count if storing history
    if request.store_history {
        ctx.increment_count();
    }

    let execution_count = ctx.execution_count();

    // Publish busy status
    publish_status(sockets, msg, "busy").await?;

    // Publish execute_input if not silent
    if !request.silent {
        let input_content = ExecuteInputContent {
            code: request.code.clone(),
            execution_count,
        };
        let input_msg = msg.reply("execute_input", serde_json::to_value(input_content)?);
        sockets.publish(&input_msg).await?;
    }

    // Execute the code
    let result = ctx.eval(&request.code);

    // Handle stdout
    let stdout = ctx.stdout().to_string();
    if !stdout.is_empty() && !request.silent {
        let stream_content = StreamContent {
            name: "stdout".to_string(),
            text: stdout,
        };
        let stream_msg = msg.reply("stream", serde_json::to_value(stream_content)?);
        sockets.publish(&stream_msg).await?;
    }

    // Handle stderr
    let stderr = ctx.stderr().to_string();
    if !stderr.is_empty() && !request.silent {
        let stream_content = StreamContent {
            name: "stderr".to_string(),
            text: stderr,
        };
        let stream_msg = msg.reply("stream", serde_json::to_value(stream_content)?);
        sockets.publish(&stream_msg).await?;
    }

    let reply = match result {
        Ok(eval_result) => {
            // Publish result if there's output
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
            // Publish error
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

    // Publish idle status
    publish_status(sockets, msg, "idle").await?;

    Ok(msg.reply("execute_reply", serde_json::to_value(reply)?))
}

/// Handle complete_request (tab completion)
pub fn handle_complete(msg: &JupyterMessage, _ctx: &ReplContext) -> JupyterMessage {
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
