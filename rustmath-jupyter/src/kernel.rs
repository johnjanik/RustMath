//! Main kernel implementation.
//!
//! The REPL runs on its own thread ("rustmath-eval", `EVAL_STACK_BYTES` of stack), which owns
//! the `ReplContext`. The async loop serves the shell, control and heartbeat sockets and SIGINT
//! at once (one `tokio::select!`), so `interrupt_request`, `shutdown_request` and the heartbeat
//! are answered while a cell runs (S13 of the HANDOFF follow-ups, unit J1b).
//!
//! What an interrupt stops. MAGMA's `<Ctrl>-C` "interrupts at a convenient point", and a second
//! one within half a second (or `<Ctrl>-\`) quits at once (Handbook ch. 1 1.2,
//! `rustmath-lava/docs/Chapter_01_Statements_and_Expressions.md:24-31`); `Alarm(s)` ends the
//! process after s seconds (ch. 3 3.12, `Chapter_03_Input_and_Output.md:400`). Here:
//!
//! - An interrupt (`interrupt_request` on the control channel, or SIGINT) and the optional time
//!   limit (`time_limit(s)`) stop a cell at its next checkpoint: before each line, between the
//!   sub-expressions of a line, before a result is stored, between the products of
//!   `matrix_power` and between plot samples. An interrupted line has no effect.
//! - A single library call (`factor`, `det`, `simplify`, `taylor`, `nintegrate`, `find_root`,
//!   one big product, one plot sample, a display) runs to its end: Rust cannot stop a thread
//!   from outside, and the libraries do not check a flag. After `INTERRUPT_NOTICE_AFTER` the
//!   kernel says so on stderr.
//! - The hard stop is the process exit: `shutdown_request` (a restart sends it) is answered at
//!   once, `run` returns, and the process ends with the evaluation thread.
//!
//! A panic in a library call becomes an `InternalError` (`eval_guarded`); the kernel and the
//! variables defined before the cell survive. Jupyter messaging protocol 5.3 (`interrupt_request`,
//! `stop_on_error`, status `aborted`) and the kernel spec field `interrupt_mode`.

use crate::connection::{ConnectionInfo, KernelSockets};
use crate::handlers::{
    aborted_reply, begin_execute, create_idle_status, create_starting_status, finish_execute,
    handle_complete, handle_interrupt, handle_is_complete, handle_kernel_info, handle_shutdown,
    publish_status, publish_stream, EvalOutcome,
};
use crate::protocol::{ExecuteRequest, JupyterMessage};
use crate::repl::{EvalError, EvalResult, InterruptHandle, ReplContext};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use zeromq::{SocketRecv, SocketSend};
use uuid::Uuid;

/// Stack of the evaluation thread: deep recursion in the libraries (expression trees) gets
/// 8 times the 8 MiB of a main thread. Only touched pages are committed.
pub const EVAL_STACK_BYTES: usize = 64 << 20;
/// After an interrupt, how long the kernel waits for a checkpoint before it tells the user
/// that the running library call does not check for interrupts.
pub const INTERRUPT_NOTICE_AFTER: Duration = Duration::from_secs(2);
/// After an error with stop_on_error, the execute requests already queued are answered with
/// status "aborted"; the queue is polled without waiting (as ipykernel's default
/// stop_on_error_timeout = 0). Unlike ipykernel, which sends the error reply first and aborts
/// after it, the kernel aborts the queue before it publishes the failed cell's outputs and
/// reply, so a request sent after the client saw the error is never aborted.
pub const ABORT_WINDOW: Duration = Duration::ZERO;

struct EvalJob {
    code: String,
    reply: oneshot::Sender<EvalOutcome>,
}

/// What the evaluation thread runs: `ReplContext` in the kernel, a stub in the tests.
trait Evaluator: Send + 'static {
    fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError>;
    /// stdout and stderr of the last `eval`.
    fn output(&self) -> (String, String);
}

impl Evaluator for ReplContext {
    fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError> {
        ReplContext::eval(self, code)
    }
    fn output(&self) -> (String, String) {
        (self.stdout().to_string(), self.stderr().to_string())
    }
}

/// The evaluation thread and the handle that interrupts it.
struct EvalWorker {
    jobs: std::sync::mpsc::Sender<EvalJob>,
    interrupt: InterruptHandle,
}

impl EvalWorker {
    fn spawn() -> std::io::Result<Self> {
        let ctx = ReplContext::new();
        let interrupt = ctx.interrupt_handle();
        Self::spawn_with(ctx, interrupt)
    }

    /// Starts the thread on `ev`; `interrupt` is the handle that `ev`'s checkpoints read.
    fn spawn_with<E: Evaluator>(ev: E, interrupt: InterruptHandle) -> std::io::Result<Self> {
        let (jobs, rx) = std::sync::mpsc::channel::<EvalJob>();
        std::thread::Builder::new()
            .name("rustmath-eval".to_string())
            .stack_size(EVAL_STACK_BYTES)
            .spawn(move || {
                let mut ev = ev;
                while let Ok(job) = rx.recv() {
                    let result = eval_guarded(|| ev.eval(&job.code));
                    let (stdout, stderr) = ev.output();
                    // The kernel may be gone (shutdown); nothing to report then.
                    let _ = job.reply.send(EvalOutcome { result, stdout, stderr });
                }
            })?;
        Ok(Self { jobs, interrupt })
    }

    /// Sends a job; the reply comes through the returned receiver.
    fn submit(&self, code: String) -> Result<oneshot::Receiver<EvalOutcome>, &'static str> {
        let (tx, rx) = oneshot::channel();
        self.jobs.send(EvalJob { code, reply: tx }).map_err(|_| "the evaluation thread stopped")?;
        Ok(rx)
    }
}

/// Runs `f`, turning a panic into an `InternalError`, so a panic in a library call no longer
/// ends the kernel. `AssertUnwindSafe` is sound for the REPL: it stores a variable only after
/// its value is computed, `eval` clears the output buffers and sets the deadline at its start,
/// and the kernel clears the interrupt flag before each job. What a panicking library leaves
/// behind inside itself (a poisoned mutex, a half-updated cache) is not repaired, and the
/// message does not claim otherwise. A stack overflow still aborts the process.
pub fn eval_guarded<F>(f: F) -> Result<EvalResult, EvalError>
where
    F: FnOnce() -> Result<EvalResult, EvalError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(EvalError::new("InternalError", format!(
                "a library call panicked: {}. The variables defined before this cell are kept.", what)))
        }
    }
}

/// The execution in progress.
struct Running {
    msg: JupyterMessage,
    request: ExecuteRequest,
    execution_count: u64,
    outcome: oneshot::Receiver<EvalOutcome>,
    interrupted_at: Option<Instant>,
    notice_sent: bool,
}

/// The RustMath Jupyter Kernel
pub struct RustMathKernel {
    sockets: KernelSockets,
    worker: EvalWorker,
    session: String,
    execution_count: u64,
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
            worker: EvalWorker::spawn()?,
            session,
            execution_count: 0,
            running: true,
        })
    }

    /// Run the kernel main loop
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let starting_msg = create_starting_status(&self.session);
        self.sockets.publish(&starting_msg).await?;
        let idle_msg = create_idle_status(&self.session);
        self.sockets.publish(&idle_msg).await?;
        eprintln!("RustMath kernel started");

        #[cfg(unix)]
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;

        let mut current: Option<Running> = None;
        while self.running {
            let busy = current.is_some();
            let notice_at = current.as_ref().and_then(|r| {
                if r.notice_sent { None } else { r.interrupted_at.map(|t| t + INTERRUPT_NOTICE_AFTER) }
            });
            let outcome_rx = current.as_mut().map(|r| &mut r.outcome);
            #[cfg(unix)]
            let sigint_fut = sigint.recv();
            #[cfg(not(unix))]
            let sigint_fut = std::future::pending::<Option<()>>();
            tokio::select! {
                r = self.sockets.shell.recv(), if !busy => {
                    let msg = match r { Ok(m) => self.sockets.deserialize_message(m), Err(e) => Err(e.into()) };
                    match msg {
                        Ok(msg) => match self.start_shell_message(msg).await {
                            Ok(run) => current = run,
                            Err(e) => eprintln!("Error handling shell message: {}", e),
                        },
                        Err(e) => eprintln!("Error receiving shell message: {}", e),
                    }
                }
                r = self.sockets.control.recv() => {
                    let msg = match r { Ok(m) => self.sockets.deserialize_message(m), Err(e) => Err(e.into()) };
                    match msg {
                        Ok(msg) => if let Err(e) = self.handle_control_message(msg, current.as_mut()).await {
                            eprintln!("Error handling control message: {}", e);
                        },
                        Err(e) => eprintln!("Error receiving control message: {}", e),
                    }
                }
                r = self.sockets.heartbeat.recv() => {
                    if let Ok(m) = r {
                        if let Err(e) = self.sockets.heartbeat.send(m).await {
                            eprintln!("Heartbeat error: {}", e);
                        }
                    }
                }
                out = async { match outcome_rx { Some(rx) => rx.await, None => std::future::pending().await } } => {
                    let run = current.take().expect("busy");
                    // The job's sender is gone only if the evaluation thread ended (a bug: a
                    // panic is caught, and a stack overflow aborts the whole process). The cell
                    // gets an error and the kernel exits, so that Jupyter restarts it.
                    let worker_gone = out.is_err();
                    let outcome = out.unwrap_or_else(|_| EvalOutcome {
                        result: Err(EvalError::new("InternalError", "the evaluation thread stopped")),
                        stdout: String::new(),
                        stderr: String::new(),
                    });
                    if worker_gone {
                        self.running = false;
                    }
                    // Abort the requests queued behind a failed one before its reply and idle
                    // status go out, so that a request sent after them is never aborted.
                    if outcome.result.is_err() && run.request.stop_on_error && !run.request.silent {
                        self.abort_queued_executes().await?;
                    }
                    let reply = finish_execute(&run.msg, &run.request, run.execution_count, outcome, &mut self.sockets).await?;
                    self.sockets.send_shell(&reply).await?;
                }
                s = sigint_fut => {
                    if s.is_some() {
                        if let Some(run) = current.as_mut() {
                            self.worker.interrupt.interrupt();
                            run.interrupted_at.get_or_insert_with(Instant::now);
                        }
                    }
                }
                _ = async { match notice_at { Some(t) => tokio::time::sleep_until(t.into()).await, None => std::future::pending().await } } => {
                    if let Some(run) = current.as_mut() {
                        run.notice_sent = true;
                        let text = format!(
                            "Interrupt requested {} s ago. The running command is inside a library call that does not \
                             check for interrupts; it stops when that call returns. Restart the kernel to stop it now \
                             (all variables are lost).\n", INTERRUPT_NOTICE_AFTER.as_secs());
                        publish_stream(&mut self.sockets, &run.msg, "stderr", &text).await?;
                    }
                }
            }
        }

        eprintln!("RustMath kernel shutting down");
        Ok(())
    }

    /// Handles one shell message while idle; an execute_request starts a job and returns it.
    async fn start_shell_message(&mut self, msg: JupyterMessage) -> Result<Option<Running>, Box<dyn std::error::Error>> {
        match msg.header.msg_type.as_str() {
            "kernel_info_request" => {
                publish_status(&mut self.sockets, &msg, "busy").await?;
                let reply = handle_kernel_info(&msg);
                self.sockets.send_shell(&reply).await?;
                publish_status(&mut self.sockets, &msg, "idle").await?;
            }
            "execute_request" => {
                let store = msg.content.get("store_history").and_then(|v| v.as_bool()).unwrap_or(true);
                if store {
                    self.execution_count += 1;
                }
                let request = begin_execute(&msg, self.execution_count, &mut self.sockets).await?;
                // Discard an interrupt that arrived while nothing ran. Interrupts are requested
                // only on this task (the control and SIGINT branches of the same loop), so none
                // can fall between the clear and the send; one after the send is seen by the
                // job's checkpoints, one after its last checkpoint is cleared by `eval`.
                self.worker.interrupt.clear();
                let rx = self.worker.submit(request.code.clone())?;
                return Ok(Some(Running {
                    msg,
                    request,
                    execution_count: self.execution_count,
                    outcome: rx,
                    interrupted_at: None,
                    notice_sent: false,
                }));
            }
            "complete_request" => {
                let reply = handle_complete(&msg);
                self.sockets.send_shell(&reply).await?;
            }
            "is_complete_request" => {
                let reply = handle_is_complete(&msg);
                self.sockets.send_shell(&reply).await?;
            }
            other => eprintln!("Unknown shell message type: {}", other),
        }
        Ok(None)
    }

    /// After an error with stop_on_error: answer the execute requests already queued with
    /// "aborted" (Jupyter messaging protocol; ipykernel does the same).
    async fn abort_queued_executes(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        loop {
            let m = match tokio::time::timeout(ABORT_WINDOW, self.sockets.shell.recv()).await {
                Ok(Ok(m)) => m,
                _ => return Ok(()),
            };
            let msg = self.sockets.deserialize_message(m)?;
            if msg.header.msg_type == "execute_request" {
                publish_status(&mut self.sockets, &msg, "busy").await?;
                let reply = aborted_reply(&msg, self.execution_count);
                self.sockets.send_shell(&reply).await?;
                publish_status(&mut self.sockets, &msg, "idle").await?;
            } else if let Some(run) = self.start_shell_message(msg).await? {
                // Not reachable: start_shell_message only starts jobs for execute_request.
                drop(run);
            }
        }
    }

    async fn handle_control_message(&mut self, msg: JupyterMessage, current: Option<&mut Running>) -> Result<(), Box<dyn std::error::Error>> {
        match msg.header.msg_type.as_str() {
            "kernel_info_request" => {
                let reply = handle_kernel_info(&msg);
                self.sockets.send_control(&reply).await?;
            }
            "shutdown_request" => {
                // Also while an evaluation runs: the process exits when run() returns, which
                // stops the evaluation thread (a thread cannot be stopped any other way).
                let (reply, restart) = handle_shutdown(&msg);
                self.sockets.send_control(&reply).await?;
                self.running = false;
                if restart {
                    eprintln!("Kernel restart requested");
                }
            }
            "interrupt_request" => {
                if let Some(run) = current {
                    self.worker.interrupt.interrupt();
                    run.interrupted_at.get_or_insert_with(Instant::now);
                }
                let reply = handle_interrupt(&msg);
                self.sockets.send_control(&reply).await?;
            }
            other => eprintln!("Unknown control message type: {}", other),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_guarded_turns_panics_into_errors() {
        let e = eval_guarded(|| panic!("boom")).unwrap_err();
        assert_eq!(e.name, "InternalError");
        assert!(e.message.contains("boom"), "{}", e.message);
        assert!(e.message.contains("variables defined before this cell are kept"), "{}", e.message);
        let s = String::from("formatted");
        let e = eval_guarded(|| panic!("{} panic", s)).unwrap_err();
        assert!(e.message.contains("formatted panic"), "{}", e.message);
        assert_eq!(eval_guarded(|| Ok(EvalResult::text("x"))).unwrap().text, "x");
    }

    /// A stub that counts its evaluations and panics on "boom".
    struct Stub {
        count: u64,
    }

    impl Evaluator for Stub {
        fn eval(&mut self, code: &str) -> Result<EvalResult, EvalError> {
            if code == "boom" {
                panic!("boom in the stub");
            }
            self.count += 1;
            Ok(EvalResult::text(self.count.to_string()))
        }
        fn output(&self) -> (String, String) {
            (format!("out {}", self.count), String::new())
        }
    }

    /// The real evaluation loop survives a panic and keeps its state (replaces an integration
    /// test that needed a panicking REPL command).
    #[test]
    fn test_eval_worker_survives_panic() {
        let worker = EvalWorker::spawn_with(Stub { count: 0 }, InterruptHandle::default()).unwrap();
        let a = worker.submit("a".to_string()).unwrap().blocking_recv().unwrap();
        assert_eq!(a.result.unwrap().text, "1");
        assert_eq!(a.stdout, "out 1");
        let boom = worker.submit("boom".to_string()).unwrap().blocking_recv().unwrap();
        let e = boom.result.unwrap_err();
        assert_eq!(e.name, "InternalError");
        assert!(e.message.contains("boom in the stub"), "{}", e.message);
        let b = worker.submit("b".to_string()).unwrap().blocking_recv().unwrap();
        assert_eq!(b.result.unwrap().text, "2");
    }

    /// The kernel's worker runs the REPL on a thread of EVAL_STACK_BYTES.
    #[test]
    fn test_eval_worker_runs_the_repl() {
        let worker = EvalWorker::spawn().unwrap();
        let r = worker.submit("a = 2 + 3".to_string()).unwrap().blocking_recv().unwrap();
        assert_eq!(r.result.unwrap().text, "5");
        let r = worker.submit("a * a".to_string()).unwrap().blocking_recv().unwrap();
        assert_eq!(r.result.unwrap().text, "25");
        worker.interrupt.interrupt();
        let r = worker.submit("b = 1".to_string()).unwrap().blocking_recv().unwrap();
        assert_eq!(r.result.unwrap_err().name, "KeyboardInterrupt");
    }
}
