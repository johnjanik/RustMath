//! The kernel answers while a cell runs (S13 of the HANDOFF follow-ups, unit J1b).
//!
//! MAGMA Handbook ch. 1 1.2: `<Ctrl>-C` "interrupts at a convenient point", and `<Ctrl>-\`
//! quits at once (`rustmath-lava/docs/Chapter_01_Statements_and_Expressions.md:24-31`);
//! ch. 3 3.12 `Alarm(s)` (`Chapter_03_Input_and_Output.md:400`). The Jupyter messaging protocol
//! (5.3 and later): `interrupt_request` on the control channel, `execute_request` field
//! `stop_on_error` (default true) and `execute_reply` status `aborted`; the kernel spec field
//! `interrupt_mode`.
//!
//! Each test starts the kernel binary with a connection file in a temporary directory (five
//! free local ports, an empty key, so no HMAC) and drives it like a Jupyter client: DEALER
//! sockets for shell and control, SUB for iopub, REQ for the heartbeat. Before the J1b change
//! the kernel evaluated on its main loop, so none of these requests was answered during a
//! cell, and SIGINT killed it.

use bytes::Bytes;
use serde_json::{json, Value};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use zeromq::{DealerSocket, ReqSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

/// A product of two 100-digit primes (gp: setrand(20261112); p = randomprime([10^99, 10^100]);
/// q = randomprime([10^99, 10^100]); p*q). The REPL's `factor` is trial division, so the call
/// is one library call that runs for far longer than the test.
const SEMIPRIME: &str = "56265256014966897789815307676575655213155187188640868279434630869656784436560427178262558957748157589423366406142190461229644562799099646663196635340923360377107699471706747773488052056948343747546823";

/// A cell of many cooperative steps: each line is short, so an interrupt stops it at the next
/// line (over 20 s in a debug build without an interrupt).
fn long_cell() -> String {
    vec!["x = factorial(3000)"; 2000].join("\n")
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn frames(msg_type: &str, content: Value, id: u64) -> ZmqMessage {
    let header = json!({"msg_id": format!("m{}", id), "session": "test", "username": "test",
        "date": "2026-11-14T00:00:00Z", "msg_type": msg_type, "version": "5.4"});
    let parts: Vec<Bytes> = vec![
        Bytes::from_static(b"<IDS|MSG>"),
        Bytes::from_static(b""),
        Bytes::from(header.to_string()),
        Bytes::from_static(b"{}"),
        Bytes::from_static(b"{}"),
        Bytes::from(content.to_string()),
    ];
    ZmqMessage::try_from(parts).unwrap()
}

/// (msg_type, content) of a received message.
fn parse(m: ZmqMessage) -> (String, Value) {
    let parts: Vec<Vec<u8>> = m.iter().map(|f| f.to_vec()).collect();
    let d = parts.iter().position(|p| p.as_slice() == b"<IDS|MSG>").expect("delimiter");
    let header: Value = serde_json::from_slice(&parts[d + 2]).unwrap();
    let content: Value = serde_json::from_slice(&parts[d + 5]).unwrap();
    (header["msg_type"].as_str().unwrap().to_string(), content)
}

struct Kernel {
    child: Child,
    shell: DealerSocket,
    control: DealerSocket,
    iopub: SubSocket,
    hb: ReqSocket,
    n: u64,
    _dir: tempfile::TempDir,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn connect<S: Socket>(mut s: S, endpoint: &str) -> S {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match s.connect(endpoint).await {
            Ok(()) => return s,
            Err(e) if Instant::now() < deadline => {
                let _ = e;
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(e) => panic!("cannot connect to {endpoint}: {e}"),
        }
    }
}

impl Kernel {
    async fn start() -> Kernel {
        let ports: Vec<u16> = (0..5).map(|_| free_port()).collect();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("connection.json");
        std::fs::write(&file, json!({"shell_port": ports[0], "iopub_port": ports[1], "stdin_port": ports[2],
            "control_port": ports[3], "hb_port": ports[4], "ip": "127.0.0.1", "key": "", "transport": "tcp",
            "signature_scheme": "hmac-sha256", "kernel_name": "rustmath"}).to_string()).unwrap();
        let child = Command::new(env!("CARGO_BIN_EXE_rustmath-kernel"))
            .args(["start", "-f", file.to_str().unwrap()])
            .spawn()
            .unwrap();
        let ep = |p: u16| format!("tcp://127.0.0.1:{}", p);
        let shell = connect(DealerSocket::new(), &ep(ports[0])).await;
        let control = connect(DealerSocket::new(), &ep(ports[3])).await;
        let mut iopub = connect(SubSocket::new(), &ep(ports[1])).await;
        iopub.subscribe("").await.unwrap();
        let hb = connect(ReqSocket::new(), &ep(ports[4])).await;
        let mut k = Kernel { child, shell, control, iopub, hb, n: 0, _dir: dir };
        // Wait for the kernel instead of sleeping: it answers kernel_info_request once its loop runs.
        k.n += 1;
        k.shell.send(frames("kernel_info_request", json!({}), k.n)).await.unwrap();
        let (t, _) = k.shell_reply(30).await.expect("kernel_info_reply");
        assert_eq!(t, "kernel_info_reply");
        k
    }

    async fn execute(&mut self, code: &str) {
        self.n += 1;
        let m = frames("execute_request", json!({"code": code, "silent": false, "store_history": true,
            "user_expressions": {}, "allow_stdin": false, "stop_on_error": true}), self.n);
        self.shell.send(m).await.unwrap();
    }

    async fn shell_reply(&mut self, secs: u64) -> Option<(String, Value)> {
        match tokio::time::timeout(Duration::from_secs(secs), self.shell.recv()).await {
            Ok(Ok(m)) => Some(parse(m)),
            _ => None,
        }
    }

    /// Sends a control request; its reply and how long it took, or None after `secs`.
    async fn control(&mut self, msg_type: &str, content: Value, secs: u64) -> Option<(String, Value, Duration)> {
        self.n += 1;
        let sent = Instant::now();
        self.control.send(frames(msg_type, content, self.n)).await.unwrap();
        match tokio::time::timeout(Duration::from_secs(secs), self.control.recv()).await {
            Ok(Ok(m)) => {
                let (t, c) = parse(m);
                Some((t, c, sent.elapsed()))
            }
            _ => None,
        }
    }

    async fn ping(&mut self) -> Option<Duration> {
        let sent = Instant::now();
        self.hb.send(ZmqMessage::from("ping")).await.unwrap();
        match tokio::time::timeout(Duration::from_secs(5), self.hb.recv()).await {
            Ok(Ok(_)) => Some(sent.elapsed()),
            _ => None,
        }
    }

    /// The text of an execute_result for `code` (the reply must be ok).
    async fn run(&mut self, code: &str) -> String {
        self.execute(code).await;
        let (t, c) = self.shell_reply(60).await.expect("execute_reply");
        assert_eq!(t, "execute_reply");
        assert_eq!(c["status"], "ok", "{code}: {c}");
        // The execute_result went out on iopub before the reply.
        let end = Instant::now() + Duration::from_secs(5);
        while let Ok(Ok(m)) = tokio::time::timeout_at(end.into(), self.iopub.recv()).await {
            let (t, c) = parse(m);
            if t == "execute_result" {
                return c["data"]["text/plain"].as_str().unwrap_or_default().to_string();
            }
        }
        panic!("{code}: no execute_result");
    }

    fn alive(&mut self) -> bool {
        self.child.try_wait().unwrap().is_none()
    }

    #[cfg(unix)]
    fn sigint(&self) {
        let ok = Command::new("kill").args(["-INT", &self.child.id().to_string()]).status().unwrap();
        assert!(ok.success());
    }
}

/// While a cell runs: the heartbeat, kernel_info_request and interrupt_request are answered
/// at once; the cell stops with KeyboardInterrupt; the variables set before it are kept.
#[tokio::test]
async fn kernel_answers_while_busy() {
    let mut k = Kernel::start().await;
    assert_eq!(k.run("a = 5").await, "5");
    k.execute(&long_cell()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let echo = k.ping().await.expect("heartbeat echoed during the cell");
    assert!(echo < Duration::from_secs(1), "{echo:?}");
    let (t, _, d) = k.control("kernel_info_request", json!({}), 5).await.expect("kernel_info_reply");
    assert_eq!(t, "kernel_info_reply");
    assert!(d < Duration::from_secs(1), "{d:?}");
    let (t, _, d) = k.control("interrupt_request", json!({}), 5).await.expect("interrupt_reply");
    assert_eq!(t, "interrupt_reply");
    assert!(d < Duration::from_secs(1), "{d:?}");
    let (t, c) = k.shell_reply(5).await.expect("the cell stops within 5 s");
    assert_eq!(t, "execute_reply");
    assert_eq!(c["status"], "error");
    assert_eq!(c["ename"], "KeyboardInterrupt");
    assert_eq!(k.run("a").await, "5");
}

/// SIGINT (Jupyter's default interrupt_mode "signal") interrupts the cell; it killed the
/// kernel before, even while idle.
#[cfg(unix)]
#[tokio::test]
async fn kernel_survives_sigint() {
    let mut k = Kernel::start().await;
    k.sigint();
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(k.alive(), "an idle SIGINT ended the kernel");
    assert_eq!(k.run("2+2").await, "4");
    k.execute(&long_cell()).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    k.sigint();
    let (_, c) = k.shell_reply(5).await.expect("the cell stops within 5 s");
    assert_eq!(c["ename"], "KeyboardInterrupt");
    assert!(k.alive());
    assert_eq!(k.run("3+3").await, "6");
}

/// After an error, the execute requests already queued are answered "aborted" (stop_on_error,
/// the protocol default), before the failed cell's own reply.
#[tokio::test]
async fn kernel_aborts_queued_cells() {
    let mut k = Kernel::start().await;
    k.execute(&long_cell()).await;
    k.execute("y = 1").await;
    k.execute("y").await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let (t, _, _) = k.control("interrupt_request", json!({}), 5).await.expect("interrupt_reply");
    assert_eq!(t, "interrupt_reply");
    let mut statuses = Vec::new();
    for _ in 0..3 {
        let (t, c) = k.shell_reply(10).await.expect("three replies");
        assert_eq!(t, "execute_reply");
        statuses.push((c["status"].as_str().unwrap().to_string(), c["ename"].as_str().map(String::from)));
    }
    assert_eq!(statuses, vec![
        ("aborted".to_string(), None),
        ("aborted".to_string(), None),
        ("error".to_string(), Some("KeyboardInterrupt".to_string())),
    ]);
    // y was never assigned, so it is the symbol y.
    assert_eq!(k.run("y").await, "y");
}

/// A single library call does not check for interrupts: the kernel says so after 2 s, and a
/// shutdown request is still answered at once and ends the process.
#[tokio::test]
async fn kernel_stops_on_shutdown_during_library_call() {
    let mut k = Kernel::start().await;
    k.execute(&format!("factor({})", SEMIPRIME)).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let (t, _, d) = k.control("interrupt_request", json!({}), 5).await.expect("interrupt_reply");
    assert_eq!(t, "interrupt_reply");
    assert!(d < Duration::from_secs(1), "{d:?}");
    let mut notice = false;
    let end = Instant::now() + Duration::from_secs(4);
    while let Ok(Ok(m)) = tokio::time::timeout_at(end.into(), k.iopub.recv()).await {
        let (t, c) = parse(m);
        if t == "stream" && c["name"] == "stderr" && c["text"].as_str().unwrap_or("").contains("does not check for interrupts") {
            notice = true;
            break;
        }
    }
    assert!(notice, "no stderr notice within 4 s of the interrupt");
    let (t, c, d) = k.control("shutdown_request", json!({"restart": false}), 5).await.expect("shutdown_reply");
    assert_eq!(t, "shutdown_reply");
    assert_eq!(c["restart"], false);
    assert!(d < Duration::from_secs(1), "{d:?}");
    let start = Instant::now();
    while k.alive() && start.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(!k.alive(), "the process is still running 5 s after the shutdown reply");
}
