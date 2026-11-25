use alice_native_tts::session::{Device, Session, SessionError, SessionResult};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Synthesize {
        request_id: String,
        text: String,
        language: Option<String>,
    },
    Cancel {
        request_id: String,
    },
    Shutdown,
}

struct Active {
    request_id: String,
    cancelled: Arc<AtomicBool>,
}

struct Job {
    request_id: String,
    text: String,
    language: Option<String>,
    cancelled: Arc<AtomicBool>,
}

type Output = Mutex<BufWriter<io::Stdout>>;
type State = Mutex<Option<Active>>;

fn send(output: &Output, event: Value) -> SessionResult<()> {
    let mut output = output.lock().expect("stdout writer mutex");
    serde_json::to_writer(&mut *output, &event)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn reject(output: &Output, request_id: Option<&str>, message: &str) -> SessionResult<()> {
    send(
        output,
        json!({ "type": "rejected", "request_id": request_id, "message": message }),
    )
}

fn read_requests(output: &Output, state: &State, jobs: Sender<Job>) -> SessionResult<()> {
    for line in io::stdin().lock().lines() {
        let line = line?;
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                reject(output, None, &error.to_string())?;
                continue;
            }
        };
        let mut active = state.lock().expect("request state mutex");
        match request {
            Request::Synthesize {
                request_id,
                text,
                language,
            } => {
                if request_id.trim().is_empty() {
                    reject(output, Some(&request_id), "request_id is empty")?;
                } else if active.is_some() {
                    reject(output, Some(&request_id), "worker is already synthesizing")?;
                } else {
                    let cancelled = Arc::new(AtomicBool::new(false));
                    *active = Some(Active {
                        request_id: request_id.clone(),
                        cancelled: cancelled.clone(),
                    });
                    if jobs
                        .send(Job {
                            request_id,
                            text,
                            language,
                            cancelled,
                        })
                        .is_err()
                    {
                        return Ok(());
                    }
                }
            }
            Request::Cancel { request_id } => {
                if let Some(active) = active
                    .as_ref()
                    .filter(|active| active.request_id == request_id)
                {
                    active.cancelled.store(true, Ordering::Relaxed);
                } else {
                    reject(output, Some(&request_id), "request is not active")?;
                }
            }
            Request::Shutdown => {
                if let Some(active) = active.as_ref() {
                    active.cancelled.store(true, Ordering::Relaxed);
                }
                return Ok(());
            }
        }
    }
    Ok(())
}

pub(super) fn run(mut session: Session, device: Device) -> SessionResult<()> {
    let sample_rate = session
        .info()
        .sample_rate
        .expect("loaded model sample rate");
    let output = Arc::new(Mutex::new(BufWriter::new(io::stdout())));
    send(
        &output,
        json!({
            "type": "ready",
            "protocol_version": 1,
            "pcm_format": "s16le",
            "channels": 1,
            "device": device,
            "model": session.info(),
        }),
    )?;
    let state: Arc<State> = Arc::new(Mutex::new(None));
    let (jobs, pending) = mpsc::channel();
    let reader_output = output.clone();
    let reader_state = state.clone();
    std::thread::Builder::new()
        .name("alice-tts-input".into())
        .spawn(move || {
            if let Err(error) = read_requests(&reader_output, &reader_state, jobs) {
                if let Some(active) = reader_state.lock().expect("request state mutex").as_ref() {
                    active.cancelled.store(true, Ordering::Relaxed);
                }
                eprintln!("alice-tts input: {error}");
            }
        })?;

    for job in pending {
        let cancelled = || job.cancelled.load(Ordering::Relaxed);
        let result = session.synthesize(&job.text, job.language.as_deref(), &cancelled, |pcm| {
            let bytes: Vec<_> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
            send(
                &output,
                json!({
                    "type": "audio",
                    "request_id": &job.request_id,
                    "sample_rate": sample_rate,
                    "pcm": STANDARD.encode(bytes),
                }),
            )
        });
        let mut active = state.lock().expect("request state mutex");
        let event = if cancelled() {
            json!({ "type": "cancelled", "request_id": job.request_id })
        } else {
            match result {
                Ok(samples) => {
                    json!({ "type": "done", "request_id": job.request_id, "samples": samples })
                }
                Err(SessionError::Cancelled) => {
                    json!({ "type": "cancelled", "request_id": job.request_id })
                }
                Err(error) => {
                    json!({ "type": "error", "request_id": job.request_id, "message": error.to_string() })
                }
            }
        };
        send(&output, event)?;
        *active = None;
    }
    Ok(())
}
