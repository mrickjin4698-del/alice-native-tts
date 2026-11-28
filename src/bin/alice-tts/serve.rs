use super::{invalid, wav_spec};
use alice_native_tts::session::{
    Device, Model, ModelInfo, Session, SessionError, SessionOptions, SessionResult,
};
use axum::extract::{
    rejection::JsonRejection,
    ws::{close_code, CloseFrame, Message, WebSocket, WebSocketUpgrade},
    DefaultBodyLimit, State,
};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::future::IntoFuture;
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout;

const MAX_REQUEST_BYTES: usize = 16 * 1024;
const MAX_TEXT_CHARS: usize = 4096;
const MAX_WAV_SAMPLES: usize = 16 * 1024 * 1024;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechRequest {
    text: String,
    language: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Control {
    Cancel,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({"error": {"code": self.code, "message": self.message}})),
        )
            .into_response()
    }
}

fn synthesis_error(error: impl std::fmt::Display) -> ApiError {
    eprintln!("alice-tts serve: {error}");
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "synthesis_failed",
        "synthesis failed",
    )
}

struct Job {
    request: SpeechRequest,
    cancelled: Arc<AtomicBool>,
    audio: mpsc::Sender<Vec<i16>>,
    done: oneshot::Sender<SessionResult<u64>>,
    permit: OwnedSemaphorePermit,
}

struct Synthesis {
    cancelled: Arc<AtomicBool>,
    audio: mpsc::Receiver<Vec<i16>>,
    done: oneshot::Receiver<SessionResult<u64>>,
}

impl Synthesis {
    async fn finish(&mut self) -> Result<u64, ApiError> {
        (&mut self.done)
            .await
            .map_err(synthesis_error)?
            .map_err(synthesis_error)
    }
}

impl Drop for Synthesis {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct ServerState {
    model: Arc<ModelInfo>,
    device: Device,
    jobs: mpsc::Sender<Job>,
    available: Arc<Semaphore>,
}

impl ServerState {
    fn sample_rate(&self) -> u32 {
        self.model.sample_rate.expect("loaded model sample rate")
    }

    fn max_text_chars(&self) -> usize {
        self.model
            .max_text_chars()
            .unwrap_or(MAX_TEXT_CHARS)
            .min(MAX_TEXT_CHARS)
    }

    fn start(&self, request: SpeechRequest) -> Result<Synthesis, ApiError> {
        let limit = self.max_text_chars();
        if request.text.trim().is_empty() || request.text.chars().count() > limit {
            return Err(ApiError::invalid(format!(
                "text must be non-empty and at most {limit} characters"
            )));
        }
        if request
            .language
            .as_ref()
            .is_some_and(|language| !self.model.languages.contains(language))
        {
            return Err(ApiError::invalid("language is unsupported by this model"));
        }
        let permit =
            self.available.clone().try_acquire_owned().map_err(|_| {
                ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "busy", "model is busy")
            })?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (audio, pcm) = mpsc::channel(2);
        let (done, completion) = oneshot::channel();
        self.jobs
            .try_send(Job {
                request,
                cancelled: cancelled.clone(),
                audio,
                done,
                permit,
            })
            .map_err(|_| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "unavailable",
                    "model worker is unavailable",
                )
            })?;
        Ok(Synthesis {
            cancelled,
            audio: pcm,
            done: completion,
        })
    }
}

async fn health(State(state): State<ServerState>) -> Json<Value> {
    Json(json!({
        "status": "ready",
        "protocol_version": 1,
        "max_text_chars": state.max_text_chars(),
        "device": state.device,
        "model": &*state.model,
    }))
}

async fn synthesize_http(
    State(state): State<ServerState>,
    input: Result<Json<SpeechRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = input.map_err(|rejection| {
        ApiError::new(rejection.status(), "invalid_request", rejection.body_text())
    })?;
    let mut synthesis = state.start(request)?;
    let mut bytes = Cursor::new(Vec::new());
    let mut wav = hound::WavWriter::new(&mut bytes, wav_spec(state.sample_rate()))
        .map_err(synthesis_error)?;
    let mut samples = 0;
    while let Some(pcm) = synthesis.audio.recv().await {
        samples += pcm.len();
        if samples > MAX_WAV_SAMPLES {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "audio_too_large",
                "result exceeds the WAV size limit; use the WebSocket endpoint",
            ));
        }
        let mut writer = wav.get_i16_writer(pcm.len() as u32);
        for sample in pcm {
            writer.write_sample(sample);
        }
        writer.flush().map_err(synthesis_error)?;
    }
    synthesis.finish().await?;
    wav.finalize().map_err(synthesis_error)?;
    Ok((
        [
            (header::CONTENT_TYPE, "audio/wav"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        bytes.into_inner(),
    )
        .into_response())
}

async fn synthesize_ws(State(state): State<ServerState>, ws: WebSocketUpgrade) -> Response {
    ws.max_message_size(MAX_REQUEST_BYTES)
        .max_frame_size(MAX_REQUEST_BYTES)
        .on_upgrade(move |socket| websocket_session(socket, state))
}

async fn send(socket: &mut WebSocket, message: Message) -> bool {
    matches!(
        timeout(SOCKET_TIMEOUT, socket.send(message)).await,
        Ok(Ok(()))
    )
}

async fn finish_close(socket: &mut WebSocket) {
    // Polling the socket flushes tungstenite's queued close reply.
    let _ = timeout(SOCKET_TIMEOUT, async {
        while matches!(socket.recv().await, Some(Ok(_))) {}
    })
    .await;
}

async fn read_request(socket: &mut WebSocket) -> Option<Result<SpeechRequest, ApiError>> {
    loop {
        match socket.recv().await? {
            Ok(Message::Text(text)) => {
                return Some(
                    serde_json::from_str(text.as_str())
                        .map_err(|error| ApiError::invalid(error.to_string())),
                );
            }
            Ok(Message::Ping(_) | Message::Pong(_)) => {}
            Ok(Message::Close(_)) => {
                finish_close(socket).await;
                return None;
            }
            Err(_) => return None,
            Ok(Message::Binary(_)) => return Some(Err(ApiError::invalid("expected JSON text"))),
        }
    }
}

async fn stream_audio(
    socket: &mut WebSocket,
    state: &ServerState,
) -> Result<Option<Value>, ApiError> {
    let request = match timeout(SOCKET_TIMEOUT, read_request(socket)).await {
        Ok(Some(request)) => request?,
        Ok(None) => return Ok(None),
        Err(_) => {
            return Err(ApiError::invalid(
                "timed out waiting for a synthesis request",
            ))
        }
    };
    let mut synthesis = state.start(request)?;
    let started = json!({
        "type": "start",
        "sample_rate": state.sample_rate(),
        "channels": 1,
        "pcm_format": "s16le",
        "streaming": state.model.streaming,
    });
    if !send(socket, Message::Text(started.to_string().into())).await {
        return Ok(None);
    }

    loop {
        tokio::select! {
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                Some(Ok(Message::Text(text))) if matches!(
                    serde_json::from_str(text.as_str()), Ok(Control::Cancel)
                ) => return Ok(Some(json!({"type": "cancelled"}))),
                Some(Ok(Message::Text(_) | Message::Binary(_))) => {
                    return Err(ApiError::invalid("send only a cancel operation during synthesis"));
                }
                Some(Ok(Message::Close(_))) => {
                    drop(synthesis);
                    finish_close(socket).await;
                    return Ok(None);
                }
                Some(Err(_)) | None => return Ok(None),
            },
            audio = synthesis.audio.recv() => match audio {
                Some(pcm) => {
                    let bytes: Vec<_> = pcm.into_iter().flat_map(i16::to_le_bytes).collect();
                    if !send(socket, Message::Binary(bytes.into())).await {
                        return Ok(None);
                    }
                }
                None => {
                    let samples = synthesis.finish().await?;
                    return Ok(Some(json!({"type": "done", "samples": samples})));
                }
            },
        }
    }
}

async fn websocket_session(mut socket: WebSocket, state: ServerState) {
    let terminal = match stream_audio(&mut socket, &state).await {
        Ok(Some(event)) => event,
        Ok(None) => return,
        Err(error) => json!({"type": "error", "code": error.code, "message": error.message}),
    };
    let close = CloseFrame {
        code: close_code::NORMAL,
        reason: "".into(),
    };
    if send(&mut socket, Message::Text(terminal.to_string().into())).await
        && send(&mut socket, Message::Close(Some(close))).await
    {
        finish_close(&mut socket).await;
    }
}

fn run_job(session: &mut Session, job: Job, stopping: &AtomicBool) {
    let cancelled = || job.cancelled.load(Ordering::Relaxed) || stopping.load(Ordering::Relaxed);
    let result = session.synthesize(
        &job.request.text,
        job.request.language.as_deref(),
        &cancelled,
        |pcm| {
            job.audio
                .blocking_send(pcm.to_vec())
                .map_err(|_| SessionError::Cancelled)
        },
    );
    drop(job.permit);
    let _ = job.done.send(result);
}

pub(super) fn run(
    model: Model,
    options: SessionOptions,
    warmup: Option<String>,
    bind: SocketAddr,
) -> SessionResult<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let listener = runtime.block_on(TcpListener::bind(bind))?;
    let address = listener.local_addr()?;
    let (jobs, mut pending) = mpsc::channel(1);
    let (ready, startup) = std::sync::mpsc::sync_channel::<SessionResult<ModelInfo>>(1);
    let stopping = Arc::new(AtomicBool::new(false));
    let worker_stopping = stopping.clone();
    let device = options.device;
    // Keep model creation, all inference, and runtime teardown on one thread.
    let worker = std::thread::Builder::new()
        .name("alice-tts-inference".into())
        .spawn(move || {
            let load = || {
                let mut session = Session::load(model, &options, &|| false)?;
                if let Some(text) = warmup {
                    session.synthesize(&text, None, &|| false, |_| Ok(()))?;
                }
                Ok(session)
            };
            let mut session = match load() {
                Ok(session) => session,
                Err(error) => {
                    let _ = ready.send(Err(error));
                    return;
                }
            };
            if ready.send(Ok(session.info().clone())).is_err() {
                return;
            }
            while let Some(job) = pending.blocking_recv() {
                run_job(&mut session, job, &worker_stopping);
            }
        })?;
    let model = startup
        .recv()
        .map_err(|_| invalid("model worker stopped during startup"))??;
    let state = ServerState {
        model: Arc::new(model),
        device,
        jobs,
        available: Arc::new(Semaphore::new(1)),
    };
    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/tts", post(synthesize_http))
        .route("/v1/tts/ws", get(synthesize_ws))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state);
    eprintln!("alice-tts listening on http://{address}");
    let result = runtime.block_on(async move {
        tokio::select! {
            result = axum::serve(listener, app).into_future() => result,
            signal = tokio::signal::ctrl_c() => signal,
        }
    });
    stopping.store(true, Ordering::Relaxed);
    drop(runtime);
    worker
        .join()
        .map_err(|_| invalid("model worker panicked"))?;
    Ok(result?)
}
