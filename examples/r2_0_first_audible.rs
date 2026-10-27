//! R2.0: conservative leading-silence trim experiment.
//!
//! This wraps only the first PCM callback. It does not alter token sampling,
//! VITS inputs, decoder state, or the default streaming API.

use alice_native_tts::{Backend, Config, NativeTts, SynthesisTimings};
use serde_json::{json, Value};
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

const SAMPLE_RATE: usize = 32_000;
const WINDOW_SAMPLES: usize = SAMPLE_RATE / 100; // 10 ms
const ACTIVE_RMS_THRESHOLD: f64 = 1e-3;
const STRICT_SILENCE_RMS_THRESHOLD: f64 = 1e-7;
const STRICT_SILENCE_PEAK_THRESHOLD: f32 = 1e-6;
const MIN_QUIET_PREFIX_SAMPLES: usize = 100 * SAMPLE_RATE / 1000;
const ONSET_PREROLL_SAMPLES: usize = 80 * SAMPLE_RATE / 1000;
const MIN_TRIM_SAMPLES: usize = 20 * SAMPLE_RATE / 1000;
const WARMUPS: usize = 30;
const SAMPLES: usize = 20;

const CORPUS: [(&str, &str); 8] = [
    ("zh_short", "你好。"),
    ("zh_poly_short", "行长来了。"),
    (
        "zh_long",
        "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
    ),
    (
        "zh_poly_long",
        "银行行长在重庆重看那段长长的乐章，朋友们还在行走；后来他着重介绍了重要的行程。",
    ),
    ("en_short", "Hello, Alice."),
    (
        "en_long",
        "The quick brown fox jumps over the lazy dog while Alice reads a book and welcomes everyone home.",
    ),
    ("mixed_short", "你好 Alice，请看这个 report。"),
    (
        "mixed_long",
        "你好 Alice，请 read the report about 银行行长 who will travel to 重庆 tomorrow.",
    ),
];

#[derive(Clone, Copy, Debug)]
enum Strategy {
    Chunk24,
    StrictLeadingSilenceTrim,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Self::Chunk24 => "chunk24",
            Self::StrictLeadingSilenceTrim => "strict_prefix_trim",
        }
    }

    fn trims(self) -> bool {
        matches!(self, Self::StrictLeadingSilenceTrim)
    }
}

const STRATEGIES: [Strategy; 2] = [Strategy::Chunk24, Strategy::StrictLeadingSilenceTrim];

#[derive(Clone, serde::Serialize)]
struct CallbackRecord {
    at_ms: f64,
    sample_count: usize,
    first_active_offset_ms: Option<f64>,
    final_chunk: bool,
    boundary_jump_before: Option<f64>,
}

struct CapturedRun {
    row: Value,
    audio: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: r2_0_first_audible PACK OUTPUT_DIR".into());
    }
    let pack = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    if output.exists() && fs::read_dir(&output)?.next().is_some() {
        return Err(format!(
            "refusing to overwrite non-empty output directory: {}",
            output.display()
        )
        .into());
    }
    fs::create_dir_all(&output)?;
    let frontend_backend = env::var("R2_0_FRONTEND_BACKEND").unwrap_or_else(|_| "cuda".to_owned());
    if !matches!(frontend_backend.as_str(), "cpu" | "cuda") {
        return Err("R2_0_FRONTEND_BACKEND must be cpu or cuda".into());
    }
    configure_runtime(&frontend_backend);
    let load_started = Instant::now();
    let mut model = NativeTts::load(
        &Config {
            model_dir: pack,
            reference_audio: None,
            reference_text: None,
            backend: Backend::Cuda,
            streaming: true,
        },
        &|| false,
    )?;
    eprintln!(
        "R2.0 model load complete in {:.1} ms",
        load_started.elapsed().as_secs_f64() * 1000.0
    );
    if env::var("R2_0_LOAD_ONLY").ok().as_deref() == Some("1") {
        eprintln!("R2.0 load-only diagnostic complete; benchmark skipped");
        return Ok(());
    }
    run_benchmark(&mut model, &output, &frontend_backend)?;
    Ok(())
}

fn configure_runtime(frontend_backend: &str) {
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_BACKEND", frontend_backend);
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_BERT_BACKEND", frontend_backend);
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_G2PW_PARALLEL", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_PARALLEL", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE", "1");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CACHE_FORCE_MISS", "0");
    env::set_var("ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_CUDA_VITS", "1");
    env::set_var("ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_GPU_DIAGNOSTIC_ALLOW_CPU", "1");
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_BINDING_MODE", "clear_rebind");
    env::set_var("ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH", "0");
    env::remove_var("ALICE_NATIVE_TTS_GPT_STEP");
}

fn run_benchmark(
    model: &mut NativeTts,
    output: &Path,
    frontend_backend: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut all_runs = Vec::new();
    let mut blind_tracks = Vec::<(String, String, String, bool, Vec<f32>)>::new();

    for (group_id, text) in CORPUS {
        let mut measured: [Vec<Value>; 2] = [Vec::new(), Vec::new()];
        let mut first_audio: [Option<Vec<f32>>; 2] = [None, None];
        for warmup in 0..WARMUPS {
            if warmup == 0 {
                eprintln!("R2.0 {group_id}: first control inference starting");
            }
            for strategy in STRATEGIES {
                let _ = synthesize(model, group_id, text, strategy)?;
            }
            if warmup == 0 {
                eprintln!("R2.0 {group_id}: first A/B pair complete");
            }
            if (warmup + 1) % 10 == 0 {
                eprintln!("R2.0 {group_id}: warmup {}/{}", warmup + 1, WARMUPS);
            }
        }

        for sample in 0..SAMPLES {
            // Alternate order after warmup to reduce drift/order bias.
            let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut sample_runs: [Option<CapturedRun>; 2] = [None, None];
            for index in order {
                sample_runs[index] = Some(synthesize(model, group_id, text, STRATEGIES[index])?);
            }
            let baseline = sample_runs[0].take().ok_or("missing Chunk24 run")?;
            let trimmed = sample_runs[1].take().ok_or("missing trim run")?;
            let trim_samples = trimmed.row["trimmed_samples"].as_u64().unwrap_or(0) as usize;
            let waveform_check =
                compare_after_prefix_trim(&baseline.audio, &trimmed.audio, trim_samples);
            let mut baseline_row = baseline.row;
            baseline_row["output_matches_control_after_trim"] = json!(true);
            let mut trim_row = trimmed.row;
            trim_row["output_matches_control_after_trim"] = json!(waveform_check.0);
            trim_row["post_trim_waveform_max_abs_error"] = json!(waveform_check.1);

            if sample == 0 {
                first_audio[0] = Some(baseline.audio);
                first_audio[1] = Some(trimmed.audio);
            }
            measured[0].push(baseline_row.clone());
            measured[1].push(trim_row.clone());
            all_runs.push(baseline_row);
            all_runs.push(trim_row);
            if (sample + 1) % 5 == 0 {
                eprintln!("R2.0 {group_id}: measured {}/{}", sample + 1, SAMPLES);
            }
        }

        for (index, strategy) in STRATEGIES.iter().copied().enumerate() {
            let audio = first_audio[index]
                .take()
                .ok_or("missing first measured waveform")?;
            blind_tracks.push((
                group_id.to_owned(),
                strategy.name().to_owned(),
                text.to_owned(),
                true,
                audio,
            ));
            println!(
                "R2.0 {group_id}/{} {}",
                strategy.name(),
                summarize(&measured[index])
            );
        }
    }

    let report = json!({
        "schema": "alice-native-tts-r2.0-first-audible-v1",
        "runtime": "CUDA GPT + CUDA VITS, CPU token sampling, ABI-1",
        "frontend_backend": frontend_backend,
        "seed": 1234,
        "default_behavior_changed": false,
        "model_or_sampling_changed": false,
        "quiet_rms_threshold": STRICT_SILENCE_RMS_THRESHOLD,
        "quiet_peak_threshold": STRICT_SILENCE_PEAK_THRESHOLD,
        "min_quiet_prefix_ms": MIN_QUIET_PREFIX_SAMPLES * 1000 / SAMPLE_RATE,
        "onset_preroll_ms": ONSET_PREROLL_SAMPLES * 1000 / SAMPLE_RATE,
        "min_trim_ms": MIN_TRIM_SAMPLES * 1000 / SAMPLE_RATE,
        "warmups_per_group_strategy": WARMUPS,
        "samples_per_group_strategy": SAMPLES,
        "buffer_simulation_ms": [32, 64],
        "groups": summarize_all(&all_runs),
        "runs": all_runs,
        "blind_package": "blind-listening/",
        "blind_key": "blind-key.json",
        "human_blind_audition": "NOT_RUN; anonymous package exported",
        "real_playback_acceptance": "NOT_RUN"
    });
    fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    export_blind_package(output, blind_tracks)?;
    Ok(())
}

fn synthesize(
    model: &mut NativeTts,
    group_id: &str,
    text: &str,
    strategy: Strategy,
) -> Result<CapturedRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut audio = Vec::<f32>::new();
    let mut callbacks = Vec::<CallbackRecord>::new();
    let mut trimmed_samples = 0usize;
    let timings: SynthesisTimings =
        model.synthesize_streaming(text, &|| false, |samples, info| {
            let at_ms = started.elapsed().as_secs_f64() * 1000.0;
            let trim = if callbacks.is_empty() && strategy.trims() {
                strict_leading_silence_trim(samples)
            } else {
                0
            };
            if callbacks.is_empty() {
                trimmed_samples = trim;
            }
            let outgoing = &samples[trim..];
            let active_offset = first_active_sample(outgoing)
                .map(|sample| sample as f64 * 1000.0 / SAMPLE_RATE as f64);
            callbacks.push(CallbackRecord {
                at_ms,
                sample_count: outgoing.len(),
                first_active_offset_ms: active_offset,
                final_chunk: info.final_chunk,
                boundary_jump_before: audio
                    .last()
                    .zip(outgoing.first())
                    .map(|(before, after)| f64::from((*before - *after).abs())),
            });
            audio.extend_from_slice(outgoing);
        })?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let first = callbacks.first().ok_or("stream produced no callback")?;
    let second = callbacks.get(1);
    let first_active_pcm_ms = callbacks.iter().find_map(|callback| {
        callback
            .first_active_offset_ms
            .map(|offset| callback.at_ms + offset)
    });
    let generated_samples = audio.len() + trimmed_samples;
    let generated_seconds = generated_samples as f64 / SAMPLE_RATE as f64;
    let rtf = elapsed_ms / (generated_seconds * 1000.0);
    let finite = audio.iter().all(|sample| sample.is_finite());
    if audio.is_empty() || !finite {
        return Err(format!("{group_id}/{} produced invalid PCM", strategy.name()).into());
    }
    let boundary_jumps = callbacks
        .iter()
        .filter_map(|callback| callback.boundary_jump_before)
        .collect::<Vec<_>>();
    let row = json!({
        "group_id": group_id,
        "strategy": strategy.name(),
        "completed": true,
        "elapsed_ms": elapsed_ms,
        "ttfa_callback_ms": first.at_ms,
        "first_active_pcm_ms": first_active_pcm_ms,
        "first_active_offset_in_emitted_audio_ms": first.first_active_offset_ms,
        "second_callback_ms": second.map(|callback| callback.at_ms),
        "first_callback_samples_original": first.sample_count + trimmed_samples,
        "first_callback_samples_emitted": first.sample_count,
        "trimmed_samples": trimmed_samples,
        "trimmed_ms": trimmed_samples as f64 * 1000.0 / SAMPLE_RATE as f64,
        "generated_audio_seconds": generated_seconds,
        "emitted_audio_seconds": audio.len() as f64 / SAMPLE_RATE as f64,
        "audio_samples_emitted": audio.len(),
        "audio_finite": finite,
        "semantic_tokens": timings.semantic_tokens,
        "first_audio_tokens": timings.first_audio_tokens,
        "audio_chunks": timings.audio_chunks,
        "frontend_ms": timings.frontend_ms,
        "gpt_ms": timings.gpt_ms,
        "vocoder_ms": timings.vocoder_ms,
        "rtf_original_audio_duration": rtf,
        "callbacks": callbacks,
        "boundary_jumps": boundary_jumps,
        "playback_buffer_simulation": [simulate_buffer(&callbacks, 32, elapsed_ms), simulate_buffer(&callbacks, 64, elapsed_ms)]
    });
    Ok(CapturedRun { row, audio })
}

/// Trim only a continuous, near-machine-zero prefix. A low but nonzero onset
/// stops trimming; a minimum 80 ms quiet pre-roll remains before the first
/// non-quiet window. Interior pauses are never removed.
fn strict_leading_silence_trim(audio: &[f32]) -> usize {
    let Some(active_onset) = first_active_sample(audio) else {
        return 0;
    };
    let mut quiet_prefix = 0usize;
    for frame in audio.chunks_exact(WINDOW_SAMPLES) {
        let energy = frame
            .iter()
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>();
        let rms = (energy / frame.len() as f64).sqrt();
        let peak = frame
            .iter()
            .fold(0.0f32, |maximum, sample| maximum.max(sample.abs()));
        if rms > STRICT_SILENCE_RMS_THRESHOLD || peak > STRICT_SILENCE_PEAK_THRESHOLD {
            break;
        }
        quiet_prefix += frame.len();
    }
    if quiet_prefix < MIN_QUIET_PREFIX_SAMPLES {
        return 0;
    }
    let before_guard = active_onset.saturating_sub(ONSET_PREROLL_SAMPLES);
    let trim = quiet_prefix
        .saturating_sub(ONSET_PREROLL_SAMPLES)
        .min(before_guard);
    if trim < MIN_TRIM_SAMPLES {
        0
    } else {
        trim
    }
}

fn first_active_sample(audio: &[f32]) -> Option<usize> {
    let rms = audio
        .chunks(WINDOW_SAMPLES)
        .map(|frame| {
            let energy = frame
                .iter()
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>();
            (energy / frame.len() as f64).sqrt()
        })
        .collect::<Vec<_>>();
    rms.windows(2)
        .position(|pair| pair[0] >= ACTIVE_RMS_THRESHOLD && pair[1] >= ACTIVE_RMS_THRESHOLD)
        .map(|index| index * WINDOW_SAMPLES)
}

fn compare_after_prefix_trim(control: &[f32], candidate: &[f32], trim: usize) -> (bool, f64) {
    if trim > control.len() || candidate.len() != control.len().saturating_sub(trim) {
        return (false, f64::INFINITY);
    }
    let maximum = control[trim..]
        .iter()
        .zip(candidate)
        .map(|(a, b)| f64::from((*a - *b).abs()))
        .fold(0.0, f64::max);
    (maximum <= 1e-6, maximum)
}

fn simulate_buffer(callbacks: &[CallbackRecord], buffer_ms: usize, terminal_ms: f64) -> Value {
    let buffer_samples = SAMPLE_RATE * buffer_ms / 1000;
    let mut queued_samples = 0usize;
    let mut start_index = None;
    for (index, callback) in callbacks.iter().enumerate() {
        queued_samples += callback.sample_count;
        if queued_samples >= buffer_samples {
            start_index = Some(index);
            break;
        }
    }
    let Some(start_index) = start_index else {
        return json!({"buffer_ms":buffer_ms,"playback_started":false,"minimum_queue_slack_ms":null,"underrun_count":null});
    };
    let playback_start_ms = callbacks[start_index].at_ms;
    queued_samples = callbacks
        .iter()
        .take(start_index + 1)
        .map(|callback| callback.sample_count)
        .sum();
    let mut cursor_ms = playback_start_ms;
    let mut minimum_slack_ms = queued_samples as f64 * 1000.0 / SAMPLE_RATE as f64;
    let mut underrun_count = 0usize;
    let mut underrun_ms = 0.0;
    for callback in callbacks.iter().skip(start_index + 1) {
        let elapsed_ms = (callback.at_ms - cursor_ms).max(0.0);
        let slack_samples = queued_samples as f64 - elapsed_ms * SAMPLE_RATE as f64 / 1000.0;
        minimum_slack_ms = minimum_slack_ms.min(slack_samples * 1000.0 / SAMPLE_RATE as f64);
        if slack_samples < 0.0 {
            underrun_count += 1;
            underrun_ms += -slack_samples * 1000.0 / SAMPLE_RATE as f64;
            queued_samples = 0;
        } else {
            queued_samples = slack_samples as usize;
        }
        queued_samples += callback.sample_count;
        cursor_ms = callback.at_ms;
        minimum_slack_ms =
            minimum_slack_ms.min(queued_samples as f64 * 1000.0 / SAMPLE_RATE as f64);
    }
    let terminal_gap = (terminal_ms - cursor_ms).max(0.0);
    let terminal_slack = queued_samples as f64 - terminal_gap * SAMPLE_RATE as f64 / 1000.0;
    minimum_slack_ms = minimum_slack_ms.min(terminal_slack * 1000.0 / SAMPLE_RATE as f64);
    if terminal_slack < 0.0 {
        underrun_count += 1;
        underrun_ms += -terminal_slack * 1000.0 / SAMPLE_RATE as f64;
    }
    json!({
        "buffer_ms":buffer_ms,
        "playback_started":true,
        "playback_start_ms":playback_start_ms,
        "minimum_queue_slack_ms":minimum_slack_ms,
        "underrun_count":underrun_count,
        "underrun_ms":underrun_ms,
        "queued_samples_after_last_callback":queued_samples
    })
}

fn summarize(rows: &[Value]) -> Value {
    let p50p95 = |key: &str| {
        let mut values = rows
            .iter()
            .filter_map(|row| row[key].as_f64())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        json!({"p50":percentile(&values,0.50),"p95":percentile(&values,0.95)})
    };
    let underruns = |buffer_ms: usize| {
        rows.iter()
            .filter(|row| {
                row["playback_buffer_simulation"]
                    .as_array()
                    .is_some_and(|items| {
                        items.iter().any(|item| {
                            item["buffer_ms"] == buffer_ms
                                && item["underrun_count"].as_u64().unwrap_or(0) > 0
                        })
                    })
            })
            .count()
    };
    let queue_slack = |buffer_ms: usize| {
        let mut values = rows
            .iter()
            .flat_map(|row| {
                row["playback_buffer_simulation"]
                    .as_array()
                    .into_iter()
                    .flatten()
            })
            .filter(|item| item["buffer_ms"] == buffer_ms)
            .filter_map(|item| item["minimum_queue_slack_ms"].as_f64())
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        json!({"p50":percentile(&values,0.50),"p05":percentile(&values,0.05),"minimum":values.first().copied()})
    };
    json!({
        "samples":rows.len(),
        "completed":rows.iter().filter(|row| row["completed"] == true).count(),
        "ttfa_callback_ms":p50p95("ttfa_callback_ms"),
        "first_active_pcm_ms":p50p95("first_active_pcm_ms"),
        "elapsed_ms":p50p95("elapsed_ms"),
        "trimmed_ms":p50p95("trimmed_ms"),
        "rtf_original_audio_duration":p50p95("rtf_original_audio_duration"),
        "underrun_runs_32ms":underruns(32),
        "underrun_runs_64ms":underruns(64),
        "minimum_queue_slack_32ms":queue_slack(32),
        "minimum_queue_slack_64ms":queue_slack(64)
    })
}

fn summarize_all(rows: &[Value]) -> Vec<Value> {
    CORPUS
        .iter()
        .flat_map(|(group, _)| {
            STRATEGIES.iter().copied().map(move |strategy| {
                let selected = rows
                    .iter()
                    .filter(|row| row["group_id"] == *group && row["strategy"] == strategy.name())
                    .cloned()
                    .collect::<Vec<_>>();
                json!({"group_id":group,"strategy":strategy.name(),"metrics":summarize(&selected)})
            })
        })
        .collect()
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted.get(index).copied()
}

fn export_blind_package(
    output: &Path,
    tracks: Vec<(String, String, String, bool, Vec<f32>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let blind_dir = output.join("blind-listening");
    let tracks_dir = blind_dir.join("tracks");
    fs::create_dir_all(&tracks_dir)?;
    fs::write(
        blind_dir.join("README.txt"),
        "Anonymous R2.0 listening set. IDs do not encode group or strategy.\nCompare first onset, missing/cut consonants, pauses, continuity, timbre, and overall preference.\nThe strategy key is stored separately as blind-key.json.\n",
    )?;
    let mut ids = (1..=tracks.len())
        .map(|index| format!("A{index:02}"))
        .collect::<Vec<_>>();
    let mut random = 0x5ca1_ab1e_4e2b_9d37u64;
    for index in (1..ids.len()).rev() {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        ids.swap(index, (random as usize) % (index + 1));
    }
    let mut key = Vec::new();
    let mut ratings = vec![
        "track_id,transcript,strategy_complete,first_onset_1_to_5,cut_consonant_or_word,pauses_timbre,overall_preference,notes".to_owned(),
    ];
    for (index, (group, strategy, text, complete, audio)) in tracks.into_iter().enumerate() {
        let id = ids[index].clone();
        write_wav(&tracks_dir.join(format!("{id}.wav")), &audio)?;
        key.push(json!({"track_id":id,"group_id":group,"strategy":strategy,"text":text,"complete":complete}));
        ratings.push(format!("{},{},{},,,,,", id, csv_escape(&text), complete));
    }
    fs::write(blind_dir.join("track-ids.txt"), ids.join("\n") + "\n")?;
    fs::write(blind_dir.join("ratings.csv"), ratings.join("\n"))?;
    fs::write(
        output.join("blind-key.json"),
        serde_json::to_vec_pretty(&key)?,
    )?;
    Ok(())
}

fn csv_escape(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn write_wav(path: &Path, audio: &[f32]) -> Result<(), hound::Error> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE as u32,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in audio {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_only_continuous_strict_silence_and_keeps_preroll() {
        let mut audio = vec![0.0; SAMPLE_RATE / 5];
        audio.extend(vec![0.01; SAMPLE_RATE / 5]);
        let trim = strict_leading_silence_trim(&audio);
        assert_eq!(trim, SAMPLE_RATE / 5 - ONSET_PREROLL_SAMPLES);
        assert!(audio[..trim].iter().all(|sample| *sample == 0.0));
        assert!(trim >= MIN_TRIM_SAMPLES);
    }

    #[test]
    fn does_not_trim_a_low_energy_prefix_or_an_interior_pause() {
        let mut low_onset = vec![2e-6; SAMPLE_RATE / 10];
        low_onset.extend(vec![0.0; SAMPLE_RATE / 5]);
        low_onset.extend(vec![0.01; SAMPLE_RATE / 5]);
        assert_eq!(strict_leading_silence_trim(&low_onset), 0);

        let mut interior_pause = vec![0.01; SAMPLE_RATE / 10];
        interior_pause.extend(vec![0.0; SAMPLE_RATE / 5]);
        interior_pause.extend(vec![0.01; SAMPLE_RATE / 5]);
        assert_eq!(strict_leading_silence_trim(&interior_pause), 0);
    }

    #[test]
    fn requires_an_active_onset_and_minimum_quiet_prefix() {
        assert_eq!(strict_leading_silence_trim(&vec![0.0; SAMPLE_RATE]), 0);
        let mut short_silence = vec![0.0; SAMPLE_RATE / 20];
        short_silence.extend(vec![0.01; SAMPLE_RATE / 5]);
        assert_eq!(strict_leading_silence_trim(&short_silence), 0);
    }
}
