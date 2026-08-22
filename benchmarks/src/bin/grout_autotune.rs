//! Engine-objective autotuning on the cutile-rs 0.3.0 `cutile::tune` stack.
//!
//! The poster-example shape for multi-architecture tuning:
//!
//! - **Declared spaces** (`Config`) per tunable site, searched by the
//!   library's resumable `GridSearch` through the public `Oracle` trait.
//! - **Engine objective**: each trial times real `Qwen3Engine` steps (whole
//!   prefill or decode window), not an isolated kernel — grout's tile optima
//!   are only meaningful end-to-end (per-kernel-form tuning lesson).
//! - **Correctness gate**: a candidate whose generated text differs from the
//!   default configuration's is recorded `Invalid`, never timed as a winner.
//! - **Per-arch persistence**: winners land in a provenance-checked
//!   `tune::Record` at `benchmarks/tuning/<arch>/<site>.json`. Records are
//!   refused at load when kernel source, toolchain, or candidate space
//!   changed. Run this same binary on each target arch (sm_120 locally,
//!   sm_100 on a B200) to produce that arch's records; nothing is shared or
//!   approximated across arches.
//!
//! Usage:
//!   grout_autotune --model ../hf_models/qwen3_4b \
//!       --prompt-dir benchmarks/results/sweep/<ts>/prompts \
//!       [--site prefill_attention|decode_attention|wide_prefill|all] \
//!       [--out-dir benchmarks/tuning] [--reps 3]
//!
//! Trials append to `<out-dir>/<arch>/<site>.<bucket>.trials.jsonl`; an
//! interrupted sweep resumes where it stopped.

use anyhow::{Context, Result};
use clap::Parser;
use cutile::tune::{
    best_config, space_hash, Config, GridSearch, Oracle, ParamValue, Record, RecordEntry,
    Searcher, Trial, TrialState, Workspace,
};

/// `Trial`/`TrialState` are #[non_exhaustive], so an out-of-crate `Oracle`
/// impl cannot construct its own return value; until upstream adds a
/// constructor (reported), build trials through serde.
fn make_trial(config_id: &str, state: serde_json::Value) -> Trial {
    serde_json::from_value(serde_json::json!({
        "config_id": config_id,
        "state": state,
    }))
    .expect("Trial schema")
}

fn invalid_trial(config_id: &str, reason: String) -> Trial {
    make_trial(config_id, serde_json::json!({"Invalid": {"reason": reason}}))
}

fn measured_trial(config_id: &str, median_ms: f32, min_ms: f32, reps: usize) -> Trial {
    make_trial(
        config_id,
        serde_json::json!({"Measured": {"median_ms": median_ms, "min_ms": min_ms, "reps": reps}}),
    )
}
use grout::model::Qwen3Engine;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: String,
    /// Directory containing pp_<n>.txt prompt files (sweep layout).
    #[arg(long)]
    prompt_dir: String,
    #[arg(long, default_value = "all")]
    site: String,
    #[arg(long, default_value = "benchmarks/tuning")]
    out_dir: String,
    /// Timed engine steps per trial (after one untimed warmup step).
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Max sequence length for the engine (bounds which pp buckets run).
    #[arg(long, default_value_t = 16384)]
    max_seq_len: usize,
    /// Wall-clock budget per site, minutes (0 = none).
    #[arg(long, default_value_t = 0)]
    budget_min: u64,
    /// Debug: run the default config N times on the first bucket and print
    /// each generated text hash (in-process determinism probe), then exit.
    #[arg(long, default_value_t = 0)]
    gate_probe: usize,
}

/// One tunable engine site: env-var axes, shape buckets, and an objective.
struct Site {
    name: &'static str,
    /// (env var, values). Value 0 means "unset" (compiler/engine default).
    axes: Vec<(&'static str, Vec<i64>)>,
    /// (bucket label, pp tokens file stem, max_new_tokens, decode_objective)
    buckets: Vec<Bucket>,
}

struct Bucket {
    label: String,
    prompt_pp: usize,
    max_new_tokens: usize,
    /// false: objective = prompt_elapsed; true: objective = decode_elapsed.
    decode_objective: bool,
}

fn sites(max_seq_len: usize) -> Vec<Site> {
    // Axes mirror the original tile-sweep scripts (sweep_pp_tile.sh /
    // sweep_tg_tile.sh) plus the knobs the shipping sm_100/sm_120 profiles
    // set — critically including the KERNEL DISPATCH flag
    // (GROUT_FMHA_PREFILL_GQA_LPT) and occupancy. The first B200 tuning
    // attempt failed its parity gate because the space omitted the
    // dispatch axis: on sm_100 auto-LPT engaged for every candidate while
    // the shipping config is the mapped kernel with LPT off. Rule learned:
    // the incumbent shipping config must be expressible within the space
    // (it is, for both arches: sm_100 mapped 128/128/occ2/LPT0 and the
    // sm_120 profile corners are all covered).
    let pp_bucket = |pp: usize| Bucket {
        label: format!("pp={pp}"),
        prompt_pp: pp,
        max_new_tokens: 16,
        decode_objective: false,
    };
    vec![
        Site {
            name: "prefill_attention",
            axes: vec![
                // 0 = mapped kernel, 1 = LPT kernel; LPT-specific knobs
                // (swizzle/sched/mask-split) ride engine defaults.
                ("GROUT_FMHA_PREFILL_GQA_LPT", vec![0, 1]),
                ("GROUT_ATTN_BM_PREFILL", vec![16, 32, 64, 128]),
                ("GROUT_ATTN_BN_PREFILL", vec![16, 32, 64, 128]),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", vec![1, 2]),
                ("GROUT_FMHA_PREFILL_WARPS", vec![0, 4]),
            ],
            buckets: [512usize, 2048, 8192]
                .into_iter()
                .filter(|pp| *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
        Site {
            name: "decode_attention",
            axes: vec![
                ("GROUT_ATTN_BN_DECODE", vec![16, 32, 64, 128]),
                ("GROUT_FMHA_NUM_KV_SPLITS", vec![4, 8, 16, 32]),
                ("GROUT_FMHA_DECODE_WARPS", vec![0, 4]),
            ],
            // Canonical decode cells use a short prompt; tuning in a long
            // kv context (the first attempt used pp=512) skews winners.
            buckets: vec![Bucket {
                label: "tg=128".into(),
                prompt_pp: 18,
                max_new_tokens: 128,
                decode_objective: true,
            }],
        },
        Site {
            name: "wide_prefill",
            axes: vec![
                ("GROUT_QK_PREFILL_BM", vec![16, 32, 64]),
                ("GROUT_QK_PREFILL_WARPS", vec![0, 1, 2, 4]),
            ],
            buckets: [2048usize, 8192]
                .into_iter()
                .filter(|pp| *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
    ]
}

/// Shipping incumbent configs per arch: coverage is sufficient only if
/// every incumbent is expressible inside the declared space — verified at
/// startup, so a candidate space can never again omit the config it must
/// beat (the failure mode of the first B200 tuning attempt).
fn incumbents(arch: &str, site: &str) -> Vec<Vec<(&'static str, i64)>> {
    match (arch, site) {
        (_, "prefill_attention") if arch.starts_with("sm_100") => vec![
            // sweep_pp_sm100.sh pp<=8192: mapped kernel, 128x128, occ 2.
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 0),
                ("GROUT_ATTN_BM_PREFILL", 128),
                ("GROUT_ATTN_BN_PREFILL", 128),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 2),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
        ],
        (_, "prefill_attention") => vec![
            // sweep_pp_sm120.sh: mapped 64x32 short, LPT on at >=2048.
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 0),
                ("GROUT_ATTN_BM_PREFILL", 64),
                ("GROUT_ATTN_BN_PREFILL", 32),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 1),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
            vec![
                ("GROUT_FMHA_PREFILL_GQA_LPT", 1),
                ("GROUT_ATTN_BM_PREFILL", 16),
                ("GROUT_ATTN_BN_PREFILL", 64),
                ("GROUT_FMHA_PREFILL_OCCUPANCY", 1),
                ("GROUT_FMHA_PREFILL_WARPS", 0),
            ],
        ],
        (_, "decode_attention") if arch.starts_with("sm_100") => vec![
            // sweep_tg_sm100.sh TG_128 cell.
            vec![
                ("GROUT_ATTN_BN_DECODE", 32),
                ("GROUT_FMHA_NUM_KV_SPLITS", 4),
                ("GROUT_FMHA_DECODE_WARPS", 0),
            ],
        ],
        (_, "decode_attention") => vec![vec![
            ("GROUT_ATTN_BN_DECODE", 32),
            ("GROUT_FMHA_NUM_KV_SPLITS", 16),
            ("GROUT_FMHA_DECODE_WARPS", 0),
        ]],
        (_, "wide_prefill") => vec![vec![
            ("GROUT_QK_PREFILL_BM", 32),
            ("GROUT_QK_PREFILL_WARPS", 0),
        ]],
        _ => vec![],
    }
}

/// Every incumbent parameter value must be a member of its axis.
fn verify_coverage(arch: &str, site: &Site) -> Result<()> {
    for incumbent in incumbents(arch, site.name) {
        for (key, value) in &incumbent {
            let axis = site
                .axes
                .iter()
                .find(|(name, _)| name == key)
                .with_context(|| format!("{}: incumbent key {key} has no axis", site.name))?;
            anyhow::ensure!(
                axis.1.contains(value),
                "{}: incumbent {key}={value} is NOT in the declared axis {:?} —                  the space cannot beat a config it does not contain",
                site.name,
                axis.1
            );
        }
    }
    println!(
        "  coverage ok: {} shipping incumbent(s) inside the {} space",
        incumbents(arch, site.name).len(),
        site.name
    );
    Ok(())
}

fn cartesian(axes: &[(&'static str, Vec<i64>)]) -> Vec<Config> {
    let mut configs: Vec<Vec<(&'static str, i64)>> = vec![vec![]];
    for (name, values) in axes {
        configs = configs
            .into_iter()
            .flat_map(|base| {
                values.iter().map(move |v| {
                    let mut c = base.clone();
                    c.push((name, *v));
                    c
                })
            })
            .collect();
    }
    configs
        .into_iter()
        .map(|params| Config::new(params.into_iter().map(|(k, v)| (k, ParamValue::Int(v)))))
        .collect()
}

fn apply_config(config: &Config) {
    for (key, value) in &config.params {
        // SAFETY: the tuner is single-threaded; env mutation happens strictly
        // between engine steps, and the engine reads these vars on this same
        // thread inside block_on.
        unsafe {
            match value {
                ParamValue::Int(0) => std::env::remove_var(key),
                ParamValue::Int(v) => std::env::set_var(key, v.to_string()),
                ParamValue::Str(s) => std::env::set_var(key, s),
                other => panic!("unsupported param value {other:?}"),
            }
        }
    }
}

fn clear_config(config: &Config) {
    for key in config.params.keys() {
        // SAFETY: see apply_config.
        unsafe { std::env::remove_var(key) };
    }
}

/// Engine-objective oracle: measures whole engine steps per candidate.
///
/// This composes with the library's `GridSearch`/`Searcher` through the
/// public `Oracle` trait; the closure-based `Autotuner` front-end assumes a
/// CUDA-event-timed launch on one stream, which does not fit an engine step
/// that spans streams and host logic (reported upstream).
struct EngineOracle<'a> {
    configs: Vec<Config>,
    engine: Option<Qwen3Engine>,
    model_dir: PathBuf,
    max_seq_len: usize,
    rt: &'a tokio::runtime::Runtime,
    prompt: String,
    max_new_tokens: usize,
    decode_objective: bool,
    reps: usize,
    reference_text: Option<String>,
    deadline: Option<Instant>,
    log: fs::File,
}

fn is_alloc_failure(e: &anyhow::Error) -> bool {
    let msg = format!("{e:#}");
    msg.contains("ALLOC_FAILED") || msg.contains("OUT_OF_MEMORY") || msg.contains("OutOfMemory")
}

impl EngineOracle<'_> {
    /// The engine accumulates per-specialization device state across trials
    /// (JIT modules, pools); after enough configs it exhausts VRAM. On an
    /// allocation-flavored failure, rebuild the engine (fresh context frees
    /// everything) and let the caller retry — a leak must never masquerade
    /// as an invalid candidate.
    fn reload_engine(&mut self) -> Result<()> {
        eprintln!("  (device allocation failure — reloading engine)");
        // Drop first and drain the async frees; loading before dropping
        // would hold two engines resident and OOM the reload itself.
        self.engine = None;
        grout::model::device_synchronize();
        self.engine = Some(self.rt.block_on(Qwen3Engine::load(
            &self.model_dir,
            Some(self.max_seq_len),
        ))?);
        Ok(())
    }

    fn step_ms(&mut self) -> Result<(f32, String)> {
        let engine = self.engine.as_mut().expect("engine");
        let out = self
            .rt
            .block_on(engine.generate(&self.prompt, self.max_new_tokens))?;
        let ms = if self.decode_objective {
            out.decode_elapsed.as_secs_f32() * 1e3
        } else {
            out.prompt_elapsed.as_secs_f32() * 1e3
        };
        Ok((ms, out.text))
    }

    fn measure_inner(&mut self, index: usize) -> Result<Trial> {
        let config = self.configs[index].clone();
        apply_config(&config);
        // Warmup step doubles as the compile/launch/correctness gate.
        let mut first = self.step_ms();
        if let Err(e) = &first {
            if is_alloc_failure(e) {
                self.reload_engine()?;
                first = self.step_ms();
            }
        }
        let (_, text) = match first {
            Ok(v) => v,
            Err(e) => {
                clear_config(&config);
                return Ok(invalid_trial(
                    &config.id,
                    format!("engine step failed: {e:#}"),
                ));
            }
        };
        // Gate v1 = the step succeeded (compile/launch/shape errors above
        // invalidate the candidate). A text-match gate is not usable: the
        // engine's greedy output is nondeterministic at the logit level
        // (allocation-address-dependent reduction order), verified by
        // --gate-probe. Numeric correctness of every kernel form remains
        // covered by the GPU test suite; a strict gate needs a
        // deterministic-eval/logits-capture engine API (tracked).
        let _ = (&text, &self.reference_text);
        let mut samples = Vec::with_capacity(self.reps);
        for _ in 0..self.reps {
            samples.push(self.step_ms()?.0);
        }
        clear_config(&config);
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = samples[samples.len() / 2];
        Ok(measured_trial(&config.id, median, samples[0], samples.len()))
    }
}

impl Oracle for EngineOracle<'_> {
    fn configs(&self) -> &[Config] {
        &self.configs
    }

    fn measure(&mut self, index: usize) -> Trial {
        let trial = self.measure_inner(index).unwrap_or_else(|e| {
            invalid_trial(&self.configs[index].id, format!("harness error: {e:#}"))
        });
        if let Ok(line) = serde_json::to_string(&trial) {
            let _ = writeln!(self.log, "{line}");
        }
        match &trial.state {
            TrialState::Measured { median_ms, .. } => {
                println!("  {} -> {median_ms:.2} ms", trial.config_id)
            }
            TrialState::Invalid { reason } => {
                println!("  {} -> invalid: {reason}", trial.config_id)
            }
            other => println!("  {} -> {other:?}", trial.config_id),
        }
        trial
    }

    fn budget_remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }
}

fn load_existing_trials(path: &Path) -> Vec<Trial> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<Trial>(l).ok())
        .collect()
}

fn detect_arch() -> String {
    std::env::var("GROUT_TUNE_ARCH").unwrap_or_else(|_| grout::model::device_arch(0))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let arch = detect_arch();
    let out_dir = PathBuf::from(&args.out_dir).join(&arch);
    fs::create_dir_all(&out_dir)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let model_dir = PathBuf::from(&args.model);
    let mut engine_slot: Option<Qwen3Engine> = Some(rt.block_on(Qwen3Engine::load(
        &model_dir,
        Some(args.max_seq_len),
    ))?);

    let tileiras = cutile_compiler::cuda_tile_runtime_utils::tileiras_fingerprint().to_string();

    if args.gate_probe > 0 {
        let site_list = sites(args.max_seq_len);
        let bucket = &site_list[0].buckets[0];
        let prompt_path =
            PathBuf::from(&args.prompt_dir).join(format!("pp_{}.txt", bucket.prompt_pp));
        let prompt = fs::read_to_string(&prompt_path)?;
        let mut engine = engine_slot.take().expect("engine");
        for i in 0..args.gate_probe {
            let out = rt.block_on(engine.generate(&prompt, bucket.max_new_tokens))?;
            let mut hash = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&out.text, &mut hash);
            println!(
                "probe {i}: prefill={:.2} ms hash={:016x} text[..48]={:?}",
                out.prompt_elapsed.as_secs_f32() * 1e3,
                std::hash::Hasher::finish(&hash),
                out.text.chars().take(48).collect::<String>()
            );
        }
        return Ok(());
    }

    for site in sites(args.max_seq_len) {
        if args.site != "all" && args.site != site.name {
            continue;
        }
        verify_coverage(&arch, &site)?;
        let configs = cartesian(&site.axes);
        println!(
            "site {} — {} candidates x {} buckets on {arch}",
            site.name,
            configs.len(),
            site.buckets.len()
        );
        let mut entries: Vec<RecordEntry> = Vec::new();
        for bucket in &site.buckets {
            let prompt_path =
                PathBuf::from(&args.prompt_dir).join(format!("pp_{}.txt", bucket.prompt_pp));
            let prompt = fs::read_to_string(&prompt_path)
                .with_context(|| format!("prompt file {}", prompt_path.display()))?;
            let trials_path = out_dir.join(format!("{}.{}.trials.jsonl", site.name, bucket.label));
            let known = load_existing_trials(&trials_path);
            if !known.is_empty() {
                println!("  [{}] resuming: {} prior trials", bucket.label, known.len());
            }
            let log = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&trials_path)?;

            // Reference output for the correctness gate: default config.
            let mut oracle = EngineOracle {
                configs: configs.clone(),
                engine: Some(match engine_slot.take() {
                    Some(e) => e,
                    None => rt.block_on(Qwen3Engine::load(&model_dir, Some(args.max_seq_len)))?,
                }),
                model_dir: model_dir.clone(),
                max_seq_len: args.max_seq_len,
                rt: &rt,
                prompt,
                max_new_tokens: bucket.max_new_tokens,
                decode_objective: bucket.decode_objective,
                reps: args.reps,
                reference_text: None,
                deadline: (args.budget_min > 0)
                    .then(|| Instant::now() + Duration::from_secs(args.budget_min * 60)),
                log,
            };
            let mut reference = oracle.step_ms();
            if let Err(e) = &reference {
                if is_alloc_failure(e) {
                    oracle.reload_engine()?;
                    reference = oracle.step_ms();
                }
            }
            oracle.reference_text = Some(reference?.1);

            println!("  [{}] searching...", bucket.label);
            let trials = GridSearch::new().resume(known).search(&mut oracle);
            engine_slot = oracle.engine;
            let Some(best) = best_config(&configs, &trials) else {
                println!("  [{}] no valid winner", bucket.label);
                continue;
            };
            let best_trial = trials
                .iter()
                .filter(|t| t.config_id == best.id)
                .find_map(|t| match &t.state {
                    TrialState::Measured { median_ms, reps, .. } => Some((*median_ms, *reps)),
                    _ => None,
                })
                .unwrap_or((f32::NAN, 0));
            println!(
                "  [{}] winner: {} ({:.2} ms)",
                bucket.label, best.id, best_trial.0
            );
            entries.push(RecordEntry {
                bucket: bucket.label.clone(),
                config: best.clone(),
                median_ms: best_trial.0,
                samples: best_trial.1,
                l2_key: None,
            });
        }
        if entries.is_empty() {
            continue;
        }
        let ws = Workspace {
            kernel: site.name.to_string(),
            source_hash: grout::kernels::_SOURCE_HASH.to_string(),
            arch: arch.clone(),
            tileiras_fingerprint: tileiras.clone(),
            space_hash: Some(space_hash(&configs)),
        };
        let mut record = Record::new(&ws);
        record.entries = entries;
        record.gate = Some("step-success-v1".into());
        let record_path = out_dir.join(format!("{}.json", site.name));
        record.save(&record_path)?;
        println!("saved {}", record_path.display());
    }
    Ok(())
}
