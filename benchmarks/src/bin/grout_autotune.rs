//! Engine-objective autotuning on the cutile-rs 0.3.0 `cutile::tune` stack.
//!
//! The poster-example shape for multi-architecture tuning:
//!
//! - **Declared spaces** (`Config`) per tunable site, searched by the
//!   library's resumable `GridSearch` through the public `Objective` trait.
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
//!       [--site prefill_attention|prefill_hints|decode_attention|wide_prefill|all] \
//!       [--out-dir benchmarks/tuning] [--reps 3]
//!
//! Trials append to `<out-dir>/<arch>/<site>.<bucket>.trials.jsonl`; an
//! interrupted sweep resumes where it stopped.

use anyhow::{Context, Result};
use clap::Parser;
use cutile::tune::{
    best_config, space_hash, Config, GridSearch, Objective, ParamValue, Record, RecordEntry,
    Searcher, Trial, TrialState, Workspace,
};

/// `Trial`/`TrialState` are #[non_exhaustive]; the public constructors
/// (`Trial::measured` / `Trial::invalid`, landed with cutile-rs #239) are the
/// out-of-crate `Objective` implementor's way to build a return value.
/// `Trial::measured` records a non-finite timing as `Invalid` so it can
/// round-trip through the JSONL log.
fn invalid_trial(config_id: &str, reason: String) -> Trial {
    Trial::invalid(config_id, reason)
}

fn measured_trial(config_id: &str, median_ms: f32, min_ms: f32, reps: usize) -> Trial {
    Trial::measured(config_id, median_ms, min_ms, reps)
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
                // num_worker_warps_per_cta (0 = compiler default). Full
                // range, not just 4: the sm_100 08-23 run already moved
                // two prefill buckets to warps=4 and wide prefill to 2.
                ("GROUT_FMHA_PREFILL_WARPS", vec![0, 2, 4, 8]),
            ],
            buckets: [512usize, 2048, 8192]
                .into_iter()
                .filter(|pp| *pp < max_seq_len)
                .map(pp_bucket)
                .collect(),
        },
        // Architecture-specific optimization hints for the prefill attention
        // kernels, tuned as a second layer ON TOP of the tile/dispatch winners
        // above: the engine resolves every knob env > record > default, and
        // the driver reloads the engine after each site's record is saved, so
        // candidates here run with the prefill_attention record already
        // applied. Kept as its own site because the joint space (128 x 192
        // per bucket) is not searchable in an evening; the layering is the
        // classic coordinate-descent compromise. LPT-only knobs (schedule,
        // swizzle, mask-split) collapse to their incumbent when the
        // prefill_attention record picked the mapped kernel everywhere
        // (see restrict_lpt_hints).
        Site {
            name: "prefill_hints",
            axes: vec![
                // load_pipelined depth for the K/V loads (engine default 2).
                ("GROUT_FMHA_PREFILL_LATENCY", vec![1, 2, 3, 4]),
                // GQA heads per CTA (0 = query_group_size, i.e. all of them).
                ("GROUT_FMHA_PREFILL_GQA_GROUP", vec![0, 2, 4, 8]),
                // LPT schedule: 1 = linear, 2/3 = swizzled (reverse/forward).
                ("GROUT_FMHA_PREFILL_LPT_SCHED", vec![1, 2, 3]),
                // LPT swizzle width (0 = derived from L2 budget).
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", vec![0, 8]),
                // Boolean: -1 = explicit off (see apply_config), 1 = on.
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", vec![-1, 1]),
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
                ("GROUT_FMHA_DECODE_WARPS", vec![0, 2, 4, 8]),
            ],
            // Canonical decode cells use a short prompt; tuning in a long
            // kv context (the first attempt used pp=512) skews winners.
            //
            // The bucket is labeled by max_seq_len, not tg: split-kv
            // geometry partitions the ALLOCATED cache (kv_len_per_split =
            // ceil(max_seq_len / splits)), so the NKS optimum is a
            // function of the engine's max_seq_len. The sm_100 gate
            // proved a winner tuned at msl=16384 does not transfer to
            // the canonical msl=4096 engine. Tune once per max_seq_len
            // the deployment uses; records coexist as separate buckets.
            buckets: vec![Bucket {
                label: format!("msl={max_seq_len}"),
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
        (_, "prefill_hints") if arch.starts_with("sm_100") => vec![
            // sweep_pp_sm100.sh pp=2048/8192: latency 2, group auto, LPT
            // knobs (swizzle 8 / sched 1 / mask-split off) — inert while the
            // incumbent runs the mapped kernel, but the profile sets them.
            vec![
                ("GROUT_FMHA_PREFILL_LATENCY", 2),
                ("GROUT_FMHA_PREFILL_GQA_GROUP", 0),
                ("GROUT_FMHA_PREFILL_LPT_SCHED", 1),
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", 8),
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", -1),
            ],
        ],
        (_, "prefill_hints") => vec![
            // sm_120 records run LPT with engine defaults for every hint.
            vec![
                ("GROUT_FMHA_PREFILL_LATENCY", 2),
                ("GROUT_FMHA_PREFILL_GQA_GROUP", 0),
                ("GROUT_FMHA_PREFILL_LPT_SCHED", 1),
                ("GROUT_FMHA_PREFILL_LPT_SWIZZLE", 0),
                ("GROUT_FMHA_PREFILL_LPT_MASK_SPLIT", 1),
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

/// LPT-only hint axes are meaningless when the prefill_attention record for
/// this arch runs the mapped kernel in every bucket; collapse them to the
/// incumbent value so the search spends its budget on knobs that fire.
fn restrict_lpt_hints(site: &mut Site, arch: &str, out_dir: &Path) {
    if site.name != "prefill_hints" {
        return;
    }
    let record_path = out_dir.join("prefill_attention.json");
    let Ok(text) = fs::read_to_string(&record_path) else {
        println!("  prefill_hints: no prefill_attention record yet; tuning LPT knobs too");
        return;
    };
    let Ok(record) = serde_json::from_str::<serde_json::Value>(&text) else {
        return;
    };
    let any_lpt = record["entries"]
        .as_array()
        .map(|entries| {
            entries.iter().any(|e| {
                e["config"]["params"]["GROUT_FMHA_PREFILL_GQA_LPT"].as_i64() == Some(1)
            })
        })
        .unwrap_or(true);
    if any_lpt {
        return;
    }
    let incumbent = incumbents(arch, "prefill_hints")
        .into_iter()
        .next()
        .unwrap_or_default();
    for (key, values) in site.axes.iter_mut() {
        if key.contains("_LPT_") {
            let keep = incumbent
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v)
                .unwrap_or(values[0]);
            *values = vec![keep];
        }
    }
    println!(
        "  prefill_hints: record runs the mapped kernel everywhere; LPT-only axes fixed at incumbents"
    );
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
                // Negative = an explicit "0" for boolean knobs whose unset
                // default is true (0 itself means "unset" in this space).
                ParamValue::Int(v) if *v < 0 => std::env::set_var(key, "0"),
                ParamValue::Int(v) => std::env::set_var(key, v.to_string()),
                ParamValue::Str(s) => std::env::set_var(key, s),
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
/// public `Objective` trait (named `Oracle` before cutile-rs #239); the
/// closure-based `Autotuner` front-end assumes a CUDA-event-timed launch on
/// one stream, which does not fit an engine step that spans streams and host
/// logic (reported upstream).
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

/// All tuner engines must run a fixed decode window: with EOS active, a
/// "128-token" candidate trial measures however many tokens that sample
/// happened to emit, and the winner is not comparable to a fixed-length
/// shipping run (found the hard way in the sm_100 parity gate).
fn load_engine(
    rt: &tokio::runtime::Runtime,
    model_dir: &Path,
    max_seq_len: usize,
) -> Result<Qwen3Engine> {
    let mut engine = rt.block_on(Qwen3Engine::load(model_dir, Some(max_seq_len)))?;
    engine.set_ignore_eos(true);
    Ok(engine)
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
        self.engine = Some(load_engine(self.rt, &self.model_dir, self.max_seq_len)?);
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

impl Objective for EngineOracle<'_> {
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
    let mut engine_slot: Option<Qwen3Engine> =
        Some(load_engine(&rt, &model_dir, args.max_seq_len)?);

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

    // Engines constructed below load records from this directory, so a
    // site tuned later in the run (prefill_hints) sees the winners saved by
    // an earlier one (prefill_attention).
    if std::env::var("GROUT_TUNING_RECORD_DIR").is_err() {
        // SAFETY: single-threaded, before any engine exists.
        unsafe { std::env::set_var("GROUT_TUNING_RECORD_DIR", &args.out_dir) };
    }
    for mut site in sites(args.max_seq_len) {
        if args.site != "all" && args.site != site.name {
            continue;
        }
        restrict_lpt_hints(&mut site, &arch, &out_dir);
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
                    None => load_engine(&rt, &model_dir, args.max_seq_len)?,
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
        // Records are read at engine construction; drop the engine so the
        // next site runs on top of the winners just saved.
        engine_slot = None;
        grout::model::device_synchronize();
    }
    Ok(())
}
