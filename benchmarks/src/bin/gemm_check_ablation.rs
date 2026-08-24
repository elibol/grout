//! Bounds-check cost ablation on the persistent GEMM (paper exp4).
//!
//! Three arms of the same kernel body over square f16 GEMM (M=N=K):
//!   checked    — shipped gemm_persistent_f16 (deny_in_kernel_checks; every
//!                check discharged or launch-validated; zero in kernel)
//!   forced     — gemm_persistent_nodeny_f16 under
//!                CUTILE_FORCE_DEVICE_CHECKS=1 (set by the harness): every
//!                access checked in the kernel, per k-iteration
//!   unchecked  — gemm_persistent_unchecked_f16 (exact body, no checks)
//!
//! GEMM is the sharpest probe for check-execution cost: the inner loop is
//! load/load/mma with none of the ALU slack that hides checks in attention,
//! and check executions scale with K/BK per tile. Samples are paired and
//! order-alternating; outputs are cross-validated between arms first.

use anyhow::{anyhow, ensure, Context, Result};
use clap::Parser;
use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::kernels::{
    gemm_persistent_f16, gemm_persistent_nodeny_f16, gemm_persistent_unchecked_f16,
};
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Args {
    /// Square problem sizes, comma-separated (M=N=K).
    #[arg(long, default_value = "2048,4096,8192")]
    sizes: String,
    #[arg(long, default_value_t = 128)]
    bm: usize,
    #[arg(long, default_value_t = 128)]
    bn: usize,
    #[arg(long, default_value_t = 64)]
    bk: usize,
    /// Paired samples per arm (order alternates every sample).
    #[arg(long, default_value_t = 7)]
    samples: usize,
    /// Launches inside each CUDA-event window.
    #[arg(long, default_value_t = 5)]
    iters: usize,
    #[arg(long, default_value_t = 3)]
    warmup_iters: usize,
    /// Also print one raw row per (sample, arm): sample,<n>,<arm>,<idx>,<us>
    #[arg(long, default_value_t = false)]
    emit_samples: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Checked,
    Forced,
    Unchecked,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Checked => "checked",
            Arm::Forced => "forced",
            Arm::Unchecked => "unchecked",
        }
    }
}

fn launch(
    arm: Arm,
    stream: &Arc<Stream>,
    n: usize,
    args: &Args,
    grid_x: u32,
    x: &Arc<Tensor<f16>>,
    y: &Arc<Tensor<f16>>,
    z: &mut Tensor<f16>,
) -> Result<()> {
    let generics = vec![
        args.bm.to_string(),
        args.bn.to_string(),
        args.bk.to_string(),
        "8".to_string(),
        "1".to_string(),
    ];
    let _ = n;
    let mapped_z =
        cutile::tensor::PartitionMut::partition(z, [args.bm, args.bn]).map([8, 1], grid_x);
    match arm {
        // SAFETY: async_on is the unsafe launch API; buffers outlive the
        // stream sync that follows every timing window.
        Arm::Checked => unsafe {
            gemm_persistent_f16(mapped_z, &**x, &**y)
                .generics(generics)
                .async_on(stream)
        }
        .map_err(|e| anyhow!("checked launch failed: {e:?}"))?,
        Arm::Forced => unsafe {
            gemm_persistent_nodeny_f16(mapped_z, &**x, &**y)
                .generics(generics)
                .async_on(stream)
        }
        .map_err(|e| anyhow!("forced launch failed: {e:?}"))?,
        Arm::Unchecked => unsafe {
            gemm_persistent_unchecked_f16(mapped_z, &**x, &**y)
                .generics(generics)
                .async_on(stream)
        }
        .map_err(|e| anyhow!("unchecked launch failed: {e:?}"))?,
    };
    Ok(())
}

fn time_arm(
    arm: Arm,
    stream: &Arc<Stream>,
    n: usize,
    args: &Args,
    grid_x: u32,
    x: &Arc<Tensor<f16>>,
    y: &Arc<Tensor<f16>>,
    z: &mut Tensor<f16>,
) -> Result<f64> {
    for _ in 0..args.warmup_iters {
        launch(arm, stream, n, args, grid_x, x, y, z)?;
    }
    unsafe { stream.synchronize() }.map_err(|e| anyhow!("sync: {e:?}"))?;
    let start = std::time::Instant::now();
    for _ in 0..args.iters {
        launch(arm, stream, n, args, grid_x, x, y, z)?;
    }
    unsafe { stream.synchronize() }.map_err(|e| anyhow!("sync: {e:?}"))?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / args.iters as f64)
}

fn quartiles(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let q = |f: f64| values[((values.len() - 1) as f64 * f).round() as usize];
    (q(0.25), q(0.5), q(0.75))
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        std::env::var("CUTILE_FORCE_DEVICE_CHECKS").is_err(),
        "unset CUTILE_FORCE_DEVICE_CHECKS; the harness sets it per-arm"
    );
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let sizes: Vec<usize> = args
        .sizes
        .split(',')
        .map(|s| s.trim().parse::<usize>().context("bad --sizes"))
        .collect::<Result<_>>()?;

    println!("n,arm,p25_us,median_us,p75_us,samples,iters,bm,bn,bk");
    for &n in &sizes {
        let host_random = |seed: u32| -> Arc<Vec<f16>> {
            let mut v = Vec::with_capacity(n * n);
            let mut st = seed;
            for _ in 0..n * n {
                st = st.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push(f16::from_f32(
                    ((st >> 8) as f32 / (1u32 << 24) as f32) - 0.5,
                ));
            }
            Arc::new(v)
        };
        let x: Arc<Tensor<f16>> = Arc::new(
            api::copy_host_vec_to_device(&host_random(1))
                .reshape(&[n, n])
                .sync_on(&stream)
                .map_err(|e| anyhow!("alloc x: {e:?}"))?,
        );
        let y: Arc<Tensor<f16>> = Arc::new(
            api::copy_host_vec_to_device(&host_random(2))
                .reshape(&[n, n])
                .sync_on(&stream)
                .map_err(|e| anyhow!("alloc y: {e:?}"))?,
        );
        let total_tiles = (n / args.bm) * (n / args.bn);
        let grid_x = {
            // persistent grid: min(total tile groups, SMs/2); tiles are
            // consumed 8 per CTA step via MAP_SHAPE [8, 1].
            let groups = total_tiles.div_ceil(8);
            (groups.min(128).max(1)) as u32
        };

        // Correctness gate: all three arms must agree bitwise on the same
        // inputs before anything is timed. Forced runs in a child process
        // env: the env var must be process-wide at JIT time, so the forced
        // arm is compiled in THIS process by setting the var around its
        // first (gate) launch only — JIT happens once, at first use.
        let mut outs: Vec<(Arm, Vec<f16>)> = Vec::new();
        for arm in [Arm::Checked, Arm::Forced, Arm::Unchecked] {
            let mut z = api::zeros::<f16>(&[n, n])
                .sync_on(&stream)
                .map_err(|e| anyhow!("alloc z: {e:?}"))?;
            if arm == Arm::Forced {
                // SAFETY: single-threaded harness.
                unsafe { std::env::set_var("CUTILE_FORCE_DEVICE_CHECKS", "1") };
            }
            launch(arm, &stream, n, &args, grid_x, &x, &y, &mut z)?;
            unsafe { stream.synchronize() }.map_err(|e| anyhow!("sync: {e:?}"))?;
            if arm == Arm::Forced {
                // SAFETY: single-threaded harness.
                unsafe { std::env::remove_var("CUTILE_FORCE_DEVICE_CHECKS") };
            }
            outs.push((arm, z.to_host_vec().sync_on(&stream)?));
        }
        let reference = &outs[0].1;
        for (arm, out) in &outs[1..] {
            let bad = reference
                .iter()
                .zip(out.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                bad == 0,
                "{} arm diverges from checked on {bad}/{} elements at n={n}",
                arm.name(),
                reference.len()
            );
        }
        eprintln!("n={n}: all arms bitwise-identical; timing...");

        let mut z = api::zeros::<f16>(&[n, n])
            .sync_on(&stream)
            .map_err(|e| anyhow!("alloc z: {e:?}"))?;
        let mut med: Vec<(Arm, Vec<f64>)> = [Arm::Checked, Arm::Forced, Arm::Unchecked]
            .into_iter()
            .map(|a| (a, Vec::new()))
            .collect();
        for s in 0..args.samples {
            // order alternates each sample
            let order: Vec<Arm> = if s % 2 == 0 {
                vec![Arm::Checked, Arm::Forced, Arm::Unchecked]
            } else {
                vec![Arm::Unchecked, Arm::Forced, Arm::Checked]
            };
            for arm in order {
                let t = time_arm(arm, &stream, n, &args, grid_x, &x, &y, &mut z)?;
                if args.emit_samples {
                    println!("sample,{n},{},{s},{t:.3}", arm.name());
                }
                med.iter_mut().find(|(a, _)| *a == arm).unwrap().1.push(t);
            }
        }
        for (arm, mut ts) in med {
            let (p25, p50, p75) = quartiles(&mut ts);
            println!(
                "{n},{},{p25:.3},{p50:.3},{p75:.3},{},{},{},{},{}",
                arm.name(),
                args.samples,
                args.iters,
                args.bm,
                args.bn,
                args.bk
            );
        }
    }
    Ok(())
}
