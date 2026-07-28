use anyhow::Result;
use cuda_async::device_operation::{DeviceOp, value};
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::core::f16;
use cutile::tensor::{IntoPartition, ToHostVec};
use cutile::tile_kernel::TileKernel;
use grout::kernels::add_2d_f16;
use std::sync::Arc;

#[test]
fn add_2d_kernel_compiles_and_executes() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        Ok(_) => {
            eprintln!("skipping CUDA kernel integration test: no CUDA devices found");
            return Ok(());
        }
        Err(err) => {
            eprintln!("skipping CUDA kernel integration test: CUDA unavailable: {err:?}");
            return Ok(());
        }
    }

    const BLOCK: usize = 4;

    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    let lhs_host = Arc::new(vec![
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ]);
    let rhs_host = Arc::new(vec![
        f16::from_f32(10.0),
        f16::from_f32(20.0),
        f16::from_f32(30.0),
        f16::from_f32(40.0),
    ]);

    let lhs = Arc::new(
        api::copy_host_vec_to_device(&lhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let rhs = Arc::new(
        api::copy_host_vec_to_device(&rhs_host)
            .reshape(&[1, BLOCK])
            .sync_on(&stream)?,
    );
    let out = api::zeros::<f16>(&[1, BLOCK]).sync_on(&stream)?;

    let result = add_2d_f16(value(out.partition([1, BLOCK])), value(lhs), value(rhs))
        .generics(vec![BLOCK.to_string()])
        .sync_on(&stream)?;
    let out = result.0.unpartition();
    let actual = out.to_host_vec().sync_on(&stream)?;

    let actual: Vec<f32> = actual.into_iter().map(|x| x.to_f32()).collect();
    assert_eq!(actual, vec![11.0, 22.0, 33.0, 44.0]);
    Ok(())
}

#[test]
fn add_rms_norm_decode_bounded_matches_raw() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => {
            eprintln!("skipping: no CUDA device");
            return Ok(());
        }
    }
    use grout::kernels::{add_rms_norm_decode_bounded_f16, add_rms_norm_decode_raw_f16};

    const N: usize = 2560;
    const BS: usize = 512;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;

    // Deterministic pseudo-random inputs spanning sign/magnitude range.
    let gen_vals = |seed: u32| -> Arc<Vec<f16>> {
        let mut v = Vec::with_capacity(N);
        let mut x = seed;
        for _ in 0..N {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(f16::from_f32(((x >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0));
        }
        Arc::new(v)
    };
    let residual = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(1))
            .reshape(&[1, N])
            .sync_on(&stream)?,
    );
    let x = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(2))
            .reshape(&[1, N])
            .sync_on(&stream)?,
    );
    let w = Arc::new(
        api::copy_host_vec_to_device(&gen_vals(3))
            .reshape(&[N])
            .sync_on(&stream)?,
    );
    let eps = 1e-6f32;

    // Raw reference.
    let out_raw = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    let res_raw = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    unsafe {
        add_rms_norm_decode_raw_f16(
            residual.device_pointer().clone(),
            x.device_pointer().clone(),
            w.device_pointer().clone(),
            out_raw.device_pointer().clone(),
            res_raw.device_pointer().clone(),
            eps,
        )
    }
    .generics(vec![N.to_string(), BS.to_string()])
    .grid((1u32, 1u32, 1u32))
    .sync_on(&stream)?;

    // Bounded kernel.
    let out_b = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    let res_b = api::zeros::<f16>(&[1, N]).sync_on(&stream)?;
    let result = add_rms_norm_decode_bounded_f16(
        &residual,
        &x,
        &w,
        value(out_b.partition([1usize, N])),
        value(res_b.partition([1usize, N])),
        eps,
    )
    .generics(vec![N.to_string(), BS.to_string()])
    .grid((1u32, 1u32, 1u32))
    .sync_on(&stream)?;
    let out_b = result.3.unpartition();
    let res_b = result.4.unpartition();

    let or = out_raw.to_host_vec().sync_on(&stream)?;
    let ob = out_b.to_host_vec().sync_on(&stream)?;
    let rr = res_raw.to_host_vec().sync_on(&stream)?;
    let rb = res_b.to_host_vec().sync_on(&stream)?;
    let mut bad = 0;
    for i in 0..N {
        if or[i].to_f32() != ob[i].to_f32() || rr[i].to_f32() != rb[i].to_f32() {
            if bad < 5 {
                eprintln!(
                    "mismatch @{i}: out raw={} bounded={} | res raw={} bounded={}",
                    or[i].to_f32(),
                    ob[i].to_f32(),
                    rr[i].to_f32(),
                    rb[i].to_f32()
                );
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{N} elements differ");
    Ok(())
}

/// Acceptance spec for cutile-rs owned-axis row branding: the fully safe
/// row-wise bounded kernel must JIT clean once grid-axis branding lands.
/// Today it must FAIL to JIT (unbranded `get_tile_block_id` row coordinate).
/// Run with `cargo test -- --ignored` to see the current error.
#[test]
#[ignore = "acceptance spec: expected to fail JIT until cutile-rs row branding lands"]
fn rowwise_bounded_spec_jit_error() -> Result<()> {
    match Device::device_count() {
        Ok(count) if count > 0 => {}
        _ => return Ok(()),
    }
    use grout::kernels::add_rms_norm_rows_bounded_spec_f16;
    const N: usize = 2560;
    const BS: usize = 512;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let z = |shape: &[usize]| api::zeros::<f16>(shape);
    let residual = Arc::new(z(&[4, N]).sync_on(&stream)?);
    let x = Arc::new(z(&[4, N]).sync_on(&stream)?);
    let w = Arc::new(z(&[N]).sync_on(&stream)?);
    let out = z(&[4, N]).sync_on(&stream)?;
    let r = add_rms_norm_rows_bounded_spec_f16(
        &residual,
        &x,
        &w,
        value(out.partition([1usize, N])),
        1e-6f32,
    )
    .generics(vec![N.to_string(), BS.to_string()])
    .grid((4u32, 1u32, 1u32))
    .sync_on(&stream);
    match r {
        Ok(_) => println!("SPEC NOW PASSES: owned-axis row branding has landed"),
        Err(e) => println!("SPEC JIT ERROR (expected today): {e:?}"),
    }
    Ok(())
}
