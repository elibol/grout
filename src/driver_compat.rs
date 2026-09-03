//! Source-level compatibility for the raw driver wrappers across cutile-rs
//! 0.3.0 and 0.3.1.
//!
//! In 0.3.0, `cuda_core::{malloc_async, memcpy_*_async, free_async}` returned
//! the bare value (`CUdeviceptr` / `()`) and panicked internally on a driver
//! error. In 0.3.1 they return `Result<_, DriverError>` so a failed device
//! allocation can be handled instead of aborting the process. Grout wants the
//! 0.3.1 behavior (propagate) but also needs to build against 0.3.0 for
//! side-by-side validation, so every call site goes through
//! [`DriverCall::driver_result`], which is the identity on a `Result` and
//! wraps a bare value in `Ok`.

use cuda_core::DriverError;

pub trait DriverCall<T> {
    fn driver_result(self) -> Result<T, DriverError>;
}

impl DriverCall<()> for () {
    fn driver_result(self) -> Result<(), DriverError> {
        Ok(())
    }
}

impl DriverCall<u64> for u64 {
    fn driver_result(self) -> Result<u64, DriverError> {
        Ok(self)
    }
}

impl<T> DriverCall<T> for Result<T, DriverError> {
    fn driver_result(self) -> Result<T, DriverError> {
        self
    }
}
