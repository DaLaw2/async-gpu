//! Stateless GPU operations — functional API wrapping GPU kernels.
//!
//! Each function takes [`GpuTensor`] inputs and returns a new [`GpuTensor`].
//! Launch configuration is handled automatically via [`KernelRegistry`].
//!
//! [`GpuTensor`]: super::GpuTensor
//! [`KernelRegistry`]: super::KernelRegistry
