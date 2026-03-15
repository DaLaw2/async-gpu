//! N-dimensional GPU tensor with shape, strides, and device memory.
//!
//! [`GpuTensor`] is the core data type for the nn module. It owns device memory
//! via `CudaSlice<f32>`, tracks shape and strides via `SmallVec`, and holds an
//! `Arc<CudaDevice>` so it can allocate new memory without passing the device
//! around.
//!
//! # Design choices
//!
//! - **N-dimensional**: GPT-2 needs 2D/3D/4D tensors; a fixed-4D type is too rigid.
//! - **SmallVec<[usize; 4]>**: Avoids heap allocation for the common case (<=4 dims).
//! - **f32 only (v1)**: All existing kernels work with f32. f16 packing is internal.
//! - **C-contiguous by default**: Strides computed from shape. Non-contiguous views
//!   are materialized (copied) before kernel launch.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice};
use smallvec::SmallVec;

use super::error::{NnError, Result};

/// N-dimensional GPU tensor with device memory and shape metadata.
///
/// Owns its device memory via [`CudaSlice<f32>`]. When dropped, the device
/// memory is freed. Holds an [`Arc<CudaDevice>`] so new allocations can be
/// created from the tensor itself (e.g., `reshape`, `transpose`).
///
/// # Example
///
/// ```no_run
/// use gpu_host::nn::GpuTensor;
/// use std::sync::Arc;
///
/// # fn example() -> gpu_host::nn::error::Result<()> {
/// let dev = cudarc::driver::CudaDevice::new(0)?;
/// let t = GpuTensor::from_host(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &dev)?;
/// assert_eq!(t.shape(), &[2, 2]);
/// assert_eq!(t.numel(), 4);
/// let host = t.to_host()?;
/// assert_eq!(host, vec![1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub struct GpuTensor {
    /// Device memory (f32). Owned — dropped when tensor is dropped.
    data: CudaSlice<f32>,
    /// Shape dimensions, e.g. `[batch, channels, height, width]`.
    /// SmallVec avoids heap allocation for <= 4 dims.
    shape: SmallVec<[usize; 4]>,
    /// Strides in element count per dimension (C-contiguous by default).
    strides: SmallVec<[usize; 4]>,
    /// Device reference for memory operations.
    device: Arc<CudaDevice>,
}

impl GpuTensor {
    /// Create a tensor from existing device memory and a shape.
    ///
    /// Computes C-contiguous strides from the given shape. The caller must
    /// ensure `data.len() == shape.iter().product()`.
    pub fn from_data(data: CudaSlice<f32>, shape: &[usize], device: Arc<CudaDevice>) -> Self {
        let strides = compute_strides(shape);
        Self {
            data,
            shape: SmallVec::from_slice(shape),
            strides,
            device,
        }
    }

    /// Upload host data to GPU, returning a new tensor.
    ///
    /// `data.len()` must equal `shape.iter().product()`, otherwise returns
    /// [`NnError::ShapeMismatch`].
    pub fn from_host(data: &[f32], shape: &[usize], device: &Arc<CudaDevice>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(NnError::ShapeMismatch {
                expected: format!("numel={numel} from shape {shape:?}"),
                actual: format!("data.len()={}", data.len()),
            });
        }
        let slice = device.htod_sync_copy(data)?;
        let strides = compute_strides(shape);
        Ok(Self {
            data: slice,
            shape: SmallVec::from_slice(shape),
            strides,
            device: Arc::clone(device),
        })
    }

    /// Download tensor data to host.
    pub fn to_host(&self) -> Result<Vec<f32>> {
        let host = self.device.dtoh_sync_copy(&self.data)?;
        Ok(host)
    }

    /// Allocate a zeroed tensor with the given shape.
    pub fn zeros(shape: &[usize], device: &Arc<CudaDevice>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let slice = device.alloc_zeros::<f32>(numel)?;
        let strides = compute_strides(shape);
        Ok(Self {
            data: slice,
            shape: SmallVec::from_slice(shape),
            strides,
            device: Arc::clone(device),
        })
    }

    /// Number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides as a slice.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Whether the tensor is C-contiguous (row-major, no gaps).
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Reference to the underlying device memory slice.
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Mutable reference to the underlying device memory slice.
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
    }

    /// Reference to the CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Raw device pointer (for kernel launch arguments).
    pub fn as_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        *self.data.device_ptr()
    }

    /// Number of bytes occupied on device.
    pub fn size_bytes(&self) -> usize {
        self.data.num_bytes()
    }

    /// Reshape to a new shape with the same number of elements.
    ///
    /// The tensor must be contiguous. Returns a new tensor sharing the same
    /// device memory if contiguous (zero-copy view is not yet supported, so
    /// this currently copies).
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(NnError::ShapeMismatch {
                expected: format!(
                    "numel={} matching current shape {:?}",
                    self.numel(),
                    self.shape
                ),
                actual: format!("numel={new_numel} from shape {new_shape:?}"),
            });
        }
        // Copy data to new allocation with new shape
        let mut new_data = self.device.alloc_zeros::<f32>(new_numel)?;
        self.device.dtod_copy(&self.data, &mut new_data)?;
        Ok(Self {
            data: new_data,
            shape: SmallVec::from_slice(new_shape),
            strides: compute_strides(new_shape),
            device: Arc::clone(&self.device),
        })
    }

    /// Transpose two dimensions. Always produces a contiguous copy.
    ///
    /// This is needed for attention (transpose seq and head dims).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let ndim = self.ndim();
        if dim0 >= ndim {
            return Err(NnError::DimOutOfRange { dim: dim0, ndim });
        }
        if dim1 >= ndim {
            return Err(NnError::DimOutOfRange { dim: dim1, ndim });
        }
        if dim0 == dim1 {
            return self.clone_tensor();
        }

        // Build transposed shape and strides
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        let mut src_strides = self.strides.clone();
        src_strides.swap(dim0, dim1);

        // Materialize contiguous copy with transposed layout
        let numel = self.numel();
        let host = self.to_host()?;
        let mut transposed = vec![0.0f32; numel];

        // Generic N-dim transpose via index mapping
        let new_strides = compute_strides(&new_shape);
        for flat_dst in 0..numel {
            // Convert flat_dst to multi-index in new_shape
            let mut remaining = flat_dst;
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = remaining / new_strides[d];
                remaining %= new_strides[d];
                src_flat += idx * src_strides[d];
            }
            transposed[flat_dst] = host[src_flat];
        }

        Self::from_host(&transposed, &new_shape, &self.device)
    }

    /// Deep copy on device (device-to-device copy).
    pub fn clone_tensor(&self) -> Result<Self> {
        let numel = self.numel();
        let mut new_data = self.device.alloc_zeros::<f32>(numel)?;
        self.device.dtod_copy(&self.data, &mut new_data)?;
        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            device: Arc::clone(&self.device),
        })
    }

    /// Concatenate tensors along the given dimension.
    ///
    /// All tensors must have the same shape except along `dim`.
    pub fn concat(tensors: &[&GpuTensor], dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(NnError::ShapeMismatch {
                expected: "at least one tensor".to_string(),
                actual: "empty list".to_string(),
            });
        }
        let ndim = tensors[0].ndim();
        if dim >= ndim {
            return Err(NnError::DimOutOfRange { dim, ndim });
        }
        // Validate shapes match except along dim
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.ndim() != ndim {
                return Err(NnError::ShapeMismatch {
                    expected: format!("ndim={ndim}"),
                    actual: format!("tensor[{i}] has ndim={}", t.ndim()),
                });
            }
            for d in 0..ndim {
                if d != dim && t.shape()[d] != tensors[0].shape()[d] {
                    return Err(NnError::ShapeMismatch {
                        expected: format!("dim {d} size {}", tensors[0].shape()[d]),
                        actual: format!("tensor[{i}] dim {d} size {}", t.shape()[d]),
                    });
                }
            }
        }

        // Compute output shape
        let mut out_shape: SmallVec<[usize; 4]> = tensors[0].shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape()[dim]).sum();

        // Download all to host, concat, upload
        let out_numel: usize = out_shape.iter().product();
        let mut out_host = vec![0.0f32; out_numel];
        let out_strides = compute_strides(&out_shape);

        let mut dim_offset = 0;
        for t in tensors {
            let t_host = t.to_host()?;
            let t_strides = compute_strides(&t.shape);
            let t_numel = t.numel();
            for flat_src in 0..t_numel {
                // Convert flat_src to multi-index in t.shape
                let mut remaining = flat_src;
                let mut dst_flat = 0;
                for d in 0..ndim {
                    let idx = remaining / t_strides[d];
                    remaining %= t_strides[d];
                    let dst_idx = if d == dim { idx + dim_offset } else { idx };
                    dst_flat += dst_idx * out_strides[d];
                }
                out_host[dst_flat] = t_host[flat_src];
            }
            dim_offset += t.shape()[dim];
        }

        Self::from_host(&out_host, &out_shape, &tensors[0].device)
    }

    /// Split a tensor along the given dimension into chunks of the specified sizes.
    ///
    /// `sizes` must sum to `self.shape()[dim]`.
    pub fn split(&self, dim: usize, sizes: &[usize]) -> Result<Vec<Self>> {
        let ndim = self.ndim();
        if dim >= ndim {
            return Err(NnError::DimOutOfRange { dim, ndim });
        }
        let total: usize = sizes.iter().sum();
        if total != self.shape()[dim] {
            return Err(NnError::ShapeMismatch {
                expected: format!("sizes sum to {}", self.shape()[dim]),
                actual: format!("sizes sum to {total}"),
            });
        }

        let host = self.to_host()?;
        let src_strides = compute_strides(&self.shape);
        let mut results = Vec::with_capacity(sizes.len());
        let mut dim_offset = 0;

        for &sz in sizes {
            let mut chunk_shape = self.shape.clone();
            chunk_shape[dim] = sz;
            let chunk_numel: usize = chunk_shape.iter().product();
            let chunk_strides = compute_strides(&chunk_shape);
            let mut chunk_host = vec![0.0f32; chunk_numel];

            for flat_dst in 0..chunk_numel {
                let mut remaining = flat_dst;
                let mut src_flat = 0;
                for d in 0..ndim {
                    let idx = remaining / chunk_strides[d];
                    remaining %= chunk_strides[d];
                    let src_idx = if d == dim { idx + dim_offset } else { idx };
                    src_flat += src_idx * src_strides[d];
                }
                chunk_host[flat_dst] = host[src_flat];
            }

            results.push(Self::from_host(&chunk_host, &chunk_shape, &self.device)?);
            dim_offset += sz;
        }

        Ok(results)
    }
}

/// Compute C-contiguous (row-major) strides for a given shape.
fn compute_strides(shape: &[usize]) -> SmallVec<[usize; 4]> {
    let mut strides = SmallVec::with_capacity(shape.len());
    if shape.is_empty() {
        return strides;
    }
    strides.resize(shape.len(), 1);
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_slice())
            .field("numel", &self.numel())
            .finish()
    }
}

use std::fmt;

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Arc<CudaDevice> {
        CudaDevice::new(0).expect("CUDA device")
    }

    #[test]
    fn test_from_host_roundtrip() {
        let dev = test_device();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::from_host(&data, &[2, 3], &dev).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.ndim(), 2);
        assert!(t.is_contiguous());
        let back = t.to_host().unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn test_zeros() {
        let dev = test_device();
        let t = GpuTensor::zeros(&[3, 4], &dev).unwrap();
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.numel(), 12);
        let host = t.to_host().unwrap();
        assert!(host.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_reshape() {
        let dev = test_device();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = GpuTensor::from_host(&data, &[2, 3, 4], &dev).unwrap();
        let r = t.reshape(&[6, 4]).unwrap();
        assert_eq!(r.shape(), &[6, 4]);
        assert_eq!(r.to_host().unwrap(), data);
    }

    #[test]
    fn test_reshape_numel_mismatch() {
        let dev = test_device();
        let t = GpuTensor::from_host(&[1.0, 2.0, 3.0], &[3], &dev).unwrap();
        assert!(t.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_transpose_2d() {
        let dev = test_device();
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = GpuTensor::from_host(&data, &[2, 3], &dev).unwrap();
        let tr = t.transpose(0, 1).unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.to_host().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_clone_tensor() {
        let dev = test_device();
        let data = vec![1.0, 2.0, 3.0];
        let t = GpuTensor::from_host(&data, &[3], &dev).unwrap();
        let c = t.clone_tensor().unwrap();
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.to_host().unwrap(), data);
    }

    #[test]
    fn test_concat_and_split() {
        let dev = test_device();
        let a = GpuTensor::from_host(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &dev).unwrap();
        let b = GpuTensor::from_host(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &dev).unwrap();

        // Concat along dim 0: [2,2] + [2,2] -> [4,2]
        let cat = GpuTensor::concat(&[&a, &b], 0).unwrap();
        assert_eq!(cat.shape(), &[4, 2]);
        assert_eq!(
            cat.to_host().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );

        // Split back
        let parts = cat.split(0, &[2, 2]).unwrap();
        assert_eq!(parts[0].to_host().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(parts[1].to_host().unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_concat_dim1() {
        let dev = test_device();
        let a = GpuTensor::from_host(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &dev).unwrap();
        let b = GpuTensor::from_host(&[5.0, 6.0], &[2, 1], &dev).unwrap();

        // Concat along dim 1: [2,2] + [2,1] -> [2,3]
        let cat = GpuTensor::concat(&[&a, &b], 1).unwrap();
        assert_eq!(cat.shape(), &[2, 3]);
        assert_eq!(cat.to_host().unwrap(), vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }
}
