//! Optimizers — SGD and Adam for GPU tensor parameter updates.

use std::collections::HashMap;
use std::sync::Arc;

use super::TensorId;
use crate::nn::error::Result;
use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;

/// SGD optimizer with optional momentum.
pub struct Sgd {
    lr: f32,
    momentum: f32,
    velocity: HashMap<TensorId, GpuTensor>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }

    /// Update parameters in-place using gradients.
    ///
    /// `params`: map of TensorId → mutable GpuTensor (the weights to update).
    /// `grads`: map of TensorId → gradient GpuTensor (from backward()).
    pub fn step(
        &mut self,
        params: &mut HashMap<TensorId, GpuTensor>,
        grads: &HashMap<TensorId, GpuTensor>,
        registry: &Arc<KernelRegistry>,
    ) -> Result<()> {
        for (id, param) in params.iter_mut() {
            if let Some(grad) = grads.get(id) {
                if self.momentum > 0.0 {
                    // Momentum SGD: still CPU for now (needs momentum buffer on GPU)
                    let grad_host = grad.to_host()?;
                    let mut param_host = param.to_host()?;
                    let vel = self.velocity.entry(*id).or_insert_with(|| {
                        let zeros = vec![0.0f32; param_host.len()];
                        GpuTensor::from_host(&zeros, param.shape(), registry.device()).unwrap()
                    });
                    let mut vel_host = vel.to_host()?;
                    for i in 0..param_host.len() {
                        vel_host[i] = self.momentum * vel_host[i] + grad_host[i];
                        param_host[i] -= self.lr * vel_host[i];
                    }
                    *vel = GpuTensor::from_host(&vel_host, vel.shape(), registry.device())?;
                    *param = GpuTensor::from_host(&param_host, param.shape(), registry.device())?;
                } else {
                    // GPU-side SGD: param -= lr * grad (no host roundtrip!)
                    let n = param.numel() as u32;
                    let func = registry.get("sgd_step")?;
                    let config = crate::nn::registry::KernelRegistry::config_1d(n);
                    let status = registry.device().htod_sync_copy(&[0u32])?;
                    unsafe {
                        cudarc::driver::LaunchAsync::launch(
                            func,
                            config,
                            (param.data_mut(), grad.data(), self.lr, n, &status),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Adam optimizer with m,v state tensors.
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step_count: u32,
    m: HashMap<TensorId, Vec<f32>>, // first moment
    v: HashMap<TensorId, Vec<f32>>, // second moment
}

impl Adam {
    /// Create a new Adam optimizer with default betas (0.9, 0.999) and epsilon (1e-8).
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step_count: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Update parameters in-place using gradients.
    pub fn step(
        &mut self,
        params: &mut HashMap<TensorId, GpuTensor>,
        grads: &HashMap<TensorId, GpuTensor>,
        registry: &Arc<KernelRegistry>,
    ) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count;
        let bc1 = 1.0 - self.beta1.powi(t as i32);
        let bc2 = 1.0 - self.beta2.powi(t as i32);

        for (id, param) in params.iter_mut() {
            if let Some(grad) = grads.get(id) {
                let grad_host = grad.to_host()?;
                let mut param_host = param.to_host()?;
                let n = param_host.len();

                let m = self.m.entry(*id).or_insert_with(|| vec![0.0; n]);
                let v = self.v.entry(*id).or_insert_with(|| vec![0.0; n]);

                for i in 0..n {
                    let g = grad_host[i];
                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;

                    let m_hat = m[i] / bc1;
                    let v_hat = v[i] / bc2;
                    param_host[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }

                *param = GpuTensor::from_host(&param_host, param.shape(), registry.device())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_step_decreases_param() {
        // Simple test: param starts at 1.0, grad is positive → param should decrease
        let dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let registry =
            Arc::new(crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).unwrap());

        let id = TensorId(0);
        let param = GpuTensor::from_host(&[1.0, 2.0, 3.0], &[3], &dev).unwrap();
        let grad = GpuTensor::from_host(&[0.1, 0.2, 0.3], &[3], &dev).unwrap();

        let mut params = HashMap::new();
        params.insert(id, param);
        let mut grads = HashMap::new();
        grads.insert(id, grad);

        let mut sgd = Sgd::new(0.1, 0.0);
        sgd.step(&mut params, &grads, &registry).unwrap();

        let updated = params.get(&id).unwrap().to_host().unwrap();
        // param -= lr * grad: [1-0.01, 2-0.02, 3-0.03]
        assert!((updated[0] - 0.99).abs() < 1e-5);
        assert!((updated[1] - 1.98).abs() < 1e-5);
        assert!((updated[2] - 2.97).abs() < 1e-5);
    }

    #[test]
    fn test_adam_step_decreases_param() {
        let dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let registry =
            Arc::new(crate::nn::KernelRegistry::new(Arc::clone(&dev), crate::ptx::KERNEL).unwrap());

        let id = TensorId(0);
        let param = GpuTensor::from_host(&[1.0, 2.0], &[2], &dev).unwrap();
        let grad = GpuTensor::from_host(&[0.5, -0.5], &[2], &dev).unwrap();

        let mut params = HashMap::new();
        params.insert(id, param);
        let mut grads = HashMap::new();
        grads.insert(id, grad);

        let mut adam = Adam::new(0.01);
        adam.step(&mut params, &grads, &registry).unwrap();

        let updated = params.get(&id).unwrap().to_host().unwrap();
        // After 1 step, param[0] should decrease (positive grad), param[1] should increase
        assert!(updated[0] < 1.0, "param[0] should decrease: {}", updated[0]);
        assert!(updated[1] > 2.0, "param[1] should increase: {}", updated[1]);
    }
}
