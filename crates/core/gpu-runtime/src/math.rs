/// Approximate square root. PTX: `sqrt.approx.f32` (~1 ULP).
#[inline(always)]
pub fn sqrt_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "sqrt.approx.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) x,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate reciprocal square root. PTX: `rsqrt.approx.f32`.
#[inline(always)]
pub fn rsqrt_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "rsqrt.approx.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) x,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate exponential (e^x). Uses PTX `ex2.approx.f32` with log2(e) scaling.
#[inline(always)]
pub fn exp_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        let t = x * 1.442695_f32; // log2(e)
        unsafe {
            core::arch::asm!(
                "ex2.approx.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) t,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate natural logarithm (ln x). Uses PTX `lg2.approx.f32` with ln(2) scaling.
#[inline(always)]
pub fn log_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let lg2: f32;
        unsafe {
            core::arch::asm!(
                "lg2.approx.f32 {out}, {inp};",
                out = out(reg32) lg2,
                inp = in(reg32) x,
            );
        }
        lg2 * 0.693147_f32 // ln(2)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate sine. PTX: `sin.approx.f32`.
#[inline(always)]
pub fn sin_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "sin.approx.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) x,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Approximate cosine. PTX: `cos.approx.f32`.
#[inline(always)]
pub fn cos_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "cos.approx.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) x,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Absolute value. PTX: `abs.f32`.
#[inline(always)]
pub fn abs_f32(x: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "abs.f32 {out}, {inp};",
                out = out(reg32) result,
                inp = in(reg32) x,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = x;
        0.0
    }
}

/// Minimum of two floats. PTX: `min.f32`.
#[inline(always)]
pub fn min_f32(a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "min.f32 {out}, {a}, {b};",
                out = out(reg32) result,
                a = in(reg32) a,
                b = in(reg32) b,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b);
        0.0
    }
}

/// Maximum of two floats. PTX: `max.f32`.
#[inline(always)]
pub fn max_f32(a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "max.f32 {out}, {a}, {b};",
                out = out(reg32) result,
                a = in(reg32) a,
                b = in(reg32) b,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b);
        0.0
    }
}

/// Fused multiply-add: `a * b + c` in a single instruction. PTX: `fma.rn.f32`.
#[inline(always)]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: f32;
        unsafe {
            core::arch::asm!(
                "fma.rn.f32 {out}, {a}, {b}, {c};",
                out = out(reg32) result,
                a = in(reg32) a,
                b = in(reg32) b,
                c = in(reg32) c,
            );
        }
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, c);
        0.0
    }
}

/// Hyperbolic tangent. Computed as `tanh(x) = 1 - 2/(exp(2x)+1)`.
#[inline(always)]
pub fn tanh_f32(x: f32) -> f32 {
    let e2x = exp_f32(2.0 * x);
    1.0 - 2.0 / (e2x + 1.0)
}

/// Sigmoid function: `1 / (1 + exp(-x))`.
#[inline(always)]
pub fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + exp_f32(-x))
}
