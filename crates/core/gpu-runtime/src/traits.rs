//! User-extensible GPU traits for generic kernel algorithms.
//!
//! These traits enable writing generic kernel functions bounded by
//! user-defined behavior — the compiler monomorphizes each instantiation
//! to type-specific PTX instructions, with zero runtime overhead.
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::traits::GpuReducible;
//!
//! #[inline(always)]
//! fn parallel_sum<T: GpuReducible>(data: &[T]) -> T {
//!     let mut acc = T::identity();
//!     for &x in data {
//!         acc = acc.combine(x);
//!     }
//!     acc
//! }
//! ```

/// Trait for types that can be reduced (summed, combined) on GPU.
///
/// Provides an identity element and a binary combine operation.
/// The compiler monomorphizes generic functions bounded by `GpuReducible`
/// to type-specific PTX — `combine` for `f32` emits `add.rn.f32`,
/// for `i32` emits `add.s32`, etc.
///
/// # Laws
///
/// Implementations should satisfy:
/// - `identity().combine(x) == x` (left identity)
/// - `x.combine(identity()) == x` (right identity)
/// - `a.combine(b).combine(c) == a.combine(b.combine(c))` (associativity)
///
/// Associativity is required for correct parallel reduction (warps reduce
/// partitions independently and combine results).
pub trait GpuReducible: Copy {
    /// The identity element for the combine operation.
    ///
    /// For addition: `0`. For multiplication: `1`.
    fn identity() -> Self;

    /// Combine two values into one.
    ///
    /// For addition: `self + other`. For multiplication: `self * other`.
    fn combine(self, other: Self) -> Self;
}

/// Trait for types that support an elementwise transform on GPU.
///
/// This demonstrates `where` bounds in generic GPU code — functions
/// bounded by `where T: GpuTransformable` monomorphize identically
/// to those using trait syntax `<T: GpuTransformable>`.
pub trait GpuTransformable: Copy {
    /// The default/zero value for this type.
    fn default_value() -> Self;

    /// Apply a scaling transform: `self * factor`.
    fn scale(self, factor: Self) -> Self;

    /// Apply an additive offset: `self + offset`.
    fn offset(self, amount: Self) -> Self;
}

// ============================================================
// Built-in implementations for primitive types
// ============================================================

macro_rules! impl_gpu_reducible_additive {
    ($($t:ty, $zero:expr);* $(;)?) => {
        $(
            impl GpuReducible for $t {
                #[inline(always)]
                fn identity() -> Self { $zero }
                #[inline(always)]
                fn combine(self, other: Self) -> Self { self + other }
            }
        )*
    };
}

impl_gpu_reducible_additive! {
    f32, 0.0;
    f64, 0.0;
    u32, 0;
    u64, 0;
    i32, 0;
    i64, 0;
    usize, 0;
}

macro_rules! impl_gpu_transformable {
    ($($t:ty, $zero:expr);* $(;)?) => {
        $(
            impl GpuTransformable for $t {
                #[inline(always)]
                fn default_value() -> Self { $zero }
                #[inline(always)]
                fn scale(self, factor: Self) -> Self { self * factor }
                #[inline(always)]
                fn offset(self, amount: Self) -> Self { self + amount }
            }
        )*
    };
}

impl_gpu_transformable! {
    f32, 0.0;
    f64, 0.0;
    u32, 0;
    u64, 0;
    i32, 0;
    i64, 0;
    usize, 0;
}
