//! Autograd — tape-based automatic differentiation for GPU tensors.
//!
//! Records forward operations on a [`Tape`] and computes gradients via
//! [`backward()`] in reverse topological order.

mod pool;
mod tape;

pub use pool::TensorPool;
pub use tape::{OpKind, OpMeta, Reduction, Tape, TapeEntry, TensorId};
