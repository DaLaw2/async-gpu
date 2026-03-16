//! Autograd — tape-based automatic differentiation for GPU tensors.
//!
//! Records forward operations on a [`Tape`] and computes gradients via
//! [`backward()`] in reverse topological order.
//!
//! # Thread-local tape
//!
//! Use [`with_tape`] to run a closure with an active tape. Forward ops
//! automatically record onto the active tape when tensors have `requires_grad`.
//!
//! ```no_run
//! # fn example() {
//! use gpu_host::nn::autograd::{with_tape, Tape};
//!
//! let tape = Tape::new();
//! with_tape(tape, || {
//!     // Forward ops here will record onto the tape
//! });
//! # }
//! ```

pub mod backward;
pub mod checkpoint;
pub mod loss;
pub mod optim;
mod pool;
mod tape;

pub use pool::TensorPool;
pub use tape::{OpKind, OpMeta, Reduction, Tape, TapeEntry, TensorId};

use std::cell::RefCell;

thread_local! {
    static ACTIVE_TAPE: RefCell<Option<Tape>> = const { RefCell::new(None) };
}

/// Run a closure with an active autograd tape.
///
/// After the closure returns, the tape is extracted and returned.
/// Any forward ops on tensors with `requires_grad = true` will be recorded.
pub fn with_tape<F, R>(tape: Tape, f: F) -> (R, Tape)
where
    F: FnOnce() -> R,
{
    ACTIVE_TAPE.with(|cell| {
        *cell.borrow_mut() = Some(tape);
    });
    let result = f();
    let tape = ACTIVE_TAPE.with(|cell| cell.borrow_mut().take().unwrap_or_default());
    (result, tape)
}

/// Record an operation on the active thread-local tape (if any).
///
/// Called by forward ops when any input tensor has `requires_grad = true`.
pub fn record_op(entry: TapeEntry) {
    ACTIVE_TAPE.with(|cell| {
        if let Some(ref mut tape) = *cell.borrow_mut() {
            tape.record(entry);
        }
    });
}

/// Allocate a new tensor ID from the active tape (if any).
///
/// Returns `None` if no tape is active.
pub fn alloc_tensor_id() -> Option<TensorId> {
    ACTIVE_TAPE.with(|cell| cell.borrow_mut().as_mut().map(|tape| tape.alloc_id()))
}
