//! MIR pass: `WarpCooperativeTransform`
//!
//! A skeleton MIR pass that runs **after** `StateTransform` (coroutine → state machine)
//! and identifies yield/poll sites in coroutine bodies compiled for `nvptx64`.
//!
//! # Phase 1 (this file)
//!
//! - Detects `#[warp_cooperative]` attribute on the source function
//! - Identifies `TerminatorKind::Yield` terminators (pre-StateTransform residuals, if any)
//! - Identifies `TerminatorKind::Call` terminators that resolve to `Future::poll`
//! - Identifies the dispatch `SwitchInt` on the coroutine discriminant
//! - Emits diagnostic notes about discovered sites — no MIR rewriting yet
//!
//! # Future phases
//!
//! - Phase 2: Broadcast the dispatch discriminant via `shfl.sync.idx.b32`
//! - Phase 3: Gate `Future::poll` calls behind lane-0 predication
//! - Phase 4: Broadcast `Poll::Ready(T)` payloads (u32 → u64 → structs)
//! - Phase 5: Handle `?` operator, barriers before `Return`, validation

use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_span::def_id::DefId;
use rustc_span::sym;

use crate::MirPass;

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

pub(crate) struct WarpCooperativeTransform;

impl<'tcx> MirPass<'tcx> for WarpCooperativeTransform {
    fn is_enabled(&self, sess: &Session) -> bool {
        sess.target.arch == "nvptx64"
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();

        // Only transform functions annotated with `#[warp_cooperative]`.
        if !has_warp_cooperative_attr(tcx, def_id) {
            return;
        }

        // After StateTransform the body retains `body.coroutine` metadata even
        // though Yield terminators have been lowered to a dispatch switch.
        // If that metadata is absent the function was never a coroutine — skip.
        if body.coroutine.is_none() {
            tcx.dcx().span_warn(
                body.span,
                "#[warp_cooperative] on a non-coroutine function has no effect",
            );
            return;
        }

        let fn_name = tcx.def_path_str(def_id);
        let analysis = CoroutineAnalysis::run(tcx, body);
        analysis.emit_diagnostics(tcx, body, &fn_name);

        // TODO(phase2): Insert discriminant broadcast at the dispatch switch.
        // TODO(phase3): Gate poll calls behind lane-0 predication and broadcast
        //               the Poll discriminant.
        // TODO(phase4): Broadcast Poll::Ready(T) payloads.
        // TODO(phase5): Handle `?` (Result broadcasting), add syncwarp before
        //               Return terminators, validate against self-referential
        //               borrows / dyn Future / Drop output types.
    }
}

// ---------------------------------------------------------------------------
// Attribute detection
// ---------------------------------------------------------------------------

fn has_warp_cooperative_attr(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // `sym::warp_cooperative` must be added to `rustc_span::symbol::sym`.
    // For the skeleton, fall back to scanning attributes by string name.
    //
    // Canonical API (once the symbol is registered):
    //   tcx.get_attrs(def_id, sym::warp_cooperative).next().is_some()
    //
    // Temporary fallback that works without modifying rustc_span:
    tcx.get_attrs_unchecked(def_id).iter().any(|attr| attr.has_name(sym::warp_cooperative))
}

// ---------------------------------------------------------------------------
// Analysis results
// ---------------------------------------------------------------------------

/// Collected information about the coroutine body that will drive future
/// rewrite phases.
#[allow(dead_code)]
struct CoroutineAnalysis {
    /// Basic block indices that contain `TerminatorKind::Yield`.
    /// After `StateTransform` this list is normally empty — yields are
    /// converted into the dispatch switch.  If non-empty it means the pass
    /// ran before StateTransform (a pipeline ordering bug).
    yield_points: Vec<BasicBlock>,

    /// `(basic_block, callee_def_id)` pairs for every `Call` terminator
    /// whose callee resolves to `<T as Future>::poll`.
    poll_call_sites: Vec<(BasicBlock, DefId)>,

    /// The basic block containing the coroutine dispatch `SwitchInt`, if found.
    dispatch_switch_bb: Option<BasicBlock>,

    /// Number of suspension-point targets in the dispatch switch
    /// (discriminant values >= RESERVED_VARIANTS = 3).
    suspension_point_count: usize,

    /// Basic blocks that end with `TerminatorKind::Return`.
    return_blocks: Vec<BasicBlock>,
}

impl CoroutineAnalysis {
    fn run<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut visitor = AnalysisVisitor::new(tcx);
        visitor.visit_body(body);

        // Find the dispatch switch — the first SwitchInt whose scrutinee is
        // the discriminant of the coroutine self parameter (`_1`).
        let (dispatch_switch_bb, suspension_point_count) =
            find_dispatch_switch(body).unwrap_or((None, 0));

        CoroutineAnalysis {
            yield_points: visitor.yield_points,
            poll_call_sites: visitor.poll_call_sites,
            dispatch_switch_bb,
            suspension_point_count,
            return_blocks: visitor.return_blocks,
        }
    }

    fn emit_diagnostics(&self, tcx: TyCtxt<'_>, body: &Body<'_>, fn_name: &str) {
        let span = body.span;

        // Summary note.
        tcx.dcx().span_note(
            span,
            format!(
                "warp_cooperative: analyzing `{fn_name}` — \
                 {yp} yield point(s), \
                 {pc} poll-call site(s), \
                 {sp} suspension point(s), \
                 {rb} return block(s)",
                yp = self.yield_points.len(),
                pc = self.poll_call_sites.len(),
                sp = self.suspension_point_count,
                rb = self.return_blocks.len(),
            ),
        );

        // Warn if yields are still present (pipeline ordering error).
        if !self.yield_points.is_empty() {
            tcx.dcx().span_warn(
                span,
                format!(
                    "warp_cooperative: {} Yield terminator(s) still present — \
                     pass may have run before StateTransform",
                    self.yield_points.len(),
                ),
            );
        }

        // Detail: dispatch switch.
        if let Some(bb) = self.dispatch_switch_bb {
            tcx.dcx().span_note(
                span,
                format!(
                    "warp_cooperative: dispatch switch at {bb:?} \
                     with {n} suspension point(s)",
                    n = self.suspension_point_count,
                ),
            );
        } else {
            tcx.dcx().span_note(
                span,
                "warp_cooperative: no dispatch switch found \
                 (function may not be a transformed coroutine)",
            );
        }

        // Detail: poll call sites.
        for (bb, callee) in &self.poll_call_sites {
            let callee_name = tcx.def_path_str(*callee);
            tcx.dcx().span_note(
                span,
                format!("warp_cooperative: poll call at {bb:?} → `{callee_name}`"),
            );
        }

        // Detail: return blocks.
        for bb in &self.return_blocks {
            tcx.dcx().span_note(
                span,
                format!("warp_cooperative: return at {bb:?}"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MIR visitor that collects yield points, poll calls, and return blocks
// ---------------------------------------------------------------------------

struct AnalysisVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    yield_points: Vec<BasicBlock>,
    poll_call_sites: Vec<(BasicBlock, DefId)>,
    return_blocks: Vec<BasicBlock>,
}

impl<'tcx> AnalysisVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self {
            tcx,
            yield_points: Vec::new(),
            poll_call_sites: Vec::new(),
            return_blocks: Vec::new(),
        }
    }
}

impl<'tcx> Visitor<'tcx> for AnalysisVisitor<'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let bb = location.block;

        match &terminator.kind {
            // ----------------------------------------------------------
            // Yield — should be absent after StateTransform
            // ----------------------------------------------------------
            TerminatorKind::Yield { .. } => {
                self.yield_points.push(bb);
            }

            // ----------------------------------------------------------
            // Call — check if callee is `<T as Future>::poll`
            // ----------------------------------------------------------
            TerminatorKind::Call { func, .. } => {
                if let Some(callee_def_id) = resolve_callee_def_id(self.tcx, func) {
                    if is_future_poll(self.tcx, callee_def_id) {
                        self.poll_call_sites.push((bb, callee_def_id));
                    }
                }
            }

            // ----------------------------------------------------------
            // Return — marks completion (Poll::Ready of outermost future)
            // ----------------------------------------------------------
            TerminatorKind::Return => {
                self.return_blocks.push(bb);
            }

            _ => {}
        }

        self.super_terminator(terminator, location);
    }
}

// ---------------------------------------------------------------------------
// Callee resolution helpers
// ---------------------------------------------------------------------------

/// Extract the `DefId` of the callee from a `Call` terminator's function
/// operand, if it is a statically-known function (`FnDef`).
fn resolve_callee_def_id<'tcx>(
    _tcx: TyCtxt<'tcx>,
    func: &Operand<'tcx>,
) -> Option<DefId> {
    match func {
        Operand::Constant(c) => {
            match c.const_.ty().kind() {
                ty::FnDef(def_id, _substs) => Some(*def_id),
                _ => None,
            }
        }
        // Indirect calls (fn pointers, closures) cannot be resolved
        // statically.  We conservatively skip them.
        Operand::Copy(_) | Operand::Move(_) => None,
    }
}

/// Returns `true` if `def_id` refers to the `poll` method on the
/// `core::future::Future` trait.
fn is_future_poll(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // Approach: check whether the item's parent trait is `Future` and
    // the item name is `poll`.
    //
    // `tcx.trait_of_item(def_id)` returns `Some(trait_def_id)` when
    // `def_id` is an associated item of a trait impl or trait definition.
    let Some(trait_def_id) = tcx.trait_of_item(def_id) else {
        return false;
    };

    // Compare against the lang-item `Future` trait.
    let Some(future_trait) = tcx.lang_items().future_trait() else {
        return false;
    };

    if trait_def_id != future_trait {
        return false;
    }

    // Confirm the method name is `poll`.
    tcx.item_name(def_id) == sym::poll
}

// ---------------------------------------------------------------------------
// Dispatch-switch detection
// ---------------------------------------------------------------------------

/// RESERVED_VARIANTS mirrors `CoroutineArgs::RESERVED_VARIANTS` in
/// `rustc_middle::ty::sty`.  Discriminant values 0, 1, 2 correspond to
/// UNRESUMED, RETURNED, POISONED.  Values >= 3 are suspension points.
const RESERVED_VARIANTS: u128 = 3;

/// Locate the dispatch `SwitchInt` in the body.
///
/// After `StateTransform`, the entry block (or a block reachable from it)
/// contains a `SwitchInt` on `Discriminant((*_1))` — the coroutine state
/// discriminant.  We identify it by:
///
/// 1. Looking for `SwitchInt` terminators.
/// 2. Checking that the scrutinee is a `Rvalue::Discriminant` applied to a
///    projection through the first argument (`_1`, the `&mut Self` of the
///    resume function).
/// 3. Counting targets with values >= `RESERVED_VARIANTS`.
fn find_dispatch_switch(body: &Body<'_>) -> Option<(Option<BasicBlock>, usize)> {
    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        let terminator = bb_data.terminator();

        if let TerminatorKind::SwitchInt { discr, targets, .. } = &terminator.kind {
            // The scrutinee should be a local that was assigned via
            // `Discriminant((*_1))`.  We check a simpler heuristic: the
            // scrutinee is a `Copy` or `Move` of a local, and one of the
            // preceding statements in this (or a predecessor) block writes
            // that local as `Rvalue::Discriminant` of a place rooted at `_1`.
            if is_discriminant_of_self(discr, bb_data, body) {
                let suspension_points = targets
                    .iter()
                    .filter(|(val, _)| *val >= RESERVED_VARIANTS)
                    .count();
                return Some((Some(bb), suspension_points));
            }
        }
    }
    // No dispatch switch found — the function may not be a coroutine,
    // or StateTransform has not run yet.
    Some((None, 0))
}

/// Check whether the `SwitchInt` scrutinee originates from
/// `Discriminant((*_1))`.
///
/// We inspect the statements in `bb_data` looking for an assignment
/// `scrutinee_local = Discriminant(place)` where `place` is rooted at
/// `Local::from(1u32)` (the `self` parameter of the resume fn).
fn is_discriminant_of_self(
    discr: &Operand<'_>,
    bb_data: &BasicBlockData<'_>,
    body: &Body<'_>,
) -> bool {
    // Extract the local that the SwitchInt reads.
    let scrutinee_local = match discr {
        Operand::Copy(place) | Operand::Move(place) => {
            if place.projection.is_empty() {
                place.local
            } else {
                return false;
            }
        }
        _ => return false,
    };

    // Walk statements in the same block looking for the assignment.
    for stmt in &bb_data.statements {
        if let StatementKind::Assign(box (lhs, rvalue)) = &stmt.kind {
            if lhs.local == scrutinee_local && lhs.projection.is_empty() {
                if let Rvalue::Discriminant(place) = rvalue {
                    return is_rooted_at_self(place, body);
                }
            }
        }
    }
    false
}

/// Returns `true` if `place` is (transitively through derefs) rooted at
/// `_1`, which is the `&mut Self` parameter in the resume function
/// generated by `StateTransform`.
fn is_rooted_at_self(place: &Place<'_>, _body: &Body<'_>) -> bool {
    // _1 is Local::from(1u32).
    let self_local = Local::from(1u32);
    place.local == self_local
}

// ---------------------------------------------------------------------------
// Future MIR rewriting helpers (stubs for Phase 2+)
// ---------------------------------------------------------------------------

// TODO(phase2): fn insert_discriminant_broadcast(...)
//   - Split the dispatch switch block: interpose shfl.sync.idx.b32
//     between the discriminant read and the SwitchInt.
//   - New locals: _mask (u32), _broadcast_discr (u32).
//   - New blocks: activemask call, shfl_sync call, then original SwitchInt
//     on the broadcast discriminant.

// TODO(phase3): fn gate_poll_behind_leader(...)
//   - For each poll call site, insert lane-id check.
//   - Lane 0 → call Future::poll; other lanes → skip to broadcast block.
//   - Broadcast the Poll discriminant via shfl.sync.idx.b32.

// TODO(phase4): fn broadcast_ready_payload(...)
//   - Determine Output type size via tcx.layout_of().
//   - <= 4 bytes: single shfl.sync.idx.b32.
//   - <= 8 bytes: two shfl.sync calls (hi/lo halves).
//   - <= 256 bytes: decompose into u32 words.
//   - > 256 bytes: shared memory fallback.

// TODO(phase5): fn validate_output_type(...)
//   - Reject dyn Future, types with Drop, self-referential borrows.
//   - Handle Result<T, E> broadcasting for `?` operator.
//   - Insert syncwarp barriers before Return terminators.

// TODO(phase2+): fn emit_inline_asm_shfl_sync(...)
//   - Create TerminatorKind::InlineAsm with template:
//     "shfl.sync.idx.b32 $0, $1, $2, 31, $3;"
//   - Operands: Out(dest), In(src), In(src_lane=0), In(mask).

// TODO(phase2+): fn emit_inline_asm_activemask(...)
//   - Template: "activemask.b32 $0;"
//   - Operands: Out(mask).

// TODO(phase2+): fn emit_inline_asm_syncwarp(...)
//   - Template: "bar.warp.sync $0;"
//   - Operands: In(mask).

// TODO(phase3): fn emit_inline_asm_lane_id(...)
//   - Template: "mov.u32 $0, %laneid;"
//   - Operands: Out(lane_id).
