//! MIR pass: `WarpCooperativeTransform`
//!
//! Runs **after** `StateTransform` (coroutine → state machine) and rewrites the
//! generated dispatch loop for warp-cooperative execution on `nvptx64`.
//!
//! ## What it does
//!
//! 1. **Dispatch discriminant broadcast**: At the entry `SwitchInt` that dispatches
//!    on the coroutine state, inserts `activemask` + `shfl.sync.idx.b32` so all
//!    32 lanes in a warp agree on the current state.
//!
//! 2. **Barrier before Return**: Inserts `bar.warp.sync` before every `Return`
//!    terminator so all lanes exit together.
//!
//! 3. **Analysis**: Emits diagnostic notes about poll-call sites and suspension
//!    points for debugging.
//!
//! ## Activation
//!
//! - Only on `nvptx64` targets (`is_enabled` check)
//! - Only on functions annotated with `#[warp_cooperative]`
//! - Only on coroutine bodies (post-StateTransform)

use std::borrow::Cow;

use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::Session;
use rustc_span::Symbol;
use rustc_span::def_id::DefId;
use rustc_target::asm::{InlineAsmRegClass, InlineAsmRegOrRegClass};
use rustc_target::asm::nvptx::NvptxInlineAsmRegClass;
use tracing::debug;

use crate::MirPass;

// ---------------------------------------------------------------------------
// Arena allocation helper
// ---------------------------------------------------------------------------

/// Allocate a `&'tcx [InlineAsmTemplatePiece]` from a `Vec`.
///
/// The MIR type `TerminatorKind::InlineAsm` stores the template as `&'tcx [...]`.
/// Normally this comes from the HIR arena, but since we're synthesizing inline asm
/// in a MIR pass, we leak the allocation. This is fine — compiler arenas would
/// free it at the same time anyway (end of compilation).
fn alloc_template(
    pieces: impl IntoIterator<Item = InlineAsmTemplatePiece>,
) -> &'static [InlineAsmTemplatePiece] {
    let v: Vec<InlineAsmTemplatePiece> = pieces.into_iter().collect();
    Box::leak(v.into_boxed_slice())
}

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

pub(super) struct WarpCooperativeTransform;

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
        if body.coroutine.is_none() {
            tcx.dcx().span_warn(
                body.span,
                "#[warp_cooperative] on a non-coroutine function has no effect",
            );
            return;
        }

        let fn_name = tcx.def_path_str(def_id);
        debug!("WarpCooperativeTransform: processing `{fn_name}`");

        // Phase 1: Analysis — collect structural info about the post-StateTransform MIR.
        let analysis = CoroutineAnalysis::run(tcx, body);
        analysis.emit_diagnostics(tcx, body, &fn_name);

        // Phase 2: Broadcast the dispatch discriminant via shfl.sync.
        if let Some(dispatch_bb) = analysis.dispatch_switch_bb {
            insert_discriminant_broadcast(tcx, body, dispatch_bb);
        }

        // Phase 4 (Rule 4): Barrier before Return terminators.
        // Collect return blocks first, then mutate — avoids borrow conflict.
        let return_blocks: Vec<BasicBlock> = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(bb, data)| {
                if matches!(data.terminator().kind, TerminatorKind::Return) {
                    Some(bb)
                } else {
                    None
                }
            })
            .collect();
        for bb in return_blocks {
            insert_barrier_before_return(tcx, body, bb);
        }
    }
}

// ---------------------------------------------------------------------------
// Attribute detection
// ---------------------------------------------------------------------------

fn has_warp_cooperative_attr(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let warp_sym = Symbol::intern("warp_cooperative");
    tcx.get_attrs_unchecked(def_id)
        .iter()
        .any(|attr| attr.has_name(warp_sym))
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

struct CoroutineAnalysis {
    yield_points: Vec<BasicBlock>,
    poll_call_sites: Vec<(BasicBlock, DefId)>,
    dispatch_switch_bb: Option<BasicBlock>,
    suspension_point_count: usize,
    return_blocks: Vec<BasicBlock>,
}

impl CoroutineAnalysis {
    fn run<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>) -> Self {
        let mut yield_points = Vec::new();
        let mut poll_call_sites = Vec::new();
        let mut return_blocks = Vec::new();

        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            match &bb_data.terminator().kind {
                TerminatorKind::Yield { .. } => {
                    yield_points.push(bb);
                }
                TerminatorKind::Call { func, .. } => {
                    if let Some(callee_def_id) = resolve_callee_def_id(func) {
                        if is_future_poll(tcx, callee_def_id) {
                            poll_call_sites.push((bb, callee_def_id));
                        }
                    }
                }
                TerminatorKind::Return => {
                    return_blocks.push(bb);
                }
                _ => {}
            }
        }

        let (dispatch_switch_bb, suspension_point_count) =
            find_dispatch_switch(body);

        CoroutineAnalysis {
            yield_points,
            poll_call_sites,
            dispatch_switch_bb,
            suspension_point_count,
            return_blocks,
        }
    }

    fn emit_diagnostics(&self, tcx: TyCtxt<'_>, body: &Body<'_>, fn_name: &str) {
        let span = body.span;
        tcx.dcx().span_note(
            span,
            format!(
                "warp_cooperative: `{fn_name}` — \
                 {yp} yield(s), {pc} poll(s), {sp} suspension(s), {rb} return(s)",
                yp = self.yield_points.len(),
                pc = self.poll_call_sites.len(),
                sp = self.suspension_point_count,
                rb = self.return_blocks.len(),
            ),
        );

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
    }
}

// ---------------------------------------------------------------------------
// Callee resolution helpers
// ---------------------------------------------------------------------------

fn resolve_callee_def_id<'tcx>(func: &Operand<'tcx>) -> Option<DefId> {
    if let Operand::Constant(c) = func {
        if let ty::FnDef(def_id, _) = *c.const_.ty().kind() {
            return Some(def_id);
        }
    }
    None
}

fn is_future_poll(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let Some(trait_def_id) = tcx.trait_of_item(def_id) else {
        return false;
    };
    let Some(future_trait) = tcx.lang_items().future_trait() else {
        return false;
    };
    if trait_def_id != future_trait {
        return false;
    }
    let poll_sym = Symbol::intern("poll");
    tcx.item_name(def_id) == poll_sym
}

// ---------------------------------------------------------------------------
// Dispatch switch detection
// ---------------------------------------------------------------------------

const RESERVED_VARIANTS: u128 = 3;

fn find_dispatch_switch(body: &Body<'_>) -> (Option<BasicBlock>, usize) {
    for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
        if let TerminatorKind::SwitchInt { discr, targets, .. } = &bb_data.terminator().kind {
            if is_discriminant_of_self(discr, bb_data) {
                let suspension_points = targets
                    .iter()
                    .filter(|(val, _)| *val >= RESERVED_VARIANTS)
                    .count();
                return (Some(bb), suspension_points);
            }
        }
    }
    (None, 0)
}

fn is_discriminant_of_self(discr: &Operand<'_>, bb_data: &BasicBlockData<'_>) -> bool {
    let scrutinee_local = match discr {
        Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
            place.local
        }
        _ => return false,
    };

    for stmt in &bb_data.statements {
        if let StatementKind::Assign(box (lhs, Rvalue::Discriminant(place))) = &stmt.kind {
            if lhs.local == scrutinee_local
                && lhs.projection.is_empty()
                && place.local == Local::from(1u32)
            {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Phase 2: Discriminant broadcast at the dispatch switch
// ---------------------------------------------------------------------------
//
// BEFORE (post-StateTransform):
//   bb0: {
//       _discr = discriminant((*_1));
//       switchInt(_discr) -> [0: .., 3: .., ...]
//   }
//
// AFTER:
//   bb0: {
//       _discr = discriminant((*_1));
//       // → falls through to activemask block
//   }
//   bb_activemask: {
//       asm!("activemask.b32 $0", out(reg32) _mask);
//       → bb_shfl
//   }
//   bb_shfl: {
//       asm!("shfl.sync.idx.b32 $0, $1, 0, 31, $2",
//            out(reg32) _bc_discr, in(reg32) _discr, in(reg32) _mask);
//       → bb_switch
//   }
//   bb_switch: {
//       switchInt(_bc_discr) -> [0: .., 3: .., ...]
//   }

fn insert_discriminant_broadcast<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    dispatch_bb: BasicBlock,
) {
    let source_info = SourceInfo::outermost(body.span);
    let u32_ty = tcx.types.u32;

    // Find the discriminant local from the existing SwitchInt.
    let discr_local = {
        let term = body.basic_blocks[dispatch_bb].terminator();
        match &term.kind {
            TerminatorKind::SwitchInt { discr, .. } => {
                match discr {
                    Operand::Copy(p) | Operand::Move(p) if p.projection.is_empty() => {
                        p.local
                    }
                    _ => return, // Can't handle complex scrutinee
                }
            }
            _ => return,
        }
    };

    // Allocate new locals: _mask (u32), _bc_discr (u32).
    let mask_local = body.local_decls.push(LocalDecl::new(u32_ty, body.span));
    let bc_discr_local = body.local_decls.push(LocalDecl::new(u32_ty, body.span));

    let reg32 = InlineAsmRegOrRegClass::RegClass(
        InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32),
    );

    // --- Create bb_switch: the original SwitchInt on the broadcast discriminant ---
    let original_switch = body.basic_blocks[dispatch_bb].terminator().clone();
    let new_switch_kind = match original_switch.kind {
        TerminatorKind::SwitchInt { targets, .. } => {
            TerminatorKind::SwitchInt {
                discr: Operand::Move(bc_discr_local.into()),
                targets,
            }
        }
        _ => return,
    };
    let bb_switch = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator { source_info, kind: new_switch_kind }),
        false,
    ));

    // --- Create bb_shfl: shfl.sync.idx.b32 ---
    // Template: "shfl.sync.idx.b32 $0, $1, 0, 31, $2;"
    // Operands: out(reg32) _bc_discr, in(reg32) _discr, in(reg32) _mask
    let shfl_template: &[InlineAsmTemplatePiece] = alloc_template([
        InlineAsmTemplatePiece::String(Cow::Borrowed("shfl.sync.idx.b32 ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 0,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(", ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 1,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(", 0, 31, ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 2,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(";")),
    ]);

    let shfl_operands: Box<[InlineAsmOperand<'tcx>]> = Box::new([
        InlineAsmOperand::Out {
            reg: reg32,
            late: false,
            place: Some(bc_discr_local.into()),
        },
        InlineAsmOperand::In {
            reg: reg32,
            value: Operand::Copy(discr_local.into()),
        },
        InlineAsmOperand::In {
            reg: reg32,
            value: Operand::Copy(mask_local.into()),
        },
    ]);

    let bb_shfl = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::InlineAsm {
                asm_macro: InlineAsmMacro::Asm,
                template: shfl_template,
                operands: shfl_operands,
                options: InlineAsmOptions::NOSTACK,
                line_spans: &[],
                targets: Box::new([bb_switch]),
                unwind: UnwindAction::Unreachable,
            },
        }),
        false,
    ));

    // --- Create bb_activemask: activemask.b32 ---
    // Template: "activemask.b32 $0;"
    // Operands: out(reg32) _mask
    let activemask_template: &[InlineAsmTemplatePiece] = alloc_template([
        InlineAsmTemplatePiece::String(Cow::Borrowed("activemask.b32 ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 0,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(";")),
    ]);

    let activemask_operands: Box<[InlineAsmOperand<'tcx>]> = Box::new([
        InlineAsmOperand::Out {
            reg: reg32,
            late: false,
            place: Some(mask_local.into()),
        },
    ]);

    let bb_activemask = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::InlineAsm {
                asm_macro: InlineAsmMacro::Asm,
                template: activemask_template,
                operands: activemask_operands,
                options: InlineAsmOptions::NOSTACK,
                line_spans: &[],
                targets: Box::new([bb_shfl]),
                unwind: UnwindAction::Unreachable,
            },
        }),
        false,
    ));

    // --- Patch the dispatch block: keep the discriminant read, replace terminator ---
    // The dispatch block keeps its statements (including the discriminant assignment)
    // but now jumps to the activemask block instead of switching.
    body.basic_blocks_mut()[dispatch_bb].terminator_mut().kind =
        TerminatorKind::Goto { target: bb_activemask };
}

// ---------------------------------------------------------------------------
// Phase 4 (Rule 4): Barrier before Return
// ---------------------------------------------------------------------------
//
// BEFORE:
//   bb_ret: {
//       _0 = Poll::Ready(value);
//       return;
//   }
//
// AFTER:
//   bb_ret: {
//       _0 = Poll::Ready(value);
//       // → falls through to activemask block
//   }
//   bb_mask: {
//       asm!("activemask.b32 $0", out(reg32) _ret_mask);
//       → bb_barrier
//   }
//   bb_barrier: {
//       asm!("bar.warp.sync $0", in(reg32) _ret_mask);
//       → bb_actual_ret
//   }
//   bb_actual_ret: {
//       return;
//   }

fn insert_barrier_before_return<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    return_bb: BasicBlock,
) {
    let source_info = SourceInfo::outermost(body.span);
    let u32_ty = tcx.types.u32;

    let ret_mask_local = body.local_decls.push(LocalDecl::new(u32_ty, body.span));

    let reg32 = InlineAsmRegOrRegClass::RegClass(
        InlineAsmRegClass::Nvptx(NvptxInlineAsmRegClass::reg32),
    );

    // --- bb_actual_ret: just `return` ---
    let bb_actual_ret = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator { source_info, kind: TerminatorKind::Return }),
        false,
    ));

    // --- bb_barrier: bar.warp.sync ---
    let barrier_template: &[InlineAsmTemplatePiece] = alloc_template([
        InlineAsmTemplatePiece::String(Cow::Borrowed("bar.warp.sync ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 0,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(";")),
    ]);

    let barrier_operands: Box<[InlineAsmOperand<'tcx>]> = Box::new([
        InlineAsmOperand::In {
            reg: reg32,
            value: Operand::Copy(ret_mask_local.into()),
        },
    ]);

    let bb_barrier = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::InlineAsm {
                asm_macro: InlineAsmMacro::Asm,
                template: barrier_template,
                operands: barrier_operands,
                options: InlineAsmOptions::NOSTACK,
                line_spans: &[],
                targets: Box::new([bb_actual_ret]),
                unwind: UnwindAction::Unreachable,
            },
        }),
        false,
    ));

    // --- bb_mask: activemask ---
    let activemask_template: &[InlineAsmTemplatePiece] = alloc_template([
        InlineAsmTemplatePiece::String(Cow::Borrowed("activemask.b32 ")),
        InlineAsmTemplatePiece::Placeholder {
            operand_idx: 0,
            modifier: None,
            span: source_info.span,
        },
        InlineAsmTemplatePiece::String(Cow::Borrowed(";")),
    ]);

    let activemask_operands: Box<[InlineAsmOperand<'tcx>]> = Box::new([
        InlineAsmOperand::Out {
            reg: reg32,
            late: false,
            place: Some(ret_mask_local.into()),
        },
    ]);

    let bb_mask = body.basic_blocks_mut().push(BasicBlockData::new(
        Some(Terminator {
            source_info,
            kind: TerminatorKind::InlineAsm {
                asm_macro: InlineAsmMacro::Asm,
                template: activemask_template,
                operands: activemask_operands,
                options: InlineAsmOptions::NOSTACK,
                line_spans: &[],
                targets: Box::new([bb_barrier]),
                unwind: UnwindAction::Unreachable,
            },
        }),
        false,
    ));

    // --- Patch the return block: redirect to the mask block ---
    body.basic_blocks_mut()[return_bb].terminator_mut().kind =
        TerminatorKind::Goto { target: bb_mask };
}
