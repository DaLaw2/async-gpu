//! Proc macro for generating WarpFuture state machines from sequential code.
//!
//! `#[warp_async]` transforms a function containing `warp_print!()` calls
//! into a WarpFuture struct + impl, where each `warp_print!()` becomes an
//! INIT + WAIT state pair in the generated state machine.
//!
//! # Constraints
//!
//! - The function body must contain ONLY `warp_print!(buf, msg_bytes)` calls.
//! - No other statements, variable bindings, or expressions are supported.
//! - The first parameter must be `buf: *mut u8`.
//! - The return type determines `WarpFuture::Output`.
//! - The macro always generates `Ready(true)` on completion. For other return
//!   values, use a hand-written WarpFuture.
//!
//! # Example
//!
//! ```rust,ignore
//! #[warp_async]
//! unsafe fn my_pipeline(buf: *mut u8) -> bool {
//!     warp_print!(buf, b"First message");
//!     warp_print!(buf, b"Second message");
//! }
//! ```
//!
//! Generates a `MyPipeline` struct implementing `WarpFuture<Output = bool>`,
//! plus a `my_pipeline` kernel entry point that launches the state machine.
//!
//! # Message Length
//!
//! Messages longer than 32 bytes use a hybrid write strategy: lanes 0..31
//! write the first 32 bytes cooperatively, then lane 0 writes the remaining
//! bytes sequentially. Maximum message length is 56 bytes (PRINT_MAX_MSG_LEN).

use proc_macro::TokenStream;
use quote::{quote, format_ident};
use syn::{parse_macro_input, ItemFn, Stmt, Expr, ExprMacro, ReturnType};
use syn::parse::{Parse, ParseStream};

/// Parsed arguments from `warp_print!(buf_expr, msg_expr)`.
struct WarpPrintArgs {
    buf_ident: syn::Ident,
    msg_expr: syn::Expr,
}

impl Parse for WarpPrintArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let buf_ident: syn::Ident = input.parse()?;
        let _comma: syn::Token![,] = input.parse()?;
        let msg_expr: syn::Expr = input.parse()?;
        Ok(WarpPrintArgs { buf_ident, msg_expr })
    }
}

/// A recognized `warp_print!(buf, msg)` call extracted from the function body.
struct WarpPrintCall {
    msg_expr: proc_macro2::TokenStream,
}

/// Parse the function body and extract `warp_print!()` calls.
/// Returns Err with a compile error if non-warp_print statements are found.
fn extract_warp_prints(stmts: &[Stmt], buf_name: &str) -> Result<Vec<WarpPrintCall>, proc_macro2::TokenStream> {
    let mut calls = Vec::new();
    for stmt in stmts {
        match stmt {
            // warp_print!(buf, msg); — macro invocation with semicolon
            Stmt::Macro(stmt_mac) => {
                if let Some(call) = try_extract_from_macro(&stmt_mac.mac, buf_name)? {
                    calls.push(call);
                } else {
                    let mac_name = stmt_mac.mac.path.get_ident()
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(syn::Error::new_spanned(
                        &stmt_mac.mac.path,
                        format!(
                            "#[warp_async] only supports warp_print!() calls, found `{}!`",
                            mac_name,
                        ),
                    ).to_compile_error());
                }
            }
            // warp_print!(buf, msg) — expression without semicolon
            Stmt::Expr(Expr::Macro(ExprMacro { mac, .. }), _) => {
                if let Some(call) = try_extract_from_macro(mac, buf_name)? {
                    calls.push(call);
                } else {
                    let mac_name = mac.path.get_ident()
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    return Err(syn::Error::new_spanned(
                        &mac.path,
                        format!(
                            "#[warp_async] only supports warp_print!() calls, found `{}!`",
                            mac_name,
                        ),
                    ).to_compile_error());
                }
            }
            // Any other statement type is not supported
            other => {
                return Err(syn::Error::new_spanned(
                    quote::quote! { #other },
                    "#[warp_async] function body must contain only warp_print!() calls. \
                     Variable bindings, expressions, and other statements are not supported.",
                ).to_compile_error());
            }
        }
    }
    Ok(calls)
}

/// Try to parse `warp_print!(buf, msg)` from a `Macro` node.
fn try_extract_from_macro(mac: &syn::Macro, expected_buf: &str) -> Result<Option<WarpPrintCall>, proc_macro2::TokenStream> {
    if !mac.path.is_ident("warp_print") {
        return Ok(None);
    }

    let args: WarpPrintArgs = syn::parse2(mac.tokens.clone()).map_err(|e| {
        syn::Error::new_spanned(
            &mac.tokens,
            format!("warp_print! expects (buf, msg_bytes): {}", e),
        ).to_compile_error()
    })?;

    // Validate that the buf argument matches the function's buf parameter
    if args.buf_ident != expected_buf {
        return Err(syn::Error::new_spanned(
            &args.buf_ident,
            format!(
                "warp_print! first argument must be `{}`, found `{}`",
                expected_buf, args.buf_ident,
            ),
        ).to_compile_error());
    }

    let msg = &args.msg_expr;
    Ok(Some(WarpPrintCall { msg_expr: quote::quote! { #msg } }))
}

/// Convert snake_case to PascalCase.
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(ch) => ch.to_uppercase().chain(c).collect(),
            }
        })
        .collect()
}

#[proc_macro_attribute]
pub fn warp_async(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Validate no attribute arguments
    if !attr.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[warp_async] does not accept arguments",
        )
        .to_compile_error()
        .into();
    }

    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let struct_name = format_ident!("{}", to_pascal_case(&fn_name.to_string()));

    // Validate: function must have at least one parameter, first must be buf: *mut u8
    if input_fn.sig.inputs.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[warp_async] function must have at least one parameter: buf: *mut u8",
        )
        .to_compile_error()
        .into();
    }

    // Extract the first parameter name for buf validation
    let first_param = input_fn.sig.inputs.first().unwrap();
    let buf_name = match first_param {
        syn::FnArg::Typed(pat_type) => {
            if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                pat_ident.ident.to_string()
            } else {
                return syn::Error::new_spanned(
                    first_param,
                    "First parameter must be a simple identifier (e.g., `buf: *mut u8`)",
                )
                .to_compile_error()
                .into();
            }
        }
        _ => {
            return syn::Error::new_spanned(
                first_param,
                "First parameter must be `buf: *mut u8`",
            )
            .to_compile_error()
            .into();
        }
    };

    // Parse return type
    let return_type = match &input_fn.sig.output {
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, ty) => quote! { #ty },
    };

    // For now, we only support `-> bool` return type with Ready(true)
    // Future: parse the tail expression as the ready value
    let ready_value = quote! { true };

    // Extract warp_print! calls
    let calls = match extract_warp_prints(&input_fn.block.stmts, &buf_name) {
        Ok(calls) => calls,
        Err(err) => return err.into(),
    };
    let num_calls = calls.len();

    if num_calls == 0 {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "#[warp_async] requires at least one warp_print!() call",
        )
        .to_compile_error()
        .into();
    }

    let done_state = (num_calls * 2) as u32;

    // Generate match arms: each warp_print! → INIT + WAIT state pair
    let mut arms = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        let init_state = (i * 2) as u32;
        let wait_state = (i * 2 + 1) as u32;
        let next_after_wait = if i + 1 < num_calls {
            ((i + 1) * 2) as u32
        } else {
            done_state
        };
        let is_last = i + 1 == num_calls;
        let msg_expr = &call.msg_expr;

        // INIT: pop packet, cooperative payload write, submit
        // Messages >32 bytes: first 32 bytes written cooperatively by all lanes,
        // remaining bytes written by lane 0 sequentially.
        arms.push(quote! {
            #init_state => unsafe {
                let mut idx_raw: u32 = gpu_protocol::NULL_INDEX as u32;
                if wcx.is_leader() {
                    idx_raw = gpu_runtime::hostcall::hc_pop_free(self.buf) as u32;
                }

                let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
                if idx == gpu_protocol::NULL_INDEX {
                    return WarpPoll::Pending;
                }
                self.pkt_idx = idx;

                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let payload = pkt.add(gpu_protocol::PKT_OFF_PAYLOAD);

                let msg: &[u8] = #msg_expr;
                let msg_len = msg.len() as u32;

                if wcx.is_leader() {
                    core::ptr::write_volatile(payload as *mut u64, msg_len as u64);
                }

                // Cooperative write: all 32 lanes write first 32 bytes
                let msg_base = payload.add(8);
                let lid = wcx.lane_id;
                if lid < msg_len && lid < 32 {
                    core::ptr::write_volatile(
                        msg_base.add(lid as usize),
                        msg[lid as usize],
                    );
                }

                // Sequential write: lane 0 writes remaining bytes (33..msg_len)
                if wcx.is_leader() && msg_len > 32 {
                    let mut j: u32 = 32;
                    while j < msg_len {
                        core::ptr::write_volatile(
                            msg_base.add(j as usize),
                            msg[j as usize],
                        );
                        j += 1;
                    }
                }

                // Lane 0: write thread/block metadata at payload+64
                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        payload.add(64) as *mut u32,
                        core::arch::nvptx::_block_idx_x() as u32,
                    );
                    core::ptr::write_volatile(
                        payload.add(68) as *mut u32,
                        core::arch::nvptx::_thread_idx_x() as u32,
                    );
                }

                gpu_atomics::syncwarp(wcx.active_mask);

                if wcx.is_leader() {
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_ACTIVE_MASK) as *mut u32,
                        wcx.active_mask,
                    );
                    core::ptr::write_volatile(
                        pkt.add(gpu_protocol::PKT_OFF_SERVICE) as *mut u32,
                        gpu_protocol::SERVICE_PRINT,
                    );
                    gpu_atomics::sys_store_release_u32(
                        pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
                        gpu_protocol::CONTROL_FILLED,
                    );

                    let (num_shards, shard_off, _) =
                        gpu_runtime::hostcall::read_shard_info(self.buf as *const u8);
                    let ready_ptr = gpu_runtime::hostcall::get_ready_stack_ptr(
                        self.buf, num_shards, shard_off,
                    );
                    gpu_runtime::hostcall::hc_push(ready_ptr, self.buf, idx);
                    gpu_atomics::sys_fetch_add_u64(
                        self.buf.add(gpu_protocol::BUF_OFF_DOORBELL) as *mut u64,
                        1,
                    );

                    self.state = #wait_state;
                }

                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            }
        });

        // WAIT: convergent spin-wait, release, transition
        let on_ready = if is_last {
            quote! { return WarpPoll::Ready(#ready_value); }
        } else {
            quote! { return WarpPoll::Pending; }
        };

        arms.push(quote! {
            #wait_state => unsafe {
                let idx = broadcast_u32(wcx.active_mask, self.pkt_idx as u32) as u16;
                let pkt_off = gpu_runtime::hostcall::pkt_offset(self.buf as *const u8, idx);
                let pkt = self.buf.add(pkt_off);
                let ctrl = gpu_atomics::sys_spin_load_acquire_u32(
                    pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *const u32,
                );

                if ctrl & gpu_protocol::CONTROL_READY != 0 {
                    if wcx.is_leader() {
                        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
                        self.state = #next_after_wait;
                    }
                    gpu_atomics::syncwarp(wcx.active_mask);
                    #on_ready
                }

                WarpPoll::Pending
            }
        });
    }

    let output = quote! {
        struct #struct_name {
            buf: *mut u8,
            state: u32,
            pkt_idx: u16,
        }

        impl #struct_name {
            #[inline(always)]
            fn new(buf: *mut u8) -> Self {
                Self {
                    buf,
                    state: 0,
                    pkt_idx: gpu_protocol::NULL_INDEX,
                }
            }
        }

        unsafe impl gpu_runtime::warp_future::WarpFuture for #struct_name {
            type Output = #return_type;

            #[inline(always)]
            fn poll_warp(
                &mut self,
                wcx: &mut gpu_runtime::warp_future::WarpContext,
            ) -> gpu_runtime::warp_future::WarpPoll<#return_type> {
                use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

                let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

                match state {
                    #(#arms,)*
                    #done_state => WarpPoll::Ready(#ready_value),
                    _ => WarpPoll::Pending,
                }
            }
        }

        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn #fn_name(
            buf: *mut u8,
            result: *mut u32,
        ) {
            gpu_runtime::panic::gpu_panic_init(buf);

            let mut future = #struct_name::new(buf);
            let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

            if gpu_atomics::lane_id() == 0 {
                core::ptr::write_volatile(result, if ok { 1 } else { 0 });
            }
        }
    };

    output.into()
}
