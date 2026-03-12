//! Proc macro for generating WarpFuture state machines from sequential code.
//!
//! `#[warp_async]` transforms a function containing `warp_*!()` calls
//! into a WarpFuture struct + impl, where each `warp_*!()` becomes an
//! INIT + WAIT state pair in the generated state machine.
//!
//! # Supported Macros
//!
//! | Macro | Service | Args | Returns |
//! |-------|---------|------|---------|
//! | `warp_print!(buf, msg_bytes)` | PRINT | message bytes | — |
//! | `warp_open!(buf, path_bytes, flags)` | OPEN | path, flags | fd: u64 |
//! | `warp_close!(buf, fd)` | CLOSE | fd | — |
//! | `warp_read!(buf, fd, max_bytes)` | READ | fd, max_bytes | bytes_read: u64 |
//! | `warp_write!(buf, fd, data_bytes, data_len)` | WRITE | fd, data, len | bytes_written: u64 |
//! | `warp_bulk_read!(buf, fd, sb_offset, length)` | BULK_READ | fd, offset, len | bytes_read: u64 |
//! | `warp_bulk_write!(buf, fd, sb_offset, length)` | BULK_WRITE | fd, offset, len | bytes_written: u64 |
//!
//! # Variable Bindings
//!
//! Return values from hostcalls can be captured with `let`:
//! ```rust,ignore
//! let fd = warp_open!(buf, b"data.txt", FILE_OPEN_READ);
//! warp_close!(buf, fd);
//! ```
//! Each captured variable becomes a `u64` field in the generated struct.
//! Variables can be referenced in subsequent macro arguments.
//!
//! # Example
//!
//! ```rust,ignore
//! #[warp_async]
//! unsafe fn file_pipeline(buf: *mut u8) -> bool {
//!     let fd = warp_open!(buf, b"input.txt", FILE_OPEN_READ);
//!     warp_close!(buf, fd);
//!     warp_print!(buf, b"Done");
//! }
//! ```
//!
//! Generates a `FilePipeline` struct implementing `WarpFuture<Output = bool>`,
//! plus a `file_pipeline` kernel entry point.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Expr, ExprMacro, ItemFn, ReturnType, Stmt};

// ============================================================
// Service definitions
// ============================================================

#[derive(Clone, Copy)]
enum ServiceKind {
    Print,
    Open,
    Close,
    Read,
    Write,
    BulkRead,
    BulkWrite,
}

impl ServiceKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "warp_print" => Some(Self::Print),
            "warp_open" => Some(Self::Open),
            "warp_close" => Some(Self::Close),
            "warp_read" => Some(Self::Read),
            "warp_write" => Some(Self::Write),
            "warp_bulk_read" => Some(Self::BulkRead),
            "warp_bulk_write" => Some(Self::BulkWrite),
            _ => None,
        }
    }

    fn service_const(&self) -> proc_macro2::TokenStream {
        match self {
            Self::Print => quote! { gpu_protocol::SERVICE_PRINT },
            Self::Open => quote! { gpu_protocol::SERVICE_OPEN },
            Self::Close => quote! { gpu_protocol::SERVICE_CLOSE },
            Self::Read => quote! { gpu_protocol::SERVICE_READ },
            Self::Write => quote! { gpu_protocol::SERVICE_WRITE },
            Self::BulkRead => quote! { gpu_protocol::SERVICE_BULK_READ },
            Self::BulkWrite => quote! { gpu_protocol::SERVICE_BULK_WRITE },
        }
    }

    fn expected_args(&self) -> usize {
        match self {
            Self::Print => 1,     // msg_bytes
            Self::Open => 2,      // path_bytes, flags
            Self::Close => 1,     // fd
            Self::Read => 2,      // fd, max_bytes
            Self::Write => 3,     // fd, data_bytes, data_len
            Self::BulkRead => 3,  // fd, sb_offset, length
            Self::BulkWrite => 3, // fd, sb_offset, length
        }
    }

}

// ============================================================
// Parsed warp call
// ============================================================

struct WarpCall {
    service: ServiceKind,
    result_var: Option<syn::Ident>,
    /// Arguments after `buf` — their meaning depends on ServiceKind.
    args: Vec<syn::Expr>,
}

// ============================================================
// Generic comma-separated macro argument parser
// ============================================================

struct MacroArgs {
    buf_ident: syn::Ident,
    args: Vec<syn::Expr>,
}

impl Parse for MacroArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let buf_ident: syn::Ident = input.parse()?;
        let mut args = Vec::new();
        while input.peek(syn::Token![,]) {
            let _: syn::Token![,] = input.parse()?;
            if !input.is_empty() {
                args.push(input.parse()?);
            }
        }
        Ok(MacroArgs { buf_ident, args })
    }
}

// ============================================================
// Statement extraction
// ============================================================

/// Supported macro names for error messages.
const SUPPORTED_MACROS: &str =
    "warp_print, warp_open, warp_close, warp_read, warp_write, warp_bulk_read, warp_bulk_write";

/// Extract warp calls from the function body.
/// Handles both `warp_xxx!(buf, ...);` and `let var = warp_xxx!(buf, ...);`.
fn extract_warp_calls(
    stmts: &[Stmt],
    buf_name: &str,
) -> Result<Vec<WarpCall>, proc_macro2::TokenStream> {
    let mut calls = Vec::new();

    for stmt in stmts {
        match stmt {
            // `warp_xxx!(buf, args...);` — macro statement with semicolon
            Stmt::Macro(stmt_mac) => {
                let call = try_parse_macro_call(&stmt_mac.mac, buf_name, None)?;
                match call {
                    Some(c) => calls.push(c),
                    None => {
                        let name = macro_name_str(&stmt_mac.mac);
                        return Err(syn::Error::new_spanned(
                            &stmt_mac.mac.path,
                            format!(
                                "#[warp_async] unsupported macro `{}!`. Supported: {}",
                                name, SUPPORTED_MACROS,
                            ),
                        )
                        .to_compile_error());
                    }
                }
            }

            // `warp_xxx!(buf, args...)` — expression without semicolon
            Stmt::Expr(Expr::Macro(ExprMacro { mac, .. }), _) => {
                let call = try_parse_macro_call(mac, buf_name, None)?;
                match call {
                    Some(c) => calls.push(c),
                    None => {
                        let name = macro_name_str(mac);
                        return Err(syn::Error::new_spanned(
                            &mac.path,
                            format!(
                                "#[warp_async] unsupported macro `{}!`. Supported: {}",
                                name, SUPPORTED_MACROS,
                            ),
                        )
                        .to_compile_error());
                    }
                }
            }

            // `let var = warp_xxx!(buf, args...);` — local binding
            Stmt::Local(local) => {
                let var_name = extract_local_ident(local)?;
                let init_expr = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(
                        local,
                        "#[warp_async] `let` bindings must have an initializer: \
                         `let var = warp_xxx!(buf, ...);`",
                    )
                    .to_compile_error()
                })?;

                // The initializer must be a macro call
                if let Expr::Macro(ExprMacro { mac, .. }) = init_expr.expr.as_ref() {
                    let call = try_parse_macro_call(mac, buf_name, Some(var_name))?;
                    match call {
                        Some(c) => calls.push(c),
                        None => {
                            let name = macro_name_str(mac);
                            return Err(syn::Error::new_spanned(
                                &mac.path,
                                format!(
                                    "#[warp_async] unsupported macro `{}!`. Supported: {}",
                                    name, SUPPORTED_MACROS,
                                ),
                            )
                            .to_compile_error());
                        }
                    }
                } else {
                    return Err(syn::Error::new_spanned(
                        &init_expr.expr,
                        "#[warp_async] `let` bindings must initialize from a warp_*!() call",
                    )
                    .to_compile_error());
                }
            }

            // Any other statement type
            other => {
                return Err(syn::Error::new_spanned(
                    quote! { #other },
                    format!(
                        "#[warp_async] function body must contain only warp_*!() calls and \
                         `let var = warp_*!()` bindings. Supported macros: {}",
                        SUPPORTED_MACROS,
                    ),
                )
                .to_compile_error());
            }
        }
    }

    Ok(calls)
}

/// Extract the identifier from a `let` pattern. Only simple `let x = ...` supported.
fn extract_local_ident(
    local: &syn::Local,
) -> Result<syn::Ident, proc_macro2::TokenStream> {
    if let syn::Pat::Ident(pat_ident) = &local.pat {
        Ok(pat_ident.ident.clone())
    } else {
        Err(syn::Error::new_spanned(
            &local.pat,
            "#[warp_async] only simple `let var = ...` bindings are supported \
             (no destructuring, no patterns)",
        )
        .to_compile_error())
    }
}

/// Try to parse a macro invocation as a warp_xxx! call.
fn try_parse_macro_call(
    mac: &syn::Macro,
    expected_buf: &str,
    result_var: Option<syn::Ident>,
) -> Result<Option<WarpCall>, proc_macro2::TokenStream> {
    let name = macro_name_str(mac);
    let service = match ServiceKind::from_name(&name) {
        Some(s) => s,
        None => return Ok(None),
    };

    let parsed: MacroArgs = syn::parse2(mac.tokens.clone()).map_err(|e| {
        syn::Error::new_spanned(
            &mac.tokens,
            format!("{}! parse error: {}", name, e),
        )
        .to_compile_error()
    })?;

    // Validate buf argument
    if parsed.buf_ident != expected_buf {
        return Err(syn::Error::new_spanned(
            &parsed.buf_ident,
            format!(
                "{}! first argument must be `{}`, found `{}`",
                name, expected_buf, parsed.buf_ident,
            ),
        )
        .to_compile_error());
    }

    // Validate argument count
    let expected = service.expected_args();
    if parsed.args.len() != expected {
        return Err(syn::Error::new_spanned(
            &mac.tokens,
            format!(
                "{}! expects {} argument(s) after `buf`, found {}",
                name, expected, parsed.args.len(),
            ),
        )
        .to_compile_error());
    }

    Ok(Some(WarpCall {
        service,
        result_var,
        args: parsed.args,
    }))
}

/// Get the macro name as a string.
fn macro_name_str(mac: &syn::Macro) -> String {
    mac.path
        .get_ident()
        .map(|id| id.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ============================================================
// Code generation
// ============================================================

/// Generate the payload fill closure body for a given service.
fn gen_payload_fill(
    service: ServiceKind,
    args: &[syn::Expr],
    known_vars: &[syn::Ident],
) -> proc_macro2::TokenStream {
    // Generate captures for all known variables from self
    let captures: Vec<_> = known_vars
        .iter()
        .map(|v| quote! { let #v = self.#v; })
        .collect();

    match service {
        ServiceKind::Print => {
            // args[0] = msg_bytes (e.g., b"Hello")
            let msg = &args[0];
            quote! {
                #(#captures)*
                let msg: &[u8] = #msg;
                let msg_len = msg.len();
                let copy_len = if msg_len > gpu_protocol::PRINT_MAX_MSG_LEN {
                    gpu_protocol::PRINT_MAX_MSG_LEN
                } else {
                    msg_len
                };
                core::ptr::write_volatile(payload as *mut u64, copy_len as u64);
                let dst = payload.add(8);
                let mut __i = 0usize;
                while __i < copy_len {
                    core::ptr::write_volatile(dst.add(__i), msg[__i]);
                    __i += 1;
                }
                // Write block/thread metadata at payload+64
                core::ptr::write_volatile(
                    payload.add(64) as *mut u32,
                    core::arch::nvptx::_block_idx_x() as u32,
                );
                core::ptr::write_volatile(
                    payload.add(68) as *mut u32,
                    core::arch::nvptx::_thread_idx_x() as u32,
                );
            }
        }

        ServiceKind::Open => {
            // args[0] = path_bytes, args[1] = flags
            let path = &args[0];
            let flags = &args[1];
            quote! {
                #(#captures)*
                let path: &[u8] = #path;
                let flags = (#flags) as u64;
                let path_len = if path.len() > gpu_protocol::FILE_MAX_PATH_LEN {
                    gpu_protocol::FILE_MAX_PATH_LEN
                } else {
                    path.len()
                };
                let slot0 = (path_len as u64) | (flags << 32);
                core::ptr::write_volatile(payload as *mut u64, slot0);
                let dst = payload.add(8);
                let mut __i = 0usize;
                while __i < path_len {
                    core::ptr::write_volatile(dst.add(__i), path[__i]);
                    __i += 1;
                }
            }
        }

        ServiceKind::Close => {
            // args[0] = fd
            let fd = &args[0];
            quote! {
                #(#captures)*
                let fd_val = (#fd) as u64;
                core::ptr::write_volatile(payload as *mut u64, fd_val);
            }
        }

        ServiceKind::Read => {
            // args[0] = fd, args[1] = max_bytes
            let fd = &args[0];
            let max_bytes = &args[1];
            quote! {
                #(#captures)*
                let fd_val = (#fd) as u64;
                let max_val = (#max_bytes) as u64;
                core::ptr::write_volatile(payload as *mut u64, fd_val);
                core::ptr::write_volatile(payload.add(8) as *mut u64, max_val);
            }
        }

        ServiceKind::Write => {
            // args[0] = fd, args[1] = data_bytes, args[2] = data_len
            let fd = &args[0];
            let data = &args[1];
            let data_len = &args[2];
            quote! {
                #(#captures)*
                let fd_val = (#fd) as u64;
                let dlen = (#data_len) as u64;
                core::ptr::write_volatile(payload as *mut u64, fd_val);
                core::ptr::write_volatile(payload.add(8) as *mut u64, dlen);
                let data: &[u8] = #data;
                let dst = payload.add(16);
                let mut __i = 0usize;
                while __i < data.len() && __i < gpu_protocol::FILE_MAX_WRITE_LEN {
                    core::ptr::write_volatile(dst.add(__i), data[__i]);
                    __i += 1;
                }
            }
        }

        ServiceKind::BulkRead | ServiceKind::BulkWrite => {
            // args[0] = fd, args[1] = sb_offset, args[2] = length
            let fd = &args[0];
            let sb_off = &args[1];
            let length = &args[2];
            quote! {
                #(#captures)*
                let fd_val = (#fd) as u64;
                let sb_off_val = (#sb_off) as u64;
                let len_val = (#length) as u64;
                core::ptr::write_volatile(payload as *mut u64, fd_val);
                core::ptr::write_volatile(payload.add(8) as *mut u64, sb_off_val);
                core::ptr::write_volatile(payload.add(16) as *mut u64, len_val);
            }
        }
    }
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

// ============================================================
// Main proc macro
// ============================================================

#[proc_macro_attribute]
pub fn warp_async(attr: TokenStream, item: TokenStream) -> TokenStream {
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

    // ---- Parse function parameters ----
    if input_fn.sig.inputs.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[warp_async] function must have at least one parameter: buf: *mut u8",
        )
        .to_compile_error()
        .into();
    }

    // Collect all parameters: (name, type_tokens)
    let mut params: Vec<(syn::Ident, proc_macro2::TokenStream)> = Vec::new();
    for param in &input_fn.sig.inputs {
        match param {
            syn::FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                    let ty = &pat_type.ty;
                    params.push((pat_ident.ident.clone(), quote! { #ty }));
                } else {
                    return syn::Error::new_spanned(
                        param,
                        "#[warp_async] parameters must be simple identifiers",
                    )
                    .to_compile_error()
                    .into();
                }
            }
            _ => {
                return syn::Error::new_spanned(
                    param,
                    "#[warp_async] does not support `self` parameters",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let buf_name = params[0].0.to_string();

    // ---- Parse return type ----
    let is_bool_return;
    let (return_type, ready_value) = match &input_fn.sig.output {
        ReturnType::Default => {
            is_bool_return = false;
            (quote! { () }, quote! { () })
        }
        ReturnType::Type(_, ty) => {
            let ty_str = quote! { #ty }.to_string();
            if ty_str == "bool" {
                is_bool_return = true;
                (quote! { #ty }, quote! { true })
            } else {
                return syn::Error::new_spanned(
                    ty,
                    "#[warp_async] only supports `-> bool` or no return type (-> ()). \
                     For other return types, use a hand-written WarpFuture.",
                )
                .to_compile_error()
                .into();
            }
        }
    };

    // ---- Extract warp calls ----
    let calls = match extract_warp_calls(&input_fn.block.stmts, &buf_name) {
        Ok(c) => c,
        Err(e) => return e.into(),
    };

    if calls.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "#[warp_async] requires at least one warp_*!() call",
        )
        .to_compile_error()
        .into();
    }

    let num_calls = calls.len();
    let done_state = (num_calls * 2) as u32;

    // ---- Collect user variable fields (detect duplicates) ----
    let mut user_vars: Vec<syn::Ident> = Vec::new();
    for call in &calls {
        if let Some(ref var) = call.result_var {
            if user_vars.iter().any(|v| v == var) {
                return syn::Error::new_spanned(
                    var,
                    format!(
                        "#[warp_async] duplicate variable name `{}`. Each `let` binding \
                         must use a unique name (e.g., `fd_in`, `fd_out`).",
                        var,
                    ),
                )
                .to_compile_error()
                .into();
            }
            user_vars.push(var.clone());
        }
    }

    // ---- Generate struct fields ----
    let param_fields: Vec<_> = params
        .iter()
        .map(|(name, ty)| quote! { #name: #ty })
        .collect();
    let user_var_fields: Vec<_> = user_vars
        .iter()
        .map(|v| quote! { #v: u64 })
        .collect();

    // ---- Generate constructor ----
    let param_names: Vec<_> = params.iter().map(|(name, _)| name).collect();
    let user_var_inits: Vec<_> = user_vars
        .iter()
        .map(|v| quote! { #v: 0 })
        .collect();

    // ---- Generate match arms ----
    let mut arms = Vec::new();
    let mut known_vars_so_far: Vec<syn::Ident> = Vec::new();

    for (i, call) in calls.iter().enumerate() {
        let init_state = (i * 2) as u32;
        let wait_state = (i * 2 + 1) as u32;
        let next_after_wait = if i + 1 < num_calls {
            ((i + 1) * 2) as u32
        } else {
            done_state
        };
        let is_last = i + 1 == num_calls;

        let service_const = call.service.service_const();
        let payload_fill = gen_payload_fill(call.service, &call.args, &known_vars_so_far);

        // INIT state: warp_hostcall_submit
        arms.push(quote! {
            #init_state => unsafe {
                gpu_runtime::warp_future::warp_hostcall_submit(
                    self.buf, wcx, #service_const,
                    |payload| {
                        #payload_fill
                    },
                    #wait_state,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            }
        });

        // WAIT state: warp_hostcall_wait_u64
        let on_ready = if let Some(ref var) = call.result_var {
            if is_last {
                quote! {
                    if wcx.is_leader() { self.#var = val; }
                    return WarpPoll::Ready(#ready_value);
                }
            } else {
                quote! {
                    if wcx.is_leader() { self.#var = val; }
                    return WarpPoll::Pending;
                }
            }
        } else if is_last {
            quote! { return WarpPoll::Ready(#ready_value); }
        } else {
            quote! { return WarpPoll::Pending; }
        };

        arms.push(quote! {
            #wait_state => unsafe {
                if let Some(val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    #next_after_wait, &mut self.state,
                ) {
                    #on_ready
                }
                WarpPoll::Pending
            }
        });

        // Track this variable for subsequent payload fills
        if let Some(ref var) = call.result_var {
            known_vars_so_far.push(var.clone());
        }
    }

    // ---- Generate kernel entry point parameters ----
    let kernel_params: Vec<_> = params
        .iter()
        .map(|(name, ty)| quote! { #name: #ty })
        .collect();
    let struct_init_args: Vec<_> = params
        .iter()
        .map(|(name, _)| quote! { #name })
        .collect();

    // ---- Kernel result expression ----
    let kernel_result_expr = if is_bool_return {
        quote! { if __output { 1u32 } else { 0u32 } }
    } else {
        quote! { 1u32 } // () return → always success
    };

    // ---- Assemble output ----
    let output = quote! {
        struct #struct_name {
            #(#param_fields,)*
            state: u32,
            pkt_idx: u16,
            #(#user_var_fields,)*
        }

        impl #struct_name {
            #[inline(always)]
            fn new(#(#param_fields),*) -> Self {
                Self {
                    #(#param_names,)*
                    state: 0,
                    pkt_idx: gpu_protocol::NULL_INDEX,
                    #(#user_var_inits,)*
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
            #(#kernel_params,)*
            result: *mut u32,
        ) {
            gpu_runtime::panic::gpu_panic_init(buf);

            let mut future = #struct_name::new(#(#struct_init_args),*);
            let __output = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

            if gpu_atomics::lane_id() == 0 {
                core::ptr::write_volatile(result, #kernel_result_expr);
            }
        }
    };

    output.into()
}
