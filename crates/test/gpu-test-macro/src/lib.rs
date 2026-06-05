//! `#[gpu_test]` proc macro — transforms a stub function into a `#[test]`
//! that loads the unified PTX and launches the eponymous kernel on the GPU.
//!
//! # Usage
//!
//! ```ignore
//! use gpu_test_macro::gpu_test;
//!
//! #[gpu_test]
//! fn test_vector_sum() {}
//!
//! // Expands to:
//! #[test]
//! fn test_vector_sum() {
//!     gpu_host::gpu::run_zero_param(
//!         gpu_host::ptx::KERNEL_STD,
//!         "test_vector_sum",
//!     ).expect("GPU test 'test_vector_sum' failed");
//! }
//! ```
//!
//! # Attributes
//!
//! - `#[gpu_test]` — launch with default 128 threads, (1,1,1) grid
//! - `#[gpu_test(threads = 256)]` — custom thread count
//! - `#[gpu_test(threads = 128, grid = (2, 1, 1))]` — custom grid

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Ident, ItemFn, LitInt, Token};

/// Parsed configuration from the `#[gpu_test(...)]` attribute.
struct GpuTestConfig {
    threads: u32,
    grid: (u32, u32, u32),
}

impl Default for GpuTestConfig {
    fn default() -> Self {
        Self {
            threads: 128,
            grid: (1, 1, 1),
        }
    }
}

impl Parse for GpuTestConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut config = GpuTestConfig::default();

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "threads" => {
                    let lit: LitInt = input.parse()?;
                    config.threads = lit.base10_parse()?;
                }
                "grid" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let x: LitInt = content.parse()?;
                    content.parse::<Token![,]>()?;
                    let y: LitInt = content.parse()?;
                    content.parse::<Token![,]>()?;
                    let z: LitInt = content.parse()?;
                    config.grid = (x.base10_parse()?, y.base10_parse()?, z.base10_parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown gpu_test attribute: `{other}` (expected `threads` or `grid`)"
                        ),
                    ));
                }
            }

            // Consume optional trailing comma
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(config)
    }
}

/// Mark a function as a GPU test.
///
/// The function body is ignored — the macro generates a `#[test]` function
/// that loads the unified kernel PTX and launches the kernel whose name
/// matches the function name.
///
/// The corresponding `extern "gpu-kernel" fn <name>()` must exist in the
/// kernel crate (gpu-kernel-std) and be compiled into the PTX.
///
/// # Examples
///
/// ```ignore
/// // Basic: 128 threads, 1 block
/// #[gpu_test]
/// fn test_arithmetic() {}
///
/// // Custom thread count
/// #[gpu_test(threads = 256)]
/// fn test_multithread() {}
///
/// // Custom grid
/// #[gpu_test(threads = 128, grid = (2, 1, 1))]
/// fn test_multiblock() {}
/// ```
#[proc_macro_attribute]
pub fn gpu_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_macro_input!(attr as GpuTestConfig);
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let kernel_name = fn_name.to_string();

    let threads = config.threads;
    let grid_x = config.grid.0;
    let grid_y = config.grid.1;
    let grid_z = config.grid.2;

    let fail_msg = format!("GPU test '{}' failed", kernel_name);

    // Generate code that tries to load cubin from a well-known path at runtime
    // for fast loading (sub-second), falling back to PTX JIT if unavailable.
    let expanded = quote! {
        #[test]
        fn #fn_name() {
            // Try to load cubin from the gpu-host crate directory for fast loading.
            // Falls back to PTX JIT (slow, 10+ minutes) if cubin is not found.
            let cubin = {
                let manifest = env!("CARGO_MANIFEST_DIR");
                let cubin_path = std::path::Path::new(manifest)
                    .join("../../core/gpu-host/kernel_std.cubin");
                std::fs::read(&cubin_path).unwrap_or_default()
            };

            gpu_host::gpu::run_zero_param_with_cubin(
                gpu_host::ptx::KERNEL_STD,
                &cubin,
                #kernel_name,
                #threads,
                (#grid_x, #grid_y, #grid_z),
            ).expect(#fail_msg);
        }
    };

    expanded.into()
}
