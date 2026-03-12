@echo off
REM Run the hello-gpu example — builds kernel + host automatically.
REM Prerequisites: Rust nightly (auto-installed via rust-toolchain.toml), NVIDIA GPU with CUDA driver.
cd /d "%~dp0examples\hello-gpu\host"
cargo run --release %*
