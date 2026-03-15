@echo off
REM Run the hello-gpu example.
cd /d "%~dp0host"
cargo run --release %*
