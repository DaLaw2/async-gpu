@echo off
REM Run the async-io example.
cd /d "%~dp0host"
cargo run --release %*
