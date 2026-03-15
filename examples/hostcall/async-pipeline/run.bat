@echo off
REM Run the async-pipeline example.
cd /d "%~dp0host"
cargo run --release %*
