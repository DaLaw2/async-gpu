@echo off
REM Run the parallel-search example.
cd /d "%~dp0host"
cargo run --release %*
