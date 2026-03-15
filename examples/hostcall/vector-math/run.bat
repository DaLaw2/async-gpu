@echo off
REM Run the vector-math example.
cd /d "%~dp0host"
cargo run --release %*
