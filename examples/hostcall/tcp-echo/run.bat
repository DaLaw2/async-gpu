@echo off
REM Run the tcp-echo example.
cd /d "%~dp0host"
cargo run --release %*
