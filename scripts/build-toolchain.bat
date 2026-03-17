@echo off
setlocal enabledelayedexpansion

REM Build patched Rust toolchain with nvptx64 support.
REM Full pipeline: clone rustc -> apply patches -> build.
REM Can run from any command prompt — loads MSVC environment automatically.
REM On Linux/macOS, use build-toolchain.sh instead.
REM
REM Usage:
REM   scripts\build-toolchain.bat [--from-scratch]
REM
REM Prerequisites:
REM   - Visual Studio 2022 with C++ workload (MSVC 17.14+)
REM   - Python 3, git, bash (Git Bash)
REM   - ~30GB disk space

set "REPO_DIR=%~dp0.."
set "SCRIPTS_DIR=%~dp0"
set "RUSTC_SRC=%REPO_DIR%\rustc-src"
set "PATCHED_RUSTC=%REPO_DIR%\patched-rustc"
set "PATCH_DIR_STD=%REPO_DIR%\std-patches"
set "FROM_SCRATCH=0"

if "%1"=="--from-scratch" set "FROM_SCRATCH=1"

REM ============================================================
REM Step 1: Load MSVC environment
REM ============================================================

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo === Loading MSVC environment ===
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    where cl.exe >nul 2>&1
    if errorlevel 1 (
        echo ERROR: MSVC environment setup failed
        exit /b 1
    )
    echo MSVC loaded
) else (
    echo === MSVC already available ===
)

set CC=cl.exe
set CXX=cl.exe

REM ============================================================
REM Step 2: Clone rustc source
REM ============================================================

echo.
echo === Step 2: rustc source ===

if "%FROM_SCRATCH%"=="1" (
    if exist "%RUSTC_SRC%" rmdir /s /q "%RUSTC_SRC%"
)

set "RECLONED=0"
if not exist "%RUSTC_SRC%\compiler" (
    echo Cloning rust-lang/rust ^(depth 1^)...
    git clone --depth 1 https://github.com/rust-lang/rust.git "%RUSTC_SRC%"
    if errorlevel 1 (
        echo ERROR: git clone failed
        exit /b 1
    )
    echo Initializing required submodules...
    pushd "%RUSTC_SRC%"
    git submodule update --init --depth 1 library/backtrace library/stdarch src/llvm-project
    popd
    set "RECLONED=1"
) else (
    echo Already present ^(use --from-scratch to reclone^)
    set "NEED_SUBMODULES=0"
    if not exist "%RUSTC_SRC%\library\backtrace\src" set "NEED_SUBMODULES=1"
    if not exist "%RUSTC_SRC%\src\llvm-project\llvm" set "NEED_SUBMODULES=1"
    if "!NEED_SUBMODULES!"=="1" (
        echo Initializing missing submodules...
        pushd "%RUSTC_SRC%"
        git submodule update --init --depth 1 library/backtrace library/stdarch src/llvm-project
        popd
    )
)

REM ============================================================
REM Step 3: Apply compiler patches
REM ============================================================

echo.
echo === Step 3: Compiler patches ===

if "%FROM_SCRATCH%"=="1" (
    if exist "%PATCHED_RUSTC%" rmdir /s /q "%PATCHED_RUSTC%"
)
if "!RECLONED!"=="1" (
    if exist "%PATCHED_RUSTC%" rmdir /s /q "%PATCHED_RUSTC%"
)

if not exist "%PATCHED_RUSTC%\compiler" (
    echo Copying rustc-src to patched-rustc...
    robocopy "%RUSTC_SRC%" "%PATCHED_RUSTC%" /E /XD .git build /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    echo Applying rustc patches...
    bash "%SCRIPTS_DIR%apply-rustc-patches.sh" "%PATCHED_RUSTC:\=/%"
    if errorlevel 1 (
        echo ERROR: Rustc patch application failed
        exit /b 1
    )
) else (
    echo Already present ^(use --from-scratch to reapply^)
)

REM ============================================================
REM Step 4: Apply std patches
REM ============================================================

echo.
echo === Step 4: Std patches ===

set "PATCHED_STD=%PATCHED_RUSTC%\library\std"
set "MARKER=%PATCHED_STD%\.async_gpu_std_patched"

if "%FROM_SCRATCH%"=="1" (
    if exist "%MARKER%" del "%MARKER%"
)

if not exist "%MARKER%" (
    REM Reset std/src to stock
    if exist "%PATCHED_STD%\src" rmdir /s /q "%PATCHED_STD%\src"
    xcopy "%RUSTC_SRC%\library\std\src" "%PATCHED_STD%\src" /E /I /Q >nul

    REM Apply .patch files via bash
    echo Applying std patches...
    for %%p in ("%PATCH_DIR_STD%\*.patch") do (
        echo     [PATCH] %%~nxp
        bash -c "cd '%PATCHED_STD:\=/%' && patch -p1 --binary < '%PATCH_DIR_STD:\=/%/%%~nxp'"
        if errorlevel 1 (
            echo ERROR: Failed to apply patch %%~nxp
            exit /b 1
        )
    )

    REM Copy new .rs files
    echo Copying new source files...
    call :copy_new sys_alloc_cuda.rs              src\sys\alloc\cuda.rs
    call :copy_new sys_fs_cuda.rs                 src\sys\fs\cuda.rs
    call :copy_new sys_io_error_cuda.rs           src\sys\io\error\cuda.rs
    call :copy_new sys_stdio_cuda.rs              src\sys\stdio\cuda.rs
    call :copy_new sys_thread_local_gpu_threads.rs src\sys\thread_local\gpu_threads.rs

    echo Patched on %DATE% %TIME% > "%MARKER%"
) else (
    echo Already applied ^(use --from-scratch to reapply^)
)

REM ============================================================
REM Step 5: Write bootstrap.toml
REM ============================================================

echo.
echo === Step 5: Writing bootstrap.toml ===

cd /d "%PATCHED_RUSTC%"

> bootstrap.toml (
    echo change-id = "ignore"
    echo.
    echo [build]
    echo build = "x86_64-pc-windows-msvc"
    echo host = ["x86_64-pc-windows-msvc"]
    echo target = ["x86_64-pc-windows-msvc", "nvptx64-nvidia-cuda"]
    echo.
    echo [rust]
    echo incremental = false
    echo debug-assertions = false
    echo optimize = 2
    echo codegen-units = 1
    echo.
    echo [llvm]
    echo download-ci-llvm = false
    echo.
    echo [target.nvptx64-nvidia-cuda]
)

REM ============================================================
REM Step 6: Build
REM ============================================================

echo.
echo === Step 6: Building toolchain ===
echo Working dir: %CD%

set CI=
python x.py build --stage 1 compiler library
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo BUILD SUCCEEDED
echo.

REM Find sysroot
for %%s in (stage2 stage1) do (
    if exist "build\x86_64-pc-windows-msvc\%%s\bin\rustc.exe" (
        echo Sysroot: %CD%\build\x86_64-pc-windows-msvc\%%s
        echo Usage: set RUSTC=%CD%\build\x86_64-pc-windows-msvc\%%s\bin\rustc.exe
        goto :eof
    )
)
echo WARNING: Could not find sysroot
goto :eof

REM ============================================================
REM Helpers
REM ============================================================

:copy_new
REM %1 = source filename in std-patches, %2 = relative dest in patched-std
set "DEST_DIR=%PATCHED_STD%\%~dp2"
if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"
copy "%PATCH_DIR_STD%\%1" "%PATCHED_STD%\%2" >nul
echo     [NEW]   %2
goto :eof
