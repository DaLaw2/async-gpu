# async_gpu - environment check and setup guide (Windows)
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Checks all prerequisites for building and running async_gpu examples.
# Does NOT modify system state - only reports what is present/missing.

$ErrorActionPreference = "Continue"
$Issues = 0

function Write-Ok($msg)   { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $msg }
function Write-Warn($msg) { Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Write-Fail($msg) { Write-Host "[MISSING] " -ForegroundColor Red -NoNewline; Write-Host $msg; $script:Issues++ }

Write-Host "====================================="
Write-Host "  async_gpu - Environment Check"
Write-Host "====================================="
Write-Host ""

# -- 1. Rust toolchain --
Write-Host "--- Rust Toolchain ---"

$rustup = Get-Command rustup -ErrorAction SilentlyContinue
if ($rustup) {
    $ver = (rustup --version 2>$null | Select-Object -First 1) -replace '.*(\d+\.\d+\.\d+).*','$1'
    Write-Ok "rustup $ver"
} else {
    Write-Fail "rustup not found - install from https://rustup.rs"
}

$rustc = Get-Command rustc -ErrorAction SilentlyContinue
if ($rustc) {
    $ver = (rustc --version) -replace 'rustc ',''
    Write-Ok "rustc $ver"
} else {
    Write-Fail "rustc not found"
}

# Check nightly from rust-toolchain.toml
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Nightly = ""
$tomlPath = Join-Path $RepoRoot "rust-toolchain.toml"
if (Test-Path $tomlPath) {
    $content = Get-Content $tomlPath -Raw
    if ($content -match 'channel\s*=\s*"([^"]+)"') {
        $Nightly = $Matches[1]
    }
}

if ($Nightly) {
    $installed = rustup toolchain list 2>$null
    if ($installed -match [regex]::Escape($Nightly)) {
        Write-Ok "Nightly toolchain: $Nightly"
    } else {
        Write-Fail "Nightly toolchain $Nightly not installed"
        Write-Host "      Run: rustup toolchain install $Nightly"
    }
}

# Check nvptx64 target
$targets = rustup target list --installed 2>$null
if ($targets -match "nvptx64-nvidia-cuda") {
    Write-Ok "nvptx64-nvidia-cuda target installed"
} else {
    Write-Fail "nvptx64-nvidia-cuda target not installed"
    Write-Host "      Run: rustup target add nvptx64-nvidia-cuda --toolchain $Nightly"
}

# Check rust-src
$components = rustup component list --installed 2>$null
if ($components -match "rust-src") {
    Write-Ok "rust-src component installed"
} else {
    Write-Fail "rust-src component not installed (needed for -Zbuild-std)"
    Write-Host "      Run: rustup component add rust-src --toolchain $Nightly"
}

# Check llvm-bitcode-linker
if ($components -match "llvm-bitcode-linker") {
    Write-Ok "llvm-bitcode-linker component installed"
} elseif (Get-Command llvm-bitcode-linker -ErrorAction SilentlyContinue) {
    Write-Ok "llvm-bitcode-linker found in PATH"
} else {
    Write-Fail "llvm-bitcode-linker not installed (needed for nvptx64 linking)"
    Write-Host "      Run: rustup component add llvm-bitcode-linker --toolchain $Nightly"
}

Write-Host ""

# -- 2. CUDA --
Write-Host "--- CUDA ---"

$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $gpuName = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    $driverVer = (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    Write-Ok "GPU: $gpuName (driver $driverVer)"

    $computeCap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    $sm = [int]($computeCap -replace '\.','')
    if ($sm -ge 70) {
        Write-Ok "Compute capability: $computeCap (SM 70+ required)"
    } else {
        Write-Fail "Compute capability $computeCap is below minimum (SM 70+ required)"
    }
} else {
    Write-Fail "nvidia-smi not found - NVIDIA GPU driver not installed"
}

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    $cudaVer = (nvcc --version 2>$null | Select-String "release") -replace '.*release\s+([\d.]+).*','$1'
    Write-Ok "CUDA toolkit: $cudaVer"
} else {
    Write-Warn "nvcc not found - CUDA toolkit may not be in PATH"
}

Write-Host ""

# -- 3. Build tools --
Write-Host "--- Build Tools ---"

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $ver = (cargo --version) -replace 'cargo ',''
    Write-Ok "cargo $ver"
} else {
    Write-Fail "cargo not found"
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $ver = (git --version) -replace 'git version ',''
    Write-Ok "git $ver"
} else {
    Write-Fail "git not found"
}

Write-Host ""

# -- Summary --
Write-Host "====================================="
if ($Issues -eq 0) {
    Write-Host "All prerequisites met!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Quick start:"
    Write-Host "  cargo run --manifest-path examples\hello-gpu\host\Cargo.toml"
    Write-Host ""
    Write-Host "Or use xtask to build all GPU kernels:"
    Write-Host "  cargo xtask gpu-build"
} else {
    Write-Host "$Issues issue(s) found." -ForegroundColor Red -NoNewline
    Write-Host " Fix the items above, then re-run this script."
}
Write-Host "====================================="
