# Build a patched Rust toolchain with warp-cooperative async/await support.
# For Windows only. On Linux/macOS, use build-toolchain.sh instead.
#
# Usage:
#   .\scripts\build-toolchain.ps1 [-FromScratch] [-PrintSysroot] [-Targets "t1,t2"]
#
# Prerequisites:
#   - Visual Studio 2019+ with C++ workload (MSVC + Windows SDK)
#   - Python 3, git
#   - ~30GB disk space

param(
    [switch]$FromScratch,
    [switch]$PrintSysroot,
    [string]$Targets = ""
)

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ScriptsDir = Join-Path $RepoDir "scripts"
$RustcSrc = Join-Path $RepoDir "rustc-src"
$PatchedRustc = Join-Path $RepoDir "patched-rustc"
$PatchDirStd = Join-Path $RepoDir "std-patches"

# Detect host triple
$HostTriple = "x86_64-pc-windows-msvc"
try {
    $ver = rustc -vV 2>$null | Where-Object { $_ -match "^host:" }
    if ($ver -match "host:\s+(.+)") { $HostTriple = $Matches[1].Trim() }
} catch {}

if (-not $Targets) { $Targets = "$HostTriple,nvptx64-nvidia-cuda" }

# ============================================================
# Print sysroot mode
# ============================================================

if ($PrintSysroot) {
    foreach ($stage in "stage2", "stage1") {
        foreach ($dir in "$PatchedRustc\build\host\$stage", "$PatchedRustc\build\$HostTriple\$stage") {
            if (Test-Path $dir) { Write-Output $dir; exit 0 }
        }
    }
    Write-Error "No sysroot found. Run build-toolchain.ps1 first."
    exit 1
}

Write-Host ""
Write-Host "  async_gpu - Patched Toolchain Builder (Windows)" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Step 1: Ensure rustc-src/ exists
# ============================================================

Write-Host "=== Step 1: rustc source ===" -ForegroundColor Yellow

if ($FromScratch -or -not (Test-Path "$RustcSrc\compiler")) {
    if (Test-Path $RustcSrc) { Remove-Item $RustcSrc -Recurse -Force }
    Write-Host "  Cloning rust-lang/rust (depth 1)..."
    git clone --depth 1 https://github.com/rust-lang/rust.git $RustcSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
} else {
    Write-Host "  Already present (use -FromScratch to reclone)"
}

$version = Get-Content "$RustcSrc\src\version" -ErrorAction SilentlyContinue
Write-Host "  Version: $version"
Write-Host ""

# ============================================================
# Step 2: Create/refresh patched-rustc/ with compiler patches
# ============================================================

Write-Host "=== Step 2: Compiler patches ===" -ForegroundColor Yellow

if ($FromScratch -or -not (Test-Path "$PatchedRustc\compiler")) {
    if (Test-Path $PatchedRustc) { Remove-Item $PatchedRustc -Recurse -Force }
    Write-Host "  Copying rustc-src -> patched-rustc..."
    # Use robocopy (fast, excludes .git and build)
    robocopy $RustcSrc $PatchedRustc /E /XD .git build /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
    Write-Host "  Applying rustc patches..."
    bash "$ScriptsDir\apply-rustc-patches.sh" ($PatchedRustc -replace '\\', '/')
    if ($LASTEXITCODE -ne 0) { throw "Rustc patch application failed" }
} else {
    Write-Host "  Already present (use -FromScratch to reapply)"
}
Write-Host ""

# ============================================================
# Step 3: Apply std patches into patched-rustc/library/std/
# ============================================================

Write-Host "=== Step 3: Std patches ===" -ForegroundColor Yellow

$PatchedStd = Join-Path $PatchedRustc "library\std"
$Marker = Join-Path $PatchedStd ".async_gpu_std_patched"

if ($FromScratch -or -not (Test-Path $Marker)) {
    # Reset std/src to stock
    $StdSrc = Join-Path $PatchedStd "src"
    if (Test-Path $StdSrc) { Remove-Item $StdSrc -Recurse -Force }
    Copy-Item -Path "$RustcSrc\library\std\src" -Destination $StdSrc -Recurse

    # Apply patches (using git apply which handles a/ b/ prefixes)
    Push-Location $PatchedStd
    foreach ($pf in Get-ChildItem "$PatchDirStd\*.patch") {
        Write-Host "    [PATCH] $($pf.Name)"
        git apply --directory=. $pf.FullName 2>$null
        if ($LASTEXITCODE -ne 0) {
            # Fallback: try patch command via bash
            bash -c "cd '$($PatchedStd -replace '\\','/')' && patch -p1 --binary < '$($pf.FullName -replace '\\','/')'"
            if ($LASTEXITCODE -ne 0) { throw "Failed to apply patch: $($pf.Name)" }
        }
    }
    Pop-Location

    # Copy new .rs files
    $newFiles = @{
        "sys_alloc_cuda.rs"              = "src\sys\alloc\cuda.rs"
        "sys_fs_cuda.rs"                 = "src\sys\fs\cuda.rs"
        "sys_io_error_cuda.rs"           = "src\sys\io\error\cuda.rs"
        "sys_stdio_cuda.rs"              = "src\sys\stdio\cuda.rs"
        "sys_thread_local_gpu_threads.rs" = "src\sys\thread_local\gpu_threads.rs"
    }
    foreach ($entry in $newFiles.GetEnumerator()) {
        $dest = Join-Path $PatchedStd $entry.Value
        $destDir = Split-Path $dest -Parent
        if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
        Copy-Item (Join-Path $PatchDirStd $entry.Key) $dest
        Write-Host "    [NEW]   $($entry.Value)"
    }

    "Patched on $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ')" | Set-Content $Marker
} else {
    Write-Host "  Already applied (use -FromScratch to reapply)"
}
Write-Host ""

# ============================================================
# Step 4: Set up MSVC environment and build
# ============================================================

Write-Host "=== Step 4: Building patched toolchain ===" -ForegroundColor Yellow
Write-Host "  Host: $HostTriple"
Write-Host "  Targets: $Targets"

# Load MSVC environment via vswhere + vcvarsall
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "  Loading MSVC environment..."
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Install Visual Studio with C++ workload."
    }
    $vsInstall = & $vswhere -latest -property installationPath
    $vcvarsall = Join-Path $vsInstall "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        throw "vcvarsall.bat not found at $vcvarsall"
    }

    # Capture environment from vcvarsall
    $envLines = cmd /c "`"$vcvarsall`" x64 >nul 2>&1 && set" 2>$null
    foreach ($line in $envLines) {
        if ($line -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }

    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        $clVer = (cl.exe 2>&1 | Out-String).Trim().Split("`n")[0]
        Write-Host "  MSVC: $clVer"
    } else {
        throw "Failed to load MSVC environment"
    }
} else {
    Write-Host "  MSVC already available"
}

# Generate bootstrap.toml
$targetList = ($Targets -split ',') | ForEach-Object { "`"$_`"" }
$targetStr = $targetList -join ', '

$bootstrapToml = @"
change-id = "ignore"

[build]
build = "$HostTriple"
host = ["$HostTriple"]
target = [$targetStr]

[rust]
incremental = true
debug-assertions = true
optimize = 1

[llvm]
download-ci-llvm = true

[target.nvptx64-nvidia-cuda]
"@

$bootstrapToml | Set-Content (Join-Path $PatchedRustc "bootstrap.toml") -Encoding UTF8
Write-Host "  Generated bootstrap.toml"
Write-Host ""
Write-Host "  Starting x.py build (this may take 20-60 minutes)..." -ForegroundColor Cyan
Write-Host ""

# Build
Push-Location $PatchedRustc
try {
    python x.py build compiler library 2>&1 | Tee-Object -FilePath "$RepoDir\.research\toolchain-build.log"
    if ($LASTEXITCODE -ne 0) { throw "x.py build failed" }
} catch {
    Write-Host ""
    Write-Host "  BUILD FAILED" -ForegroundColor Red
    Write-Host "  Log: .research\toolchain-build.log"
    Pop-Location
    exit 1
}
Pop-Location

Write-Host ""
Write-Host "  BUILD SUCCEEDED" -ForegroundColor Green
Write-Host ""

# Find sysroot
$sysroot = $null
foreach ($stage in "stage2", "stage1") {
    foreach ($dir in "$PatchedRustc\build\$HostTriple\$stage", "$PatchedRustc\build\host\$stage") {
        if (Test-Path $dir) { $sysroot = $dir; break }
    }
    if ($sysroot) { break }
}

if ($sysroot) {
    Write-Host "Sysroot: $sysroot"
    $nvptxLib = Join-Path $sysroot "lib\rustlib\nvptx64-nvidia-cuda\lib"
    if (Test-Path $nvptxLib) {
        Write-Host "nvptx64: PRESENT" -ForegroundColor Green
    } else {
        Write-Host "WARNING: nvptx64 libs not found" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  `$env:RUSTC = `"$sysroot\bin\rustc.exe`""
} else {
    Write-Host "WARNING: Could not find sysroot. Check: $PatchedRustc\build\" -ForegroundColor Yellow
}
