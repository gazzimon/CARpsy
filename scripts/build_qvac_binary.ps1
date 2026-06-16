# build_qvac_binary.ps1
# Compiles a patched llama-finetune-lora.exe with Flash Attention disabled for training.
#
# Prerequisites (run once):
#   winget install --id Kitware.CMake --silent
#   winget install --id Microsoft.VisualStudio.2022.BuildTools --silent `
#     --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
#
# The patch: examples/training/finetune-lora.cpp forces LLAMA_FLASH_ATTN_TYPE_DISABLED
# so the binary never auto-enables Flash Attention during training (backward pass unsupported).

param(
    [string]$RepoPath  = "C:\Users\User\Documents\qvac-fabric-llm.cpp",
    [string]$OutputDir = "C:\Users\User\Documents\llama-b7349-patched"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Find cmake
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue)?.Source
if (-not $cmake) {
    $cmake = "C:\Program Files\CMake\bin\cmake.exe"
    if (-not (Test-Path $cmake)) {
        Write-Error "cmake not found. Run: winget install --id Kitware.CMake --silent"
        exit 1
    }
}
Write-Host "[cmake] $cmake"

# Find MSVC via vswhere
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "Visual Studio Build Tools not found. Install with:"
    Write-Error "  winget install Microsoft.VisualStudio.2022.BuildTools"
    exit 1
}
$vsInstallPath = & $vswhere -latest -property installationPath
Write-Host "[vs]    $vsInstallPath"

$buildDir = Join-Path $RepoPath "build-patched"

Write-Host ""
Write-Host "=== Configuring QVAC Fabric (Flash Attn disabled patch) ==="
& $cmake `
    -S $RepoPath `
    -B $buildDir `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_SERVER=OFF `
    -DGGML_CUDA=OFF `
    -DGGML_METAL=OFF `
    -DGGML_VULKAN=OFF `
    -DGGML_OPENCL=OFF `
    -DGGML_SYCL=OFF `
    -A x64

if ($LASTEXITCODE -ne 0) { Write-Error "cmake configure failed"; exit 1 }

Write-Host ""
Write-Host "=== Building llama-finetune-lora ==="
& $cmake --build $buildDir --config Release --target llama-finetune-lora -- /m

if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

# Copy the binary and required DLLs
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$builtExe = Join-Path $buildDir "bin\Release\llama-finetune-lora.exe"
Copy-Item $builtExe $OutputDir -Force
Write-Host ""
Write-Host "=== Build complete ==="
Write-Host "Binary: $OutputDir\llama-finetune-lora.exe"
Write-Host ""
Write-Host "Update your .env:"
Write-Host "  FABRIC_PATH=$OutputDir\llama-finetune-lora.exe"
Write-Host ""
Write-Host "Then run:  python scripts/04_run_finetune.py"
