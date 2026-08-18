# Launch TRELLIS.2 on Windows (RTX 5090 / Blackwell sm_120)
# Usage:  .\run_trellis2.ps1              -> runs example.py (image -> 3D, writes sample.glb + sample.mp4)
#         .\run_trellis2.ps1 app.py       -> runs the Gradio web demo
param([string]$Script = "example.py")

$ErrorActionPreference = "Stop"
$repo = Join-Path $PSScriptRoot "TRELLIS.2"

# Blackwell has no flash-attn/xformers kernels; use the patched torch-native SDPA backend.
$env:ATTN_BACKEND        = "sdpa"
$env:SPARSE_ATTN_BACKEND = "sdpa"
# Sparse conv via the FlexGEMM extension we compiled for sm_120.
$env:SPARSE_CONV_BACKEND = "flex_gemm"
# Keep the big model cache off the C: drive.
# NOTE: use HF_HUB_CACHE, not HF_HOME -- HF_HOME also relocates the auth token,
# which would break `hf auth login` and cause 401s on gated repos (e.g. DINOv3).
$env:HF_HUB_CACHE        = Join-Path $PSScriptRoot "hf_cache\hub"
# Needed for the .exr HDRI environment maps.
$env:OPENCV_IO_ENABLE_OPENEXR = "1"
$env:PYTHONUNBUFFERED    = "1"

Set-Location $repo
& "$repo\.venv\Scripts\python.exe" $Script
