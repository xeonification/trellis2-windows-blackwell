@echo off
REM Launch TRELLIS.2 on Windows (RTX 5090 / Blackwell sm_120)
REM Usage:  run_trellis2.bat            -> example.py (image -> 3D, writes sample.glb + sample.mp4)
REM         run_trellis2.bat app.py     -> Gradio web demo

setlocal

set "SCRIPT=%~1"
if "%SCRIPT%"=="" set "SCRIPT=example.py"

REM Blackwell has no flash-attn/xformers kernels; use the patched torch-native SDPA backend.
set "ATTN_BACKEND=sdpa"
set "SPARSE_ATTN_BACKEND=sdpa"
REM Sparse conv via the FlexGEMM extension compiled for sm_120.
set "SPARSE_CONV_BACKEND=flex_gemm"
REM Keep the big model cache off the C: drive.
REM NOTE: use HF_HUB_CACHE, not HF_HOME -- HF_HOME also relocates the auth token,
REM which would break `hf auth login` and cause 401s on gated repos (e.g. DINOv3).
set "HF_HUB_CACHE=%~dp0hf_cache\hub"
REM Needed for the .exr HDRI environment maps.
set "OPENCV_IO_ENABLE_OPENEXR=1"
set "PYTHONUNBUFFERED=1"

cd /d "%~dp0TRELLIS.2"
"%~dp0TRELLIS.2\.venv\Scripts\python.exe" "%SCRIPT%"

REM Propagate python's exit code through endlocal so failures are visible to callers.
endlocal & exit /b %ERRORLEVEL%
