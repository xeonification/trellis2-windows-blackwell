@echo off
REM Launch the TRELLIS.2 Studio web UI (batch queue + gallery) on http://127.0.0.1:7861

setlocal

REM Blackwell has no flash-attn/xformers kernels; use the patched torch-native SDPA backend.
set "ATTN_BACKEND=sdpa"
set "SPARSE_ATTN_BACKEND=sdpa"
REM Sparse conv via the FlexGEMM extension compiled for sm_120.
set "SPARSE_CONV_BACKEND=flex_gemm"
REM Use HF_HUB_CACHE (not HF_HOME) so the auth token stays where `hf auth login` put it.
set "HF_HUB_CACHE=%~dp0hf_cache\hub"
REM Needed for the .exr HDRI environment maps.
set "OPENCV_IO_ENABLE_OPENEXR=1"
set "PYTHONUNBUFFERED=1"

cd /d "%~dp0TRELLIS.2"
"%~dp0TRELLIS.2\.venv\Scripts\python.exe" trellis_studio.py

REM Propagate python's exit code through endlocal so failures are visible to callers.
endlocal & exit /b %ERRORLEVEL%
