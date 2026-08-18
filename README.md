

https://github.com/user-attachments/assets/a3711f70-289c-4b5b-bf20-a85622f6515f


# TRELLIS.2 on Windows + Blackwell (RTX 50-series)

Patches, launchers, and a batch web UI for running [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
natively on Windows with an RTX 50-series (Blackwell, `sm_120`) GPU.

Upstream states *"the code is currently tested only on Linux"* and pins PyTorch 2.6 / CUDA 12.4.
On a Blackwell card that combination has no usable compute kernels, and several of the CUDA
extensions do not compile under MSVC. This repo carries the changes needed to close both gaps.

No WSL, no VM, no Docker.

---

## What was actually broken

### 1. Sparse attention had no Blackwell-capable backend

This is the blocker that stops the model running at all, and it is easy to miss.

TRELLIS.2 dispatches attention through a backend selector. **Dense** attention supports
`flash_attn`, `flash_attn_3`, `xformers`, `sdpa`, and `naive`. **Sparse** attention — which is
where this model does most of its work — supports only `flash_attn`, `flash_attn_3`, and
`xformers`.

On `sm_120`, none of those three has kernels in any released wheel:

```
NotImplementedError: No operator found for `memory_efficient_attention_forward`
  `fa3F@0.0.0` is not supported because:
      requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
  `cutlassF-pt` is not supported because:
      requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```

PyTorch's own `scaled_dot_product_attention` *does* support `sm_120`, so the fix is a
variable-length SDPA backend: [`src/sdpa_varlen.py`](src/sdpa_varlen.py). It pads ragged
sequences into a batch, builds a block-diagonal mask, and runs native SDPA. Packing is
index-based rather than a Python loop over sequences, because windowed attention produces
thousands of windows per call.

It is wired into all three sparse call sites (`full_attn`, windowed self-attention, windowed
cross-attention) and validated against a naive reference implementation across ragged,
equal-length, cross-attention, and single-sequence cases — **max error ~1e-7**.

### 2. CUDA extensions that only ever saw GCC

| Project | Files | Problem |
|---|---|---|
| `o-voxel` | 4 | Narrowing `size_t` → `int64_t` in tensor-shape initializer lists; non-standard `1e-6d` / `0.0d` literal suffixes |
| `FlexGEMM` | 2 (20 lines) | `x.data_ptr<T>()` on a dependent type needs `x.template data_ptr<T>()`; MSVC enforces this, GCC does not |

`CuMesh`, `nvdiffrast`, and `nvdiffrec` compile clean — no patches needed.

### 3. Version drift

Upstream pins almost nothing, so a fresh install pulls breaking majors:

- **transformers 5.x** nests the DINOv3 encoder — `DINOv3ViTModel.layer` moved to `.model.layer`.
  Patched to accept either layout.
- **opencv-python 5.x** ships with `OpenEXR: NO` and returns `None` from `cv2.imread` on the
  `.exr` HDRIs, with no error. **Pin `opencv-python-headless<5`.**
- **flash-attn / xformers** will silently drag `torch` to a version whose `nvcc` breaks the MSVC
  build. Install torch *after* them, or skip both (see above).

### 4. Gated Hugging Face models

- [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  — the image encoder. Genuinely required; accept the license.
- [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) — background removal. Only used for
  inputs **without** an alpha channel, but upstream loads it eagerly, so an inaccessible RMBG
  blocks the entire pipeline. Patched to degrade gracefully: RGBA inputs work regardless, and RGB
  inputs raise an explanatory error instead of a confusing one.

> **Do not use `HF_HOME` to relocate the model cache.** It also relocates the auth token, so
> `hf auth login` appears to work while every gated download returns 401. Use `HF_HUB_CACHE`.

---

## Verified environment

| | |
|---|---|
| GPU | RTX 5090 Laptop (24 GB, `sm_120`, capability 12.0) |
| OS | Windows 11 |
| Python | 3.11 (via `uv`; no conda) |
| PyTorch | 2.8.0+cu128 → later 2.11.0+cu128 |
| CUDA Toolkit | 12.8 |
| Compiler | MSVC 14.44 (VS 2022 Build Tools) |
| Blender | 5.1 (optional, for FBX export) |

Output: valid glTF 2.0 meshes at ~9.5M vertices / 19M faces with 4096² PBR textures.

---

## Install

Prerequisites — install in this order (CUDA integrates with MSBuild):

```bat
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --includeRecommended"
winget install --id Nvidia.CUDA --version 12.8 -e
```

Then:

```bat
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
cd TRELLIS.2
uv venv --python 3.11 .venv

:: Blackwell needs cu128. Do NOT use the cu124 wheels from upstream's setup.sh.
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install --python .venv/Scripts/python.exe imageio imageio-ffmpeg tqdm easydict "opencv-python-headless<5" ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard kornia timm
uv pip install --python .venv/Scripts/python.exe "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
```

Apply the patches from this repo:

```bat
git apply path\to\patches\trellis2.patch
copy path\to\src\sdpa_varlen.py trellis2\modules\sparse\attention\
```

Build the extensions (from a **Developer Command Prompt for VS 2022**, so `cl.exe` is on PATH):

```bat
set TORCH_CUDA_ARCH_LIST=12.0
uv pip install --python .venv/Scripts/python.exe ./o-voxel --no-build-isolation

git clone https://github.com/JeffreyXiang/FlexGEMM.git --recursive
cd FlexGEMM && git apply path\to\patches\flexgemm.patch && cd ..
uv pip install --python .venv/Scripts/python.exe ./FlexGEMM --no-build-isolation

:: these need no patches
uv pip install --python .venv/Scripts/python.exe ./CuMesh --no-build-isolation
uv pip install --python .venv/Scripts/python.exe ./nvdiffrast --no-build-isolation
uv pip install --python .venv/Scripts/python.exe ./nvdiffrec --no-build-isolation
```

Authenticate for the gated encoder:

```bat
.venv\Scripts\huggingface-cli.exe login
```

### Required environment

The launchers in [`scripts/`](scripts) set all of these for you. They resolve paths relative to
their own location, so place them in the directory *containing* your `TRELLIS.2` checkout:

```
<your dir>/
  run_studio.bat        <- from scripts/
  run_trellis2.bat
  TRELLIS.2/            <- the upstream clone, patched
  hf_cache/             <- created on first run
```

Override the output locations with `TRELLIS_OUTPUT_DIR` / `TRELLIS_INPUT_DIR` if you want them
elsewhere. The variables the launchers set are:

```bat
set ATTN_BACKEND=sdpa
set SPARSE_ATTN_BACKEND=sdpa
set SPARSE_CONV_BACKEND=flex_gemm
set HF_HUB_CACHE=<your cache dir>
set OPENCV_IO_ENABLE_OPENEXR=1
```

---

## TRELLIS.2 Studio

[`studio/trellis_studio.py`](studio/trellis_studio.py) — a batch-oriented Gradio UI, for when the
bundled single-image `app.py` isn't the workflow you want.

- **Queue** — drop in many images, run unattended; one failure doesn't stop the rest
- **Gallery** — every generation persisted with the exact seed/steps/guidance that made it
- **Turntable is opt-in** — the video render costs ~20 min on top of a ~2 min generation, so it is
  off by default. This is the single biggest speed lever.
- **Export** — GLB always; OBJ and FBX optional (FBX on by default)

FBX goes through Blender in background mode ([`studio/glb_to_fbx.py`](studio/glb_to_fbx.py)),
because trimesh cannot write the format:

```bat
blender --background --python glb_to_fbx.py -- input.glb output.fbx
```

### Hard-won defaults

- Inputs are copied to a staging directory on queue — Gradio's cache cleanup will otherwise
  delete an uploaded image out from under a 20-minute run.
- `source.png` / `meta.json` are written **before** the mesh, and the mesh before the video, so a
  cheap failure at the end can't discard expensive work.
- Filenames are sanitized. Windows strips trailing spaces when *creating* a path but not from a
  string handed to a subprocess, so `foo ` + ffmpeg = `No such file or directory`.

---

## Known issues

**Performance is unbenchmarked.** The SDPA backend is verified *correct*, not fast. It is not
flash-attention; expect slower generation than upstream's H100 figures.

**Sustained load is a real stress test.** On the test machine this surfaced pre-existing
platform instability (WHEA fatal hardware errors, including a `VIDEO_TDR_FAILURE`) that predated
TRELLIS by months. If you hit hard reboots under load, check Event Viewer for WHEA-Logger entries
before assuming it's this software — and update GPU drivers and system firmware.

**RGB inputs need RMBG access.** Without it, supply RGBA images with the background already
removed.

---

## License

MIT, matching upstream. TRELLIS.2 is © Microsoft Corporation; the patches here are derivative of
that MIT-licensed work. FlexGEMM, CuMesh, o-voxel, nvdiffrast, and nvdiffrec belong to their
respective authors.
