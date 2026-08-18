"""
TRELLIS.2 Studio -- a batch-oriented web UI for image-to-3D generation.

Adds what the bundled app.py lacks: a multi-image queue that runs unattended, and a
persistent gallery of past generations with the settings used to produce them.

Launch via:  D:\\AI\\Trellis2\\run_studio.bat
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import re
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

REPO_DIR = Path(__file__).parent
# Default to siblings of the TRELLIS.2 checkout; override with env vars if you want them elsewhere.
OUTPUT_DIR = Path(os.environ.get("TRELLIS_OUTPUT_DIR", REPO_DIR.parent / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Gradio purges its upload cache on a timer, and a 1536 generation outlives any sane timer.
# Copy every queued image somewhere we control so a long run can't lose its own input.
INPUT_DIR = Path(os.environ.get("TRELLIS_INPUT_DIR", REPO_DIR.parent / "queue_inputs"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SEED = 2**31 - 1

# Rough wall-clock cost per resolution on a 24GB card, used to warn before long runs.
RESOLUTION_INFO = {
    "512":  ("512\u00b3",  "~5 min",  "512"),
    "1024": ("1024\u00b3", "~20 min", "1024_cascade"),
    "1536": ("1536\u00b3", "~45 min, heavy VRAM", "1536_cascade"),
}

def find_blender():
    """FBX is proprietary and trimesh cannot write it, so we drive Blender in background mode."""
    roots = [Path(r"C:\Program Files\Blender Foundation"), Path(r"C:\Program Files (x86)\Blender Foundation")]
    found = []
    for r in roots:
        if r.is_dir():
            found += list(r.glob("Blender */blender.exe"))
    if found:
        return str(sorted(found)[-1])          # newest version wins
    return shutil.which("blender")


BLENDER_EXE = find_blender()
FBX_SCRIPT = REPO_DIR / "glb_to_fbx.py"
print(f"Blender for FBX export: {BLENDER_EXE or 'NOT FOUND (FBX export will be skipped)'}")

print("Loading TRELLIS.2 pipeline (this takes a couple of minutes)...")
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread(str(REPO_DIR / 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))
print("Pipeline ready.")


# ---------------------------------------------------------------- helpers

def safe_name(stem):
    """
    Make a filename safe to use as a Windows directory component.

    Windows strips trailing spaces and dots when *creating* a path, but a string handed to a
    child process (ffmpeg) keeps them -- so the folder we made and the folder ffmpeg looks for
    are different, and the write fails. Strip them here, plus the reserved characters.
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem)
    cleaned = cleaned.strip().rstrip('. ')
    return cleaned or "untitled"


def has_real_alpha(path):
    """The background remover is gated, so inputs without true transparency will fail."""
    try:
        im = Image.open(path)
        if im.mode != 'RGBA':
            return False
        return not (np.array(im)[:, :, 3] == 255).all()
    except Exception:
        return False


def queue_to_rows(queue):
    return [[i + 1, item['name'], item['resolution'], item['seed'],
             "yes" if item.get('render_turntable') else "no", item['status']]
            for i, item in enumerate(queue)]


def scan_gallery():
    """Each generation lives in its own timestamped folder with a meta.json."""
    entries = []
    for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        meta_path = d / 'meta.json'
        thumb = d / 'source.png'
        if d.is_dir() and meta_path.exists() and thumb.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            entries.append((str(thumb), f"{meta.get('name', d.name)} \u00b7 {meta.get('resolution', '?')}\u00b3"))
    return entries


def gallery_dirs():
    return [d for d in sorted(OUTPUT_DIR.iterdir(), reverse=True)
            if d.is_dir() and (d / 'meta.json').exists() and (d / 'source.png').exists()]


# ---------------------------------------------------------------- queue actions

def add_to_queue(files, queue, resolution, seed, randomize_seed, steps, guidance,
                 decimation, texture_size, render_turntable, formats):
    if not files:
        return queue, queue_to_rows(queue), "No images selected."

    warnings = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        name = safe_name(Path(path).stem)
        if not has_real_alpha(path):
            warnings.append(name)

        # Take our own copy immediately -- the Gradio-managed temp file may be deleted while
        # this item is still sitting in the queue or mid-generation.
        try:
            stable = INPUT_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{name}.png"
            Image.open(path).convert('RGBA').save(stable)
            path = str(stable)
        except Exception as e:
            print(f"[QUEUE] Could not stage a copy of {name} ({type(e).__name__}); using original path.")

        queue.append({
            'path': path,
            'name': name,
            'resolution': resolution,
            'seed': int(np.random.randint(0, MAX_SEED)) if randomize_seed else int(seed),
            'steps': int(steps),
            'guidance': float(guidance),
            'decimation': int(decimation),
            'texture_size': int(texture_size),
            'render_turntable': bool(render_turntable),
            'formats': list(formats or []),
            'status': 'pending',
        })

    msg = f"Added {len(files)} image(s). Queue length: {len(queue)}."
    if warnings:
        msg += ("\n\n\u26a0 No transparent background detected in: " + ", ".join(warnings) +
                ".\nTRELLIS needs a cut-out subject, and the automatic background remover "
                "(briaai/RMBG-2.0) is a gated model that isn't available here. "
                "These will likely fail \u2014 supply RGBA images with the background already removed.")
    return queue, queue_to_rows(queue), msg


def clear_queue(queue):
    queue = [item for item in queue if item['status'] == 'running']
    return queue, queue_to_rows(queue), "Queue cleared."


def export_extra_formats(glb_scene, glb_path, out_dir, formats):
    """Write the optional side formats. GLB is always written separately; these are extras."""
    written = []
    if 'OBJ' in formats:
        try:
            glb_scene.export(str(out_dir / 'model.obj'))
            written.append('OBJ')
        except Exception as e:
            print(f"[EXPORT] OBJ export failed ({type(e).__name__}: {e}).")
    if 'FBX' in formats:
        if not BLENDER_EXE:
            print("[EXPORT] FBX requested but Blender was not found; skipping.")
        else:
            try:
                r = subprocess.run(
                    [BLENDER_EXE, "--background", "--python", str(FBX_SCRIPT),
                     "--", str(glb_path), str(out_dir / 'model.fbx')],
                    capture_output=True, text=True, timeout=1800,
                )
                if (out_dir / 'model.fbx').exists():
                    written.append('FBX')
                else:
                    print(f"[EXPORT] FBX conversion produced no file. Blender said:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            except Exception as e:
                print(f"[EXPORT] FBX export failed ({type(e).__name__}: {e}).")
    return written


def generate_one(item, progress):
    """Run the full pipeline for a single queued image and write its output folder."""
    image = Image.open(item['path'])
    res = item['resolution']

    outputs = pipeline.run(
        image,
        seed=item['seed'],
        preprocess_image=True,
        sparse_structure_sampler_params={
            "steps": item['steps'], "guidance_strength": 7.5,
            "guidance_rescale": 0.7, "rescale_t": 5.0,
        },
        shape_slat_sampler_params={
            "steps": item['steps'], "guidance_strength": item['guidance'],
            "guidance_rescale": 0.5, "rescale_t": 3.0,
        },
        tex_slat_sampler_params={
            "steps": item['steps'], "guidance_strength": 1.0,
            "guidance_rescale": 0.0, "rescale_t": 3.0,
        },
        pipeline_type=RESOLUTION_INFO[res][2],
    )
    mesh = outputs[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / safe_name(f"{stamp}_{item['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write the cheap bookkeeping first: it costs milliseconds, and doing it last means a
    # hiccup here throws away a generation that already cost 20 minutes.
    try:
        Image.open(item['path']).convert('RGBA').save(out_dir / 'source.png')
    except Exception as e:
        print(f"[SOURCE] Could not copy the source image ({type(e).__name__}: {e}).")
    (out_dir / 'meta.json').write_text(json.dumps({
        'name': item['name'], 'resolution': res, 'seed': item['seed'],
        'steps': item['steps'], 'guidance': item['guidance'],
        'decimation': item['decimation'], 'texture_size': item['texture_size'],
        'render_turntable': bool(item.get('render_turntable')),
        'formats': item.get('formats', []),
        'source': item['path'], 'created': stamp,
    }, indent=2))

    # Then the mesh -- the expensive artifact. The turntable render is a nicety, and a failure
    # there must never discard a generation that already succeeded.
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=item['decimation'], texture_size=item['texture_size'],
        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    glb.export(str(out_dir / 'model.glb'), extension_webp=True)
    extra = export_extra_formats(glb, out_dir / 'model.glb', out_dir, item.get('formats', []))
    if extra:
        print(f"[EXPORT] Also wrote: {', '.join(extra)}")

    # The turntable is ~20 minutes of rendering on top of a ~2 minute generation, so it is
    # opt-in per item rather than an unavoidable tax on every mesh.
    if item.get('render_turntable'):
        try:
            video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            imageio.mimsave(str(out_dir / 'preview.mp4'), video, fps=15)
        except Exception as e:
            print(f"[RENDER] Turntable render failed ({type(e).__name__}: {e}); mesh was saved regardless.")

    torch.cuda.empty_cache()
    return out_dir


def run_queue(queue, progress=gr.Progress(track_tqdm=True)):
    if not queue:
        yield queue, queue_to_rows(queue), "Queue is empty.", None, None, scan_gallery()
        return

    pending = [i for i, it in enumerate(queue) if it['status'] == 'pending']
    if not pending:
        yield queue, queue_to_rows(queue), "Nothing pending.", None, None, scan_gallery()
        return

    last_glb = last_video = None
    for n, idx in enumerate(pending, 1):
        item = queue[idx]
        item['status'] = 'running'
        yield (queue, queue_to_rows(queue),
               f"[{n}/{len(pending)}] Generating '{item['name']}' at {item['resolution']}\u00b3 "
               f"({RESOLUTION_INFO[item['resolution']][1]})...",
               last_glb, last_video, scan_gallery())

        try:
            out_dir = generate_one(item, progress)
            item['status'] = 'done'
            last_glb = str(out_dir / 'model.glb')
            vid = out_dir / 'preview.mp4'
            last_video = str(vid) if vid.exists() else None
            msg = f"[{n}/{len(pending)}] Finished '{item['name']}' \u2192 {out_dir.name}"
        except Exception as e:
            item['status'] = 'failed'
            traceback.print_exc()
            torch.cuda.empty_cache()
            msg = f"[{n}/{len(pending)}] FAILED on '{item['name']}': {type(e).__name__}: {e}"

        yield queue, queue_to_rows(queue), msg, last_glb, last_video, scan_gallery()

    done = sum(1 for it in queue if it['status'] == 'done')
    failed = sum(1 for it in queue if it['status'] == 'failed')
    yield (queue, queue_to_rows(queue),
           f"Queue complete \u2014 {done} succeeded, {failed} failed.",
           last_glb, last_video, scan_gallery())


def load_from_gallery(evt: gr.SelectData):
    dirs = gallery_dirs()
    if evt.index is None or evt.index >= len(dirs):
        return None, None, "Could not load that item."
    d = dirs[evt.index]
    meta = json.loads((d / 'meta.json').read_text())
    info = (f"**{meta.get('name')}** \u2014 {meta.get('resolution')}\u00b3, seed {meta.get('seed')}, "
            f"{meta.get('steps')} steps, guidance {meta.get('guidance')}, "
            f"{meta.get('texture_size')}px textures\n\n`{d}`")
    vid = d / 'preview.mp4'
    return str(d / 'model.glb'), (str(vid) if vid.exists() else None), info


# ---------------------------------------------------------------- ui

# Cache cleanup must not outpace a generation: purge hourly, and only files over a day old.
with gr.Blocks(title="TRELLIS.2 Studio", delete_cache=(3600, 86400)) as demo:
    gr.Markdown("# TRELLIS.2 Studio\nDrop in a batch of cut-out images, queue them, and collect the meshes.")
    queue_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            files_in = gr.File(label="Images (RGBA with transparent background)",
                               file_count="multiple", file_types=["image"])

            resolution = gr.Radio(["512", "1024", "1536"], value="1024", label="Resolution",
                                  info="512 ~5min \u00b7 1024 ~20min \u00b7 1536 ~45min and VRAM-heavy")
            with gr.Row():
                seed = gr.Number(value=0, label="Seed", precision=0)
                randomize_seed = gr.Checkbox(value=True, label="Randomize")
            steps = gr.Slider(1, 50, value=12, step=1, label="Sampling steps")
            guidance = gr.Slider(1.0, 10.0, value=7.5, step=0.1, label="Shape guidance strength")
            formats = gr.CheckboxGroup(
                ["OBJ", "FBX"], value=["FBX"], label="Additional export formats",
                info="GLB is always written (the 3D viewer reads it). FBX goes through Blender.")
            render_turntable = gr.Checkbox(
                value=False, label="Render turntable video",
                info="Adds roughly 20 min per item. Off = mesh only, ready in ~2 min.")
            with gr.Accordion("Mesh output", open=False):
                decimation = gr.Slider(100_000, 1_000_000, value=500_000, step=10_000, label="Decimation target (faces)")
                texture_size = gr.Slider(1024, 4096, value=2048, step=1024, label="Texture size (px)")

            with gr.Row():
                add_btn = gr.Button("Add to queue", variant="secondary")
                clear_btn = gr.Button("Clear queue")
            run_btn = gr.Button("Run queue", variant="primary", size="lg")

        with gr.Column(scale=2):
            status = gr.Markdown("Ready.")
            queue_table = gr.Dataframe(
                headers=["#", "Name", "Res", "Seed", "Video", "Status"],
                datatype=["number", "str", "str", "number", "str", "str"],
                label="Queue", interactive=False, wrap=True,
            )
            with gr.Tabs():
                with gr.Tab("Result"):
                    model_out = gr.Model3D(label="Mesh", height=480, clear_color=(0.25, 0.25, 0.25, 1.0))
                    video_out = gr.Video(label="Turntable", autoplay=True, height=320)
                with gr.Tab("Gallery"):
                    gr.Markdown("Past generations \u2014 click one to load it.")
                    gallery = gr.Gallery(value=scan_gallery(), label="Generations",
                                         columns=4, height=380, allow_preview=False)
                    gallery_info = gr.Markdown()

    add_btn.click(add_to_queue,
                  [files_in, queue_state, resolution, seed, randomize_seed, steps, guidance,
                   decimation, texture_size, render_turntable, formats],
                  [queue_state, queue_table, status])
    clear_btn.click(clear_queue, [queue_state], [queue_state, queue_table, status])
    run_btn.click(run_queue, [queue_state],
                  [queue_state, queue_table, status, model_out, video_out, gallery])
    gallery.select(load_from_gallery, None, [model_out, video_out, gallery_info])

if __name__ == "__main__":
    # Gradio 6 moved `theme` off Blocks and onto launch().
    demo.queue().launch(server_name="127.0.0.1", server_port=7861,
                        inbrowser=False, theme=gr.themes.Soft())
