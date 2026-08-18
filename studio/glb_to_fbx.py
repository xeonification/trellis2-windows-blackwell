"""
Convert a .glb to .fbx using Blender in background mode.

trimesh cannot write FBX (it is a proprietary Autodesk format), so we shell out to Blender.
Invoked as:  blender --background --python glb_to_fbx.py -- <input.glb> <output.fbx>
"""
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
bpy.ops.export_scene.fbx(
    filepath=dst,
    path_mode='COPY',        # bring textures along
    embed_textures=True,     # ...packed into the .fbx itself
    apply_scale_options='FBX_SCALE_ALL',
    use_mesh_modifiers=True,
)
print(f"FBX_EXPORT_OK {dst}")
