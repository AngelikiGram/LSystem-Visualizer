import bpy
import os
import math
import numpy as np
import re

import sys

# Ensure local imports (e.g. pipeline.py) resolve even when Blender runs from a different cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from pipeline import NUM_BINS_THETA, NUM_BINS_PHI, NUM_BINS_F
import pipeline
NUM_BINS_THETA, NUM_BINS_PHI, NUM_BINS_F = 12, 12, 10
# ============================================================
# CONFIG
# ============================================================

INPUT_FOLDER = r"C:\Users\angie\Documents\lsys_gen\results\final"
SIMPLIFIED_FOLDER = r"C:\Users\angie\Documents\lsys_gen\results\simplified"

INPUT_FOLDER = r"E:\TREES_DATASET_P3_LSTRING\LSTRINGS_RESULTS\LSTRINGS_FINAL_SMALL"

# INPUT_FOLDER = r"C:\Users\angie\Documents\lsys_gen\lstrings_generator\results\final"
# SIMPLIFIED_FOLDER = r"C:\Users\angie\Documents\lsys_gen\lstrings_generator\results\simplified"


# Dynamically determine from simplified to match encoding
max_len_found = 0.0
if os.path.exists(SIMPLIFIED_FOLDER) and os.path.exists(INPUT_FOLDER):
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(SIMPLIFIED_FOLDER, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r") as f:
            data = f.read()
        for m in re.finditer(r"F\(([+-]?\d+(?:\.\d+)?)\)", data):
            max_len_found = max(max_len_found, float(m.group(1)))

GLOBAL_LENGTH_MAX = max_len_found if max_len_found > 0 else pipeline.GLOBAL_LENGTH_MAX
print(f"Viz set GLOBAL_LENGTH_MAX to: {GLOBAL_LENGTH_MAX}")

MAX_TREES       = 3000 # 20
LENGTH_VARIANCE = 0.05
ANGLE_VARIANCE  = 2.0   # degrees jitter per bin

# ============================================================
# CLEAR SCENE
# ============================================================

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ============================================================
# BIN DECODING  (absolute world-space angles, matches pipeline.py)
# ============================================================

def decode_theta(bin_id):
    """Inclination: bin → degrees in [0, 180] with small jitter."""
    theta = (bin_id + 0.5) / NUM_BINS_THETA * 180.0
    theta += np.random.uniform(-ANGLE_VARIANCE, ANGLE_VARIANCE)
    return max(0.0, min(180.0, theta))

def decode_phi(bin_id):
    """Azimuth: bin → degrees in [0, 360)."""
    return (bin_id + 0.5) / NUM_BINS_PHI * 360.0

def decode_length(bin_id):
    length = (bin_id + 0.5) / NUM_BINS_F * GLOBAL_LENGTH_MAX
    length *= np.random.uniform(1 - LENGTH_VARIANCE, 1 + LENGTH_VARIANCE)
    return max(0.0, length)

def direction_from_angles(theta_deg, phi_deg):
    """Absolute world-space direction from spherical coordinates."""
    theta = math.radians(theta_deg)
    phi   = math.radians(phi_deg)
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ])

# ============================================================
# TURTLE + MESH BUILDING
# ============================================================

txt_files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt"))

for tree_idx, filename in enumerate(txt_files):
    if tree_idx >= MAX_TREES:
        break

    if '1445' not in filename: 
        continue
    print(f"Processing {filename}...")

    with open(os.path.join(INPUT_FOLDER, filename), "r") as f:
        program = f.read().strip()

    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", program)

    pos         = np.array([0.0, 0.0, 0.0])
    current_dir = np.array([0.0, 0.0, 1.0])
    stack       = []   # (pos, current_dir)
    verts, edges = [], []

    for token in tokens:
        if token.startswith("B"):
            m = re.match(r"B(\d+)_(\d+)F(\d+)", token)
            if not m: continue
            theta_bin  = int(m.group(1))
            phi_bin    = int(m.group(2))
            length_bin = int(m.group(3))

            theta  = decode_theta(theta_bin)
            phi    = decode_phi(phi_bin)
            length = decode_length(length_bin)

            current_dir = direction_from_angles(theta, phi)
            new_pos     = pos + current_dir * length

            i0 = len(verts); verts.append(tuple(pos))
            i1 = len(verts); verts.append(tuple(new_pos))
            edges.append((i0, i1))
            pos = new_pos

        elif token == "[":
            stack.append((pos.copy(), current_dir.copy()))

        elif token == "]":
            if stack:
                pos, current_dir = stack.pop()

    if not verts:
        print(f"  {filename}: no vertices, skipping")
        continue

    # Create mesh
    mesh = bpy.data.meshes.new(filename)
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    obj = bpy.data.objects.new(filename, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location.y = tree_idx * 1.5

    # Convert to Grease Pencil with thickness
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target='GREASEPENCIL')
    bpy.ops.object.modifier_add(type='GREASE_PENCIL_THICKNESS')
    mod = obj.modifiers[-1]
    mod.use_uniform_thickness = True
    mod.thickness = 0.005

    print(f"  {filename}: {len(verts)//2} segments")

# ============================================================
# CAMERA + RENDER SETTINGS
# ============================================================

bpy.context.scene.render.film_transparent = True

cam_data = bpy.data.cameras.new("Camera")
cam_obj  = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
cam_obj.location       = (0, -3, 0)
cam_obj.rotation_euler = (math.radians(90), 0, 0)
bpy.context.scene.camera = cam_obj

bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1920

print("✔ Scene built.")
