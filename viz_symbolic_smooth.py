import bpy
import os
import math
import numpy as np
import re

# ============================================================
# THE "SMOOTHNESS" SLIDER (0.0 to 1.0)
# ============================================================
# 0.0 = Perfectly straight
# 0.5 = Natural wandering branches
# 1.0 = Dramatic, swirling curves
SMOOTH_FACTOR = 0.5  

# ============================================================
# CONFIG
# ============================================================
INPUT_FOLDER = r"./results/final" # growth/dense"  
STEPS = 40 # Increased to give enough geometry for deep curves

NUM_BINS_THETA, NUM_BINS_PHI, NUM_BINS_F = 12, 12, 15
MAX_LENGTH = 1.0

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def direction_from_angles(theta_deg, phi_deg):
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.sin(t)*math.cos(p), math.sin(t)*math.sin(p), math.cos(t)])

clear_scene()
txt_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
i = 0

intensity = np.clip(SMOOTH_FACTOR, 0.0, 1.0)

for filename in txt_files:
    if i >= 10: break
    
    with open(os.path.join(INPUT_FOLDER, filename), "r") as f:
        program = f.read().strip()

    pos, stack = np.array([0.0, 0.0, 0.0]), []
    verts, edges = [tuple(pos)], []
    current_vert_idx = 0

    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", program)

    for token in tokens:
        if token.startswith("B"):
            m = re.match(r"B(\d+)_(\d+)F(\d+)", token)
            if m:
                t_bin, p_bin, f_bin = map(int, m.groups())
                target_theta = (t_bin + 0.5) / NUM_BINS_THETA * 180.0
                target_phi = (p_bin + 0.5) / NUM_BINS_PHI * 360.0
                length = (f_bin + 0.5) / NUM_BINS_F * MAX_LENGTH
                
                segment_start = pos.copy()
                
                # --- CONTINUOUS CURVATURE LOGIC ---
                # We initialize the "running" angles
                curr_theta, curr_phi = target_theta, target_phi
                
                for s in range(STEPS):
                    # Instead of absolute random, we add a "Steering Drift"
                    # This causes the angle to accumulate error, creating a CURVE
                    # Higher intensity = tighter, more frequent bends
                    steering_force = intensity * 4.0 
                    curr_theta += np.random.uniform(-steering_force, steering_force)
                    curr_phi   += np.random.uniform(-steering_force, steering_force)
                    
                    m_dir = direction_from_angles(curr_theta, curr_phi)
                    step_pos = segment_start + m_dir * (length / STEPS)
                    
                    verts.append(tuple(step_pos))
                    edges.append((current_vert_idx, len(verts)-1))
                    
                    segment_start = step_pos
                    current_vert_idx = len(verts) - 1
                
                pos = segment_start

        elif token == "[":
            stack.append((pos.copy(), current_vert_idx))
        elif token == "]":
            pos, current_vert_idx = stack.pop()

    # --- Mesh to Grease Pencil ---
    mesh = bpy.data.meshes.new(filename)
    mesh.from_pydata(verts, edges, [])
    obj = bpy.data.objects.new(filename, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location.x = i * 2.5

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target='GREASEPENCIL')
    gp_obj = bpy.context.active_object

    # 1. SUBDIVIDE: High detail for smooth arcs
    sub = gp_obj.modifiers.new(name="Sub", type='GREASE_PENCIL_SUBDIV')
    sub.level = 2 if intensity > 0.1 else 0

    # 2. SMOOTH: The "Melt" effect
    smooth = gp_obj.modifiers.new(name="Smooth", type='GREASE_PENCIL_SMOOTH')
    smooth.factor = 1.0
    # Higher intensity = more iterations to pull the "wandering" into a smooth arc
    smooth.step = int(intensity * 80) 

    # 3. THICKNESS
    thick = gp_obj.modifiers.new(name="Thick", type='GREASE_PENCIL_THICKNESS')
    thick.use_uniform_thickness = True
    thick.thickness = 0.012
    
    i += 1