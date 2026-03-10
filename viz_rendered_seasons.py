import bpy, os, re
import math
import numpy as np
import bmesh
from mathutils import Vector, Matrix

# ============================================================
# TREE LOGIC PARAMETERS
# ============================================================
SPECIES = "Spruce"

# Season/Appearance Sliders (0.0 to 1.0)
LEAF_DENSITY = 1.0            # 0.0 = Winter (no leaves), 1.0 = Summer (full leaves)
LEAF_COLOR_ORANGENESS = 0.0   # 0.0 = Green (summer), 1.0 = Orange (autumn)

BASE_LEAF_DIR = r"C:\Users\angie\Documents\the_grove_22_indie\the_grove_22\templates\Transparency_Twigs_modify\TwigsLibrary"
BARK_DIR = r"C:\Users\angie\Documents\the_grove_22_indie\the_grove_22\templates\BarkTextures"

SMOOTH_FACTOR = 0.1 
GRAVITY_STRENGTH = 0.03 
STEPS = 5  

BASE_RADIUS    = 0.15
DEPTH_DECAY    = 0.72 
SEGMENT_TAPER  = 0.94 # Keep closer to 1.0 for smoother long branches
MIN_RADIUS     = 0.006 
BRANCH_SIDES   = 16

LEAF_SIZE = 0.12
LEAF_MIN_DEPTH = 2 

LEAVES_PER_TERMINAL = 1

# Maximum depth for branches to keep (prune deeper branches before adding leaves)
DEPTH_MAX = 2

INPUT_FOLDER = r"./results/final" 
# INPUT_FOLDER = r"./results/growth/dense" 
SIMPLIFIED_FOLDER = r"./results/simplified"
import sys
sys.path.append(r"C:\Users\angie\Documents\lsys_gen")
from pipeline import NUM_BINS_THETA, NUM_BINS_PHI, NUM_BINS_F

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

MAX_LENGTH = max_len_found if max_len_found > 0 else 3.0
# ============================================================
# REFINED GEOMETRY ENGINE
# ============================================================

def create_smooth_branch(bm, uv_layer, pts, radii):
    import math
    """Creates a single, continuous tube for a list of points."""
    if len(pts) < 2: return
    
    rings = []
    prev_side = None
    
    # Calculate total length for UV V-coordinates
    lengths = [0.0]
    for i in range(1, len(pts)):
        lengths.append(lengths[-1] + (Vector(pts[i]) - Vector(pts[i-1])).length)
        
    for i in range(len(pts)):
        pt, rad = Vector(pts[i]), radii[i]
        # Calculate consistent tangent
        if i < len(pts) - 1:
            direction = (Vector(pts[i+1]) - pt).normalized()
        else:
            direction = (pt - Vector(pts[i-1])).normalized()
        
        # Parallel Transport Frame to eliminate 'twisting' at seams
        if prev_side is None:
            ref = Vector((0,0,1)) if abs(direction.z) < 0.99 else Vector((1,0,0))
            side = direction.cross(ref).normalized()
        else:
            side = (prev_side - direction * prev_side.dot(direction)).normalized()
        
        up = side.cross(direction).normalized()
        prev_side = side
        
        ring = [bm.verts.new(pt + (side * math.cos(2*math.pi*j/BRANCH_SIDES) + 
                                   up * math.sin(2*math.pi*j/BRANCH_SIDES)) * rad) 
                for j in range(BRANCH_SIDES)]
        rings.append(ring)

    # Bridge rings into a single mesh shell
    for i in range(len(rings) - 1):
        r1, r2 = rings[i], rings[i+1]
        v1_uv = lengths[i] * 4.0  # Tile scaling
        v2_uv = lengths[i+1] * 4.0
        for j in range(BRANCH_SIDES):
            nj = (j + 1) % BRANCH_SIDES
            f = bm.faces.new([r1[j], r1[nj], r2[nj], r2[j]])
            
            u0 = j / BRANCH_SIDES
            u1 = 1.0 if j == BRANCH_SIDES - 1 else (j + 1) / BRANCH_SIDES
            
            f.loops[0][uv_layer].uv = (u0, v1_uv)
            f.loops[1][uv_layer].uv = (u1, v1_uv)
            f.loops[2][uv_layer].uv = (u1, v2_uv)
            f.loops[3][uv_layer].uv = (u0, v2_uv)
            
    # Cap terminal ends
    for r_idx in [0, -1]:
        ring = rings[r_idx]
        center = sum((v.co for v in ring), Vector()) / len(ring)
        v_cap = bm.verts.new(center)
        try:
            for j in range(BRANCH_SIDES):
                nj = (j + 1) % BRANCH_SIDES
                if r_idx == 0: 
                    f = bm.faces.new([ring[nj], ring[j], v_cap])
                else: 
                    f = bm.faces.new([ring[j], ring[nj], v_cap])
                
                # Default cap UVs mapping
                for loop in f.loops: loop[uv_layer].uv = (0.5, 0.5)
        except ValueError:
            pass

def add_logical_leaves(bm_leaf, uv_layer, start_pos, end_pos, direction, radius, leaf_size):
    """
    Distributes leaves along the branch segment and ensures 
    the base of the leaf quad is pinned to the branch surface.
    """
    d = Vector(direction).normalized()
    up = Vector((0, 0, 1))
    side = d.cross(up).normalized() if abs(d.z) < 0.99 else d.cross(Vector((1,0,0))).normalized()
    perp = side.cross(d).normalized()

    # Distribute 3-5 leaves per segment step to avoid "clumping" at nodes
    num_leaves_this_step = 3 
    
    for i in range(num_leaves_this_step):
        # Apply the seasonal leaf density factor randomly
        if np.random.uniform(0, 1) > LEAF_DENSITY:
            continue
            
        # 1. SLIDE: Randomize position along the segment length
        t = np.random.uniform(0, 1)
        interp_pos = start_pos.lerp(end_pos, t)
        
        # 2. ROTATE: Randomize angle around the branch
        angle = np.random.uniform(0, 2 * math.pi)
        outward_dir = (side * math.cos(angle) + perp * math.sin(angle)).normalized()
        
        # 3. POSITION: Pin to the surface (the "bark")
        surface_origin = interp_pos + (outward_dir * radius)
        
        # 4. VECTORS: Define Growth (Length) and Width (Axis)
        # We use the same scale for both to prevent "stretched" leaves
        growth_dir = (d * 0.4 + outward_dir * 0.6).normalized()
        width_vec = growth_dir.cross(outward_dir).normalized()
        
        # Consistent scaling on both axes
        full_length = growth_dir * (leaf_size * 2.5)
        half_width = width_vec * (leaf_size * 1.25) # Width is roughly half the length
        
        # 5. QUAD CONSTRUCTION: v1 and v2 are the "stem" edge
        v1 = bm_leaf.verts.new(surface_origin - half_width)
        v2 = bm_leaf.verts.new(surface_origin + half_width)
        v3 = bm_leaf.verts.new(surface_origin + half_width + full_length)
        v4 = bm_leaf.verts.new(surface_origin - half_width + full_length)
        
        try:
            f = bm_leaf.faces.new([v1, v2, v3, v4])
            f.loops[0][uv_layer].uv = (0.0, 0.0)
            f.loops[1][uv_layer].uv = (1.0, 0.0)
            f.loops[2][uv_layer].uv = (1.0, 1.0)
            f.loops[3][uv_layer].uv = (0.0, 1.0)
        except ValueError:
            pass

# ============================================================
# MAIN ASSEMBLY
# ============================================================

def build_organic_tree(filepath):
    import math
    with open(filepath, "r") as f:
        program = f.read().strip()
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", program)

    # Prune all branches deeper than DEPTH_MAX
    def prune_to_max_depth(tokens, max_depth):
        pruned = []
        depth = 0
        for t in tokens:
            if t == '[':
                if depth < max_depth:
                    pruned.append(t)
                depth += 1
            elif t == ']':
                depth -= 1
                if depth < max_depth:
                    pruned.append(t)
            else:
                if depth <= max_depth:
                    pruned.append(t)
        return pruned

    tokens = prune_to_max_depth(tokens, DEPTH_MAX)
    
    # Calculate a GLOBAL average leaf size based on terminal depths
    tmp_depth = 0
    branch_lengths = []
    current_branch_length = 0.0
    for t in tokens:
        if t == '[':
            if tmp_depth >= LEAF_MIN_DEPTH and current_branch_length > 0:
                branch_lengths.append(current_branch_length)
            tmp_depth += 1
            current_branch_length = 0.0
        elif t == ']':
            if tmp_depth >= LEAF_MIN_DEPTH and current_branch_length > 0:
                branch_lengths.append(current_branch_length)
            tmp_depth -= 1
            current_branch_length = 0.0
        elif t.startswith("B") and tmp_depth >= LEAF_MIN_DEPTH:
            m = re.match(r"B\d+_\d+F(\d+)", t)
            if m:
                f_bin = int(m.group(1))
                current_branch_length += (f_bin + 0.5) / NUM_BINS_F * MAX_LENGTH

    if current_branch_length > 0 and tmp_depth >= LEAF_MIN_DEPTH:
        branch_lengths.append(current_branch_length)

    avg_last_depth_length = sum(branch_lengths) / len(branch_lengths) if branch_lengths else 0.5
    
    # A single consistent size for all leaves everywhere (1/2 of average size)
    fixed_global_leaf_size = avg_last_depth_length * 0.5
    
    bm_tree = bmesh.new()
    bm_leaf = bmesh.new()
    
    uv_tree = bm_tree.loops.layers.uv.new("UVMap")
    uv_leaf = bm_leaf.loops.layers.uv.new("UVMap")
    
    # State: pos, angle_t, angle_p, radius, depth
    pos, curr_t, curr_p = np.array([0.,0.,0.]), 0.0, 0.0
    rad, depth = BASE_RADIUS, 0
    stack = []
    max_depth = 0
    depth_track = []
    # Points collected for the CURRENT branch segment
    pts, radii = [pos.copy()], [rad]

    # First pass: find max depth
    tmp_depth = 0
    for token in tokens:
        if token == '[':
            tmp_depth += 1
            if tmp_depth > max_depth:
                max_depth = tmp_depth
        elif token == ']':
            tmp_depth -= 1
    print(f"Detected last (maximum) depth in tree: {max_depth}")

    # Second pass: build tree and add leaves only at max depth
    pos, curr_t, curr_p = np.array([0.,0.,0.]), 0.0, 0.0
    rad, depth = BASE_RADIUS, 0
    stack = []
    pts, radii = [pos.copy()], [rad]
    curr_depth = 0
    for token in tokens:
        if token.startswith("B"):
            m = re.match(r"B(\d+)_(\d+)F(\d+)", token)
            t_bin, p_bin, f_bin = map(int, m.groups())
            target_t = (t_bin+0.5)/NUM_BINS_THETA*180
            target_p = (p_bin+0.5)/NUM_BINS_PHI*360
            length   = (f_bin+0.5)/NUM_BINS_F*MAX_LENGTH

            for s in range(STEPS):
                curr_t += (target_t - curr_t) * (1.0 / STEPS)
                curr_p += (target_p - curr_p) * (1.0 / STEPS)
                curr_t += (BASE_RADIUS / max(rad, 0.01)) * GRAVITY_STRENGTH
                curr_t += np.random.uniform(-SMOOTH_FACTOR, SMOOTH_FACTOR)
                direction = np.array([math.sin(math.radians(curr_t))*math.cos(math.radians(curr_p)),
                                     math.sin(math.radians(curr_t))*math.sin(math.radians(curr_p)),
                                     math.cos(math.radians(curr_t))])
                old_pos = pos.copy()
                pos += direction * (length / STEPS)
                rad = max(rad * (SEGMENT_TAPER ** (1/STEPS)), MIN_RADIUS)
                pts.append(pos.copy())
                radii.append(rad)
                # Only add leaves at max depth
                if curr_depth == max_depth:
                    add_logical_leaves(bm_leaf, uv_leaf, Vector(old_pos), Vector(pos), direction, rad, fixed_global_leaf_size)

        elif token == "[":
            create_smooth_branch(bm_tree, uv_tree, pts, radii)
            stack.append((pos.copy(), curr_t, curr_p, rad, curr_depth))
            curr_depth += 1
            rad *= DEPTH_DECAY
            pts, radii = [pos.copy()], [rad]

        elif token == "]":
            create_smooth_branch(bm_tree, uv_tree, pts, radii)
            pos, curr_t, curr_p, rad, curr_depth = stack.pop()
            pts, radii = [pos.copy()], [rad]

    create_smooth_branch(bm_tree, uv_tree, pts, radii)

    # ----------------------------------------------------
    # MATERIAL SETUP
    # ----------------------------------------------------
    species_folder = SPECIES.capitalize() # e.g. "Spruce"
    leaf_img_path = os.path.join(BASE_LEAF_DIR, species_folder, f"{species_folder}.png")
    bark_img_path = os.path.join(BARK_DIR, f"{species_folder}.jpg")

    # Bark Material
    mat_tree = bpy.data.materials.new(name="BarkMat")
    mat_tree.use_nodes = True
    bsdf = mat_tree.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs['Roughness'].default_value = 0.0
    if os.path.exists(bark_img_path):
        tex_image = mat_tree.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(bark_img_path)
        # Connect UV mapping for Bark mapping
        uv_map_node = mat_tree.node_tree.nodes.new('ShaderNodeUVMap')
        uv_map_node.uv_map = "UVMap"
        mat_tree.node_tree.links.new(uv_map_node.outputs['UV'], tex_image.inputs['Vector'])
        mat_tree.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Leaf Material
    mat_leaf = bpy.data.materials.new(name="LeafMat")
    mat_leaf.use_nodes = True
    mat_leaf.blend_method = 'CLIP'
    mat_leaf.use_backface_culling = False
    bsdf_leaf = mat_leaf.node_tree.nodes.get("Principled BSDF")
    if bsdf_leaf is not None:
        bsdf_leaf.inputs['Roughness'].default_value = 0.0
    if os.path.exists(leaf_img_path):
        tex_image_leaf = mat_leaf.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image_leaf.image = bpy.data.images.load(leaf_img_path)
        # Connect UV mapping for Leaf mapping
        uv_map_node_leaf = mat_leaf.node_tree.nodes.new('ShaderNodeUVMap')
        uv_map_node_leaf.uv_map = "UVMap"
        
        # Add a seasonal color shift node
        hsv_node = mat_leaf.node_tree.nodes.new('ShaderNodeHueSaturation')
        hsv_node.inputs['Hue'].default_value = 0.5 - (0.12 * LEAF_COLOR_ORANGENESS)
        hsv_node.inputs['Saturation'].default_value = 1.0 + (0.3 * LEAF_COLOR_ORANGENESS)
        hsv_node.inputs['Value'].default_value = 1.0 - (0.15 * LEAF_COLOR_ORANGENESS)
        
        mat_leaf.node_tree.links.new(uv_map_node_leaf.outputs['UV'], tex_image_leaf.inputs['Vector'])
        mat_leaf.node_tree.links.new(hsv_node.inputs['Color'], tex_image_leaf.outputs['Color'])
        mat_leaf.node_tree.links.new(bsdf_leaf.inputs['Base Color'], hsv_node.outputs['Color'])
        mat_leaf.node_tree.links.new(bsdf_leaf.inputs['Alpha'], tex_image_leaf.outputs['Alpha'])

    # Bake to Scene
    tree_obj = None
    for bm, name in [(bm_tree, "Tree"), (bm_leaf, "Leaves")]:
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(obj)
        if name == "Tree":
            obj.data.materials.append(mat_tree)
            for p in me.polygons: p.use_smooth = True
            tree_obj = obj
        elif name == "Leaves":
            obj.data.materials.append(mat_leaf)

    # ----------------------------------------------------
    # SUN LIGHT SETUP
    # ----------------------------------------------------
    # Remove existing sun lights
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add a new sun light
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0, -10, 10)
    import math
    light_obj.rotation_euler = (0, -math.radians(90), 0)

    # ----------------------------------------------------
    # CAMERA SETUP
    # ----------------------------------------------------
    # Remove existing cameras
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add a new camera
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    # Compute bounding box of the tree
    if tree_obj is not None:
        bpy.context.view_layer.update()
        bbox = [tree_obj.matrix_world @ Vector(corner) for corner in tree_obj.bound_box]
        min_x = min(v.x for v in bbox)
        max_x = max(v.x for v in bbox)
        min_y = min(v.y for v in bbox)
        max_y = max(v.y for v in bbox)
        min_z = min(v.z for v in bbox)
        max_z = max(v.z for v in bbox)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        size = max(max_x - min_x, max_y - min_y, max_z - min_z)
        # Place camera in front of the tree, looking at its center
        cam_dist = size * 3.5 + 1.0
        cam_obj.location = (center_x - cam_dist, center_y, center_z)
        # Make camera look at the tree center
        direction = Vector((center_x, center_y, center_z)) - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()
        # Set camera as active
        bpy.context.scene.camera = cam_obj

# ============================================================
# EXECUTION
# ============================================================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
if files:
    build_organic_tree(os.path.join(INPUT_FOLDER, files[0]))