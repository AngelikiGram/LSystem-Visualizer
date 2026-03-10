bl_info = {
    "name": "L-System Visualizer & Seasons",
    "author": "L-System Addon Generator",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > L-System",
    "description": "Renders an L-System tree from a txt file with seasonal leaf controls.",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

import bpy
import bmesh
import os
import re
import math
import random
import numpy as np
from mathutils import Vector

# Fallback values if pipeline.py can't be imported
NUM_BINS_THETA = 12
NUM_BINS_PHI = 10
NUM_BINS_F = 10

def create_smooth_branch(bm, uv_layer, pts, radii, branch_sides):
    if len(pts) < 2: return
    
    rings = []
    prev_side = None
    lengths = [0.0]
    
    for i in range(1, len(pts)):
        lengths.append(lengths[-1] + (Vector(pts[i]) - Vector(pts[i-1])).length)
        
    for i in range(len(pts)):
        pt, rad = Vector(pts[i]), radii[i]
        if i < len(pts) - 1:
            direction = (Vector(pts[i+1]) - pt).normalized()
        else:
            direction = (pt - Vector(pts[i-1])).normalized()
        
        if prev_side is None:
            ref = Vector((0,0,1)) if abs(direction.z) < 0.99 else Vector((1,0,0))
            side = direction.cross(ref).normalized()
        else:
            side = (prev_side - direction * prev_side.dot(direction)).normalized()
        
        up = side.cross(direction).normalized()
        prev_side = side
        
        ring = [bm.verts.new(pt + (side * math.cos(2*math.pi*j/branch_sides) + 
                                   up * math.sin(2*math.pi*j/branch_sides)) * rad) 
                for j in range(branch_sides)]
        rings.append(ring)

    for i in range(len(rings) - 1):
        r1, r2 = rings[i], rings[i+1]
        v1_uv = lengths[i] * 4.0
        v2_uv = lengths[i+1] * 4.0
        for j in range(branch_sides):
            nj = (j + 1) % branch_sides
            f = bm.faces.new([r1[j], r1[nj], r2[nj], r2[j]])
            u0 = j / branch_sides
            u1 = 1.0 if j == branch_sides - 1 else (j + 1) / branch_sides
            f.loops[0][uv_layer].uv = (u0, v1_uv)
            f.loops[1][uv_layer].uv = (u1, v1_uv)
            f.loops[2][uv_layer].uv = (u1, v2_uv)
            f.loops[3][uv_layer].uv = (u0, v2_uv)
            
    for r_idx in [0, -1]:
        ring = rings[r_idx]
        center = sum((v.co for v in ring), Vector()) / len(ring)
        v_cap = bm.verts.new(center)
        try:
            for j in range(branch_sides):
                nj = (j + 1) % branch_sides
                if r_idx == 0: 
                    f = bm.faces.new([ring[nj], ring[j], v_cap])
                else: 
                    f = bm.faces.new([ring[j], ring[nj], v_cap])
                for loop in f.loops: loop[uv_layer].uv = (0.5, 0.5)
        except ValueError:
            pass

def add_logical_leaves(bm_leaf, uv_layer, start_pos, end_pos, direction, radius, leaf_size, density, leaf_size_mult, leaf_angle):
    d = Vector(direction).normalized()
    up = Vector((0, 0, 1))
    side = d.cross(up).normalized() if abs(d.z) < 0.99 else d.cross(Vector((1,0,0))).normalized()
    perp = side.cross(d).normalized()
    num_leaves_this_step = 3 
    
    for i in range(num_leaves_this_step):
        if np.random.uniform(0, 1) > density:
            continue
            
        t = np.random.uniform(0, 1)
        interp_pos = start_pos.lerp(end_pos, t)
        angle = np.random.uniform(0, 2 * math.pi)
        outward_dir = (side * math.cos(angle) + perp * math.sin(angle)).normalized()
        surface_origin = interp_pos + (outward_dir * radius)
        
        growth_dir = (d * (1.0 - leaf_angle) + outward_dir * leaf_angle).normalized()
        width_vec = growth_dir.cross(outward_dir).normalized()
        full_length = growth_dir * (leaf_size * leaf_size_mult * 2.5)
        half_width = width_vec * (leaf_size * leaf_size_mult * 1.25)
        
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


def _expand_tree_to_level(raw_string, target_level):
    """Prune (target_level=1) or grow terminal branches to reach target_level."""
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", raw_string)

    if target_level <= 1:
        # Remove all branches at depth 2 and deeper
        output, depth, i = [], 0, 0
        while i < len(tokens):
            t = tokens[i]
            if t == '[':
                depth += 1
                if depth == 2:
                    balance = 1
                    i += 1
                    while i < len(tokens) and balance > 0:
                        if tokens[i] == '[': balance += 1
                        elif tokens[i] == ']': balance -= 1
                        i += 1
                    depth -= 1
                    continue
                else:
                    output.append('[')
            elif t == ']':
                if depth > 0:
                    depth -= 1
                output.append(']')
            else:
                if depth < 2:
                    output.append(t)
            i += 1
        return "".join(output)

    def _forced_growth(d_remaining):
        if d_remaining <= 0:
            return []
        result = []
        for _ in range(2):
            result.append('[')
            f_val = max(0, 2 - (target_level - d_remaining))
            result.append(f"B{random.randint(0,4)}_{random.randint(0,4)}F{f_val}")
            result.extend(_forced_growth(d_remaining - 1))
            result.append(']')
        return result

    def _process(token_list, depth):
        if depth > target_level:
            return token_list
        output, i = [], 0
        while i < len(token_list):
            t = token_list[i]
            if t == '[':
                block, balance, j = [], 1, i + 1
                while j < len(token_list) and balance > 0:
                    if token_list[j] == '[': balance += 1
                    elif token_list[j] == ']': balance -= 1
                    if balance > 0: block.append(token_list[j])
                    j += 1
                output.append('[')
                output.extend(_process(block, depth + 1))
                output.append(']')
                i = j - 1
            elif t.startswith('B'):
                output.append(t)
                next_is_bracket = (i + 1 < len(token_list) and token_list[i + 1] == '[')
                if not next_is_bracket and depth < target_level:
                    output.extend(_forced_growth(target_level - depth))
            i += 1
        return output

    return "".join(_process(tokens, 0))


def _enrich_at_depth(lstring, enrich_depth, scale=5):
    """Add 'scale' new sub-branches alongside existing ones at (enrich_depth + 1).
    New branches sample theta/F values from existing children to match their length."""
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", lstring)
    enriched, depth, i = [], 0, 0
    while i < len(tokens):
        t = tokens[i]
        if t == '[':
            depth += 1
            enriched.append(t)
        elif t == ']':
            enriched.append(t)
            depth -= 1
        elif t.startswith('B'):
            enriched.append(t)
            if depth == enrich_depth:
                m = re.match(r"B(\d+)_(\d+)F(\d+)", t)
                if m:
                    t_bin = int(m.group(1))
                    # Collect existing child blocks
                    child_blocks, j = [], i + 1
                    while j < len(tokens) and tokens[j] == '[':
                        block, balance, k = [tokens[j]], 1, j + 1
                        while k < len(tokens) and balance > 0:
                            block.append(tokens[k])
                            if tokens[k] == '[': balance += 1
                            elif tokens[k] == ']': balance -= 1
                            k += 1
                        child_blocks.append(block)
                        j = k
                    # Extract F and theta values from existing children for length-matching
                    child_f = []
                    child_t = []
                    for block in child_blocks:
                        for tok in block:
                            cm = re.match(r"B(\d+)_\d+F(\d+)", tok)
                            if cm:
                                child_t.append(int(cm.group(1)))
                                child_f.append(int(cm.group(2)))
                                break
                    ref_f = child_f if child_f else [int(m.group(3))]
                    ref_t = child_t if child_t else [t_bin]
                    new_blocks = []
                    for _ in range(scale):
                        t_r = min(max(random.choice(ref_t) + random.choice([-1, 0, 0]), 0), 4)
                        p_r = random.randint(0, 4)
                        f_r = max(random.choice(ref_f) + random.choice([-1, 0, 0]), 0)
                        new_blocks.append([f"[B{t_r}_{p_r}F{f_r}]"])
                    all_blocks = child_blocks + new_blocks
                    random.shuffle(all_blocks)
                    for block in all_blocks:
                        enriched.extend(block)
                    i = j - 1
        i += 1
    return "".join(enriched)


_generating = False


def on_property_update(self, context):
    if getattr(self, "auto_update", False):
        try:
            generate_tree(context)
        except Exception as e:
            print(f"L-System: auto-update error: {e}")
            import traceback
            traceback.print_exc()


def generate_tree(context):
    global _generating
    if _generating:
        return
    _generating = True
    try:
        _generate_tree_inner(context)
    finally:
        _generating = False


def _generate_tree_inner(context):
    props = context.scene.lsystem_props
    
    filepath = bpy.path.abspath(props.filepath)
    if not os.path.exists(filepath):
        return
        
    # Use a fixed random seed so procedural randomness (leaves, smooth factor) creates the exact same visual structure each rebuild
    np.random.seed(props.random_seed)

    # Clean up old generated meshes/objects cleanly to allow live updates without destroying the scene
    for obj_name in ["Tree", "Leaves"]:
        obj = bpy.data.objects.get(obj_name)
        if obj:
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data and getattr(mesh_data, "users", 0) == 0:
                bpy.data.meshes.remove(mesh_data, do_unlink=True)
                
    # Clean up old materials
    for mat_name in ["BarkMat", "LeafMat"]:
        old_mat = bpy.data.materials.get(mat_name)
        if old_mat:
            bpy.data.materials.remove(old_mat, do_unlink=True)

    # Try to import num bins from pipeline if possible
    num_theta, num_phi, num_f = NUM_BINS_THETA, NUM_BINS_PHI, NUM_BINS_F
    try:
        import sys
        script_dir = r"C:\Users\angie\Documents\visualizer_Lsystem"
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        from pipeline import NUM_BINS_THETA as th, NUM_BINS_PHI as ph, NUM_BINS_F as fp
        num_theta, num_phi, num_f = th, ph, fp
    except ImportError:
        pass

    with open(filepath, "r") as f:
        program = f.read().strip()

    # Apply growth / enrichment from growth.py logic
    random.seed(props.random_seed)
    if props.growth_target_level > 0:
        program = _expand_tree_to_level(program, props.growth_target_level)
    if props.growth_enrich:
        # enrich_depth is the PARENT depth: new sub-branches appear at (enrich_depth+1) = target_level
        enrich_depth = (props.growth_target_level - 1) if props.growth_target_level > 0 else 0
        program = _enrich_at_depth(program, enrich_depth, props.growth_enrich_scale)

    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", program)

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

    tokens = prune_to_max_depth(tokens, props.depth_max)

    # Max length estimation
    max_len_found = props.max_length

    tmp_depth = 0
    branch_lengths = []
    current_branch_length = 0.0
    for t in tokens:
        if t == '[':
            if tmp_depth >= props.leaf_min_depth and current_branch_length > 0:
                branch_lengths.append(current_branch_length)
            tmp_depth += 1
            current_branch_length = 0.0
        elif t == ']':
            if tmp_depth >= props.leaf_min_depth and current_branch_length > 0:
                branch_lengths.append(current_branch_length)
            tmp_depth -= 1
            current_branch_length = 0.0
        elif t.startswith("B") and tmp_depth >= props.leaf_min_depth:
            m = re.match(r"B\d+_\d+F(\d+)", t)
            if m:
                f_bin = int(m.group(1))
                current_branch_length += (f_bin + 0.5) / num_f * max_len_found

    if current_branch_length > 0 and tmp_depth >= props.leaf_min_depth:
        branch_lengths.append(current_branch_length)

    avg_last_depth_length = sum(branch_lengths) / len(branch_lengths) if branch_lengths else 0.5
    fixed_global_leaf_size = avg_last_depth_length * 0.5

    bm_tree = bmesh.new()
    bm_leaf = bmesh.new()
    uv_tree = bm_tree.loops.layers.uv.new("UVMap")
    uv_leaf = bm_leaf.loops.layers.uv.new("UVMap")

    pos, curr_t, curr_p = np.array([0.,0.,0.]), 0.0, 0.0
    rad, depth = props.base_radius, 0
    stack = []
    max_depth = 0
    pts, radii = [pos.copy()], [rad]

    tmp_depth = 0
    for token in tokens:
        if token == '[':
            tmp_depth += 1
            if tmp_depth > max_depth:
                max_depth = tmp_depth
        elif token == ']':
            tmp_depth -= 1

    pos, curr_t, curr_p = np.array([0.,0.,0.]), 0.0, 0.0
    rad, depth = props.base_radius, 0
    stack = []
    pts, radii = [pos.copy()], [rad]
    curr_depth = 0

    for token in tokens:
        if token.startswith("B"):
            m = re.match(r"B(\d+)_(\d+)F(\d+)", token)
            t_bin, p_bin, f_bin = map(int, m.groups())
            target_t = (t_bin+0.5)/num_theta*180
            target_p = (p_bin+0.5)/num_phi*360
            length   = (f_bin+0.5)/num_f*max_len_found

            for s in range(props.steps):
                curr_t += (target_t - curr_t) * (1.0 / props.steps)
                curr_p += (target_p - curr_p) * (1.0 / props.steps)
                curr_t += (props.base_radius / max(rad, 0.01)) * props.gravity_strength
                curr_t += np.random.uniform(-props.smooth_factor, props.smooth_factor)
                direction = np.array([math.sin(math.radians(curr_t))*math.cos(math.radians(curr_p)),
                                      math.sin(math.radians(curr_t))*math.sin(math.radians(curr_p)),
                                      math.cos(math.radians(curr_t))])
                old_pos = pos.copy()
                pos += direction * (length / props.steps)
                rad = max(rad * (props.segment_taper ** (1.0 / props.steps)), props.min_radius)
                pts.append(pos.copy())
                radii.append(rad)
                # Only add leaves at max depth
                if curr_depth == max_depth:
                    add_logical_leaves(bm_leaf, uv_leaf, Vector(old_pos), Vector(pos), direction, rad, fixed_global_leaf_size, props.leaf_density, props.leaf_size, props.leaf_angle)

        elif token == "[":
            create_smooth_branch(bm_tree, uv_tree, pts, radii, props.branch_sides)
            stack.append((pos.copy(), curr_t, curr_p, rad, curr_depth))
            curr_depth += 1
            rad *= props.depth_decay
            pts, radii = [pos.copy()], [rad]

        elif token == "]":
            create_smooth_branch(bm_tree, uv_tree, pts, radii, props.branch_sides)
            pos, curr_t, curr_p, rad, curr_depth = stack.pop()
            pts, radii = [pos.copy()], [rad]

    create_smooth_branch(bm_tree, uv_tree, pts, radii, props.branch_sides)

    # Materials
    leaf_img_path = os.path.join(bpy.path.abspath(props.base_leaf_dir), props.species.capitalize(), f"{props.species.capitalize()}.png")
    bark_img_path = os.path.join(bpy.path.abspath(props.bark_dir), f"{props.species.capitalize()}.jpg")

    mat_tree = bpy.data.materials.new(name="BarkMat")
    mat_tree.use_nodes = True
    bsdf = mat_tree.node_tree.nodes.get("Principled BSDF")
    if bsdf: bsdf.inputs['Roughness'].default_value = 0.0
    if os.path.exists(bark_img_path):
        tex_image = mat_tree.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(bark_img_path)
        uv_map_node = mat_tree.node_tree.nodes.new('ShaderNodeUVMap')
        uv_map_node.uv_map = "UVMap"
        mat_tree.node_tree.links.new(uv_map_node.outputs['UV'], tex_image.inputs['Vector'])
        mat_tree.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    mat_leaf = bpy.data.materials.new(name="LeafMat")
    mat_leaf.use_nodes = True
    mat_leaf.blend_method = 'CLIP'
    mat_leaf.use_backface_culling = False
    bsdf_leaf = mat_leaf.node_tree.nodes.get("Principled BSDF")
    if bsdf_leaf: bsdf_leaf.inputs['Roughness'].default_value = 0.0

    if os.path.exists(leaf_img_path):
        tex_image_leaf = mat_leaf.node_tree.nodes.new('ShaderNodeTexImage')
        tex_image_leaf.image = bpy.data.images.load(leaf_img_path)
        uv_map_node_leaf = mat_leaf.node_tree.nodes.new('ShaderNodeUVMap')
        uv_map_node_leaf.uv_map = "UVMap"

        hsv_node = mat_leaf.node_tree.nodes.new('ShaderNodeHueSaturation')
        hsv_node.inputs['Hue'].default_value = 0.5 - (0.12 * props.leaf_orangeness)
        hsv_node.inputs['Saturation'].default_value = 1.0 + (0.3 * props.leaf_orangeness)
        hsv_node.inputs['Value'].default_value = 1.0 - (0.15 * props.leaf_orangeness)

        mat_leaf.node_tree.links.new(uv_map_node_leaf.outputs['UV'], tex_image_leaf.inputs['Vector'])
        mat_leaf.node_tree.links.new(hsv_node.inputs['Color'], tex_image_leaf.outputs['Color'])
        mat_leaf.node_tree.links.new(bsdf_leaf.inputs['Base Color'], hsv_node.outputs['Color'])
        mat_leaf.node_tree.links.new(bsdf_leaf.inputs['Alpha'], tex_image_leaf.outputs['Alpha'])

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


class LSYSTEM_OT_BuildTree(bpy.types.Operator):
    bl_idname = "object.lsystem_build_tree"
    bl_label = "Force Render Tree"
    bl_description = "Manually trigger a tree generation from the L-System txt"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        generate_tree(context)
        return {'FINISHED'}


class LSYSTEM_OT_AddCameraSun(bpy.types.Operator):
    bl_idname = "object.lsystem_add_camera_sun"
    bl_label = "Add Camera & Sun"
    bl_description = "Adds a sun light and a camera pointing at the tree"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # SUN LIGHT SETUP
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT' and obj.data.type == 'SUN':
                bpy.data.objects.remove(obj, do_unlink=True)
                
        light_data = bpy.data.lights.new(name="Sun", type='SUN')
        light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = (0, -10, 10)
        light_obj.rotation_euler = (0, -math.radians(90), 0)

        # CAMERA SETUP
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        cam_data = bpy.data.cameras.new(name="Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.collection.objects.link(cam_obj)

        tree_obj = bpy.data.objects.get("Tree")
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
            
            cam_dist = size * 3.5 + 1.0
            cam_obj.location = (center_x - cam_dist, center_y, center_z)
            direction = Vector((center_x, center_y, center_z)) - cam_obj.location
            rot_quat = direction.to_track_quat('-Z', 'Y')
            cam_obj.rotation_euler = rot_quat.to_euler()
        else:
            cam_obj.location = (-10, 0, 5)
            cam_obj.rotation_euler = (math.radians(60), 0, math.radians(-90))
            
        bpy.context.scene.camera = cam_obj
        
        return {'FINISHED'}


class LSYSTEM_OT_ExportGrowth(bpy.types.Operator):
    bl_idname = "object.lsystem_export_growth"
    bl_label = "Export Processed L-String"
    bl_description = "Saves the growth/enrichment processed L-System string to a txt file"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.lsystem_props
        filepath = bpy.path.abspath(props.filepath)
        export_path = bpy.path.abspath(props.export_filepath)

        if not os.path.exists(filepath):
            self.report({'ERROR'}, "Input file not found")
            return {'CANCELLED'}
        if not export_path:
            self.report({'ERROR'}, "Export path not set")
            return {'CANCELLED'}

        with open(filepath, "r") as f:
            program = f.read().strip()

        random.seed(props.random_seed)
        if props.growth_target_level > 0:
            program = _expand_tree_to_level(program, props.growth_target_level)
        if props.growth_enrich:
            enrich_depth = (props.growth_target_level - 1) if props.growth_target_level > 0 else 0
            program = _enrich_at_depth(program, enrich_depth, props.growth_enrich_scale)

        export_dir = os.path.dirname(export_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        with open(export_path, "w") as f:
            f.write(program)

        self.report({'INFO'}, f"Exported to {export_path}")
        return {'FINISHED'}


class LSYSTEM_OT_ResetDefaults(bpy.types.Operator):
    bl_idname = "object.lsystem_reset_defaults"
    bl_label = "Reset Defaults"
    bl_description = "Resets all L-System visualizer parameters to their default values"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.lsystem_props
        # Disable auto_update so each property_unset below doesn't trigger
        # a separate generate_tree call; we do one explicit rebuild at the end
        props.auto_update = False
        props.property_unset("filepath")
        props.property_unset("species")
        props.property_unset("leaf_density")
        props.property_unset("leaf_orangeness")
        props.property_unset("leaf_size")
        props.property_unset("leaf_angle")
        props.property_unset("base_leaf_dir")
        props.property_unset("bark_dir")
        props.property_unset("base_radius")
        props.property_unset("max_length")
        props.property_unset("depth_decay")
        props.property_unset("segment_taper")
        props.property_unset("min_radius")
        props.property_unset("gravity_strength")
        props.property_unset("smooth_factor")
        props.property_unset("steps")
        props.property_unset("branch_sides")
        props.property_unset("depth_max")
        props.property_unset("leaf_min_depth")
        props.property_unset("random_seed")
        props.property_unset("growth_target_level")
        props.property_unset("growth_enrich")
        props.property_unset("growth_enrich_scale")
        props.property_unset("auto_update")  # restores default=True
        generate_tree(context)  # single rebuild with default values
        return {'FINISHED'}


class LSYSTEM_PT_Panel(bpy.types.Panel):
    bl_label = "L-System Visualizer"
    bl_idname = "LSYSTEM_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'L-System'

    def draw(self, context):
        layout = self.layout
        props = context.scene.lsystem_props

        # ── Input ────────────────────────────────────────────
        box = layout.box()
        box.label(text="Input", icon='FILE_TEXT')
        box.prop(props, "filepath")
        box.prop(props, "species")

        # ── Seasons / Leaves ─────────────────────────────────
        box = layout.box()
        box.label(text="Seasons & Leaves", icon='OUTLINER_OB_POINTCLOUD')
        box.prop(props, "leaf_density",    text="Density  (Summer → Winter)")
        box.prop(props, "leaf_orangeness", text="Orangeness  (Autumn)")
        box.prop(props, "leaf_size",       text="Size Multiplier")
        box.prop(props, "leaf_angle",      text="Orientation")

        # ── Geometry & Look ───────────────────────────────────
        box = layout.box()
        box.label(text="Geometry & Look", icon='MESH_DATA')
        col = box.column(align=True)
        col.prop(props, "base_radius")
        col.prop(props, "max_length")
        col.prop(props, "depth_decay")
        col.prop(props, "segment_taper")
        col.prop(props, "min_radius")
        col = box.column(align=True)
        col.prop(props, "gravity_strength")
        col.prop(props, "smooth_factor")
        col = box.column(align=True)
        col.prop(props, "steps")
        col.prop(props, "branch_sides")
        col.prop(props, "depth_max")
        col.prop(props, "leaf_min_depth")

        # ── Growth ────────────────────────────────────────────
        box = layout.box()
        box.label(text="Growth & Enrichment", icon='FORCE_TURBULENCE')
        box.prop(props, "growth_target_level", text="Target Depth  (0 = off)")
        row = box.row(align=True)
        row.prop(props, "growth_enrich", text="Enrich at Target Depth")
        if props.growth_enrich:
            row.prop(props, "growth_enrich_scale", text="Scale")

        # ── Texture Paths ─────────────────────────────────────
        box = layout.box()
        box.label(text="Texture Paths", icon='TEXTURE')
        box.prop(props, "base_leaf_dir")
        box.prop(props, "bark_dir")

        # ── Settings ──────────────────────────────────────────
        box = layout.box()
        box.label(text="Settings", icon='PREFERENCES')
        row = box.row(align=True)
        row.prop(props, "auto_update", text="Auto-Update")
        row.prop(props, "random_seed",  text="Seed")

        # ── Export ────────────────────────────────────────────
        box = layout.box()
        box.label(text="Export Processed L-String", icon='EXPORT')
        box.prop(props, "export_filepath", text="Output Path")
        box.operator(LSYSTEM_OT_ExportGrowth.bl_idname, text="Export to .txt", icon='EXPORT')

        # ── Actions ───────────────────────────────────────────
        layout.separator()
        layout.operator(LSYSTEM_OT_BuildTree.bl_idname,    text="Force Render Tree",  icon='OUTLINER_OB_CURVE')
        layout.operator(LSYSTEM_OT_AddCameraSun.bl_idname, text="Add Camera & Sun",    icon='LIGHT_SUN')
        layout.operator(LSYSTEM_OT_ResetDefaults.bl_idname, text="Reset Defaults",     icon='FILE_REFRESH')


class LSystemProperties(bpy.types.PropertyGroup):
    auto_update: bpy.props.BoolProperty(
        name="Auto-Update Live", 
        default=True, 
        description="Rebuilds the tree automatically on parameter change"
    )
    random_seed: bpy.props.IntProperty(
        name="Random Seed", 
        default=42, 
        min=0, 
        description="Seed for random variations (leaves, branch twisting)",
        update=on_property_update
    )
    filepath: bpy.props.StringProperty(
        name="L-System Text File",
        subtype='FILE_PATH',
        default=r"C:\Users\angie\Documents\visualizer_Lsystem\results\txt\symbolic_txt2\tree_0020.txt",
        update=on_property_update
    )
    species: bpy.props.StringProperty(
        name="Species",
        default="Spruce",
        update=on_property_update
    )
    leaf_density: bpy.props.FloatProperty(
        name="Leaf Density",
        default=1.0, min=0.0, max=1.0,
        update=on_property_update
    )
    leaf_orangeness: bpy.props.FloatProperty(
        name="Leaf Orangeness",
        default=0.0, min=0.0, max=1.0,
        update=on_property_update
    )
    leaf_size: bpy.props.FloatProperty(
        name="Leaf Size Multiplier",
        default=1.0, min=0.1, soft_min=0.1, max=5.0, soft_max=5.0,
        update=on_property_update
    )
    leaf_angle: bpy.props.FloatProperty(
        name="Leaf Angle / Orientation",
        default=0.6, min=0.0, max=1.0, description="0.0 for parallel to branch, 1.0 for perpendicular",
        update=on_property_update
    )
    base_leaf_dir: bpy.props.StringProperty(
        name="Leaf Material Dir",
        subtype='DIR_PATH',
        default=r"C:\Users\angie\Documents\the_grove_22_indie\the_grove_22\templates\Transparency_Twigs_modify\TwigsLibrary",
        update=on_property_update
    )
    bark_dir: bpy.props.StringProperty(
        name="Bark Material Dir",
        subtype='DIR_PATH',
        default=r"C:\Users\angie\Documents\the_grove_22_indie\the_grove_22\templates\BarkTextures",
        update=on_property_update
    )
    base_radius: bpy.props.FloatProperty(name="Base Radius", default=0.15, min=0.001, soft_min=0.01, update=on_property_update)
    max_length: bpy.props.FloatProperty(name="Max Length", default=6.8, min=0.1, soft_min=0.1, update=on_property_update)
    depth_decay: bpy.props.FloatProperty(name="Depth Decay", default=0.72, min=0.001, max=1.0, update=on_property_update)
    segment_taper: bpy.props.FloatProperty(name="Segment Taper", default=0.94, min=0.001, max=1.0, update=on_property_update)
    min_radius: bpy.props.FloatProperty(name="Min Radius", default=0.006, min=0.0001, soft_min=0.001, update=on_property_update)
    gravity_strength: bpy.props.FloatProperty(name="Gravity Strength", default=0.03, min=0.0, soft_min=0.0, update=on_property_update)
    smooth_factor: bpy.props.FloatProperty(name="Smooth Factor", default=0.1, min=0.0, soft_min=0.0, update=on_property_update)
    steps: bpy.props.IntProperty(name="Steps (Resolution)", default=5, min=1, max=50, update=on_property_update)
    branch_sides: bpy.props.IntProperty(name="Branch Sides", default=16, min=3, max=128, update=on_property_update)
    depth_max: bpy.props.IntProperty(name="Depth Max", default=2, min=0, update=on_property_update)
    leaf_min_depth: bpy.props.IntProperty(name="Leaf Min Depth", default=2, min=0, update=on_property_update)
    growth_target_level: bpy.props.IntProperty(
        name="Target Depth Level",
        default=0, min=0, max=10,
        description="0 = use file as-is. 1 = prune to depth 1. 2+ = grow/fill branches up to that depth",
        update=on_property_update
    )
    growth_enrich: bpy.props.BoolProperty(
        name="Enrich Branches",
        default=False,
        description="Add extra sub-branches at the target depth level (same-level densification)",
        update=on_property_update
    )
    growth_enrich_scale: bpy.props.IntProperty(
        name="Enrich Scale",
        default=5, min=1, max=50,
        description="Number of new branches added per parent when Enrich Branches is enabled",
        update=on_property_update
    )
    export_filepath: bpy.props.StringProperty(
        name="Export Path",
        subtype='FILE_PATH',
        default=r"C:\Users\angie\Documents\visualizer_Lsystem\results\growth_export\tree_export.txt",
        description="Output file path for the exported processed L-System string"
    )


classes = (
    LSYSTEM_OT_BuildTree,
    LSYSTEM_OT_AddCameraSun,
    LSYSTEM_OT_ExportGrowth,
    LSYSTEM_OT_ResetDefaults,
    LSYSTEM_PT_Panel,
    LSystemProperties
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.lsystem_props = bpy.props.PointerProperty(type=LSystemProperties)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.lsystem_props

if __name__ == "__main__":
    register()
