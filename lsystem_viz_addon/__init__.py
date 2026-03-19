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

def _safe_normalized(vec, fallback=None):
    if vec.length > 1e-8:
        return vec.normalized()
    if fallback is not None and fallback.length > 1e-8:
        return fallback.normalized()
    return Vector((1, 0, 0))

def create_smooth_branch(bm, uv_layer, pts, radii, branch_sides):
    if len(pts) < 2: return
    
    rings = []
    prev_side = None
    lengths = [0.0]
    
    for i in range(1, len(pts)):
        lengths.append(lengths[-1] + (Vector(pts[i]) - Vector(pts[i-1])).length)
        
    for i in range(len(pts)):
        pt, rad = Vector(pts[i]), radii[i]
        if i == 0:
            direction = _safe_normalized(Vector(pts[1]) - pt)
        elif i == len(pts) - 1:
            direction = _safe_normalized(pt - Vector(pts[i-1]))
        else:
            d_in = _safe_normalized(pt - Vector(pts[i-1]))
            d_out = _safe_normalized(Vector(pts[i+1]) - pt, d_in)
            direction = _safe_normalized(d_in + d_out, d_out)
            if (d_in + d_out).length < 1e-4:
                direction = d_out
        
        if prev_side is None:
            ref = Vector((0,0,1)) if abs(direction.z) < 0.99 else Vector((1,0,0))
            side = _safe_normalized(direction.cross(ref), Vector((1, 0, 0)))
        else:
            side = _safe_normalized(prev_side - direction * prev_side.dot(direction), direction.cross(Vector((0, 0, 1))))
        
        up = _safe_normalized(side.cross(direction), Vector((0, 0, 1)))
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


def _expand_tree_to_level(raw_string, target_level, num_branches, branch_length_scale=0.5, num_steps=1, d1_branches=0, d1_branch_length_scale=0.5):
    """Prune (target_level=1) or grow terminal branches to reach target_level."""
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", raw_string)

    # Determine the maximum F-bin present in the source tree so child F values
    # can be capped to a valid range (prevents children longer than parent).
    existing_f_vals = [int(re.match(r"B\d+_\d+F(\d+)", t).group(1))
                       for t in tokens if t.startswith('B')]
    max_f_in_tree = max(existing_f_vals) if existing_f_vals else 9

    if target_level <= 1 and d1_branches == 0:
        # Pure prune: remove all branches deeper than target_level, no growth
        output, depth, i = [], 0, 0
        limit_depth = target_level + 1
        while i < len(tokens):
            t = tokens[i]
            if t == '[':
                depth += 1
                if depth == limit_depth:
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
                if depth < limit_depth:
                    output.append(t)
            i += 1
        return "".join(output)

    # Growth path — also entered when d1_branches > 0 even if target_level <= 1
    existing_t_vals = [int(re.match(r"B(\d+)_\d+F\d+", t).group(1)) for t in tokens if t.startswith('B')]
    existing_p_vals = [int(re.match(r"B\d+_(\d+)F\d+", t).group(1)) for t in tokens if t.startswith('B')]
    max_t_bin = max(existing_t_vals) if existing_t_vals else 11
    max_p_bin = max(existing_p_vals) if existing_p_vals else 11

    # When d1_branches > 0 we need to traverse at least to depth=2 to add sub-branches
    # to depth-1 branches; otherwise stop at target_level
    depth_limit = max(target_level, 2) if d1_branches > 0 else target_level

    def _rand_branch_angles(parent_t, parent_p):
        p_r = (parent_p + random.randint(1, max_p_bin)) % (max_p_bin + 1)
        t_offset = random.randint(-1, 3)
        t_r = max(0, min(max_t_bin, parent_t + t_offset))
        return t_r, p_r

    def _forced_growth(d_remaining, parent_f, parent_t=None, parent_p=None):
        if d_remaining <= 0:
            return []
        if parent_t is None: parent_t = max_t_bin // 2
        if parent_p is None: parent_p = 0
        child_f = max(0, round(parent_f * 0.5))
        result = []
        for _ in range(num_branches):
            result.append('[')
            t_r, p_r = _rand_branch_angles(parent_t, parent_p)
            f_r = max(0, child_f + random.choice([-1, 0, 0]))
            result.append(f"B{t_r}_{p_r}F{f_r}")
            result.extend(_forced_growth(d_remaining - 1, child_f, t_r, p_r))
            result.append(']')
        return result

    def _process(token_list, depth):
        if depth > depth_limit:
            return token_list
        output, i = [], 0
        b_tokens_indices = []

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
                b_tokens_indices.append(len(output))
                output.append(t)
            else:
                output.append(t)
            i += 1

        # Count of new branches to insert at this depth:
        #   depth == 1: add d1_branches extra sub-branches to this D1 branch (keeping existing ones)
        #   depth >= 1 and < target_level: add num_branches for general growth
        # Both can apply at depth=1 simultaneously when target_level >= 2
        count = 0
        if depth == 1 and d1_branches > 0:
            count += d1_branches
        if depth >= 1 and depth < target_level:
            count += num_branches

        if count > 0 and b_tokens_indices:
            # At depth=1, d1_branches use their own length scale; all other depths use branch_length_scale
            scale = d1_branch_length_scale if (depth == 1 and d1_branches > 0) else branch_length_scale
            child_f = max(0, min(round(max_f_in_tree * scale), max_f_in_tree))
            # New branches at depth d become depth d+1; they need (target_level - (d+1)) more forced levels
            levels_extra = max(0, target_level - depth - 1)

            chosen_positions = random.choices(b_tokens_indices, k=count)
            insertions = {}
            for idx in chosen_positions:
                insertions[idx] = insertions.get(idx, 0) + 1

            new_output = []
            for idx, item in enumerate(output):
                if idx in insertions and item.startswith('B') and num_steps > 1:
                    m_seg = re.match(r"B(\d+)_(\d+)F(\d+)", item)
                    t_seg = int(m_seg.group(1)); p_seg = int(m_seg.group(2)); f_seg = int(m_seg.group(3))
                    split = random.randint(1, num_steps - 1)
                    f_before = max(0, round(f_seg * split / num_steps))
                    f_after  = max(0, f_seg - f_before)
                    new_output.append(f"B{t_seg}_{p_seg}F{f_before}")
                    for _ in range(insertions[idx]):
                        t_r, p_r = _rand_branch_angles(t_seg, p_seg)
                        f_r = max(0, child_f + random.choice([-1, 0, 0]))
                        sub = ['[', f"B{t_r}_{p_r}F{f_r}"]
                        sub.extend(_forced_growth(levels_extra, child_f, t_r, p_r))
                        sub.append(']')
                        new_output.extend(sub)
                    new_output.append(f"B{t_seg}_{p_seg}F{f_after}")
                else:
                    new_output.append(item)
                    if idx in insertions:
                        m_par = re.match(r"B(\d+)_(\d+)F\d+", item)
                        pt = int(m_par.group(1)) if m_par else max_t_bin // 2
                        pp = int(m_par.group(2)) if m_par else 0
                        for _ in range(insertions[idx]):
                            t_r, p_r = _rand_branch_angles(pt, pp)
                            f_r = max(0, child_f + random.choice([-1, 0, 0]))
                            sub = ['[', f"B{t_r}_{p_r}F{f_r}"]
                            sub.extend(_forced_growth(levels_extra, child_f, t_r, p_r))
                            sub.append(']')
                            new_output.extend(sub)
            output = new_output

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
    for obj_name in ["Tree", "Leaves", "Tree_Skeleton"]:
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

    with open(filepath, "r") as f:
        program = f.read().strip()

    # Auto-detect bin counts from the raw file so the tree is always decoded correctly
    # regardless of what the UI sliders show. UI values act as a minimum override.
    raw_b_tokens = re.findall(r"B(\d+)_(\d+)F(\d+)", program)
    if raw_b_tokens:
        max_t = max(int(t) for t, p, f in raw_b_tokens)
        max_p = max(int(p) for t, p, f in raw_b_tokens)
        max_f = max(int(f) for t, p, f in raw_b_tokens)
        num_theta = max(max_t + 1, props.num_bins_theta)
        num_phi   = max(max_p + 1, props.num_bins_phi)
        # If all F bins are 0 the file uses a 1-bin F encoding; treat it as num_bins_f
        # so length = (0+0.5)/num_bins_f*max_len gives a sensible per-segment length
        num_f     = max(max_f + 1, props.num_bins_f)
    else:
        num_theta = props.num_bins_theta
        num_phi   = props.num_bins_phi
        num_f     = props.num_bins_f

    # Apply growth / enrichment from growth.py logic
    random.seed(props.random_seed)
    if props.growth_target_level > 0 or props.growth_d1_branches > 0:
        program = _expand_tree_to_level(program, props.growth_target_level, props.growth_branches, props.growth_branch_length, props.steps, props.growth_d1_branches, props.growth_d1_branch_length)
    if props.growth_enrich:
        # enrich_depth is the PARENT depth: new sub-branches appear at (enrich_depth+1) = target_level
        enrich_depth = (props.growth_target_level - 1) if props.growth_target_level > 0 else 0
        program = _enrich_at_depth(program, enrich_depth, props.growth_enrich_scale)

    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", program)

    def prune_to_max_depth(tokens, max_depth):
        pruned = []
        depth = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == '[':
                if depth < max_depth:
                    pruned.append(t)
                    depth += 1
                else:
                    # Skip entire subtree at this bracket
                    balance = 1
                    i += 1
                    while i < len(tokens) and balance > 0:
                        if tokens[i] == '[': balance += 1
                        elif tokens[i] == ']': balance -= 1
                        i += 1
                    continue  # i already advanced past the closing ]
            elif t == ']':
                depth -= 1
                pruned.append(t)
            else:
                pruned.append(t)
            i += 1
        return pruned

    # d1_branches adds depth-2 branches to depth-1 branches, so ensure depth_max >= 2 when active
    d1_min_depth = 2 if props.growth_d1_branches > 0 else 0
    effective_max_depth = max(props.depth_max, props.growth_target_level + 1, d1_min_depth) if (props.growth_target_level > 0 or props.growth_d1_branches > 0) else props.depth_max
    tokens = prune_to_max_depth(tokens, effective_max_depth)

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
    bm_skel = bmesh.new()
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

    # Clamp leaf_min_depth to the actual tree depth so leaves always appear
    # on the deepest available branches even when growth_target_level < leaf_min_depth
    effective_leaf_min_depth = min(props.leaf_min_depth, max_depth)

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

            start_t = curr_t
            start_p = curr_p
            gravity_acc = 0.0
            n_steps = props.steps_d1 if curr_depth == 1 else props.steps

            for s in range(n_steps):
                progress = (s + 1) / n_steps

                # Power-curve smoothing: sf=1 → linear, sf>1 → ease-in (droopy/natural)
                # depth 0 (trunk): always linear
                # depth 1: smooth_factor_d1
                # depth 2+: smooth_factor
                if curr_depth == 1:
                    sf = max(0.01, props.smooth_factor_d1)
                elif curr_depth >= 2:
                    sf = max(0.01, props.smooth_factor)
                else:
                    sf = 1.0
                weight = progress ** sf
                
                curr_t = start_t + (target_t - start_t) * weight
                curr_p = start_p + (target_p - start_p) * weight

                # Accumulate gravity evenly across the steps
                gravity_acc += ((props.base_radius / max(rad, 0.01)) * props.gravity_strength) / n_steps
                curr_t += gravity_acc
                
                direction = np.array([math.sin(math.radians(curr_t))*math.cos(math.radians(curr_p)),
                                      math.sin(math.radians(curr_t))*math.sin(math.radians(curr_p)),
                                      math.cos(math.radians(curr_t))])
                old_pos = pos.copy()
                pos += direction * (length / n_steps)
                rad = max(rad * (props.segment_taper ** (1.0 / n_steps)), props.min_radius)
                pts.append(pos.copy())
                radii.append(rad)
                
                # Skeleton
                v0_skel = bm_skel.verts.new(old_pos)
                v1_skel = bm_skel.verts.new(pos)
                try:
                    bm_skel.edges.new((v0_skel, v1_skel))
                except ValueError:
                    pass

                # Add leaves at all depths >= effective_leaf_min_depth
                if curr_depth >= effective_leaf_min_depth:
                    add_logical_leaves(bm_leaf, uv_leaf, Vector(old_pos), Vector(pos), direction, rad, fixed_global_leaf_size, props.leaf_density, props.leaf_size, props.leaf_angle)

        elif token == "[":
            stack.append((pos.copy(), curr_t, curr_p, rad, curr_depth, pts, radii))
            curr_depth += 1
            rad *= props.depth_decay
            pts, radii = [pos.copy()], [rad]

        elif token == "]":
            create_smooth_branch(bm_tree, uv_tree, pts, radii, props.branch_sides)
            pos, curr_t, curr_p, rad, curr_depth, pts, radii = stack.pop()

    create_smooth_branch(bm_tree, uv_tree, pts, radii, props.branch_sides)

    # Materials
    leaf_img_path = os.path.join(bpy.path.abspath(props.base_leaf_dir), props.species.capitalize(), f"{props.species.capitalize()}.png")
    bark_img_path = os.path.join(bpy.path.abspath(props.bark_dir), f"{props.species.capitalize()}.jpg")

    mat_tree = bpy.data.materials.new(name="BarkMat")
    mat_tree.use_nodes = True
    bsdf = mat_tree.node_tree.nodes.get("Principled BSDF")
    if bsdf: 
        bsdf.inputs['Roughness'].default_value = 0.0
        if 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.0
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
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
    if bsdf_leaf: 
        bsdf_leaf.inputs['Roughness'].default_value = 0.0
        if 'Specular' in bsdf_leaf.inputs:
            bsdf_leaf.inputs['Specular'].default_value = 0.0
        if 'Specular IOR Level' in bsdf_leaf.inputs:
            bsdf_leaf.inputs['Specular IOR Level'].default_value = 0.0

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
    for bm, name in [(bm_tree, "Tree"), (bm_leaf, "Leaves"), (bm_skel, "Tree_Skeleton")]:
        if name == "Tree_Skeleton":
            me = bpy.data.meshes.new(name)
            bm.to_mesh(me)
            bm.free()
            obj = bpy.data.objects.new(name, me)
            bpy.context.collection.objects.link(obj)
            obj.hide_viewport = True
            obj.hide_render = True
            continue

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


class LSYSTEM_OT_RenderOriginal(bpy.types.Operator):
    bl_idname = "object.lsystem_render_original"
    bl_label = "Original"
    bl_description = "Render the original tree from the source file without growth or enrichment"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.lsystem_props
        auto_update = props.auto_update
        props.auto_update = False
        props.growth_target_level = 0
        props.growth_enrich = False
        props.auto_update = auto_update
        generate_tree(context)
        return {'FINISHED'}


class LSYSTEM_OT_ConvertToGreasePencil(bpy.types.Operator):
    bl_idname = "object.lsystem_convert_to_grease_pencil"
    bl_label = "Convert to GREASE_PENCIL"
    bl_description = "Convert the Tree skeleton to a uniform outline Grease Pencil object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        skeleton = bpy.data.objects.get("Tree_Skeleton")
        if not skeleton:
            self.report({'WARNING'}, "No 'Tree_Skeleton' object found. Generate a tree first.")
            return {'CANCELLED'}
        
        # Deselect all
        for obj in context.selected_objects:
            obj.select_set(False)
            
        # Duplicate the skeleton so we don't consume it
        new_skel = skeleton.copy()
        new_skel.data = skeleton.data.copy()
        new_skel.name = "Tree_GreasePencil"
        context.collection.objects.link(new_skel)
        
        # Select and make it active
        new_skel.hide_viewport = False
        new_skel.hide_render = False
        new_skel.select_set(True)
        context.view_layer.objects.active = new_skel
        
        # Convert
        bpy.ops.object.convert(target='GREASEPENCIL')
        
        gp_obj = context.view_layer.objects.active
        if gp_obj and gp_obj.type == 'GREASEPENCIL':
            bpy.ops.object.modifier_add(type='GREASE_PENCIL_THICKNESS')
            mod = gp_obj.modifiers[-1]
            mod.use_uniform_thickness = True
            mod.thickness = 0.005
            
        self.report({'INFO'}, "Created uniform Tree Grease Pencil object.")
        return {'FINISHED'}


class LSYSTEM_OT_AddGroundPlane(bpy.types.Operator):
    bl_idname = "object.lsystem_add_ground_plane"
    bl_label = "Add Ground Plane & White BG"
    bl_description = "Adds a white plane precisely under the tree and sets the world background to pure white"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tree_obj = bpy.data.objects.get("Tree")
        if not tree_obj:
            self.report({'WARNING'}, "No 'Tree' object found.")
            return {'CANCELLED'}

        # Calculate bbox specifically to place the plane directly beneath
        import mathutils
        bbox_world = [tree_obj.matrix_world @ mathutils.Vector(corner) for corner in tree_obj.bound_box]
        min_z = min(v.z for v in bbox_world)
        center_x = sum(v.x for v in bbox_world) / 8.0
        center_y = sum(v.y for v in bbox_world) / 8.0

        # Remove old GroundPlane if it exists to avoid piling them up
        old_plane = bpy.data.objects.get("GroundPlane")
        if old_plane:
            bpy.data.objects.remove(old_plane, do_unlink=True)
            
        bpy.ops.mesh.primitive_plane_add(
            size=10,
            location=(center_x, center_y, min_z - 0.01)
        )
        plane = context.active_object
        plane.name = "GroundPlane"

        # Create white material for plane
        mat = bpy.data.materials.get("WhitePlaneMaterial")
        if not mat:
            mat = bpy.data.materials.new(name="WhitePlaneMaterial")
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
            bsdf.inputs["Roughness"].default_value = 1.0 # purely matte
            if 'Specular' in bsdf.inputs: bsdf.inputs['Specular'].default_value = 0.0
            if 'Specular IOR Level' in bsdf.inputs: bsdf.inputs['Specular IOR Level'].default_value = 0.0

        if not plane.data.materials:
            plane.data.materials.append(mat)
        else:
            plane.data.materials[0] = mat

        # Set white world background
        world = context.scene.world
        if world:
            world.use_nodes = True
            bg = world.node_tree.nodes.get("Background")
            if bg:
                bg.inputs[0].default_value = (1, 1, 1, 1) # Pure White
                bg.inputs[1].default_value = 1.0          # Strength

        self.report({'INFO'}, "Added ground plane and white environment.")
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
        if props.growth_target_level > 0 or props.growth_d1_branches > 0:
            program = _expand_tree_to_level(program, props.growth_target_level, props.growth_branches, props.growth_branch_length, props.steps, props.growth_d1_branches, props.growth_d1_branch_length)
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
        props.property_unset("num_bins_theta")
        props.property_unset("num_bins_phi")
        props.property_unset("num_bins_f")
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
        props.property_unset("smooth_factor_d1")
        props.property_unset("steps")
        props.property_unset("steps_d1")
        props.property_unset("branch_sides")
        props.property_unset("depth_max")
        props.property_unset("leaf_min_depth")
        props.property_unset("random_seed")
        props.property_unset("growth_target_level")
        props.property_unset("growth_d1_branches")
        props.property_unset("growth_d1_branch_length")
        props.property_unset("growth_branches")
        props.property_unset("growth_branch_length")
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
        col = box.column(align=True)
        col.prop(props, "num_bins_theta")
        col.prop(props, "num_bins_phi")
        col.prop(props, "num_bins_f")

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
        col.prop(props, "smooth_factor_d1")
        col = box.column(align=True)
        col.prop(props, "steps")
        col.prop(props, "steps_d1")
        col.prop(props, "branch_sides")
        col.prop(props, "depth_max")
        col.prop(props, "leaf_min_depth")

        # ── Growth ────────────────────────────────────────────
        box = layout.box()
        box.label(text="Growth & Enrichment", icon='FORCE_TURBULENCE')
        box.prop(props, "growth_target_level", text="Target Depth  (0 = off)")
        box.prop(props, "growth_d1_branches", text="Extra D1 Sub-Branches")
        box.prop(props, "growth_d1_branch_length", text="D1 Sub-Branch Length")
        box.prop(props, "growth_branches", text="New Branches Per Parent")
        box.prop(props, "growth_branch_length", text="New Branch Length")

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
        row = layout.row(align=True)
        row.operator(LSYSTEM_OT_BuildTree.bl_idname, text="Force Render Tree", icon='OUTLINER_OB_CURVE')
        row.operator(LSYSTEM_OT_RenderOriginal.bl_idname, text="Original", icon='FILE_REFRESH')
        layout.operator(LSYSTEM_OT_ConvertToGreasePencil.bl_idname, text="Convert to GREASE_PENCIL", icon='GREASEPENCIL')
        layout.operator(LSYSTEM_OT_AddGroundPlane.bl_idname, text="Ground Plane & White BG", icon='MESH_PLANE')
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
        default=r"C:\Users\angie\Documents\treesformer_new\results\austria_1290\tree_3.txt",
        update=on_property_update
    )
    num_bins_theta: bpy.props.IntProperty(
        name="Num Bins Theta",
        default=12, min=1, max=100,
        update=on_property_update
    )
    num_bins_phi: bpy.props.IntProperty(
        name="Num Bins Phi",
        default=12, min=1, max=100,
        update=on_property_update
    )
    num_bins_f: bpy.props.IntProperty(
        name="Num Bins F",
        default=10, min=1, max=100,
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
    smooth_factor: bpy.props.FloatProperty(
        name="Smooth Factor (D2+)",
        default=2.0, min=0.1, soft_min=0.5, soft_max=5.0,
        description="Branch curvature for depth 2 and deeper: 1=linear, 2=natural droop, 3=strong droop. Depth 1 uses Smooth Factor D1.",
        update=on_property_update
    )
    smooth_factor_d1: bpy.props.FloatProperty(
        name="Smooth Factor D1",
        default=3.0, min=0.1, soft_min=0.5, soft_max=8.0,
        description="Extra curvature specifically for depth-1 branches (direct children of trunk). Higher = more arching/drooping.",
        update=on_property_update
    )
    steps: bpy.props.IntProperty(name="Steps (Resolution)", default=5, min=1, max=50, update=on_property_update)
    steps_d1: bpy.props.IntProperty(name="Steps D1", default=10, min=1, max=50, description="Sub-segments per depth-1 branch segment. More steps = smoother curvature for primary branches.", update=on_property_update)
    branch_sides: bpy.props.IntProperty(name="Branch Sides", default=16, min=3, max=128, update=on_property_update)
    depth_max: bpy.props.IntProperty(name="Depth Max", default=2, min=0, update=on_property_update)
    leaf_min_depth: bpy.props.IntProperty(name="Leaf Min Depth", default=2, min=0, update=on_property_update)
    growth_target_level: bpy.props.IntProperty(
        name="Target Depth Level",
        default=0, min=0, max=10,
        description="0 = use file as-is. 1 = prune to depth 1. 2+ = grow/fill branches up to that depth",
        update=on_property_update
    )
    growth_d1_branches: bpy.props.IntProperty(
        name="Extra D1 Sub-Branches",
        default=0, min=0, max=30,
        description="Extra sub-branches to add to each depth-1 branch. These become depth-2 branches. When Target Depth >= 2, they also get their own children.",
        update=on_property_update
    )
    growth_d1_branch_length: bpy.props.FloatProperty(
        name="D1 Sub-Branch Length",
        default=0.5, min=0.05, max=1.0,
        description="Length scale of branches added by Extra D1 Sub-Branches (relative to parent segment max length)",
        update=on_property_update
    )
    growth_branches: bpy.props.IntProperty(
        name="New Branches",
        default=2, min=0, max=20,
        description="Number of new sub-branches generated when adding a new level. 0 means it behaves effectively like depth_max.",
        update=on_property_update
    )
    growth_branch_length: bpy.props.FloatProperty(
        name="New Branch Length",
        default=0.5, min=0.05, max=1.0,
        description="Length scale of newly added branches relative to their parent segment (0.05 = very short, 1.0 = same length as parent)",
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
    LSYSTEM_OT_RenderOriginal,
    LSYSTEM_OT_ConvertToGreasePencil,
    LSYSTEM_OT_AddGroundPlane,
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
