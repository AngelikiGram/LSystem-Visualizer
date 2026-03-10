import bpy
import numpy as np
import math
import re


# ── Radius config ──────────────────────────────────────────
BASE_RADIUS   = 0.04   # trunk radius in metres
SEGMENT_TAPER = 0.92   # each consecutive F segment shrinks radius by this factor
BRANCH_DECAY  = 0.65   # child branch starts at this fraction of parent radius
MIN_RADIUS    = 0.002  # thinnest allowed twig
# ──────────────────────────────────────────────────────────


# ============================================================
#  TokenType and LSystemTokenizerV2 for discretization
# ============================================================

class TokenType:
    F   = 0   # F(x)
    R   = 1   # R(a,b,c)
    LBR = 2   # "["
    RBR = 3   # "]"
    A   = 4   # "A->"
    EOS = 5   # end-of-sequence
    PAD = 6   # padding

NUM_TYPES = 7

class LSystemTokenizerV2:
    """
    Converts L-system strings into:
        type_ids:    list[int]
        value_ids:   list[[ax,by,cz]]  (padded to length 3)
    """
    def __init__(self, f_bins=128, r_bins=360):
        self.f_bins = f_bins
        self.r_bins = r_bins
        # Regex patterns
        self.re_F = re.compile(r"F\(([+-]?\d+(\.\d+)?)\)")
        self.re_R = re.compile(
            r"R\(([+-]?\d+(\.\d+)?),([+-]?\d+(\.\d+)?),([+-]?\d+(\.\d+)?)\)"
        )
        # Value ranges
        self.f_min, self.f_max = 0.0, 1.0
        self.r_min, self.r_max = -180.0, 180.0

    def bin_F(self, x):
        x = float(x)
        x = np.clip(x, self.f_min, self.f_max)
        idx = int((x - self.f_min) / (self.f_max - self.f_min) * (self.f_bins - 1))
        return idx

    def bin_R(self, a, b, c):
        def bin_single(v):
            v = np.clip(v, self.r_min, self.r_max)
            return int((v - self.r_min) / (self.r_max - self.r_min) * (self.r_bins - 1))
        return bin_single(a), bin_single(b), bin_single(c)

    def encode(self, s):
        types, values = [], []
        i = 0
        L = len(s)
        while i < L:
            if s.startswith("A->", i):
                types.append(TokenType.A)
                values.append([0,0,0])
                i += 3
                continue
            if s[i] == "[":
                types.append(TokenType.LBR)
                values.append([0,0,0])
                i += 1
                continue
            if s[i] == "]":
                types.append(TokenType.RBR)
                values.append([0,0,0])
                i += 1
                continue
            mF = self.re_F.match(s, i)
            if mF:
                x = float(mF.group(1))
                types.append(TokenType.F)
                values.append([self.bin_F(x), 0, 0])
                i = mF.end()
                continue
            mR = self.re_R.match(s, i)
            if mR:
                a, b, c = float(mR.group(1)), float(mR.group(3)), float(mR.group(5))
                ax, by, cz = self.bin_R(a, b, c)
                types.append(TokenType.R)
                values.append([ax, by, cz])
                i = mR.end()
                continue
            i += 1
        types.append(TokenType.EOS)
        values.append([0,0,0])
        return types, values

    def decode(self, types, values):
        out = []
        for t, v in zip(types, values):
            if t == TokenType.F:
                f = self.f_min + v[0] / (self.f_bins - 1) * (self.f_max - self.f_min)
                out.append(f"F({f:.3f})")
            elif t == TokenType.R:
                def inv(idx):
                    return self.r_min + idx / (self.r_bins - 1) * (self.r_max - self.r_min)
                out.append(f"R({inv(v[0]):.3f},{inv(v[1]):.3f},{inv(v[2]):.3f})")
            elif t == TokenType.LBR:
                out.append("[")
            elif t == TokenType.RBR:
                out.append("]")
            elif t == TokenType.A:
                out.append("A->")
            elif t == TokenType.EOS:
                break
        return "".join(out)


# ------------------------------------------------------------
# Tokenization (same as simplification)
# ------------------------------------------------------------

def load_lsystem(path):
    txt = open(path, "r").read()
    if "->" in txt:
        txt = txt.split("->", 1)[1].strip()

    token_pattern = r"R\([^)]*\)|F\([^)]*\)|\[|\]"
    return re.findall(token_pattern, txt)


# ------------------------------------------------------------
# Rotation matrices
# ------------------------------------------------------------

def rot_x(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(a):
    a = math.radians(a)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


# ------------------------------------------------------------
# Turtle class
# ------------------------------------------------------------

class Turtle:
    def __init__(self, pos=None, R=None, radius=None, vert_idx=0):
        self.pos      = np.array([0,0,0], float) if pos is None else np.array(pos, float)
        self.R        = np.eye(3) if R is None else np.array(R, float)
        self.radius   = BASE_RADIUS if radius is None else radius
        self.vert_idx = vert_idx

    def copy(self):
        return Turtle(self.pos.copy(), self.R.copy(), self.radius, self.vert_idx)


# ------------------------------------------------------------
# Build tree geometry  (with smooth radius decay)
# ------------------------------------------------------------


def build_tree(tokens, obj_name="LSystemTree"):
    verts  = []
    edges  = []
    radii  = []   # one radius value per vertex

    turtle = Turtle()
    verts.append(tuple(turtle.pos))
    radii.append(turtle.radius)
    turtle.vert_idx = 0

    stack = []

    for t in tokens:
        if t.startswith("R("):
            a = t[2:-1].split(",")
            rx, ry, rz = map(float, a)
            R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
            turtle.R = turtle.R @ R

        elif t.startswith("F("):
            d = float(t[2:-1])
            forward  = turtle.R @ np.array([0, 0, 1])
            new_pos  = turtle.pos + forward * d
            r_end    = max(turtle.radius * SEGMENT_TAPER, MIN_RADIUS)

            new_idx = len(verts)
            verts.append(tuple(new_pos))
            radii.append(r_end)
            edges.append((turtle.vert_idx, new_idx))

            turtle.pos      = new_pos
            turtle.vert_idx = new_idx
            turtle.radius   = r_end

        elif t == "[":
            stack.append(turtle.copy())
            # child branch starts thinner than parent at this junction
            turtle.radius = max(turtle.radius * BRANCH_DECAY, MIN_RADIUS)

        elif t == "]":
            if stack:
                turtle = stack.pop()
            else:
                print("Warning: Stack is empty on ']', possible unbalanced L-system string.")

    # ── Create Grease Pencil object ──────────────────────────
    gp_data = bpy.data.grease_pencils.new(obj_name)
    gp_layer = gp_data.layers.new("TreeLayer")
    gp_frame = gp_layer.frames.new(1)
    drawing = gp_frame.drawing
    
    # We have `len(edges)` strokes, each with 2 points.
    drawing.add_strokes([2] * len(edges))
    
    # Set coordinates and thickness (radius)
    pos_attr = drawing.attributes['position']
    if 'radius' not in drawing.attributes:
        drawing.attributes.new('radius', 'FLOAT', 'POINT')
    radius_attr = drawing.attributes['radius']
    
    points_flat = []
    radii_flat = []
    
    for (start_idx, end_idx) in edges:
        points_flat.extend(verts[start_idx])
        points_flat.extend(verts[end_idx])
        radii_flat.append(radii[start_idx])
        radii_flat.append(radii[end_idx])
        
    pos_attr.data.foreach_set('vector', points_flat)
    radius_attr.data.foreach_set('value', radii_flat)

    # Optional: material for Grease Pencil
    mat = bpy.data.materials.new(name=f"{obj_name}_Mat")
    gp_data.materials.append(mat)

    obj = bpy.data.objects.new(obj_name, gp_data)
    bpy.context.scene.collection.objects.link(obj)

    return obj


# ------------------------------------------------------------
# RUN INSIDE BLENDER
# ------------------------------------------------------------


import glob
import os

folder = r"C:\Users\angie\Documents\P3\inference_data\exports\val_p3_eff16"

folder = "./results/simplified" # _motif"
# folder = "./results/enriched" # _motif"
txt_files = sorted(glob.glob(os.path.join(folder, "*.txt")))

for i, path in enumerate(txt_files):
    print(f"Processing file {i+1}/{len(txt_files)}: {path}")
    tokens = load_lsystem(path)
    base_name = os.path.splitext(os.path.basename(path))[0]
    print(f"Loaded tokens from {base_name}: {len(tokens)}")
    obj = build_tree(tokens, obj_name=base_name)
    obj.location.y = i * 5
    print(f"Tree built: {obj}")

# & "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --python .\visualize_03.py