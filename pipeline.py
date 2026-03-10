import os
import gzip
import json
import math
import re
import numpy as np
import argparse
from typing import List

# ===================================================================
# CONFIG
# ===================================================================
# Pass 1 Config
EPS = 1e-8
FWD = np.array([0, 0, 1.0])
MIN_SEG = 0.0
MIN_BRANCH = 0.0
MIN_E2E = 0.0
MERGE_DIR_TOL = 0.0
ROT_TOL = 0.0
ANGLE_DEC = 1
LEN_DEC = 1

# Pass 2 Config
MAX_TOKENS_NUM = 1024 * 2 # 4 

# Simplification Step Limits & Params
SIMPL_MAX_ITERATIONS   = 100
SIMPL_MIN_CHAIN_INIT   = 2
SIMPL_BASE_ROT_INIT    = 5.0 
SIMPL_K_INIT           = 8.0
SIMPL_DELTA_INIT       = 0.1
SIMPL_PRUNE_LEN_INIT   = 0.0

SIMPL_ROT_MULTIPLIER   = 1.3
SIMPL_K_MULTIPLIER     = 1.2

SIMPL_PRUNE_SOFT_CAP   = 0.05
SIMPL_PRUNE_STEP_SMALL = 0.01 # increase: branches removed faster 
SIMPL_PRUNE_STEP_LARGE = 0.01
SIMPL_MAX_KEEP_EVERY   = 15 # decrease: more branches kept at high depth

AGGR_MAX_ATTEMPTS      = 4
AGGR_PRUNE_MULTIPLIER  = 0.01

# Pass 3 Config
NUM_BINS_THETA = 12
NUM_BINS_PHI   = 10
NUM_BINS_F     = 10
GLOBAL_LENGTH_MAX = 10.0 # Will be populated dynamically

# ===================================================================
# STEP 1: GROVE TO L-SYSTEM PARSING
# ===================================================================

def v3(p): return np.array([p["x"], p["y"], p["z"]], float)

def safe_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > EPS else v

def rot_from(v1, v2):
    v1 = safe_norm(v1); v2 = safe_norm(v2)
    dot = float(np.clip(np.dot(v1, v2), -1, 1))
    if dot > 1 - EPS: return np.eye(3)
    if dot < -1 + EPS:
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        axis = safe_norm(np.cross(v1, perp))
        x, y, z = axis; c, s, C = -1, 0, 2
        return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                         [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                         [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])
    cross = np.cross(v1, v2)
    k = np.array([[0,-cross[2],cross[1]],
                  [cross[2],0,-cross[0]],
                  [-cross[1],cross[0],0]])
    return np.eye(3) + k + k @ k * ((1 - dot) / (np.linalg.norm(cross)**2))

def euler(R):
    if abs(R[2,0]) < 1 - EPS:
        ry = math.degrees(math.asin(-R[2,0]))
        rx = math.degrees(math.atan2(R[2,1], R[2,2]))
        rz = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        ry = math.degrees(math.asin(-np.sign(R[2,0])))
        rx = math.degrees(math.atan2(-R[0,1], R[1,1]))
        rz = 0
    return rx, ry, rz

def flush_rot(acc, out):
    rx, ry, rz = acc
    if abs(rx) < ROT_TOL and abs(ry) < ROT_TOL and abs(rz) < ROT_TOL:
        acc[:] = [0, 0, 0]
        return
    out.append(f"R({round(rx,ANGLE_DEC)},{round(ry,ANGLE_DEC)},{round(rz,ANGLE_DEC)})")
    acc[:] = [0, 0, 0]

def flush_fwd(acc, out):
    if acc <= 0: return 0.0
    out.append(f"F({round(acc,LEN_DEC)})")
    return 0.0

def process(nodes, out, Rcur):
    acc_rot = [0, 0, 0]
    acc_fwd = 0.0
    prev = None

    for i in range(1, len(nodes)):
        p0 = v3(nodes[i-1]["pos"])
        p1 = v3(nodes[i]["pos"])
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        if L < MIN_SEG: continue

        seg_dir = safe_norm(seg)

        for br in nodes[i-1].get("side_branches", []):
            brn = br["nodes"]
            if len(brn) < 2: continue
            tot = sum(np.linalg.norm(v3(brn[j]["pos"])-v3(brn[j-1]["pos"])) for j in range(1, len(brn)))
            e2e = np.linalg.norm(v3(brn[-1]["pos"])-v3(brn[0]["pos"]))
            if tot < MIN_BRANCH or e2e < MIN_E2E: continue

            flush_rot(acc_rot, out)
            acc_fwd = flush_fwd(acc_fwd, out)
            out.append("[")

            br_dir = safe_norm(v3(brn[1]["pos"])-v3(brn[0]["pos"]))
            br_local = Rcur.T @ br_dir
            Rloc = rot_from(FWD, br_local)
            rx, ry, rz = euler(Rloc)
            acc_rot[0] += rx; acc_rot[1] += ry; acc_rot[2] += rz
            flush_rot(acc_rot, out)

            process(brn, out, Rcur @ Rloc)
            flush_rot(acc_rot, out)
            acc_fwd = flush_fwd(acc_fwd, out)
            out.append("]")

        local = Rcur.T @ seg_dir
        Rloc = rot_from(FWD, local)
        rx, ry, rz = euler(Rloc)

        dir_change = np.linalg.norm(local - (prev if prev is not None else local))

        if prev is None or dir_change > MERGE_DIR_TOL:
            acc_fwd = flush_fwd(acc_fwd, out)
            flush_rot(acc_rot, out)
            acc_rot[0] += rx; acc_rot[1] += ry; acc_rot[2] += rz
            flush_rot(acc_rot, out)
            Rcur = Rcur @ Rloc
            prev = local
        else:
            Rcur = Rcur @ Rloc
            prev = local

        acc_fwd += L

    flush_rot(acc_rot, out)
    acc_fwd = flush_fwd(acc_fwd, out)
    return Rcur

def generate_tokens(nodes):
    out = []
    first = safe_norm(v3(nodes[1]["pos"]) - v3(nodes[0]["pos"]))
    R0 = rot_from(FWD, first)
    rx, ry, rz = euler(R0)
    out.append(f"R({round(rx,1)},{round(ry,1)},{round(rz,1)})")
    process(nodes, out, R0)
    return out

# ===================================================================
# STEP 2: SIMPLIFICATION
# ===================================================================

def compose_rot(r1, r2):
    return (r1[0] + r2[0], r1[1] + r2[1], r1[2] + r2[2])

def enforce_no_consecutive_R(tokens):
    out = []
    pending_rot = None

    def is_zero(rot):
        return all(abs(x) < 1e-8 for x in rot)

    for t in tokens:
        if t.typ == "R":
            rot = tuple(float(x) for x in t.val)
            if is_zero(rot): continue
            if pending_rot is None:
                pending_rot = rot
            else:
                pending_rot = compose_rot(pending_rot, rot)
        else:
            if pending_rot is not None and not is_zero(pending_rot):
                out.append(Token("R", pending_rot))
            pending_rot = None
            out.append(t)

    if pending_rot is not None and not is_zero(pending_rot):
        out.append(Token("R", pending_rot))
    return out

RE_F = re.compile(r"F\(([+-]?\d+(\.\d+)?)\)")
RE_R = re.compile(r"R\(([+-]?\d+(\.\d+)?),([+-]?\d+(\.\d+)?),([+-]?\d+(\.\d+)?)\)")

class Token:
    __slots__ = ['typ', 'val']  # Drastically reduces memory and speeds up attribute access for millions of tokens
    def __init__(self, typ, val=None):
        self.typ = typ
        self.val = val

    def __repr__(self):
        return f"{self.typ}:{self.val}"

def parse_lstring(s: str) -> List[Token]:
    tokens = []
    i = 0
    while i < len(s):
        if s.startswith("A->", i):
            tokens.append(Token("A"))
            i += 3
            continue
        if s[i] == "[":
            tokens.append(Token("["))
            i += 1
            continue
        if s[i] == "]":
            tokens.append(Token("]"))
            i += 1
            continue

        mF = RE_F.match(s, i)
        if mF:
            tokens.append(Token("F", float(mF.group(1))))
            i = mF.end()
            continue

        mR = RE_R.match(s, i)
        if mR:
            tokens.append(Token("R", (float(mR.group(1)), float(mR.group(3)), float(mR.group(5)))))
            i = mR.end()
            continue
        i += 1
    return tokens

def merge_consecutive_F(tokens):
    out, acc = [], 0.0
    for t in tokens:
        if t.typ == "F":
            acc += t.val
        else:
            if acc > 0:
                out.append(Token("F", acc))
                acc = 0.0
            out.append(t)
    if acc > 0:
        out.append(Token("F", acc))
    return out

def collapse_RF_chains(tokens, min_chain, base_rot, k, delta):
    out = []
    i, N = 0, len(tokens)
    while i < N:
        if tokens[i].typ == "R" and i+1 < N and tokens[i+1].typ == "F":
            Rs, Fs = [], []
            j = i
            while j+1 < N and tokens[j].typ == "R" and tokens[j+1].typ == "F":
                Rs.append(tokens[j].val)
                Fs.append(tokens[j+1].val)
                j += 2
            if len(Rs) >= min_chain:
                total_len = sum(Fs)
                # Keep first rotation
                out.append(Token("R", Rs[0]))
                out.append(Token("F", total_len))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return out

def prune_small_branches(tokens, min_len):
    # First pass: compute direct F length and close-bracket index for each branch
    open_stack = []
    branch_len = {}  # open-bracket index -> direct F total
    close_of = {}    # open-bracket index -> close-bracket index
    for i, t in enumerate(tokens):
        if t.typ == "[":
            open_stack.append(i)
            branch_len[i] = 0.0
        elif t.typ == "F" and open_stack:
            branch_len[open_stack[-1]] += t.val
        elif t.typ == "]" and open_stack:
            close_of[open_stack.pop()] = i

    # Second pass: skip pruned branches in O(n)
    out = []
    skip_to = -1
    for i, t in enumerate(tokens):
        if i <= skip_to:
            continue
        if t.typ == "[" and i in close_of and branch_len[i] < min_len:
            skip_to = close_of[i]
            continue
        out.append(t)
    return out

def keep_max_depth_balanced(tokens, max_depth=3, keep_every=2):
    out = []
    depth = 0
    branch_counter = {}
    keep_stack = []
    current_height = 0.0
    height_stack = []

    for t in tokens:
        if t.typ == "F":
            if not keep_stack or keep_stack[-1]:
                current_height += t.val
                out.append(t)
        elif t.typ == "[":
            depth += 1
            height_stack.append(current_height)

            if depth <= max_depth:
                keep = True
            else:
                height_band = int(current_height)
                parent_depth = depth - 1
                key = (parent_depth, height_band)
                branch_counter.setdefault(key, 0)
                keep = (branch_counter[key] % keep_every == 0)
                branch_counter[key] += 1

            keep_stack.append(keep)
            if keep: out.append(t)
        elif t.typ == "]":
            if keep_stack and keep_stack[-1]: out.append(t)
            if keep_stack: keep_stack.pop()
            if height_stack: current_height = height_stack.pop()
            depth -= 1
        else:
            if not keep_stack or keep_stack[-1]: out.append(t)
    return out

def serialize(tokens):
    def is_zero_rot(rot):
        return all(abs(float(x)) < 1e-8 for x in rot)

    out = []
    for t in tokens:
        if t.typ == "A": out.append("A->")
        elif t.typ in "[]": out.append(t.typ)
        elif t.typ == "F": out.append(f"F({t.val:.2f})")
        elif t.typ == "R":
            a, b, c = (float(x) for x in t.val)
            if is_zero_rot((a, b, c)): continue
            out.append(f"R({a:.1f},{b:.1f},{c:.1f})")
    return "".join(out)

def simplify_until_limit(tokens, max_tokens=MAX_TOKENS_NUM, target_depth=3):
    min_chain = SIMPL_MIN_CHAIN_INIT
    base_rot = SIMPL_BASE_ROT_INIT
    k = SIMPL_K_INIT
    delta = SIMPL_DELTA_INIT
    prune_len = SIMPL_PRUNE_LEN_INIT
    last_len = len(tokens)
    target_keep_every = 2
    current_max_depth = max(1, target_depth - 1)
    consecutive_stalls = 0

    # Analyze the tree's actual scale to find the bounds automatically
    p10, p50, p90 = get_branch_stats(tokens)
    auto_soft_cap = p50 * 0.75
    auto_step_small = max(0.001, p10 / 2.0)
    auto_step_large = max(0.005, p50 / 5.0)

    # Start more aggressively when far over the token limit
    ratio = last_len / max(1, max_tokens)
    if ratio > 3:
        prune_len = auto_step_small * min(ratio - 1, 10)
        min_chain = 1

    for _ in range(SIMPL_MAX_ITERATIONS):
        tokens = merge_consecutive_F(tokens)
        tokens = collapse_RF_chains(tokens, min_chain, base_rot, k, delta)
        if prune_len > 0:
            tokens = prune_small_branches(tokens, prune_len)
        tokens = enforce_no_consecutive_R(tokens)
        cur_len = len(tokens)

        if cur_len <= max_tokens:
            break

        # If we didn't crunch at least 2% of the tree in this pass, we are stalling.
        # Escalate step size exponentially after repeated stalls.
        if cur_len >= last_len * 0.98:
            consecutive_stalls += 1
            step_mult = min(2 ** (consecutive_stalls // 2), 8)  # doubles every 2 stalls, capped at 8x
            base_rot *= SIMPL_ROT_MULTIPLIER
            k *= SIMPL_K_MULTIPLIER
            min_chain = max(1, min_chain - 1)

            if prune_len < auto_soft_cap:
                prune_len = min(prune_len + auto_step_small * step_mult, auto_soft_cap)
            else:
                tokens = keep_max_depth_balanced(tokens, max_depth=current_max_depth, keep_every=target_keep_every)
                target_keep_every += 1

                if target_keep_every > SIMPL_MAX_KEEP_EVERY:
                    if current_max_depth > 1:
                        current_max_depth -= 1
                        target_keep_every = 2
                    else:
                        prune_len += auto_step_large * step_mult

                    print("DEPTH REDUCED", target_keep_every, SIMPL_MAX_KEEP_EVERY)
        else:
            consecutive_stalls = 0

        last_len = cur_len
    return tokens

def aggressive_simplify(tokens, max_tokens=MAX_TOKENS_NUM, attempts=AGGR_MAX_ATTEMPTS, target_depth=3):
    cur = simplify_until_limit(tokens, max_tokens=max_tokens, target_depth=target_depth)
    if len(cur) <= max_tokens: return cur

    for attempt in range(attempts):
        prune_len = AGGR_PRUNE_MULTIPLIER * (attempt + 1)
        cur = prune_small_branches(cur, prune_len)
        cur = collapse_RF_chains(cur, min_chain=1, base_rot=0.0, k=1.0, delta=0.0)
        cur = merge_consecutive_F(cur)
        cur = enforce_no_consecutive_R(cur)
        if attempt >= attempts - 1:
            cur = [t for t in cur if t.typ != "R"]
        cur = simplify_until_limit(cur, max_tokens=max_tokens, target_depth=target_depth)
        if len(cur) <= max_tokens: break
    return cur

def get_branch_stats(tokens):
    """Scans the tree to find the mathematical percentiles of branch lengths."""
    lengths = []
    stack = []
    for t in tokens:
        if t.typ == "[":
            stack.append(0.0)
        elif t.typ == "F" and stack:
            stack[-1] += t.val
        elif t.typ == "]":
            lengths.append(stack.pop())
            
    if not lengths:
        return 0.01, 0.05, 0.10
        
    lengths = np.array(lengths)
    # Return 10th percentile (tiny branches), 50th (median), and 90th
    return float(np.percentile(lengths, 10)), float(np.percentile(lengths, 50)), float(np.percentile(lengths, 90))

# ===================================================================
# STEP 3: SYMBOLIC ENCODING
# ===================================================================

def rot_x_mat(a):
    a = np.radians(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y_mat(a):
    a = np.radians(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z_mat(a):
    a = np.radians(a)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def bin_theta(theta_deg):
    idx = int(theta_deg / 180.0 * NUM_BINS_THETA)
    return str(max(0, min(NUM_BINS_THETA-1, idx)))

def bin_phi(phi_deg):
    idx = int(phi_deg / 360.0 * NUM_BINS_PHI)
    return str(max(0, min(NUM_BINS_PHI-1, idx)))

def bin_length(x, local_global_len_max):
    x = max(0.0, min(float(x), local_global_len_max))
    idx = int(x / local_global_len_max * NUM_BINS_F)
    return str(max(0, min(NUM_BINS_F-1, idx)))

def convert_program(txt, local_global_len_max):
    re_token = re.compile(r"R\([^)]*\)|F\([^)]*\)|\[|\]")
    tokens = re_token.findall(txt)
    orientation = np.eye(3)
    stack = []
    output = []

    for token in tokens:
        if token.startswith("R("):
            rx, ry, rz = map(float, token[2:-1].split(","))
            R = rot_z_mat(rz) @ rot_y_mat(ry) @ rot_x_mat(rx)
            orientation = orientation @ R
        elif token.startswith("F("):
            length = float(token[2:-1])
            direction = orientation @ np.array([0,0,1])
            direction = direction / np.linalg.norm(direction)

            theta = np.degrees(np.arccos(direction[2]))
            phi = np.degrees(np.arctan2(direction[1], direction[0]))
            if phi < 0: phi += 360

            t_bin = bin_theta(theta)
            p_bin = bin_phi(phi)
            f_bin = bin_length(length, local_global_len_max)
            output.append(f"B{t_bin}_{p_bin}F{f_bin}")
        elif token == "[":
            stack.append(orientation.copy())
            output.append("[")
        elif token == "]":
            orientation = stack.pop()
            output.append("]")
    return "".join(output)

# ===================================================================
# MAIN PIPELINE 
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end L-System Pipeline")
    parser.add_argument("--in_dir", default="./", help="Directory with .grove files")
    parser.add_argument("--depth", type=int, choices=[1,2,3], default=3, help="Max depth to keep")
    parser.add_argument("--keep_every", type=int, default=1, help="Keep 1/N branches at max depth")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS_NUM, help="Simplification budget")
    parser.add_argument("--aggressive", action="store_true", help="Aggressively adhere to token limit")
    args = parser.parse_args()

    out_raw = os.path.join(args.in_dir, "results")
    out_simp = os.path.join(out_raw, "simplified")
    out_final = os.path.join(out_raw, "final")

    os.makedirs(out_raw, exist_ok=True)
    os.makedirs(out_simp, exist_ok=True)
    os.makedirs(out_final, exist_ok=True)

    grove_files = [f for f in os.listdir(args.in_dir) if f.endswith(".grove")]
    
    print("\n" + "="*50 + "\nSTEP 1: .grove -> Raw .txt L-system\n" + "="*50)
    for g_file in grove_files:
        path = os.path.join(args.in_dir, g_file)
        name = os.path.splitext(g_file)[0]
        out_path = os.path.join(out_raw, name + ".txt")
        print(f"Parsing {g_file}...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        nodes = data["trees"][0]["nodes"]
        tokens = generate_tokens(nodes)
        with open(out_path, "w", encoding="utf-8") as f_out:
            f_out.write("A->" + "".join(tokens))

    print("\n" + "="*50 + "\nSTEP 2: Adaptive Simplification\n" + "="*50)
    for raw_file in os.listdir(out_raw):
        if not raw_file.endswith(".txt"): continue
        inp = os.path.join(out_raw, raw_file)
        out = os.path.join(out_simp, raw_file)
        print(f"Simplifying {raw_file}...")
        with open(inp, 'r') as f:
            s = f.read().strip()
        tokens = parse_lstring(s)
        tokens = keep_max_depth_balanced(tokens, args.depth, keep_every=args.keep_every)
        if args.aggressive:
            tokens = aggressive_simplify(tokens, max_tokens=args.max_tokens, target_depth=args.depth)
        else:
            tokens = simplify_until_limit(tokens, max_tokens=args.max_tokens, target_depth=args.depth)
        print(f"  -> Tokens kept: {len(tokens)}")
        with open(out, "w") as f:
            f.write(serialize(tokens))

    print("\n" + "="*50 + "\nSTEP 3: Symbolic Encoding\n" + "="*50)
    max_len_found = 0.0
    for simp_file in os.listdir(out_simp):
        if not simp_file.endswith(".txt"): continue
        with open(os.path.join(out_simp, simp_file), "r") as f:
            data = f.read()
        for m in re.finditer(r"F\(([+-]?\d+(?:\.\d+)?)\)", data):
            max_len_found = max(max_len_found, float(m.group(1)))

    if max_len_found > 0:
        global GLOBAL_LENGTH_MAX
        GLOBAL_LENGTH_MAX = max_len_found
        print(f"Data-driven GLOBAL_LENGTH_MAX set to: {GLOBAL_LENGTH_MAX}")

    for simp_file in os.listdir(out_simp):
        if not simp_file.endswith(".txt"): continue
        inp = os.path.join(out_simp, simp_file)
        out = os.path.join(out_final, simp_file)
        with open(inp, "r") as f:
            data = f.read()
        symbolic = convert_program(data, GLOBAL_LENGTH_MAX)
        with open(out, "w") as f:
            f.write(symbolic)
        print(f"Encoded {simp_file} successfully.")

    print("\n✅ Pipeline completed saving to ./results/")

if __name__ == "__main__":
    main()