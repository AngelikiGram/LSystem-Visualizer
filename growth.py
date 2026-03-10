import re
import os
import numpy as np
import random

# ============================================================
# SETTINGS
# ============================================================
INPUT_FOLDER = "./results/final"
# INPUT_FOLDER = r"C:\Users\angie\Documents\treesformer\results\inference"
OUTPUT_FOLDER = "./results/growth/dense" # SMALL"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


enrich = True  # If True, enriches the last existing level with similar branches
enrich_scale = 20  # Number of new branches to add per terminal at last depth (set higher for more enrichment)

test_dataset = False # True
if test_dataset:
    ORTHO_IMAGE_PATH = r"E:\TREES_DATASET_P3_LSTRING\TREES_DATASET\ORTHOPHOTOS\tree_0013\rendering\view_003.png"
    INPUT_FOLDER = r"C:\Users\angie\Documents\treesformer\results\test" # network_architecture
    OUTPUT_FOLDER = "./GROWTH_INFERENCE_TEST" # SMALL"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Interaction Variables
TARGET_LEVEL = 2      # Depth: 0=Trunk, 1=L1, 2=L2, 3=L3
DENSIFY_SCALE = 5     # Number of additional L2 branches to spawn
HEAL_ENABLED = False   # MASTER TOGGLE: If False, Densify Scale is ignored for L2

# ============================================================
# HIERARCHICAL ENGINE (Flask-Logic Aligned)
# ============================================================

MAX_F_BINS = 5
def process_recursive1(token_list, current_depth):
    output = []
    i = 0

    while i < len(token_list):
        t = token_list[i]

        if t == '[':
            # Extract block
            block, balance, j = [], 1, i + 1
            while j < len(token_list) and balance > 0:
                if token_list[j] == '[': balance += 1
                elif token_list[j] == ']': balance -= 1
                if balance > 0:
                    block.append(token_list[j])
                j += 1

            output.append('[')
            output.extend(process_recursive(block, current_depth + 1))
            output.append(']')
            i = j - 1

        elif 'B' in t:
            output.append(t)

            # Check if terminal (no following bracket)
            next_is_bracket = (i + 1 < len(token_list) and token_list[i + 1] == '[')

            if not next_is_bracket:
                # Force-grow until TARGET_LEVEL
                depth_needed = TARGET_LEVEL - current_depth - 1
                parent_depth = current_depth

                for d in range(depth_needed):
                    output.append('[')

                    t_r = random.randint(0, 4)
                    p_r = random.randint(0, 4)

                    # shrink length with depth
                    f_r = max(0, MAX_F_BINS - (current_depth + d))

                    output.append(f"B{t_r}_{p_r}F{f_r}")

                for d in range(depth_needed):
                    output.append(']')

        i += 1

    return output

def expand_tree_hierarchical(raw_string, target_level, densify_scale, heal_enabled):
    """
    Recursive expansion logic:
    - If HEAL is OFF: densify_scale is ignored for Level 2 spawning.
    - If HEAL is ON: densify_scale spawns N branches on every Level 1 segment.
    """
    if target_level == 1:
        # Remove all branches at level 2 and deeper
        tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", raw_string)
        output = []
        depth = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == '[':
                depth += 1
                if depth == 2:
                    # Skip everything until matching ]
                    balance = 1
                    i += 1
                    while i < len(tokens) and balance > 0:
                        if tokens[i] == '[':
                            balance += 1
                        elif tokens[i] == ']':
                            balance -= 1
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

    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", raw_string)
    import math
    def compute_max_depth(s):
        depth = 0
        max_depth = 0
        for c in s:
            if c == '[':
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ']':
                depth -= 1
        return max_depth
    
    def process_recursive(token_list, current_depth, prev_level_branches=1):
        # Base case: only stop if we are strictly BEYOND the target level
        if current_depth > target_level:
            return token_list
            
        output = []
        i = 0
        
        while i < len(token_list):
            t = token_list[i]
            
            if t == '[':
                # 1. EXTRACT NESTED BLOCK
                block, balance, j = [], 1, i + 1
                while j < len(token_list) and balance > 0:
                    if token_list[j] == '[': balance += 1
                    elif token_list[j] == ']': balance -= 1
                    if balance > 0: block.append(token_list[j])
                    j += 1
                
                # 2. RECURSE
                # We pass the depth incremented. 
                # Branch scaling: 2.5 is good for density, but we cap it to prevent string explosion
                next_scaling = min(prev_level_branches * 2.1, 15) 
                processed_block = process_recursive(block, current_depth + 1, next_scaling)
                
                output.append('[')
                output.extend(processed_block)
                output.append(']')
                i = j - 1
                
            elif 'B' in t:
                output.append(t)
                
                # 3. TERMINAL GROWTH (The "Heal" and "Fill" Logic)
                # If a branch segment exists but has no children, and we haven't reached Target Level:
                next_is_bracket = (i + 1 < len(token_list) and token_list[i + 1] == '[')
                
                if not next_is_bracket and current_depth < target_level:
                    # Determine how many levels of growth we need to add to reach Level 5
                    levels_to_grow = target_level - current_depth
                    
                    def generate_forced_growth(d_remaining, branching_factor):
                        if d_remaining <= 0: return []
                        
                        growth = []
                        # Spawn multiple sub-branches based on depth
                        num_sub_branches = int(math.ceil(branching_factor))
                        
                        for _ in range(num_sub_branches):
                            growth.append('[')
                            t_r, p_r = random.randint(0,4), random.randint(0,4)
                            # Length F decreases as we get closer to the tips
                            f_val = max(0, 2 - (target_level - d_remaining)) 
                            growth.append(f"B{t_r}_{p_r}F{f_val}")
                            # Recursive sub-growth
                            growth.extend(generate_forced_growth(d_remaining - 1, 1.8))
                            growth.append(']')
                        return growth

                    # Apply the forced growth to the terminal segment
                    output.extend(generate_forced_growth(levels_to_grow, 2.0))
                    
            i += 1
        return output

    result_tokens = process_recursive(tokens, 0 , 1)
    return "".join(result_tokens)

def enrich_last_level(lstring, scale=2):
    """
    Enriches the last (deepest) level of the tree by adding 'scale' new branches per parent branch at the previous depth.
    New subbranches are spread across the parent's block, with similar properties but varied phi angle.
    """
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", lstring)
    # Enrich all branches at depth 1 (L1) by adding 'scale' new branches to each
    enriched = []
    depth = 0
    i = 0
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
            # If this is a branch at depth 1, enrich it
            if depth == 1:
                m = re.match(r"B(\d+)_(\d+)F(\d+)", t)
                if m:
                    t_bin, p_bin, parent_f_bin = map(int, m.groups())
                    print(f"Enriching L1 parent {t} at depth {depth} with {scale} new branches")
                    # Find the block of children for this L1 branch
                    children = []
                    child_blocks = []
                    j = i + 1
                    while j < len(tokens) and tokens[j] == '[':
                        # Find the matching closing bracket for this child
                        block, balance, k = [], 1, j + 1
                        block.append(tokens[j])
                        while k < len(tokens) and balance > 0:
                            block.append(tokens[k])
                            if tokens[k] == '[':
                                balance += 1
                            elif tokens[k] == ']':
                                balance -= 1
                            k += 1
                        child_blocks.append(block)
                        j = k
                    # Generate new subbranches, each with a forward movement to spread them
                    new_blocks = []
                    for k in range(scale):
                        new_p = random.randint(0, 4)
                        t_r = min(max(t_bin + random.choice([-1, 0, 1]), 0), 4)
                        f_r = max(parent_f_bin + random.choice([-1, 0, 1]), 0)
                        # Spread F value along the parent branch
                        spread_f = max(1, int((k + 1) * MAX_F_BINS / (scale + 1)))
                        new_blocks.append([f"F{spread_f}", f"[B{t_r}_{new_p}F{f_r}]"])
                    # Interleave new_blocks with child_blocks at random positions
                    total_blocks = child_blocks + new_blocks
                    random.shuffle(total_blocks)
                    for block in total_blocks:
                        enriched.extend(block)
                    # Skip over the original children in the main loop
                    i = j - 1
        i += 1
    return "".join(enriched)

# ============================================================
# BATCH EXECUTION
# ============================================================

def run_batch():
    if not os.path.exists(OUTPUT_FOLDER): 
        os.makedirs(OUTPUT_FOLDER)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: {INPUT_FOLDER} does not exist.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"Processing {len(files)} files...")
    print(f"Mode: {'HEAL+DENSIFY' if HEAL_ENABLED else 'PRUNE/ORIGINAL ONLY'}")

    for filename in files:
        with open(os.path.join(INPUT_FOLDER, filename), "r") as f:
            raw_str = f.read().strip()
        
        processed_str = expand_tree_hierarchical(
            raw_str, 
            TARGET_LEVEL, 
            DENSIFY_SCALE, 
            HEAL_ENABLED
        )
        if enrich:
            print(f"\n--- {filename} ---")
            print("Initial tokens:", len(processed_str), "characters")
            processed_str = enrich_last_level(processed_str, scale=enrich_scale)
            print("Final tokens after enrichment:", len(processed_str), "characters")
        
        with open(os.path.join(OUTPUT_FOLDER, filename), "w") as f:
            f.write(processed_str)
        
    print(f"Done. Outputs saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_batch()