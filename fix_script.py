import re
with open(r'c:\Users\angie\Documents\visualizer_Lsystem\lsystem_viz_addon\__init__.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_func = '''def _expand_tree_to_level(raw_string, target_level, enrich_scale=0):
    \"\"\"Prune or grow the L-string to target_level depth.\"\"\"
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", raw_string)

    if target_level <= 1:
        skip_at = target_level + 1
        output, depth, i = [], 0, 0
        while i < len(tokens):
            if tokens[i] == '[':
                depth += 1
                if depth == skip_at:
                    bal = 1
                    i += 1
                    while i < len(tokens) and bal > 0:
                        if tokens[i] == '[': bal += 1
                        elif tokens[i] == ']': bal -= 1
                        i += 1
                    depth -= 1
                    continue
                else:
                    output.append('[')
            elif tokens[i] == ']':
                if depth > 0:
                    depth -= 1
                output.append(']')
            else:
                if depth < skip_at:
                    output.append(tokens[i])
            i += 1
        return "".join(output)

    def parse_branch(pos):
        branch = []
        while pos < len(tokens):
            if tokens[pos] == '[':
                child, pos = parse_branch(pos+1)
                branch.append(child)
            elif tokens[pos] == ']':
                return branch, pos+1
            else:
                branch.append(tokens[pos])
                pos += 1
        return branch, pos

    tree, _ = parse_branch(0)

    def generate_random_branch(d_remaining, parent_total_f):
        if d_remaining <= 0: return []
        import random
        child_f = max(1, (parent_total_f + 1) // 2)
        n = parent_total_f + enrich_scale
        if n < 1: n = 1

        t_base = random.randint(0, 11)
        p_base = random.randint(0, 9)
        base_f = parent_total_f // n
        rem_f = parent_total_f % n

        child = []
        for seg in range(n):
            seg_f = base_f + (1 if seg < rem_f else 0)
            child.append(f"B{t_base}_{p_base}F{seg_f}")
            if d_remaining - 1 > 0:
                f_r = max(1, child_f + random.choice([-1, 0, 0]))
                sub = generate_random_branch(d_remaining - 1, f_r)
                if sub:
                    child.append(sub)
        return child

    def process_node(node, depth):
        if depth >= target_level:
            return node

        total_f = 0
        terminal = True
        for item in node:
            if isinstance(item, list):
                terminal = False
            else:
                m = re.match(r"B\d+_\d+F(\d+)", item)
                if m:
                    total_f += int(m.group(1))

        import random
        child_f = max(1, (total_f + 1) // 2)

        result = []
        for i, item in enumerate(node):
            if isinstance(item, list):
                result.append(process_node(item, depth + 1))
            else:
                m = re.match(r"B(\d+)_(\d+)F(\d+)", item)
                if m:
                    t_bin, p_bin, f_bin = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    next_is_bracket = (i + 1 < len(node) and isinstance(node[i+1], list))
                    
                    will_expand = False
                    if terminal and depth < target_level:
                        will_expand = True
                    elif not next_is_bracket and depth < target_level:
                        will_expand = True

                    if will_expand:
                        n = f_bin + enrich_scale
                        if n < 1: n = 1
                        base_f = f_bin // n
                        rem_f = f_bin % n

                        for seg in range(n):
                            seg_f = base_f + (1 if seg < rem_f else 0)
                            result.append(f"B{t_bin}_{p_bin}F{seg_f}")
                            
                            f_r = max(1, child_f + random.choice([-1, 0, 0]))
                            new_branch = generate_random_branch(target_level - depth, f_r)
                            if new_branch:
                                result.append(new_branch)
                    else:
                        result.append(item)
                else:
                    result.append(item)

        return result

    processed_tree = process_node(tree, 0)

    def serialize(node):
        out = []
        for item in node:
            if isinstance(item, list):
                out.append('[')
                out.append(serialize(item))
                out.append(']')
            else:
                out.append(item)
        return "".join(out)

    return serialize(processed_tree)

def _enrich_at_depth(lstring, enrich_depth, scale=5):
    tokens = re.findall(r"B\d+_\d+F\d+|\[|\]", lstring)

    def _collect_children(toks, start):
        children, j = [], start
        while j < len(toks) and toks[j] == '[':
            inner, balance, k = [], 1, j + 1
            while k < len(toks) and balance > 0:
                if toks[k] == '[':   balance += 1
                elif toks[k] == ']': balance -= 1
                if balance > 0:
                    inner.append(toks[k])
                k += 1
            children.append(inner)
            j = k
        return children, j

    def _process(toks, depth):
        out = []
        i = 0
        import random
        while i < len(toks):
            t = toks[i]
            if t == '[':
                inner, balance, j = [], 1, i + 1
                while j < len(toks) and balance > 0:
                    if toks[j] == '[':   balance += 1
                    elif toks[j] == ']': balance -= 1
                    if balance > 0:
                        inner.append(toks[j])
                    j += 1
                out.append('[')
                out.extend(_process(inner, depth + 1))
                out.append(']')
                i = j
            elif t.startswith('B'):
                out.append(t)
                if depth == enrich_depth:
                    children_inner, j = _collect_children(toks, i + 1)
                    processed = [_process(ci, depth + 1) for ci in children_inner]

                    child_f, child_t = [], []
                    for ci in children_inner:
                        for tok in ci:
                            cm = re.match(r"B(\d+)_\d+F(\d+)", tok)
                            if cm:
                                child_t.append(int(cm.group(1)))
                                child_f.append(int(cm.group(2)))
                                break

                    m = re.match(r"B(\d+)_\d+F(\d+)", t)
                    ref_f = child_f if child_f else ([int(m.group(2))] if m else [2])
                    ref_t = child_t if child_t else ([int(m.group(1))] if m else [2])

                    new_branches = []
                    for _ in range(scale):
                        t_r = min(max(random.choice(ref_t) + random.choice([-1, 0, 0]), 0), 11)
                        p_r = random.randint(0, 9)
                        f_r = max(random.choice(ref_f) + random.choice([-1, 0, 0]), 0)
                        new_branches.append([f"B{t_r}_{p_r}F{f_r}"])

                    all_children = processed + new_branches
                    random.shuffle(all_children)
                    for child_toks in all_children:
                        out.append('[')
                        out.extend(child_toks)
                        out.append(']')
                    i = j
                else:
                    i += 1
            else:
                out.append(t)
                i += 1
        return out

    return "".join(_process(tokens, 0))
'''

pattern = r'\"\"\"\nReplacement for _expand_tree_to_level.*?(?=\n_generating = False)'

new_text, count = re.subn(pattern, new_func, text, flags=re.DOTALL)
if count > 0:
    with open(r'c:\Users\angie\Documents\visualizer_Lsystem\lsystem_viz_addon\__init__.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
    print("Success")
else:
    print("Failed")