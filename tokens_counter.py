import os
import numpy as np
import re


# ---------------------------------------------------------
# Token Types
# ---------------------------------------------------------

class TokenType:
    F   = 0
    LBR = 1
    RBR = 2
    EOS = 3


# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------

class LSystemTokenizerV2:
    """
    Tokenizer for compressed symbolic L-strings:

        B{theta}_{phi}F{length}
        [
        ]

    Produces:
        type_ids:  list[int]
        value_ids: list[[len, theta, phi]]
    """

    def __init__(self, f_bins=10, theta_bins=6, phi_bins=6):

        self.f_bins = f_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins

        # Strict regex for your dataset format
        self.re_SEGMENT = re.compile(r"B(\d+)_(\d+)F(\d+)")

    # ------------------------------------------------------
    # Encoding
    # ------------------------------------------------------

    def encode(self, s):

        types = []
        values = []

        i = 0
        L = len(s)

        while i < L:

            c = s[i]

            # Branch start
            if c == "[":
                types.append(TokenType.LBR)
                values.append([0, 0, 0])
                i += 1
                continue

            # Branch end
            if c == "]":
                types.append(TokenType.RBR)
                values.append([0, 0, 0])
                i += 1
                continue

            # Segment
            m = self.re_SEGMENT.match(s, i)

            if m:
                theta = int(m.group(1))
                phi   = int(m.group(2))
                f_bin = int(m.group(3))

                types.append(TokenType.F)
                values.append([f_bin, theta, phi])

                i = m.end()
                continue

            # Skip unknown char
            i += 1

        # Append EOS
        types.append(TokenType.EOS)
        values.append([0, 0, 0])

        return types, values


    # ------------------------------------------------------
    # Decoding
    # ------------------------------------------------------

    def decode(self, types, values):

        out = []

        for t, v in zip(types, values):

            if t == TokenType.F:
                f_bin, theta, phi = v
                out.append(f"B{theta}_{phi}F{f_bin}")

            elif t == TokenType.LBR:
                out.append("[")

            elif t == TokenType.RBR:
                out.append("]")

            elif t == TokenType.EOS:
                break

        return "".join(out)


    # ------------------------------------------------------
    # Count tokens
    # ------------------------------------------------------

    def count_tokens(self, s, exclude_eos=True):

        types, _ = self.encode(s)

        if exclude_eos and types[-1] == TokenType.EOS:
            return len(types) - 1

        return len(types)


    # ------------------------------------------------------
    # Truncate sequence
    # ------------------------------------------------------

    def truncate_to_max_tokens(self, s, max_tokens):

        types, values = self.encode(s)

        # Remove EOS for truncation
        if types[-1] == TokenType.EOS:
            types = types[:-1]
            values = values[:-1]

        if len(types) > max_tokens:
            types = types[:max_tokens]
            values = values[:max_tokens]

        # Re-add EOS
        types.append(TokenType.EOS)
        values.append([0, 0, 0])

        return types, values


# ---------------------------------------------------------
# Token statistics
# ---------------------------------------------------------

def analyze_dataset(folder, tokenizer):

    txts = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    if len(txts) == 0:
        print("No .txt files found.")
        return

    print(f"Found {len(txts)} L-strings\n")

    lengths = []

    for fname in txts:

        path = os.path.join(folder, fname)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        n = tokenizer.count_tokens(content)

        lengths.append(n)

        print(f"{fname:30s} → {n} tokens")

    lengths = np.array(lengths)

    print("\n============================")
    print("TOKEN STATISTICS")
    print("============================")

    print(f"Min tokens      : {lengths.min()}")
    print(f"Max tokens      : {lengths.max()}")
    print(f"Mean tokens     : {lengths.mean():.1f}")
    print(f"Median tokens   : {np.median(lengths):.1f}")
    print(f"90th percentile : {np.percentile(lengths, 90):.1f}")
    print(f"95th percentile : {np.percentile(lengths, 95):.1f}")
    print(f"99th percentile : {np.percentile(lengths, 99):.1f}")

    print("\n============================")
    print("RECOMMENDED WINDOW")
    print("============================")

    recommended = int(np.percentile(lengths, 85))

    print(f"Safe window size (99% coverage): {recommended}")

    if recommended > 2048:
        print("⚠ Trees are very long. Consider truncated BPTT.")
    elif recommended > 1024:
        print("⚠ Consider window=1024 with overlap.")
    else:
        print("✓ window=1024 is safe.")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    FOLDER = r"./results/final"
    FOLDER = r"E:\TREES_DATASET_P3_LSTRING\LSTRINGS_RESULTS\LSTRINGS_FINAL_SMALL"

    from pipeline import GLOBAL_LENGTH_MAX, NUM_BINS_F, NUM_BINS_THETA, NUM_BINS_PHI

    tokenizer = LSystemTokenizerV2(
        f_bins=NUM_BINS_F,
        theta_bins=NUM_BINS_THETA,
        phi_bins=NUM_BINS_PHI
    )

    analyze_dataset(FOLDER, tokenizer)


if __name__ == "__main__":
    main()