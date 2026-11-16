#!/usr/bin/env python3
"""Build the evaluation text from a public repository at a fixed commit.

The text alternates slices of prose (README and docs) and of source code from
llama.cpp, read with `git show` so that the result does not depend on the
state of the checkout. Nothing is downloaded.

Usage:
  python3 harness/build_eval_text.py /path/to/llama.cpp OUT.txt
"""
from __future__ import annotations

import hashlib
import subprocess
import sys

COMMIT = "427291b"
SLICE = 6000        # characters per slice
TOTAL = 260_000     # characters in all
CODE = ["src/llama-sampler.cpp", "common/common.cpp", "gguf-py/gguf/gguf_writer.py",
        "tools/server/server-context.cpp", "convert_hf_to_gguf.py", "src/llama-vocab.cpp"]


def show(repo, path):
    return subprocess.run(["git", "-C", repo, "show", f"{COMMIT}:{path}"], check=True,
                          capture_output=True).stdout.decode("utf-8", errors="replace")


def main():
    repo, out = sys.argv[1], sys.argv[2]
    names = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", COMMIT], check=True,
                           capture_output=True, text=True).stdout.split("\n")
    prose_files = ["README.md"] + sorted(n for n in names if n.startswith("docs/") and n.endswith(".md"))
    code_files = [n for n in CODE if n in names]
    prose = "\n\n".join(show(repo, n) for n in prose_files)
    code = "\n\n".join(show(repo, n) for n in code_files)
    parts, i = [], 0
    while sum(len(p) for p in parts) < TOTAL:
        a, b = prose[i * SLICE:(i + 1) * SLICE], code[i * SLICE:(i + 1) * SLICE]
        if not a or not b:
            break
        parts += [a, b]
        i += 1
    text = "\n\n".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"commit {COMMIT}, {len(prose_files)} prose files, code files {code_files}")
    print(f"{len(parts)} slices, {len(text)} characters, sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
