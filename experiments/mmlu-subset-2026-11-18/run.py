#!/usr/bin/env python3
"""A subset of MMLU, scored question by question, for the arms of the model-output records.

The plan is PREREG.md in this folder. Each variant is scored with llama.cpp's own tool in its
multiple-choice mode on the same questions: the tool picks its subset with a fixed seed. The
tool is built with the one-line patch of this folder, so that for each question it prints the
answer it chose beside its running accuracy.

This program works out the same subset from the questions file, so it knows the right answer of
each question. It then checks the two things the tool printed against each other: a question
counts as right by the running accuracy exactly when the answer chosen is the one the file
marks. If they ever disagree it stops.

A variant is "reference" (the model file as given), "file:PATH" (a file llama.cpp made), or the
name of an arm of the record model-output-qwen25-2026-11-17, which is built by that record's
program, written as 16-bit floats, scored and deleted.

Usage, with NumPy and torch, llama.cpp's gguf package on GGUF_PY and the patched tool:
  python3 experiments/mmlu-subset-2026-11-18/run.py --model qwen2.5-1.5b-instruct-fp16.gguf --data mmlu-test.bin \
      --tasks 2000 --work DIR --out experiments/mmlu-subset-2026-11-18/results.jsonl --bin llama-perplexity-mc \
      --imatrix qwen15.imatrix.gguf reference file:qwen15-Q4_K_M.gguf ZBG128:b560s16 hZBG128:b560s16
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_SHA256 = {
    "88bbb0b99f0ac75dc63b82017d301be2e6b1b922b6a6eb508df41e5ac4396b5c": "mmlu-test.bin",
    "470af3a74eccacfaf6f43b08aabf510f61e6c92fe20d17241ded934151e225fa": "mmlu-validation.bin",     # for tests of this program only
}

spec = importlib.util.spec_from_file_location("qwen_run", HERE.parent / "model-output-qwen25-2026-11-17" / "run.py")
Q = importlib.util.module_from_spec(spec)
sys.modules["qwen_run"] = Q
spec.loader.exec_module(Q)
Q.MODELS["cf3f0569a9d0b56d5d3b5b7944768c71bd2fef789f8b36b1b85a05d390850852"] = "SmolLM2-1.7B-Instruct-f16.gguf"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_task(blob, at):
    """One question of the file: (question, answers, index of the answer marked right)."""
    def text(at):
        (n,) = struct.unpack_from("<I", blob, at)
        return blob[at + 4: at + 4 + n].decode("utf-8", "replace"), at + 4 + n
    question, at = text(at)
    (n,) = struct.unpack_from("<I", blob, at)
    at += 4
    answers = []
    for _ in range(n):
        a, at = text(at)
        answers.append(a)
    labels = struct.unpack_from(f"<{n}i", blob, at)
    if sum(labels) != 1:
        raise SystemExit("FAIL: a question does not have exactly one answer marked right")
    return question, answers, labels.index(1)


def subset(path, tasks):
    """The questions the tool picks, in its order: (index in the file, index of the right answer, number of answers).

    The tool draws with the generator std::mt19937 seeded with 1 and 32-bit float arithmetic:
    j = (int)(scale * rng() * aux.size()), then the last index is moved into place j.
    """
    blob = open(path, "rb").read()
    (n,) = struct.unpack_from("<I", blob, 0)
    pos = struct.unpack_from(f"<{n}I", blob, 4)
    if tasks <= 0 or tasks >= n:
        raise SystemExit("FAIL: this program reads the tool's random subset only: 0 < tasks < the number in the file")
    gen = np.random.MT19937()
    gen._legacy_seeding(1)
    if int(gen.random_raw(1)[0]) != 1791095845:
        raise SystemExit("FAIL: NumPy's generator does not give std::mt19937's first value")
    gen._legacy_seeding(1)
    raw = gen.random_raw(tasks)
    scale = np.float32(1.0) / (np.float32(1.0) + np.float32(4294967295))
    aux = list(range(n))
    out = []
    for r in raw:
        j = int(np.float32(scale * np.float32(int(r))) * np.float32(len(aux)))
        idx = aux[j]
        aux[j] = aux[-1]
        aux.pop()
        _, answers, right = read_task(blob, pos[idx])
        out.append((idx, right, len(answers)))
    return n, out


def build(variant, path, workers):
    names = [t.name for t in Q.targets(Q.reader())]
    replace, infos = {}, {}
    with Pool(workers) as pool:
        for name, Wh, info in pool.imap_unordered(Q.round_one, [(variant, n) for n in names], chunksize=1):
            replace[name], infos[name] = Wh, info
    Q.ggufedit.write_copy(Q.reader(), path, replace)
    return Q.stats_of(infos)


def score(binary, path, data, tasks, threads, picked):
    """For each question of the subset, in the tool's order: the answer chosen and whether it is right."""
    cmd = [binary, "-m", path, "-bf", data, "--multiple-choice", "--multiple-choice-tasks", str(tasks), "-t", str(threads)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    text = done.stdout + "\n" + done.stderr
    lines = [(int(a), float(b), int(c)) for a, b, c in re.findall(r"^(\d+)\t(\d+\.\d+)\t(\d+)\s*$", text, re.M)]
    if done.returncode != 0 or len(lines) != tasks:
        raise SystemExit(f"FAIL: the tool gave {len(lines)} results for {path}, not {tasks}\n{text[-2000:]}")
    chosen, right, before = [], [], 0
    for (n, acc, choice), (_, answer, count) in zip(lines, picked):
        now = round(acc * n / 100.0)
        if n != len(right) + 1 or now - before not in (0, 1) or not 0 <= choice < count:
            raise SystemExit(f"FAIL: cannot read question {n} from the tool's output")
        if (now - before == 1) != (choice == answer):
            raise SystemExit(f"FAIL: question {n}: the running accuracy and the answer chosen disagree. The subset worked out here is not the tool's.")
        chosen.append(choice)
        right.append(now - before)
        before = now
    final = re.search(r"Final result: ([\d.]+) \+/- ([\d.]+)", text)
    return chosen, right, (float(final.group(1)) if final else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tasks", type=int, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", default="llama-perplexity-mc")
    ap.add_argument("--imatrix", help="llama.cpp's importance matrix file, for the arms given one")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    data_digest = sha256_of(args.data)
    if data_digest not in DATA_SHA256:
        raise SystemExit("FAIL: the questions file is not one the plan names")
    digest = sha256_of(args.model)
    if digest not in Q.MODELS:
        raise SystemExit("FAIL: the model file is not one the plan names")
    Q._model_path, Q._imatrix_path = args.model, args.imatrix
    in_file, picked = subset(args.data, args.tasks)
    subset_digest = hashlib.sha256(",".join(str(i) for i, _, _ in picked).encode()).hexdigest()
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r["model"], r["variant"], r["tasks"], r["data"]) for r in map(json.loads, filter(str.strip, f))}
    os.makedirs(args.work, exist_ok=True)
    tool_digest = sha256_of(args.bin)
    for v in args.variants:
        label = "file:" + os.path.basename(v[5:]) if v.startswith("file:") else v
        if (Q.MODELS[digest], label, args.tasks, DATA_SHA256[data_digest]) in done:
            print(f"{label}: already in the results file", file=sys.stderr, flush=True)
            continue
        t0 = time.time()
        row = {"variant": label, "model": Q.MODELS[digest], "data": DATA_SHA256[data_digest], "questions_in_file": in_file, "tasks": args.tasks,
               "subset_sha256": subset_digest, "tool_sha256": tool_digest}
        if v == "reference":
            path, made = args.model, False
        elif v.startswith("file:"):
            path, made = v[5:], False
            row["file_sha256"] = sha256_of(path)
        else:
            if not (Q.GROUP_BALL.match(v) or Q.WEIGHTED_BALL.match(v) or Q.VARIANT.match(v) or Q.ZERO_CENTRE.match(v)):
                raise SystemExit(f"unknown variant {v}")
            if Q.WEIGHTED_BALL.match(v) and not args.imatrix:
                raise SystemExit(f"{v} needs --imatrix")
            path, made = os.path.join(args.work, "mc_" + re.sub(r"[^A-Za-z0-9_.+-]", "_", v) + ".gguf"), True
            stats = build(v, path, args.workers)
            row.update({k: stats[k] for k in ("matrices", "bits_per_weight", "weight_err_db_mean")})
        t1 = time.time()
        chosen, right, final = score(args.bin, path, args.data, args.tasks, args.threads, picked)
        if made:
            os.remove(path)
        row.update({"right": sum(right), "accuracy": 100.0 * sum(right) / len(right), "tool_final": final,
                    "build_seconds": round(t1 - t0, 1), "score_seconds": round(time.time() - t1, 1),
                    "chosen": "".join(map(str, chosen)), "answer": "".join(str(a) for _, a, _ in picked)})
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in row if k not in ("chosen", "answer")}), flush=True)


if __name__ == "__main__":
    main()
