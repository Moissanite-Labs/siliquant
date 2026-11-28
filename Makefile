# SPDX-License-Identifier: Apache-2.0
PYTHON ?= python3

.PHONY: help test test-3mode clean

help:
	@echo "SiliQuant V2 targets:"
	@echo "  make test       Run pure-CPU V2 codec, gauge, Hadamard, Pareto BPW, and wire tests"
	@echo "  make test-3mode Run 3x -I isolated parity across -B, -O -B, and -B -W error"
	@echo "  make clean      Remove build and Python cache artifacts"

test:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_siliquant_v2_cpu.py

test-3mode:
	@set -eu; \
	out_b="$$(mktemp)"; out_ob="$$(mktemp)"; out_we="$$(mktemp)"; \
	trap 'rm -f "$${out_b:?}" "$${out_ob:?}" "$${out_we:?}"' EXIT; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B tests/test_siliquant_v2_cpu.py > "$$out_b"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -O -B tests/test_siliquant_v2_cpu.py > "$$out_ob"; \
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -W error tests/test_siliquant_v2_cpu.py > "$$out_we"; \
	cmp -s "$$out_b" "$$out_ob"; \
	cmp -s "$$out_b" "$$out_we"; \
	cat "$$out_b"; \
	echo "R4_3X_ISOLATED_PARITY_PASS=1"

clean:
	CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -c 'import pathlib, shutil; r = pathlib.Path("."); [shutil.rmtree(p, ignore_errors=True) for p in list(r.glob("*.egg-info")) + [r / "build", r / "dist", r / ".pytest_cache"] if p.exists()]; [shutil.rmtree(p, ignore_errors=True) for p in r.rglob("__pycache__") if ".git" not in p.parts]; [p.unlink(missing_ok=True) for ext in ("*.pyc", "*.pyo") for p in r.rglob(ext) if ".git" not in p.parts]'
