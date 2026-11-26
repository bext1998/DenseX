#!/usr/bin/env python3
"""DenseX env-default snapshot check.

Parses the two documented read points and fails when a DENSEX_* switch is added,
removed, or gets a different value when the environment is unset:

  ggml/src/ggml-cuda/ggml-cuda.cu   (CUDA backend)
  src/llama-densex-env.cpp          (llama layer)

The check is deliberately narrow: it understands the current expression shapes and
reports "cannot parse" instead of guessing.  A legitimate default change updates
scripts/densex_env_defaults.json in the same commit, so the diff is visible.

No GPU, no build: pure source parsing.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_REL = "ggml/src/ggml-cuda/ggml-cuda.cu"
LLAMA_REL = "src/llama-densex-env.cpp"
SNAPSHOT = ROOT / "scripts/densex_env_defaults.json"

# field name in the struct -> source file (relative to the repo root)
SITES = {
    "DENSEX_FA_DECODE_MMA": ("ggml/src/ggml-cuda/ggml-cuda.cu", "fa_decode_mma"),
    "DENSEX_FA_DECODE_MMA_ADA": ("ggml/src/ggml-cuda/ggml-cuda.cu", "fa_decode_mma_ada"),
    "DENSEX_MMQ_SLOTS": ("ggml/src/ggml-cuda/ggml-cuda.cu", "mmq_slots"),
    "DENSEX_MMVQ_MAX": ("ggml/src/ggml-cuda/ggml-cuda.cu", "mmvq_max"),
    "DENSEX_GRAPH_VERIFY": ("ggml/src/ggml-cuda/ggml-cuda.cu", "graph_verify"),
    "DENSEX_GDN_DEFERRED": ("src/llama-densex-env.cpp", "gdn_deferred"),
    "DENSEX_DRAFT_VOCAB_ORDER": ("src/llama-densex-env.cpp", "draft_vocab_order"),
    "DENSEX_DRAFT_VOCAB": ("src/llama-densex-env.cpp", "draft_vocab"),
    "DENSEX_KV_VMM": ("src/llama-densex-env.cpp", "kv_vmm"),
    "DENSEX_KV_LAZY": ("src/llama-densex-env.cpp", "kv_lazy"),
    "DENSEX_KV_LAZY_CELLS": ("src/llama-densex-env.cpp", "kv_lazy_cells"),
    "DENSEX_LOGITS_ARGMAX": ("tools/server/server-context.cpp", "__return__"),
}
SCAN_DIRS = ("src", "ggml", "common", "tools")
SCAN_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def _line(text, pat):
    for line in text.splitlines():
        if re.search(pat, line):
            return line.strip()
    return None


def _default(expr):
    e = " ".join(expr.split())
    if "== nullptr || atoi(" in e:
        return True
    if "!= nullptr && atoi(" in e and "? 1 : 0" in e:
        return False
    if "!= nullptr && atoi(" in e:
        return False
    if "!= nullptr ? atoi(" in e and ": -1" in e:
        return -1
    if "== nullptr ? -1" in e:
        return -1
    if "!= nullptr ? std::string(" in e and ": std::string()" in e:
        return ""
    if "!= nullptr ? (int64_t) atoll(" in e and ": (int64_t) 0" in e:
        return 0
    if "!= nullptr ? (int64_t) atoll(" in e:
        m = re.search(r": \(int64_t\) (\d+)$", e)
        if m:
            return int(m.group(1))
    raise ValueError(f"cannot parse default expression: {expr!r}")


def read_defaults(root=ROOT):
    out = {}
    for name, (rel, field) in SITES.items():
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if f'getenv("{name}")' not in text:
            raise ValueError(f"{name}: getenv not found in {rel}")
        if field == "__return__":
            m = re.search(rf'getenv\("{name}"\)[\s\S]{{0,400}}?^\s*return\s+(.+?);', text, re.MULTILINE)
            if not m:
                raise ValueError(f"{name}: return expression not found in {rel}")
            expr = m.group(1).strip()
        else:
            line = _line(text, rf"\be\.{re.escape(field)}\s*=")
            if line is None:
                raise ValueError(f"{name}: assignment e.{field} not found in {rel}")
            expr = line.split("=", 1)[1].strip().rstrip(";")
        out[name] = {
            "default": _default(expr),
            "field": field,
            "file": rel,
            "expr": " ".join(expr.split()),
        }
    # every runtime DENSEX_* getenv in src/ggml/common/tools must be registered;
    # tests/ are app/test-only and intentionally excluded.
    found = set()
    for dirname in SCAN_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            rel_parts = path.relative_to(root).parts
            if any(part in ("tests", "test", "build", ".git") for part in rel_parts):
                continue
            found |= set(re.findall(r'getenv\("(DENSEX_[A-Z0-9_]+)"\)', path.read_text(encoding="utf-8", errors="replace")))
    if found != set(SITES):
        raise ValueError(f"switch set changed: missing={sorted(set(SITES) - found)} new={sorted(found - set(SITES))}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--check", action="store_true", help="check the snapshot (default)")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    snap_path = Path(args.snapshot) if args.snapshot else root / "scripts/densex_env_defaults.json"
    try:
        got = read_defaults(root)
    except (ValueError, OSError) as exc:
        print(f"DENSEX_ENV_DEFAULTS_ERROR: {exc}", file=sys.stderr)
        return 2
    if args.update:
        snap_path.write_text(json.dumps({"version": 1, "switches": got}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {snap_path}")
        return 0
    if not snap_path.is_file():
        print(f"DENSEX_ENV_DEFAULTS_MISSING: {snap_path}", file=sys.stderr)
        return 2
    want = json.loads(snap_path.read_text(encoding="utf-8"))
    want_sw = want.get("switches", want)
    bad = []
    for name in sorted(set(got) | set(want_sw)):
        if name not in got:
            bad.append(f"{name}: removed")
            continue
        if name not in want_sw:
            bad.append(f"{name}: new (default {got[name]['default']!r})")
            continue
        if got[name]["default"] != want_sw[name]["default"]:
            bad.append(f"{name}: {want_sw[name]['default']!r} -> {got[name]['default']!r}")
    if bad:
        print("DENSEX_ENV_DEFAULTS_CHANGED", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        return 1
    print(f"DENSEX_ENV_DEFAULTS_OK {len(got)} switches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
