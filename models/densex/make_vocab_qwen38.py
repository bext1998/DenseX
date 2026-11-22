#!/usr/bin/env python3
"""DenseX: make a vocab-only GGUF from a full model GGUF (issue #82).

Input:  --src  full model GGUF (qwen35)
Output: --out  vocab-only GGUF with general.architecture + tokenizer.* only

No tensors are copied and no qwen35.* hparam is needed: for vocab_only loads
llama_model_load() returns right after llama_vocab::load(), which reads only
tokenizer.* keys. Keeping the file free of hparams also makes a full-model
load of this file fail early instead of looking like a real model.

Requires gguf-py from this repo (added to sys.path automatically).

Example:
  python models/densex/make_vocab_qwen38.py \
      --src models/Qwen3.8-27B.gguf --out models/ggml-vocab-qwen38.gguf
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_vocab_only(src: Path, out: Path) -> tuple[str, int, int]:
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(str(src))
    if "general.architecture" not in reader.fields:
        raise RuntimeError(f"source has no general.architecture: {src}")
    arch = str(reader.fields["general.architecture"].contents())
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        raise RuntimeError(f"source has no tokenizer.ggml.tokens: {src}")
    n_tokens = len(tokens_field.data)
    if len(reader.tensors) != 0:
        print(f"source tensors: {len(reader.tensors)} (not copied)")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), arch)
    for key in sorted(reader.fields):
        if not key.startswith("tokenizer."):
            continue
        field = reader.fields[key]
        vtype = field.types[0]
        sub_type = field.types[1] if len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), vtype, sub_type)
    if "general.name" in reader.fields:
        writer.add_string("general.name", str(reader.fields["general.name"].contents()))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return arch, n_tokens, len(reader.tensors)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract a vocab-only GGUF for the tokenizer test")
    ap.add_argument("--src", required=True, help="full model GGUF")
    ap.add_argument("--out", required=True, help="output vocab-only GGUF")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        print(f"make-vocab: source not found: {src}")
        return 1
    arch, n_tokens, n_tensors = build_vocab_only(src, out)

    reader = _reread(out)
    print(f"make-vocab: arch={arch} tokens={n_tokens} tensors_dropped={n_tensors}")
    print(f"make-vocab: out={out} size={out.stat().st_size} fields={len(reader.fields)} tensors={len(reader.tensors)}")
    return 0


def _reread(path: Path):
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    import gguf

    return gguf.GGUFReader(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
