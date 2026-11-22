#!/usr/bin/env python3
"""DenseX: make the .inp/.out fixtures for test-tokenizer-0-qwen38 (issue #82).

Input:
  --tokenizer    HF tokenizer.json of Qwen 3.8
  --base-inp     upstream test string set (models/ggml-vocab-qwen35.gguf.inp)
  --extra-tests  extra strings (models/densex/qwen38_extra_tests.txt)
  --out          vocab-only GGUF; writes <out>.inp and <out>.out next to it

The test runs llama_tokenize(..., parse_special=false): CONTROL token text is
tokenized as ordinary text, while USER_DEFINED added tokens (for example
<tool_call>) are still matched as single tokens. HF tokenizers match every added
token, so for CONTROL tokens we replace the added-token text with an unused
sentinel string; this keeps the original token ids (all added tokens stay in the
JSON) but stops them from matching, so the text falls through to plain BPE.
Qwen marks CONTROL tokens with special=true or with the <|...|> spelling, while
USER_DEFINED ones look like <tool_call>. Verified against the C++ test.

The HF tokenizer.json declares an NFC normalizer, but the formal Qwen 3.8 GGUF
does not carry tokenizer.ggml.normalizer.nfc, so llama.cpp loads it without NFC.
The fixtures therefore disable the normalizer too; otherwise decomposed accents
would differ from what DenseX actually tokenizes.

Requires the `tokenizers` package.
"""

import argparse
import json
import re
from pathlib import Path

SEP = "\n__ggml_vocab_test__\n"


def load_strings(base_inp: Path) -> list[str]:
    text = base_inp.read_text(encoding="utf-8")
    parts = text.split(SEP)
    if parts and parts[-1] == "":
        parts.pop()
    if not parts:
        raise RuntimeError(f"no test strings in {base_inp}")
    return parts


class TokenizerBundle:
    """HF tokenizer patched to match llama.cpp with parse_special=false."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids


CONTROL_TOKEN_TEXT = re.compile(r"^<\|[^|]+\|>$")


def load_tokenizers(tokenizer_json: Path) -> tuple[TokenizerBundle, int, int]:
    import tokenizers

    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    added = data.get("added_tokens", [])
    n_control = 0
    for t in added:
        is_control = bool(t.get("special", False)) or bool(CONTROL_TOKEN_TEXT.match(t["content"]))
        if is_control:
            # 保留條目與 id，換成文字裡不會出現的 sentinel，讓它不再被匹配
            t["content"] = f"\x00densex-control-{t['id']}\x00"
            n_control += 1
    data["normalizer"] = None
    tok = tokenizers.Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    return TokenizerBundle(tok), len(added), n_control


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate tokenizer test fixtures for Qwen 3.8")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer.json")
    ap.add_argument("--base-inp", required=True, help="upstream .inp with the base test strings")
    ap.add_argument("--extra-tests", required=True, help="extra test strings in .inp format")
    ap.add_argument("--out", required=True, help="vocab-only GGUF path (fixtures go next to it)")
    args = ap.parse_args()

    tokenizer_json = Path(args.tokenizer)
    base_inp = Path(args.base_inp)
    extra_tests = Path(args.extra_tests)
    out = Path(args.out)
    if not tokenizer_json.is_file():
        print(f"make-vocab-tests: tokenizer not found: {tokenizer_json}")
        return 1
    if not base_inp.is_file():
        print(f"make-vocab-tests: base inp not found: {base_inp}")
        return 1
    if not extra_tests.is_file():
        print(f"make-vocab-tests: extra tests not found: {extra_tests}")
        return 1

    strings = load_strings(base_inp) + load_strings(extra_tests)
    bundle, n_added, n_control = load_tokenizers(tokenizer_json)

    inp_path = out.with_name(out.name + ".inp")
    out_path = out.with_name(out.name + ".out")
    inp_path.write_text(SEP.join(strings) + SEP, encoding="utf-8", newline="\n")

    ids_lines = []
    for s in strings:
        ids = bundle.encode(s)
        ids_lines.append(" ".join(str(i) for i in ids))
    out_path.write_text("\n".join(ids_lines) + "\n", encoding="utf-8", newline="\n")

    print(f"make-vocab-tests: strings={len(strings)} added_tokens={n_added} control_tokens={n_control}")
    print(f"make-vocab-tests: wrote {inp_path} ({inp_path.stat().st_size} bytes)")
    print(f"make-vocab-tests: wrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
