# -*- coding: utf-8 -*-
"""DenseX 轉檔白名單（issue #86 Phase 7 待決策 7，擁有者選 B）。

做什麼：
    在轉檔入口決定「這份 HuggingFace 模型能不能轉成 GGUF」。不在白名單裡的架構，在
    讀任何權重之前就停下，印以 `densex-convert:` 開頭的錯誤訊息，並以 exit code 1 結束
    （不印 Python traceback）。載入端另有 `src/llama-model.cpp` 的 factory 拒絕，兩邊一起
    構成「轉檔就擋、載入也擋」。

輸入輸出：
    輸入 config.json 的架構名稱（`architectures[0]`）與模式（一般文字模型，或 `--mmproj`
    的視覺塔）。通過時不做任何事；不通過時拋 `ConversionNotAllowedError`，訊息含偵測到的
    架構名稱、DenseX 支援的完整清單，以及「要支援其他模型必須重新審議」的說明。
    入口檔用 `require_supported_or_exit()` 一行完成檢查、印訊息、結束。

依賴：
    只依賴 Python 標準庫（`logging`、`typing`），不 import torch 與 `conversion.base`，
    讓入口檔能在建立 model class、讀取權重之前就做檢查。

修改規則：
    本檔是 DenseX 支援範圍的唯一來源（白名單只放在這裡）。改這裡等於放寬或收窄支援
    範圍，必須走 `AGENTS.md`「基本規則 11」的重新審議流程：開 issue 說明要支援什麼、
    為什麼、維護成本與對精度／穩定度／VRAM 的影響，提出 2～3 個做法交擁有者決定，同意後
    白名單、載入 factory、文件、測試在同一個 commit 裡一起更新並跑完整驗證。
    agent 不得自行放寬本白名單，也不得從 git 歷史救回已刪除的架構。
"""

from __future__ import annotations

import logging
from typing import Final

# 重新審議的入口（AGENTS.md 基本規則 11）
_REAPPROVAL_URL: Final = "https://github.com/bext1998/DenseX/issues"

# 主模型（文字）：Qwen 3.8 系列，轉出 `general.architecture = qwen35`。
# 三個名稱都是 `conversion/qwen.py` 的 `Qwen3_5TextModel`；Qwen 3.8 的 HF config 回報的就是
# `Qwen3_5ForConditionalGeneration`（研究實測見 docs/Qwen38_GSQ_RCO_Inference_Research_zh-TW(1).md:15、
# docs/tasks/CODEX_71_BENCH_DESIGN_REPORT.md:151），產品名 3.8 不代表原始碼有 Qwen3_8* 類別。
ALLOWED_TEXT_ARCHS: Final = frozenset({
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5TextModel",
})

# draft 模型：DenseX 正式在用的 DFlash／DFlash2 草稿，轉出 `general.architecture = dflash`。
# 對應 `conversion/qwen.py` 的 `DFlashModel`（@ModelBase.example("z-lab/Qwen3.5-9B-DFlash")）；
# 實際使用的是 z-lab/Qwen3.8-27B-DFlash2-GGUF（`DenseX/bench/common.ps1:107` 的 $DenseXDflash2）。
ALLOWED_DRAFT_ARCHS: Final = frozenset({
    "DFlashDraftModel",
    "DFlash2DraftModel",
})

# 視覺編碼器（--mmproj）：同系列的 mmproj，轉出 `general.architecture = clip`。
# `conversion/__init__.py` 的 MMPROJ_MODEL_MAP 只有這個 Qwen 3.8 名稱對到 `qwen3vl` 模組。
ALLOWED_MMPROJ_ARCHS: Final = frozenset({
    "Qwen3_5ForConditionalGeneration",
})


class ConversionNotAllowedError(Exception):
    """架構不在白名單；訊息已含完整說明，呼叫端逐行印出後以 exit 1 結束。"""


def _message(architecture: str, mmproj: bool) -> str:
    mode = "mmproj" if mmproj else "text"
    return "\n".join([
        f"densex-convert: model architecture {architecture!r} ({mode} mode) is not supported by DenseX.",
        "densex-convert: DenseX converts only:",
        f"densex-convert:   text:   {', '.join(sorted(ALLOWED_TEXT_ARCHS))}  (dense Qwen 3.8 main models)",
        f"densex-convert:   draft:  {', '.join(sorted(ALLOWED_DRAFT_ARCHS))}  (DFlash draft models)",
        f"densex-convert:   mmproj: {', '.join(sorted(ALLOWED_MMPROJ_ARCHS))}  (vision projector, use --mmproj)",
        f"densex-convert: supporting any other model or architecture requires re-approval:"
        f" open an issue at {_REAPPROVAL_URL} (AGENTS.md base rule 11)",
    ])


def require_supported(architecture: str, *, mmproj: bool = False) -> None:
    """通過就回傳；不通過就拋 `ConversionNotAllowedError`。"""
    allowed = ALLOWED_MMPROJ_ARCHS if mmproj else (ALLOWED_TEXT_ARCHS | ALLOWED_DRAFT_ARCHS)
    if architecture in allowed:
        return
    raise ConversionNotAllowedError(_message(architecture, mmproj))


def require_supported_or_exit(architecture: str, *, mmproj: bool, logger: logging.Logger) -> None:
    """轉檔入口用：不通過就逐行印錯誤並以 exit 1 結束，不印 traceback。"""
    try:
        require_supported(architecture, mmproj=mmproj)
    except ConversionNotAllowedError as exc:
        for line in str(exc).splitlines():
            logger.error(line)
        raise SystemExit(1) from None


def supported_models_lines() -> list[str]:
    """`--print-supported-models` 用：只列白名單，不列其餘已註冊的上游架構。"""
    return [
        "text (dense Qwen 3.8 main models):",
        *[f"  - {name}" for name in sorted(ALLOWED_TEXT_ARCHS)],
        "draft (DFlash):",
        *[f"  - {name}" for name in sorted(ALLOWED_DRAFT_ARCHS)],
        "mmproj (vision projector, --mmproj):",
        *[f"  - {name}" for name in sorted(ALLOWED_MMPROJ_ARCHS)],
    ]
