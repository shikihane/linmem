"""Utility functions for linmem."""

from __future__ import annotations

import hashlib
import re
import string
from typing import List

import torch


def detect_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def text_hash(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- sentence splitting ---

_ZH_SENT_RE = re.compile(r"(?<=[。！？；\n])")
_EN_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, language: str = "zh") -> List[str]:
    """Split text into sentences by punctuation."""
    pattern = _ZH_SENT_RE if language == "zh" else _EN_SENT_RE
    parts = pattern.split(text)
    return [s.strip() for s in parts if s.strip()]


# --- normalize answer (language-aware) ---

def normalize_answer(text: str, language: str = "zh") -> str:
    """Normalize answer text for comparison."""
    text = text.lower().strip()
    if language == "zh":
        # remove Chinese punctuation + whitespace
        text = re.sub(r"[\s，。！？、；：""''（）【】《》…—]+", "", text)
    else:
        # English: remove articles, punctuation, extra whitespace
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
    return text
