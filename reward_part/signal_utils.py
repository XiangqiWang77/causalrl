import re
from typing import Tuple

# Strict tags: <finalanswer>...</finalanswer>
_TAG_OPEN  = re.compile(r'(?is)<\s*finalanswer\s*>')
_TAG_CLOSE = re.compile(r'(?is)<\s*/\s*finalanswer\s*>')

# Minimal compatibility: "finalanswer>" (no '<', no closing tag)
_ARROW_OPEN = re.compile(r'(?is)\bfinalanswer\s*>')

_SENT_BOUNDARY = re.compile(
    r'(?us)(?<=[\.\!\?…。！？])(?:\s+|$)'
)

def _strip_keep(s: str) -> str:
    return s.strip() if isinstance(s, str) else ""

def _extract_last_sentence_fallback(text: str) -> Tuple[str, str]:
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    s = raw.rstrip()

    if not s:
        return "", ""

    parts = re.split(_SENT_BOUNDARY, s)
    if len(parts) >= 2:
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx].strip():
                last_sent = parts[idx]
                break
        else:
            return "", s.strip()

        pos = s.rfind(last_sent)
        if pos != -1:
            x = s[:pos].rstrip()
            y = s[pos:].strip()
            return x, y

    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2:
        last_line = lines[-1]
        pos = s.rfind(last_line)
        if pos != -1:
            x = s[:pos].rstrip()
            y = s[pos:].strip()
            return x, y

    return "", s.strip()

def extract_XY(resp: str) -> Tuple[str, str]:
    """
    X = text before the final-answer marker
    Y = text inside <finalanswer>...</finalanswer>, or after 'finalanswer>' if tags are absent
    Fallback: if no markers, return (X=all before last sentence, Y=last sentence)
    No fuzzy markers. No heuristics beyond sentence fallback.
    """
    if not isinstance(resp, str):
        resp = "" if resp is None else str(resp)
    text = resp

    # 1) Strict <finalanswer>...</finalanswer>
    m_open = _TAG_OPEN.search(text)
    if m_open:
        x = _strip_keep(text[:m_open.start()])
        m_close = _TAG_CLOSE.search(text, m_open.end())
        if m_close:
            y = _strip_keep(text[m_open.end():m_close.start()])
        else:
            y = _strip_keep(text[m_open.end():])
        return x, y

    # 2) Minimal fall-back: 'finalanswer>' (case-insensitive)
    m_arrow = _ARROW_OPEN.search(text)
    if m_arrow:
        x = _strip_keep(text[:m_arrow.start()])
        y = _strip_keep(text[m_arrow.end():])
        return x, y

    # 3) No marker found: fallback to "last sentence" logic
    return _extract_last_sentence_fallback(text)
