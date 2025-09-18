import re
from typing import Tuple

# Strict tags: <finalanswer>...</finalanswer>
_TAG_OPEN  = re.compile(r'(?is)<\s*finalanswer\s*>')
_TAG_CLOSE = re.compile(r'(?is)<\s*/\s*finalanswer\s*>')

# Minimal compatibility: "finalanswer>" (no '<', no closing tag)
_ARROW_OPEN = re.compile(r'(?is)\bfinalanswer\s*>')

# 句末边界：句号/问号/感叹号/省略号 + 可选右引号/右括号/方括号/书名号
_SENT_BOUNDARY = re.compile(
    r'(?us)(?<=[\.\!\?…。！？])(?:\s+|$)'
)

def _strip_keep(s: str) -> str:
    """与原版一致：两端去空白，不改变中间格式"""
    return s.strip() if isinstance(s, str) else ""

def _extract_last_sentence_fallback(text: str) -> Tuple[str, str]:
    """
    无标记时：将最后一句作为 Y，之前的作为 X。
    - 优先按句末边界切分（中英符号）
    - 若无法可靠切分，则用最后一个非空行
    - 若仍无法定位，则 (X="", Y=text)
    """
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    s = raw.rstrip()

    if not s:
        return "", ""

    # 用句末边界进行“近似句子分割”，尽量保留原始空白
    # 思路：通过边界位置 split，不破坏标点本身
    parts = re.split(_SENT_BOUNDARY, s)
    # re.split 会丢失分隔符，但我们只需要最后一段文本作为“最后一句”
    if len(parts) >= 2:
        # 最后一段非空即为最后一句；若最后一段为空（比如文本以换行结尾），找倒数第一个非空
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx].strip():
                last_sent = parts[idx]
                break
        else:
            # 都是空的：退化到整段
            return "", s.strip()

        # 在原始文本中从右往左定位该最后一句，保持原格式
        pos = s.rfind(last_sent)
        if pos != -1:
            x = s[:pos].rstrip()
            y = s[pos:].strip()
            return x, y

    # 若句末分割失败，退化为“最后一个非空行”
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2:
        last_line = lines[-1]
        pos = s.rfind(last_line)
        if pos != -1:
            x = s[:pos].rstrip()
            y = s[pos:].strip()
            return x, y

    # 再不行：把整段当作最后一句
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
