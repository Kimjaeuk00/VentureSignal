"""보고서 문장 조립용 작은 헬퍼."""

import re
from typing import Iterable

from .references import TOKEN_RE

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SOURCE_START = re.compile(r"^[\[(]?\s*(?:S\d{3}|https?://)")  # 마침표 뒤에 따로 떨어진 출처 표기
_CLAUSE_END = re.compile(r"[;,]\s|\s(?:—|-)\s")


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for s in _SENTENCE_SPLIT.split(text):
        if out and _SOURCE_START.match(s):
            out[-1] += " " + s  # "…아니다. [S006]" 처럼 앞 문장에 딸린 출처는 붙여 둔다
        else:
            out.append(s)
    return out


def _cut(text: str, limit: int) -> str:
    """limit 안쪽의 절 경계(쉼표·세미콜론·공백)에서 자른다. 출처 URL·S번호와 괄호는 자르지 않는다."""
    pos = limit
    for m in _CLAUSE_END.finditer(text[:limit]):
        if m.end() >= limit * 0.5:
            pos = m.start() + 1 if text[m.start()] in ";," else m.start()
    else:
        if pos == limit:
            space = text.rfind(" ", 0, limit)
            pos = space if space > limit * 0.5 else limit
    for m in TOKEN_RE.finditer(text):  # 출처 표기 한가운데면 끝까지 포함한다
        if m.start() < pos < m.end():
            pos = m.end()
    head = text[:pos].rstrip(" ,;")
    for open_ch, close_ch in (("(", ")"), ("[", "]")):  # 열린 괄호는 닫힌 곳까지 포함하거나 그 앞에서 자른다
        if head.count(open_ch) > head.count(close_ch):
            close = text.find(close_ch, pos)
            if 0 <= close <= pos + 40:
                return text[: close + 1] + "…"
            head = head[: head.rfind(open_ch)].rstrip(" ,;")
    return head.rstrip(".。 ,;") + "…"


def first_sentence(text: str, limit: int = 160) -> str:
    """첫 문장. 길면 출처 URL·괄호를 자르지 않고 절 경계에서 줄인다."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    sentence = sentences(text)[0]
    return sentence if len(sentence) <= limit else _cut(sentence, limit)


def brief(text: str, limit: int = 110) -> str:
    """읽기 좋은 길이로 줄인다: limit 안에 들어가는 만큼의 문장(최소 1문장), 첫 문장이 길면 절 경계에서 자른다."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    kept: list[str] = []
    for s in sentences(text):
        if kept and len(" ".join([*kept, s])) > limit:
            break
        kept.append(s)
    out = " ".join(kept)
    return out if len(out) <= limit * 1.25 else _cut(out, limit)


def dedupe(items: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        item = " ".join((item or "").split())
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def bullets(items: Iterable[str], limit: int = 5, cap: int = 0) -> str:
    """줄바꿈으로 나눈 글머리 목록. 비어 있으면 빈 문자열. cap 이 있으면 항목마다 그 길이로 줄인다."""
    return "\n".join(f"• {brief(i, cap) if cap else i}" for i in dedupe(items)[:limit])


def join(items: Iterable[str], sep: str = "; ", limit: int = 3, cap: int = 0) -> str:
    return sep.join(brief(i, cap) if cap else i for i in dedupe(items)[:limit])
