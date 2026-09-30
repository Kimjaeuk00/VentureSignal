"""분량 줄이기 — PDF 가 쪽수 제한을 넘으면 보고서의 긴 칸·문단을 LLM 으로 요약한다.

글자 크기와 여백은 그대로 두고 내용을 줄인다. 표의 항목명(첫 열)과 출처 칸, REFERENCE 는 건드리지 않는다.
요약은 원본 문서에서 다시 시작하므로 거듭 요약해 뭉개지지 않는다.

검증: 요약문에는 원문에 없는 숫자와 원문에 없는 인용 번호([n])가 있으면 안 되고, 목표 글자 수를 넘으면 안 된다.
어긴 칸과, LLM 을 쓰지 않는 단계(호출 상한·실패)의 칸은 결정적으로(문장 경계에서) 자른다. LLM 호출은 condense 한 번에 최대 1회다.
"""

import copy
import logging
import re
from dataclasses import dataclass
from typing import Callable

from langsmith import traceable

from . import llm_writer
from .document import Document, Paragraph, Table
from .prompts import CONDENSE_SYSTEM
from .schemas import CondensedCells
from .textutil import brief

logger = logging.getLogger(__name__)

LLM_ROUNDS = 2  # write_pdf_fit 에서 LLM 요약을 쓰는 최대 횟수. 그 뒤에도 넘으면 LLM 없이 결정적으로 더 줄인다
MAX_STEPS = 6  # 요약 비율을 낮춰 다시 만드는 전체 최대 횟수 (PDF 만들기만 반복 — 비용이 적다)
MIN_LEN = 90  # 이보다 짧은 칸은 줄이지 않는다
MIN_TARGET = 50
LENGTH_SLACK = 1.5  # LLM 은 글자 수를 정확히 못 맞춘다. 결정적으로 자르는 것보다 조금 긴 요약이 낫다 — 못 맞추면 다음 단계에서 더 줄인다
CITE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
ONLY_CITE_RE = re.compile(r"^\s*(?:\[[\d,\s]*\]\s*)*$")
DANGLING_CITE_RE = re.compile(r"\s*\[[\d,\s]*…$")


@dataclass
class Slot:
    get: Callable[[], str]
    set: Callable[[str], None]


def _slots(doc: Document) -> list[Slot]:
    """줄일 수 있는 칸: 표의 본문 칸(첫 열·출처 번호만 있는 칸·짧은 칸 제외)과 문단 (REFERENCE 제외)."""
    slots: list[Slot] = []

    def add(get, set_):
        if len(get()) > MIN_LEN and not ONLY_CITE_RE.match(get()):
            slots.append(Slot(get, set_))

    for section in doc.sections:
        if "REFERENCE" in section.title:
            continue
        for block in section.blocks:
            if isinstance(block, Paragraph):
                add(lambda b=block: b.text, lambda v, b=block: setattr(b, "text", v))
            elif isinstance(block, Table):
                for row in block.rows:
                    if row and row[0] == "출처":  # 비교표의 출처 행
                        continue
                    for c in range(1, len(row)):
                        add(lambda r=row, c=c: r[c], lambda v, r=row, c=c: r.__setitem__(c, v))
    return slots


def first_ratio(doc: Document, exact_pages: float, max_pages: int) -> float:
    """넘친 분량을 줄이려면 요약 대상 칸을 얼마로 줄여야 하는지 가늠한다 (표 머리·항목명·REFERENCE 등 줄일 수 없는 몫은 그대로다)."""
    from .render_md import render_md

    condensable = sum(len(s.get()) for s in _slots(doc))
    share = condensable / max(1, len(render_md(doc)))
    need = 1 - (max_pages - 0.1) / exact_pages  # 전체에서 덜어 내야 하는 비율
    return min(0.9, max(0.25, 1 - 1.6 * need / max(share, 0.2)))  # LLM 이 목표보다 길게 쓰는 것을 감안해 여유를 둔다


def _citations(text: str) -> set[str]:
    return {n for m in CITE_RE.finditer(text) for n in re.findall(r"\d+", m.group(1))}


def _problem(original: str, out: str, target: int) -> str:
    """요약문이 규칙을 어겼으면 이유, 아니면 빈 문자열."""
    if not out.strip():
        return "비어 있다"
    if len(out) > target * LENGTH_SLACK:
        return f"{len(out)}자로 목표 {target}자를 넘는다"
    bad = llm_writer.ungrounded_numbers(out, original)
    if bad:
        return f"원문에 없는 숫자: {', '.join(bad)}"
    extra = _citations(out) - _citations(original)
    if extra:
        return f"원문에 없는 인용 번호: {', '.join(sorted(extra, key=int))}"
    return ""


def _cut(text: str, target: int) -> str:
    """결정적 요약 — 줄(글머리)마다 문장 경계에서 자른다. 인용 번호가 반쯤 잘리지 않게 한다."""
    lines = text.split("\n")
    per_line = max(MIN_TARGET // 2, target // len(lines))
    return "\n".join(DANGLING_CITE_RE.sub("…", brief(line, per_line)) for line in lines)


def _ask(texts: dict[int, str], targets: dict[int, int]) -> dict[int, str]:
    user = "\n\n".join(f"### id={i} (원문 {len(t)}자 → 목표 {targets[i]}자 이내)\n{t}" for i, t in texts.items())
    result: CondensedCells = llm_writer._invoke(CondensedCells, CONDENSE_SYSTEM, user)
    return {c.id: c.text.strip() for c in result.cells if c.id in texts}


@traceable(name="report_writer.condense", run_type="chain")
def condense(doc: Document, ratio: float, use_llm: bool = True) -> Document:
    """긴 칸을 원문 길이의 ratio 배 안팎으로 요약한 새 문서를 돌려준다. LLM 호출은 최대 1회, 예외를 내지 않는다."""
    short = copy.deepcopy(doc)
    slots = _slots(short)
    originals = {i: s.get() for i, s in enumerate(slots)}
    targets = {i: max(MIN_TARGET, int(len(t) * ratio)) for i, t in originals.items()}

    done: dict[int, str] = {}
    if use_llm and originals:
        try:
            answers = _ask(originals, targets)
        except llm_writer.LlmBudgetExceeded as exc:
            logger.warning("%s → 요약은 결정적으로 자릅니다", exc)
            answers = {}
        except Exception:
            logger.exception("LLM 요약 호출 실패 → 결정적으로 자릅니다")
            answers = {}
        problems = {}
        for i, text in originals.items():
            problem = _problem(text, answers.get(i, ""), targets[i]) if i in answers else "응답 없음"
            if problem:
                problems[i] = problem
            else:
                done[i] = answers[i]
        if answers and problems:
            logger.warning("LLM 요약 %d/%d칸이 검증에 실패해 그 칸만 자릅니다: %s", len(problems), len(originals),
                           "; ".join(f"id={i} {p}" for i, p in list(problems.items())[:5]))
    for i, text in originals.items():
        slots[i].set(done.get(i) or _cut(text, targets[i]))
    return short
