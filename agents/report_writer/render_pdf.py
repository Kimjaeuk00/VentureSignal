"""
Document → PDF. 기본은 Chromium(render_chrome.py, report.css 서식)이고, 브라우저를 쓸 수 없으면
이 모듈의 PyMuPDF Story(HTML+CSS)로 기본 서식 PDF 를 만든다. Story 방식은 새 라이브러리와 폰트 파일이 필요 없다.

한글은 MuPDF 의 내장 CJK 폴백 폰트로 그려지고 PDF 에 포함(임베드)된다. 머리글과 쪽번호는 본문을 그린 뒤
각 쪽에 HTML 상자로 덧붙인다(같은 폴백 폰트를 써서 한글도 뷰어에 의존하지 않는다).

MuPDF Story 의 제약(실험으로 확인)
- 표 열 너비는 % 나 colgroup 은 무시하고 셀의 style="width:Npt" 만 따른다 → Table.widths(%)를 pt 로 바꿔 준다.
- page-break-inside/after: avoid 는 무시되고 page-break-before: always 만 동작한다 → 쪽 아래에 제목만 홀로 남으면
  그 제목 앞에서만 쪽을 나누도록 배치 결과를 읽어(element_positions) 다시 그린다.
- 내장 폴백 폰트에는 굵은 한글이 없다 → 굵게(<b>)는 색으로 구분한다.
- 표 머리 배경색(background-color)은 이전 쪽들의 배경이 뒤 쪽마다 다시 그려지는 문제가 있다(th·tr·bgcolor 모두 동일) → 배경 없이
  굵은 아래 테두리와 글자색으로 머리 행을 구분한다.
"""

import html
import logging
import math
import re
from pathlib import Path
from typing import Optional

import pymupdf

from core.config import OUTPUT_DIR

from .document import Document, Paragraph, SubHeading, Table

PAGE = pymupdf.paper_rect("a4")
MARGIN = (48, 62, -48, -56)  # 좌, 상, 우, 하 (상·하는 머리글·쪽번호 자리를 남긴다)

CSS = """
body { font-size: 9.5pt; line-height: 1.5; color: #1f2933; }
h1 { font-size: 19pt; margin: 0 0 3pt 0; color: #10233a; }
h2 { font-size: 13pt; margin: 15pt 0 5pt 0; color: #10233a; border-bottom: 1px solid #4a5b70; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 11pt 0 4pt 0; color: #10233a; }
p { margin: 3pt 0; }
p.meta { font-size: 9pt; color: #52606d; margin-bottom: 6pt; }
b { color: #10233a; }
table { width: 100%; border-collapse: collapse; margin: 5pt 0 7pt 0; }
th { border: 1px solid #9aa7b5; border-bottom: 2px solid #4a5b70; padding: 4pt; font-size: 8.5pt; text-align: left; vertical-align: top; color: #10233a; }
td { border: 1px solid #9aa7b5; padding: 4pt; font-size: 8.5pt; vertical-align: top; }
"""
BAR_CSS = "body { font-size: 8pt; color: #6b7785; }"
CELL_OVERHEAD = 13.5  # 셀 하나가 내용 너비 밖에서 차지하는 폭(pt): 좌우 padding 8 + 테두리 등. Story 로 실측
MIN_ROOM = 110  # 제목 아래에 이만큼(pt)의 공간이 없으면 제목 앞에서 쪽을 나눈다
MAX_PASSES = 8

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")


def _inline(text: str) -> str:
    """이스케이프한 뒤 **굵게**, *기울임*, 줄바꿈만 태그로 바꾼다."""
    out = html.escape(text or "")
    out = _BOLD.sub(r"<b>\1</b>", out)
    out = _ITALIC.sub(r"<i>\1</i>", out)
    return out.replace("\n", "<br>")


def _table(t: Table, n: int = 0) -> str:
    """열 너비(%)는 셀의 width(pt)로 준다. Story 는 % 나 colgroup 은 무시하지만 pt 는 따른다."""
    content_w = PAGE.width - MARGIN[0] + MARGIN[2] - len(t.headers) * CELL_OVERHEAD - 2  # width 는 셀 안쪽(내용) 너비라 셀마다 여백·테두리를 뺀다
    widths = [f' style="width:{content_w * w / sum(t.widths):.0f}pt"' for w in t.widths] if len(t.widths) == len(t.headers) else [""] * len(t.headers)
    head = "".join(f"<th{widths[i]}>{_inline(h)}</th>" for i, h in enumerate(t.headers))
    rows = []
    for r, row in enumerate(t.rows):
        # id 가 있는 요소만 element_positions 로 위치를 알 수 있다: 머리 행(h)과 첫 본문 행(b)이 다른 쪽에 놓였는지 본다
        tag = f'<tr id="b{n}">' if r == 0 else "<tr>"
        rows.append(tag + "".join(f"<td{widths[i]}>{_inline(c)}</td>" for i, c in enumerate(row)) + "</tr>")
    return f'<table><tr id="h{n}">{head}</tr>{"".join(rows)}</table>'


def _table_owners(doc: Document) -> list[str]:
    """표마다 바로 앞의 제목 텍스트 (문서 순서). 표 머리만 홀로 남으면 이 제목 앞에서 쪽을 나눈다."""
    owners, current = [], ""
    for section in doc.sections:
        current = section.title
        for block in section.blocks:
            if isinstance(block, SubHeading):
                current = block.text
            elif isinstance(block, Table):
                owners.append(current)
    return owners


def render_html(doc: Document, breaks: frozenset = frozenset()) -> str:
    """breaks 에 든 제목(텍스트) 앞에서는 쪽을 나눈다."""
    brk = lambda text: ' style="page-break-before:always"' if text in breaks else ""
    meta = " · ".join(f"<b>{html.escape(k)}</b> {html.escape(v)}" for k, v in doc.meta)
    parts = [f"<h1>{html.escape(doc.title)}</h1>", f'<p class="meta">{meta}</p>']
    n = 0
    for section in doc.sections:
        parts.append(f"<h2{brk(section.title)}>{html.escape(section.title)}</h2>")
        for block in section.blocks:
            if isinstance(block, SubHeading):
                parts.append(f"<h3{brk(block.text)}>{_inline(block.text)}</h3>")
            elif isinstance(block, Paragraph):
                parts.append(f"<p>{_inline(block.text)}</p>")
            elif isinstance(block, Table):
                parts.append(_table(block, n))
                n += 1
    return "<html><body>" + "".join(parts) + "</body></html>"


def _draw_story(html_text: str, path: Path, owners: list[str] = ()) -> list[str]:
    """PDF 로 그리고, 홀로 남은 제목(아래 공간이 MIN_ROOM 보다 작음)과 본문 행 없이 머리만 남은 표의 제목 텍스트를 돌려준다."""
    story = pymupdf.Story(html=html_text, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    where = PAGE + MARGIN
    stranded: list[str] = []
    page_of: dict[str, int] = {}  # 표의 머리 행(h)/첫 본문 행(b) id → 놓인 쪽
    page_no = 0

    def on_element(position) -> None:
        rect = pymupdf.Rect(position.rect)
        if position.id and position.open_close & 1:
            page_of.setdefault(position.id, page_no)
        # open_close 1 = 여는 태그. 쪽 맨 위의 제목은 나눌 필요가 없다.
        if position.heading and position.open_close & 1 and rect.y0 > where.y0 + 4 and where.y1 - rect.y1 < MIN_ROOM:
            stranded.append(position.text)

    more = True
    while more:
        page_no += 1
        device = writer.begin_page(PAGE)
        more, _ = story.place(where)
        story.element_positions(on_element)
        story.draw(device)
        writer.end_page()
    writer.close()
    for n, owner in enumerate(owners):
        if f"h{n}" in page_of and f"b{n}" in page_of and page_of[f"h{n}"] != page_of[f"b{n}"]:
            stranded.append(owner)
    return stranded


def _decorate(path: Path, doc: Document) -> None:
    """머리글(제목)과 쪽번호를 덧붙이고 메타데이터를 넣은 뒤 압축해 다시 저장한다."""
    pdf = pymupdf.open(str(path))
    total = len(pdf)
    for i, page in enumerate(pdf, start=1):
        page.insert_htmlbox(pymupdf.Rect(48, 22, 547, 46), f"<p>VentureSignal · {html.escape(doc.title)}</p>", css=BAR_CSS)
        page.insert_htmlbox(pymupdf.Rect(48, 806, 547, 830), f'<p style="text-align:center">{i} / {total}</p>', css=BAR_CSS)
    pdf.set_metadata({"title": doc.title, "author": "VentureSignal", "subject": "투자 검토 보고서"})
    final = path.with_suffix(".tmp.pdf")
    pdf.ez_save(str(final), garbage=4, deflate=True)
    pdf.close()
    final.replace(path)


log = logging.getLogger(__name__)
USE_CHROMIUM = True  # False 면 항상 Story 로 만든다 (테스트·Chromium 없는 환경)

# 보고서 PDF 는 MAX_PAGES 쪽 이내여야 한다(요구사항). 넘으면 글자 크기는 그대로 두고 내용을 LLM 으로 요약해 다시 만든다(condense.py).
MAX_PAGES = 5
BODY_Y = (60.0, 786.0)  # A4 본문 영역의 세로 범위(pt) — Chromium(여백 18mm)과 Story(MARGIN) 모두에 대략 맞다


def write_pdf(doc: Document, out_dir: Optional[Path] = None) -> Path:
    """Chromium 으로 만들고, 실패하면(미설치·실행 오류 등) 기본 서식으로 대신 만든다."""
    out_dir = Path(out_dir) if out_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc.file_stem}.pdf"
    if USE_CHROMIUM:
        try:
            from .render_chrome import write_pdf_chrome

            return write_pdf_chrome(doc, path)
        except Exception as exc:  # ImportError(패키지 없음), 브라우저 실행 실패, 시간 초과 …
            log.warning("Chromium PDF 를 만들지 못해 기본 서식으로 대신 만든다: %s", exc)
    return _write_story(doc, path)


def _write_story(doc: Document, path: Path) -> Path:
    breaks: set[str] = set()
    for _ in range(MAX_PASSES):  # 홀로 남은 제목 앞에서 쪽을 나누면 뒤의 배치가 바뀌므로 안정될 때까지 다시 그린다
        stranded = _draw_story(render_html(doc, frozenset(breaks)), path, _table_owners(doc))
        new = set(stranded) - breaks
        if not new:
            break
        breaks |= new
    _decorate(path, doc)
    return path


def _pages_exact(path: Path) -> float:
    """쪽수(소수): 마지막 쪽은 내용이 차지한 만큼만 센다. 얼마나 줄여야 하는지 가늠하는 데 쓴다."""
    with pymupdf.open(str(path)) as pdf:
        last = pdf[-1]
        top, bottom = BODY_Y  # 머리글·쪽번호(가장자리)는 제외하고 본문이 차지한 높이만 본다
        body = [b for b in last.get_text("blocks") if top <= b[1] <= bottom]
        filled = (max((b[3] for b in body), default=top) - top) / (bottom - top)
        return len(pdf) - 1 + max(filled, 0.05)


def write_pdf_fit(doc: Document, out_dir: Optional[Path] = None) -> tuple[Document, Path, int]:
    """PDF 를 만들어 MAX_PAGES 쪽을 넘으면 내용을 요약(condense)해 다시 만든다. (문서, PDF 경로, 쪽수).

    글자 크기·여백은 그대로다. 요약은 원본 문서에서 다시 시작해 줄이는 비율만 키운다(요약을 거듭 요약하지 않는다).
    LLM 요약은 최대 LLM_ROUNDS 번이고(호출 상한은 llm_writer.llm_budget), 그래도 넘으면 LLM 없이 결정적으로 더 줄여
    5쪽 안에 들게 한다 — 이 단계는 PDF 만 다시 만들어 비용이 적다.
    """
    from .condense import LLM_ROUNDS, MAX_STEPS, condense, first_ratio

    path = write_pdf(doc, out_dir)
    exact = _pages_exact(path)
    pages = math.ceil(exact - 1e-6)
    if pages <= MAX_PAGES:
        return doc, path, pages
    ratio = first_ratio(doc, exact, MAX_PAGES)  # 처음에는 넘친 만큼만 줄여 본다
    for step in range(MAX_STEPS):
        use_llm = step < LLM_ROUNDS
        log.info("PDF %d쪽 (%.1f쪽 분량)으로 %d쪽을 넘어 내용을 %d%%로 %s (%d/%d)", pages, exact, MAX_PAGES, round(ratio * 100),
                 "LLM 요약한다" if use_llm else "결정적으로 줄인다", step + 1, MAX_STEPS)
        short = condense(doc, ratio, use_llm)
        path = write_pdf(short, out_dir)
        exact = _pages_exact(path)
        pages = math.ceil(exact - 1e-6)
        if pages <= MAX_PAGES:
            return short, path, pages
        ratio *= 0.8
    log.warning("가장 많이 줄여도 PDF 가 %d쪽이다 (제한 %d쪽)", pages, MAX_PAGES)
    return short, path, pages
