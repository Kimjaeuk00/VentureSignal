"""마크다운 보고서 → HTML → Chromium(Playwright) → PDF.

VS Code 의 MarkdownPDF 와 같은 방식이라 report.css 서식이 그대로 나온다. 브라우저를 못 띄우면 예외를 내고,
호출한 render_pdf.write_pdf 가 기본 서식(PyMuPDF Story)으로 대신 만든다.

브라우저: 내려받은 Chromium(`playwright install chromium`)을 먼저 쓰고, 없으면 설치된 Chrome 을 쓴다.
"""

import html
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import markdown
import pymupdf

from .document import Document
from .render_md import render_md

CSS_PATH = Path(__file__).with_name("report.css")
MARGIN = {"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"}
LAUNCH_TIMEOUT_MS = 30_000
PDF_TIMEOUT_MS = 60_000
INSTALL_HINT = "playwright install chromium (또는 Chrome 설치)"

KIND_LABEL = {"invest": "투자 검토 보고서", "hold": "보류 보고서"}
BAR_STYLE = "font-size:8px; width:100%; padding:0 16mm; color:#9B9A97; display:flex; justify-content:space-between;"


def to_html(md_text: str, title: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"])
    css = CSS_PATH.read_text(encoding="utf-8")
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f"<style>{css}</style></head><body>{body}</body></html>")


def _templates(doc: Document) -> tuple[str, str]:
    header = (f'<div style="{BAR_STYLE}"><span>VentureSignal · {html.escape(doc.title)}</span>'
              f"<span>{KIND_LABEL.get(doc.kind, '')}</span></div>")
    footer = ('<div style="font-size:8px; width:100%; text-align:center; color:#9B9A97;">'
              '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    return header, footer


def _launch(playwright):
    """내려받은 Chromium → 설치된 Chrome 순서로 시도한다."""
    errors = []
    for options in ({}, {"channel": "chrome"}):
        try:
            return playwright.chromium.launch(timeout=LAUNCH_TIMEOUT_MS, **options)
        except Exception as exc:  # 브라우저 없음·실행 실패
            errors.append(re.sub(r"\s+", " ", str(exc))[:120])
    raise RuntimeError(f"Chromium 을 실행하지 못했다 — {INSTALL_HINT}: {' | '.join(errors)}")


def _print(html_text: str, path: Path, doc: Document) -> None:
    from playwright.sync_api import sync_playwright  # 설치되지 않았으면 ImportError 로 폴백한다

    header, footer = _templates(doc)
    with sync_playwright() as p:
        browser = _launch(p)
        try:
            page = browser.new_page()
            page.set_default_timeout(PDF_TIMEOUT_MS)
            page.set_content(html_text, wait_until="load")
            page.pdf(path=str(path), format="A4", print_background=True, display_header_footer=True,
                     header_template=header, footer_template=footer, margin=MARGIN)
        finally:
            browser.close()


def _set_metadata(path: Path, doc: Document) -> None:
    pdf = pymupdf.open(str(path))
    pdf.set_metadata({"title": doc.title, "author": "VentureSignal", "subject": "투자 검토 보고서"})
    tmp = path.with_suffix(".tmp.pdf")
    pdf.ez_save(str(tmp), garbage=4, deflate=True)
    pdf.close()
    tmp.replace(path)


def write_pdf_chrome(doc: Document, path: Path) -> Path:
    """path 에 PDF 를 쓴다. 실패하면 예외(호출자가 폴백). 비동기 루프 안에서도 안전하도록 별도 스레드에서 실행한다."""
    html_text = to_html(render_md(doc), doc.title)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_print, html_text, path, doc).result()
    _set_metadata(path, doc)
    return path
