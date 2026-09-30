"""
보고서 문서 모델 — 마크다운(render_md)과 PDF(render_pdf)가 같은 모델을 각자의 형식으로 변환한다.

텍스트에는 `**굵게**`, `*기울임*`, 줄바꿈(\\n)만 쓴다. 표 셀도 같다.
"""

from dataclasses import dataclass, field
from typing import Union


@dataclass
class Paragraph:
    text: str


@dataclass
class SubHeading:
    text: str


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    widths: list[int] = field(default_factory=list)  # 열 너비(%). 비우면 균등


Block = Union[Paragraph, SubHeading, Table]


@dataclass
class Section:
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Document:
    title: str
    meta: list[tuple[str, str]]  # 작성일·작성자·기준일 등
    sections: list[Section]
    kind: str = "invest"  # invest / hold
    file_stem: str = ""  # PDF 파일명(확장자 제외)
