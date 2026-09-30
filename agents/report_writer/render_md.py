"""Document → 마크다운. 표 셀의 줄바꿈은 <br>, 파이프는 이스케이프한다."""

from .document import Document, Paragraph, SubHeading, Table


def _cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", "<br>")


def _table(t: Table) -> list[str]:
    lines = ["| " + " | ".join(_cell(h) for h in t.headers) + " |", "|" + "|".join("---" for _ in t.headers) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in t.rows]
    return lines


def render_md(doc: Document) -> str:
    lines = [f"# {doc.title}", "", " · ".join(f"**{k}** {v}" for k, v in doc.meta), ""]
    for section in doc.sections:
        lines += [f"## {section.title}", ""]
        for block in section.blocks:
            if isinstance(block, SubHeading):
                lines += [f"### {block.text}", ""]
            elif isinstance(block, Paragraph):
                lines += [block.text.replace("\n", "  \n"), ""]
            elif isinstance(block, Table):
                lines += [*_table(block), ""]
    return "\n".join(lines).rstrip() + "\n"
