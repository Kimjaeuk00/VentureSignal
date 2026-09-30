"""
출처 번호 레지스트리 — 본문 인용 [번호]와 REFERENCE 목록을 만든다.

출처는 두 종류다.
- RAG 출처 ID (S006 …): rag.sources.load_sources 의 publisher/title/url 로 서지정보를 만든다.
- 웹 URL: 제목을 알면 제목과 함께, 모르면 URL 만 적고 접속일(기준일)을 붙인다.

번호는 본문에서 처음 인용된 순서로 붙고, REFERENCE 에는 실제로 인용된 출처만 들어간다.
본문에 이미 들어 있는 인라인 표기("[S006; 분석]", "(https://…)")는 rewrite 로 같은 번호 체계에 맞춘다.
"""

import re
from typing import Iterable, Optional
from urllib.parse import urlsplit

from core.sources import source_key
from rag.sources import load_sources

# S번호는 다른 글자에 붙어 있으면(DX-S010) 출처가 아니다. URL 끝의 문장부호는 URL 이 아니다.
TOKEN = r"(?<![\w-])S\d{3}\b|https?://[^\s)\]>,;]+"
GROUP = rf"[\[(]\s*(?:{TOKEN})(?:\s*[,;·/]\s*(?:{TOKEN}|분석|공개 자료))*\s*[\])]"
INLINE_RE = re.compile(rf"(?P<group>{GROUP})|(?P<token>{TOKEN})")
TOKEN_RE = re.compile(TOKEN)
S_ID_RE = re.compile(r"^S\d{3}$")
DATE_RE = re.compile(r"\d{4}(?:-\d{2}-\d{2})?")

NO_DATE = "발행일 미표기"
REPEATED_CITE_RE = re.compile(r"(\[\d+(?:, \d+)*\])(?:\s*\1)+")

# sources.json 일부 항목의 url 뒤에 PDF 쪽 하단("05 • SOURCE REGISTER …")이 붙어 있다. 쪽 번호와 함께 잘라낸다.
PAGE_FOOTER_RE = re.compile(r"\d{2}\s*•\s*SOURCE REGISTER.*$")


def _domain(url: str) -> str:
    return urlsplit(url).netloc.removeprefix("www.")


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", text.lower())


def _loosely_in(needle: str, hay: str) -> bool:
    """needle 의 글자가 hay 안에 순서대로, 길이의 2배 이내 구간에서 나오면 참 (gminsights ↔ Global Market Insights)."""
    if len(needle) < 4:
        return False
    if needle in hay:
        return True
    for i, ch in enumerate(hay):
        if ch != needle[0]:
            continue
        j = i
        for c in needle[1:]:
            j = hay.find(c, j + 1)
            if j < 0 or j - i > 2 * len(needle) + 2:
                break
        else:
            return True
    return False


class SourceRegistry:
    def __init__(self, titles: Optional[dict[str, str]] = None, accessed: str = ""):
        self._order: list[str] = []
        self._index: dict[str, int] = {}  # 같은 페이지의 출처(S번호와 URL 포함)는 한 번호를 쓴다
        self._titles: dict[str, str] = dict(titles or {})
        self._titles_by_key: dict[str, str] = {source_key(u): t for u, t in self._titles.items()}
        self.accessed = accessed
        try:
            self._rag = load_sources()
        except Exception:  # 인덱스 미생성(python -m rag.ingest 필요) 등 — 보고서는 계속 만든다
            self._rag = {}

    # --- 번호 -----------------------------------------------------------------

    def add_titles(self, titles: dict[str, str]) -> None:
        self._titles.update({u: t for u, t in titles.items() if u and t})
        self._titles_by_key.update({source_key(u): t for u, t in titles.items() if u and t})

    def _key(self, source: str) -> str:
        if S_ID_RE.match(source):
            info = self._rag.get(source)
            if info:
                return source_key(PAGE_FOOTER_RE.sub("", info["url"]).strip())
            return source
        return source_key(source) if source.startswith(("http://", "https://")) else source

    def number(self, source: str) -> int:
        source = source.strip()
        key = self._key(source)
        if key not in self._index:
            self._index[key] = len(self._order)
            self._order.append(source)
        elif S_ID_RE.match(source) and not S_ID_RE.match(self._order[self._index[key]]):
            self._order[self._index[key]] = source  # 서지정보가 있는 S번호 표기를 우선한다
        return self._index[key] + 1

    def _labels(self, source: str) -> list[str]:
        """본문에서 이 출처를 가리킬 때 쓸 만한 이름들 (도메인 이름, 발행처, 제목)."""
        if S_ID_RE.match(source):
            info = self._rag.get(source) or {}
            names = [info.get("publisher", "").partition(" · ")[0], info.get("title", "")]
            url = PAGE_FOOTER_RE.sub("", info.get("url", "")).strip()
        else:
            names, url = [self._titles.get(source) or self._titles_by_key.get(source_key(source), "")], source
        host = _domain(url).split(".") if url else []
        names.append(host[-2] if len(host) >= 2 and len(host[-2]) > 3 else (host[0] if host else ""))
        return [n for n in map(_alnum, names) if n]

    def pick(self, text: str, sources: Iterable[str], limit: int = 4) -> str:
        """본문에 인용 표기가 없을 때 출처 칸에 적을 번호. 본문이 이름을 언급한 출처를 우선하고, 없으면 앞의 limit 개."""
        sources = list(dict.fromkeys(s.strip() for s in sources if s and s.strip()))
        hay = _alnum(text)
        named = [s for s in sources if any(_loosely_in(n, hay) for n in self._labels(s))]
        return self.cite((named or sources)[:limit])

    def cite(self, sources: Iterable[str]) -> str:
        """출처 목록 → '[1, 3]'. 중복은 한 번만, 번호 오름차순. 출처가 없으면 빈 문자열."""
        numbers = sorted({self.number(s) for s in sources if s and s.strip()})
        return f"[{', '.join(map(str, numbers))}]" if numbers else ""

    def rewrite(self, text: str) -> str:
        """본문에 들어 있는 인라인 출처(S번호·URL)를 [번호]로 바꾼다."""
        if not text:
            return text

        def replace(m: re.Match) -> str:
            if m.group("group"):
                return self.cite(self._clean(t) for t in TOKEN_RE.findall(m.group("group")))
            token = m.group("token")
            clean = self._clean(token)
            return self.cite([clean]) + token[len(clean):]  # 뒤에 딸려 온 문장부호는 본문에 남긴다

        # 한 번의 패스로 처리해야 번호가 본문 등장 순서를 따른다
        out = INLINE_RE.sub(replace, text)
        return REPEATED_CITE_RE.sub(r"\1", out)  # 같은 페이지를 S번호와 URL 로 잇달아 인용하면 [1] [1] 이 된다

    @staticmethod
    def _clean(token: str) -> str:
        return token.rstrip(".:")  # 문장 끝 마침표 등은 URL 에 속하지 않는다

    # --- REFERENCE ---------------------------------------------------------------

    def _format(self, source: str) -> str:
        if S_ID_RE.match(source):
            info = self._rag.get(source)
            if not info:
                return f"{source} (출처 목록에 없음)"
            org, _, tail = info["publisher"].partition(" · ")
            date = DATE_RE.search(tail)
            url = PAGE_FOOTER_RE.sub("", info["url"]).strip()
            return f"{org}({date.group(0) if date else NO_DATE}). *{info['title']}*. {_domain(url)}, {url}"
        if source.startswith(("http://", "https://")):
            domain = _domain(source)
            accessed = f"접속 {self.accessed}" if self.accessed else NO_DATE
            title = self._titles.get(source) or self._titles_by_key.get(source_key(source))
            return f"{domain}({accessed}). *{title}*. {domain}, {source}" if title else f"{domain}({accessed}). {source}"
        return source  # 알 수 없는 형태는 그대로 보존한다

    def references(self) -> list[str]:
        return [f"[{i}] {self._format(s)}" for i, s in enumerate(self._order, start=1)]
