"""출처 표기 공용 모듈 — LLM 이 URL 을 직접 쓰지 않게 하고, State 에는 표준 URL 만 남긴다.

LLM 은 긴 URL 을 글자 그대로 베끼지 못하고 가끔 한 조각을 반복하거나 바꾼다(예: embeddedvisionsummit → embeddedvisionsionsummit).
그래서 프롬프트에는 URL 대신 짧은 라벨(`U1`, `U2` …)을 보여 주고 LLM 은 라벨만 인용하게 한다. 에이전트 출구에서 이 모듈이
라벨을 표준 URL 로 되돌리고, 목록에 없는 라벨은 근거에서 빼서(예외 없이) 기록한다.

- 수집하는 에이전트(웹 검색 결과를 받는 쪽): `SourceBook(canonicalize=True)` — 검색 결과를 표준 URL 로 정리해 라벨을 붙인다.
- 읽기만 하는 에이전트(앞 노드의 State 를 채점·요약하는 쪽): `SourceBook()` — State 의 URL 문자열을 그대로 유지하고
  `mask` 로 프롬프트용 텍스트의 URL 을 라벨로 바꾼다.
- RAG 출처 ID(`S006`)는 90개짜리 닫힌 집합이라 그대로 쓴다. `rag_ids` 에 든 것만 유효하다.

State 의 evidence 는 계속 URL(또는 S번호) 문자열이라 뒤 노드와 보고서 코드는 그대로 동작한다.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "viewasmember", "igshid"}
URL_RE = re.compile(r"https?://[^\s<>\"'\[\]{}]+")
TRAILING = ".,;:!?)"
LABEL_GROUP_RE = re.compile(r"[\[(]\s*U0*\d+(?:\s*[,;/·]\s*U?0*\d+)*\s*[\])]", re.IGNORECASE)
LABEL_RE = re.compile(r"(?<![A-Za-z0-9])U0*(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
RAG_ID_RE = re.compile(r"^S\d{3}$")


def _is_web(url: str) -> bool:
    return isinstance(url, str) and url.strip().lower().startswith(("http://", "https://"))


def canonical_url(url: str) -> str:
    """가져올 수 있는 형태는 유지하면서 같은 페이지의 표기 차이를 줄인다: 조각(#)·추적 파라미터·끝 슬래시 제거, 호스트 소문자."""
    url = (url or "").strip()
    if not _is_web(url):
        return url
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(query), ""))


def source_key(url: str) -> str:
    """같은 페이지인지 판단하는 키 (스킴·www·끝 슬래시·추적 파라미터 차이는 무시)."""
    if not _is_web(url):
        return (url or "").strip()
    parts = urlsplit(canonical_url(url))
    return f"{parts.netloc.removeprefix('www.')}{parts.path}" + (f"?{parts.query}" if parts.query else "")


def strip_trailing(token: str) -> str:
    """본문에서 URL 뒤에 딸려 온 문장부호를 뗀다."""
    return token.rstrip(TRAILING)


@dataclass
class Source:
    label: str
    url: str
    title: str = ""
    content: str = ""


@dataclass
class SourceBook:
    """한 에이전트 실행 안에서 출처 라벨(U1…)과 URL 을 오가게 하는 장부."""

    canonicalize: bool = False
    rag_ids: set[str] = field(default_factory=set)
    _by_key: dict[str, Source] = field(default_factory=dict)
    _by_label: dict[str, Source] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)  # 목록에 없어 버린 라벨·URL (진단·기록용)

    # --- 등록 ---------------------------------------------------------------------------

    def add_url(self, url: str, title: str = "", content: str = "") -> Optional[Source]:
        if not _is_web(url):
            return None
        key = source_key(url)
        found = self._by_key.get(key)
        if found is None:
            found = Source(f"U{len(self._by_key) + 1}", canonical_url(url) if self.canonicalize else url.strip(), title, content)
            self._by_key[key] = found
            self._by_label[found.label] = found
        else:  # 같은 페이지를 다시 만났다: 비어 있던 제목·본문만 채운다
            found.title = found.title or title
            found.content = found.content or content
        return found

    def add(self, results: Iterable[dict]) -> list[Source]:
        """웹 검색 결과({url, title, content})를 등록하고 (중복 제거한) Source 목록을 돌려준다."""
        out: list[Source] = []
        for r in results:
            src = self.add_url(r.get("url", ""), r.get("title", ""), r.get("content", ""))
            if src and src not in out:
                out.append(src)
        return out

    @property
    def sources(self) -> list[Source]:
        return list(self._by_label.values())

    def urls(self) -> set[str]:
        return {s.url for s in self._by_label.values()}

    def label_of(self, url: str) -> Optional[str]:
        src = self._by_key.get(source_key(url))
        return src.label if src else None

    # --- 프롬프트용 ---------------------------------------------------------------------

    def render(self, source: Source, limit: int = 0) -> str:
        """LLM 에게 보여 줄 한 건. URL 은 넣지 않는다."""
        body = source.content[:limit] if limit else source.content
        return f"[{source.label}] {source.title}\n{body}".strip()

    def mask(self, text: str) -> str:
        """텍스트 안의 URL 을 라벨로 바꾼다(처음 보는 URL 은 등록). 읽기 전용 에이전트가 앞 노드 결과를 프롬프트에 넣을 때 쓴다."""
        def replace(m: re.Match) -> str:
            token = m.group(0)
            core = strip_trailing(token)
            src = self.add_url(core)
            return f"[{src.label}]" + token[len(core):] if src else token

        return URL_RE.sub(replace, text)

    # --- 복원 (LLM 출력 → State) -----------------------------------------------------------

    def _label(self, raw: str) -> Optional[str]:
        m = LABEL_RE.fullmatch(raw.strip().strip("[]() "))
        return f"U{int(m.group(1))}" if m else None

    def resolve(self, ids: Iterable[str]) -> list[str]:
        """evidence 목록의 라벨·RAG ID·URL 을 State 에 넣을 문자열로. 모르는 것은 버리고 unknown 에 기록한다."""
        out: list[str] = []
        for raw in ids:
            raw = (raw or "").strip()
            if not raw:
                continue
            label = self._label(raw)
            if label and label in self._by_label:
                value = self._by_label[label].url
            elif RAG_ID_RE.match(raw.strip("[] ")) and raw.strip("[] ") in self.rag_ids:
                value = raw.strip("[] ")
            elif _is_web(raw) and source_key(strip_trailing(raw)) in self._by_key:  # LLM 이 URL 을 썼더라도 목록에 있는 페이지면 인정
                value = self._by_key[source_key(strip_trailing(raw))].url
            else:
                self.unknown.append(raw)
                continue
            if value not in out:
                out.append(value)
        return out

    def restore(self, text: str) -> str:
        """본문의 [U1]·(U2)·U3 표기를 `[URL]` 로 되돌린다. 목록에 없는 라벨과 URL 은 지운다(unknown 에 기록)."""
        def group(m: re.Match) -> str:
            urls = self.resolve(re.findall(r"U?0*\d+", m.group(0).strip("[]() "), flags=re.IGNORECASE))
            return " ".join(f"[{u}]" for u in urls)

        def bare(m: re.Match) -> str:
            urls = self.resolve([m.group(0)])
            return f"[{urls[0]}]" if urls else ""

        def raw_url(m: re.Match) -> str:
            # LLM 이 라벨 대신 URL 을 써 버린 경우: 장부에 있는 페이지면 표준 URL 로, 없으면 지운다(오탈자·지어낸 URL 차단)
            token = m.group(0)
            core = strip_trailing(token)
            found = self._by_key.get(source_key(core))
            if found:
                return found.url + token[len(core):]
            self.unknown.append(core)
            return token[len(core):]

        text = LABEL_GROUP_RE.sub(group, text or "")
        text = URL_RE.sub(raw_url, LABEL_RE.sub(bare, text))
        text = re.sub(r"\[\s*\]|\(\s*\)", "", text)  # 지운 인용이 남긴 빈 괄호
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    def unknown_labels_in(self, text: str) -> list[str]:
        """restore 하기 전에, 본문에 목록에 없는 라벨이 있는지 미리 본다."""
        return sorted({f"U{int(n)}" for n in LABEL_RE.findall(text or "") if f"U{int(n)}" not in self._by_label})

    # --- State 에 남길 메타데이터 --------------------------------------------------------------

    def registry(self, used: Optional[Iterable[str]] = None) -> dict[str, dict]:
        """{URL: {title, retrieved_at}} — 보고서 REFERENCE 가 제목을 쓰도록 State 의 sources 에 병합한다. used 가 있으면 그 URL 만."""
        wanted = set(used) if used is not None else None
        today = date.today().isoformat()
        return {s.url: {"title": s.title, "retrieved_at": today}
                for s in self._by_label.values() if wanted is None or s.url in wanted}
