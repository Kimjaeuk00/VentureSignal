import math
import re
from datetime import datetime

from .settings import MISSING_CATEGORY_TEXT, SUMMARY_CATEGORY_ORDER


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def valid_evidence(evidence, sources) -> bool:
    return bool(evidence) and all(
        item.source_id in sources
        and len(normalized(item.quote)) >= 6
        and normalized(item.quote) in normalized(sources[item.source_id]["content"])
        for item in evidence
    )


def profile_sections(text: str) -> dict[str, str]:
    result = {}
    headings = {
        "experience": r"experience|경력",
        "education": r"education|학력",
        "skills": r"skills|보유기술|기술",
        "publications": r"publications|논문",
        "activity": r"activity|활동|posts|게시물",
    }
    for key, names in headings.items():
        match = re.search(rf"(?im)^\s*(?:#+\s*)?(?:{names})\s*:?[ \t]*$", text)
        if not match:
            result[key] = "not_returned"
            continue
        tail = text[match.end():]
        next_heading = re.search(
            r"(?im)^\s*(?:#{1,4}\s+|[가-힣A-Za-z][^\n]{0,40}\n(?:[-=]{3,})$)", tail
        )
        section = (tail[:next_heading.start()] if next_heading else tail[:1200]).strip()
        result[key] = "empty_or_NA" if not section or re.match(r"(?i)^N/?A\b", section) else "returned"
    return result


def estimate_runway(facts, company, as_of):
    """동일 기준일·단위의 검증된 현금/월 순소진액만 사용한다."""
    cash = [fact for fact in facts if fact.subject == company and fact.category == "cash"]
    burn = [fact for fact in facts if fact.subject == company and fact.category == "burn"]
    if len(cash) != 1 or len(burn) != 1:
        return None
    cash_attributes, burn_attributes = cash[0].attributes, burn[0].attributes
    try:
        amount = float(cash_attributes["value"])
        rate = float(burn_attributes["value"])
        age = (datetime.fromisoformat(as_of) - datetime.fromisoformat(cash_attributes["as_of"])).days
        if not (
            math.isfinite(amount)
            and math.isfinite(rate)
            and amount >= 0
            and rate > 0
            and 0 <= age <= 180
            and cash_attributes["unit"]
            and cash_attributes["unit"] == burn_attributes["unit"]
            and cash_attributes["as_of"] == burn_attributes["as_of"]
            and burn_attributes["period"] == "month"
        ):
            return None
        return {
            "months": round(amount / rate, 2),
            "as_of": cash_attributes["as_of"],
            "formula": "cash / monthly net burn",
            "fact_ids": [cash[0].id, burn[0].id],
            "note": "기준일 단순 추정. 이후 현금 변동과 향후 지출 미반영.",
        }
    except (KeyError, ValueError, TypeError):
        return None


def build_verified_fact_summary(section, facts, target_subjects, missing_categories):
    """승인된 대상 Fact를 생략 없이 category 순서로 연결하는 결정론적 fallback."""
    order = {category: index for index, category in enumerate(SUMMARY_CATEGORY_ORDER[section])}
    selected = sorted(
        (fact for fact in facts if fact.subject in target_subjects and fact.category in order),
        key=lambda fact: order[fact.category],
    )
    texts, fact_ids, seen = [], [], set()
    for fact in selected:
        key = normalized(fact.text)
        if not key or key in seen:
            continue
        seen.add(key)
        texts.append(key)
        fact_ids.append(fact.id)

    if not texts:
        return "확인 가능한 검증 사실을 확보하지 못했습니다.", []

    missing = [
        MISSING_CATEGORY_TEXT[category]
        for category in missing_categories
        if category in MISSING_CATEGORY_TEXT
    ]
    if missing:
        texts.append(f"공개 자료에서 {', '.join(missing)}은 확인되지 않았습니다.")
    return " ".join(texts), fact_ids
