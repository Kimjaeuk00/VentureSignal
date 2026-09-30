import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# =========================
# LLM
# =========================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")

# =========================
# RAG
# =========================

RESEARCH_PDF = ROOT_DIR / "data" / "AI_Chip_Startup_RAG_Research_150p.pdf"

# python -m rag.ingest 로 생성되는 파일 (git 미추적)
INDEX_DIR = ROOT_DIR / "data" / "index"
QDRANT_PATH = INDEX_DIR / "qdrant"
SOURCES_PATH = INDEX_DIR / "sources.json"
QDRANT_COLLECTION = "ai_chip_research"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# =========================
# Graph
# =========================

TOP_K_CANDIDATES = 3
OUTPUT_DIR = ROOT_DIR / "outputs"

# =========================
# Scorecard (설계 문서 C절)
# =========================

SCORECARD_WEIGHTS = {
    "technology": 0.35,
    "competition": 0.25,
    "market": 0.15,
    "team": 0.10,
    "traction": 0.10,
    "deal_terms": 0.05,
}

# 항목별 점수 범위 0 ~ SCORE_MAX, total_score 는 0 ~ 100 으로 환산
SCORE_MAX = 5

# TODO: 설계 문서에 미정 — 팀 합의 후 확정
INVEST_THRESHOLD = 70.0
