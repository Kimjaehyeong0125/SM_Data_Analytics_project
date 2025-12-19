from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import re
import yaml


LEVEL_POINTS = {"low": 1, "mid": 2, "high": 3}


@dataclass(frozen=True)
class CategoryScore:
    category: str
    matched_low: List[str]
    matched_mid: List[str]
    matched_high: List[str]
    points_0_to_6: int
    score_0_to_100: float


class KeywordRubric:
    """Substring 기반 키워드 매칭 루브릭."""

    def __init__(self, rubric: Dict[str, Dict[str, List[str]]], *, case_insensitive: bool = True):
        self.rubric = rubric
        self.case_insensitive = case_insensitive

        # keyword -> compiled regex (escape 처리)
        flags = re.IGNORECASE if case_insensitive else 0
        self._compiled: Dict[Tuple[str, str, str], re.Pattern] = {}
        for cat, levels in rubric.items():
            for level_name in ("low", "mid", "high"):
                for kw in levels.get(level_name, []):
            
                    pat = re.compile(re.escape(kw), flags)
                    self._compiled[(cat, level_name, kw)] = pat

    @staticmethod
    def load(path: str | Path) -> "KeywordRubric":
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
        match_cfg = obj.get("match", {})
        case_insensitive = bool(match_cfg.get("case_insensitive", True))
        rubric = obj.get("rubric", {})
        return KeywordRubric(rubric, case_insensitive=case_insensitive)


def _is_likely_keyword_dump_line(line: str) -> bool:
    """키워드 나열(예: 'A | B | C')로 보이는 라인을 점수화에서 제외하기 위한 휴리스틱."""
    s = line.strip()
    if not s:
        return True

    # 구분자 과다
    if s.count("|") >= 3:
        return True

    # 쉼표/슬래시가 과다한 "나열" 패턴
    if s.count(",") >= 6:
        return True

    # 너무 짧은 토큰들이 반복되는 경우
    tokens = re.split(r"[|,/]", s)
    if len(tokens) >= 10 and sum(1 for t in tokens if len(t.strip()) <= 4) >= 8:
        return True

    return False


def score_text(text: str, rubric: KeywordRubric) -> Dict[str, CategoryScore]:
    """각 카테고리별로 low/mid/high 검출 여부를 기반으로 0~6점(=1+2+3) 부여."""

    # 라인 단위로 점수화 제외 휴리스틱 적용
    lines = [ln for ln in text.splitlines() if not _is_likely_keyword_dump_line(ln)]
    filtered_text = "\n".join(lines)

    result: Dict[str, CategoryScore] = {}
    for category, levels in rubric.rubric.items():
        matched: Dict[str, Set[str]] = {"low": set(), "mid": set(), "high": set()}

        for level_name in ("low", "mid", "high"):
            for kw in levels.get(level_name, []):
                pat = rubric._compiled.get((category, level_name, kw))
                if pat and pat.search(filtered_text):
                    matched[level_name].add(kw)

        points = sum(LEVEL_POINTS[level] for level in ("low", "mid", "high") if matched[level])
        score_100 = (points / 6.0) * 100.0 if points > 0 else 0.0

        result[category] = CategoryScore(
            category=category,
            matched_low=sorted(matched["low"]),
            matched_mid=sorted(matched["mid"]),
            matched_high=sorted(matched["high"]),
            points_0_to_6=points,
            score_0_to_100=score_100,
        )

    return result


def total_unstructured_score_0_to_100(category_scores: Dict[str, CategoryScore]) -> float:
    """10개 카테고리 합(최대 60점)을 100점으로 환산."""
    total_points = sum(cs.points_0_to_6 for cs in category_scores.values())
    return (total_points / 60.0) * 100.0
