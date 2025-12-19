from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .keyword_scoring import KeywordRubric, score_text
from .structured_scoring import (
    StructuredRow,
    Attendance,
    StructuredRules,
    compute_structured_score_0_to_100,
)


DEFAULT_KEYWORD_RULES = Path(__file__).resolve().parents[1] / "config" / "keyword_rules.yaml"
DEFAULT_SCORE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "score_config.yaml"
DEFAULT_STRUCTURED_RULES = Path(__file__).resolve().parents[1] / "config" / "structured_rules.yaml"


def _load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _category_points_0_to_6(category_score: Dict[str, Any]) -> int:
    return int(category_score.get("points_0_to_6", 0))


def _group_score_0_to_100(category_scores: Dict[str, Dict[str, Any]], categories: List[str]) -> float:
    if not categories:
        return 0.0
    total = 0
    for c in categories:
        total += _category_points_0_to_6(category_scores.get(c, {}))
    max_points = 6 * len(categories)
    return (total / max_points) * 100.0


def aggregate_total_score(
    *,
    unstructured_text: str,
    structured_rows: Optional[List[StructuredRow]] = None,
    structured_score_0_to_100: Optional[float] = None,
    attendance: Optional[Attendance] = None,
    graduation_year: Optional[int] = None,
    keyword_rules_path: Union[str, Path] = DEFAULT_KEYWORD_RULES,
    score_config_path: Union[str, Path] = DEFAULT_SCORE_CONFIG,
    structured_rules_path: Union[str, Path] = DEFAULT_STRUCTURED_RULES,
) -> Dict[str, Any]:
    """총점(0~100) 산출.

    - 비정형: keyword_rules.yaml 기반 (카테고리별 0~6점)
    - 정형: 1~9등급 → 점수 변환 후 단위수 가중 평균(0~100), 출결 패널티 차감
    - 학업성취도 그룹: (정형 50% + 비정형 학업성취도 50%) 혼합
    - 나머지 그룹: 비정형만 사용

    structured_score_0_to_100을 직접 넣으면 structured_rows/attendance는 무시 가능.
    """

    # 입력 유연성: dict → dataclass 변환 허용
    if structured_rows is not None:
        structured_rows = [
            StructuredRow(**r) if isinstance(r, dict) else r
            for r in structured_rows
        ]
    if attendance is not None and isinstance(attendance, dict):
        attendance = Attendance(**attendance)

    # 1) 비정형 점수
    rubric = KeywordRubric.load(keyword_rules_path)
    category_scores = score_text(unstructured_text, rubric)
    category_scores_dict = {k: asdict(v) for k, v in category_scores.items()}

    # 2) 정형 점수
    if structured_score_0_to_100 is None:
        if structured_rows is None:
            structured_score_0_to_100 = 0.0
        else:
            rules = StructuredRules.load(structured_rules_path)
            structured_score_0_to_100 = compute_structured_score_0_to_100(
                structured_rows, rules=rules, attendance=attendance, graduation_year=graduation_year
            )

    # 3) 그룹/가중치
    cfg = _load_yaml(score_config_path)
    weights = cfg.get("weights", {})
    groups = cfg.get("groups", {})
    mix_cfg = (cfg.get("structured_unstructured_mix") or {}).get("academic_achievement") or {
        "structured": 0.5,
        "unstructured": 0.5,
    }

    # 각 그룹 점수(0~100)
    group_scores: Dict[str, float] = {}
    for group_name, group_def in groups.items():
        cats = group_def.get("categories", [])
        group_scores[group_name] = _group_score_0_to_100(category_scores_dict, cats)

    # 학업성취도는 혼합
    unstructured_academic = group_scores.get("academic_achievement", 0.0)
    mixed_academic = (
        float(mix_cfg.get("structured", 0.5)) * float(structured_score_0_to_100)
        + float(mix_cfg.get("unstructured", 0.5)) * float(unstructured_academic)
    )
    group_scores["academic_achievement_mixed"] = mixed_academic

    # 최종 총점
    final = 0.0
    final += float(weights.get("academic_achievement", 0.0)) * mixed_academic
    final += float(weights.get("academic_attitude_inquiry", 0.0)) * group_scores.get("academic_attitude_inquiry", 0.0)
    final += float(weights.get("career_exploration", 0.0)) * group_scores.get("career_exploration", 0.0)
    final += float(weights.get("self_directed_major", 0.0)) * group_scores.get("self_directed_major", 0.0)
    final += float(weights.get("diligence_collaboration", 0.0)) * group_scores.get("diligence_collaboration", 0.0)

    final = max(0.0, min(100.0, final))

    unstructured_serializable = category_scores_dict

    return {
        "final_score_0_to_100": round(final, 3),
        "structured_score_0_to_100": round(float(structured_score_0_to_100), 3),
        "unstructured_categories": unstructured_serializable,
        "group_scores_0_to_100": {k: round(v, 3) for k, v in group_scores.items()},
        "weights": weights,
    }
