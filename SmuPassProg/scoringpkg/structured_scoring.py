from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# =========================
# Data models
# =========================

@dataclass(frozen=True)
class StructuredRow:
    subject: str
    grade: int                     # 1~9 석차등급
    achievement: Optional[str] = None  # A~E (D/E는 C로 처리)
    unit: float = 1.0
    year: Optional[int] = None
    semester: Optional[int] = None


@dataclass(frozen=True)
class Attendance:
    absence: int = 0
    tardy: int = 0
    early_leave: int = 0
    etc: int = 0


# =========================
# Rules
# =========================

class StructuredRules:
    def __init__(
        self,
        grade_points_default: Dict[int, float],
        year_overrides: Dict[int, Dict[int, float]],
        attendance_penalty: Dict[str, float],
        achievement_points: Dict[str, float],
    ):
        self.grade_points_default = grade_points_default
        self.year_overrides = year_overrides
        self.attendance_penalty = attendance_penalty
        self.achievement_points = achievement_points

    @staticmethod
    def load(path: str | Path) -> "StructuredRules":
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)

        # 1~9 등급 점수
        gp = obj.get("grade_points", {}) or {}
        default = {int(k): float(v) for k, v in (gp.get("default", {}) or {}).items()}

        year_overrides_raw = obj.get("year_overrides", {}) or {}
        year_overrides = {
            int(year): {int(k): float(v) for k, v in mapping.items()}
            for year, mapping in year_overrides_raw.items()
        }

        # 출결 패널티
        ap = obj.get("attendance_penalty", {}) or {}
        attendance_penalty = {
            "absence": float(ap.get("absence", 1.0)),
            "tardy": float(ap.get("tardy", 0.5)),
            "early_leave": float(ap.get("early_leave", 0.5)),
            "etc": float(ap.get("etc", 0.0)),
        }

        # A~C 성취도 점수 (D/E는 C로 처리)
        ach = obj.get("achievement_points", {}) or {}
        achievement_points = {str(k).upper(): float(v) for k, v in ach.items()}

        return StructuredRules(default, year_overrides, attendance_penalty, achievement_points)

    def grade_to_points(self, grade: int, graduation_year: Optional[int] = None) -> float:
        if graduation_year is not None and graduation_year in self.year_overrides:
            mapping = self.year_overrides[graduation_year]
            if grade in mapping:
                return mapping[grade]
        return self.grade_points_default.get(grade, 0.0)

    def achievement_to_points(self, achievement: Optional[str]) -> Optional[float]:
        """
        A/B/C → 점수
        D/E/기타 → C 점수로 처리
        None → 성취도 미사용(=석차등급만 반영)
        """
        if achievement is None:
            return None

        a = achievement.upper()
        if a in self.achievement_points:
            return self.achievement_points[a]

        # D/E 등은 C와 동일 처리
        return self.achievement_points.get("C")


# =========================
# Scoring
# =========================

def compute_structured_score_0_to_100(
    rows: List[StructuredRow],
    rules: StructuredRules,
    *,
    graduation_year: Optional[int] = None,
    attendance: Optional[Attendance] = None,
) -> float:
    """
    정형 점수(0~100):
    - (1~9 석차등급 점수 + A~C 성취도 점수) 를 5:5로 결합
    - 단위(unit) 가중평균
    - 출결 패널티 차감
    """
    if not rows:
        base_score = 0.0
    else:
        total_units = sum(max(0.0, r.unit) for r in rows)
        if total_units <= 0:
            total_units = float(len(rows))

        weighted_sum = 0.0
        for r in rows:
            rank_score = rules.grade_to_points(r.grade, graduation_year)
            ach_score = rules.achievement_to_points(r.achievement)

            if ach_score is not None:
                combined_score = 0.5 * rank_score + 0.5 * ach_score
            else:
                combined_score = rank_score

            weighted_sum += combined_score * max(0.0, r.unit)

        base_score = weighted_sum / total_units

    # 출결 패널티
    penalty = 0.0
    if attendance is not None:
        penalty += attendance.absence * rules.attendance_penalty["absence"]
        penalty += attendance.tardy * rules.attendance_penalty["tardy"]
        penalty += attendance.early_leave * rules.attendance_penalty["early_leave"]
        penalty += attendance.etc * rules.attendance_penalty["etc"]

    return float(max(0.0, min(100.0, base_score - penalty)))
