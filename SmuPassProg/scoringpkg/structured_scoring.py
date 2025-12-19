from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


@dataclass(frozen=True)
class StructuredRow:
    subject: str
    grade: int  # 1~9
    unit: float = 1.0
    year: Optional[int] = None
    semester: Optional[int] = None


@dataclass(frozen=True)
class Attendance:
    absence: int = 0
    tardy: int = 0
    early_leave: int = 0
    etc: int = 0


class StructuredRules:
    def __init__(self, grade_points_default: Dict[int, float], year_overrides: Dict[int, Dict[int, float]],
                 attendance_penalty: Dict[str, float]):
        self.grade_points_default = grade_points_default
        self.year_overrides = year_overrides
        self.attendance_penalty = attendance_penalty

    @staticmethod
    def load(path: str | Path) -> "StructuredRules":
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)

        gp = obj.get("grade_points", {})
        default = {int(k): float(v) for k, v in gp.get("default", {}).items()}
        year_overrides_raw = obj.get("year_overrides", {}) or {}
        year_overrides = {int(year): {int(k): float(v) for k, v in mapping.items()}
                          for year, mapping in year_overrides_raw.items()}

        ap = obj.get("attendance_penalty", {}) or {}
        attendance_penalty = {
            "absence": float(ap.get("absence", 1.0)),
            "tardy": float(ap.get("tardy", 0.5)),
            "early_leave": float(ap.get("early_leave", 0.5)),
            "etc": float(ap.get("etc", 0.0)),
        }

        return StructuredRules(default, year_overrides, attendance_penalty)

    def grade_to_points(self, grade: int, graduation_year: Optional[int] = None) -> float:
        if graduation_year is not None and graduation_year in self.year_overrides:
            mapping = self.year_overrides[graduation_year]
            if grade in mapping:
                return mapping[grade]
        return self.grade_points_default.get(grade, 0.0)


def compute_structured_score_0_to_100(
    rows: List[StructuredRow],
    rules: StructuredRules,
    *,
    graduation_year: Optional[int] = None,
    attendance: Optional[Attendance] = None,
) -> float:
    """정형 점수(0~100): 등급→점수 환산 후 (점수 * 단위) 가중평균, 출결 페널티 차감.

    - A~E 성취도는 입력에서 제외하는 것으로 가정
    - 단위(unit)가 없다면 1.0으로 취급
    """

    if not rows:
        base_score = 0.0
    else:
        total_units = sum(max(0.0, r.unit) for r in rows)
        if total_units <= 0:
            total_units = float(len(rows))
            weighted = sum(rules.grade_to_points(r.grade, graduation_year) for r in rows)
            base_score = weighted / max(1.0, total_units)
        else:
            weighted = sum(rules.grade_to_points(r.grade, graduation_year) * max(0.0, r.unit) for r in rows)
            base_score = weighted / total_units

    penalty = 0.0
    if attendance is not None:
        penalty += attendance.absence * rules.attendance_penalty["absence"]
        penalty += attendance.tardy * rules.attendance_penalty["tardy"]
        penalty += attendance.early_leave * rules.attendance_penalty["early_leave"]
        penalty += attendance.etc * rules.attendance_penalty["etc"]

    final_score = max(0.0, min(100.0, base_score - penalty))
    return float(final_score)
