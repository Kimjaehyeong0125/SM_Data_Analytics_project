from __future__ import annotations

from typing import List, Tuple
import re

from .structured_scoring import StructuredRow, Attendance


# 1~9 석차등급 (예: 3(251))
_GRADE_TOKEN_RE = re.compile(r"(?<!\d)([1-9])\s*\(\s*([0-9]{2,4})\s*\)")

# 성취도 A~E (예: A(537), B(251), D(123), E(999))
# - D/E는 structured_scoring에서 C와 동일 점수 처리됨
_ACHIEVEMENT_RE = re.compile(r"\b([A-E])\s*\(", re.IGNORECASE)


def extract_structured_rows(text: str) -> Tuple[List[StructuredRow], Attendance]:
    """
    Extract structured rows + attendance from a single OCR txt.

    Returns:
        (rows, attendance)
    """

    # =========================
    # 1) Attendance (기존 그대로)
    # =========================
    def _find_attendance_block(t: str) -> str:
        m1 = re.search(r"결\s*석\s*일\s*수(.{0,1400})", t, flags=re.S)
        if m1:
            return "결석일수" + m1.group(1)

        m2 = re.search(r"출\s*\S?\s*상\s*항(.{0,1400})", t, flags=re.S)
        if m2:
            return m2.group(0)

        return ""

    def _sum_small_ints(seg: str) -> int:
        nums = [int(x) for x in re.findall(r"\b\d{1,3}\b", seg)]
        nums = [n for n in nums if 0 <= n <= 60]
        return int(sum(nums)) if nums else 0

    block = _find_attendance_block(text)

    if block and ("개근" in block):
        attendance = Attendance()
    elif block:
        def _seg(start_kw: str, end_kw: str | None) -> str:
            s = block.find(start_kw)
            if s == -1:
                return ""
            s += len(start_kw)
            e = block.find(end_kw, s) if end_kw else -1
            if e == -1:
                e = len(block)
            return block[s:e]

        abs_seg = _seg("결석", "지각") or _seg("결석일수", "지각")
        tardy_seg = _seg("지각", "조퇴")
        early_seg = _seg("조퇴", "결과")
        etc_seg = _seg("결과", "수업일수") or _seg("결과", None)

        attendance = Attendance(
            absence=_sum_small_ints(abs_seg),
            tardy=_sum_small_ints(tardy_seg),
            early_leave=_sum_small_ints(early_seg),
            etc=_sum_small_ints(etc_seg),
        )
    else:
        attendance = Attendance()

    # =========================
    # 2) 과목 성적 파싱
    # =========================
    rows: List[StructuredRow] = []

    grade_matches = list(_GRADE_TOKEN_RE.finditer(text))
    achievement_matches = list(_ACHIEVEMENT_RE.finditer(text))

    for idx, m in enumerate(grade_matches, start=1):
        grade = int(m.group(1))

        # 성취도: 순서 기반 매칭 (없으면 None)
        achievement = None
        if idx - 1 < len(achievement_matches):
            achievement = achievement_matches[idx - 1].group(1).upper()

        rows.append(
            StructuredRow(
                subject=f"COURSE_{idx:02d}",
                grade=grade,                 # 1~9 석차등급
                achievement=achievement,     # A~E (D/E는 C로 처리됨)
                unit=1.0,
                year=None,
                semester=None,
            )
        )

    return rows, attendance
