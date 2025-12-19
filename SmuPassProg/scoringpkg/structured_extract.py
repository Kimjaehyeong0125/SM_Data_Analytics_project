from __future__ import annotations

from typing import List, Tuple
import re

from .structured_scoring import StructuredRow, Attendance


# ---------------------------------------------------------------------------
# OCR-robust structured extraction
# ---------------------------------------------------------------------------
# Target pattern (common in 생기부 OCR):
#   ... 석차등급 ... 5(344) ... 8(180) ...
# where:
#   - "5" is 석차등급(1~9)
#   - "(344)" is 수강자수, typically digits-only
#
# We intentionally exclude patterns such as:
#   - 78.2(10.8)  (decimal in parentheses)
#   - (4위), (319명) (non-digits in parentheses)
# ---------------------------------------------------------------------------

_GRADE_TOKEN_RE = re.compile(r"(?<!\d)([1-9])\s*\(\s*([0-9]{2,4})\s*\)")
# We don't currently use the student-count number, but keeping it helps validate the match.


def extract_structured_rows(text: str) -> Tuple[List[StructuredRow], Attendance]:
    """
    Extract structured rows + attendance from a single OCR txt.

    Returns:
        (rows, attendance)
    """
    # 1) Attendance (best effort)
    # Many OCR outputs do not keep the table layout. We support:
    # - "개근" -> all zeros
    # - else: try to find a compact block and pick first 4 integers as
    #   (absence, tardy, early_leave, etc). If not found -> zeros.
    if "개근" in text:
        attendance = Attendance()
    else:
        window = text
        m = re.search(r"출\s*결\s*상\s*항(.{0,600})", text)
        if m:
            window = m.group(1)
        nums = [int(x) for x in re.findall(r"\b\d+\b", window)]
        if len(nums) >= 4:
            attendance = Attendance(absence=nums[0], tardy=nums[1], early_leave=nums[2], etc=nums[3])
        else:
            attendance = Attendance()

    # 2) Grade tokens -> StructuredRow list
    rows: List[StructuredRow] = []
    for idx, m in enumerate(_GRADE_TOKEN_RE.finditer(text), start=1):
        grade = int(m.group(1))
        rows.append(
            StructuredRow(
                subject=f"COURSE_{idx:02d}",
                grade=grade,
                unit=1.0,
                year=None,
                semester=None,
            )
        )

    return rows, attendance
