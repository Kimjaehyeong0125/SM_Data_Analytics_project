from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from scoringpkg.structured_extract import extract_structured_rows
from scoringpkg.aggregate import aggregate_total_score


VERSION = "1.0"


def _default_txt_dir() -> Path:
    """
    Default input directory when running with no CLI args (VSCode 'Run' convenience).

    Priority:
      1) ./censored_txt (inside project)
      2) ~/OneDrive/바탕 화면/censored_txt (common on Windows Korea OneDrive)
      3) ./data
    """
    here = Path(__file__).resolve().parent
    p1 = here / "censored_txt"
    if p1.exists():
        return p1

    home = Path.home()
    p2 = home / "OneDrive" / "바탕 화면" / "censored_txt"
    if p2.exists():
        return p2

    p3 = here / "data"
    return p3


def read_text_file(path: Path) -> str:
    # Try common encodings; OCR outputs vary.
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    # Last resort
    return path.read_text(encoding="utf-8", errors="ignore")


def list_txt_files(dir_path: Path) -> List[Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Input directory not found: {dir_path}")
    files = [p for p in dir_path.rglob("*.txt") if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def calc_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0,
            "trimmed_mean_10pct": 0.0,
        }

    vals = sorted(values)
    n = len(vals)
    vmin, vmax = vals[0], vals[-1]
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = math.sqrt(var)

    def percentile(p: float) -> float:
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] * (c - k) + vals[c] * (k - f)

    cut = int(math.floor(0.10 * n))
    trimmed = vals[cut:n - cut] if n - 2 * cut > 0 else vals
    trimmed_mean = sum(trimmed) / len(trimmed)

    return {
        "min": round(vmin, 6),
        "max": round(vmax, 6),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "p10": round(percentile(0.10), 6),
        "p50": round(percentile(0.50), 6),
        "p90": round(percentile(0.90), 6),
        "trimmed_mean_10pct": round(trimmed_mean, 6),
    }


def compute_unstructured_total(group_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Unstructured-only total score (i.e., academic group is NOT mixed with structured score).
    This matches the interpretation:
      - unstructured_score: pure keyword-based group aggregate
      - total_score: includes structured score via the configured mix in academic_achievement
    """
    total = 0.0
    for g, w in weights.items():
        total += float(w) * float(group_scores.get(g, 0.0))
    return max(0.0, min(100.0, total))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SMU pass probability scorer (batch on txt directory or single txt)."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", type=str, help="Single txt file path.")
    src.add_argument("--dir", type=str, help="Directory containing txt files (recursive).")

    parser.add_argument("--ratio", type=float, default=10.0, help="Competition ratio (e.g., 10 means 10:1). Default=10.")
    parser.add_argument("--grad-year", type=int, default=None, help="Graduation year (optional).")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory. Default=./output")
    parser.add_argument("--top-n", type=int, default=0, help="How many ranked lines to print. 0=all.")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print ranked list to console.")
    parser.add_argument("--no-print", dest="do_print", action="store_false", help="Do not print ranked list.")
    parser.set_defaults(do_print=True)

    # VSCode convenience: if user runs with no args, auto use default dir & ratio 10
    import sys
    if len(sys.argv) == 1:
        defaults_dir = _default_txt_dir()
        sys.argv.extend(["--dir", str(defaults_dir), "--ratio", "10"])

    args = parser.parse_args()

    if args.ratio <= 0:
        raise ValueError("--ratio must be > 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect inputs
    inputs: List[Path]
    source_label: str
    if args.input:
        p = Path(args.input)
        inputs = [p]
        source_label = str(p)
    else:
        dir_path = Path(args.dir) if args.dir else _default_txt_dir()
        inputs = list_txt_files(dir_path)
        source_label = str(dir_path)

    results = []
    for path in inputs:
        text = read_text_file(path)
        structured_rows, attendance = extract_structured_rows(text)

        score_detail = aggregate_total_score(
            unstructured_text=text,
            structured_rows=structured_rows,
            attendance=attendance,
            graduation_year=args.grad_year,
        )

        total_score = float(score_detail.get("final_score_0_to_100", 0.0))
        structured_score = float(score_detail.get("structured_score_0_to_100", 0.0))

        group_scores = score_detail.get("group_scores_0_to_100", {}) or {}
        weights = score_detail.get("weights", {}) or {}
        unstructured_score = compute_unstructured_total(group_scores, weights)

        results.append({
            "filename": path.name,
            "path": str(path),
            "total_score": total_score,
            "structured_score": structured_score,
            "unstructured_score": unstructured_score,
            "debug": {
                "structured_rows_found": len(structured_rows),
            },
        })

    # Rank
    results.sort(key=lambda r: (-r["total_score"], r["filename"]))

    total_files = len(results)
    accepted_count = min(total_files, int(math.ceil(total_files / args.ratio))) if total_files else 0
    cutoff_total_score = results[accepted_count - 1]["total_score"] if accepted_count > 0 else 0.0

    for i, r in enumerate(results, start=1):
        r["rank"] = i
        r["pass"] = (i <= accepted_count)

    cutoff_record = results[accepted_count - 1] if accepted_count > 0 else None

    # Stats
    stats = calc_stats([r["total_score"] for r in results])

    # Save outputs
    scores_csv = out_dir / "scores.csv"
    scores_json = out_dir / "scores.json"

    # CSV
    try:
        import csv
        with scores_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "filename", "total_score", "structured_score", "unstructured_score", "pass", "path", "structured_rows_found"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "rank": r["rank"],
                    "filename": r["filename"],
                    "total_score": round(r["total_score"], 6),
                    "structured_score": round(r["structured_score"], 6),
                    "unstructured_score": round(r["unstructured_score"], 6),
                    "pass": r["pass"],
                    "path": r["path"],
                    "structured_rows_found": r["debug"]["structured_rows_found"],
                })
    except Exception:
        # If CSV writing fails, continue; JSON is still written.
        pass

    summary = {
        "version": VERSION,
        "source": source_label,
        "total_files": total_files,
        "ratio": args.ratio,
        "accepted_count": accepted_count,
        "cutoff_total_score": round(float(cutoff_total_score), 6),
        "score_stats": stats,
        "cutoff_record": {
            "filename": cutoff_record["filename"],
            "total_score": round(float(cutoff_record["total_score"]), 6),
            "structured_score": round(float(cutoff_record["structured_score"]), 6),
            "unstructured_score": round(float(cutoff_record["unstructured_score"]), 6),
        } if cutoff_record else None,
        "outputs": {
            "scores_csv": str(scores_csv),
            "scores_json": str(scores_json),
        },
    }

    scores_json.write_text(json.dumps({
        "summary": summary,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.do_print and results:
        to_print = results if (args.top_n is None or args.top_n <= 0) else results[:args.top_n]
        print("\nRanked list (total_score):")
        for r in to_print:
            print(f"{r['rank']:02d}. {r['filename']}: {r['total_score']:.3f}  pass={str(r['pass'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
