from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from scoringpkg.structured_extract import extract_structured_rows
from scoringpkg.aggregate import aggregate_total_score
from scoringpkg.pca import run_pca_and_save_artifacts

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
    """
    total = 0.0
    for g, w in weights.items():
        total += float(w) * float(group_scores.get(g, 0.0))
    return max(0.0, min(100.0, total))


def _extract_category_scores_for_pca(score_detail: Dict[str, Any]) -> Dict[str, float]:
    """
    PCA에 쓸 '비교과 10개 항목(키워드 점수)'를 results에 넣기 위한 함수.
    aggregate_total_score가 주는 unstructured_categories를 0~100으로 뽑아낸다.
    """
    categories = score_detail.get("unstructured_categories", {}) or {}
    category_scores: Dict[str, float] = {}
    for k, v in categories.items():
        if isinstance(v, dict):
            category_scores[k] = float(v.get("score_0_to_100", 0.0))
        else:
            category_scores[k] = 0.0
    return category_scores


def _choose_txt_file_gui() -> Optional[Path]:
    """
    Windows에서 파일 선택창으로 txt를 고르게 함.
    tkinter 없으면 None 반환.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

        file_path = filedialog.askopenfilename(
            title="임의 입력 TXT 파일을 선택하세요 (취소하면 스킵)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        try:
            root.destroy()
        except Exception:
            pass

        if not file_path:
            return None
        return Path(file_path)
    except Exception:
        return None


def _choose_txt_file_fallback() -> Optional[Path]:
    """
    GUI가 안 되면, 콘솔에 드래그&드롭(경로 붙여넣기)로 받음.
    """
    try:
        s = input("\n[INPUT] 임의 txt 파일을 드래그해서 붙여넣고 Enter (그냥 Enter=스킵): ").strip().strip('"')
        if not s:
            return None
        p = Path(s)
        return p
    except Exception:
        return None


def _student_type(pc1: float, pc2: float) -> str:
    if pc1 >= 0 and pc2 < 0:
        return "전공역량형"
    if pc1 >= 0 and pc2 >= 0:
        return "균형형"
    if pc1 < 0 and pc2 >= 0:
        return "공동체형"
    return "발전필요형"


def _project_one_with_saved_pca_model(
    model_path: Path,
    category_scores: Dict[str, float],
) -> Dict[str, Any]:
    """
    output/pca_model.json을 이용해서 임의 1개 샘플을 PC1/PC2로 투영.
    """
    import numpy as np

    model = json.loads(model_path.read_text(encoding="utf-8"))
    feature_names = model["feature_names"]
    mean_ = np.array(model["scaler"]["mean_"], dtype=float)
    scale_ = np.array(model["scaler"]["scale_"], dtype=float)
    comps = np.array(model["pca"]["components_"], dtype=float)  # (2, F)

    x = np.array([float(category_scores.get(k, 0.0)) for k in feature_names], dtype=float)

    safe_scale = scale_.copy()
    safe_scale[safe_scale == 0] = 1.0
    xs = (x - mean_) / safe_scale

    z = comps @ xs  # (2,)
    pc1 = float(z[0])
    pc2 = float(z[1])

    pc1_cut = float(model.get("pc1_cutoff", 0.0))
    centroid = np.array(model.get("pass_centroid", [0.0, 0.0]), dtype=float)
    dist = float(np.linalg.norm(np.array([pc1, pc2], dtype=float) - centroid))

    return {
        "pc1": pc1,
        "pc2": pc2,
        "pc1_cutoff": pc1_cut,
        "pca_pass_pc1": bool(pc1 >= pc1_cut),
        "student_type": _student_type(pc1, pc2),
        "dist_to_pass_centroid": dist,
        "explained_variance_ratio": model.get("pca", {}).get("explained_variance_ratio_", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SMU pass probability scorer (batch on txt directory or single txt)."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", type=str, help="Single txt file path.")
    src.add_argument("--dir", type=str, help="Directory containing txt files (recursive).")

    parser.add_argument("--ratio", type=float, default=10.0, help="Competition ratio (e.g., 10 means 10:1). Default=10.")
    parser.add_argument("--trim-pct", type=float, default=0.10,
                        help="Trim fraction applied BEFORE pass decision (default: 0.10).")
    parser.add_argument("--trim-mode", type=str, default="both", choices=["none", "bottom", "both"],
                        help="Where to trim before pass decision: none|bottom|both (default: both).")
    parser.add_argument("--grad-year", type=int, default=None, help="Graduation year (optional).")
    parser.add_argument("--out-dir", type=str, default="output", help="Output directory. Default=./output")
    parser.add_argument("--top-n", type=int, default=0, help="How many ranked lines to print. 0=all.")
    parser.add_argument("--print", dest="do_print", action="store_true", help="Print ranked list to console.")
    parser.add_argument("--no-print", dest="do_print", action="store_false", help="Do not print ranked list.")
    parser.add_argument("--no-open", dest="open_outdir", action="store_false", help="Do not open output folder.")
    parser.set_defaults(do_print=True, open_outdir=True)

    # VSCode convenience: if user runs with no args, auto use default dir & ratio 10
    import sys
    if len(sys.argv) == 1:
        defaults_dir = _default_txt_dir()
        sys.argv.extend(["--dir", str(defaults_dir), "--ratio", "10"])

    args = parser.parse_args()

    if args.ratio <= 0:
        raise ValueError("--ratio must be > 0")

    # IMPORTANT: out_dir를 '현재 작업폴더'가 아니라 '프로젝트(main.py 위치)' 기준으로 고정
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = here / out_dir
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

        category_scores = _extract_category_scores_for_pca(score_detail)

        results.append({
            "filename": path.name,
            "path": str(path),
            "total_score": total_score,
            "structured_score": structured_score,
            "unstructured_score": unstructured_score,
            "group_scores": group_scores,
            "category_scores": category_scores,  # <-- PCA 입력
            "debug": {
                "structured_rows_found": len(structured_rows),
            },
        })

    # Rank by total_score
    results.sort(key=lambda r: (-r["total_score"], r["filename"]))

    total_files = len(results)

    # Trim BEFORE pass decision
    trim_mode = getattr(args, "trim_mode", "bottom")
    trim_pct = max(0.0, float(getattr(args, "trim_pct", 0.0)))

    trim_top = 0
    trim_bottom = 0
    if total_files > 0 and trim_mode != "none" and trim_pct > 0:
        k = int(math.ceil(total_files * trim_pct))
        if trim_mode == "both":
            k = min(k, (total_files - 1) // 2)
            trim_top = k
            trim_bottom = k
        elif trim_mode == "bottom":
            k = min(k, total_files - 1)
            trim_bottom = k

    eligible_results = results[trim_top: total_files - trim_bottom] if total_files else []
    eligible_count = len(eligible_results)

    accepted_count = min(
        eligible_count,
        int(math.ceil(eligible_count / args.ratio))
    ) if eligible_count > 0 else 0

    cutoff_total_score = eligible_results[accepted_count - 1]["total_score"] if accepted_count > 0 else 0.0

    for i, r in enumerate(results, start=1):
        r["rank"] = i

        idx0 = i - 1
        in_eligible = (trim_top <= idx0 < (total_files - trim_bottom))
        r["eligible"] = in_eligible
        if in_eligible:
            eligible_rank = (idx0 - trim_top) + 1
            r["eligible_rank"] = eligible_rank
            r["pass"] = (eligible_rank <= accepted_count)
        else:
            r["eligible_rank"] = None
            r["pass"] = False

        r["status"] = (
            "TRIMMED_TOP" if (not in_eligible and idx0 < trim_top)
            else "TRIMMED_BOTTOM" if (not in_eligible)
            else "PASS" if r["pass"]
            else "FAIL"
        )

    cutoff_record = eligible_results[accepted_count - 1] if accepted_count > 0 else None

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
                fieldnames=[
                    "rank",
                    "filename",
                    "total_score",
                    "structured_score",
                    "unstructured_score",
                    "status",
                    "eligible",
                    "eligible_rank",
                    "pass",
                    "path",
                    "structured_rows_found",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "rank": r["rank"],
                    "filename": r["filename"],
                    "total_score": round(r["total_score"], 6),
                    "structured_score": round(r["structured_score"], 6),
                    "unstructured_score": round(r["unstructured_score"], 6),
                    "status": r.get("status"),
                    "eligible": r.get("eligible"),
                    "eligible_rank": r.get("eligible_rank"),
                    "pass": r["pass"],
                    "path": r["path"],
                    "structured_rows_found": r["debug"]["structured_rows_found"],
                })
    except Exception:
        pass

    trimmed_top_list = [
        {"rank": r["rank"], "filename": r["filename"], "total_score": round(float(r["total_score"]), 6)}
        for r in results
        if r.get("status") == "TRIMMED_TOP"
    ]
    trimmed_bottom_list = [
        {"rank": r["rank"], "filename": r["filename"], "total_score": round(float(r["total_score"]), 6)}
        for r in results
        if r.get("status") == "TRIMMED_BOTTOM"
    ]

    summary = {
        "version": VERSION,
        "source": source_label,
        "total_files": total_files,
        "ratio": args.ratio,
        "accepted_count": accepted_count,
        "trim": {
            "mode": trim_mode,
            "pct": trim_pct,
            "top_trimmed": trim_top,
            "bottom_trimmed": trim_bottom,
            "eligible_count": eligible_count,
            "trimmed_top": trimmed_top_list,
            "trimmed_bottom": trimmed_bottom_list,
        },
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

    scores_json.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # --- PCA (PC1/PC2 + 히트맵) ---
    try:
        run_pca_and_save_artifacts(
            results=results,
            out_dir=out_dir,
            accepted_count=accepted_count,
        )
        print("[PCA] Saved: pca_scores.csv, pca_scatter.png, pca_heatmap.png, pca_model.json")
    except Exception as e:
        print(f"[PCA] skipped due to error: {e}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.do_print and results:
        to_print = results if (args.top_n is None or args.top_n <= 0) else results[:args.top_n]
        print(f"\n[Selection] trim_mode={trim_mode}, trim_pct={trim_pct:.2f}, trim_top={trim_top}, trim_bottom={trim_bottom}, eligible={eligible_count}, ratio={args.ratio}:1, accepted={accepted_count}, cutoff={cutoff_total_score:.3f}")

        if trim_top > 0:
            print(f"\nTrimmed TOP {trim_top} (excluded from pass calculation):")
            for r in results[:trim_top]:
                print(f"- {r['rank']:02d}. {r['filename']}: {r['total_score']:.3f}")
        if trim_bottom > 0:
            print(f"\nTrimmed BOTTOM {trim_bottom} (excluded from pass calculation):")
            for r in results[-trim_bottom:]:
                print(f"- {r['rank']:02d}. {r['filename']}: {r['total_score']:.3f}")

        print("\nRanked list (total_score):")
        for r in to_print:
            if r.get("eligible"):
                er = r.get("eligible_rank")
                print(
                    f"{r['rank']:02d}. {r['filename']}: {r['total_score']:.3f}  "
                    f"status={r.get('status')}  eligible_rank={er}/{eligible_count}  pass={str(r.get('pass'))}"
                )
            else:
                print(
                    f"{r['rank']:02d}. {r['filename']}: {r['total_score']:.3f}  "
                    f"status={r.get('status')}  pass={str(r.get('pass'))}"
                )

    # ==========================
    # 임의 입력 1개 파일 선택 → 동일 채점 → PCA 투영 비교
    # ==========================
    model_path = out_dir / "pca_model.json"
    if model_path.exists():
        p = _choose_txt_file_gui()
        if p is None:
            p = _choose_txt_file_fallback()

        if p is not None and p.exists():
            try:
                text = read_text_file(p)
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
                category_scores = _extract_category_scores_for_pca(score_detail)

                proj = _project_one_with_saved_pca_model(model_path, category_scores)

                pass_total = bool(total_score >= cutoff_total_score)
                pass_pc1 = bool(proj.get("pca_pass_pc1", False))

                print("\n" + "=" * 70)
                print("[INPUT] 임의 파일 평가 결과")
                print(f"- file: {p}")
                print(f"- total_score: {total_score:.3f} (cutoff_total_score={cutoff_total_score:.3f}) -> {'PASS' if pass_total else 'FAIL'}")
                print(f"- structured_score: {structured_score:.3f}")
                print(f"- unstructured_score: {unstructured_score:.3f}")

                print("\n[INPUT][PCA] 투영 결과")
                print(f"- PC1: {proj['pc1']:.4f} (pc1_cutoff={proj['pc1_cutoff']:.4f}) -> {'PASS' if pass_pc1 else 'FAIL'}")
                print(f"- PC2: {proj['pc2']:.4f}")
                print(f"- student_type: {proj['student_type']}")
                print(f"- dist_to_pass_centroid: {proj['dist_to_pass_centroid']:.4f}")
                evr = proj.get("explained_variance_ratio", [])
                if isinstance(evr, list) and len(evr) >= 2:
                    print(f"- explained_variance_ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")

                print("\n[DECISION] (둘 다 만족하면 합격권으로 판단하는 방식)")
                final_pass = bool(pass_total and pass_pc1)
                print(f"- PASS(total_score) AND PASS(PC1) => {'PASS' if final_pass else 'FAIL'}")
                print("=" * 70)

                # 저장
                input_out = {
                    "file": str(p),
                    "total_score": total_score,
                    "cutoff_total_score": float(cutoff_total_score),
                    "pass_total_score": pass_total,
                    "structured_score": structured_score,
                    "unstructured_score": unstructured_score,
                    "pca": proj,
                    "pass_pc1": pass_pc1,
                    "final_pass_both": final_pass,
                }
                (out_dir / "input_result.json").write_text(
                    json.dumps(input_out, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print(f"[INPUT] Saved: {out_dir / 'input_result.json'}")

            except Exception as e:
                print(f"[INPUT] evaluation failed: {e}")
        else:
            print("\n[INPUT] 파일 선택이 취소되어 임의 입력 평가는 스킵되었습니다.")
    else:
        print("\n[INPUT] pca_model.json이 없어서 임의 입력 PCA 비교를 스킵합니다.")

    # output 폴더 열기 (Windows)
    if args.open_outdir:
        try:
            if os.name == "nt":
                os.startfile(str(out_dir))
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
