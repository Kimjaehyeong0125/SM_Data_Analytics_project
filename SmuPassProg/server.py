from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
OUT_DIR = BASE_DIR / "output"
TMP_DIR = BASE_DIR / "_tmp"

# User will drop this into project root:
DEFAULT_COHORT_ZIP = BASE_DIR / "censored_txt.zip"
DEFAULT_COHORT_DIR = BASE_DIR / "censored_txt"

app = Flask(__name__, static_folder=None)


# -------------------------
# Static routes (front)
# -------------------------
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "reportPage.html")


@app.get("/app.js")
def app_js():
    return send_from_directory(WEB_DIR, "app.js")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/output/<path:filename>")
def output_files(filename: str):
    return send_from_directory(OUT_DIR, filename)


# -------------------------
# Helpers
# -------------------------
def _find_txt_root(extract_root: Path) -> Path:
    """
    Zip 구조가 아래 둘 중 무엇이든 처리:
      A) extract_root/*.txt
      B) extract_root/censored_txt/*.txt
    """
    children = [p for p in extract_root.iterdir()]

    # Case B: single top folder
    if len(children) == 1 and children[0].is_dir():
        candidate = children[0]
        if any(candidate.rglob("*.txt")):
            return candidate

    # Case A: txts directly under extract_root (or deeper)
    if any(extract_root.rglob("*.txt")):
        return extract_root

    raise FileNotFoundError("No .txt files found after extracting cohort zip.")


def _ensure_cohort_dir() -> Path:
    # Prefer already-unzipped folder if user has it
    if DEFAULT_COHORT_DIR.exists() and any(DEFAULT_COHORT_DIR.rglob("*.txt")):
        return DEFAULT_COHORT_DIR

    if not DEFAULT_COHORT_ZIP.exists():
        raise FileNotFoundError(
            "censored_txt.zip not found.\n"
            "Place it in the project root:\n"
            f"  {DEFAULT_COHORT_ZIP}"
        )

    # Extract zip to _tmp/cohort_extracted
    extract_root = TMP_DIR / "cohort_extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DEFAULT_COHORT_ZIP, "r") as zf:
        zf.extractall(extract_root)

    return _find_txt_root(extract_root)


def _merge_pca_into_results(payload: dict) -> dict:
    """
    main.py가 scores.json을 먼저 만들고, PCA 산출물(pca_scores.csv)을 별도로 만드는 경우가 있어
    pca_scores.csv 내용을 payload['results']에 병합합니다.
    """
    pca_csv = OUT_DIR / "pca_scores.csv"
    if not pca_csv.exists():
        return payload

    by_filename: dict[str, dict] = {}
    with pca_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_filename[row.get("filename", "")] = row

    def ffloat(x):
        try:
            return float(x)
        except Exception:
            return None

    def fbool(x):
        if x is None:
            return None
        s = str(x).strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None

    for r in payload.get("results", []) or []:
        fn = r.get("filename", "")
        row = by_filename.get(fn)
        if not row:
            continue

        r["pc1"] = ffloat(row.get("pc1"))
        r["pc2"] = ffloat(row.get("pc2"))
        r["pc1_rank"] = int(row["pc1_rank"]) if str(row.get("pc1_rank", "")).isdigit() else None
        r["pca_pass_pc1"] = fbool(row.get("pca_pass_pc1"))
        r["student_type"] = row.get("student_type")
        r["dist_to_pass_centroid"] = ffloat(row.get("dist_to_pass_centroid"))
        r["pc1_cutoff"] = ffloat(row.get("pc1_cutoff"))

    payload["artifacts"] = {
        "pca_scatter_url": "/output/pca_scatter.png",
        "pca_heatmap_url": "/output/pca_heatmap.png",
    }
    return payload


# -------------------------
# API (ONE CLICK)
# -------------------------
@app.post("/api/run_all")
def api_run_all():
    """
    한 번 클릭으로 main.py를 '터미널에서 실행한 것과 동일하게' 1회만 실행합니다.

    - 모집단: ./censored_txt.zip 또는 ./censored_txt 를 자동 사용
    - 입력 파일 선택: main.py가 내부적으로 파일 선택(Tkinter 등)을 띄운다면 그 1회만 수행
    - 결과: main.py 표준출력(stdout)을 그대로 반환 + output/*.json 및 PCA 이미지 URL 반환
    """
    try:
        cohort_dir = _ensure_cohort_dir()
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        import subprocess

        cmd = [
            sys.executable,
            str(BASE_DIR / "main.py"),
            "--dir", str(cohort_dir),
            "--out-dir", str(OUT_DIR),
            # Explorer 자동 오픈만 막고, 출력/파일선택 흐름은 main.py 그대로 유지
            "--no-open",
            "--print",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        stdout_text = proc.stdout or ""
        ok = (proc.returncode == 0)

        cohort_payload = None
        scores_json = OUT_DIR / "scores.json"
        if scores_json.exists():
            cohort_payload = json.loads(scores_json.read_text(encoding="utf-8"))
            cohort_payload = _merge_pca_into_results(cohort_payload)

        input_payload = None
        input_json = OUT_DIR / "input_result.json"
        if input_json.exists():
            input_payload = json.loads(input_json.read_text(encoding="utf-8"))

        return jsonify({
            "ok": ok,
            "returncode": proc.returncode,
            "stdout": stdout_text,
            "cohort": cohort_payload,
            "input": input_payload,
            "artifacts": {
                "pca_scatter_url": "/output/pca_scatter.png",
                "pca_heatmap_url": "/output/pca_heatmap.png",
            }
        }), (200 if ok else 500)

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
