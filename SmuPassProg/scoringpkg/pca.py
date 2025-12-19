from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ACADEMIC_KEYS_DEFAULT = [
    "academic_achievement",
    "major_fit",
    "inquiry",
    "career_exploration",
]

COMMUNITY_KEYS_DEFAULT = [
    "community",
    "leadership",
    "communication",
    "empathy",
]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _collect_feature_names(results: List[Dict[str, Any]]) -> List[str]:
    keys = set()
    for r in results:
        cs = r.get("category_scores") or {}
        if isinstance(cs, dict):
            keys.update(cs.keys())
    return sorted(keys)


def _build_matrix(results: List[Dict[str, Any]], feature_names: List[str]) -> np.ndarray:
    X = np.zeros((len(results), len(feature_names)), dtype=float)
    for i, r in enumerate(results):
        cs = r.get("category_scores") or {}
        for j, k in enumerate(feature_names):
            X[i, j] = _safe_float(cs.get(k, 0.0))
    return X


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.allclose(np.std(a), 0) or np.allclose(np.std(b), 0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _student_type(pc1: float, pc2: float) -> str:
    # 기준은 0 (표준화 후 PCA score는 평균 0 근처)
    if pc1 >= 0 and pc2 < 0:
        return "전공역량형"
    if pc1 >= 0 and pc2 >= 0:
        return "균형형"
    if pc1 < 0 and pc2 >= 0:
        return "공동체형"
    return "발전필요형"


def run_pca_and_save_artifacts(
    results: List[Dict[str, Any]],
    out_dir: str | Path,
    *,
    accepted_count: int,
    academic_keys: Optional[List[str]] = None,
    community_keys: Optional[List[str]] = None,
) -> None:
    """
    - results: main.py에서 만든 results 리스트(각 원소에 category_scores 필수)
    - accepted_count: 기존 total_score 기준 '합격자 수'를 그대로 PCA 컷 산정에 사용
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    feature_names = _collect_feature_names(results)
    if not feature_names:
        raise ValueError("PCA needs category_scores in results (feature_names is empty).")

    X = _build_matrix(results, feature_names)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)  # (n,2)
    pc1 = Z[:, 0].copy()
    pc2 = Z[:, 1].copy()

    # --- 방향(부호) 정렬: PC1은 '학업/전공'과 양의 상관, PC2는 '공동체'와 양의 상관이 되도록 뒤집기 ---
    academic_keys = academic_keys or ACADEMIC_KEYS_DEFAULT
    community_keys = community_keys or COMMUNITY_KEYS_DEFAULT

    # academic/community bundle (원점수 기반)
    def bundle(keys: List[str]) -> np.ndarray:
        idx = [feature_names.index(k) for k in keys if k in feature_names]
        if not idx:
            return np.zeros((len(results),), dtype=float)
        return np.mean(X[:, idx], axis=1)

    academic_bundle = bundle(academic_keys)
    community_bundle = bundle(community_keys)

    if _corr(pc1, academic_bundle) < 0:
        pc1 *= -1
        pca.components_[0, :] *= -1

    if _corr(pc2, community_bundle) < 0:
        pc2 *= -1
        pca.components_[1, :] *= -1

    # --- PC1 기반 합격 컷: (eligible만 대상으로) PC1 내림차순 정렬 후 accepted_count 위치를 컷으로 ---
    eligible = [r for r in results if r.get("eligible", True)]
    eligible_idx = [i for i, r in enumerate(results) if r.get("eligible", True)]

    if len(eligible) == 0:
        pc1_cut = float(np.max(pc1))
    else:
        # eligible pc1만 따로 정렬
        elig_pc1 = [(i, pc1[i]) for i in eligible_idx]
        elig_pc1.sort(key=lambda t: -t[1])
        if accepted_count <= 0:
            pc1_cut = float(elig_pc1[0][1])  # 의미상 top
        else:
            k = min(accepted_count, len(elig_pc1)) - 1
            pc1_cut = float(elig_pc1[k][1])

    # pass centroid (기존 total_score PASS 집단 중심)
    pass_idx = [i for i, r in enumerate(results) if r.get("pass") is True]
    if pass_idx:
        centroid = np.array([np.mean(pc1[pass_idx]), np.mean(pc2[pass_idx])], dtype=float)
    else:
        centroid = np.array([0.0, 0.0], dtype=float)

    # 결과에 PCA 결과 주입
    for i, r in enumerate(results):
        r["pc1"] = float(pc1[i])
        r["pc2"] = float(pc2[i])
        r["student_type"] = _student_type(float(pc1[i]), float(pc2[i]))
        r["pca_pass_pc1"] = bool(pc1[i] >= pc1_cut)
        r["dist_to_pass_centroid"] = float(np.linalg.norm(np.array([pc1[i], pc2[i]]) - centroid))

    # PC1 rank (전체 기준)
    order = sorted(range(len(results)), key=lambda i: (-pc1[i], results[i].get("filename", "")))
    for rank, i in enumerate(order, start=1):
        results[i]["pc1_rank"] = rank

    # --- 저장: pca_scores.csv ---
    pca_scores_csv = out_dir / "pca_scores.csv"
    try:
        import csv
        with pca_scores_csv.open("w", newline="", encoding="utf-8-sig") as f:
            fieldnames = [
                "rank",
                "filename",
                "total_score",
                "pass",
                "pc1",
                "pc2",
                "pc1_rank",
                "pca_pass_pc1",
                "student_type",
                "dist_to_pass_centroid",
                "pc1_cutoff",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({
                    "rank": r.get("rank"),
                    "filename": r.get("filename"),
                    "total_score": round(_safe_float(r.get("total_score")), 6),
                    "pass": r.get("pass"),
                    "pc1": round(_safe_float(r.get("pc1")), 6),
                    "pc2": round(_safe_float(r.get("pc2")), 6),
                    "pc1_rank": r.get("pc1_rank"),
                    "pca_pass_pc1": r.get("pca_pass_pc1"),
                    "student_type": r.get("student_type"),
                    "dist_to_pass_centroid": round(_safe_float(r.get("dist_to_pass_centroid")), 6),
                    "pc1_cutoff": round(float(pc1_cut), 6),
                })
    except Exception:
        pass

    # --- 시각화: scatter / heatmap ---
    import matplotlib.pyplot as plt

    # scatter: PASS/FAIL 마커만 다르게(색 지정 없음)
    scatter_png = out_dir / "pca_scatter.png"
    plt.figure(figsize=(8, 6))

    pass_mask = np.array([r.get("pass") is True for r in results], dtype=bool)
    plt.scatter(pc1[~pass_mask], pc2[~pass_mask], marker="x", alpha=0.8, label="FAIL")
    plt.scatter(pc1[pass_mask], pc2[pass_mask], marker="o", alpha=0.8, label="PASS")

    plt.axvline(0, linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.title("PCA Scatter (PC1 vs PC2)")
    plt.xlabel("PC1 (학업/전공 축 방향 정렬)")
    plt.ylabel("PC2 (공동체 축 방향 정렬)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=200)
    plt.close()

    # heatmap: PASS 집단 밀도 (hexbin)
    heatmap_png = out_dir / "pca_heatmap.png"
    plt.figure(figsize=(8, 6))
    if pass_idx:
        plt.hexbin(pc1[pass_idx], pc2[pass_idx], gridsize=25, mincnt=1)
        plt.colorbar(label="PASS density")
    else:
        plt.text(0.5, 0.5, "No PASS samples", ha="center", va="center", transform=plt.gca().transAxes)

    plt.axvline(0, linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.title("PASS Heatmap (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=200)
    plt.close()

    # --- 모델 저장: 나중에 임의 입력 1명 투영에 그대로 사용 가능 ---
    model = {
        "feature_names": feature_names,
        "scaler": {
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        },
        "pca": {
            "components_": pca.components_.tolist(),
            "explained_variance_ratio_": pca.explained_variance_ratio_.tolist(),
        },
        "pc1_cutoff": float(pc1_cut),
        "pass_centroid": centroid.tolist(),
        "academic_keys_used": academic_keys,
        "community_keys_used": community_keys,
    }
    (out_dir / "pca_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
