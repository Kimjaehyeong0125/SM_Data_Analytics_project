import pandas as pd


def compute_total_score(
    df: pd.DataFrame,
    structured_z_cols: list[str],
    unstructured_z_cols: list[str],
    w_structured: float = 0.6,
    w_unstructured: float = 0.4,
) -> pd.DataFrame:
    result = df.copy()

    result["score_structured"] = result[structured_z_cols].mean(axis=1)
    result["score_unstructured"] = result[unstructured_z_cols].mean(axis=1)

    result["total_score"] = (
        w_structured * result["score_structured"]
        + w_unstructured * result["score_unstructured"]
    )

    result = result.sort_values("total_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1

    return result
