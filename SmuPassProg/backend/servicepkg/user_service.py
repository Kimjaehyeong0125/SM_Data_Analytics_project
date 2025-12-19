import pandas as pd

from datapkg.io import load_new_applicant
from scoringpkg.aggregate import compute_total_score


def evaluate_new_applicants(
    pop_df: pd.DataFrame,
    new_applicant_path: str,
    structured_cols: list[str],
    unstructured_cols: list[str],
    cutoff_score: float,
) -> pd.DataFrame:
    new_df = load_new_applicant(new_applicant_path)

    for col in structured_cols + unstructured_cols:
        mean = pop_df[col].mean()
        std = pop_df[col].std(ddof=0)
        if std == 0:
            new_df[col + "_z"] = 0.0
        else:
            new_df[col + "_z"] = (new_df[col] - mean) / std

    structured_z_cols = [c + "_z" for c in structured_cols]
    unstructured_z_cols = [c + "_z" for c in unstructured_cols]

    scored = compute_total_score(
        new_df,
        structured_z_cols=structured_z_cols,
        unstructured_z_cols=unstructured_z_cols,
    )

    scored["pass_by_cutoff"] = scored["total_score"] >= cutoff_score
    return scored
