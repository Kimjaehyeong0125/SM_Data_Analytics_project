import pandas as pd
from utilpkg.stats import trimmed_mean


def compute_cutoff_from_accepted(
    df: pd.DataFrame,
    trim_ratio: float = 0.1,
) -> dict:
    accepted_df = df[df["accepted"] == 1]

    scores = accepted_df["total_score"]
    mean_val = float(scores.mean())
    median_val = float(scores.median())
    trimmed_val = trimmed_mean(scores, trim_ratio=trim_ratio)

    return {
        "mean_cutoff": mean_val,
        "median_cutoff": median_val,
        "trimmed_cutoff": trimmed_val,
    }

