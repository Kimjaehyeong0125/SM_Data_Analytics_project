import numpy as np
import pandas as pd


def zscore_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    result = df.copy()
    for col in cols:
        mean = result[col].mean()
        std = result[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            result[col + "_z"] = 0.0
        else:
            result[col + "_z"] = (result[col] - mean) / std
    return result


def trimmed_mean(series: pd.Series, trim_ratio: float = 0.1) -> float:
    arr = np.sort(series.to_numpy())
    n = len(arr)
    k = int(n * trim_ratio)
    if 2 * k >= n:
        return float(series.mean())
    trimmed = arr[k:n - k]
    return float(trimmed.mean())
