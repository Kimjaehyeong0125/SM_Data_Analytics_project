import pandas as pd


def load_population(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_population(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_new_applicant(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
