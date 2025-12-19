import pandas as pd
from sklearn.decomposition import PCA


def fit_pca_on_unstructured(
    df: pd.DataFrame,
    unstructured_z_cols: list[str],
    n_components: int = 2,
):
    X = df[unstructured_z_cols].to_numpy()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    result = df.copy()
    for i in range(n_components):
        result[f"pc{i+1}"] = X_pca[:, i]

    return result, pca
