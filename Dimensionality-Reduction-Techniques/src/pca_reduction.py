import time
import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X, variance_threshold=0.95):
    start_time = time.time()


    # Fit PCA with all components
    pca = PCA()
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)


    # Find minimum number of components
    optimal_components = np.argmax(
        cumulative_variance >= variance_threshold
    ) + 1


    # Refit PCA
    pca_model = PCA(n_components=optimal_components)
    X_pca = pca_model.fit_transform(X)

    execution_time = time.time() - start_time

    return (
        X_pca,
        pca_model,
        optimal_components,
        explained_variance,
        cumulative_variance,
        execution_time,
    )