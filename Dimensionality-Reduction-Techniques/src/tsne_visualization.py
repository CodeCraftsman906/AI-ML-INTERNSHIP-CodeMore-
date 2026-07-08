import time
from sklearn.manifold import TSNE


def apply_tsne(
    X,
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    max_iter=1000,
    random_state=42,
):
    
    start_time = time.time()

    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )

    X_tsne = tsne_model.fit_transform(X)

    execution_time = time.time() - start_time

    return (
        X_tsne,
        tsne_model,
        execution_time,
    )