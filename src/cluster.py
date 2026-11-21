"""Group messages into variant families (TF-IDF and multilingual embeddings)."""
import hashlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from . import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as C


def tfidf_vectors(texts):
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)


def embed(texts, cache=C.PROC / "emb_cache.npz"):
    """Sentence embeddings; cached on disk and reused only if the texts are identical."""
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["emb"]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(C.EMB_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    if cache is not None:
        np.savez(cache, key=key, emb=emb)
    return emb


def knn(X, k=C.K_NEIGHBORS):
    """Nearest neighbours once; reuse for every threshold. Returns (similarity, index)."""
    k = min(k, X.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1).fit(X)
    dist, idx = nn.kneighbors(X)
    return 1 - dist, idx


def labels_from_knn(sim, idx, th):
    """Connect messages that are mutual neighbours above `th`; components = families."""
    n = sim.shape[0]
    keep = (sim >= th) & (idx != np.arange(n)[:, None])
    rows = np.repeat(np.arange(n), idx.shape[1])[keep.ravel()]
    cols = idx.ravel()[keep.ravel()]
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    g = g.minimum(g.T)                       # mutual neighbours only (prevents chaining)
    return connected_components(g, directed=False)[1]


def cluster_sizes(labels):
    s = pd.Series(labels)
    return s.map(s.value_counts()).values


def add_clusters(agg, use_embeddings=True):
    """Add cl_*/size_* columns for TF-IDF (and optionally MiniLM)."""
    agg = agg.copy()
    sim, idx = knn(tfidf_vectors(agg["clean"]))
    lab = labels_from_knn(sim, idx, C.TFIDF_TH)
    agg["cl_tfidf"], agg["size_tfidf"] = lab, cluster_sizes(lab)
    if use_embeddings:
        sim, idx = knn(embed(agg["clean"].tolist()))
        lab = labels_from_knn(sim, idx, C.EMB_TH)
        agg["cl_minilm"], agg["size_minilm"] = lab, cluster_sizes(lab)
    return agg
