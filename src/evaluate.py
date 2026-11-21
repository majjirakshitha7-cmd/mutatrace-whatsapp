"""Synthetic benchmark: plant known mutations of real messages and measure
whether each clustering method puts every variant back with its source."""
import random
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix

try:
    from . import config as C
    from .cluster import knn, labels_from_knn, embed
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as C
    from src.cluster import knn, labels_from_knn, embed

FOOTER = "compartilhe com todos os seus contatos agora mesmo"
HEADER = "atenção urgente leia até o final"


def _mutations(rng):
    def drop(frac):
        def f(t):
            w = [x for x in t.split() if rng.random() > frac]
            return " ".join(w) if len(w) >= 5 else None
        return f

    def swaps(t):
        w = t.split()
        for i in range(len(w) - 1):
            if rng.random() < 0.15:
                w[i], w[i + 1] = w[i + 1], w[i]
        return " ".join(w)

    def numbers(t):
        if not re.search(r"\d", t):
            return None
        return re.sub(r"\d{1,9}", lambda m: str(int(m.group()) + rng.randint(1, 50)), t)

    return {
        "numbers_changed": numbers,
        "footer_added": lambda t: t + " " + FOOTER,
        "header_added": lambda t: HEADER + " " + t,
        "drop_15pct": drop(0.15),
        "drop_30pct": drop(0.30),
        "truncate_60pct": lambda t: " ".join(t.split()[:int(len(t.split()) * 0.6)]),
        "word_swaps": swaps,
    }


def make_planted(agg, n=C.N_PLANTED, seed=C.SEED):
    """Pick messages that are not in any natural family and create known variants."""
    rng = random.Random(seed)
    muts = _mutations(rng)
    words = agg["clean"].str.split().str.len()
    mask = (agg["size_tfidf"] == 1) & (words >= 15)
    if "size_minilm" in agg:
        mask &= agg["size_minilm"] == 1
    cand = agg[mask]
    src_ids = cand.sample(min(n, len(cand)), random_state=seed).index.tolist()
    rows = []
    for s in src_ids:
        base = agg.loc[s, "clean"]
        for name, fn in muts.items():
            v = fn(base)
            if v is not None and v != base:
                rows.append({"src": s, "mut": name, "text": v})
    return pd.DataFrame(rows), src_ids


def run_benchmark(agg, use_embeddings=True):
    synth, src_ids = make_planted(agg)
    n = len(agg)
    texts_all = agg["clean"].tolist() + synth["text"].tolist()
    spaces = {"tfidf": (TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                        .fit_transform(texts_all), C.BENCH_TFIDF_TH)}
    if use_embeddings:
        emb = np.vstack([embed(agg["clean"].tolist()), embed(synth["text"].tolist(), cache=None)])
        spaces["minilm"] = (emb, C.BENCH_EMB_TH)

    src_pos = np.array(src_ids)
    var_pos = n + np.arange(len(synth))
    planted = np.concatenate([src_pos, var_pos])
    truth = np.concatenate([src_pos, synth["src"].values])

    res = []
    for name, (X, ths) in spaces.items():
        sim, idx = knn(X)
        for th in ths:
            labels = labels_from_knn(sim, idx, th)
            pred = labels[planted]
            M = pair_confusion_matrix(truth, pred)
            tp, fp, fn = M[1, 1], M[0, 1], M[1, 0]
            row = {"method": name, "th": th,
                   "precision": round(tp / max(tp + fp, 1), 3),
                   "recall": round(tp / max(tp + fn, 1), 3),
                   "ARI": round(adjusted_rand_score(truth, pred), 3)}
            same = labels[var_pos] == labels[synth["src"].values]
            for m in synth["mut"].unique():
                row[m] = round(same[(synth["mut"] == m).values].mean(), 2)
            res.append(row)
    return pd.DataFrame(res), synth
