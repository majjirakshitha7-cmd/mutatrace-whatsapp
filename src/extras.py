"""Extra experiments:
   1. validate the mutation detectors on planted variants (true edit is known),
   2. hybrid clustering (TF-IDF OR embedding agreement),
   3. sensitivity of the main findings to the clustering threshold."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix

try:
    from . import config as C, analysis, mutations
    from .cluster import knn, labels_from_knn, embed, cluster_sizes, tfidf_vectors
    from .evaluate import make_planted
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as C, analysis, mutations
    from src.cluster import knn, labels_from_knn, embed, cluster_sizes, tfidf_vectors
    from src.evaluate import make_planted


# ---------------------------------------------------------------- 1. detectors
DETECTORS = {                                   # detector -> planted edit types it should fire on
    "header_added": {"header_added"},
    "footer_added": {"footer_added"},
    "numbers_changed": {"numbers_changed"},
    "numbers_replaced (strict)": {"numbers_changed"},
    "words_dropped>3": {"drop_15pct", "drop_30pct", "truncate_60pct"},
}


def _numbers_replaced(a, b):
    """Strict rule: a number token is replaced by another number token."""
    import difflib
    ta, tb = a.split(), b.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "replace" and mutations.NUM.search(" ".join(ta[i1:i2])) \
                and mutations.NUM.search(" ".join(tb[j1:j2])):
            return True
    return False


def _flags(a, b):
    ins, dele, rep, prefix, suffix, sim = mutations.diff_features(a, b)
    na, nb = set(mutations.NUM.findall(a)), set(mutations.NUM.findall(b))
    return {"header_added": prefix, "footer_added": suffix,
            "numbers_changed": bool(na and nb and na != nb),
            "numbers_replaced (strict)": _numbers_replaced(a, b),
            "words_dropped>3": dele > 3}


def validate_detectors(agg):
    """Run the real detectors on planted variants and score them against the known edit."""
    synth, _ = make_planted(agg)
    F = pd.DataFrame([_flags(agg.loc[s, "clean"], t) for s, t in zip(synth["src"], synth["text"])])
    rows = []
    for det, true_types in DETECTORS.items():
        pos = synth["mut"].isin(true_types).values
        flag = F[det].values.astype(bool)
        tp, fn = int((flag & pos).sum()), int((~flag & pos).sum())
        fp, tn = int((flag & ~pos).sum()), int((~flag & ~pos).sum())
        rows.append(dict(detector=det, should_fire_on="+".join(sorted(true_types)),
                         n_true=int(pos.sum()), n_other=int((~pos).sum()),
                         recall=round(tp / max(tp + fn, 1), 3),
                         precision=round(tp / max(tp + fp, 1), 3),
                         false_positive_rate=round(fp / max(fp + tn, 1), 3)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. hybrid clustering
def _graph(sim, idx, th):
    n = sim.shape[0]
    keep = (sim >= th) & (idx != np.arange(n)[:, None])
    rows = np.repeat(np.arange(n), idx.shape[1])[keep.ravel()]
    cols = idx.ravel()[keep.ravel()]
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return g.minimum(g.T)                       # mutual neighbours only


def _score(labels, planted, truth, var_pos, synth):
    pred = labels[planted]
    M = pair_confusion_matrix(truth, pred)
    tp, fp, fn = M[1, 1], M[0, 1], M[1, 0]
    row = {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
           "ARI": round(adjusted_rand_score(truth, pred), 3)}
    same = labels[var_pos] == labels[synth["src"].values]
    for m in synth["mut"].unique():
        row[m] = round(same[(synth["mut"] == m).values].mean(), 2)
    return row


def hybrid_benchmark(agg):
    """Compare TF-IDF, MiniLM and their union on the planted benchmark, and report how
    big the largest natural family gets (a sign of chaining)."""
    synth, src_ids = make_planted(agg)
    n = len(agg)
    texts_all = agg["clean"].tolist() + synth["text"].tolist()
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts_all)
    emb = np.vstack([embed(agg["clean"].tolist()), embed(synth["text"].tolist(), cache=None)])
    st, it = knn(X)
    se, ie = knn(emb)

    src_pos = np.array(src_ids)
    var_pos = n + np.arange(len(synth))
    planted = np.concatenate([src_pos, var_pos])
    truth = np.concatenate([src_pos, synth["src"].values])

    configs = [("tfidf", 0.6, None), ("minilm", None, 0.85), ("minilm", None, 0.90),
               ("hybrid", 0.6, 0.90), ("hybrid", 0.6, 0.85), ("hybrid", 0.7, 0.90)]
    rows = []
    for name, tth, eth in configs:
        graphs = []
        if tth is not None:
            graphs.append(_graph(st, it, tth))
        if eth is not None:
            graphs.append(_graph(se, ie, eth))
        g = graphs[0]
        for h in graphs[1:]:
            g = g.maximum(h)
        labels = connected_components(g, directed=False)[1]
        real = connected_components(g[:n, :n], directed=False)[1]      # real messages only
        sizes = pd.Series(real).value_counts()
        row = {"method": name, "tfidf_th": tth, "minilm_th": eth}
        row.update(_score(labels, planted, truth, var_pos, synth))
        row["real_families"] = int((sizes >= 2).sum())
        row["largest_real_family"] = int(sizes.max())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. threshold sensitivity
def threshold_sensitivity(agg, ths=(0.5, 0.6, 0.7), B=100):
    """Repeat the main analysis at several TF-IDF thresholds."""
    sim, idx = knn(tfidf_vectors(agg["clean"]))
    controls = ["days_since_root", "root_len"]
    rows = []
    for th in ths:
        a = agg.copy()
        lab = labels_from_knn(sim, idx, th)
        a["cl_tfidf"], a["size_tfidf"] = lab, cluster_sizes(lab)
        p = mutations.build_pairs(a)
        if p.empty:
            continue
        shares = mutations.mutation_shares(p)
        part = analysis._partial_spearman(p, "real_rewrites", "sharings_rel", controls)
        groups = {k: v for k, v in p.groupby("cluster")}
        keys = list(groups)
        rng = np.random.default_rng(C.SEED)
        boots = [analysis._partial_spearman(pd.concat([groups[k] for k in rng.choice(keys, len(keys))],
                                                      ignore_index=True),
                                            "real_rewrites", "sharings_rel", controls)
                 for _ in range(B)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(
            tfidf_th=th, families=p["cluster"].nunique(), variant_pairs=len(p),
            largest_family=int(p["family_size"].max()),
            header_added=round(shares["header added (start)"], 3),
            footer_added=round(shares["footer added (end)"], 3),
            numbers_changed=round(shares["numbers changed"], 3),
            words_rewritten=round(shares["words rewritten (>3)"], 3),
            rho_edit_size_vs_sharings=round(spearmanr(p["edit_size"], p["max_sharings"])[0], 3),
            auc_logreg=analysis.predict_auc(p).get("logreg"),
            partial_rho_rewrites=round(part, 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
    return pd.DataFrame(rows)
