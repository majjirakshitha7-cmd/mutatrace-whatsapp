"""Statistics: do edits relate to spread, and do COVID messages mutate differently?"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import config as C

BINARY = ["numbers_changed", "prefix_added", "suffix_added", "url_added", "cta_added", "cta_removed"]
CONT = ["edit_size", "words_added", "words_removed", "words_replaced", "real_rewrites"]


def spread_tests(p):
    """Mann-Whitney for yes/no edits, Spearman for counts (vs max_sharings)."""
    rows = []
    for f in BINARY:
        a, b = p.loc[p[f], "max_sharings"], p.loc[~p[f], "max_sharings"]
        if len(a) and len(b):
            rows.append(dict(feature=f, kind="mannwhitney", n_with=len(a),
                             mean_with=round(a.mean(), 3), mean_without=round(b.mean(), 3),
                             stat=np.nan, p=round(mannwhitneyu(a, b).pvalue, 4)))
    for c in CONT:
        r, pv = spearmanr(p[c], p["max_sharings"])
        rows.append(dict(feature=c, kind="spearman", n_with=len(p), mean_with=np.nan,
                         mean_without=np.nan, stat=round(r, 3), p=round(pv, 4)))
    return pd.DataFrame(rows)


def predict_auc(p):
    """Can the edit features predict whether a variant circulated (2+ sharings)?"""
    cols = ["edit_size", "words_added", "words_removed", "words_replaced",
            "numbers_changed", "prefix_added", "suffix_added", "url_added"]
    X, y, g = p[cols].astype(float), p["viral_variant"], p["cluster"]
    n_splits = min(5, g.nunique())
    if n_splits < 2 or y.nunique() < 2:
        return {}
    out = {}
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "gboost": GradientBoostingClassifier(random_state=C.SEED)}
    for name, clf in models.items():
        pr = cross_val_predict(clf, X, y, cv=GroupKFold(n_splits=n_splits), groups=g,
                               method="predict_proba")[:, 1]
        out[name] = round(roc_auc_score(y, pr), 3)
    return out


def _partial_spearman(df, x, y, controls):
    R = df[[x, y] + controls].rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c] for c in controls])
    rx = R[x] - Z @ np.linalg.lstsq(Z, R[x], rcond=None)[0]
    ry = R[y] - Z @ np.linalg.lstsq(Z, R[y], rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def within_family(p, B=C.BOOTSTRAP_B):
    """Edits vs sharings relative to the family mean, controlling for timing and
    length, with a family-level bootstrap confidence interval."""
    controls = ["days_since_root", "root_len"]
    groups = {k: v for k, v in p.groupby("cluster")}
    keys = list(groups)
    rng = np.random.default_rng(C.SEED)
    rows = []
    for x in ["real_rewrites", "words_added", "rewrite_rate"]:
        raw = spearmanr(p[x], p["sharings_rel"])[0]
        part = _partial_spearman(p, x, "sharings_rel", controls)
        boots = []
        for _ in range(B):
            pick = rng.choice(keys, len(keys), replace=True)
            boots.append(_partial_spearman(pd.concat([groups[k] for k in pick], ignore_index=True),
                                           x, "sharings_rel", controls))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(feature=x, raw_rho=round(raw, 3), partial_rho=round(part, 3),
                         ci_low=round(lo, 3), ci_high=round(hi, 3)))
    r, pv = spearmanr(p["days_since_root"], p["sharings_rel"])
    return pd.DataFrame(rows), dict(days_vs_sharings_rho=round(r, 3), p=round(pv, 4))


def covid_tests(p):
    """COVID vs non-COVID families: Fisher tests on yes/no edits and
    length-fair rate comparisons (Mann-Whitney)."""
    rows = []
    flags = {"header_added": p["prefix_added"], "words_added>3": p["words_added"] > 3,
             "real_rewrite>3": p["real_rewrites"] > 3, "words_dropped>3": p["words_removed"] > 3,
             "numbers_changed": p["numbers_changed"], "footer_added": p["suffix_added"]}
    for name, m in flags.items():
        tab = pd.crosstab(p["covid_family"], m).reindex(index=[False, True],
                                                        columns=[False, True], fill_value=0)
        odds, pv = fisher_exact(tab.values)
        rows.append(dict(measure=name, test="fisher", covid=round(m[p["covid_family"]].mean(), 3),
                         non_covid=round(m[~p["covid_family"]].mean(), 3),
                         effect=round(odds, 2), p=round(pv, 4)))
    for c in ["add_rate", "drop_rate", "rewrite_rate", "edit_size"]:
        a, b = p.loc[p["covid_family"], c], p.loc[~p["covid_family"], c]
        if len(a) and len(b):
            rows.append(dict(measure=c, test="mannwhitney (length-fair)",
                             covid=round(a.median(), 3), non_covid=round(b.median(), 3),
                             effect=np.nan, p=round(mannwhitneyu(a, b).pvalue, 4)))
    return pd.DataFrame(rows)
