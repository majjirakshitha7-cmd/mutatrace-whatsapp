"""Misinformation analysis: link variant families with fact-checked labels
from covidbr_labeled.csv to measure fact-checking propagation and compare mutation patterns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from . import config as C, data

LABELED_CSV = C.ROOT / "data" / "raw" / "covidbr_labeled.csv"


def load_labeled_data():
    """Load and clean fact-checked messages."""
    df = pd.read_csv(LABELED_CSV)
    df["clean"] = df["text"].astype(str).map(data.clean_text)
    # Take the consensus label per cleaned text (if any duplicate was marked misinfo, flag it)
    lab_map = df.groupby("clean")["misinformation"].max().to_dict()
    src_map = df.groupby("clean")["source"].first().to_dict()
    return lab_map, src_map


def analyze_propagation(clustered):
    """Measure how clustering propagates fact-check labels to unflagged variants."""
    lab_map, _ = load_labeled_data()
    df = clustered.copy()
    df["misinfo_direct"] = df["clean"].map(lab_map)

    # Propagate to families
    fam_misinfo = df.groupby("cl_tfidf")["misinfo_direct"].max().fillna(0).astype(int)
    df["fam_has_misinfo"] = df["cl_tfidf"].map(fam_misinfo)

    fam_df = df[df["size_tfidf"] >= 2]
    misinfo_fams = fam_df.groupby("cl_tfidf")["misinfo_direct"].max().eq(1).sum()
    total_in_fams = int((fam_df["fam_has_misinfo"] == 1).sum())
    direct_labeled = int((fam_df.loc[fam_df["fam_has_misinfo"] == 1, "misinfo_direct"] == 1).sum())
    unflagged_variants = total_in_fams - direct_labeled

    unflagged_sub = fam_df[(fam_df["fam_has_misinfo"] == 1) & (fam_df["misinfo_direct"] != 1)]
    unflagged_shares = int(unflagged_sub["max_sharings"].sum())
    unflagged_occurrences = int(unflagged_sub["n_occurrences"].sum())

    res = pd.DataFrame([{
        "misinfo_families": int(misinfo_fams),
        "total_messages_in_misinfo_families": total_in_fams,
        "direct_factchecked_messages": direct_labeled,
        "unflagged_mutated_variants": unflagged_variants,
        "coverage_boost_pct": round(100.0 * unflagged_variants / max(1, direct_labeled), 1),
        "unflagged_variants_sharings": unflagged_shares,
        "unflagged_variants_occurrences": unflagged_occurrences
    }])
    return res, fam_misinfo


def analyze_misinfo_mutations(pairs, fam_misinfo):
    """Compare mutation patterns in misinformation families vs non-misinformation families."""
    p = pairs.copy()
    p["misinfo_family"] = p["cluster"].map(fam_misinfo).fillna(0).astype(int)

    tests = {
        "cta_urgency_added": p["cta_added"],
        "header_added": p["prefix_added"],
        "words_added>3": p["words_added"] > 3,
        "real_rewrite>3": p["real_rewrites"] > 3,
        "footer_added": p["suffix_added"],
        "numbers_changed": p["numbers_changed"],
        "words_dropped>3": p["words_removed"] > 3,
    }

    rows = []
    for name, s in tests.items():
        t = pd.crosstab(p["misinfo_family"] == 1, s)
        if t.shape == (2, 2):
            odds, pv = fisher_exact(t)
            mis_rate = s[p["misinfo_family"] == 1].mean()
            non_rate = s[p["misinfo_family"] == 0].mean()
            rows.append({
                "measure": name,
                "test": "fisher",
                "misinfo_share": round(float(mis_rate), 3),
                "non_misinfo_share": round(float(non_rate), 3),
                "odds_ratio": round(float(odds), 2),
                "p_value": round(float(pv), 4)
            })

    for name in ["edit_size", "add_rate", "rewrite_rate", "drop_rate"]:
        a = p.loc[p["misinfo_family"] == 1, name]
        b = p.loc[p["misinfo_family"] == 0, name]
        pv = mannwhitneyu(a, b).pvalue
        rows.append({
            "measure": name,
            "test": "mannwhitney",
            "misinfo_share": round(float(a.median()), 3),
            "non_misinfo_share": round(float(b.median()), 3),
            "odds_ratio": round(float(a.mean() - b.mean()), 3),
            "p_value": round(float(pv), 4)
        })

    return pd.DataFrame(rows)


def plot_misinfo_comparison(test_df, path=C.OUT / "misinformation_mutations.png"):
    """Plot comparison of mutation rates for key features."""
    fisher_df = test_df[test_df["test"] == "fisher"].copy()
    fisher_df = fisher_df.sort_values("odds_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(fisher_df))
    height = 0.35

    ax.barh(y + height / 2, fisher_df["misinfo_share"], height, label="Misinformation Families", color="#d9534f")
    ax.barh(y - height / 2, fisher_df["non_misinfo_share"], height, label="Non-Misinformation Families", color="#428bca")

    ax.set_yticks(y)
    ax.set_yticklabels(fisher_df["measure"], fontsize=9)
    ax.set_xlabel("Share of Variant Messages")
    ax.set_title("Mutation Frequency: Misinformation vs Non-Misinformation Families", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
