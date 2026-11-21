"""Figures for the report."""
import difflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as C


def mutation_chart(shares, path=C.OUT / "mutation_types.png"):
    ax = shares.plot.barh(figsize=(7, 4.5))
    ax.set_xlabel("share of variant messages")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def benchmark_chart(results, path=C.OUT / "benchmark_recall.png"):
    meta = {"method", "th", "precision", "recall", "ARI"}
    muts = [c for c in results.columns if c not in meta]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(results)
    for i, (_, r) in enumerate(results.iterrows()):
        ax.bar(np.arange(len(muts)) + i * width, [r[m] for m in muts], width,
               label=f"{r['method']} {r['th']}")
    ax.set_xticks(np.arange(len(muts)) + 0.4 - width / 2)
    ax.set_xticklabels(muts, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("share of planted variants recovered")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _lineage(g):
    g = g.sort_values("first_dt").reset_index(drop=True)
    parents = [-1]
    for i in range(1, len(g)):
        sims = [difflib.SequenceMatcher(None, g.loc[j, "clean"].split(),
                                        g.loc[i, "clean"].split(), autojunk=False).ratio()
                for j in range(i)]
        parents.append(int(np.argmax(sims)))
    g["parent"] = parents
    g["x"] = mdates.date2num(g["first_dt"])
    return g


def lineage_figure(agg, cl="cl_tfidf", sz="size_tfidf", path=C.OUT / "lineage_examples.png"):
    fam = agg[agg[sz] >= 2]
    ids = fam.groupby(cl).size().sort_values(ascending=False).index[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, cid in zip(axes.ravel(), ids):
        g = _lineage(fam[fam[cl] == cid])
        for i, r in g.iterrows():
            if r["parent"] >= 0:
                p = g.loc[int(r["parent"])]
                ax.plot([p["x"], r["x"]], [int(r["parent"]), i], color="gray", lw=1, zorder=1)
        ax.scatter(g["x"], range(len(g)), s=30 + 40 * g["max_sharings"], zorder=2)
        ax.set_title(f"family {cid}: {len(g)} messages")
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"#{i}" for i in range(len(g))], fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.suptitle("Approximate lineage: each message linked to its most similar earlier message "
                 "(dot size = sharings)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
