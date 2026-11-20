"""Run the whole project from the command line.

    python run_pipeline.py                      # every stage
    python run_pipeline.py --stage prepare      # one stage
    python run_pipeline.py --no-embeddings      # TF-IDF only (fast, no model download)
    python run_pipeline.py --stage translate    # optional English examples

Stages: prepare -> cluster -> evaluate -> mutations -> analyze -> extras -> plots
"""
import argparse
import time
import pandas as pd

from src import config as C
from src import analysis, cluster, data, evaluate, extras, misinformation, mutations, plots

CLUSTERED = C.PROC / "messages_clustered.csv"
PAIRS = C.PROC / "mutation_pairs_final.csv"


def read_clustered():
    return pd.read_csv(CLUSTERED, parse_dates=["first_dt"])


def read_pairs():
    return pd.read_csv(PAIRS)


def stage_prepare(**_):
    n_raw, d = data.load_messages()
    agg = data.aggregate(d)
    agg_f, stats = data.remove_spam(agg)
    agg_f.to_csv(C.PROC / "messages_filtered.csv", index=False)
    print(f"raw messages: {n_raw}")
    print(f"after removing media rows and chat noise: {len(d)} messages")
    print(f"unique cleaned texts: {len(agg)}")
    print(f"removed spam: {stats['spam']}  removed ASCII art: {stats['art']}  kept: {stats['kept']}")
    print(f"covid-related: {int(agg_f['is_covid'].sum())}")


def stage_cluster(no_embeddings=False, **_):
    agg = pd.read_csv(C.PROC / "messages_filtered.csv", parse_dates=["first_dt"])
    agg = cluster.add_clusters(agg, use_embeddings=not no_embeddings)
    agg.to_csv(CLUSTERED, index=False)
    for m in ["tfidf"] + ([] if no_embeddings else ["minilm"]):
        sz = agg[f"size_{m}"]
        n_fam = agg.loc[sz >= 2, f"cl_{m}"].nunique()
        print(f"[{m}] families (2+ messages): {n_fam}, messages in them: {int((sz >= 2).sum())}, "
              f"largest: {int(sz.max())}")


def stage_evaluate(no_embeddings=False, **_):
    agg = read_clustered()
    res, synth = evaluate.run_benchmark(agg, use_embeddings=not no_embeddings and "cl_minilm" in agg)
    res.to_csv(C.OUT / "synthetic_eval.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(synth)} planted variants")
    print(res.to_string(index=False))


def stage_mutations(**_):
    agg = read_clustered()
    pairs = mutations.build_pairs(agg)
    pairs.to_csv(PAIRS, index=False)
    shares = mutations.mutation_shares(pairs)
    shares.round(3).to_csv(C.OUT / "mutation_shares.csv", header=["share"])
    print(f"families: {pairs['cluster'].nunique()}, variant pairs: {len(pairs)}")
    print(shares.round(3).to_string())


def stage_analyze(**_):
    p = read_pairs()
    pd.set_option("display.width", 250)
    st = analysis.spread_tests(p)
    st.to_csv(C.OUT / "spread_tests.csv", index=False)
    print("\n== edits vs. sharings ==\n", st.to_string(index=False))
    print("\n== can edits predict spread? (AUC, 0.5 = guessing) ==\n", analysis.predict_auc(p))
    wf, timing = analysis.within_family(p)
    wf.to_csv(C.OUT / "within_family.csv", index=False)
    print("\n== within-family, controlling for timing and length ==")
    print("later variants vs relative sharings:", timing)
    print(wf.to_string(index=False))
    ct = analysis.covid_tests(p)
    ct.to_csv(C.OUT / "covid_tests.csv", index=False)
    print("\n== COVID vs non-COVID families ==\n", ct.to_string(index=False))
    print("\nmedian original length (words):\n", p.groupby("covid_family")["root_len"].median())


def stage_plots(**_):
    agg = read_clustered()
    plots.mutation_chart(mutations.mutation_shares(read_pairs()))
    bench = C.OUT / "synthetic_eval.csv"
    if bench.exists():
        plots.benchmark_chart(pd.read_csv(bench))
    plots.lineage_figure(agg)
    print("figures saved in", C.OUT)


def stage_extras(no_embeddings=False, **_):
    pd.set_option("display.width", 250)
    agg = read_clustered()
    dv = extras.validate_detectors(agg)
    dv.to_csv(C.OUT / "detector_validation.csv", index=False)
    print("\n== mutation detectors on planted variants (known truth) ==\n", dv.to_string(index=False))
    ts = extras.threshold_sensitivity(agg)
    ts.to_csv(C.OUT / "threshold_sensitivity.csv", index=False)
    print("\n== sensitivity to the TF-IDF threshold ==\n", ts.to_string(index=False))
    if no_embeddings or "cl_minilm" not in agg:
        print("\n(skipping hybrid clustering: needs embeddings)")
        return
    hy = extras.hybrid_benchmark(agg)
    hy.to_csv(C.OUT / "hybrid_eval.csv", index=False)
    print("\n== hybrid clustering (tfidf OR minilm) ==\n", hy.to_string(index=False))


def stage_translate(**_):
    from src import translate
    agg, pairs = read_clustered(), read_pairs()
    fam, pairs = translate.add_english(agg, pairs)
    fam.to_csv(C.PROC / "families_english.csv", index=False)
    pairs.to_csv(C.PROC / "mutation_pairs_en.csv", index=False)
    for title, mask in [("header added", pairs["prefix_added"]),
                        ("footer added", pairs["suffix_added"]),
                        ("numbers changed", pairs["numbers_changed"])]:
        print("\n#####", title)
        translate.english_examples(pairs, mask)


def stage_misinformation(**_):
    pd.set_option("display.width", 250)
    agg, pairs = read_clustered(), read_pairs()
    prop_df, fam_misinfo = misinformation.analyze_propagation(agg)
    prop_df.to_csv(C.OUT / "misinformation_propagation.csv", index=False)
    print("\n== Misinformation Fact-Checking Propagation ==\n", prop_df.to_string(index=False))

    test_df = misinformation.analyze_misinfo_mutations(pairs, fam_misinfo)
    test_df.to_csv(C.OUT / "misinformation_tests.csv", index=False)
    print("\n== Misinformation vs Non-Misinformation Mutations ==\n", test_df.to_string(index=False))

    misinformation.plot_misinfo_comparison(test_df)
    print("saved figure to", C.OUT / "misinformation_mutations.png")


STAGES = {"prepare": stage_prepare, "cluster": stage_cluster, "evaluate": stage_evaluate,
          "mutations": stage_mutations, "analyze": stage_analyze, "extras": stage_extras,
          "misinformation": stage_misinformation, "plots": stage_plots,
          "translate": stage_translate}
DEFAULT = ["prepare", "cluster", "evaluate", "mutations", "analyze", "extras", "misinformation", "plots"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    ap.add_argument("--no-embeddings", action="store_true",
                    help="skip the MiniLM comparison (no model download, much faster)")
    a = ap.parse_args()
    C.PROC.mkdir(parents=True, exist_ok=True)
    C.OUT.mkdir(parents=True, exist_ok=True)
    for name in (DEFAULT if a.stage == "all" else [a.stage]):
        t0 = time.time()
        print(f"\n=========== {name} ===========")
        STAGES[name](no_embeddings=a.no_embeddings)
        print(f"[{name}] finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
