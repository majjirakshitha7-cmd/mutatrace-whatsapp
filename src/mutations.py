"""Compare every variant with the earliest message of its family and
describe how it was edited."""
import difflib
import re
import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config as C

NUM = re.compile(r"\d+(?:[.,]\d+)?")
URL = re.compile(r"https?://\S+|www\.\S+")


def diff_features(a, b):
    """Word-level differences between original `a` and variant `b`."""
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    ins = dele = rep = 0
    ops = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in ops:
        if tag == "insert":
            ins += j2 - j1
        elif tag == "delete":
            dele += i2 - i1
        elif tag == "replace":
            rep += max(i2 - i1, j2 - j1)
    prefix = ops[0][0] == "insert" and ops[0][1] == 0
    suffix = ops[-1][0] == "insert" and ops[-1][1] == len(ta)
    return ins, dele, rep, prefix, suffix, sm.ratio()


def edit_kinds(a, b):
    """Split replacements into spelling-level fixes and genuine rewrites."""
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    typo = real = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            old, new = " ".join(ta[i1:i2]), " ".join(tb[j1:j2])
            if difflib.SequenceMatcher(None, old, new).ratio() >= 0.8:
                typo += 1
            else:
                real += max(i2 - i1, j2 - j1)
    return typo, real


def numbers_replaced(a, b):
    """Strict rule: a number token is replaced by another number token."""
    ta, tb = a.split(), b.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "replace" and NUM.search(" ".join(ta[i1:i2])) and NUM.search(" ".join(tb[j1:j2])):
            return True
    return False


def build_pairs(agg, cl="cl_tfidf", sz="size_tfidf"):
    fam = agg[agg[sz] >= 2]
    covid = fam.groupby(cl)["is_covid"].any()
    rows = []
    for cid, g in fam.groupby(cl):
        g = g.sort_values("first_dt")
        root = g.iloc[0]
        for k, (_, r) in enumerate(g.iloc[1:].iterrows(), start=1):
            ins, dele, rep, prefix, suffix, sim = diff_features(root["clean"], r["clean"])
            typo, real = edit_kinds(root["clean"], r["clean"])
            na, nb = set(NUM.findall(root["clean"])), set(NUM.findall(r["clean"]))
            rows.append(dict(
                cluster=cid, family_size=len(g), order_in_family=k,
                root_text=root["clean"], var_text=r["clean"],
                words_added=ins, words_removed=dele, words_replaced=rep,
                typo_fixes=typo, real_rewrites=real,
                prefix_added=prefix, suffix_added=suffix, similarity=sim,
                numbers_changed=numbers_replaced(root["clean"], r["clean"]),
                url_added=bool(URL.search(r["text"])) and not bool(URL.search(root["text"])),
                n_groups=r["n_groups"], max_sharings=r["max_sharings"],
                days_since_root=(r["first_dt"] - root["first_dt"]).total_seconds() / 86400,
                root_len=len(root["clean"].split()),
                covid_family=bool(covid[cid]),
            ))
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p["edit_size"] = 1 - p["similarity"]
    p["rewrite_rate"] = p["real_rewrites"] / p["root_len"]
    p["add_rate"] = p["words_added"] / p["root_len"]
    p["drop_rate"] = p["words_removed"] / p["root_len"]
    r_has = p["root_text"].str.contains(C.CTA_RE, regex=True)
    v_has = p["var_text"].str.contains(C.CTA_RE, regex=True)
    p["cta_added"] = v_has & ~r_has
    p["cta_removed"] = r_has & ~v_has
    p["viral_variant"] = (p["max_sharings"] >= 2).astype(int)
    p["sharings_rel"] = p["max_sharings"] - p.groupby("cluster")["max_sharings"].transform("mean")
    return p


def mutation_shares(p):
    return pd.Series({
        "numbers changed": p["numbers_changed"].mean(),
        "header added (start)": p["prefix_added"].mean(),
        "footer added (end)": p["suffix_added"].mean(),
        "link added": p["url_added"].mean(),
        "urgency/share wording added": p["cta_added"].mean(),
        "words dropped (>3)": (p["words_removed"] > 3).mean(),
        "words added (>3)": (p["words_added"] > 3).mean(),
        "words rewritten (>3)": (p["real_rewrites"] > 3).mean(),
        "spelling-level fixes": (p["typo_fixes"] > 0).mean(),
    }).sort_values()
