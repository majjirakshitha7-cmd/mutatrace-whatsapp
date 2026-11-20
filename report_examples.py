import re, difflib
import pandas as pd
from src import config as C
def _numbers_replaced(a, b):
    ta, tb = a.split(), b.split()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "replace" and NUM.search(" ".join(ta[i1:i2])) and NUM.search(" ".join(tb[j1:j2])):
            return True
    return False
from src.mutations import NUM

pairs = pd.read_csv(C.PROC / "mutation_pairs_en.csv").dropna(subset=["root_en", "var_en"])
BAD = re.compile(r"fuck|shit|bitch|asshole|idiot|stupid|porn|whore", re.I)
pairs = pairs[~(pairs["root_en"].str.contains(BAD) | pairs["var_en"].str.contains(BAD))]
pairs = pairs[pairs["root_len"] <= 80].copy()
pairs["numbers_strict"] = [_numbers_replaced(a, b) for a, b in zip(pairs["root_text"], pairs["var_text"])]


def ops(a, b, k=2):
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    out = [f"{tag}: {' '.join(ta[i1:i2])[:100]!r} -> {' '.join(tb[j1:j2])[:100]!r}"
           for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    return out[:k]


def show(title, mask, k=3, seed=5, numbers=False):
    sub = pairs[mask]
    print(f"\n##### {title}  ({len(sub)} candidates)")
    for _, r in sub.sample(min(k, len(sub)), random_state=seed).iterrows():
        print(f"-- family {r['cluster']}")
        if numbers:
            print("   numbers in original:", sorted(NUM.findall(r["root_text"])))
            print("   numbers in variant :", sorted(NUM.findall(r["var_text"])))
            print("   context:", r["root_en"][:160])
        else:
            for o in ops(r["root_en"], r["var_en"]):
                print("  ", o)


show("header added", pairs["prefix_added"] & pairs["words_added"].between(2, 15))
show("footer added", pairs["suffix_added"] & pairs["words_added"].between(2, 15))
show("numbers replaced (strict rule)", pairs["numbers_strict"], numbers=True)
show("sentence/words dropped", (pairs["words_removed"] >= 5) & ~pairs["prefix_added"] & ~pairs["suffix_added"])
show("words substituted", pairs["real_rewrites"].between(1, 6) & (pairs["words_removed"] == 0))