"""Optional: machine-translate the family messages to English so the examples
can be read in the report. Numbers in the report always come from the original text."""
import difflib
import pandas as pd
from . import config as C

MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"


def translate(texts, batch_size=16, max_chars=600):
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = MarianTokenizer.from_pretrained(MODEL)
    mt = MarianMTModel.from_pretrained(MODEL).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:max_chars] for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = mt.generate(**enc, num_beams=2, max_length=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
        if (i // batch_size) % 10 == 0:
            print(f"  translated {i}/{len(texts)}", flush=True)
    return out


def add_english(agg, pairs):
    fam = agg[agg["size_tfidf"] >= 2].copy()
    fam["clean_en"] = translate(fam["clean"].tolist())
    en = fam.set_index("clean")["clean_en"].to_dict()
    pairs = pairs.copy()
    pairs["root_en"] = pairs["root_text"].map(en)
    pairs["var_en"] = pairs["var_text"].map(en)
    return fam, pairs


def english_examples(pairs, mask, k=4, seed=3):
    """Print readable before/after word changes for a boolean mask over pairs."""
    sub = pairs[mask].dropna(subset=["root_en"])
    for _, r in sub.sample(min(k, len(sub)), random_state=seed).iterrows():
        ta, tb = r["root_en"].split(), r["var_en"].split()
        print("-- family", r["cluster"])
        sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
        for tag, i1, i2, j1, j2 in [o for o in sm.get_opcodes() if o[0] != "equal"][:3]:
            print(f"   {tag:8s} OLD: {' '.join(ta[i1:i2])[:100]!r} -> NEW: {' '.join(tb[j1:j2])[:100]!r}")
