"""Load the raw COVID19.BR file, clean it and collapse exact duplicates."""
import re
import pandas as pd
from . import config as C

HIDDEN_MEDIA = "<Arquivo de mídia oculto>"
_URL = re.compile(r"https?://\S+|www\.\S+")
_NOISE = re.compile(r"[^\w\s.,;:!?()%$/-]")      # emojis, stars, ASCII art
_WS = re.compile(r"\s+")


def clean_text(t):
    """Lower-case, drop URLs and symbol noise, collapse whitespace."""
    t = _URL.sub(" ", str(t))
    t = _NOISE.sub(" ", t)
    return _WS.sub(" ", t).strip().lower()


def alpha_ratio(t):
    return sum(c.isalpha() for c in t) / max(len(t), 1)


def load_messages(path=C.RAW_CSV):
    """Read raw data, drop media rows and short chat messages."""
    raw = pd.read_csv(path)
    d = raw[(raw["midia"] == 0) & (raw["text"] != HIDDEN_MEDIA)]
    d = d[d["words"] >= C.MIN_WORDS].copy()
    d["clean"] = d["text"].map(clean_text)
    d = d[d["clean"].str.split().str.len() >= C.MIN_WORDS].copy()
    d["dt"] = pd.to_datetime(d["date"] + " " + d["hour"], format="%d/%m/%y %H:%M")
    return len(raw), d


def aggregate(d):
    """One row per unique cleaned text, keeping spread information."""
    agg = (d.groupby("clean")
             .agg(text=("text", "first"),
                  n_occurrences=("id", "size"),
                  n_groups=("group", "nunique"),
                  first_dt=("dt", "min"),
                  max_sharings=("sharings", "max"),
                  viral=("viral", "max"))
             .reset_index())
    agg["is_covid"] = agg["clean"].str.contains(C.COVID_RE, regex=True)
    return agg


def remove_spam(agg):
    """Drop group-invite spam and ASCII-art posts."""
    spam = agg["text"].str.contains(C.SPAM_RE, case=False, regex=True)
    art = agg["text"].map(alpha_ratio) < 0.5
    keep = ~(spam | art)
    stats = dict(spam=int(spam.sum()), art=int(art.sum()), kept=int(keep.sum()))
    return agg[keep].reset_index(drop=True), stats
