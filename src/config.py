"""Central configuration: paths, thresholds and regular expressions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "covid19br.csv"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

SEED = 0
MIN_WORDS = 10                      # shortest message kept (after cleaning)

# clustering
EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
K_NEIGHBORS = 10
TFIDF_TH = 0.6                      # main method
EMB_TH = 0.90                       # embedding comparison

# synthetic benchmark
N_PLANTED = 400
BENCH_TFIDF_TH = (0.5, 0.6, 0.7)
BENCH_EMB_TH = (0.85, 0.90, 0.95)

# text patterns (Brazilian Portuguese)
SPAM_RE = r"chat\.whatsapp\.com|bem[- ]vind|link.{0,40}grupo|grupo.{0,40}link|entrar no (?:meu )?grupo"
COVID_RE = r"covid|corona|quarentena|pandemia|cloroquina|vacina|isolamento"
CTA_RE = r"urgente|compartilh|divulg|repasse|repassem|espalh|bomba|vazou|aten[cç][aã]o|vejam|assistam"

# statistics
BOOTSTRAP_B = 300
