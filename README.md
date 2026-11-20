# Mutation Analysis of Forwarded WhatsApp Messages

NLP capstone project. Forwarded messages change as they are copied: words are dropped,
numbers altered, headers and footers added. This project

1. groups near-duplicate messages into **variant families**,
2. measures **how each variant was edited** compared with the earliest message of its family,
3. tests whether the type or size of an edit is **associated with how widely the message circulated**.

Data: [COVID19.BR](https://zenodo.org/records/5193932) (WhatsApp messages from public groups in Brazil,
Oct 2018 - Jun 2020). The text is Brazilian Portuguese; examples can be machine-translated to English
(`--stage translate`). All numbers are computed on the original text.

## Project structure

```
run_pipeline.py        one command runs every stage
build_explorer.py      generates standalone interactive HTML explorer
src/
  config.py            paths, thresholds, regex patterns
  data.py              load, clean, collapse duplicates, remove spam
  cluster.py           TF-IDF / MiniLM embeddings, mutual-nearest-neighbour clustering
  evaluate.py          synthetic benchmark (planted mutations with known answers)
  mutations.py         word-level diff features per variant
  analysis.py          statistical tests, spread prediction, COVID comparison
  misinformation.py    fact-checking propagation and misinformation mutation tests
  plots.py             report figures
  extras.py            detector validation, hybrid clustering, threshold sensitivity
  translate.py         optional English translation of examples
tests/                 unit tests
data/raw/              put covid19br.csv and covidbr_labeled.csv here
data/processed/        intermediate files (created by the pipeline)
outputs/               tables, figures, and variant_explorer.html
notebooks/             exploration notebooks (optional)
REPORT.md              complete academic capstone report
```

## Setup (Linux / Ubuntu)

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Download `covid19br.csv` from the Zenodo link above into `data/raw/`.

## Run

```
python run_pipeline.py                    # all stages (downloads the MiniLM model on first run)
python run_pipeline.py --no-embeddings    # TF-IDF only: fast, no model download
python run_pipeline.py --stage analyze    # a single stage
python run_pipeline.py --stage translate  # optional: English examples (needs transformers, sentencepiece)
```

Stages: `prepare -> cluster -> evaluate -> mutations -> analyze -> extras -> misinformation -> plots`.
Each stage reads the previous stage's file from `data/processed/`, so you can re-run one stage alone.

## Method summary

- **Cleaning:** drop media placeholders and messages under 10 words; remove URLs and symbols;
  collapse exact duplicates (keeping counts, groups, sharings, first-seen timestamp);
  remove group-invite spam and ASCII-art posts.
- **Clustering:** two messages are linked only if each is among the other's 10 nearest neighbours
  *and* cosine similarity passes a threshold; connected components are families. TF-IDF (word 1-2 grams)
  at 0.6 is the main method; multilingual MiniLM embeddings at 0.90 are the comparison.
- **Evaluation:** no labelled families exist, so 400 real messages get 7 kinds of planted edits
  (numbers, footer, header, 15% / 30% word drop, truncation, word swaps). A method is scored on whether
  each variant lands in the same cluster as its source (precision, recall, ARI, recall per edit type).
- **Mutation features:** difflib word diffs against the earliest message (by date and hour) in the family.
- **Statistics:** Mann-Whitney / Spearman for edits vs `sharings`; family-grouped cross-validated AUC;
  partial Spearman controlling for time since the original and length, with a family-level bootstrap CI;
  Fisher and length-fair tests for COVID vs non-COVID families.
- **Misinformation & Fact-Checking:** matches variant families against 2,899 fact-checked messages
  (`covidbr_labeled.csv`), uncovering a +22.1% fact-check coverage boost and finding misinformation variants
  are 2.67x more likely to add urgency CTAs and 1.79x more likely to prepend authoritative headers.

## Results Summary

All experimental outputs are saved in `outputs/`:
- `synthetic_eval.csv` & `benchmark_recall.png`: TF-IDF 0.6 achieves 1.0 precision, 0.965 recall, 0.982 ARI.
- `mutation_shares.csv` & `mutation_types.png`: 47.9% rewrites, 38.7% typos, 17.9% headers, 15.8% footers, 14.1% numbers.
- `spread_tests.csv` & `within_family.csv`: Edits do not predict virality (AUC ~ 0.50); rewrites correlate negatively with relative within-family spread (partial rho = -0.099).
- `covid_tests.csv`: COVID-19 messages show 2.18x more headers added and 2.29x more words added.
- `misinformation_propagation.csv` & `misinformation_tests.csv`: +22.1% fact-check coverage boost; urgency CTAs added at OR = 2.67 (p = 0.0066).
- `detector_validation.csv` & `hybrid_eval.csv`: Strict number detector achieves 100% precision & 100% recall.
- `variant_explorer.html`: Interactive browser dashboard to search and view word-level diffs across families.
- Full details in `REPORT.md`.

## Limitations

- Data is Portuguese; the translation stage is machine translation and only for illustration.
- `sharings` is the dataset authors' spread count; its exact definition should be checked at the source.
- Mutation types are automatic, overlapping rules and were not manually validated.
- The synthetic benchmark plants lexical edits, which favours TF-IDF over embeddings; it does not test paraphrase.
- Lineage trees are approximate (most similar earlier message, not proven copying).
- The COVID flag is a keyword match; many statistical tests were run, so single small p-values are exploratory.

## Future work

Replicate on MultiClaimNet or English/Telugu data, add a paraphrase benchmark for the embedding model,
and validate the mutation-type rules against a manually labelled sample.
