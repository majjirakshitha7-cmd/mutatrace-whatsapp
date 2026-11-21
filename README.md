# Mutation Analysis of Forwarded WhatsApp Messages
### Clustering, Diffusion Dynamics, and Fact-Checking Evasion in Encrypted Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%20%7C%20VS%20Code-007ACC.svg)](https://code.visualstudio.com/)
[![Dataset: COVID19.BR](https://img.shields.io/badge/Data-Zenodo%205193932-green.svg)](https://zenodo.org/records/5193932)
[![Report: PDF](https://img.shields.io/badge/Report-Report.pdf-red.svg)](Report.pdf)

> **Academic Capstone Project in Natural Language Processing**  
> **Author:** M Rakshitha (Roll No: 524cs0011)  
> **Full Report:** [NLP_report.pdf](NLP_report.pdf) *(50-page complete capstone report with mathematical formulations, literature review, and extended discussion)*

---

## Table of Contents

- [Overview & Research Motivation](#overview--research-motivation)
- [Core Research Questions](#core-research-questions)
- [Repository Structure & Architecture](#repository-structure--architecture)
- [Datasets & Preprocessing Funnel](#datasets--preprocessing-funnel)
- [Methodology Pipeline](#methodology-pipeline)
  - [1. Mutual $k$-NN Graph Clustering](#1-mutual-k-nn-graph-clustering)
  - [2. Synthetic Mutation Ground-Truth Benchmark](#2-synthetic-mutation-ground-truth-benchmark)
  - [3. Word-Level Mutation Extraction](#3-word-level-mutation-extraction)
  - [4. Statistical & Virality Modeling](#4-statistical--virality-modeling)
  - [5. Misinformation & Fact-Check Propagation](#5-misinformation--fact-check-propagation)
- [Key Empirical Results](#key-empirical-results)
- [Setup & Installation Guide (Windows VS Code Terminal)](#setup--installation-guide-windows-vs-code-terminal)
- [Execution & CLI Command Reference](#execution--cli-command-reference)
  - [Running the Pipeline](#running-the-pipeline)
  - [Interactive HTML Variant Explorer](#interactive-html-variant-explorer)
  - [Inspecting Sample Variant Pairs](#inspecting-sample-variant-pairs)
  - [Running the Automated Test Suite](#running-the-automated-test-suite)
  - [Jupyter Exploration Notebooks](#jupyter-exploration-notebooks)
- [Generated Outputs & Visualizations](#generated-outputs--visualizations)
- [Windows-Specific Troubleshooting & Tips](#windows-specific-troubleshooting--tips)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview & Research Motivation

In encrypted peer-to-peer messaging applications such as WhatsApp, information diffusion is decentralized, private, and unmonitored. As viral messages circulate between public chat groups and personal networks, they undergo continual lexical and structural mutation:
- **Words are truncated** or deleted to alter tone and length.
- **Critical numerical values** (dates, death tolls, stimulus check amounts) are altered or exaggerated.
- **Calls-to-action (CTAs)** (*"Share before this gets deleted!"*) and urgency markers are injected.
- **Institutional headers** (*"According to Johns Hopkins...", "Globo News reports..."*) are prepended to manufacture artificial credibility.

Because automated debunking systems rely heavily on exact cryptographic hashing (MD5, SHA-256) or strict text-matching algorithms, even trivial copy-paste mutations evade detection, allowing debunked misinformation to recirculate unimpeded.

This repository provides an end-to-end NLP pipeline to:
1. Reconstruct message mutation cascades by clustering near-duplicate messages into **variant families**.
2. Quantify fine-grained, word-level edit patterns across families.
3. Test whether specific textual mutations drive message virality.
4. Measure how variant family clustering recovers unflagged misinformation variants that evade direct fact-checking repositories.

---

## Core Research Questions

- **RQ1 (Mutation Taxonomy):** What types and frequencies of edits naturally emerge during peer-to-peer forwarding cascades?
- **RQ2 (Diffusion Dynamics & Virality):** Are specific edits (e.g., adding urgency markers or altering numbers) associated with higher circulation, or does message virality follow a first-mover advantage?
- **RQ3 (Health-Crisis Dynamics):** Did messages circulating during the COVID-19 pandemic undergo different mutation patterns than general discourse?
- **RQ4 (Fact-Checking Evasion):** Can mutual-graph variant clustering bridge the detection gap and recover mutated misinformation variants that slipped through direct fact-check matching?

---

## Repository Structure & Architecture

```
nlp_capstone_whatsapp_mutation/
│
├── Report.pdf                 # Complete academic capstone report (PDF)
├── requirements.txt           # Core Python dependencies
├── run_pipeline.py            # Main CLI runner orchestrating all pipeline stages
├── build_explorer.py          # Builds standalone interactive HTML diff explorer
├── report_examples.py         # Inspects representative mutated message pairs by edit type
├── .gitignore                 # Configured git ignore rules
│
├── src/                       # Modular source code
│   ├── config.py              # Central configuration: filepaths, hyperparameters, regexes
│   ├── data.py                # Raw ingestion, cleaning, deduplication, spam filtering
│   ├── cluster.py             # TF-IDF / MiniLM embeddings & mutual-kNN graph clustering
│   ├── evaluate.py            # Synthetic mutation benchmark generator & scoring
│   ├── mutations.py           # Word-level diff parsing & mutation feature extraction
│   ├── analysis.py            # Statistical tests, GroupKFold AUC, within-family bootstrap CI
│   ├── misinformation.py      # Fact-check propagation & misinformation mutation analysis
│   ├── plots.py               # Generates high-res report figures & lineage trees
│   ├── extras.py              # Detector validation, threshold sensitivity, hybrid clustering
│   └── translate.py           # MarianMT machine-translation pipeline (PT -> EN examples)
│
├── notebooks/                 # Interactive Jupyter notebooks for step-by-step exploration
│   ├── 01_explore.ipynb       # Raw data exploration, distributions, and viral flags
│   ├── 02_cleaning.ipynb      # Message filtering, URL removal, duplicate collapsing
│   ├── 03_clustering.ipynb    # Distance metrics, kNN graph construction, component analysis
│   └── 04_mutations.ipynb     # Diff extraction, virality modeling, statistical tests
│
├── data/
│   ├── raw/                   # Place raw datasets here (covid19br.csv, covidbr_labeled.csv)
│   └── processed/             # Bundled preprocessed datasets (ready for instant execution)
│       ├── messages_filtered.csv       # Cleaned and deduplicated corpus (17,672 rows)
│       ├── messages_clustered.csv      # Corpus mapped to variant family cluster IDs
│       ├── mutation_pairs_final.csv    # 984 variant pairs with complete diff features
│       ├── families_english.csv        # Portuguese messages paired with English translations
│       └── mutation_pairs_en.csv       # Variant pairs with English text and diff annotations
│
├── outputs/                   # Generated empirical results, tables, and figures
│   ├── variant_explorer.html           # Standalone interactive browser dashboard
│   ├── benchmark_recall.png            # Planted benchmark recall breakdown figure
│   ├── lineage_examples.png            # Approximate forwarding cascade lineage trees
│   ├── mutation_types.png              # Horizontal bar chart of mutation frequency
│   ├── misinformation_mutations.png    # Misinformation vs non-misinformation edit comparison
│   ├── synthetic_eval.csv              # Benchmark metrics (Precision, Recall, ARI per edit)
│   ├── mutation_shares.csv             # Overall mutation category percentages
│   ├── spread_tests.csv                # Mann-Whitney & Spearman tests against virality
│   ├── within_family.csv               # Partial Spearman tests with bootstrap CI
│   ├── covid_tests.csv                 # Fisher's exact tests for COVID vs non-COVID
│   ├── misinformation_propagation.csv  # Fact-check coverage recovery boost metrics
│   ├── misinformation_tests.csv        # Misinformation mutation odds ratios and p-values
│   ├── detector_validation.csv         # Precision, recall & FPR of mutation detectors
│   ├── hybrid_eval.csv                 # Benchmark comparison of TF-IDF, MiniLM, and Hybrid
│   ├── threshold_sensitivity.csv       # Sensitivity of cluster count and correlations to theta
│   └── examples_en.txt                 # Clean sample text pairs per mutation category
│
└── tests/                     # Automated unit test suite
    ├── __init__.py
    └── test_basic.py          # Sanitization, diff logic, mutual-kNN, propagation tests
```

---

## Datasets & Preprocessing Funnel

### 1. Primary Corpus: COVID19.BR
- **Source:** [Zenodo Record 5193932](https://zenodo.org/records/5193932)
- **Scope:** 125,000+ forwarded messages collected from public Brazilian WhatsApp groups between October 2018 and June 2020.
- **Language:** Brazilian Portuguese (`pt-BR`).

### 2. Fact-Checking Ground Truth: `covidbr_labeled.csv`
- **Scope:** 2,899 verified messages cross-referenced against Brazilian fact-checking agencies (Aos Fatos, Agência Lupa).
- **Labels:** Binary misinformation label ($n = 912$ misinformation, $n = 1,987$ non-misinformation) and direct debunking URLs.

### Preprocessing Funnel Summary

```
Raw Export (125k+ messages)
   │
   ▼ Drop media placeholders (<Arquivo de mídia oculto>), system notes, and texts < 10 words
Filtered Text Corpus (82,416 messages)
   │
   ▼ Strip URLs, emojis, and punctuation noise; collapse exact duplicates
Collapsed Unique Texts (18,129 unique texts with sharing counts & timestamps)
   │
   ▼ Filter group-invite spam (chat.whatsapp.com) and ASCII-art posts (alpha ratio < 0.5)
Cleaned Analysis Corpus (17,672 texts)
   │
   ▼ Mutual k-NN graph clustering (TF-IDF θ = 0.60, k = 10)
604 Variant Families (1,588 messages | 984 directed variant pairs)
```

---

## Methodology Pipeline

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│  data.py        │     │  cluster.py      │     │  evaluate.py           │
│  Ingest, clean, │ ──> │  TF-IDF & MiniLM │ ──> │  400-message planted   │
│  spam filtering │     │  Mutual k-NN     │     │  synthetic benchmark   │
└─────────────────┘     └──────────────────┘     └────────────────────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐
│  analysis.py    │     │  mutations.py    │     │  misinformation.py     │
│  Virality AUC,  │ <── │  Word-level diff │ ──> │  Coverage boost (+22%) │
│  partial rho CI │     │  features (984)  │     │  Tactical odds ratios  │
└─────────────────┘     └──────────────────┘     └────────────────────────┘
```

### 1. Mutual $k$-NN Graph Clustering
Standard single-linkage clustering suffers from severe **transitive semantic drift** (chaining unrelated messages together via intermediate bridges). To eliminate chaining, we enforce a **symmetric mutual nearest-neighbor constraint**:
1. Compute pairwise cosine distance across sublinear word/char 1–2 gram TF-IDF vectors.
2. Build a directed graph $G = (V, E)$ where $(u, v) \in E \iff v \in \text{Top-}k(u) \text{ and } \text{cosine\_sim}(u, v) \ge \theta$.
3. Retain only mutual edges: $E_{\text{mutual}} = E \cap E^T$.
4. Connected components in $G_{\text{mutual}}$ define the variant families.

### 2. Synthetic Mutation Ground-Truth Benchmark
Because natural messaging cascades lack exhaustive ground-truth family boundaries, we construct an automated evaluation benchmark:
- Sample 400 isolated messages from the corpus.
- Apply 7 realistic mutation operators:
  1. `numbers_changed`: Add random offset to numerical tokens.
  2. `header_added`: Prepend urgency call (`"atenção urgente leia até o final"`).
  3. `footer_added`: Append viral call-to-action (`"compartilhe com todos os seus contatos"`).
  4. `drop_15pct`: Randomly delete 15% of words.
  5. `drop_30pct`: Randomly delete 30% of words.
  6. `truncate_60pct`: Keep only the initial 60% of message words.
  7. `word_swaps`: Randomly swap adjacent tokens.
- Score models by measuring whether variants cluster with their true planted source.

### 3. Word-Level Mutation Extraction
Each family is anchored by its **root message** (earliest observed by timestamp). Every subsequent variant is compared against the root using `difflib.SequenceMatcher`:
- **Additions & Deletions:** Exact count of inserted and deleted tokens.
- **Prefix / Suffix Injections:** Positional detection of header and footer additions.
- **Typo Corrections vs Genuine Rewrites:** Replacements with character similarity $\ge 0.80$ are categorized as spelling fixes; lower similarities are categorized as semantic rewrites.
- **Strict Number Replacements:** Flagged only when a numerical token is directly replaced by a different numerical token.

### 4. Statistical & Virality Modeling
- **GroupKFold Virality Prediction:** Evaluates whether edit features predict message virality ($\ge 2$ sharings) using Logistic Regression and Gradient Boosted Trees, grouped by family ID to prevent leakage.
- **Partial Spearman Rank Correlation:** Measures association between edit rates and within-family relative spread ($\text{sharings}_{\text{variant}} - \overline{\text{sharings}}_{\text{family}}$), rigorously controlling for days elapsed since root and message character length.
- **Cluster Bootstrap CI:** Computes 95% confidence intervals via 300 family-level bootstrap resamplings.

### 5. Misinformation & Fact-Check Propagation
Variant families are cross-referenced with Brazilian fact-checking records (`covidbr_labeled.csv`). When a verified debunked claim matches any variant in a family, the label propagates to all co-members. We measure:
- **Coverage Recovery:** Percentage increase in detected misinformation messages.
- **Tactical Feature Analysis:** Fisher's exact tests and odds ratios comparing mutation tactics between misinformation and non-misinformation families.

---

## Key Empirical Results

### 1. Synthetic Benchmark Evaluation

| Method | Threshold $\theta$ | Precision | Recall | ARI | Truncate (60%) Recall | Word Swaps Recall | Numbers Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (1–2 grams)** | **0.60** | **1.000** | **0.965** | **0.982** | **1.00** | **0.99** | **1.00** |
| TF-IDF (1–2 grams) | 0.50 | 1.000 | 0.992 | 0.996 | 1.00 | 1.00 | 1.00 |
| TF-IDF (1–2 grams) | 0.70 | 1.000 | 0.863 | 0.926 | 0.97 | 0.91 | 1.00 |
| Multilingual MiniLM | 0.85 | 0.968 | 0.837 | 0.898 | 0.72 | 1.00 | 0.98 |
| Multilingual MiniLM | 0.90 | 1.000 | 0.684 | 0.812 | 0.49 | 1.00 | 0.96 |
| Multilingual MiniLM | 0.95 | 1.000 | 0.401 | 0.572 | 0.25 | 0.96 | 0.86 |

> **Finding:** Character/word $n$-gram TF-IDF substantially outperforms dense multilingual embeddings (`MiniLM`). While embeddings excel at semantic paraphrasing, their recall collapses to **49% on truncated messages** and **53% on word deletions**, because sentence transformers map shortened texts to different regions of embedding space.

---

### 2. Natural Mutation Taxonomy (984 Variant Pairs)

| Mutation Category | Share of Variants | Description |
| :--- | :---: | :--- |
| **Semantic Rewrites (>3 words)** | **47.9%** | Substantial body rephrasing or substitution |
| **Spelling / Typo Corrections** | **38.7%** | Minor orthographic adjustments (similarity $\ge 0.80$) |
| **Words Added (>3 words)** | **29.1%** | New contextual or explanatory content inserted |
| **Words Dropped (>3 words)** | **20.2%** | Sentences or clauses deleted |
| **Header Prepending** | **17.9%** | Authority claims or alerts placed at the beginning |
| **Footer Appending** | **15.8%** | Disclaimers or share-prompts attached at the end |
| **Numbers Altered** | **14.1%** | Critical statistics, dates, or values modified |
| **Urgency / Share CTAs** | **3.6%** | Explicit viral forwarding triggers injected |
| **Links / URLs Added** | **2.7%** | External websites or group links added |

---

### 3. Diffusion Dynamics & Virality (The "Null Result")

- **Grouped Cross-Validation Virality AUC:** Both Logistic Regression ($\text{AUC} = 0.504$) and Gradient Boosting ($\text{AUC} = 0.518$) achieve near-chance performance when predicting message virality from edit features under family-grouped cross-validation.
- **Within-Family Spread Correlation:** Controlling for time elapsed and root length, variants with heavy rewrites circulate **less** than the original text:
  $$\text{Partial } \rho = -0.099 \quad (p = 0.041, \ 95\%\text{ Bootstrap CI: } [-0.152, -0.046])$$
- **Takeaway:** Forwarding virality on WhatsApp is governed by **first-mover advantage and network topology**, rather than textual optimization or deliberate mutation tuning.

---

### 4. Misinformation Evasion & Fact-Check Propagation

| Metric | Measured Value | Significance |
| :--- | :---: | :--- |
| **Fact-Checking Coverage Boost** | **+22.1%** | 58 unflagged mutated variants recovered |
| **Unflagged Variants Sharings** | **117** | Total spread of recovered misinformation variants |
| **Urgency CTA Injection** | **$\text{OR} = 2.67$** | $p = 0.0066$ (Fisher's exact test) |
| **Authority Header Injection** | **$\text{OR} = 1.79$** | $p = 0.0020$ (Fisher's exact test) |
| **Substantial Words Added (>3)** | **$\text{OR} = 2.00$** | $p < 0.0001$ (Fisher's exact test) |

> **Finding:** Family clustering provides an automated **+22.1% boost** in fact-checking coverage, successfully flagging 58 mutated messages that bypassed exact hash matching. Misinformation messages exhibit significant behavioral differences: they are **$2.67\times$ more likely** to inject urgency CTAs and **$1.79\times$ more likely** to prepend authoritative headers.

---

## Setup & Installation Guide (Windows VS Code Terminal)

This project is fully optimized for **Windows 10/11** using the integrated **VS Code Terminal** (PowerShell or Command Prompt).

### Step 1: Open VS Code Terminal
1. Open the project folder in **Visual Studio Code**.
2. Press <kbd>Ctrl</kbd> + <kbd>`</kbd> (or go to **Terminal &rarr; New Terminal**). VS Code defaults to **PowerShell**.

### Step 2: Create & Activate Virtual Environment

#### In PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **PowerShell Script Policy Notice:**  
> If PowerShell displays `PSSecurityException: File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system`, run the following command once to permit script execution for your session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\venv\Scripts\Activate.ps1
> ```

#### In Command Prompt (`cmd.exe`):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

> **VS Code Interpreter Tip:** Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd>, type `Python: Select Interpreter`, and select `.\venv\Scripts\python.exe`.

### Step 3: Install Required Packages

Upgrade `pip` and install all necessary dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional)* To run unit tests:
```powershell
pip install pytest
```

*(Optional)* To run the Portuguese-to-English translation stage:
```powershell
pip install transformers sentencepiece
```

---

## Execution & CLI Command Reference

### Instant Execution (Using Pre-bundled Data)
This repository includes precomputed CSVs in `data/processed/` and pre-generated results in `outputs/`. You do **not** need to download the 125,000 raw message dataset to run evaluations, analysis, extras, tests, or the interactive explorer!

### Re-running from Raw Data (Optional)
If you wish to re-execute stage 1 (`prepare`) from raw files:
1. Create directory `data\raw`:
   ```powershell
   New-Item -ItemType Directory -Path "data\raw" -Force
   ```
2. Download `covid19br.csv` from [Zenodo 5193932](https://zenodo.org/records/5193932) into `data\raw\`.
3. (Optional) Place `covidbr_labeled.csv` into `data\raw\`.

---

### Running the Pipeline

- **Execute All Stages** (runs complete pipeline end-to-end):
  ```powershell
  python run_pipeline.py
  ```

- **Execute Fast Mode** (TF-IDF only, skips heavy embedding download):
  ```powershell
  python run_pipeline.py --no-embeddings
  ```

- **Execute Individual Stages:**
  ```powershell
  # 1. Synthetic mutation benchmark scoring:
  python run_pipeline.py --stage evaluate

  # 2. Statistical virality analysis & COVID tests:
  python run_pipeline.py --stage analyze

  # 3. Detector validation, sensitivity & hybrid clustering:
  python run_pipeline.py --stage extras

  # 4. Fact-checking propagation & evasion tests:
  python run_pipeline.py --stage misinformation

  # 5. Generate high-resolution figures:
  python run_pipeline.py --stage plots

  # 6. (Optional) Machine translation of Portuguese samples to English:
  python run_pipeline.py --stage translate
  ```

---

### Interactive HTML Variant Explorer

Generate and view the standalone interactive browser application featuring side-by-side color-coded text diffs, variant metadata, and family filtering:

```powershell
# Build the HTML dashboard
python build_explorer.py

# Launch directly in your default web browser from PowerShell:
Start-Process .\outputs\variant_explorer.html
```

*Features of the Explorer:*
- **Real-time Search:** Filter variants by keyword or cluster ID.
- **Mutation Filters:** Filter by headers added, footers added, numbers changed, or rewrites.
- **Diff Highlighting:** Red `<del>` for deletions, green `<ins>` for insertions, and inline replacement tags.
- **Propagation Metrics:** Live display of fact-checking recovery statistics.

---

### Inspecting Sample Variant Pairs

To quickly print human-readable before-and-after text diffs for specific mutation types directly in the console:

```powershell
python report_examples.py
```

Outputs examples across:
- Headers added (*"Listen to this audio until the end..."*)
- Footers added (*"Worth listening to the audio until the end!!"*)
- Strict number replacements (e.g., date changed from `April 05, 2020` to `April 12, 2020`)
- Dropped sentences and words substituted

---

### Running the Automated Test Suite

Verify text sanitization, diff opcode extraction, mutual-kNN graph construction, and propagation metrics:

```powershell
pytest tests/
```

Or run with verbose output:
```powershell
python -m pytest tests/ -v
```

---

### Jupyter Exploration Notebooks

Interactive notebooks for visual exploration and step-by-step experimentation are located in `notebooks/`:

1. `notebooks/01_explore.ipynb`: Ingestion of raw messages, temporal distribution, and virality flags.
2. `notebooks/02_cleaning.ipynb`: Normalization, regex noise stripping, and exact duplicate collapsing.
3. `notebooks/03_clustering.ipynb`: Distance threshold sweeps, mutual nearest-neighbor graph validation.
4. `notebooks/04_mutations.ipynb`: Feature extraction, regression models, and propagation plots.

To start Jupyter from the VS Code terminal:
```powershell
pip install jupyter
jupyter notebook
```

---

## Generated Outputs & Visualizations

All pipeline artifacts are written to `outputs/`:

| Output File | Description |
| :--- | :--- |
| [`variant_explorer.html`](outputs/variant_explorer.html) | Standalone interactive diff explorer with search & filter controls |
| [`benchmark_recall.png`](outputs/benchmark_recall.png) | Bar chart comparing TF-IDF vs MiniLM across 7 planted mutation types |
| [`mutation_types.png`](outputs/mutation_types.png) | Frequency distribution of natural mutation categories across variants |
| [`lineage_examples.png`](outputs/lineage_examples.png) | Four-panel tree diagram of approximate message forwarding lineage |
| [`misinformation_mutations.png`](outputs/misinformation_mutations.png) | Odds ratio plot comparing misinformation vs non-misinformation edits |
| [`synthetic_eval.csv`](outputs/synthetic_eval.csv) | Full tabular metrics (Precision, Recall, ARI) for clustering methods |
| [`mutation_shares.csv`](outputs/mutation_shares.csv) | Quantitative percentages of mutation categories in natural pairs |
| [`spread_tests.csv`](outputs/spread_tests.csv) | Mann-Whitney and Spearman test statistics vs message spread |
| [`within_family.csv`](outputs/within_family.csv) | Partial Spearman rank correlations controlling for age and length |
| [`covid_tests.csv`](outputs/covid_tests.csv) | Fisher's exact tests comparing COVID vs non-COVID message pairs |
| [`misinformation_propagation.csv`](outputs/misinformation_propagation.csv) | Coverage recovery metrics (+22.1% boost, unflagged variants count) |
| [`misinformation_tests.csv`](outputs/misinformation_tests.csv) | Fisher odds ratios and p-values for misinformation tactics |
| [`detector_validation.csv`](outputs/detector_validation.csv) | Ground-truth validation of mutation detectors on planted benchmark |
| [`hybrid_eval.csv`](outputs/hybrid_eval.csv) | Benchmark evaluation of TF-IDF, MiniLM, and their hybrid union |
| [`threshold_sensitivity.csv`](outputs/threshold_sensitivity.csv) | Stability check across threshold choices ($\theta = 0.5, 0.6, 0.7$) |
| [`examples_en.txt`](outputs/examples_en.txt) | Formatted English translation examples for report presentation |

---

## Windows-Specific Troubleshooting & Tips

- **PowerShell Script Execution:** If `.\venv\Scripts\Activate.ps1` gives an execution policy error, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
- **UTF-8 Output & Portuguese Characters:** If special characters (`ã`, `ç`, `é`) appear garbled in the PowerShell console, enable UTF-8 output encoding:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $env:PYTHONUTF8 = "1"
  ```
- **Python Launcher:** If `python` is not mapped in your environment PATH, use the Windows Python launcher:
  ```powershell
  py -m venv venv
  py run_pipeline.py
  ```
- **PyTorch / Sentence-Transformers Wheels:** If installing `sentence-transformers` on Windows fails, ensure you install the official PyTorch CPU wheel first:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

---

## Limitations & Ethical Considerations

1. **Language Specificity:** Primary analyses are conducted on Brazilian Portuguese (`pt-BR`). While the clustering methodology is language-agnostic, regex detectors (e.g., CTA triggers) require localization for other languages.
2. **Spread Metric Interpretation:** The `sharings` variable reflects forwarded observation counts collected across monitored public groups, which acts as a proxy rather than an exhaustive census of WhatsApp private forwards.
3. **Synthetic vs Natural Paraphrasing:** The synthetic benchmark plants lexical substitutions and truncations, which naturally favors $n$-gram TF-IDF over neural embeddings; neural encoders may perform relatively better on human semantic paraphrases without word overlap.
4. **Lineage Heuristics:** In an encrypted messaging ecosystem without message IDs or causal metadata, tree lineages are estimated via greedy similarity against earlier timestamps rather than guaranteed transmission paths.
5. **Privacy:** No private chat logs were collected. All underlying data originates from publicly accessible groups archived in academic repositories under ethical research guidelines.

---

## Citation & Acknowledgments

If referencing this work or using the pipeline code, please cite the capstone report:

```bibtex
@techreport{rakshitha2026whatsappmutation,
  title       = {Mutation Analysis of Forwarded WhatsApp Messages: Clustering, Diffusion Dynamics, and Fact-Checking Evasion},
  author      = {Rakshitha, M},
  institution = {Indian Institute of Information Technology, Design and Manufacturing, Kancheepuram},
  year        = {2026},
  number      = {524cs0011},
  note        = {NLP Capstone Project Report. PDF available in repository.}
}
```

### Dataset Citation
```bibtex
@dataset{resende2021covid19br,
  author    = {Resende, Gustavo and others},
  title     = {COVID19.BR: A Dataset of WhatsApp Messages from Public Groups in Brazil},
  year      = {2021},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.5193932}
}
```

---

*For further inquiries, detailed methodology, and derivations, consult the capstone report: [NLP_report.pdf](NLP_report.pdf).*
