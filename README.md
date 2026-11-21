# Mutation Analysis of Forwarded WhatsApp Messages

NLP capstone project investigating how viral forwarded messages mutate across peer-to-peer networks: words dropped, numbers altered, calls-to-action injected, and institutional headers added.

This project:
1. Groups near-duplicate messages into **variant families** via mutual $k$-nearest-neighbor clustering.
2. Measures **how each variant was edited** compared with the earliest message of its family using word-level diff analysis.
3. Tests whether the type or size of an edit is **associated with how widely the message circulated** (diffusion dynamics & virality).
4. Evaluates **fact-checking evasion**, quantifying coverage recovery when linking variant cascades to verified claim databases.

**Dataset:** [COVID19.BR](https://zenodo.org/records/5193932) (WhatsApp messages from Brazilian public groups, Oct 2018 – Jun 2020). The source text is Brazilian Portuguese; sample pairs can be translated to English (`--stage translate`). All statistics and diff features are computed on the original text.

---

## Project Structure

```
run_pipeline.py            Main CLI entrypoint to run any or all stages
build_explorer.py          Generates standalone interactive HTML diff explorer
report_examples.py         Prints sample mutated variant pairs by edit type
requirements.txt           Python package dependencies
Report.pdf                 Complete academic capstone report (PDF)

src/
  config.py                Paths, thresholds, hyperparameters, and regex patterns
  data.py                  Load, clean, collapse exact duplicates, and filter spam/ASCII
  cluster.py               TF-IDF / MiniLM embeddings + mutual-kNN graph clustering
  evaluate.py              Synthetic benchmark generator (planted edits with known ground truth)
  mutations.py             Word-level diff feature extraction per variant pair
  analysis.py              Statistical tests, spread prediction, and COVID comparison
  misinformation.py        Fact-checking propagation and misinformation mutation tests
  plots.py                 Generates high-resolution report figures
  extras.py                Detector validation, hybrid clustering, threshold sensitivity
  translate.py             Optional machine translation of examples to English

tests/                     Unit test suite (pytest)
data/
  raw/                     Source datasets (covid19br.csv, covidbr_labeled.csv)
  processed/               Precomputed intermediate CSVs (bundled with repo)
outputs/                   Generated tables (.csv), figures (.png), and variant_explorer.html
notebooks/                 Exploratory notebooks
```

---

## Setup & Installation (Windows - VS Code Terminal)

Open your project folder in **Visual Studio Code**. Open the integrated terminal using <kbd>Ctrl</kbd> + <kbd>`</kbd> (backtick) or go to menu **Terminal &rarr; New Terminal**. By default, VS Code opens **PowerShell**.

### 1. Create and Activate Virtual Environment

#### In PowerShell (Default VS Code Terminal):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:**  
> If you get an error saying `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system`, allow script execution for the current terminal session by running:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> .\venv\Scripts\Activate.ps1
> ```

#### In Command Prompt (`cmd.exe`):
If your VS Code terminal profile is set to Command Prompt:
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

> **VS Code Tip:** Select your virtual environment interpreter in VS Code by pressing <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd>, typing `Python: Select Interpreter`, and choosing `.\venv\Scripts\python.exe`.

### 2. Install Dependencies

Upgrade `pip` and install the required packages:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional)* If you plan to run unit tests:
```powershell
pip install pytest
```

*(Optional)* If you want to run the English translation stage:
```powershell
pip install transformers sentencepiece
```

---

## Data Availability & Fast Execution

This repository **includes preprocessed intermediate files** in `data/processed/` (`messages_filtered.csv`, `messages_clustered.csv`, `mutation_pairs_final.csv`, etc.) and all generated figures/tables in `outputs/`.

- You can **immediately run downstream stages** (`evaluate`, `analyze`, `extras`, `misinformation`, `plots`), build the interactive explorer, or run tests without downloading the full 125k message raw dataset.
- If you wish to re-run the pipeline from scratch starting from raw data:
  1. Create the `data\raw\` directory:
     ```powershell
     New-Item -ItemType Directory -Path "data\raw" -Force
     ```
  2. Download `covid19br.csv` from [Zenodo Record 5193932](https://zenodo.org/records/5193932) into `data\raw\`.
  3. (Optional) Place `covidbr_labeled.csv` (fact-checked claim labels) into `data\raw\`.

---

## Running the Project (Windows Terminal)

All pipeline commands are executed from the root workspace directory.

### 1. Run Pipeline

- **Run all stages** (downloads `paraphrase-multilingual-MiniLM-L12-v2` embedding model on first run):
  ```powershell
  python run_pipeline.py
  ```

- **Run fast mode** (TF-IDF only, skips embedding model download):
  ```powershell
  python run_pipeline.py --no-embeddings
  ```

- **Run a single stage:**
  ```powershell
  python run_pipeline.py --stage evaluate          # Synthetic mutation benchmark
  python run_pipeline.py --stage analyze           # Statistical tests & virality prediction
  python run_pipeline.py --stage extras            # Detector validation & threshold sensitivity
  python run_pipeline.py --stage misinformation    # Fact-checking propagation analysis
  python run_pipeline.py --stage plots             # Generate output figures
  python run_pipeline.py --stage translate         # Optional: English translations
  ```

Pipeline stages execute in order:
$$\text{prepare} \longrightarrow \text{cluster} \longrightarrow \text{evaluate} \longrightarrow \text{mutations} \longrightarrow \text{analyze} \longrightarrow \text{extras} \longrightarrow \text{misinformation} \longrightarrow \text{plots}$$

Because intermediate stage outputs are saved to `data\processed\`, any stage can be independently re-run.

---

### 2. Interactive Variant Explorer

Build and view the standalone interactive HTML explorer containing side-by-side word-level diffs and variant family trees:

```powershell
# Generate the explorer HTML
python build_explorer.py

# Open variant_explorer.html in your default web browser from PowerShell
Start-Process .\outputs\variant_explorer.html
```

Or right-click `outputs/variant_explorer.html` in VS Code and select **Open with Live Server** / **Reveal in File Explorer**.

---

### 3. Inspect Mutation Examples

Print sample mutated message pairs across different edit categories (headers added, footers added, numbers replaced, words dropped, rewrites):

```powershell
python report_examples.py
```

---

### 4. Run Unit Tests

Execute the automated test suite verifying text sanitization, diff parsing, mutual-kNN clustering, and misinformation propagation logic:

```powershell
pytest tests/
```

Or using python module syntax:
```powershell
python -m pytest tests\ -v
```

---

## Method Summary

- **Cleaning:** Drops media placeholders (`<Arquivo de mídia oculto>`), system messages, and texts under 10 words. Strips URLs and symbols, collapses exact duplicates (retaining sharings, counts, group IDs, and timestamps), and filters WhatsApp invite links and ASCII-art posts.
- **Mutual $k$-NN Clustering:** Connects pairs only if each message is within the other's top-$k$ nearest neighbors ($k=10$) *and* cosine similarity exceeds threshold $\theta$. Connected components form variant families. Word/char $n$-gram TF-IDF ($\theta=0.60$) serves as primary method; multilingual MiniLM embeddings ($\theta=0.90$) serve as semantic comparison.
- **Synthetic Benchmark:** Evaluates clustering against 400 real messages subjected to seven realistic planted edits (numbers altered, footers/headers added, 15%/30% word deletions, sentence truncation, word swaps). Scored via Precision, Recall, Adjusted Rand Index (ARI), and recall per edit type.
- **Mutation Diff Features:** Computes word-level diffs (`difflib.SequenceMatcher`) against the root message (earliest by timestamp) in each family, extracting additions, deletions, substitutions, spelling corrections vs semantic rewrites, and number modifications.
- **Statistical & Virality Analysis:** Grouped cross-validated AUC testing whether edits predict virality; partial Spearman correlations controlling for message age and character length; Fisher's exact tests for COVID vs non-COVID mutation patterns.
- **Misinformation Analysis:** Cross-references variant families with 2,899 fact-checked claims (`covidbr_labeled.csv`), quantifying coverage gains and mutation tactics (urgency CTAs, authority headers).

---

## Results Summary

Key empirical results stored in `outputs/`:
- **Synthetic Evaluation (`synthetic_eval.csv`, `benchmark_recall.png`):** TF-IDF 0.6 achieves **1.0 Precision, 0.965 Recall, 0.982 ARI**, outperforming embeddings (which drop to 0.490 recall on truncation).
- **Mutation Taxonomy (`mutation_shares.csv`, `mutation_types.png`):** 47.9% rewrites, 38.7% spelling/typos, 17.9% headers added, 15.8% footers added, 14.1% numbers altered.
- **Diffusion Null Result (`spread_tests.csv`, `within_family.csv`):** Textual mutations do not drive virality ($\text{AUC} \approx 0.50$). Rewrites negatively correlate with relative within-family spread ($\text{partial } \rho = -0.099, p = 0.041$).
- **Health Crisis Dynamics (`covid_tests.csv`):** COVID-19 messages exhibit $2.18\times$ more added headers and $2.29\times$ more added words.
- **Fact-Checking Recovery (`misinformation_propagation.csv`, `misinformation_tests.csv`):** Family clustering provides a **$+22.1\%$ boost** in fact-checking coverage; misinformation variants are $2.67\times$ more likely to inject urgency CTAs ($p = 0.0066$) and $1.79\times$ more likely to prepend authority headers ($p = 0.0020$).
- **Interactive UI (`variant_explorer.html`):** Browser interface to search, filter, and inspect word diffs across families.

> **Report:** For the complete academic writeup with methodology, mathematical formulations, literature review, and extended discussion, refer to [Report.pdf](file:///c:/Users/majji/OneDrive/Attachments/Desktop/nlp_capstone_whatsapp_mutation/Report.pdf).

---

## Limitations

- Data is in Portuguese; the translation stage uses machine translation for illustrative purposes.
- `sharings` is the dataset authors' spread count; its exact logging definition is inherited from the original COVID19.BR collectors.
- Mutation types are determined by automatic rule heuristics and were not exhaustively hand-labeled.
- The synthetic benchmark plants lexical edits, which naturally favors $n$-gram TF-IDF over dense embeddings; it does not evaluate deep semantic paraphrasing.
- Lineage trees represent approximate heuristics (most similar earlier message), not verified causal forwarding chains.
- COVID-19 categorization relies on regex keyword matching; individual marginal $p$-values are exploratory.

---

## Future Work

1. Replicate evaluation across multilingual and English/Telugu WhatsApp and Telegram datasets (e.g., MultiClaimNet).
2. Incorporate paraphrase-specific synthetic benchmarks to evaluate dense neural encoders.
3. Validate mutation-type extraction heuristics against human-annotated variant samples.

---

## Windows Troubleshooting & Tips

- **PowerShell Execution Policy Error:** If activating `.\venv\Scripts\Activate.ps1` produces `PSSecurityException`, run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
- **UTF-8 Output in PowerShell:** If Portuguese accent characters display incorrectly in PowerShell, enable UTF-8 console output:
  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $env:PYTHONUTF8 = "1"
  ```
- **Path Separators:** Windows supports both backslashes (`\`) and forward slashes (`/`) in Python commands, but in PowerShell, use backslashes (`.\venv\Scripts\Activate.ps1`) when executing local scripts.
- **Python Launcher:** If `python` is not recognized, try `py -m venv venv` and `py run_pipeline.py`. Ensure Python 3.10+ is added to your system `PATH`.
