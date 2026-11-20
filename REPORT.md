# Mutation Analysis of Forwarded WhatsApp Messages: Clustering, Diffusion Dynamics, and Fact-Checking Evasion

**Author:** M Rakshitha  
**Roll Number:** 524cs0011  
**Date:** November 20, 2026  
**Repository Working Directory:** `/home/iiitdmk-csecl/nlp_capstone_src`  

---

## Executive Abstract

In encrypted peer-to-peer messaging networks such as WhatsApp, information diffusion is decentralized, private, and unmonitored. As viral messages circulate between public groups and individual chats, they undergo continual lexical mutation: words are truncated, numbers altered, calls-to-action injected, and institutional headers added. These mutations degrade the efficacy of traditional hash-based fact-checking databases, allowing misinformation to evade debunking.

This capstone project presents an end-to-end natural language processing pipeline designed to reconstruct message mutation cascades, quantify edit patterns, and evaluate their relationship to message spread. Using the **COVID19.BR** corpus (125k+ messages collected across Brazilian public groups), we apply mutual $k$-nearest-neighbor graph clustering to group near-duplicate messages into variant families. On a ground-truth synthetic benchmark of 400 messages subjected to seven realistic edit types, character/word $n$-gram TF-IDF clustering achieves **$1.0$ precision, $96.5\%$ recall, and an Adjusted Rand Index (ARI) of $0.982$**, significantly outperforming multilingual sentence embeddings (`MiniLM`), which drop to $49\%$ recall on truncated texts.

We extract word-level diff features across 984 variant pairs spanning 604 natural families. Our analysis uncovers a crucial **null result**: edit types and magnitude do **not** predict message virality ($\text{AUC} \approx 0.50$ across grouped cross-validation). Controlling for arrival time and message length, variants with heavy rewrites circulate *less* than earlier messages ($\text{partial } \rho = -0.099$, $p < 0.05$), indicating that diffusion is governed by first-mover timing and network structure rather than textual optimization. Crucially, linking variant families to 2,899 fact-checked claims reveals that misinformation variants are **$2.67\times$ more likely to inject urgency calls-to-action** ($p = 0.0066$) and **$1.79\times$ more likely to inject authority headers** ($p = 0.0020$). Furthermore, our clustering pipeline provides an automated **$+22.1\%$ boost in fact-checking coverage**, recovering 58 unflagged misinformation variants that had evaded direct fact-check detection.

---

## 1. Introduction & Motivation

Private messaging applications—most notably WhatsApp—have become the primary conduit of news, political discourse, and health advisories in the Global South. WhatsApp's end-to-end encryption ensures privacy but creates an opaque information ecosystem where misinformation can spread without centralized oversight. 

When users forward messages, they frequently copy-paste and edit the text. These modifications include:
- Prepending fake authoritative endorsements (e.g., *"According to Johns Hopkins..."*, *"Globo News reports..."*).
- Altering critical numerical figures (such as government stimulus payout dates, monetary values, or COVID-19 death tolls).
- Stripping contextual disclaimers or adding urgent calls-to-action (*"Share before this gets deleted!"*).

Because conventional automated fact-checking platforms rely on exact cryptographic hashing (e.g., MD5/SHA256) or strict keyword matching, even trivial edits cause a variant to evade detection. To counter this, this project investigates four core research questions:
1. **RQ1 (Mutation Taxonomy):** What types and frequencies of edits naturally occur in forwarded message cascades?
2. **RQ2 (Diffusion & Virality):** Are specific edits (e.g., adding urgency or modifying numbers) associated with higher circulation?
3. **RQ3 (Health-Crisis Dynamics):** Did messages related to the COVID-19 pandemic mutate differently than general discourse?
4. **RQ4 (Fact-Checking Evasion):** Can variant family clustering identify unflagged misinformation variants that slip through direct fact-check repositories?

---

## 2. Dataset & Preprocessing Pipeline

The primary dataset is **COVID19.BR** ([Zenodo Record 5193932](https://zenodo.org/records/5193932)), an open archive of WhatsApp messages collected from public Brazilian groups between October 2018 and June 2020. An auxiliary verified dataset, `covidbr_labeled.csv`, contains 2,899 fact-checked messages categorized into misinformation ($n=912$) and non-misinformation ($n=1,987$) with corresponding fact-check debunking URLs.

```
Raw Data (125k+ msgs)
       │
       ▼ [Drop media placeholders, <Arquivo de mídia oculto>, and short texts < 10 words]
Chat Noise Filtered (82,416 msgs)
       │
       ▼ [Clean URLs, emojis, and symbols; collapse exact duplicates]
Aggregated Texts (18,129 unique texts)
       │
       ▼ [Remove group invite spam and ASCII-art posts]
Cleaned Corpus (17,672 texts) ──> 604 Variant Families (984 Variant Pairs)
```

### Table 1: Dataset Preprocessing Funnel
| Pipeline Stage | Records Kept | Notes |
| :--- | :--- | :--- |
| **Raw Message Database** | 125,000+ | Raw exports from WhatsApp public groups |
| **Filtered Text Records** | 82,416 | Dropped media rows, system messages, texts $< 10$ words |
| **Collapsed Aggregates** | 18,129 | Collapsed exact duplicates, recorded occurrences & groups |
| **Spam / ASCII Filtered** | 17,672 | Removed WhatsApp invite links (`chat.whatsapp.com`) & ASCII art |
| **Variant Families ($size \ge 2$)** | 604 families | Messages linked via mutual nearest-neighbor clustering |
| **Analyzed Variant Pairs** | 984 pairs | Root-to-variant comparison pairs with temporal tracking |

---

## 3. Methodology

### 3.1 Mutual Nearest-Neighbor Graph Clustering

A major pathology in clustering short, noisy text is **chaining**: if message $A$ is similar to $B$, and $B$ to $C$, single-linkage agglomeration will merge $A$ and $C$ even if they share zero semantic or lexical connection. To prevent this, we implement **Mutual $k$-Nearest Neighbors ($k\text{-NN}$)**:

$$\text{Edge}(u, v) = 1 \iff u \in \mathcal{N}_k(v) \land v \in \mathcal{N}_k(u) \land \text{CosineSim}(\mathbf{x}_u, \mathbf{x}_v) \ge \tau$$

Connected components formed by these symmetric edges define the **variant families**. This ensures that messages are grouped only when their nearest-neighbor relationship is bidirectional.

### 3.2 Feature Representation: TF-IDF vs. Sentence Transformers

We evaluate two distinct representations:
1. **Sublinear TF-IDF:** Word unigrams and bigrams, minimum document frequency $\text{min\_df}=2$, sublinear term frequency scaling ($1 + \log(\text{tf})$).
2. **Multilingual Sentence Transformers:** `paraphrase-multilingual-MiniLM-L12-v2`, producing 384-dimensional dense vectors normalized to unit length.

### 3.3 Mutation Extraction & Word-Level Diff Operators

For every family with $size \ge 2$, the **earliest message** (ordered chronologically by first-seen timestamp and group appearance) is designated as the family root ($\text{root}$). All subsequent messages in the family ($\text{var}$) are compared against the root using Python’s `difflib.SequenceMatcher`:
- **Words Added / Removed / Replaced:** Calculated from opcode operations (`insert`, `delete`, `replace`).
- **Prefix Added (Header):** Opcode insert at position 0.
- **Suffix Added (Footer):** Opcode insert at index $len(\text{root})$.
- **Typo Fixes vs. Real Rewrites:** Replaced phrases with character sequence similarity $\ge 0.80$ are categorized as typo/spelling fixes; lower similarity replacements are marked as substantive rewrites.
- **Strict Number Replacement:** Detected when a sequence replacement operation contains digits in both the deleted and inserted spans.
- **Call-to-Action (CTA) Detection:** Regex matching urgency/sharing language (*urgente, compartilhe, divulgue, repasse, bomba, vazou, atenção, veja*).

---

## 4. Synthetic Benchmark & Validation

Because real-world forwarded messages lack ground-truth family labels, we built a synthetic benchmark. We sampled 400 real messages and planted 7 realistic mutation types ($n=2,499$ total planted variants):
1. **Header Added:** Prepending fake authority or broadcast tags.
2. **Footer Added:** Appending links or disclaimer text.
3. **Number Swapped:** Replacing monetary figures, percentages, or dates.
4. **15% Word Drop:** Random deletion of 15% of tokens.
5. **30% Word Drop:** Random deletion of 30% of tokens.
6. **Truncation (60%):** Retaining only the first 60% of the message.
7. **Word Swaps:** Local token transpositions simulating paraphrasing.

### Table 2: Benchmark Performance Across Clustering Models
| Method | Threshold ($\tau$) | Precision | Recall | ARI | Header | Footer | Drop 15% | Drop 30% | Trunc 60% | Swaps | Numbers |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF** | 0.50 | 1.000 | 0.992 | 0.996 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 |
| **TF-IDF (Selected)** | **0.60** | **1.000** | **0.965** | **0.982** | **1.00** | **1.00** | **0.99** | **0.88** | **1.00** | **0.99** | **1.00** |
| **TF-IDF** | 0.70 | 1.000 | 0.863 | 0.926 | 1.00 | 1.00 | 0.97 | 0.62 | 0.97 | 0.91 | 1.00 |
| **MiniLM** | 0.85 | 0.968 | 0.837 | 0.898 | 0.99 | 0.98 | 0.94 | 0.74 | 0.72 | 1.00 | 0.98 |
| **MiniLM** | 0.90 | 1.000 | 0.684 | 0.812 | 0.96 | 0.88 | 0.85 | 0.53 | 0.49 | 1.00 | 0.96 |
| **Hybrid (TF-IDF \| MiniLM)**| 0.60 \| 0.90 | **1.000** | **0.972** | **0.986** | 1.00 | 1.00 | 0.99 | 0.90 | 1.00 | 1.00 | 1.00 |

![Figure 1: Benchmark Recall Across Mutation Types](nlp_capstone/outputs/benchmark_recall.png)
*Figure 1: Recall across planted mutation types for TF-IDF (0.50, 0.60, 0.70) and multilingual MiniLM (0.85, 0.90, 0.95). Notice the steep degradation of MiniLM on 30% word drops and 60% message truncations.*

> [!IMPORTANT]
> **Key Finding on Dense Embeddings:** Dense sentence embeddings (`MiniLM`) perform poorly when messages undergo deletions or truncations (dropping to $49\%$ recall on truncated texts), whereas TF-IDF remains resilient ($100\%$ recall). However, combining both in a hybrid architecture achieves optimal performance ($0.986$ ARI).

### Table 3: Mutation Detector Validation on Ground Truth
| Detector Rule | Target True Condition | $N_{\text{true}}$ | $N_{\text{other}}$ | Recall | Precision | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Header Added** | `header_added` | 400 | 2,099 | **1.000** | 0.897 | 0.022 |
| **Footer Added** | `footer_added` | 400 | 2,099 | **1.000** | 0.988 | 0.002 |
| **Numbers Changed (Base)** | `numbers_changed` | 127 | 2,372 | **1.000** | 0.498 | 0.054 |
| **Numbers Replaced (Strict)**| `numbers_changed` | 127 | 2,372 | **1.000** | **1.000** | **0.000** |
| **Words Dropped ($>3$)** | `drop_15pct` + `drop_30pct` | 1,187 | 1,312 | **0.848** | 0.856 | 0.129 |

---

## 5. Empirical Results & Diffusion Dynamics

### 5.1 Mutation Taxonomy in WhatsApp Cascades

Analysis of the 984 natural variant pairs reveals the prevalence of edit operations across WhatsApp networks:

```
Substantive Rewrites (>3 words)  [###############################] 47.9%
Spelling & Typo Corrections      [#########################] 38.7%
Elaborations / Additions (>3 w)  [###################] 29.1%
Words Dropped (>3 words)         [#############] 20.2%
Header Injected (Start)          [###########] 17.9%
Footer Appended (End)            [##########] 15.8%
Numbers Replaced                 [#########] 14.1%
Urgency / CTA Injected           [##] 3.6%
Hyperlink Injected               [#] 2.7%
```

![Figure 2: Distribution of Mutation Types](nlp_capstone/outputs/mutation_types.png)
*Figure 2: Prevalence of mutation types across 984 variant pairs. Lexical rewrites, typo fixes, and elaborations represent the largest share of modifications.*

### 5.2 The Virality Question: Do Mutations Drive Spread? (Null Result)

A central hypothesis in social media research is that viral content undergoes evolutionary optimization—that users tweak messages to make them more sensational, urgent, or believable, thereby increasing their circulation. We tested this systematically:

1. **Classification Modeling:** Using group 5-fold cross-validation (grouping by family ID to prevent data leakage), we trained Logistic Regression and Gradient Boosted Trees on mutation features to predict whether a variant became viral ($\ge 2$ sharings).
   - **Result:** The models achieved **$\text{AUC} = 0.50 \pm 0.03$**, equivalent to random guessing.
2. **Within-Family Partial Spearman Correlation:** Controlling for time since the root message and text length, we measured the correlation between edit features and relative within-family circulation:
   - Real Rewrites: $\text{partial } \rho = \mathbf{-0.099}$ ($95\% \text{ CI: } [-0.152, -0.046], p = 0.003$).
   - Rewrite Rate: $\text{partial } \rho = \mathbf{-0.078}$ ($95\% \text{ CI: } [-0.120, -0.034], p = 0.001$).

> [!NOTE]
> **Takeaway:** Rather than increasing reach, heavily rewritten variants circulate *less* than earlier messages. Spread is governed by first-mover advantage, timing, and network injection points, rather than fine-grained lexical optimization.

![Figure 3: Message Lineage and Mutation Propagation](nlp_capstone/outputs/lineage_examples.png)
*Figure 3: Reconstructed mutation lineage trees across sample WhatsApp message cascades, illustrating temporal propagation and topological branching.*

### 5.3 Health-Crisis Discourse: COVID-19 vs. General Discourse

Comparing COVID-19 messages to non-COVID messages shows significant differences in how health-related information is altered:

### Table 4: COVID-19 vs. Non-COVID Mutation Differences
| Feature / Edit Type | Test Type | COVID Rate | Non-COVID Rate | Effect / Odds Ratio | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Header Added** | Fisher's Exact | **0.264** | **0.142** | **OR = 2.18** | **$< 0.0001$** |
| **Words Added ($>3$)** | Fisher's Exact | **0.415** | **0.236** | **OR = 2.29** | **$< 0.0001$** |
| **Real Rewrites ($>3$)** | Fisher's Exact | **0.555** | **0.445** | **OR = 1.56** | **$0.0018$** |
| **Words Dropped ($>3$)** | Fisher's Exact | 0.234 | 0.188 | OR = 1.32 | 0.1019 |
| **Numbers Changed** | Fisher's Exact | 0.127 | 0.147 | OR = 0.84 | 0.4272 |
| **Footer Added** | Fisher's Exact | 0.171 | 0.152 | OR = 1.15 | 0.4486 |

Health crisis messages experienced more than **double the rate of added headers** and **added text**, reflecting users' attempts to attach authority, instructions, and context to alarming health claims.

---

## 6. Misinformation & Fact-Checking Evasion

Linking our clustered corpus to `covidbr_labeled.csv` (2,899 fact-checked items) reveals critical insights into how misinformation operates and mutates in encrypted networks.

### 6.1 The Fact-Checking Coverage Gap

### Table 5: Fact-Check Propagation through Variant Clustering
| Metric | Value | Significance |
| :--- | :---: | :--- |
| **Misinformation Families Identified** | 98 | Variant families containing confirmed misinformation claims |
| **Total Messages in Misinformation Families** | 321 | Messages circulating across WhatsApp public groups |
| **Directly Fact-Checked Messages** | 263 | Messages that exact-matched fact-check database entries |
| **Unflagged Mutated Variants** | **58** | Mutated variants that evaded direct fact-check detection |
| **Fact-Check Coverage Boost** | **+22.1%** | Extra misinformation volume identified by clustering |
| **Circulation of Unflagged Variants** | **117** sharings | 120 total group occurrences of unflagged hoaxes |

By clustering near-duplicates, fact-checking organizations can automatically propagate a single fact-check label across an entire variant family, closing an otherwise invisible **$22.1\%$ coverage gap**.

### 6.2 Mutation Signatures of Misinformation

Statistical tests comparing misinformation variant pairs ($n=223$) against non-misinformation variant pairs ($n=761$) reveal specific behavioral differences:

### Table 6: Mutation Features in Misinformation vs. Non-Misinformation
| Measure | Test | Misinfo Share | Non-Misinfo Share | Odds Ratio | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Urgency / CTA Added** | Fisher's Exact | **0.067** | **0.026** | **2.67** | **$0.0066$** |
| **Header Added** | Fisher's Exact | **0.251** | **0.158** | **1.79** | **$0.0020$** |
| **Words Added ($>3$)** | Fisher's Exact | **0.408** | **0.256** | **2.00** | **$< 0.0001$** |
| **Real Rewrites ($>3$)** | Fisher's Exact | **0.565** | **0.453** | **1.57** | **$0.0037$** |
| **Footer Added** | Fisher's Exact | 0.188 | 0.148 | 1.33 | 0.1738 |
| **Numbers Changed** | Fisher's Exact | 0.121 | 0.147 | 0.80 | 0.3818 |
| **Overall Edit Size** | Mann-Whitney | 0.074 | 0.175 | $-0.100$ | $< 0.0001$ |

![Figure 4: Mutation Frequency in Misinformation vs Non-Misinformation Families](nlp_capstone/outputs/misinformation_mutations.png)
*Figure 4: Comparison of edit rates showing the pronounced elevation of urgency calls-to-action (OR = 2.67, p = 0.0066) and authoritative headers (OR = 1.79, p = 0.0020) in verified misinformation cascades.*

> [!TIP]
> **The Misinformation Mutation Signature:** Misinformation messages do not alter arbitrary words; instead, they selectively inject **urgency calls-to-action** ($\text{OR} = 2.67$) and **authoritative headers** ($\text{OR} = 1.79$) to induce panic and boost credibility.

---

## 7. Qualitative Case Studies

From our translated dataset (`report_examples.py`), we highlight three representative cases demonstrating how real messages mutated in Brazilian WhatsApp groups:

### Case 1: Financial Aid Amount Alteration (Scam Escalation)
* **Family ID:** 7198
* **Original Message:**
  > *"Government has just released the schedule of the emergency withdrawal in the value of **R\$1,760.00**..."*
* **Mutated Variant:**
  > *"Government has just released the schedule of the emergency withdrawal in the value of **R\$1,200.00** and **R\$600.00**..."*
* **Analysis:** Scammers modified the monetary payout figures to align with official government emergency aid tiers (Auxílio Emergencial) to deceive citizens into entering sensitive credentials on phishing links.

### Case 2: Institutional Authority Spoofing (Header Prepending)
* **Family ID:** 7138
* **Original Message:**
  > *"Cosme: do you know about the new ministerial agenda?"*
* **Mutated Variant:**
  > *"**Globo News: Cosme from GloboNews on the agenda:** Do you know about..."*
* **Analysis:** Prepending the name of Brazil's largest news broadcaster (*GloboNews*) artificially elevates the perceived veracity of an unverified political claim.

### Case 3: Media Forwarding Bait
* **Family ID:** 3355
* **Original Message:**
  > *"Regarding the quarantine... we are seeing these consequences."*
* **Mutated Variant:**
  > *"**Listen to this audio until the end!** Regarding the quarantine, we are seeing..."*
* **Analysis:** Users frequently attach a text prefix to prompt group members to play attached external voice notes or audio recordings.

---

## 8. Interactive Artifacts & System Architecture

As part of this capstone, the following deliverables have been engineered and validated:

1. **Core Pipeline CLI (`run_pipeline.py`):**
   - Single command execution: `python run_pipeline.py --stage all`
   - Modular stages: `prepare -> cluster -> evaluate -> mutations -> analyze -> extras -> misinformation -> plots -> translate`
2. **Interactive Lineage & Variant Explorer (`nlp_capstone/outputs/variant_explorer.html`):**
   - Standalone, zero-dependency HTML dashboard allowing full-text search, mutation filtering, and side-by-side visual diff inspections.

---

## 9. Limitations & Ethical Considerations

- **Language & Cultural Context:** All messages originate from Brazilian Portuguese WhatsApp groups. While mutation mechanisms (headers, links, numbers) are universal, linguistic nuance and slang are specific to Brazil.
- **Approximation of Diffusion Lineage:** WhatsApp's encryption prevents tracking true transmission graphs. Our lineages represent the most similar earliest-observed message rather than cryptographically proven re-sharing.
- **Synthetic Benchmark Assumptions:** The synthetic benchmark simulates lexical alterations. Paraphrastic rewriting (e.g., neural paraphrasing) is less represented and would require dense embedding evaluation.

---

## 10. Conclusion

This project demonstrates that lexical mutation is widespread in encrypted WhatsApp forwarding networks, yet it operates under distinct rules:
1. **TF-IDF with Mutual $k$-NN clustering** provides an optimal defense against message drift, outperforming dense sentence embeddings when dealing with text deletions.
2. **Mutations do not predict virality:** Message spread is governed by network position and first-mover timing rather than iterative text optimization.
3. **Misinformation exhibits clear structural drift:** Disinformation variants selectively inject urgency calls ($2.67\times$) and authoritative headers ($1.79\times$).
4. **Fact-checking propagation:** Variant family clustering successfully closes a **$22.1\%$ fact-checking coverage gap**, offering an immediate, deployable capability for social media trust-and-safety systems.
