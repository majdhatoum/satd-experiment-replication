# Reproducing the SATD Experiment

This document explains how to rebuild the gold labels, compute agreement, evaluate
MT-TEXT-CNN predictions against expert gold, and regenerate all figures.

---

## 0) What’s in this repo

```bash

analysis/ # Python analysis scripts
convert_experts.py # XLSX/CSV → canonical CSV for each expert & model
build_gold_and_agreement.py # builds majority-vote gold + Fleiss' kappa
evaluate_and_plot.py # metrics + Figures (forest plot & confusion matrix)

config/
paths.yaml # (optional) central paths; not required by scripts

data/
ids/ # repo list and canonical identifiers (no raw text)
experts/ # expert annotation spreadsheets (CSV or XLSX)
text/ # (optional) locally reconstructed raw text (not committed)

model/
inference/ # model predictions (CSV or XLSX)
snapshots/ # (optional) model weights/snapshots
embeddings/ # (optional) word vectors

outputs/ # generated metrics, tables, and figures (created by scripts)

docs/
annotation_guideline.md # 2-page rater guideline
consent_expert.md # participant information & consent

REPRODUCING.md
requirements.txt
requirements.lock.txt # exact versions used in the paper run
Dockerfile
.dockerignore
```

> **Privacy/Licensing.** We publish **identifiers only** (commit SHAs, issue/PR numbers,
> file paths/line ranges). If upstream licenses restrict redistribution of raw text,
> reconstruct it locally with your own GitHub token using the included scripts.

---

## 1) Environment setup

### Option A — Python (Windows PowerShell or macOS/Linux shell)

```bash
python -V                         # 3.10–3.12 recommended
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# (Optional) reproduce exact paper environment:
# pip install -r requirements.lock.txt
```

### Option B — Docker (no local Python needed)
> The below bash loads and runs the **complete** experiment as we call all three scripts

```bash
docker build -t satd-repl .
# Windows PowerShell:
docker run --rm -v ${PWD}:/work -w /work satd-repl bash -lc \
  "python analysis/convert_experts.py && \
   python analysis/build_gold_and_agreement.py && \
   python analysis/evaluate_and_plot.py"
# macOS/Linux:
docker run --rm -v "$PWD:/work" -w /work satd-repl bash -lc \
  "python analysis/convert_experts.py && \
   python analysis/build_gold_and_agreement.py && \
   python analysis/evaluate_and_plot.py"
```
> In case you face issues such as **"Unable to find image `satd-repl:latest` locally"** 
> Just build the image first using `docker build -t satd-repl .`

## 2) Required inputs

### 2.1 Expert annotations → data/experts/

Provide three spreadsheets (CSV or XLSX), one per expert. Header names are
normalized automatically, but a clean template is:

```bash
item_id, source_type, project, origin_ref, text, stratum, SATD, SATD Type, Notes
```

- `source_type` ∈ { `commit`, `code_comment`, `issue`, `pull_request` }
- `SATD` ∈ { `yes`, `no`, `not-sure` }
- `SATD Type` ∈ { `design-debt`, `requirement-debt`, `documentation-debt`, `test-debt` }  
  (common variants like “Design”, “document”, “requirements” are normalized)

### 2.2 Model predictions → `model/inference/`

One CSV/XLSX containing at least:

```bash
item_id, source_type, satd (or label/prediction), (optional) satd type
```

Header synonyms such as `label`, `prediction`, `satd type`, `prediction type`
are handled. Type evaluation only uses rows where the model predicts
SATD = `yes` and provides a recognizable type; any dropped rows (and reasons)
are logged to `outputs/dropped_for_type_confusion.csv`


## 3) Run the pipeline

```bash

# 1) Normalize expert/model spreadsheets → canonical CSVs
python analysis/convert_experts.py

# 2) Build majority-vote gold + compute inter-rater agreement (Fleiss' κ)
python analysis/build_gold_and_agreement.py
# → outputs/gold_labels.csv
# → outputs/expert_agreement.txt

# 3) Evaluate model vs gold + produce figures
python analysis/evaluate_and_plot.py
# → outputs/metrics_detection.txt
# → outputs/per_source_ba.csv
# → outputs/fig_forest_ba_by_source.png
# → outputs/fig_type_confusion.png (if types available)
# → outputs/dropped_for_type_confusion.csv (diagnostics)
```
All scripts are deterministic (fixed seeds). Bootstrap CIs use 10,000 resamples.

## 4) Expected headline numbers (this release)

From our 32-item balanced evaluation set, the pipeline should reproduce the following
(high-level) results:

- **Expert agreement (SATD yes/no):** Fleiss’ κ ≈ **0.749**; mean item agreement ≈ **0.938**  
  (see `outputs/expert_agreement.txt`).

- **Detection (MT-TEXT-CNN vs. expert gold):**  
  Precision ≈ **0.833**, Recall ≈ **0.588**, **F1 ≈ 0.690**  
  Balanced Accuracy ≈ **0.727**  
  95% bootstrap CIs are written to `outputs/metrics_detection.txt`.

- **Balanced accuracy by source (approximate point estimates; see CSV for exact):**
  - `commit` ≈ **0.80**
  - `code_comment` ≈ **0.75**
  - `issue` ≈ **0.70**
  - `pull_request` ≈ **0.73**

Small numeric differences can arise if library versions differ from those in
`requirements.lock.txt`.

---

## 5) Re-running with your own data

1. **Replace expert files** in `data/experts/` (one CSV/XLSX per expert).  
   Be sure `item_id` and `source_type` match across all experts.

2. **Place your model predictions** in `model/inference/` as a CSV/XLSX with at least  
   `item_id`, `source_type`, and the model’s SATD decision (`yes`/`no`).  
   If you also predict types, add a column such as `satd type`/`prediction type`.

3. **Run the three scripts**:

   ```bash
   python analysis/convert_experts.py
   python analysis/build_gold_and_agreement.py
   python analysis/evaluate_and_plot.py 
   
   ```
4. **Inspect outputs**

After the three scripts finish, open the `outputs/` folder. You should see:

- `gold_labels.csv` — item-level majority gold (detection + type when a majority exists).
- `expert_agreement.txt` — Fleiss’ κ and mean item agreement for SATD yes/no.
- `metrics_detection.txt` — precision, recall, F1 and balanced accuracy, each with 95% bootstrap CIs.
- `per_source_ba.csv` — balanced accuracy by source (commit, code_comment, issue, pull_request) with CIs.
- `fig_forest_ba_by_source.png` — “Figure 2” (forest plot of balanced accuracy with 95% CIs).
- `fig_type_confusion.png` — “Figure 1” (type confusion heatmap; produced only if type data are available).
- `dropped_for_type_confusion.csv` — diagnostics listing any rows excluded from the type confusion plot
  and a `reason` column (e.g., `gold_type_missing`, `model_pred_type_unrecognized`,
  `model_pred_satd_is_no_or_missing`).

> Tip: If `fig_type_confusion.png` is missing or very sparse, check
> `dropped_for_type_confusion.csv` to understand why certain items were not included.

---

## 6) Outputs and figures (reference)

- `outputs/gold_labels.csv` — majority vote gold labels.
- `outputs/expert_agreement.txt` — inter-rater agreement summary.
- `outputs/metrics_detection.txt` — overall detection scores + CIs.
- `outputs/per_source_ba.csv` — per-source balanced accuracy + CIs.
- `outputs/fig_forest_ba_by_source.png` — forest plot (balanced accuracy by source).
- `outputs/fig_type_confusion.png` — SATD type confusion matrix (when applicable).
- `outputs/dropped_for_type_confusion.csv` — reasons for exclusions in type analysis.

---

## 7) Troubleshooting

- **“After merging gold and predictions, no rows remain.”**  
  `item_id` / `source_type` do not match between gold and predictions. Ensure the same values
  (case-insensitive for `source_type`, no extra spaces). The scripts lower-case and trim `source_type`
  but do *not* alter `item_id`.

- **Garbled characters in text.**  
  Save CSVs as UTF-8. The scripts read with `encoding="utf-8-sig"` to tolerate a BOM.

- **No type confusion figure.**  
  The model either predicted `no` for those items or emitted an unrecognized type label.
  See `dropped_for_type_confusion.csv`.

- **Numbers differ slightly from the paper.**  
  Install exact versions from `requirements.lock.txt` (or run in Docker). Bootstrap CIs are
  deterministic here due to fixed seeds, but library differences can nudge results.

- **Docker path issues on Windows.**  
  Use the PowerShell command in Section *1B* (note `${PWD}` and quoting).

---

## 8) Citation

If you use this package or its results, please cite:

- Li, Y., Soliman, M., & Avgeriou, P. (2023).
  *Automatic identification of self-admitted technical debt from four different sources.*  
  **Empirical Software Engineering, 28(3), 65.**

Please also cite the associated thesis when available.

---

## 9) Releasing an archival snapshot

1. Create a GitHub release (e.g., `v1.0.0`) once scripts and outputs are final.  
2. Connect the repository to **Zenodo** to mint a DOI; add the badge to `README.md` and this file.  
3. Ensure the release includes:
   - `analysis/`, `docs/`, `requirements*.txt`, `Dockerfile`, `.dockerignore`
   - `outputs/` with de-identified labels and generated figures
   - `data/ids/` plus deterministic reconstruction scripts
   - **No** license-restricted raw text
4. Verify the release rebuilds end-to-end by following Sections 1–5 in a clean environment.

---

*Last updated:* 2025-11-28



