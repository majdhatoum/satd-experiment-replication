# analysis/build_gold_and_agreement.py
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

EXPERTS_DIR = Path("data/experts")
OUT_DIR     = Path("outputs")

# ---------- header & value normalization ----------

def _norm_header(name: str) -> str:
    s = (name or "").replace("\ufeff", "").strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

HEADER_SYNONYMS: Dict[str, List[str]] = {
    "item_id":      ["item id", "id", "item_id"],
    "source_type":  ["source type", "source", "artifact", "artifact type", "source typ", "source ty", "source_type"],
    "satd_label":   ["satd", "satd?", "satd label", "label"],
    "type_label":   ["satd type", "type", "type label", "satd_type"],
    # optional pass-through
    "project":      ["project"],
    "origin_ref":   ["origin ref", "origin", "ref", "origin_ref"],
    "text":         ["text", "content", "body", "message"],
    "stratum":      ["stratum", "strata"],
}

def _pick_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    norm_map = {_norm_header(c): c for c in df.columns}
    for syn in HEADER_SYNONYMS.get(canonical, []):
        if syn in norm_map:
            return norm_map[syn]
    return None

def _norm_satd_value(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"y","yes","true","1"}:  return "yes"
    if s in {"n","no","false","0"}:  return "no"
    if s in {"not sure","not-sure","unsure","?"}: return "not-sure"
    if s in {"","nan","none","null"}: return None
    return s

TYPE_MAP = {
    "design": "design-debt", "code": "design-debt", "code design": "design-debt",
    "requirement": "requirement-debt", "requirements": "requirement-debt",
    "documentation": "documentation-debt", "docs": "documentation-debt",
    "doc": "documentation-debt", "document": "documentation-debt", "documents": "documentation-debt",
    "test": "test-debt", "testing": "test-debt",
}
def _norm_type_value(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"","nan","none","null"}: return None
    s = s.replace("_"," ").replace("-"," ").replace("/"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return TYPE_MAP.get(s, s)

def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    missing: List[str] = []
    required = ["item_id","source_type","satd_label","type_label"]
    for canon in required + ["project","origin_ref","text","stratum"]:
        col = _pick_column(df, canon)
        if col:
            rename[col] = canon
        elif canon in required:
            missing.append(canon)
    if missing:
        raise ValueError(
            f"CSV is missing required columns after normalization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    out = df.rename(columns=rename)
    keep = [c for c in ["item_id","source_type","satd_label","type_label","project","origin_ref","text","stratum"] if c in out.columns]
    out = out[keep].copy()
    out["satd_label"] = out["satd_label"].map(_norm_satd_value)
    out["type_label"] = out["type_label"].map(_norm_type_value)
    return out

# ---------- majority/gold helpers ----------

def majority_yes_no(rows: List[Optional[str]]) -> Optional[str]:
    votes = [str(r).lower() for r in rows if str(r).lower() in {"yes","no"}]
    if not votes: return None
    c = Counter(votes)
    return "yes" if c["yes"] >= 2 else "no"

def majority_type(rows: List[Optional[str]]) -> Optional[str]:
    votes = [str(r).strip() for r in rows if str(r).strip()]
    if not votes: return None
    label, cnt = Counter(votes).most_common(1)[0]
    return label if cnt >= 2 else None

# ---------- loading & agreement ----------

def load_expert_csvs() -> pd.DataFrame:
    csvs = sorted(EXPERTS_DIR.glob("*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(f"No expert CSVs found in {EXPERTS_DIR.resolve()}")
    frames = []
    for i, path in enumerate(csvs, start=1):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _rename_to_canonical(df)
        df.rename(columns={"satd_label": f"satd_{i}", "type_label": f"type_{i}"}, inplace=True)
        frames.append(df)
    base = frames[0]
    for df in frames[1:]:
        join_keys = [c for c in ["item_id","source_type","project","origin_ref","text","stratum"] if c in base.columns and c in df.columns]
        base = base.merge(df, on=join_keys, how="inner")
    if base.empty:
        raise ValueError("After merging expert files, no overlapping items remain. "
                         "Check that item_id/source_type match across files.")
    return base

def fleiss_table_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    mat = []
    for _, row in df.iterrows():
        votes = []
        for i in (1,2,3):
            v = str(row.get(f"satd_{i}", "")).lower()
            if v in {"yes","no"}: votes.append(v)
        mat.append([votes.count("yes"), votes.count("no")])
    return pd.DataFrame(mat, columns=["yes","no"])

# ---------- bootstrap helpers ----------

def _bootstrap_ci(values_fn, n_items: int, B: int = 10000, seed: int = 2025) -> Tuple[float,float]:
    """
    values_fn(idx_array) -> statistic computed on a bootstrap resample of item indices.
    Returns percentile 95% CI (2.5%, 97.5%).
    """
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(n_items)
    for _ in range(B):
        samp = rng.choice(idx, size=n_items, replace=True)
        vals.append(values_fn(samp))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)

# ---------- main pipeline ----------

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load experts & build gold
    allx = load_expert_csvs()
    allx["gold_satd"] = allx.apply(lambda r: majority_yes_no([r.get("satd_1"), r.get("satd_2"), r.get("satd_3")]), axis=1)
    satd_mask = allx["gold_satd"].eq("yes")
    allx.loc[satd_mask, "gold_type"] = allx.loc[satd_mask].apply(
        lambda r: majority_type([r.get("type_1"), r.get("type_2"), r.get("type_3")]), axis=1
    )

    # Save gold with pass-through
    cols = [c for c in ["item_id","source_type","project","origin_ref","text","stratum",
                        "satd_1","type_1","satd_2","type_2","satd_3","type_3",
                        "gold_satd","gold_type"] if c in allx.columns]
    allx[cols].to_csv(OUT_DIR / "gold_labels.csv", index=False, encoding="utf-8")

    # Agreement (detection)
    tab = fleiss_table_yes_no(allx)               # shape: (n_items, 2)
    kappa = fleiss_kappa(tab.values)
    per_item_agree = (tab.max(axis=1) / tab.sum(axis=1))
    mean_agree = per_item_agree.mean()

    # Vote-split diagnostics (3–0 vs 2–1, by polarity)
    split_rows = []
    yes_30 = yes_21 = no_21 = no_30 = other = 0
    for (i, row_counts), gold in zip(tab.iterrows(), allx["gold_satd"]):
        y, n = int(row_counts["yes"]), int(row_counts["no"])
        if (y + n) != 3:
            pat = "other/missing"
            other += 1
        elif y == 3:
            pat = "3-0 yes"; yes_30 += 1
        elif y == 2:
            pat = "2-1 yes"; yes_21 += 1
        elif y == 1:
            pat = "2-1 no";  no_21  += 1
        else:  # y == 0
            pat = "3-0 no";  no_30  += 1
        split_rows.append((allx.loc[i, "item_id"], allx.loc[i, "source_type"], y, n, pat, gold))
    pd.DataFrame(split_rows, columns=["item_id","source_type","yes_votes","no_votes","pattern","gold_satd"])\
      .to_csv(OUT_DIR / "expert_vote_splits.csv", index=False, encoding="utf-8")

    # Bootstrap CIs for kappa and mean agreement (percentile)
    def stat_kappa(sample_idx: np.ndarray) -> float:
        return float(fleiss_kappa(tab.values[sample_idx, :]))
    def stat_mean_agree(sample_idx: np.ndarray) -> float:
        s = per_item_agree.values[sample_idx]
        return float(np.mean(s))

    k_lo, k_hi   = _bootstrap_ci(stat_kappa,     len(tab))
    m_lo, m_hi   = _bootstrap_ci(stat_mean_agree,len(tab))

    # Write summary
    with open(OUT_DIR / "expert_agreement.txt", "w", encoding="utf-8") as fo:
        fo.write(f"Fleiss_kappa={kappa:.3f}  (bootstrap 95% CI [{k_lo:.3f},{k_hi:.3f}])\n")
        fo.write(f"Mean_item_agreement={mean_agree:.3f}  (bootstrap 95% CI [{m_lo:.3f},{m_hi:.3f}])\n")
        fo.write(f"Items_considered={len(tab)}\n")
        fo.write("Vote_splits:\n")
        fo.write(f"  3-0 yes={yes_30}, 2-1 yes={yes_21}, 2-1 no={no_21}, 3-0 no={no_30}, other/missing={other}\n")
        gold_yes = int((allx['gold_satd']=="yes").sum())
        gold_no  = int((allx['gold_satd']=="no").sum())
        fo.write(f"Gold majority: yes={gold_yes}, no={gold_no}\n")

    print("✓ Wrote:", OUT_DIR / "gold_labels.csv")
    print("✓ Wrote:", OUT_DIR / "expert_agreement.txt")
    print("✓ Wrote:", OUT_DIR / "expert_vote_splits.csv")

if __name__ == "__main__":
    main()
