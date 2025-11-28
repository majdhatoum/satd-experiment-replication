# analysis/build_gold_and_agreement.py
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

EXPERTS_DIR = Path("data/experts")
OUT_DIR = Path("outputs")

# ---------- header & value normalization ----------

def _norm_header(name: str) -> str:
    """
    Normalize a column header for robust matching:
    - lowercase
    - remove BOM/whitespace
    - replace separators (underscore, hyphen, slash) with single space
    - collapse multiple spaces
    """
    s = (name or "").replace("\ufeff", "").strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# Map from canonical name -> list of normalized header synonyms we accept.
HEADER_SYNONYMS: Dict[str, List[str]] = {
    "item_id": [
        "item id", "id", "item_id",
    ],
    "source_type": [
        "source type", "source", "artifact", "artifact type",
        "source typ", "source ty", "source_type",
    ],
    "satd_label": [
        "satd", "satd?", "satd label", "label",
    ],
    "type_label": [
        "satd type", "type", "type label", "satd_type",
    ],
    # Optional extras we’ll pass through if present:
    "project": ["project"],
    "origin_ref": ["origin ref", "origin", "ref", "origin_ref"],
    "text": ["text", "content", "body", "message"],
    "stratum": ["stratum", "strata"],
}

# Normalize SATD yes/no values
def _norm_satd_value(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return "yes"
    if s in {"n", "no", "false", "0"}:
        return "no"
    if s in {"not sure", "not-sure", "unsure", "?"}:
        return "not-sure"
    # empty or unknown → None
    if s in {"", "nan", "none", "null"}:
        return None
    # leave any other token as-is (rare)
    return s

# Normalize type values to canonical tokens
TYPE_MAP = {
    "design": "design-debt",
    "code": "design-debt",
    "code design": "design-debt",
    "Design": "design-debt",
    "requirement": "requirement-debt",
    "requirements": "requirement-debt",
    "Requirement": "requirement-debt",
    "documentation": "documentation-debt",
    "Document": "documentation-debt",
    "docs": "documentation-debt",
    "doc": "documentation-debt",
    "test": "test-debt",
    "Test": "test-debt",
    "testing": "test-debt",
}

def _norm_type_value(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    # strip common punctuation and join with spaces
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return TYPE_MAP.get(s, s)  # if unseen, keep as-is

def _pick_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the actual column name in df that maps to `canonical`, or None."""
    norm_map = {_norm_header(c): c for c in df.columns}
    for syn in HEADER_SYNONYMS.get(canonical, []):
        if syn in norm_map:
            return norm_map[syn]
    return None

def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names if we can find them via synonyms.
    Keep passthrough cols when available (project, origin_ref, text, stratum).
    """
    rename: Dict[str, str] = {}
    missing: List[str] = []
    required = ["item_id", "source_type", "satd_label", "type_label"]

    for canon in required + ["project", "origin_ref", "text", "stratum"]:
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
    # Keep only the columns we care about (plus optional pass-through if present)
    keep = [c for c in ["item_id","source_type","satd_label","type_label","project","origin_ref","text","stratum"] if c in out.columns]
    out = out[keep].copy()

    # Normalize values
    out["satd_label"] = out["satd_label"].map(_norm_satd_value)
    out["type_label"] = out["type_label"].map(_norm_type_value)

    return out

# ---------- majority/gold helpers ----------

def majority_yes_no(rows: List[Optional[str]]) -> Optional[str]:
    """Return 'yes' if >=2 yes votes among yes/no; treat other values as missing."""
    votes = [str(r).lower() for r in rows if str(r).lower() in {"yes", "no"}]
    if not votes:
        return None
    c = Counter(votes)
    return "yes" if c["yes"] >= 2 else "no"

def majority_type(rows: List[Optional[str]]) -> Optional[str]:
    """Return type label if at least 2 raters agree; otherwise None."""
    votes = [str(r).strip() for r in rows if str(r).strip()]
    if not votes:
        return None
    label, cnt = Counter(votes).most_common(1)[0]
    return label if cnt >= 2 else None

# ---------- loading & agreement ----------

def load_expert_csvs() -> pd.DataFrame:
    """
    Load all expert CSVs, normalize headers/values, and merge on (item_id, source_type).
    Produces columns: item_id, source_type, [optional passthrough], satd_1..3, type_1..3
    """
    csvs = sorted(EXPERTS_DIR.glob("*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(f"No expert CSVs found in {EXPERTS_DIR.resolve()}")

    frames = []
    for i, path in enumerate(csvs, start=1):
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = _rename_to_canonical(df)
        df.rename(columns={
            "satd_label": f"satd_{i}",
            "type_label": f"type_{i}",
        }, inplace=True)
        frames.append(df)

    # Merge on stable keys
    base = frames[0]
    for df in frames[1:]:
        base = base.merge(df, on=[c for c in ["item_id","source_type","project","origin_ref","text","stratum"] if c in base.columns and c in df.columns], how="inner")

    # Basic sanity checks
    if base.empty:
        raise ValueError("After merging expert files, no overlapping items remain. "
                         "Check that item_id/source_type match across files.")

    return base

def fleiss_table_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Build [yes,no] counts per item, ignoring 'not-sure'."""
    mat = []
    for _, row in df.iterrows():
        votes = []
        for i in (1, 2, 3):
            v = str(row.get(f"satd_{i}", "")).lower()
            if v in {"yes", "no"}:
                votes.append(v)
        mat.append([votes.count("yes"), votes.count("no")])
    return pd.DataFrame(mat, columns=["yes", "no"])

# ---------- main pipeline ----------

def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load and normalize experts
    allx = load_expert_csvs()

    # Build majority gold labels
    allx["gold_satd"] = allx.apply(
        lambda r: majority_yes_no([r.get("satd_1"), r.get("satd_2"), r.get("satd_3")]),
        axis=1
    )
    satd_mask = allx["gold_satd"].eq("yes")
    allx.loc[satd_mask, "gold_type"] = allx.loc[satd_mask].apply(
        lambda r: majority_type([r.get("type_1"), r.get("type_2"), r.get("type_3")]),
        axis=1
    )

    # Save gold labels (keep pass-through cols if we have them)
    cols = [c for c in ["item_id","source_type","project","origin_ref","text","stratum",
                        "satd_1","type_1","satd_2","type_2","satd_3","type_3",
                        "gold_satd","gold_type"] if c in allx.columns]
    allx[cols].to_csv(OUT_DIR / "gold_labels.csv", index=False, encoding="utf-8")

    # Compute expert agreement for detection (yes/no only)
    tab = fleiss_table_yes_no(allx)
    kappa = fleiss_kappa(tab.values)
    mean_agree = (tab.max(axis=1) / tab.sum(axis=1)).mean()

    with open(OUT_DIR / "expert_agreement.txt", "w", encoding="utf-8") as fo:
        fo.write(f"Fleiss_kappa={kappa:.3f}\n")
        fo.write(f"Mean_item_agreement={mean_agree:.3f}\n")
        fo.write(f"Items_considered={len(tab)}\n")

    print("✓ Wrote:", OUT_DIR / "gold_labels.csv")
    print("✓ Wrote:", OUT_DIR / "expert_agreement.txt")

if __name__ == "__main__":
    main()
