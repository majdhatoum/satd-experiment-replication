# analysis/evaluate_and_plot.py
from __future__ import annotations
import re, random
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

random.seed(2025)
np.random.seed(2025)

GOLD    = Path("outputs/gold_labels.csv")
INF_DIR = Path("model/inference")
OUTDIR  = Path("outputs")

# If True, the type confusion matrix will include an extra "unknown" column
# instead of dropping rows with missing/other model types.
INCLUDE_UNKNOWN_IN_CM = False

# -------------------- header / value normalization --------------------

def _norm_header(name: str) -> str:
    s = (name or "").replace("\ufeff", "").strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _pick_column(df: pd.DataFrame, synonyms: List[str]) -> Optional[str]:
    norm = {_norm_header(c): c for c in df.columns}
    for syn in synonyms:
        if syn in norm:
            return norm[syn]
    return None

def _norm_satd_value(v) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"y","yes","true","1"}:  return "yes"
    if s in {"n","no","false","0"}:  return "no"
    if s in {"not sure","not-sure","unsure","?"}: return "not-sure"
    if s in {"","nan","none","null"}: return None
    return s

TYPE_MAP = {
    # design
    "design":"design-debt","code":"design-debt","code design":"design-debt","design debt":"design-debt",
    # requirement
    "requirement":"requirement-debt","requirements":"requirement-debt","requirement debt":"requirement-debt",
    # documentation (cover common variants you’ve shown)
    "documentation":"documentation-debt","docs":"documentation-debt","doc":"documentation-debt",
    "document":"documentation-debt","documentation debt":"documentation-debt","documents":"documentation-debt",
    # test
    "test":"test-debt","testing":"test-debt","test debt":"test-debt",
}
def _norm_type_value(v) -> Optional[str]:
    s = str(v).strip()
    if s == "" or s.lower() in {"nan","none","null"}:
        return None
    s = s.lower().replace("_"," ").replace("-"," ").replace("/"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return TYPE_MAP.get(s, s)

# -------------------- metrics --------------------

def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["no","yes"]).ravel()
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return (tpr+tnr)/2

def bcaci(metric_fn, y_true, y_pred, B=10000):
    n = len(y_true)
    idx = np.arange(n)
    vals = []
    for _ in range(B):
        samp = np.random.choice(idx, size=n, replace=True)
        vals.append(metric_fn(y_true[samp], y_pred[samp]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)

# -------------------- predictions I/O --------------------

def _read_any_predictions() -> pd.DataFrame:
    if not INF_DIR.exists():
        raise FileNotFoundError(f"{INF_DIR.resolve()} does not exist")

    csvs = sorted(INF_DIR.glob("*.csv"))
    xlsx = sorted(INF_DIR.glob("*.xlsx"))
    path: Optional[Path] = csvs[0] if csvs else (xlsx[0] if xlsx else None)
    if not path:
        raise FileNotFoundError(
            f"No predictions file found in {INF_DIR}. "
            f"Put a CSV/XLSX there (e.g., 'balanced_sample - MT-TEXT-CNN.csv')."
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, engine="openpyxl")

    # header mapping (be lenient)
    item_col   = _pick_column(df, ["item id","item_id","id"])
    source_col = _pick_column(df, ["source type","source_type","source","artifact","source ty","source typ"])
    satd_col   = _pick_column(df, ["model pred satd","pred satd","satd","prediction","label"])
    type_col   = _pick_column(df, ["model pred type","pred type","satd type","type","prediction type"])

    missing = [n for n,c in {"item_id":item_col,"source_type":source_col,"satd":satd_col}.items() if c is None]
    if missing:
        raise ValueError(
            f"Predictions file {path.name} is missing required columns {missing}. "
            f"Headers seen: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "item_id": df[item_col].astype(str),
        "source_type": df[source_col].astype(str).str.strip().str.lower(),
        "model_pred_satd": df[satd_col].map(_norm_satd_value),
    })
    out["model_pred_type"] = df[type_col].map(_norm_type_value) if type_col else None
    return out

# -------------------- main --------------------

def main():
    OUTDIR.mkdir(exist_ok=True)

    # Gold labels
    gold = pd.read_csv(GOLD)
    gold["item_id"] = gold["item_id"].astype(str)
    gold["source_type"] = gold["source_type"].astype(str).str.strip().str.lower()
    # Normalize gold type as well (this fixes cases like 'document')
    gold["gold_type_norm"] = gold["gold_type"].map(_norm_type_value)

    preds = _read_any_predictions()

    # Merge on (item_id, source_type)
    data = gold.merge(
        preds[["item_id","source_type","model_pred_satd","model_pred_type"]],
        on=["item_id","source_type"], how="inner"
    )
    if data.empty:
        raise ValueError("After merging gold and predictions, no rows remain. "
                         "Check that item_id and source_type match across files.")

    # ---------- Detection metrics
    y_true = data["gold_satd"].values
    y_pred = data["model_pred_satd"].values

    prec = precision_score(y_true, y_pred, pos_label="yes")
    rec  = recall_score(y_true, y_pred, pos_label="yes")
    f1   = f1_score(y_true, y_pred, pos_label="yes")
    ba   = balanced_accuracy(y_true, y_pred)

    f1_lo, f1_hi = bcaci(lambda yt, yp: f1_score(yt, yp, pos_label="yes"), y_true, y_pred)
    ba_lo, ba_hi = bcaci(balanced_accuracy, y_true, y_pred)

    with open(OUTDIR / "metrics_detection.txt","w",encoding="utf-8") as fo:
        fo.write(f"Precision {prec:.3f}\nRecall {rec:.3f}\nF1 {f1:.3f} [{f1_lo:.3f},{f1_hi:.3f}]\n")
        fo.write(f"BalancedAcc {ba:.3f} [{ba_lo:.3f},{ba_hi:.3f}]\n")

    # ---------- Per-source BA (Figure 2)
    order = ["commit","code_comment","issue","pull_request"]
    rows = []
    for src in order:
        d = data[data["source_type"].eq(src)]
        if len(d) == 0:
            continue
        yt, yp = d["gold_satd"].values, d["model_pred_satd"].values
        s_ba = balanced_accuracy(yt, yp)
        lo, hi = bcaci(balanced_accuracy, yt, yp)
        rows.append((src, s_ba, lo, hi))
    per = pd.DataFrame(rows, columns=["source","ba","lo","hi"])
    per.to_csv(OUTDIR / "per_source_ba.csv", index=False)

    if not per.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        y = np.arange(len(per))[::-1]
        ax.errorbar(per["ba"], y, xerr=[per["ba"]-per["lo"], per["hi"]-per["ba"]],
                    fmt="o", capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(per["source"])
        ax.set_xlabel("Balanced accuracy")
        ax.set_title("Balanced Accuracy by Source (95% CI)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_forest_ba_by_source.png", dpi=300)

    # ---------- Type confusion (Figure 1)
    labels  = ["design-debt","requirement-debt","documentation-debt","test-debt"]
    allowed = set(labels)

    base = data[data["gold_satd"].eq("yes")].copy()
    base["gold_type_norm"]  = base["gold_type_norm"].map(_norm_type_value)  # idempotent
    base["model_type_norm"] = base["model_pred_type"].map(_norm_type_value)

    # Build an explicit audit trail for excluded rows
    reasons = []
    for _, r in base.iterrows():
        if r["gold_type_norm"] not in allowed:
            reasons.append((
                "gold_unknown_or_missing",
                r["item_id"], r["source_type"],
                r.get("gold_type"), r.get("model_pred_type"),
                r.get("model_pred_satd")
            ))
        elif r["model_type_norm"] not in allowed:
            # distinguish: model predicted NO SATD vs. type string missing/other
            if str(r.get("model_pred_satd")).lower() != "yes":
                reasons.append((
                    "model_pred_satd_is_no_or_missing",
                    r["item_id"], r["source_type"],
                    r.get("gold_type"), r.get("model_pred_type"),
                    r.get("model_pred_satd")
                ))
            else:
                reasons.append((
                    "model_type_unknown_or_missing",
                    r["item_id"], r["source_type"],
                    r.get("gold_type"), r.get("model_pred_type"),
                    r.get("model_pred_satd")
                ))

    if reasons:
        drop_df = pd.DataFrame(
            reasons,
            columns=["reason","item_id","source_type","gold_type_raw","model_pred_type_raw","model_pred_satd"]
        )
        drop_df.to_csv(OUTDIR / "dropped_for_type_confusion.csv", index=False, encoding="utf-8")

    valid_mask = base["gold_type_norm"].isin(allowed)
    if INCLUDE_UNKNOWN_IN_CM:
        # put non-allowed/missing model types into an "unknown" bin
        base["model_type_bins"] = np.where(
            base["model_type_norm"].isin(allowed), base["model_type_norm"], "unknown"
        )
        used_labels = labels + ["unknown"]
        satd_for_cm = base.loc[valid_mask].copy()
        ytrue = satd_for_cm["gold_type_norm"].values
        ypred = satd_for_cm["model_type_bins"].values
        cm = confusion_matrix(ytrue, ypred, labels=used_labels)
        fig2, ax2 = plt.subplots(figsize=(6.0,4.8))
        im = ax2.imshow(cm, interpolation="nearest")
        ax2.set_xticks(range(len(used_labels))); ax2.set_xticklabels(used_labels, rotation=30, ha="right")
        ax2.set_yticks(range(len(labels)));     ax2.set_yticklabels(labels)
        ax2.set_title("Type Confusion Matrix (SATD only)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, cm[i, j], ha="center", va="center")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(OUTDIR / "fig_type_confusion.png", dpi=300)
    else:
        # strict: keep only rows where both gold & model are one of the four labels
        strict = valid_mask & base["model_type_norm"].isin(allowed)
        satd_valid = base.loc[strict].copy()
        if satd_valid.empty:
            print("⚠ Skipping Type Confusion Matrix: no items with both gold type and valid predicted type.")
        else:
            ytrue = satd_valid["gold_type_norm"].values
            ypred = satd_valid["model_type_norm"].values
            cm = confusion_matrix(ytrue, ypred, labels=labels)
            fig2, ax2 = plt.subplots(figsize=(5.5,4.5))
            im = ax2.imshow(cm, interpolation="nearest")
            ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=30, ha="right")
            ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
            ax2.set_title("Type Confusion Matrix (SATD only)")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax2.text(j, i, cm[i, j], ha="center", va="center")
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            fig2.tight_layout()
            fig2.savefig(OUTDIR / "fig_type_confusion.png", dpi=300)

    print("✓ Wrote outputs/metrics_detection.txt")
    print("✓ Wrote outputs/per_source_ba.csv")
    if (OUTDIR / "fig_forest_ba_by_source.png").exists():
        print("✓ Wrote outputs/fig_forest_ba_by_source.png")
    if (OUTDIR / "fig_type_confusion.png").exists():
        print("✓ Wrote outputs/fig_type_confusion.png")
    if (OUTDIR / "dropped_for_type_confusion.csv").exists():
        print("ℹ See outputs/dropped_for_type_confusion.csv for exact exclusion reasons")

if __name__ == "__main__":
    main()
