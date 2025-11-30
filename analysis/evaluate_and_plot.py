# analysis/evaluate_and_plot.py
from __future__ import annotations
import re, random, math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import binomtest  # for McNemar exact two-sided

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
    """
    Robust header normalizer:
    - lowercase
    - strip BOM and whitespace (incl. NBSP)
    - replace _, -, / with spaces
    - collapse runs of whitespace to a single space
    """
    s = (name or "")
    s = s.replace("\ufeff", "")      # BOM
    s = s.replace("\xa0", " ")       # NBSP -> space
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _pick_column(df: pd.DataFrame, synonyms: list[str]) -> Optional[str]:
    norm = {_norm_header(c): c for c in df.columns}
    for syn in synonyms:
        syn_norm = _norm_header(syn)
        if syn_norm in norm:
            return norm[syn_norm]
    # fallback: case-insensitive exact match on raw names
    wanted = _norm_header(synonyms[0])
    for c in df.columns:
        if _norm_header(c) == wanted:
            return c
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
    # documentation
    "documentation":"documentation-debt","docs":"documentation-debt","doc":"documentation-debt",
    "document":"documentation-debt","documents":"documentation-debt","documentation debt":"documentation-debt",
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

# ----- type metrics (one-vs-rest) including “no-type” as a miss -----

TYPE_LABELS = ["design-debt","requirement-debt","documentation-debt","test-debt"]

def per_type_counts(gold_types: pd.Series, pred_types: pd.Series) -> Dict[str, Dict[str,int]]:
    """
    gold_types: SATD-only items with canonical gold label in TYPE_LABELS
    pred_types: model's type; values outside TYPE_LABELS (including None) are treated as 'none'
    Returns dict: cls -> {tp, fp, fn, support}
    """
    gt = gold_types.values
    pt_raw = pred_types.values
    pt = np.array([p if p in TYPE_LABELS else "none" for p in pt_raw], dtype=object)

    out: Dict[str, Dict[str,int]] = {}
    for c in TYPE_LABELS:
        support = int(np.sum(gt == c))
        tp = int(np.sum((gt == c) & (pt == c)))
        fp = int(np.sum((gt != c) & (pt == c)))
        fn = int(np.sum((gt == c) & (pt != c)))
        out[c] = {"tp":tp, "fp":fp, "fn":fn, "support":support}
    return out

def per_type_prf(counts: Dict[str, Dict[str,int]]) -> pd.DataFrame:
    rows=[]
    for c, d in counts.items():
        tp, fp, fn, sup = d["tp"], d["fp"], d["fn"], d["support"]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
        rows.append((c, sup, prec, rec, f1, tp, fp, fn))
    return pd.DataFrame(rows, columns=["type","support","precision","recall","f1","tp","fp","fn"])

def macro_f1_from_series(gold_types: np.ndarray, pred_types: np.ndarray) -> float:
    counts = per_type_counts(pd.Series(gold_types), pd.Series(pred_types))
    df = per_type_prf(counts)
    return float(df["f1"].mean())

def bcaci_macro_f1(gold_types: np.ndarray, pred_types: np.ndarray, B=10000) -> Tuple[float,float]:
    n = len(gold_types)
    idx = np.arange(n)
    vals=[]
    for _ in range(B):
        samp = np.random.choice(idx, size=n, replace=True)
        vals.append(macro_f1_from_series(gold_types[samp], pred_types[samp]))
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

    # ---------- Dataset composition summary
    comp_rows=[]
    for src, df in data.groupby("source_type", dropna=False):
        comp_rows.append((
            src, len(df),
            int((df["gold_satd"]=="yes").sum()),
            int((df["gold_satd"]=="no").sum())
        ))
    comp = pd.DataFrame(comp_rows, columns=["source","N","gold_satd_yes","gold_satd_no"]).sort_values("source")
    comp.loc["overall"] = ["overall", int(len(data)),
                           int((data["gold_satd"]=="yes").sum()),
                           int((data["gold_satd"]=="no").sum())]
    comp.to_csv(OUTDIR / "dataset_summary.csv", index=False)

    # ---------- Detection metrics (overall)
    y_true = data["gold_satd"].values
    y_pred = data["model_pred_satd"].values

    # confusion & counts
    cm = confusion_matrix(y_true, y_pred, labels=["no","yes"])
    tn, fp, fn, tp = cm.ravel()

    prec = precision_score(y_true, y_pred, pos_label="yes")
    rec  = recall_score(y_true, y_pred, pos_label="yes")
    f1   = f1_score(y_true, y_pred, pos_label="yes")
    ba   = balanced_accuracy(y_true, y_pred)

    f1_lo, f1_hi = bcaci(lambda yt, yp: f1_score(yt, yp, pos_label="yes"), y_true, y_pred)
    ba_lo, ba_hi = bcaci(balanced_accuracy, y_true, y_pred)

    # McNemar b,c and exact p
    b = int(fp)  # model=yes, gold=no
    c = int(fn)  # model=no,  gold=yes
    n_disc = b + c
    if n_disc > 0:
        mcn = binomtest(k=min(b,c), n=n_disc, p=0.5, alternative="two-sided")
        pval = float(mcn.pvalue)
        # continuity-corrected chi^2 (optional, informational)
        chi2 = ((abs(b - c) - 1)**2) / n_disc if n_disc > 0 else float("nan")
    else:
        pval, chi2 = 1.0, 0.0

    with open(OUTDIR / "metrics_detection.txt","w",encoding="utf-8") as fo:
        fo.write(f"N {len(y_true)} | TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")
        fo.write(f"Precision {prec:.3f}\nRecall {rec:.3f}\nF1 {f1:.3f} [{f1_lo:.3f},{f1_hi:.3f}]\n")
        fo.write(f"BalancedAcc {ba:.3f} [{ba_lo:.3f},{ba_hi:.3f}]\n")
        fo.write(f"McNemar b={b}, c={c}, chi2={chi2:.3f}, p={pval:.3f}\n")

    # ---------- Per-source detection (counts + P/R/F1/BA with BA CI)
    per_rows=[]
    for src in ["commit","code_comment","issue","pull_request"]:
        d = data[data["source_type"].eq(src)]
        if len(d)==0: 
            continue
        yt, yp = d["gold_satd"].values, d["model_pred_satd"].values
        cm_s = confusion_matrix(yt, yp, labels=["no","yes"])
        tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
        prec_s = precision_score(yt, yp, pos_label="yes") if (tp_s+fp_s)>0 else 0.0
        rec_s  = recall_score(yt, yp, pos_label="yes")     if (tp_s+fn_s)>0 else 0.0
        f1_s   = f1_score(yt, yp, pos_label="yes")         if (prec_s+rec_s)>0 else 0.0
        ba_s   = balanced_accuracy(yt, yp)
        lo, hi = bcaci(balanced_accuracy, yt, yp)
        per_rows.append((src, len(d), tp_s, tn_s, fp_s, fn_s, prec_s, rec_s, f1_s, ba_s, lo, hi))

    per = pd.DataFrame(
        per_rows,
        columns=["source","N","TP","TN","FP","FN","precision","recall","f1","ba","ba_lo","ba_hi"]
    )
    per.to_csv(OUTDIR / "per_source_detection_full.csv", index=False)

    # ---------- Forest plot (Figure 2: BA by source)
    if not per.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        y = np.arange(len(per))[::-1]
        ax.errorbar(per["ba"], y, xerr=[per["ba"]-per["ba_lo"], per["ba_hi"]-per["ba"]],
                    fmt="o", capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(per["source"])
        ax.set_xlabel("Balanced accuracy")
        ax.set_title("Balanced Accuracy by Source (95% CI)")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUTDIR / "fig_forest_ba_by_source.png", dpi=300)

    # ---------- Type analysis (SATD-only; do NOT drop “model said no” for metrics)
    labels  = TYPE_LABELS
    allowed = set(labels)

    satd_only = data[(data["gold_satd"].eq("yes")) & (gold["gold_type"].notna())].copy()
    satd_only["gold_type_norm"]  = satd_only["gold_type"].map(_norm_type_value)
    satd_only["model_type_norm"] = satd_only["model_pred_type"].map(_norm_type_value)

    # keep only items with canonical gold label
    satd_valid = satd_only[satd_only["gold_type_norm"].isin(allowed)].copy()

    # per-type counts/metrics (treat non-canonical/missing predicted type as 'none' → a miss)
    counts = per_type_counts(satd_valid["gold_type_norm"], satd_valid["model_type_norm"])
    type_df = per_type_prf(counts)
    type_df.to_csv(OUTDIR / "type_metrics.csv", index=False)

    # macro-F1 + CI (bootstrap over SATD-valid items)
    macro_f1 = float(type_df["f1"].mean())
    m_lo, m_hi = bcaci_macro_f1(satd_valid["gold_type_norm"].values,
                                satd_valid["model_type_norm"].values)
    with open(OUTDIR / "type_macro_f1.txt","w",encoding="utf-8") as fo:
        fo.write(f"MacroF1 {macro_f1:.3f} [{m_lo:.3f},{m_hi:.3f}]\n")
        fo.write(f"Support per type: {dict(zip(type_df['type'], type_df['support']))}\n")
        fo.write(f"Total SATD items with canonical gold type: {len(satd_valid)}\n")

    # ---------- Type confusion (Figure 1) — for visualization only
    # Build an explicit audit trail for excluded rows from the figure
    base = satd_only.copy()
    base["gold_type_norm"]  = base["gold_type_norm"].map(_norm_type_value)  # idempotent
    base["model_type_norm"] = base["model_type_norm"].map(_norm_type_value)

    reasons = []
    for _, r in base.iterrows():
        if r["gold_type_norm"] not in allowed:
            reasons.append(("gold_unknown_or_missing", r["item_id"], r["source_type"],
                            r.get("gold_type"), r.get("model_pred_type"), r.get("model_pred_satd")))
        elif r["model_type_norm"] not in allowed:
            if str(r.get("model_pred_satd")).lower() != "yes":
                reasons.append(("model_pred_satd_is_no_or_missing", r["item_id"], r["source_type"],
                                r.get("gold_type"), r.get("model_pred_type"), r.get("model_pred_satd")))
            else:
                reasons.append(("model_type_unknown_or_missing", r["item_id"], r["source_type"],
                                r.get("gold_type"), r.get("model_pred_type"), r.get("model_pred_satd")))
    if reasons:
        pd.DataFrame(reasons, columns=[
            "reason","item_id","source_type","gold_type_raw","model_pred_type_raw","model_pred_satd"
        ]).to_csv(OUTDIR / "dropped_for_type_confusion.csv", index=False, encoding="utf-8")

    valid_mask = base["gold_type_norm"].isin(allowed)
    if INCLUDE_UNKNOWN_IN_CM:
        base["model_type_bins"] = np.where(
            base["model_type_norm"].isin(allowed), base["model_type_norm"], "unknown"
        )
        used_labels = labels + ["unknown"]
        satd_for_cm = base.loc[valid_mask].copy()
        ytrue = satd_for_cm["gold_type_norm"].values
        ypred = satd_for_cm["model_type_bins"].values
        cm_t = confusion_matrix(ytrue, ypred, labels=used_labels)
        fig2, ax2 = plt.subplots(figsize=(6.0,4.8))
        im = ax2.imshow(cm_t, interpolation="nearest")
        ax2.set_xticks(range(len(used_labels))); ax2.set_xticklabels(used_labels, rotation=30, ha="right")
        ax2.set_yticks(range(len(labels)));     ax2.set_yticklabels(labels)
        ax2.set_title("Type Confusion Matrix (SATD only)")
        for i in range(cm_t.shape[0]):
            for j in range(cm_t.shape[1]):
                ax2.text(j, i, cm_t[i, j], ha="center", va="center")
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        fig2.tight_layout()
        fig2.savefig(OUTDIR / "fig_type_confusion.png", dpi=300)
    else:
        strict = valid_mask & base["model_type_norm"].isin(allowed)
        satd_fig = base.loc[strict].copy()
        if not satd_fig.empty:
            ytrue = satd_fig["gold_type_norm"].values
            ypred = satd_fig["model_type_norm"].values
            cm_t = confusion_matrix(ytrue, ypred, labels=labels)
            fig2, ax2 = plt.subplots(figsize=(5.5,4.5))
            im = ax2.imshow(cm_t, interpolation="nearest")
            ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=30, ha="right")
            ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels)
            ax2.set_title("Type Confusion Matrix (SATD only)")
            for i in range(cm_t.shape[0]):
                for j in range(cm_t.shape[1]):
                    ax2.text(j, i, cm_t[i, j], ha="center", va="center")
            fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            fig2.tight_layout()
            fig2.savefig(OUTDIR / "fig_type_confusion.png", dpi=300)
        else:
            print("⚠ Skipping Type Confusion Matrix: no items with both gold type and valid predicted type.")

    print("✓ Wrote outputs/metrics_detection.txt (incl. TP/TN/FP/FN and McNemar)")
    print("✓ Wrote outputs/per_source_detection_full.csv")
    if (OUTDIR / "fig_forest_ba_by_source.png").exists():
        print("✓ Wrote outputs/fig_forest_ba_by_source.png")
    if (OUTDIR / "type_metrics.csv").exists():
        print("✓ Wrote outputs/type_metrics.csv and type_macro_f1.txt")
    if (OUTDIR / "fig_type_confusion.png").exists():
        print("✓ Wrote outputs/fig_type_confusion.png")
    if (OUTDIR / "dataset_summary.csv").exists():
        print("✓ Wrote outputs/dataset_summary.csv")
    if (OUTDIR / "dropped_for_type_confusion.csv").exists():
        print("ℹ See outputs/dropped_for_type_confusion.csv for exact exclusion reasons")

if __name__ == "__main__":
    main()
