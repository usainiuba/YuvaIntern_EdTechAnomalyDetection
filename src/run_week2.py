"""Run the whole Week 2 pipeline and save numbers and figures.

Usage:  python -m src.run_week2
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import build_features, rolling_deviation
from src.generate_data import CENTRES, COURSES, SEED, generate
from src.prepare_data import audit, clean, outlier_counts

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
FIG = OUT / "figures"
NAVY, TEAL, ORANGE, GREY = "#1F3A5F", "#0E7C86", "#D97706", "#9AA8B8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

RAW_COLS = ["raw_total_attended", "raw_total_submitted", "raw_mean_score", "raw_mean_days_late"]


def precision_at(score, y, frac):
    k = int(round(frac * len(y)))
    return float(y[np.argsort(-score)[:k]].mean())


def run_forest(X, y, seeds=range(5)):
    """Isolation Forest on scaled features. Returns mean and std over seeds."""
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for s in seeds:
        score = -IsolationForest(n_estimators=200, random_state=s).fit(Xs).score_samples(Xs)
        rows.append([precision_at(score, y, 0.01), precision_at(score, y, 0.05), roc_auc_score(y, score)])
    rows = np.array(rows)
    return {k: [float(rows[:, i].mean()), float(rows[:, i].std())]
            for i, k in enumerate(["precision_top1", "precision_top5", "roc_auc"])}, score


def raw_features(weekly, ids):
    g = weekly.groupby("learner_id")
    r = pd.DataFrame({
        "raw_total_attended": g["sessions_attended"].sum(min_count=0),
        "raw_total_submitted": g["assignments_submitted"].sum(),
        "raw_mean_score": g["assessment_score"].mean(),
        "raw_mean_days_late": g["days_late"].mean(),
    }).reindex(ids)
    return r.fillna(0)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    res = {}

    # 1. Data and audit
    d = generate()
    for name, df in d.items():
        df.to_csv(ROOT / "data" / f"{name}.csv", index=False)
    enrol_raw, weekly_raw, labels = d["enrolments"], d["weekly"], d["labels"]
    res["audit"] = audit(enrol_raw, weekly_raw)
    res["anomaly_counts"] = labels["anomaly_type"].value_counts().to_dict()

    # 2. Cleaning
    enrol, weekly, log = clean(enrol_raw, weekly_raw, CENTRES, COURSES)
    res["cleaning"] = log
    res["audit_after"] = {"attended_missing": int(weekly["sessions_attended"].isna().sum()),
                          "missing_pct": (weekly.drop(columns=["learner_id", "week"]).isna().mean() * 100).round(2).to_dict()}

    # 3. Outliers and transformation
    res["outliers_score"] = outlier_counts(weekly["assessment_score"])
    late = weekly["days_late"].dropna()
    res["outliers_days_late"] = outlier_counts(late)
    res["skew"] = {"days_late_raw": float(skew(late)), "days_late_log1p": float(skew(np.log1p(late)))}
    res["days_late_summary"] = {"median": float(late.median()), "mean": float(late.mean()), "max": float(late.max())}

    # 4. Feature table
    feats = build_features(enrol, weekly)
    y = labels.set_index("learner_id").loc[feats.index, "is_anomaly"].to_numpy()
    types = labels.set_index("learner_id").loc[feats.index, "anomaly_type"].to_numpy()
    res["n_features_built"] = int(feats.shape[1])

    # 5. Validation: no peeking into the future
    ids = feats.index
    held = weekly.pivot(index="learner_id", columns="week", values="sessions_held").reindex(ids)
    att = weekly.pivot(index="learner_id", columns="week", values="sessions_attended").reindex(ids)
    rate = att / held
    dev_full = rolling_deviation(rate)
    dev_cut = rolling_deviation(rate.iloc[:, :8])
    res["leakage_check_passed"] = bool(np.allclose(dev_full.iloc[:, :8].fillna(-9), dev_cut.fillna(-9)))

    # 6. Selection: a 300-case labelled sample is used to choose features; it is then held out.
    rng = np.random.default_rng(SEED)
    pos = rng.choice(np.where(y == 1)[0], 75, replace=False)
    neg = rng.choice(np.where(y == 0)[0], 225, replace=False)
    lab_idx = np.concatenate([pos, neg])
    test_mask = np.ones(len(y), bool)
    test_mask[lab_idx] = False
    scaled = pd.DataFrame(StandardScaler().fit_transform(feats), index=feats.index, columns=feats.columns)
    corr = feats.corr().abs()
    dropped_corr, keep = [], list(feats.columns)
    for i, a in enumerate(feats.columns):
        for b in feats.columns[i + 1:]:
            if a in keep and b in keep and corr.loc[a, b] > 0.9:
                keep.remove(b)
                dropped_corr.append([b, a, float(corr.loc[a, b])])
    mi = mutual_info_classif(scaled.iloc[lab_idx][keep], y[lab_idx], random_state=SEED)
    mi = pd.Series(mi, index=keep).sort_values(ascending=False)
    selected = list(mi.index[:10])
    res["selection"] = {"dropped_for_correlation": dropped_corr, "mutual_information": mi.round(4).to_dict(),
                        "selected": selected, "not_selected": [c for c in keep if c not in selected],
                        "labelled_sample": 300, "evaluated_on": int(test_mask.sum())}

    # 7. Does feature engineering help? (Isolation Forest, evaluated on the held-out learners)
    raw = raw_features(weekly_raw, ids)
    y_t = y[test_mask]
    results = {}
    results["raw"], _ = run_forest(raw[test_mask], y_t)
    results["all_engineered"], _ = run_forest(feats[test_mask], y_t)
    results["selected"], best_score = run_forest(feats.loc[test_mask, selected], y_t)
    res["detector"] = results
    res["n_raw_features"] = len(RAW_COLS)
    top5 = np.argsort(-best_score)[: int(round(0.05 * len(y_t)))]
    t_t = types[test_mask]
    res["caught_in_top5"] = {k: [int((t_t[top5] == k).sum()), int((t_t == k).sum())]
                             for k in ["missing_attendance_block", "active_dropout", "duplicate_enrolment", "score_jump"]}

    # 8. Scaling on a public dataset (UCI Breast Cancer Wisconsin, bundled with scikit-learn)
    X, t = load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {"SVM (RBF)": SVC(), "KNN": KNeighborsClassifier(), "Logistic Regression": LogisticRegression(max_iter=5000)}
    res["scaling_uci"] = {n: [float(cross_val_score(m, X, t, cv=cv).mean()),
                              float(cross_val_score(make_pipeline(StandardScaler(), m), X, t, cv=cv).mean())]
                          for n, m in models.items()}
    res["uci_feature_ranges"] = {"smallest_range_feature": float(np.ptp(X, axis=0).min()), "largest_range_feature": float(np.ptp(X, axis=0).max())}

    # 9. Worked examples for the report
    def example(kind):
        lid = labels.loc[labels["anomaly_type"] == kind, "learner_id"].iloc[3]
        r = weekly_raw[weekly_raw["learner_id"] == lid].sort_values("week")
        return {"learner_id": lid,
                "attended_raw": [None if np.isnan(v) else int(v) for v in r["sessions_attended"]],
                "held": [int(v) for v in r["sessions_held"]],
                "features": {k: round(float(feats.loc[lid, k]), 2) for k in
                             ["attendance_rate", "longest_missing_run", "weeks_since_last_attendance", "active_but_inactive",
                              "rate_dev_min", "peer_attendance_z", "score_change", "dup_enrolment"]}}
    res["examples"] = {k: example(k) for k in ["active_dropout", "missing_attendance_block"]}
    typical = feats.loc[labels.loc[labels["anomaly_type"] == "none", "learner_id"]]
    res["typical_features"] = {k: round(float(typical[k].median()), 2) for k in
                               ["attendance_rate", "longest_missing_run", "weeks_since_last_attendance", "rate_dev_min", "peer_attendance_z", "score_change"]}

    (OUT / "week2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    feats.assign(is_anomaly=y).to_csv(ROOT / "data" / "features.csv")

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    items = [("Impossible days-late set to missing", log["invalid_days_late_set_missing"], NAVY),
             ("Impossible attendance set to missing", log["invalid_attended_set_missing"], NAVY),
             ("Impossible scores set to missing", log["invalid_scores_set_missing"], NAVY),
             ("Exact duplicate rows dropped", log["exact_duplicates_dropped"], NAVY),
             ("Small attendance gaps filled", log["small_gap_cells_filled"], TEAL),
             ("Long attendance gaps kept and flagged", log["block_gap_cells_kept"], ORANGE),
             ("Duplicate enrolments flagged (learners)", log["duplicate_enrolments_flagged"], ORANGE)]
    ax.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1], color=[i[2] for i in items][::-1])
    for y_, (_, v, _) in enumerate(items[::-1]):
        ax.text(v + 10, y_, f"{v:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of cells or rows")
    ax.set_title("Problems found in the raw data and what was done", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig1_issues.png", dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axs[0].hist(late.clip(upper=40), bins=40, color=NAVY)
    axs[0].set_title(f"Raw days late (skew {res['skew']['days_late_raw']:.2f})", fontsize=10, fontweight="bold")
    axs[1].hist(np.log1p(late), bins=40, color=TEAL)
    axs[1].set_title(f"After log(1 + x) (skew {res['skew']['days_late_log1p']:.2f})", fontsize=10, fontweight="bold")
    for a in axs: a.set_xlabel("Days"); a.set_ylabel("Records")
    fig.tight_layout(); fig.savefig(FIG / "fig2_skew.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    names = [("raw", "Raw columns (4)", GREY), ("all_engineered", "All engineered (21)", NAVY), ("selected", "Selected (10)", TEAL)]
    metrics = [("precision_top1", "Precision, top 1%"), ("precision_top5", "Precision, top 5%"), ("roc_auc", "ROC-AUC")]
    width = 0.26
    for j, (key, label, color) in enumerate(names):
        vals = [results[key][m][0] for m, _ in metrics]
        errs = [results[key][m][1] for m, _ in metrics]
        bars = ax.bar(np.arange(3) + (j - 1) * width, vals, width, yerr=errs, label=label, color=color, capsize=2)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in metrics]); ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_title("Same detector, different features", loc="left", fontweight="bold", pad=22)
    fig.tight_layout(); fig.savefig(FIG / "fig3_results.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    top = mi.head(12)[::-1]
    ax.barh(top.index, top.values, color=[TEAL if c in selected else GREY for c in top.index])
    ax.set_xlabel("Mutual information with the label (300 labelled cases)")
    ax.set_title("Which features carry the most signal", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig4_mi.png", dpi=170); plt.close(fig)
    print(json.dumps({k: res[k] for k in ["cleaning", "outliers_score", "outliers_days_late", "skew", "leakage_check_passed",
                                         "detector", "caught_in_top5", "scaling_uci"]}, indent=1))
    print("selected:", selected)
    print("dropped for correlation:", dropped_corr)


if __name__ == "__main__":
    main()
