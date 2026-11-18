"""Turn the cleaned weekly records into one row of behaviour features per learner."""
import numpy as np
import pandas as pd

W = 12


def _matrix(weekly, col, ids):
    return weekly.pivot(index="learner_id", columns="week", values=col).reindex(ids)


def rolling_deviation(rate, window=4):
    """Rate minus the learner's own average of the PREVIOUS weeks only (no peeking ahead)."""
    base = rate.T.rolling(window, min_periods=2).mean().shift(1).T
    return rate - base


def _longest_run(isnan_row):
    best = cur = 0
    for v in isnan_row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _robust_z(values, groups, floor):
    """Distance from the batch median in 'typical spreads' (robust to outliers)."""
    med = values.groupby(groups).transform("median")
    mad = (values - med).abs().groupby(groups).transform("median") * 1.4826
    return ((values - med) / np.maximum(mad, floor)).fillna(0)


def build_features(enrol, weekly):
    ids = enrol["learner_id"].drop_duplicates().to_numpy()
    held = _matrix(weekly, "sessions_held", ids)
    att = _matrix(weekly, "sessions_attended", ids)
    sub = _matrix(weekly, "assignments_submitted", ids)
    late = _matrix(weekly, "days_late", ids)
    score = _matrix(weekly, "assessment_score", ids)
    rate = att / held

    f = pd.DataFrame(index=pd.Index(ids, name="learner_id"))
    f["attendance_rate"] = rate.mean(axis=1)
    f["missing_weeks"] = att.isna().sum(axis=1)
    f["longest_missing_run"] = [_longest_run(r) for r in att.isna().to_numpy()]
    last_active = (att.fillna(0) > 0).to_numpy()
    last_idx = np.where(last_active.any(axis=1), W - 1 - np.argmax(last_active[:, ::-1], axis=1), -1)
    f["weeks_since_last_attendance"] = W - 1 - last_idx
    f["submit_rate"] = sub.mean(axis=1)
    f["days_late_log"] = np.log1p(late).mean(axis=1).fillna(0)
    f["score_mean"] = score.mean(axis=1)
    first = score.iloc[:, : W // 2].mean(axis=1)
    last = score.iloc[:, W // 2:].mean(axis=1)
    f["score_change"] = (last - first).fillna(0)
    f["rate_dev_min"] = rolling_deviation(rate).min(axis=1).fillna(0)

    info = enrol.drop_duplicates("learner_id").set_index("learner_id").reindex(ids)
    batch = info["batch_id"]
    f["peer_attendance_z"] = _robust_z(f["attendance_rate"], batch, 0.03)
    f["peer_score_z"] = _robust_z(f["score_mean"], batch, 3.0)
    f["dup_enrolment"] = enrol.groupby("learner_id")["dup_enrolment"].max().reindex(ids).fillna(0)
    f["active_but_inactive"] = ((info["status"] == "active") & (f["weeks_since_last_attendance"] >= 4)).astype(int)
    combo = info["centre"] + "|" + info["course"]
    f["rare_combo"] = -np.log(combo.map(combo.value_counts(normalize=True)))
    f["score_mean"] = f["score_mean"].fillna(f["score_mean"].median())

    # Encoding
    f["status_dropped"] = (info["status"] == "dropped").astype(int)
    f["centre_freq"] = info["centre"].map(info["centre"].value_counts(normalize=True))
    for course in sorted(info["course"].dropna().unique()):
        f["course_" + course.replace(" ", "_")] = (info["course"] == course).astype(int)
    return f
