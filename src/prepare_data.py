"""Audit and clean the raw tables.

Rule used throughout: fix data FAULTS, but never delete or hide possible
ANOMALIES. A learner who is enrolled twice, or who has a long block of missing
attendance, is exactly what the reviewers want to see, so we flag those.
"""
import numpy as np
import pandas as pd

GAP_LIMIT = 3          # missing runs shorter than this are treated as small gaps
WEEKS = 12


def audit(enrol, weekly):
    """Count the problems in the raw data."""
    held_ok = weekly["sessions_held"] > 0
    return {
        "enrolment_rows": int(len(enrol)),
        "weekly_rows": int(len(weekly)),
        "exact_duplicate_rows": int(enrol.duplicated().sum()),
        "unique_centre_spellings": int(enrol["centre"].nunique()),
        "unique_course_spellings": int(enrol["course"].nunique()),
        "attended_missing": int((held_ok & weekly["sessions_attended"].isna()).sum()),
        "attended_more_than_held": int((weekly["sessions_attended"] > weekly["sessions_held"]).sum()),
        "scores_outside_0_100": int(((weekly["assessment_score"] < 0) | (weekly["assessment_score"] > 100)).sum()),
        "negative_days_late": int((weekly["days_late"] < 0).sum()),
        "missing_pct": (weekly.drop(columns=["learner_id", "week"]).isna().mean() * 100).round(2).to_dict(),
    }


def _canonical(series, canon):
    """Map messy spellings to the official name (ignore case and extra spaces)."""
    lookup = {c.lower(): c for c in canon}
    cleaned = series.str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    return cleaned.map(lookup)


def _runs(isnan_row):
    """Return (start, length) of each run of True in a boolean row."""
    runs, start = [], None
    for i, v in enumerate(isnan_row):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(isnan_row) - start))
    return runs


def clean(enrol, weekly, centres, courses):
    log = {}
    e = enrol.copy()
    w = weekly.copy()

    # 1. Text spelling
    e["centre"] = _canonical(e["centre"], centres)
    e["course"] = _canonical(e["course"], courses)
    log["centre_spellings_after"] = int(e["centre"].nunique())
    log["course_spellings_after"] = int(e["course"].nunique())
    log["unmapped_text"] = int(e["centre"].isna().sum() + e["course"].isna().sum())

    # 2. Exact duplicate rows are copy-paste faults: drop them.
    n0 = len(e)
    e = e.drop_duplicates().reset_index(drop=True)
    log["exact_duplicates_dropped"] = int(n0 - len(e))

    # 3. Same learner in the same batch with a different enrolment id: flag, do not drop.
    size = e.groupby(["learner_id", "batch_id"])["enrolment_id"].transform("size")
    e["dup_enrolment"] = (size > 1).astype(int)
    log["duplicate_enrolments_flagged"] = int(e.loc[e["dup_enrolment"] == 1, "learner_id"].nunique())

    # 4. Impossible values become missing (not clipped), so they cannot pose as real data.
    bad_att = w["sessions_attended"] > w["sessions_held"]
    bad_score = (w["assessment_score"] < 0) | (w["assessment_score"] > 100)
    bad_late = w["days_late"] < 0
    w.loc[bad_att, "sessions_attended"] = np.nan
    w.loc[bad_score, "assessment_score"] = np.nan
    w.loc[bad_late, "days_late"] = np.nan
    log["invalid_attended_set_missing"] = int(bad_att.sum())
    log["invalid_scores_set_missing"] = int(bad_score.sum())
    log["invalid_days_late_set_missing"] = int(bad_late.sum())

    # 5. Missing attendance: fill small gaps, keep (and flag) long blocks.
    att = w.pivot(index="learner_id", columns="week", values="sessions_attended")
    values = att.to_numpy(copy=True)
    nan = np.isnan(values)
    small = block = learners_with_block = 0
    for i in range(len(values)):
        has_block = False
        median = np.nanmedian(values[i]) if (~nan[i]).any() else 0.0
        for start, length in _runs(nan[i]):
            if length < GAP_LIMIT:
                values[i, start:start + length] = round(median)
                small += length
            else:
                block += length
                has_block = True
        learners_with_block += int(has_block)
    filled = pd.DataFrame(values, index=att.index, columns=att.columns).stack().rename("sessions_attended").reset_index()
    w = w.drop(columns="sessions_attended").merge(filled, on=["learner_id", "week"], how="left")
    # A filled value can never be larger than the sessions held that week.
    w["sessions_attended"] = w["sessions_attended"].clip(upper=w["sessions_held"])
    log["small_gap_cells_filled"] = int(small)
    log["block_gap_cells_kept"] = int(block)
    log["learners_with_missing_block"] = int(learners_with_block)
    return e, w, log


def outlier_counts(values):
    """How many points each outlier rule flags (they disagree, which is the lesson)."""
    x = pd.Series(values).dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    z = (x - x.mean()) / x.std()
    mad = (x - x.median()).abs().median()
    robust = 0.6745 * (x - x.median()) / mad
    return {
        "n": int(len(x)),
        "iqr_1_5": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
        "zscore_3": int((z.abs() > 3).sum()),
        "mad_3_5": int((robust.abs() > 3.5).sum()),
    }
