"""Create a fictional EdTech dataset with realistic faults and known anomalies.

No real learner data is used. The data imitates a chain of coaching institutes
in Gurgaon: learners join batches, then every week we record attendance,
assignments and assessment scores. A few learners are made anomalous on purpose,
so we always know the right answer and can test our methods honestly.
"""
import numpy as np
import pandas as pd

SEED = 42
WEEKS = 12
CENTRES = ["Sector 14", "Sector 29", "DLF Phase 3", "Sohna Road", "MG Road", "Golf Course Road"]
COURSES = ["JEE Foundation", "NEET Foundation", "Class 10 Boards", "Class 12 Boards", "Spoken English"]
ANOMALY_TYPES = ["missing_attendance_block", "active_dropout", "duplicate_enrolment", "score_jump"]


def generate(n_learners=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    n, W = n_learners, WEEKS

    # Batches: each has a centre and a course.
    n_batches = 120
    batches = pd.DataFrame({
        "batch_id": [f"B{i:03d}" for i in range(n_batches)],
        "centre": rng.choice(CENTRES, n_batches, p=[.22, .20, .20, .16, .14, .08]),
        "course": rng.choice(COURSES, n_batches, p=[.30, .20, .20, .20, .10]),
        "batch_effect": rng.normal(0, 0.04, n_batches),
    })
    learners = pd.DataFrame({
        "learner_id": [f"L{i:05d}" for i in range(n)],
        "batch_id": rng.choice(batches["batch_id"], n),
    }).merge(batches, on="batch_id", how="left")

    # Each learner has their own habits.
    p_att = np.clip(rng.beta(8, 2, n) + learners["batch_effect"].to_numpy(), 0.35, 0.99)
    mu_score = np.clip(rng.normal(62, 12, n) + 25 * (p_att - 0.8), 25, 92)
    p_sub = np.clip(0.55 + 0.4 * p_att + rng.normal(0, 0.05, n), 0.2, 0.98)

    held = 3 - (rng.random((n, W)) < 0.10).astype(int)          # 2 or 3 sessions a week
    att = rng.binomial(held, p_att[:, None]).astype(float)
    sub = (rng.random((n, W)) < p_sub[:, None]).astype(float)
    late = np.round(rng.lognormal(0.2, 0.9, (n, W)), 1)         # skewed: a few very late
    score = np.full((n, W), np.nan)
    test_weeks = np.arange(1, W, 2)                             # weeks 2, 4, ..., 12
    score[:, test_weeks] = np.clip(rng.normal(mu_score[:, None], 8, (n, len(test_weeks))), 0, 100)

    status = np.array(["active"] * n, dtype=object)
    label = np.array(["none"] * n, dtype=object)

    # Honest drop-outs (correctly marked "dropped"). These are NOT anomalies.
    dropouts = rng.choice(n, int(0.06 * n), replace=False)
    for i in dropouts:
        dw = rng.integers(4, 10)
        att[i, dw:] = 0
        sub[i, dw:] = 0
        score[i, np.arange(W) >= dw] = np.nan
        status[i] = "dropped"

    # Planted anomalies: 4% of learners.
    pool = np.setdiff1d(np.arange(n), dropouts)
    chosen = rng.choice(pool, 200, replace=False)
    groups = dict(zip(ANOMALY_TYPES, [chosen[:60], chosen[60:110], chosen[110:160], chosen[160:]]))
    for i in groups["active_dropout"]:                          # stopped coming, still "active"
        att[i, 3:] = 0
        sub[i, 3:] = 0
        score[i, 3:] = np.nan
    for i in groups["missing_attendance_block"]:                # sessions held, attendance not recorded
        start, length = rng.integers(2, 6), rng.integers(4, 7)
        att[i, start:start + length] = np.nan
    for i in groups["score_jump"]:                              # weak and absent, then near-perfect scores
        att[i] = rng.binomial(held[i], 0.55)
        score[i, test_weeks] = np.concatenate([rng.normal(42, 6, 3), rng.uniform(90, 100, 3)])
    for kind, idx in groups.items():
        label[idx] = kind

    # Everyday data faults.
    att[rng.random((n, W)) < 0.015] = np.nan                    # sporadic missing attendance
    valid = ~np.isnan(score)
    bad = valid & (rng.random((n, W)) < 0.003)                  # impossible scores
    score[bad] = np.where(rng.random(bad.sum()) < 0.5, rng.integers(105, 999, bad.sum()), -5)
    ok = ~np.isnan(att)
    over = ok & (rng.random((n, W)) < 0.002)                    # attended more than held
    att[over] = held[over] + 1
    late[sub == 0] = np.nan
    neg = ~np.isnan(late) & (rng.random((n, W)) < 0.004)        # negative lateness
    late[neg] = -np.abs(late[neg])

    weekly = pd.DataFrame({
        "learner_id": np.repeat(learners["learner_id"].to_numpy(), W),
        "week": np.tile(np.arange(1, W + 1), n),
        "sessions_held": held.ravel(),
        "sessions_attended": att.ravel(),
        "assignments_submitted": sub.ravel(),
        "days_late": late.ravel(),
        "assessment_score": score.ravel(),
    })

    # Enrolment records, with typing mistakes and duplicates.
    enrol = pd.DataFrame({
        "enrolment_id": [f"E{i:06d}" for i in range(n)],
        "learner_id": learners["learner_id"],
        "batch_id": learners["batch_id"],
        "centre": learners["centre"],
        "course": learners["course"],
        "status": status,
        "enrol_date": pd.Timestamp("2026-07-01") + pd.to_timedelta(rng.integers(0, 45, n), unit="D"),
    })
    for col in ["centre", "course"]:
        typo = rng.random(n) < 0.03
        styles = rng.integers(0, 3, n)
        values = enrol[col].tolist()
        for i in np.where(typo)[0]:
            v = values[i]
            values[i] = [v.lower(), v.upper(), "  " + v + " "][styles[i]]
        enrol[col] = values
    dup_rows = enrol.iloc[groups["duplicate_enrolment"]].copy()
    dup_rows["enrolment_id"] = [f"E9{i:05d}" for i in range(len(dup_rows))]
    dup_rows["enrol_date"] = dup_rows["enrol_date"] + pd.Timedelta(days=1)
    exact = enrol.sample(25, random_state=seed)                  # copy-paste duplicates
    enrol = pd.concat([enrol, dup_rows, exact], ignore_index=True)
    enrol = enrol.sample(frac=1, random_state=seed).reset_index(drop=True)

    labels = pd.DataFrame({
        "learner_id": learners["learner_id"],
        "anomaly_type": label,
        "is_anomaly": (label != "none").astype(int),
    })
    return {"enrolments": enrol, "weekly": weekly, "labels": labels}
