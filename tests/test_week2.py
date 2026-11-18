import unittest

import numpy as np

from src.features import build_features, rolling_deviation
from src.generate_data import CENTRES, COURSES, generate
from src.prepare_data import clean


class Week2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = generate(n_learners=600, seed=7)
        cls.enrol, cls.weekly, cls.log = clean(cls.d["enrolments"], cls.d["weekly"], CENTRES, COURSES)

    def test_text_is_standardised(self):
        self.assertEqual(self.enrol["centre"].nunique(), 6)
        self.assertEqual(self.enrol["course"].nunique(), 5)

    def test_exact_duplicates_dropped_but_duplicate_enrolments_kept_and_flagged(self):
        self.assertFalse(self.enrol.duplicated().any())
        flagged = self.enrol.loc[self.enrol["dup_enrolment"] == 1]
        self.assertGreater(len(flagged), 0)

    def test_impossible_values_removed(self):
        self.assertTrue((self.weekly["assessment_score"].dropna().between(0, 100)).all())
        self.assertTrue((self.weekly["days_late"].dropna() >= 0).all())
        ok = self.weekly.dropna(subset=["sessions_attended"])
        self.assertTrue((ok["sessions_attended"] <= ok["sessions_held"]).all())

    def test_features_have_one_row_per_learner(self):
        feats = build_features(self.enrol, self.weekly)
        self.assertEqual(len(feats), 600)
        self.assertFalse(feats.isna().any().any())

    def test_rolling_baseline_never_uses_the_future(self):
        feats = build_features(self.enrol, self.weekly)
        att = self.weekly.pivot(index="learner_id", columns="week", values="sessions_attended")
        held = self.weekly.pivot(index="learner_id", columns="week", values="sessions_held")
        rate = att / held
        full = rolling_deviation(rate).iloc[:, :6].fillna(-9)
        cut = rolling_deviation(rate.iloc[:, :6]).fillna(-9)
        self.assertTrue(np.allclose(full, cut))


if __name__ == "__main__":
    unittest.main()
