import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from evaluate_models import OrdinalClassifier, log_review_classes, prepare_data, review_classes


class EvaluationTests(unittest.TestCase):
    def test_log_transform_preserves_class_boundaries(self):
        counts = np.array([0, 1, 49, 50, 51, 998, 999, 1000, 1001, 100000])
        np.testing.assert_array_equal(log_review_classes(np.log1p(counts)), review_classes(counts))
        np.testing.assert_array_equal(review_classes(counts), [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    def test_inputs_cannot_depend_on_reviews_or_ratings(self):
        df = pd.DataFrame({
            'AppID': [1, 2], 'Title': ['One', 'Two'], 'Price': [0, 10],
            'Gif_Count': [1, 2], 'Image_Count': [2, 3], 'Bullet_Points': [0, 4],
            'Word_Count': [100, 200], 'Has_Demo': [0, 1], 'Tags': ['Indie', 'RPG'],
            'Review_Count': [0, 1000], 'Positive_Ratio': [0, 0.99],
        })
        X, y = prepare_data(df)
        changed = df.copy()
        changed['Review_Count'] = [10000, 10]
        changed['Positive_Ratio'] = [1, 0]
        other_X, other_y = prepare_data(changed)
        pd.testing.assert_frame_equal(X, other_X)
        self.assertFalse(np.array_equal(y, other_y))
        self.assertTrue({'AppID', 'Review_Count', 'Positive_Ratio'}.isdisjoint(X.columns))
        with self.assertRaisesRegex(ValueError, 'Duplicate AppIDs'):
            prepare_data(pd.concat([df, df]))

    def test_ordinal_estimator_is_cloneable_and_produces_valid_probabilities(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(90, 4))
        y = np.tile([0, 1, 2], 30)
        model = clone(OrdinalClassifier(LogisticRegression(max_iter=1000))).fit(X, y)
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.shape, (90, 3))
        self.assertTrue((probabilities >= 0).all())
        self.assertTrue((probabilities <= 1).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertTrue(set(model.predict(X)).issubset({0, 1, 2}))

    def test_crossing_ordinal_probabilities_cannot_create_negative_class_probability(self):
        class FixedProbability:
            def __init__(self, probability):
                self.probability = probability
            def predict_proba(self, X):
                return np.tile([1 - self.probability, self.probability], (len(X), 1))
        model = OrdinalClassifier(LogisticRegression())
        model.classes_ = np.array([0, 1, 2])
        model.models_ = [FixedProbability(0.2), FixedProbability(0.8)]
        np.testing.assert_allclose(model.predict_proba(np.zeros((1, 4))), [[0.5, 0, 0.5]])


if __name__ == '__main__':
    unittest.main()
