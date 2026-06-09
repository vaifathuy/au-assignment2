import numpy as np
from numpy.testing import assert_array_equal, assert_raises_regex, assert_equal
from numcompute_stream.ensemble import EnsembleClassifier


class MockPredictionTree:
    def __init__(self, predictions):
        self.predictions = np.asarray(
            predictions
        )

    def predict(self, X):
        return self.predictions.copy()


class TestEnsembleClassifier:

    def test_n_estimators_controls_number_of_trees(self):
        model = EnsembleClassifier(
            n_estimators=5,
            random_state=42
        )

        assert_equal(len(model.trees), 5)

    def test_partial_fit_updates_count_and_classes(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        model.partial_fit(
            np.array([
                [1],
                [2],
                [8],
                [9]
            ]),
            np.array([
                0,
                0,
                1,
                1
            ])
        )

        assert_equal(model.count, 4)
        assert_equal(model.n_features, 1)

        assert_array_equal(
            model.classes,
            np.array([
                0,
                1
            ])
        )

    def test_partial_fit_accumulates_multiple_chunks(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        model.partial_fit(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                0
            ])
        )

        model.partial_fit(
            np.array([
                [8],
                [9]
            ]),
            np.array([
                1,
                1
            ])
        )

        assert_equal(model.count, 4)
        assert_array_equal(
            model.classes,
            np.array([
                0,
                1
            ])
        )

    def test_predict_uses_majority_voting(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        # Replace the real trees with controlled mock trees.
        model._trees = [
            MockPredictionTree([
                0,
                1,
                1
            ]),
            MockPredictionTree([
                0,
                0,
                1
            ]),
            MockPredictionTree([
                1,
                0,
                1
            ])
        ]

        model._count = 1

        predictions = model.predict(
            np.array([
                [1],
                [2],
                [3]
            ])
        )

        assert_array_equal(
            predictions,
            np.array([
                0,
                0,
                1
            ])
        )

    def test_fit_resets_previous_state(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        model.partial_fit(
            np.array([
                [1],
                [2],
                [8],
                [9]
            ]),
            np.array([
                0,
                0,
                1,
                1
            ])
        )

        assert_equal(model.count, 4)

        model.fit(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                2,
                2
            ])
        )

        assert_equal(model.count, 2)
        assert_array_equal(
            model.classes,
            np.array([
                2
            ])
        )

    def test_reset_clears_statistics_and_recreates_trees(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        model.partial_fit(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )

        previous_trees = model.trees

        model.reset()

        assert_equal(model.count, 0)
        assert_equal(model.n_features, None)
        assert_array_equal(model.classes, np.array([]))
        assert_equal(len(model.trees), 3)
        assert_equal(previous_trees[0] is model.trees[0], False)

    def test_predict_raises_before_fitting(self):
        model = EnsembleClassifier(
            n_estimators=3
        )

        assert_raises_regex(
            ValueError,
            r"The EnsembleClassifier is not fitted yet\.",
            model.predict,
            np.array([
                [1]
            ])
        )

    def test_partial_fit_rejects_different_feature_count(self):
        model = EnsembleClassifier(
            n_estimators=3,
            random_state=42
        )

        model.partial_fit(
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([
                0,
                1
            ])
        )

        assert_raises_regex(
            ValueError,
            r"Input chunk has a different number of features",
            model.partial_fit,
            np.array([
                [1, 2, 3]
            ]),
            np.array([
                0
            ])
        )
