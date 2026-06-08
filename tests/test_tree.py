import numpy as np
from numpy.testing import assert_array_equal, assert_raises_regex, assert_equal

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTreeClassifier:

    def test_fit_and_predict(self):
        tree = DecisionTreeClassifier(max_depth=2)
        X = np.array([
            [2],
            [3],
            [8],
            [9]
        ])
        y = np.array([0, 0, 1, 1])
        tree.fit(X, y)
        predictions = tree.predict(
            np.array([
                [2],
                [3],
                [8],
                [9]
            ])
        )
        assert_array_equal(predictions, np.array([0, 0, 1, 1]))

    def test_gini_split_separates_classes(self):
        tree = DecisionTreeClassifier(
            max_depth=1
        )

        tree.fit(
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

        assert_array_equal(
            tree.predict(
                np.array([
                    [1],
                    [2],
                    [8],
                    [9]
                ])
            ),
            np.array([
                0,
                0,
                1,
                1
            ])
        )

    def test_majority_class_is_used_when_tree_cannot_split(self):
        tree = DecisionTreeClassifier(
            max_depth=0
        )

        tree.fit(
            np.array([
                [1],
                [2],
                [8],
                [9],
                [10]
            ]),
            np.array([
                0,
                0,
                1,
                1,
                1
            ])
        )

        assert_array_equal(
            tree.predict(
                np.array([
                    [1],
                    [5],
                    [10]
                ])
            ),
            np.array([
                1,
                1,
                1
            ])
        )

    def test_partial_fit_grows_tree_across_chunks(self):
        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_split=4
        )

        tree.partial_fit(
            np.array([
                [2],
                [3]
            ]),
            np.array([
                0,
                0
            ])
        )

        assert_array_equal(
            tree.predict(
                np.array([
                    [2],
                    [9]
                ])
            ),
            np.array([
                0,
                0
            ])
        )

        tree.partial_fit(
            np.array([
                [8],
                [9]
            ]),
            np.array([
                1,
                1
            ])
        )

        assert_array_equal(
            tree.predict(
                np.array([
                    [2],
                    [9]
                ])
            ),
            np.array([
                0,
                1
            ])
        )

    def test_fit_resets_previous_tree(self):
        tree = DecisionTreeClassifier(max_depth=2)

        tree.fit(
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

        assert_array_equal(
            tree.predict(
                np.array([
                    [1],
                    [9]
                ])
            ),
            np.array([
                0,
                1
            ])
        )

        tree.fit(
            np.array([
                [1],
                [2],
                [8],
                [9]
            ]),
            np.array([
                2,
                2,
                2,
                2
            ])
        )

        assert_array_equal(
            tree.predict(
                np.array([
                    [1],
                    [9]
                ])
            ),
            np.array([
                2,
                2
            ])
        )

    def test_min_samples_split_prevents_early_split(self):
        tree = DecisionTreeClassifier(
            min_samples_split=5
        )

        tree.partial_fit(
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
        assert_array_equal(
            tree.predict(
                np.array([
                    [1],
                    [9]
                ])
            ),
            np.array([
                0,
                0
            ])
        )

    def test_classes_are_recorded_across_chunks(self):
        tree = DecisionTreeClassifier()

        tree.partial_fit(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                "cat",
                "dog"
            ])
        )

        tree.partial_fit(
            np.array([
                [3]
            ]),
            np.array([
                "bird"
            ])
        )

        assert_array_equal(
            tree.classes,
            np.array([
                "cat",
                "dog",
                "bird"
            ])
        )

    def test_reset_clears_tree(self):
        tree = DecisionTreeClassifier()

        tree.fit(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )

        tree.reset()
        assert_equal(tree.n_features, None)
        assert_array_equal(tree.classes, np.array([]))

        assert_raises_regex(
            ValueError,
            r"The DecisionTreeClassifier is not fitted yet\.",
            tree.predict,
            np.array([
                [1]
            ])
        )

    def test_predict_raises_before_fitting(self):
        tree = DecisionTreeClassifier()

        assert_raises_regex(
            ValueError,
            r"The DecisionTreeClassifier is not fitted yet\.",
            tree.predict,
            np.array([[1]])
        )

    def test_partial_fit_rejects_different_feature_count(self):
        tree = DecisionTreeClassifier()
        tree.partial_fit(
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([0, 1])
        )

        assert_raises_regex(
            ValueError,
            r"Input chunk has a different number of features",
            tree.partial_fit,
            np.array([
                [1, 2, 3]
            ]),
            np.array([0])
        )

    def test_fit_rejects_max_features_larger_than_input_width(self):
        tree = DecisionTreeClassifier(max_features=3)

        assert_raises_regex(
            ValueError,
            r"max_features must not exceed the number of input features\.",
            tree.fit,
            np.array([
                [1, 2],
                [3, 4]
            ]),
            np.array([0, 1])
        )

    def test_fit_rejects_mismatched_X_and_y_rows(self):
        tree = DecisionTreeClassifier()

        assert_raises_regex(
            ValueError,
            r"X and y must contain the same number of rows\.",
            tree.fit,
            np.array([
                [1],
                [2],
                [3]
            ]),
            np.array([0, 1])
        )
