import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises_regex
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler


# ------- Mock models ------- #
class MockStreamingModel:
    def __init__(self):
        self.partial_fit_calls = []
        self.predict_calls = []

        self.predictions = None
        self.is_fitted = False

    def partial_fit(self, X, y):
        self.partial_fit_calls.append({
            "X": np.asarray(X).copy(),
            "y": np.asarray(y).copy()
        })
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Mock model is not fitted yet.")

        self.predict_calls.append(np.asarray(X).copy())
        return np.asarray(self.predictions).copy()


class MockModelWithoutPartialFit:

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class MockModelWithoutPredict:
    def partial_fit(self, X, y):
        return self


class MockRecordingClassifier:

    def __init__(self):
        self.received_X = None
        self.is_fitted = False

    def partial_fit(self, X, y):
        self.received_X = np.asarray(X).copy()
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError(
                "Mock classifier is not fitted yet."
            )

        return np.zeros(len(X), dtype=int)


# ------- Test Streaming ------- #
class TestStreamTrainer:

    def test_fit_chunk_updates_model(self):
        model = MockStreamingModel()
        trainer = StreamTrainer(model=model)

        X = np.array([
            [1],
            [2]
        ])

        y = np.array([
            0,
            1
        ])

        result = trainer.fit_chunk(X, y)

        assert_equal(result, trainer)
        assert_equal(trainer.fit_chunk_count, 1)
        assert_equal(len(model.partial_fit_calls), 1)

        assert_array_equal(
            model.partial_fit_calls[0]["X"],
            X
        )

        assert_array_equal(
            model.partial_fit_calls[0]["y"],
            y
        )

    def test_score_chunk_returns_chunk_accuracy(self):
        model = MockStreamingModel()

        trainer = StreamTrainer(
            model=model
        )

        trainer.fit_chunk(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )

        model.predictions = np.array([
            1,
            0,
            1,
            1
        ])

        chunk_accuracy = trainer.score_chunk(
            np.array([
                [5],
                [6],
                [7],
                [8]
            ]),
            np.array([
                1,
                0,
                0,
                1
            ])
        )

        assert_equal(chunk_accuracy, 0.75)
        assert_equal(trainer.score_chunk_count, 1)

    def test_score_chunk_updates_cumulative_accuracy(self):
        model = MockStreamingModel()

        trainer = StreamTrainer(
            model=model
        )

        trainer.fit_chunk(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )

        # Chunk 1:
        model.predictions = np.array([
            0,
            1
        ])

        trainer.score_chunk(
            np.array([
                [3],
                [4]
            ]),
            np.array([
                0,
                1
            ])
        )

        # Chunk 2:
        model.predictions = np.array([
            1,
            0
        ])

        trainer.score_chunk(
            np.array([
                [5],
                [6]
            ]),
            np.array([
                1,
                1
            ])
        )

        assert_equal(trainer.cumulative_accuracy, 0.75)
        assert_equal(trainer.score_chunk_count, 2)

    def test_score_chunk_creates_log_entry(self):
        model = MockStreamingModel()

        trainer = StreamTrainer(
            model=model
        )

        trainer.fit_chunk(
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )

        model.predictions = np.array([
            1,
            0,
            1,
            1
        ])

        trainer.score_chunk(
            np.array([
                [5],
                [6],
                [7],
                [8]
            ]),
            np.array([
                1,
                0,
                0,
                1
            ])
        )

        logs = trainer.logs

        assert_equal(len(logs), 1)
        assert_equal(logs[0]["chunk"], 1)
        assert_equal(logs[0]["samples"], 4)
        assert_equal(logs[0]["chunk_accuracy"], 0.75)
        assert_equal(logs[0]["cumulative_accuracy"], 0.75)

        assert isinstance(logs[0]["memory_bytes"], int)
        assert isinstance(logs[0]["peak_memory_bytes"], int)
        assert logs[0]["memory_bytes"] >= 0
        assert logs[0]["peak_memory_bytes"] >= 0

    def test_score_before_fit_raises_error(self):
        model = MockStreamingModel()

        trainer = StreamTrainer(
            model=model
        )

        assert_raises_regex(
            ValueError,
            r"Mock model is not fitted yet\.",
            trainer.score_chunk,
            np.array([
                [1],
                [2]
            ]),
            np.array([
                0,
                1
            ])
        )


class TestStreamTrainerPipeline:

    def test_fit_chunk_supports_pipeline(self):
        model = MockRecordingClassifier()

        pipeline = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                model
            )
        ])

        trainer = StreamTrainer(model=pipeline)

        trainer.fit_chunk(
            np.array([
                [1.0],
                [3.0]
            ]),
            np.array([
                0,
                1
            ])
        )

        assert_array_equal(
            model.received_X,
            np.array([
                [-1.0],
                [1.0]
            ])
        )
