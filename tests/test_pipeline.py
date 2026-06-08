from numcompute_stream.pipeline import Pipeline
from numcompute_stream.pipeline import FeatureUnion
from numcompute_stream.preprocessing import (
    SimpleImputer,
    MinMaxScaler,
    StandardScaler
)
import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_raises_regex, assert_equal


class MockModel:
    """
    A mock model for testing the Pipeline class. It has a fit()
    method that does nothing and a predict() method that returns
    an array of zeros.
    """

    def __init__(self):
        self.fit_called = False
        self.predict_called = False

    def fit(self, X, y=None):
        self.fit_called = True
        return X

    def predict(self, X):
        self.predict_called = True
        return np.array([0] * len(X))

    def reset(self):
        self.fit_called = False
        self.predict_called = False


class MockModel2(MockModel):
    """
    An extended MockModel for testing the fit_predict method is called.
    """
    def __init__(self):
        super().__init__()
        self.fit_predict_called = False

    def fit_predict(self, X, y=None):
        self.fit_predict_called = True
        X = self.fit(X, y)
        return self.predict(X)


class MockScaler:
    """
    A mock scaler for testing the Pipeline class. It has a fit() method that
    does nothing and a transform() method that adds 1 to each element.
    Specifically for testing scalars not belonging to the numcompute library.

    It doesn't implement fit_transform() to ensure that the Pipeline class
    correctly calls fit() and transform() separately.
    """

    def __init__(self):
        self.fit_called = False
        self.transform_called = False

    def fit(self, X, y=None):
        self.fit_called = True

    def transform(self, X):
        self.transform_called = True
        return X + 1


class TestPipeline:

    def test_simple_chaining(self):
        imputer = SimpleImputer(fill_value=5)
        scaler = MinMaxScaler()
        pipeline = Pipeline(steps=[('imputer', imputer), ('scaler', scaler)])
        X = np.array([[0], [float('nan')], [4], [6], [np.nan], [10]])
        expected_X = np.array([[0.0], [0.5], [0.4], [0.6], [0.5], [1.0]])
        transformed_X = pipeline.fit_transform(X)
        assert_array_equal(transformed_X, expected_X)

    def test_predict_without_model(self):
        imputer = SimpleImputer(fill_value=5)
        pipeline = Pipeline(steps=[('imputer', imputer)])
        X = np.array([[0], [float('nan')], [4], [6], [np.nan], [10]])
        assert_raises_regex(
            ValueError,
            r"The final step in the pipeline must be a model with a "
            r"predict\(\) method.",
            pipeline.fit_predict,
            X
        )
        assert_raises_regex(
            ValueError,
            r"The final step in the pipeline must be a model with a "
            r"predict\(\) method.",
            pipeline.predict,
            X
        )

    def test_empty_pipeline(self):
        with pytest.raises(
                ValueError,
                match="Pipeline must have at least one step."
        ):
            Pipeline(steps=[])

    def test_fit_predict(self):
        imputer = SimpleImputer(fill_value=5)
        scaler = MinMaxScaler()
        mock_scaler = MockScaler()
        model = MockModel()
        pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler),
            ('mock_scaler', mock_scaler),
            ('model', model)
        ])
        X = np.array([[0], [float('nan')], [4], [6], [np.nan], [10]])
        expected_y = np.array([0, 0, 0, 0, 0, 0])
        y = pipeline.fit_predict(X)
        assert_array_equal(y, expected_y)
        assert mock_scaler.fit_called is True
        assert mock_scaler.transform_called is True
        assert model.fit_called is True
        assert model.predict_called is True

    def test_fit_predict2(self):
        # Test the fit_predict method is called if it exists on the model.
        model = MockModel2()
        pipeline = Pipeline(steps=[
            ('model', model)
        ])
        X = np.array([[0], [10]])
        expected_y = np.array([0, 0])
        y = pipeline.fit_predict(X)
        assert_array_equal(y, expected_y)
        assert model.fit_called is True
        assert model.predict_called is True
        assert model.fit_predict_called is True

    def test_fit(self):
        imputer = SimpleImputer(fill_value=5)
        scaler = MinMaxScaler()
        model = MockModel()
        pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler),
            ('model', model),
        ])
        X = np.array([
            [0, 20],
            [float('nan'), 30],
            [4, 40],
            [6, 60],
            [np.nan, 100],
            [10, 100],
        ])
        pipeline.fit(X)
        assert_array_equal(scaler._feature_mins, np.array([0.0, 20.0]))
        assert_array_equal(scaler._feature_ranges, np.array([10.0, 80.0]))
        assert model.fit_called is True

    def test_predict_does_not_refit(self):
        imputer = SimpleImputer(fill_value=5)
        scaler = MinMaxScaler()
        model = MockModel()
        pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler),
            ('model', model)
        ])
        X_train = np.array([[0], [float('nan')], [4], [6], [np.nan], [10]])
        X_test = np.array([[11], [float('nan')], [5], [7], [np.nan], [110]])
        pipeline.fit(X_train)
        assert model.fit_called is True
        assert model.predict_called is False
        expected_scalar_mins = np.array([0.0])
        expected_scalar_ranges = np.array([10.0])
        assert_array_equal(scaler._feature_mins, expected_scalar_mins)
        assert_array_equal(scaler._feature_ranges, expected_scalar_ranges)

        model.reset()

        y_test = pipeline.predict(X_test)
        expected_y = np.array([0, 0, 0, 0, 0, 0])
        assert_array_equal(y_test, expected_y)
        # Ensure that the pipeline hasn't been refit on the test data
        assert model.fit_called is False
        assert model.predict_called is True
        assert_array_equal(scaler._feature_mins, expected_scalar_mins)
        assert_array_equal(scaler._feature_ranges, expected_scalar_ranges)

    def test_with_external_steps(self):
        scaler = MockScaler()
        model = MockModel()
        pipeline = Pipeline(steps=[
            ('scaler', scaler),
            ('model', model)
        ])
        X = np.array([[0], [1], [2], [3], [4], [5]])
        transformed_X = pipeline.fit_transform(X)
        assert scaler.fit_called is True
        assert scaler.transform_called is True
        assert model.fit_called is True
        assert model.predict_called is False
        assert_array_equal(transformed_X, X + 1)


class TestFeatureUnion:

    def test_simple_two_transformers(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        X_train = np.array([[1, 2], [3, 4], [5, 6]])

        union = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        union.fit(X_train)
        result = union.transform(X_train)

        assert result.shape == (3, 4)

    def test_fit_transform_consistency(self):
        imputer = SimpleImputer(fill_value=5)
        scaler = MinMaxScaler()

        X = np.array([[1, 2], [3, 4], [5, 6]])

        union1 = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        union1.fit(X)
        result1 = union1.transform(X)

        union2 = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        union2.fit(X)
        result2 = union2.transform(X)

        assert_array_equal(result1, result2)

    def test_does_not_modify_original_data(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        X = np.array([[1, 2], [3, 4]])
        original_X = X.copy()

        union = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        union.fit(X)
        union.transform(X)

        assert_array_equal(X, original_X)

    def test_three_transformers(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()
        standard_scaler = StandardScaler()

        X = np.array([[1, 2], [3, 4], [5, 6]])

        union = FeatureUnion([
            ("imputer", imputer),
            ("scaler", scaler),
            ("standard", standard_scaler)
        ])
        union.fit(X)
        result = union.transform(X)

        assert result.shape == (3, 6)

    def test_train_test_split(self):
        """Test fitting on training data and transforming test data."""
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[2, 3], [4, 5]])

        union = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        union.fit(X_train)
        result_test = union.transform(X_test)

        assert result_test.shape == (2, 4)

    def test_raises_if_transform_before_fit(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        X = np.array([[1, 2], [3, 4]])

        union = FeatureUnion([("imputer", imputer), ("scaler", scaler)])

        with pytest.raises(ValueError):
            union.transform(X)

    def test_with_nan_values(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        X = np.array([[1, 2], [np.nan, 4], [5, 6]])

        union = FeatureUnion([("imputer", imputer), ("scaler", scaler)])
        with pytest.raises(ValueError):
            union.fit(X)

    def test_single_transformer(self):
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])

        union = FeatureUnion([("scaler", scaler)])
        union.fit(X)
        result = union.transform(X)

        assert result.shape == (3, 2)

    def test_feature_names_in_list(self):
        imputer = SimpleImputer(fill_value=0)
        scaler = MinMaxScaler()

        transformers = [("my_imputer", imputer), ("my_scaler", scaler)]
        union = FeatureUnion(transformers)

        assert union.transformers[0][0] == "my_imputer"
        assert union.transformers[1][0] == "my_scaler"

    def test_fit_returns_self(self):
        imputer = SimpleImputer(fill_value=0)
        X = np.array([[1, 2], [3, 4]])

        union = FeatureUnion([("imputer", imputer)])
        result = union.fit(X)

        assert result is union

    def test_output_shape_with_standard_scaler(self):
        imputer = SimpleImputer(fill_value=0)
        standard_scaler = StandardScaler()

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        union = FeatureUnion([
            ("imputer", imputer),
            ("standard", standard_scaler)
        ])
        union.fit(X)
        result = union.transform(X)

        assert result.shape == (3, 4)

        assert not np.isnan(result).any()


# Streaming
class MockStreamingTransformer:
    """
    A mock incremental transformer.

    It records the chunks received by partial_fit() and adds a fixed value
    during transform().
    """

    def __init__(self, add_value):
        self.add_value = add_value
        self.partial_fit_calls = 0
        self.transform_calls = 0
        self.received_chunks = []

    def partial_fit(self, X, y=None):
        self.partial_fit_calls += 1
        self.received_chunks.append(np.asarray(X).copy())
        return self

    def transform(self, X):
        self.transform_calls += 1
        return X + self.add_value


class MockStreamingModel:
    """
    A mock incremental model.

    It records the transformed feature chunks and target chunks received
    during partial_fit().
    """

    def __init__(self):
        self.partial_fit_calls = 0
        self.received_X_chunks = []
        self.received_y_chunks = []

    def partial_fit(self, X, y=None):
        self.partial_fit_calls += 1
        self.received_X_chunks.append(np.asarray(X).copy())

        if y is None:
            self.received_y_chunks.append(None)
        else:
            self.received_y_chunks.append(np.asarray(y).copy())

        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestPipelineStream:

    def test_partial_fit_passes_transformed_data_to_next_step(self):
        transformer_1 = MockStreamingTransformer(add_value=1)
        transformer_2 = MockStreamingTransformer(add_value=10)
        model = MockStreamingModel()

        pipeline = Pipeline([
            ("transformer_1", transformer_1),
            ("transformer_2", transformer_2),
            ("model", model)
        ])

        X = np.array([
            [1],
            [2],
            [3]
        ])

        y = np.array([0, 1, 0])

        result = pipeline.partial_fit(X, y)

        assert_equal(result, pipeline)
        assert_array_equal(transformer_1.received_chunks[0], X)
        assert_array_equal(transformer_2.received_chunks[0], X + 1)
        assert_array_equal(model.received_X_chunks[0], X + 11)
        assert_array_equal(model.received_y_chunks[0], y)

    def test_partial_fit_accumulates_multiple_chunks(self):
        transformer = MockStreamingTransformer(add_value=5)
        model = MockStreamingModel()
        pipeline = Pipeline([
            ("transformer", transformer),
            ("model", model)
        ])

        X_1 = np.array([[1], [2]])
        y_1 = np.array([0, 1])
        X_2 = np.array([[3], [4]])
        y_2 = np.array([1, 0])
        pipeline.partial_fit(X_1, y_1)
        pipeline.partial_fit(X_2, y_2)
        assert_equal(transformer.partial_fit_calls, 2)
        assert_equal(model.partial_fit_calls, 2)
        assert_array_equal(model.received_X_chunks[0], X_1 + 5)
        assert_array_equal(model.received_X_chunks[1], X_2 + 5)

    def test_partial_fit_rejects_step_without_partial_fit(self):
        pipeline = Pipeline([
            ("mock_scaler", MockScaler())
        ])

        with pytest.raises(
            ValueError,
            match="Step 'mock_scaler' does not support partial_fit"
        ):
            pipeline.partial_fit(np.array([[1], [2]]))
