from numcompute_stream.stats import (
    Statistics, Histogram, Quantile,
    histogram, quantile
)
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import numpy as np


class TestStatistics:

    def test_standard_dev(self):
        stat = Statistics()
        stat.add(1)
        stat.add(21)
        stat.add(13)
        stat.add(11)
        stat.add(2)
        stat.add(7)
        assert np.isclose(np.array([stat.std_dev()]),
                          np.array([6.841458583924598]))

    def test_mean(self):
        stat = Statistics()
        stat.add(1)
        stat.add(21)
        stat.add(13)
        stat.add(11)
        stat.add(2)
        stat.add(7)
        assert np.isclose(np.array([stat.mean]), np.array(9.166666666666666))

    def test_min(self):
        stat = Statistics()
        stat.add(2)
        stat.add(7)
        stat.add(7)
        stat.add(45)
        stat.add(6)
        stat.add(21)
        assert stat.min == 2

    def test_max(self):
        stat = Statistics()
        stat.add(5)
        stat.add(8)
        stat.add(3)
        stat.add(5)
        stat.add(2)
        stat.add(6)
        assert stat.max == 8

    def test_median(self):
        stat = Statistics()
        median = stat.median(np.array([23, 45, 1, 2, 67, 87]))
        assert median == 34.0

    def test_median_none_nan_inf(self):
        stat = Statistics()
        with pytest.raises(ValueError):
            stat.median(np.array([23, 45, 1, 2, 67, None, np.nan, np.inf]))

    def test_median_2d(self):
        stat = Statistics()
        median = stat.median(np.array([[23, 45, 1], [2, 67, 87]]))
        assert median == 34.0

    def test_median_axis_0(self):
        stat = Statistics()
        median = stat.median(np.array([[23, 45, 1], [2, 67, 87]]), axis=0)
        assert np.equal(median, np.array([12.5, 56.0, 44.0])).all()

    def test_median_axis_1(self):
        stat = Statistics()
        median = stat.median(np.array([[23, 45, 1], [2, 67, 87]]), axis=1)
        assert np.equal(median, np.array([23.0, 67.0])).all()

    def test_median_keepdims(self):
        stat = Statistics()
        median = stat.median(np.array([[23, 45, 1], [2, 67, 87]]),
                             axis=1, keepdims=True)
        assert median.ndim == 2

    def test_standard_dev_nan(self):
        stat = Statistics()
        with pytest.raises(ValueError):
            stat.add(None)
            stat.add(3)
            stat.add(5)
            stat.add(2)
            stat.add(6)
            stat.add("12")


class TestStatisticsStreaming:
    def test_update_stats_single_chunk(self):
        stat = Statistics()

        X = np.array([
            [2, 10],
            [6, 20],
            [10, 30]
        ])

        stat.update_stats(X)
        assert_equal(stat.count, 3)

        assert_allclose(
            stat.mean,
            np.array([6.0, 20.0])
        )

        assert_allclose(
            stat._M2,
            np.array([32.0, 200.0])
        )

        assert_array_equal(
            stat.min,
            np.array([2.0, 10.0])
        )

        assert_array_equal(
            stat.max,
            np.array([10.0, 30.0])
        )

    def test_update_stats_multiple_chunks(self):
        stat = Statistics()
        X_1 = np.array([
            [2, 10],
            [6, 20]
        ])
        X_2 = np.array([
            [10, 30]
        ])
        stat.update_stats(X_1)
        stat.update_stats(X_2)

        assert stat.count == 3

        assert_allclose(
            stat.mean,
            np.array([6.0, 20.0])
        )

        assert_allclose(
            stat._M2,
            np.array([32.0, 200.0])
        )

        assert_array_equal(
            stat.min,
            np.array([2.0, 10.0])
        )

        assert_array_equal(
            stat.max,
            np.array([10.0, 30.0])
        )

    def test_population_variance(self):
        stat = Statistics()

        stat.update_stats(np.array([
            [2, 10],
            [6, 20],
            [10, 30]
        ]))

        assert_allclose(
            stat.variance(ddof=0),
            np.array([10.6666666667, 66.6666666667])
        )

    def test_sample_variance(self):
        stat = Statistics()

        stat.update_stats(np.array([
            [2, 10],
            [6, 20],
            [10, 30]
        ]))

        assert_allclose(
            stat.variance(ddof=1),
            np.array([16.0, 100.0])
        )

    def test_population_standard_deviation(self):
        stat = Statistics()

        X = np.array([
            [2, 10],
            [6, 20],
            [10, 30]
        ])

        stat.update_stats(X)

        assert_allclose(
            stat.std_dev(ddof=0),
            np.std(X, axis=0, ddof=0)
        )

    def test_sample_standard_deviation(self):
        stat = Statistics()
        X = np.array([
            [2, 10],
            [6, 20],
            [10, 30]
        ])
        stat.update_stats(X)
        assert_allclose(
            stat.std_dev(ddof=1),
            np.std(X, axis=0, ddof=1)
        )


class TestHistogram:

    def test_histogram(self):
        hist, edges = histogram(np.array([1, 2, 1, 3, 4, 5, 1]))
        assert np.array_equal(hist, np.array([3, 0, 1, 0, 0, 1, 0, 1, 0, 1]))
        assert np.allclose(edges, np.array([1.0, 1.4, 1.8, 2.2, 2.6,
                                            3.0, 3.4, 3.8, 4.2, 4.6, 5.0]))

    def test_histogram_a_none(self):
        with pytest.raises(ValueError):
            histogram(np.array([1, 2, 5, 7, np.nan, np.inf, 9, 23]))

    def test_histogram_a_2d_none(self):
        with pytest.raises(ValueError):
            histogram(np.array([[1, 2, 5, 7], [np.nan, np.inf, 9, 23]]))

    def test_histogram_bins_value(self):
        hist, edges = histogram(np.array([10, 15, 10, 50, 55]), bins=12)
        assert np.array_equal(hist,
                              np.array([2, 1, 0, 0, 0,
                                        0, 0, 0, 0, 0, 1, 1]))
        assert np.allclose(edges,
                           np.array([10.0, 13.75, 17.5, 21.25, 25.0,
                                     28.75, 32.5, 36.25, 40.0, 43.75, 47.5,
                                     51.25, 55.0]))

    def test_histogram_bins_array(self):
        hist, edges = histogram(np.array([1, 1, 2, 5, 8, 10,
                                          12, 15,
                                          18, 20]), bins=[0, 5, 15, 20])
        assert np.array_equal(hist, np.array([3, 4, 3]))
        assert np.array_equal(edges, np.array([0, 5, 15, 20]))

    def test_histogram_range(self):
        hist, edges = histogram(np.array([85, 82, 88, 90, 78, 92, 999]),
                                range=(0, 100))
        assert np.array_equal(hist, np.array([0, 0, 0, 0, 0, 0, 0, 1, 3, 2]))
        assert np.allclose(edges, np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                                            60.0, 70.0, 80.0, 90.0, 100.0]))

    def test_histogram_density_true(self):
        hist, edges = histogram(np.array([10, 15, 10, 50, 55]), density=True,
                                bins=2)
        assert np.allclose(hist, np.array([0.026666666666666665,
                                           0.017777777777777778]))
        assert np.allclose(edges, np.array([10.0, 32.5, 55.0]))

    def test_histogram_density_false(self):
        hist, edges = histogram(np.array([10, 15, 10, 50, 55]), density=False,
                                bins=2)
        assert np.array_equal(hist, np.array([3, 2]))
        assert np.allclose(edges, np.array([10.0, 32.5, 55.0]))

    def test_histogram_weights(self):
        hist, edges = histogram(
            np.array([50, 90, 50]),
            bins=np.array([0, 60, 100]),
            weights=np.array([10, 5, 2])
        )
        assert np.array_equal(hist, np.array([12, 5]))
        assert np.allclose(edges, np.array([0, 60, 100]))


class TestQuantile:

    def test_quantile(self):
        q = quantile(np.array([10, 20, 30, 40]), np.array([0.5]))
        assert q == 25

    def test_quantile_axis_0(self):
        q = quantile(
            np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            np.array([0.25]),
            axis=0
        )
        assert np.allclose(q, np.array([20.0, 30.0, 40.0, 50.0]))

    def test_quantile_axis_1(self):
        q = quantile(
            np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            np.array([0.25]),
            axis=1
        )
        assert np.allclose(q, np.array([17.5, 57.5]))

    def test_quantile_out(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), np.array([0.25]), out=np.zeros(1)
        )
        assert q == 20

    def test_quantile_overwrite_input_true(self):
        orig = np.array([50, 10, 30, 20, 40])
        dup = orig.copy()
        quantile(
            orig, np.array([0.5]),
            overwrite_input=True)
        assert not np.array_equal(orig, dup)

    def test_quantile_overwrite_input_false(self):
        orig = np.array([50, 10, 30, 20, 40])
        dup = orig.copy()
        quantile(
            orig, np.array([0.5]),
            overwrite_input=False)
        assert np.array_equal(orig, dup)

    def test_quantile_method_linear(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3, method="linear")
        assert q == 22

    def test_quantile_method_lower(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3, method="lower")
        assert q == 20

    def test_quantile_method_higher(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3, method="higher")
        assert q == 30

    def test_quantile_method_midpoint(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3, method="midpoint")
        assert q == 25.0

    def test_quantile_method_nearest(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3, method="nearest")
        assert q == 20

    def test_quantile_method_inverted_cdf(self):
        q = quantile(np.array([10, 20, 30, 40, 50]), 0.3,
                     method="inverted_cdf")
        assert q == 20

    def test_quantile_method_averaged_inverted_cdf(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="averaged_inverted_cdf"
        )
        assert q == 20

    def test_quantile_method_closest_observation(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="closest_observation"
        )
        assert q == 20

    def test_quantile_method_interpolated_inverted_cdf(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="interpolated_inverted_cdf"
        )
        assert q == 15

    def test_quantile_method_hazen(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="hazen"
        )
        assert q == 20

    def test_quantile_method_weibull(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="weibull"
        )
        assert q == 18

    def test_quantile_method_median_unbiased(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="median_unbiased"
        )
        assert q == pytest.approx(19.333333333333332)

    def test_quantile_method_normal_unbiased(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.3,
            method="normal_unbiased"
        )
        assert q == pytest.approx(19.5)

    def test_quantile_keepdims_true(self):
        q = quantile(
            np.array([[10, 20, 30], [40, 50, 60]]), 0.5, axis=1, keepdims=True
        )
        assert q.shape == (2, 1)

    def test_quantile_weights(self):
        q = quantile(
            np.array([10, 20, 30, 40, 50]), 0.5,
            weights=np.array([10, 1, 1, 1, 1]),
            method="inverted_cdf"
        )
        assert q == 10


class TestHistogramStream:
    def test_explicit_edges_initial_state(self):
        hist = Histogram(
            bins=np.array([0, 10, 20, 30])
        )

        assert_array_equal(
            hist.bin_edges,
            np.array([0.0, 10.0, 20.0, 30.0])
        )

        assert_array_equal(
            hist.counts,
            np.array([0, 0, 0])
        )

        assert_equal(hist.count, 0)

    def test_update_stats_single_chunk(self):
        hist = Histogram(
            bins=np.array([0, 10, 20, 30])
        )
        hist.update_stats(np.array([2, 7, 15, 25]))
        assert_array_equal(hist.counts, [2, 1, 1])
        assert_equal(hist.count, 4)

    def test_update_stats_accumulates_multiple_chunks(self):
        hist = Histogram(
            bins=np.array([0, 10, 20, 30])
        )

        hist.update_stats(np.array([2, 7, 15]))
        assert_array_equal(hist.counts, [2, 1, 0])

        hist.update_stats(np.array([12, 25]))
        assert_array_equal(hist.counts, [2, 2, 1])

        assert_equal(hist.count, 5)

    def test_integer_bins_with_range(self):
        hist = Histogram(bins=3, range=(0, 30))

        hist.update_stats(np.array([2, 7, 15, 12, 25]))
        assert_array_equal(hist.bin_edges, [0.0, 10.0, 20.0, 30.0])
        assert_array_equal(hist.counts, [2, 2, 1])

    def test_weighted_chunks(self):
        hist = Histogram(bins=np.array([0, 10, 20, 30]))

        hist.update_stats(
            np.array([2, 7, 15]),
            weights=np.array([0.5, 1.5, 2.0])
        )

        hist.update_stats(
            np.array([12, 25]),
            weights=np.array([1.0, 4.0])
        )

        assert_allclose(
            hist.counts,
            np.array([2.0, 3.0, 4.0])
        )

    def test_bin_edges_property_returns_copy(self):
        hist = Histogram(
            bins=np.array([0, 10, 20, 30])
        )

        external_edges = hist.bin_edges
        external_edges[0] = -999

        assert_array_equal(
            hist.bin_edges,
            np.array([0.0, 10.0, 20.0, 30.0])
        )

    def test_rejects_non_numeric_explicit_edges(self):
        with pytest.raises(TypeError):
            Histogram(
                bins=np.array([
                    "low",
                    "medium",
                    "high"
                ])
            )

    def test_rejects_invalid_density_result_argument(self):
        hist = Histogram(bins=np.array([0, 10, 20]))
        hist.update_stats(np.array([2, 7, 15]))
        with pytest.raises(
            TypeError,
            match="density must be a boolean or None"
        ):
            hist.result(density="yes")


class TestQuantileStream:
    def test_update_stats_accumulates_chunks(self):
        stream = Quantile()
        stream.update_stats(np.array([50, 10, 30]))
        stream.update_stats(np.array([20, 40]))
        assert_equal(stream.count, 5)
        assert_array_equal(
            stream.values,
            np.array([
                50.0,
                10.0,
                30.0,
                20.0,
                40.0
            ])
        )

    def test_result_median(self):
        stream = Quantile()
        stream.update_stats(np.array([50, 10, 30]))
        stream.update_stats(np.array([20, 40]))
        assert_allclose(stream.result(q=0.5), 30.0)

    def test_result_accepts_random_quantile(self):
        stream = Quantile()
        stream.update_stats(np.array([10, 20, 30, 40, 50]))
        assert_allclose(stream.result(q=0.18), 17.2)

    def test_result_accepts_multiple_quantiles(self):
        stream = Quantile()
        stream.update_stats(np.array([50, 10, 30, 20, 40]))
        assert_array_equal(
            stream.result(q=np.array([0.25, 0.50, 0.75])),
            np.array([20.0, 30.0, 40.0])
        )

    def test_reset_clears_previous_observations(self):
        stream = Quantile()
        stream.update_stats(np.array([10, 20, 30]))
        stream.reset()
        assert_equal(stream.count, 0)
        assert_array_equal(stream.values, np.array([], dtype=float))

    def test_update_stats_copies_source_array(self):
        stream = Quantile()
        source_values = np.array([10, 20, 30])
        stream.update_stats(source_values)
        source_values[0] = 999

        assert_array_equal(
            stream.values,
            np.array([10.0, 20.0, 30.0])
        )

    def test_result_raises_before_receiving_observations(self):
        stream = Quantile()

        with pytest.raises(
            ValueError,
            match="No observations have been accumulated yet"
        ):
            stream.result(q=0.5)

    def test_update_stats_rejects_empty_chunk(self):
        stream = Quantile()

        with pytest.raises(
            ValueError,
            match="Input chunk must contain at least one value"
        ):
            stream.update_stats(np.array([]))

    def test_result_rejects_empty_q_array(self):
        stream = Quantile()
        stream.update_stats(np.array([10, 20, 30]))

        with pytest.raises(
            ValueError,
            match="q must contain at least one value"
        ):
            stream.result(q=np.array([]))

    def test_update_stats_rejects_non_numeric_values(self):
        stream = Quantile()

        with pytest.raises(TypeError):
            stream.update_stats(
                np.array([
                    "low",
                    "medium",
                    "high"
                ])
            )

    def test_result_rejects_unsupported_method(self):
        stream = Quantile()

        stream.update_stats(
            np.array([
                10,
                20,
                30
            ])
        )

        with pytest.raises(ValueError):
            stream.result(
                q=0.5,
                method="unsupported"
            )
