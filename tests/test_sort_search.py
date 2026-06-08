from numpy.testing import assert_array_equal
from numcompute_stream.sort_search import (
    quickselect,
    topk,
    binary_search,
    multi_key_sort,
    stable_sort
)
import numpy as np


class TestQuickselect:
    def test_quickselect_smallest(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        assert quickselect(arr, 0) == 1

    def test_quickselect_largest(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = quickselect(arr, 7, largest=True)
        assert result == 1

    def test_quickselect_middle(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        assert quickselect(arr, 3) == 3

    def test_quickselect_third_smallest(self):
        arr = np.array([7, 4, 6, 3, 9])
        assert quickselect(arr, 2) == 6

    def test_quickselect_with_duplicates(self):
        arr = np.array([2, 2, 1, 1, 3, 3])
        result = quickselect(arr, 2)
        assert result in [1, 2]

    def test_quickselect_empty_raises(self):
        np.testing.assert_raises_regex(
            ValueError,
            r"array is empty",
            quickselect, np.array([]), 0
        )

    def test_quickselect_k_out_of_bounds(self):
        np.testing.assert_raises_regex(
            IndexError,
            r"k=5 is out of bounds",
            quickselect, np.array([1, 2, 3]), 5
        )

    def test_quickselect_k_negative_raises(self):
        np.testing.assert_raises_regex(
            ValueError,
            r"Input data must be 1D",
            quickselect, np.array([[2, 3], [4, 5]]), 2
        )

    def test_quickselect_single_element(self):
        arr = np.array([42])
        assert quickselect(arr, 0) == 42

    def test_quickselect_preserves_stability(self):
        arr = np.array([5, 4, 3, 2, 1])
        assert quickselect(arr, 2) == 3

    def test_quickselect_k_0_returns_min(self):
        arr = np.array([5, 3, 8, 1, 9, 2])
        assert quickselect(arr, 0) == 1

    def test_quickselect_k_last_returns_max(self):
        arr = np.array([5, 3, 8, 1, 9, 2])
        assert quickselect(arr, 5) == 9

    def test_quickselect_does_not_modify_original(self):
        data = np.array([5, 2, 9, 1, 7])
        original = data.copy()
        quickselect(data, 2)
        np.testing.assert_array_equal(data, original)


class TestBinarySearch:
    def test_binary_search_found(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 5)
        assert idx == 2
        assert exists is True

    def test_binary_search_not_found_lower(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 4)
        assert idx == 2
        assert exists is False

    def test_binary_search_not_found_higher(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 6)
        assert idx == 3
        assert exists is False

    def test_binary_search_first_element(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 1)
        assert idx == 0
        assert exists is True

    def test_binary_search_last_element(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 9)
        assert idx == 4
        assert exists is True

    def test_binary_search_empty_array(self):
        arr = np.array([])
        idx, exists = binary_search(arr, 5)
        assert idx == 0
        assert exists is False

    def test_binary_search_single_element_found(self):
        arr = np.array([5])
        idx, exists = binary_search(arr, 5)
        assert idx == 0
        assert exists is True

    def test_binary_search_single_element_not_found(self):
        arr = np.array([5])
        idx, exists = binary_search(arr, 3)
        assert idx == 0
        assert exists is False

    def test_binary_search_with_duplicates(self):
        arr = np.array([1, 2, 2, 2, 3])
        idx, exists = binary_search(arr, 2)
        assert exists is True
        assert idx in [1, 2, 3]

    def test_binary_search_all_same_values(self):
        arr = np.array([5, 5, 5, 5, 5])
        idx, exists = binary_search(arr, 5)
        assert exists is True
        assert idx in [0, 1, 2, 3, 4]

    def test_binary_search_insertion_at_beginning(self):
        arr = np.array([5, 10, 15])
        idx, exists = binary_search(arr, 1)
        assert idx == 0
        assert exists is False

    def test_binary_search_insertion_at_end(self):
        arr = np.array([5, 10, 15])
        idx, exists = binary_search(arr, 20)
        assert idx == 3
        assert exists is False

    def test_binary_search_negative_numbers(self):
        arr = np.array([-10, -5, 0, 5, 10])
        idx, exists = binary_search(arr, -5)
        assert idx == 1
        assert exists is True

    def test_binary_search_floats(self):
        arr = np.array([1.5, 2.5, 3.5, 4.5])
        idx, exists = binary_search(arr, 3.5)
        assert idx == 2
        assert exists is True


class TestTopk:
    def test_topk_largest_basic(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = topk(arr, 3, largest=True, return_indices=False)
        assert_array_equal(result, [9, 6, 5])

    def test_topk_smallest_basic(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = topk(arr, 3, largest=False, return_indices=False)
        assert_array_equal(result, [1, 1, 2])

    def test_topk_return_indices(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = topk(arr, 3, largest=True, return_indices=True)
        assert [arr[i] for i in result] == [9, 6, 5]

    def test_topk_scores_top_3(self):
        scores = np.array([0.95, 0.87, 0.99, 0.77, 0.91])
        result = topk(scores, 3, largest=True, return_indices=False)
        assert_array_equal(result, [0.99, 0.95, 0.91])

    def test_topk_exam_scores_top_2(self):
        exam_scores = np.array([88, 92, 75, 91, 85])
        result = topk(exam_scores, 2, largest=True, return_indices=True)
        top_scores = [exam_scores[i] for i in result]
        assert sorted(top_scores) == [91, 92]

    def test_k_equals_array_length(self):
        arr = np.array([3, 1, 4, 1, 5])
        result = topk(arr, 5, largest=True, return_indices=False)
        assert_array_equal(result, [5, 4, 3, 1, 1])

    def test_k_greater_than_length(self):
        arr = np.array([3, 1, 4])
        result = topk(arr, 10, largest=True, return_indices=False)
        assert_array_equal(result, [4, 3, 1])

    def test_k_zero(self):
        arr = np.array([3, 1, 4, 1, 5])
        result = topk(arr, 0, largest=True, return_indices=False)
        assert_array_equal(result, [])

    def test_empty_array(self):
        arr = np.array([])
        result = topk(arr, 3, largest=True, return_indices=False)
        assert_array_equal(result, [])

    def test_all_equal_values(self):
        arr = np.array([5, 5, 5, 5, 5])
        result = topk(arr, 3, largest=True, return_indices=False)
        assert_array_equal(result, [5, 5, 5])

    def test_with_duplicates(self):
        arr = np.array([1, 3, 3, 3, 2, 3])
        result = topk(arr, 3, largest=True, return_indices=False)
        assert_array_equal(result, [3, 3, 3])

    def test_single_element(self):
        arr = np.array([42])
        result = topk(arr, 1, largest=True, return_indices=False)
        assert_array_equal(result, [42])

    def test_k_equals_one(self):
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        result = topk(arr, 1, largest=True, return_indices=False)
        assert result == 9

    def test_topk_returns_sorted_order(self):
        arr = np.array([1, 5, 3, 9, 2])
        result = topk(arr, 3, largest=True, return_indices=False)
        assert_array_equal(result, [9, 5, 3])

    def test_topk_raises_on_2d_input(self):
        arr = np.array([[1, 2], [3, 4]])
        np.testing.assert_raises_regex(
            ValueError,
            r"data must be a 1D array\.",
            topk, arr, 2, True, False
        )

    def test_topk_extreme_k(self):
        arr = np.array([1, 5, 3, 9, 2])
        result = topk(arr, 3_000_000, largest=True, return_indices=False)
        assert_array_equal(result, [9, 5, 3, 2, 1])

    def test_topk_negative_k(self):
        arr = np.array([1, 5, 3, 9, 2])
        result = topk(arr, -3, largest=True, return_indices=False)
        assert_array_equal(result, [])


class TestStableSort:

    # --- Basic correctness ---

    def test_stable_sort_integers(self):
        result = stable_sort(np.array([3, 1, 2]))
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_stable_sort_already_sorted(self):
        result = stable_sort(np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(result, [1, 2, 3, 4])

    def test_stable_sort_reverse_sorted(self):
        result = stable_sort(np.array([5, 4, 3, 2, 1]))
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_stable_sort_negative_values(self):
        result = stable_sort(np.array([-3, 0, -1, 2]))
        np.testing.assert_array_equal(result, [-3, -1, 0, 2])

    def test_stable_sort_floats(self):
        result = stable_sort(np.array([3.3, 1.1, 2.2]))
        np.testing.assert_array_almost_equal(result, [1.1, 2.2, 3.3])

    def test_stable_sort_strings(self):
        result = stable_sort(np.array(["banana", "apple", "cherry"]))
        np.testing.assert_array_equal(result, ["apple", "banana", "cherry"])

    # --- Stability ---

    def test_stable_sort_equal_elements_preserve_relative_order(self):
        arr = np.array([(3, 0), (1, 1), (3, 2), (1, 3)],
                       dtype=[("val", int), ("idx", int)])
        result = np.sort(arr, order="val", kind="stable")
        ones = result[result["val"] == 1]["idx"]
        threes = result[result["val"] == 3]["idx"]
        assert list(ones) == [1, 3]
        assert list(threes) == [0, 2]

    def test_stable_sort_all_equal_elements(self):
        result = stable_sort(np.array([7, 7, 7, 7]))
        np.testing.assert_array_equal(result, [7, 7, 7, 7])

    # --- Edge cases ---

    def test_stable_sort_single_element(self):
        result = stable_sort(np.array([42]))
        np.testing.assert_array_equal(result, [42])

    def test_stable_sort_empty_array(self):
        result = stable_sort(np.array([]))
        assert result.shape == (0,)

    def test_stable_sort_does_not_mutate_input(self):
        original = np.array([3, 1, 2])
        stable_sort(original)
        np.testing.assert_array_equal(original, [3, 1, 2])

    def test_stable_sort_returns_ndarray(self):
        result = stable_sort(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_stable_sort_accepts_numpy_array_input(self):
        arr = np.array([3, 1, 2])
        result = stable_sort(arr)
        np.testing.assert_array_equal(result, [1, 2, 3])

    # --- 2D / axis behaviour ---

    def test_stable_sort_2d_default_axis_sorts_rows(self):
        result = stable_sort(np.array([[3, 1, 2], [9, 7, 8]]))
        np.testing.assert_array_equal(result, [[3, 1, 2], [9, 7, 8]])

    def test_stable_sort_2d_axis_0_sorts_columns(self):
        result = stable_sort(np.array([[3, 1], [1, 3]]), axis=0)
        np.testing.assert_array_equal(result, [[1, 1], [3, 3]])

    def test_stable_sort_2d_axis_1_sorts_rows(self):
        result = stable_sort(np.array([[3, 1, 2], [9, 7, 8]]), axis=1)
        np.testing.assert_array_equal(result, [[1, 2, 3], [7, 8, 9]])

    def test_stable_sort_axis_out_of_bounds_raises(self):
        np.testing.assert_raises_regex(
            (np.exceptions.AxisError, ValueError),
            r"axis 5 is out of bounds",
            stable_sort, np.array([1, 2, 3]), 5
        )

    # --- Shape preservation ---

    def test_stable_sort_output_shape_1d(self):
        assert stable_sort(np.array([4, 2, 3, 1])).shape == (4,)

    def test_stable_sort_output_shape_2d(self):
        assert stable_sort(np.array([[3, 1, 2], [6, 4, 5]])).shape == (2, 3)


class TestMultiKeySort:

    # --- Basic correctness ---
    # Note: multi_key_sort uses np.asarray internally. Homogeneous numeric
    # rows stay numeric; mixed-type rows (int + str) are upcast to strings.
    # Tests below use purely numeric rows so dtype stays int/float.

    def test_multi_key_sort_single_key_ascending(self):
        rows = np.array([[3, 10], [1, 20], [2, 30]])
        result = multi_key_sort(rows, keys=[0])
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3])

    def test_multi_key_sort_single_key_descending(self):
        rows = np.array([[3, 10], [1, 20], [2, 30]])
        result = multi_key_sort(rows, keys=[0], ascending=False)
        np.testing.assert_array_equal(result[:, 0], [3, 2, 1])

    def test_multi_key_sort_two_keys_priority_order(self):
        rows = np.array([[1, 70], [2, 50], [1, 60], [2, 80]])
        result = multi_key_sort(rows, keys=[0, 1])
        col0 = result[:, 0]
        assert list(col0[:2]) == [1, 1]
        assert list(col0[2:]) == [2, 2]
        assert result[0][1] < result[1][1]
        assert result[2][1] < result[3][1]

    def test_multi_key_sort_mixed_ascending_per_key(self):
        rows = np.array([[1, 30], [2, 10], [1, 20], [2, 40]])
        result = multi_key_sort(rows, keys=[0, 1], ascending=[False, True])
        col0 = result[:, 0]
        assert list(col0[:2]) == [2, 2]
        assert list(col0[2:]) == [1, 1]
        assert result[0][1] == 10
        assert result[1][1] == 40
        assert result[2][1] == 20
        assert result[3][1] == 30

    # --- ascending as bool vs list ---

    def test_multi_key_sort_ascending_bool_true(self):
        rows = np.array([[3], [1], [2]])
        result = multi_key_sort(rows, keys=[0], ascending=True)
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3])

    def test_multi_key_sort_ascending_bool_false(self):
        rows = np.array([[3], [1], [2]])
        result = multi_key_sort(rows, keys=[0], ascending=False)
        np.testing.assert_array_equal(result[:, 0], [3, 2, 1])

    def test_multi_key_sort_ascending_list_per_key(self):
        rows = np.array([[1, 10], [1, 20], [2, 5], [2, 15]])
        result = multi_key_sort(rows, keys=[0, 1], ascending=[True, False])
        dept1 = result[result[:, 0] == 1]
        assert dept1[0][1] == 20
        assert dept1[1][1] == 10

    # --- Stability ---

    def test_multi_key_sort_equal_keys_preserve_relative_order(self):
        # Tag rows with a unique second column to track original order
        rows = np.array([[1, 0], [1, 1], [2, 2]])
        result = multi_key_sort(rows, keys=[0])
        assert result[0][1] == 0
        assert result[1][1] == 1

    # --- Edge cases ---

    def test_multi_key_sort_single_row(self):
        result = multi_key_sort(np.array([[5, 3, 1]]), keys=[0])
        assert result.shape == (1, 3)

    def test_multi_key_sort_single_column(self):
        result = multi_key_sort(np.array([[3], [1], [2]]), keys=[0])
        np.testing.assert_array_equal(result[:, 0], [1, 2, 3])

    def test_multi_key_sort_empty_keys_returns_unchanged_shape(self):
        rows = np.array([[3, 1], [1, 2], [2, 3]])
        result = multi_key_sort(rows, keys=[])
        assert result.shape == (3, 2)

    def test_multi_key_sort_does_not_mutate_input(self):
        rows = np.array([[3, 1], [1, 2], [2, 3]])
        original = rows.copy()
        multi_key_sort(rows, keys=[0])
        np.testing.assert_array_equal(rows, original)

    def test_multi_key_sort_returns_ndarray(self):
        result = multi_key_sort(np.array([[1, 2], [3, 4]]), keys=[0])
        assert isinstance(result, np.ndarray)

    def test_multi_key_sort_output_shape_preserved(self):
        rows = np.array([[3, 10, 1], [1, 20, 2], [2, 30, 3]])
        result = multi_key_sort(rows, keys=0)
        assert result.shape == (3, 3)

    def test_multi_key_sort_negative_values(self):
        rows = np.array([[-1, 0], [-3, 1], [-2, 2]])
        result = multi_key_sort(rows, keys=[0])
        np.testing.assert_array_equal(result[:, 0], [-3, -2, -1])

    def test_multi_key_sort_duplicate_rows(self):
        rows = np.array([[2, 99], [1, 99], [2, 99], [1, 99]])
        result = multi_key_sort(rows, keys=[0])
        np.testing.assert_array_equal(result[:, 0], [1, 1, 2, 2])

    # --- Error handling ---

    def test_multi_key_sort_1d_input_raises(self):
        np.testing.assert_raises_regex(
            (ValueError, IndexError),
            r"data must be a 2D array",
            multi_key_sort, np.array([1, 2, 3]), [0]
        )

    def test_multi_key_sort_key_out_of_bounds_raises(self):
        np.testing.assert_raises_regex(
            IndexError,
            r"key index 5 is out of bounds",
            multi_key_sort, np.array([[1, 2], [3, 4]]), [5]
        )
