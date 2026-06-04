import numpy as np
from numpy.testing import (
    assert_raises_regex,
    assert_allclose,
)
from numcompute import optim


# Test with a simple quadratic function f(x) = x^2
def sample_scalar_function(x) -> int | float:
    return np.sum(x**2)


def sample_vector_function(x) -> np.ndarray:
    r = [np.sum(x**2), np.sum(x / 2), 5]
    return np.array(r)


class TestGrad:
    def test_grad_forward(self):
        x = np.array([1.0, 2.0, 3.0])
        # forward method, we expect a small numerical error due to
        # the nature of the approximation
        expected = np.array([2.00001, 4.00001, 6.00001])
        computed = optim.grad(sample_scalar_function, x, method='forward')
        assert_allclose(computed, expected)

    def test_grad_central(self):
        x = np.array([1.0, 2.0, 3.0])
        expected = np.array([2.0, 4.0, 6.0])
        computed = optim.grad(sample_scalar_function, x, method='central')
        assert_allclose(computed, expected)

    def test_grad_invalid_method(self):
        x = np.array([1.0, 2.0])
        assert_raises_regex(
            ValueError,
            r"Invalid method. Use 'central' or 'forward'.",
            optim.grad,
            f=sample_scalar_function,
            x=x,
            method='invalid'
        )

    def test_grad_non_callable_f(self):
        x = np.array([1.0, 2.0])
        assert_raises_regex(
            ValueError,
            r"Input f must be a callable function.",
            optim.grad,
            f="not a function",  # type: ignore
            x=x
        )

    def test_grad_non_1d_x(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_raises_regex(
            ValueError,
            r"Input x must be a 1D numpy array.",
            optim.grad,
            f=sample_scalar_function,
            x=x
        )


class TestJacobian:

    def test_jacobian_forward(self):
        x = np.array([1.0, 2.0, 3.0])
        # forward method, we expect a small numerical error due to
        # the nature of the approximation
        expected = np.array([
            [2.00001, 4.00001, 6.00001],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0]
        ])
        computed = optim.jacobian(
            F=sample_vector_function,
            x=x,
            method='forward'
        )
        assert_allclose(computed, expected)

    def test_jacobian_central(self):
        x = np.array([1.0, 2.0, 3.0])
        expected = np.array([
            [2.0, 4.0, 6.0],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0]
        ])
        computed = optim.jacobian(
            F=sample_vector_function,
            x=x,
            method='central'
        )
        assert_allclose(computed, expected)

    def test_non_callable_F(self):
        x = np.array([1.0, 2.0])
        assert_raises_regex(
            ValueError,
            r"Input f must be a callable function.",
            optim.jacobian,
            F="not a function",  # type: ignore
            x=x
        )

    def test_F_returns_scalar(self):
        x = np.array([1.0, 2.0])
        assert_raises_regex(
            ValueError,
            r"Function F must return a 1D array.",
            optim.jacobian,
            F=sample_scalar_function,  # type: ignore
            x=x
        )

    def test_non_1d_x(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert_raises_regex(
            ValueError,
            r"Input x must be a 1D numpy array.",
            optim.jacobian,
            F=sample_vector_function,
            x=x
        )

    def test_invalid_method(self):
        x = np.array([1.0, 2.0])
        assert_raises_regex(
            ValueError,
            r"Invalid method. Use 'central' or 'forward'.",
            optim.jacobian,
            F=sample_vector_function,
            x=x,
            method='invalid'
        )

    def test_F_returns_not_castable(self):
        x = np.array([1.0 , 2.0])
        assert_raises_regex(
            ValueError,
            r"Function F must return a 1D array.",
            optim.jacobian,
            F=lambda _: 'non-float values',  # type: ignore
            x=x
        )
