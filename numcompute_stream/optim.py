import numpy as np
from typing import Callable


def grad(
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: float = 1e-5,
        method: str = 'central') -> np.ndarray:
    """
    Estimate the gradient of function f at point x using finite differences.
    This is used to evaluate how the loss changes with respect to the
    parameters, which is essential for optimization algorithms like
    gradient descent.

    The gradient provides the direction and magnitude of the steepest ascent,
    which can be used to update the parameters in the opposite direction to
    minimize the loss.

    Parameters
    ----------
    f : function, where f(x) -> scalar
    x : np.ndarray, shape (n,)
    h : float, step size
    method : 'central' or 'forward'
        'central' shifts x by h in both directions, while 'forward' only shifts
        x by h in the positive direction (faster, but less accurate).

    Returns
    -------
    gradients : np.ndarray, shape (n,)

    Raises
    ------
    ValueError
        If method is not 'central' or 'forward'.
        If x is not a 1D numpy array.
        If f is not a callable function.

    Complexity
    ----------
    Time Complexity: O(n) for 'forward'; O(2n) for 'central'.
    Space Complexity: O(n) for the gradient array.
    """

    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Use 'central' or 'forward'.")

    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("Input x must be a 1D numpy array.")

    if not callable(f):
        raise ValueError("Input f must be a callable function.")

    gradients = np.zeros_like(x)

    # For forward method, we pre-compute f(x) to avoid redundant calculations.
    f_x = f(x) if method == 'forward' else 0.0

    for i in range(len(x)):
        x_shifted = np.copy(x)
        x_shifted[i] += h
        if method == 'central':
            x_shifted_back = np.copy(x)
            x_shifted_back[i] -= h
            gradients[i] = (f(x_shifted) - f(x_shifted_back)) / (2 * h)
        else:
            gradients[i] = (f(x_shifted) - f_x) / h
    return gradients


def jacobian(
        F: Callable[[np.ndarray], np.ndarray],  # Capital F - returns a vector
        x: np.ndarray,
        h: float = 1e-5,
        method: str = 'central') -> np.ndarray:
    """
    Estimate the Jacobian of F at point x using finite differences. The
    Jacobian reflects how each element in F's return vector is impacted by a
    change in each element of the input vector x.

    Parameters
    ----------
    F : function, F(x) -> np.ndarray of shape (m,)
    x : np.ndarray, shape (n,)
    h : float, step size
    method : 'central' or 'forward'
        'central' shifts x by h in both directions, while 'forward' only shifts
        x by h in the positive direction (faster, but less accurate).

    Returns
    -------
    J : np.ndarray, shape (m, n)

    Raises
    ------
    ValueError
        If method is not 'central' or 'forward'.
        If x is not a 1D numpy array.
        If F is not a callable function that returns a vector.

    Complexity
    ----------
    Time Complexity: O(m * n) for 'forward'; O(2 * m * n) for 'central'.
    Space Complexity: O(m * n) for the Jacobian matrix.
    where m is the output dimension of F and n is the input dimension of x.
    """

    if method not in ['central', 'forward']:
        raise ValueError("Invalid method. Use 'central' or 'forward'.")

    if not isinstance(x, np.ndarray) or x.ndim != 1:
        raise ValueError("Input x must be a 1D numpy array.")

    if not callable(F):
        raise ValueError("Input f must be a callable function.")

    x = np.asarray(x, dtype=float)
    F_output = F(x)
    F_X = None
    try:
        F_X = np.asarray(F_output, dtype=float)
    except ValueError:
        F_X = None

    if F_X is None or not isinstance(F_X, np.ndarray) or F_X.ndim != 1:
        raise ValueError("Function F must return a 1D array.")

    m = len(F_X)                          # number of outputs
    n = len(x)                           # number of inputs
    J = np.zeros((m, n))                 # output: m x n matrix

    for i in range(n):                   # nudge each input dimension
        e_i = np.zeros_like(x)
        e_i[i] = 1.0

        if method == 'central':
            F_shifted = np.asarray(F(x + h * e_i), dtype=float)
            F_shifted_back = np.asarray(F(x - h * e_i), dtype=float)
            J[:, i] = (F_shifted - F_shifted_back) / (2 * h)  # fill column i

        else:
            F_shifted = np.asarray(F(x + h * e_i), dtype=float)
            J[:, i] = (F_shifted - F_X) / h              # fill column i

    return J
