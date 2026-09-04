import numpy as np


def forward_euler(f, t_s, x, h_s, amod):
    """
    Arguments:
        f - function representing the right-hand side of the differential equation (dx/dt = f(t, x, amod)), function
        t_s - vector of points in time at which numerical solutions will be approximated [s], 1D numpy array
        x - state history array (shape: num_states x len(t_s)) with initial conditions at x[:, 0], 2D numpy array
        h_s - integration time step size [s], scalar
        amod - additional arguments passed directly to f (e.g., amod), tuple

    Returns:
        t_s - vector of points in time at which numerical solutions were approximated [s], 1D numpy array
        x - numerically approximated state solution history across time, 2D numpy array
    """
    for i in range(1, len(t_s)):
        t_prev = t_s[i - 1]
        x_prev = x[:, i - 1]

        x[:, i] = x_prev + h_s * f(t_prev, x_prev, amod)

    return t_s, x


def rk4(f, t_s, x, h_s, amod):
    """
    Arguments:
        f - function representing the right-hand side of the differential equation (dx/dt = f(t, x, amod)), function
        t_s - vector of points in time at which numerical solutions will be approximated [s], 1D numpy array
        x - state history array (shape: num_states x len(t_s)) with initial conditions at x[:, 0], 2D numpy array
        h_s - integration time step size [s], scalar
        amod - additional arguments passed directly to f (e.g., amod), tuple

    Returns:
        t_s - vector of points in time at which numerical solutions were approximated [s], 1D numpy array
        x - numerically approximated state solution history across time, 2D numpy array
    """
    for i in range(1, len(t_s)):
        t_prev = t_s[i - 1]
        x_prev = x[:, i - 1]

        k1 = f(t_prev,                 x_prev,                     amod)
        k2 = f(t_prev + 0.5 * h_s,     x_prev + 0.5 * h_s * k1,     amod)
        k3 = f(t_prev + 0.5 * h_s,     x_prev + 0.5 * h_s * k2,     amod)
        k4 = f(t_prev + h_s,           x_prev + h_s * k3,           amod)

        x[:, i] = x_prev + (h_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_s, x