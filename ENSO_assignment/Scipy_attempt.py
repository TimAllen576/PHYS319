"""Attempt from to convert DDE to ODE and solve with SciPy
main issue is the delay"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc
from timeit import timeit


class DelayBuffer:
    """Stores the history of the system so values can be interpolated"""

    def __init__(self, delta: float | int):
        self.delta = delta
        self.buffer = np.array([-delta,
                                values_before_zero_time(-delta)],
                               dtype=float)

    def update(self, t: float, y: np.ndarray):
        """Updates the buffer"""
        self.buffer = np.block([[self.buffer], [t, y]])


def values_before_zero_time(t: int | float) -> int | float:
    """Keep the 'history' of the system"""
    return 0.55 + 0 * t


def ode_system(t: float, y: np.ndarray, buffer, delta, a, r):
    """
    Equation of the system
    """
    buffer.update(t, y)
    if t - delta <= 0:
        y_delayed = values_before_zero_time(t - delta)
    else:
        y_delayed = np.interp(t - delta, buffer.buffer[:, 0],
                              buffer.buffer[:, 1])
    return y - y ** 3 - a * y_delayed + r


def plotter(y0: np.ndarray, t_span, t_points, delta: int | float, a, r):
    """Solves and plots the system"""
    buffer = DelayBuffer(delta)
    solution = sc.solve_ivp(ode_system, t_span, y0, t_eval=t_points,
                            args=(buffer, delta, a, r))
    t_values = solution.t
    temperature = solution.y[0]
    plt.plot(t_values, temperature)
    plt.xlabel('Time')
    plt.ylabel("T'")
    plt.title('Numerical Solution of Delay Differential Equation')
    plt.grid(True)
    plt.show()


def main():
    """Does all the things"""
    delta, a, tf, dt = (8, 0.75, 25, 0.001)
    y0 = np.array([0.55])
    r = 0.0
    t_span = (0, tf)
    t_points = np.arange(0, tf, dt)
    # print(timeit("plotter(y0, t_span, t_points, delta, a, r)", globals=globals(), number=10))
    plotter(y0, t_span, t_points, delta, a, r)

    # TODO: interp opts, inline history, sort buffer, type hints
    #  check what to make arrays for sped


if __name__ == '__main__':
    main()
