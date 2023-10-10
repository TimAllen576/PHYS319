"""Attempt from to convert DDE to ODE and solve with SciPy
main issue is the delay"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc


# def history(t):
#     """Keep the 'history' of the system"""
#     return np.sin(t)    # Possibly 0.55?


def ode_system(t, y, delay_buffer, delta, a, r):
    """
    Equation of the system
    """
    x, y_delayed = y
    delay_buffer.append(x)
    if len(delay_buffer) > delta:
        delay_buffer.pop(0)
    y_delayed = delay_buffer[0]
    # print(y)
    print(t)
    return [x - x**3 - a * y_delayed + r, x]


def plotter(delay_buffer, initial_condition,
            t_span, t_points, delta, a, r):
    """Solves and plots the system"""
    solution = sc.solve_ivp(ode_system, t_span, initial_condition,
                            t_eval=t_points,
                            args=(delay_buffer, delta, a, r))
    t_values = solution.t
    y_values = solution.y[0]
    plt.plot(t_values, y_values)
    plt.xlabel('Time')
    plt.ylabel("T'")
    plt.title('Numerical Solution of Delay Differential Equation')
    plt.grid(True)
    plt.show()


def main():
    """Does all the things"""
    a = 0.75
    delta = 8
    tf = 25
    dt = 0.001

    ini_val = 0.55
    r = 0.0
    delay_buffer = [ini_val] * delta
    t_span = (0, tf)
    t_points = np.arange(0, tf, dt)
    initial_condition = np.array([ini_val, ini_val])
    plotter(delay_buffer, initial_condition,
            t_span, t_points, delta, a, r)


if __name__ == '__main__':
    main()
