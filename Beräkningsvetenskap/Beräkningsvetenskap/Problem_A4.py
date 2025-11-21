# --- Import necessary libraries ---
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def boat():
    p_f = np.array([
    [98.7, 52.9],
    [119.4, 49.4],
    [140.4, 46.4],
    [165.1, 44.1],
    [194.2, 42.5],
    [219.1, 42.1],
    [237.8, 60.1],
    [292.8, 60.5],
    [297.3, 572.6],
    [301.5, 572.6],
    [301.1, 60.4],
    [453.4, 61.2],
    [456.5, 48.7],
    [467.6, 40.3],
    [541.5, 40.1],
    [590.4, 42.7],
    [631.1, 49.4]
    ])

    p_g = np.array([
    [98.7, 52.9],
    [146.1, -0.5],
    [158.8, -14.0],
    [290.4, -26.5],
    [337.6, -93.8],
    [385.8, -93.1],
    [386.1, -29.2],
    [560.2, -15.1],
    [565.2, -60.0],
    [575.2, -60.0],
    [580.4, -6.5],
    [587.6, 0.0],
    [616.8, 23.3],
    [631.1, 49.4]
    ])

    return p_f, p_g

# --- Lagrange functions ---
def lagrange(x, x1, x2):
    l1 = (x2 - x) / (x2 - x1)
    l2 = (x - x1) / (x2 - x1)
    return l1, l2

# --- Piecewise linear interpolation function ---
def func_pw_linear(x, x_nodes, y_nodes):
    x = np.asarray(x)
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        for j in range(len(x_nodes) - 1):
            if x_nodes[j] <= x[i] <= x_nodes[j + 1]:
                l1, l2 = lagrange(x[i], x_nodes[j], x_nodes[j + 1])
                y[i] = y_nodes[j] * l1 + y_nodes[j + 1] * l2
    return y

# --- Data points ---
x_f = boat()[0][:,0]
y_f = boat()[0][:,1]
x_g = boat()[1][:,0]
y_g = boat()[1][:,1]
x = np.linspace(98.7, 631.1, 2001)

# Compute piecewise linear interpolation
f = func_pw_linear(x, x_f, y_f)
g = func_pw_linear(x, x_g, y_g)

# --- Composite Simposon's rule ---
def simpson(y, a, b, n):
    if n % 2 != 0:
        raise ValueError("Number of subintervals n must be even.")
    h = (b - a) / n
    y = np.asarray(y)
    I = (h / 3) * (
    y[0]
    + y[-1]
    + 4 * np.sum(y[1:-1:2]) # odd indices
    + 2 * np.sum(y[2:-2:2]) # even indices
    )
    return I

# --- Calculatinng center of mass ---
def mass_center(f, g):
    x_numerator = simpson(x*(f-g), -3, 3, 1280)
    x_denominator = simpson(f-g, -3, 3, 1280)
    y_numerator = simpson((f**2 - g**2)/2, -3, 3, 1280)
    y_denominator = simpson(f-g, -3, 3, 1280)
    
    x_value = x_numerator / x_denominator
    y_value = y_numerator / y_denominator
    
    return ([x_value, y_value])

print(mass_center(f, g)[0], ", ", mass_center(f, g)[1])

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x_f, y_f, 'ro', label='Data points')
plt.plot(x, f, 'r-', linewidth=1.5, label='f(x)')
plt.plot(x_g, y_g, "go", label='Data points')
plt.plot(x, g, 'g-', linewidth=1.5, label='g(x)')
plt.plot(mass_center(f, g)[0], mass_center(f, g)[1], "b*", markersize=12, label='Center of Mass')
plt.grid(True)
plt.title('Boat Shape - Piecewise Linear Interpolation')
plt.legend(loc='upper left')
plt.show()