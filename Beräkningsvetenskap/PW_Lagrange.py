# --- Import necessary libraries ---
import numpy as np
import matplotlib.pyplot as plt

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
x_f = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y_f = np.array([-3, -3, 0, 5, 12, 14, 16, 15, 12, 9, 2, -3])
x = np.linspace(0, 11, 2001)

# Compute piecewise linear interpolation
f = func_pw_linear(x, x_f, y_f)

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x_f, y_f, 'ro', label='Data points')
plt.plot(x, f, 'b-', linewidth=1.5, label='p(x)')
plt.grid(True)
plt.xlabel('Moonths')
plt.ylabel('Temperature (°C)')
plt.title('Continuous Piecewise Linear Interpolation')
plt.legend(loc='upper left')
plt.show()