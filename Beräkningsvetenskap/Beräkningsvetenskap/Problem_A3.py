# --- Import necessary libraries ---
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

#calculating the functions:
x_f = np.array([-3, -2.5, -2, 0, 2, 2.5, 3])
y_f = np.array([2, 1, 0.65, 0.5, 0.65, 1, 2])

# --- Ensure column vectors (optional) ---
x_f = x_f.reshape(-1, 1)
y_f = y_f.reshape(-1, 1)
n_f = len(x_f)
print("n=", n_f)

# --- Construct Vandermonde matrices ---
V_f = np.empty((n_f, 0))

for i in range(0, n_f):
    col = x_f ** i # compute one column
    V_f = np.hstack((V_f, col)) # append it horizontally

# --- Solve for coefficients ---
a_f = np.linalg.solve(V_f, y_f)

# --- Construct the polynomial function ---
x = np.linspace(-3, 3, 2001)
f = sum(a_f[i] * x**i for i in range(n_f))

# --- Data points g ---
x_g = np.array([-3, -2.5, -2, 0, 2, 2.5, 3])
y_g = np.array([2, 1.5, 1.2, 1, 1.2, 1.5, 2])

# --- Ensure column vectors (optional) ---
x_g = x_g.reshape(-1, 1)
y_g = y_g.reshape(-1, 1)
n_g = len(x_g)
print("n=", n_g)

# --- Construct Vandermonde matrices ---
V_g = np.empty((n_g, 0))

for i in range(0, n_g):
    col = x_g ** i # compute one column
    V_g = np.hstack((V_g, col)) # append it horizontally

# --- Solve for coefficients ---
a_g = np.linalg.solve(V_g, y_g)

# --- Construct the polynomial function ---
x = np.linspace(-3, 3, 2001)
g = sum(a_g[i] * x**i for i in range(n_g))

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
plt.plot(x, f, 'b-', linewidth=1.5, label='f(x)')
plt.plot(x_g, y_g, "*", label='Data points')
plt.plot(x, g, 'g-', linewidth=1.5, label='g(x)')
plt.plot(mass_center(f, g)[0], mass_center(f, g)[1], 'ks', label='Center of Mass')
plt.title('6th-Order Polynomial Interpolation')
plt.grid(True)
plt.legend(loc='upper center')
plt.show()

#