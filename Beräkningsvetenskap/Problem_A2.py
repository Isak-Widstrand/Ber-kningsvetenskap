# --- Import necessary libraries ---
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Define test function ---
def test_function(x):
    return np.sin(21*x)*np.exp(-0.3*x) + x 

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

# --- Convergence test ---
def test_integration():
    # reference function
    xmin = 0
    xmax = np.pi
    I_exact = 5.00096303820792
    x_fine = np.linspace(xmin, xmax, 1000)
    y_fine = test_function(x_fine)
    plt.figure(1)
    plt.plot(x_fine,y_fine, linewidth=1.5, label='f(x)')
    E = []
    H = []
    H4 = []
    N = []
    I_exact, err = quad(test_function, xmin, xmax, epsabs =1e-12, epsrel=1e-12)
    print(I_exact, err)
    n = 10

    for i in range(1,9):
        x = np.linspace(xmin, xmax, n+1)
        y = test_function(x)
        I = simpson(y, xmin, xmax, n)
        print("I = ", I, "I_ex = ", I_exact)
        E.append(np.abs(I-I_exact))
        h = (xmax - xmin)/n
        H.append(h)
        H4.append(h**4)
        n = 2*n
        print(f"n = {n:5d} | I = {I:.12f} | error = {err:.3e}")
        plt.figure(1)
        plt.plot(x,y, linewidth=1.5, label='n='+str(n))
        plt.legend(loc='upper left')
        plt.pause(0.5)
        plt.title('Test function')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        
    #--- Plot error in the loglog scale ---#
    plt.figure(2)
    plt.loglog(H, E, linewidth=1.5, label='Error')
    plt.loglog(H, H4, '--', linewidth=1.5, label='h^4')
    plt.title('Convergence plot')
    plt.legend(loc='upper right')
    plt.xlabel('ln(h)')
    plt.ylabel('ln(E)')
    plt.grid(True)
    plt.show()

# --- The main part of the program ---
if __name__ == "__main__":
    test_integration()

#Ändrade bara testfunktionen och la till vad den exakta integralens värde är.
#Ju längre vi går desto mindre blir felet, vilket är förväntat.
#Vid n = 1280 är felet ungefär 10^-11 vilket är mycket bra nog för de flesta tillämpningar.
