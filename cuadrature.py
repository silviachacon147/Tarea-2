#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad

# Función a integrar
def func(x):
    return x**6 - x**2 * np.sin(2*x)

# Gauss-Legendre quadrature
def gaussxw(N):
    # Inicialización y cálculo de los puntos y pesos de Gauss
    a = np.linspace(3, 4 * (N - 1), N) / ((4 * N) + 2)
    x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))  # Puntos de Gauss
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, dtype = float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = np.max(np.abs(dx))

    # Pesos de Gauss
    w = 2 * (N + 1) * (N + 1)/(N * N * (1 - x * x) * dp * dp)
    return x, w

# Transformación de los puntos y pesos al intervalo [a, b]
def gaussxwab(a, b, x, w):
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

# Cálculo de la integral con Gauss-Legendre
def gauss(a, b, N, f):
    x, w = gaussxw(N)
    ex, ew = gaussxwab(a, b, x, w)  # Cambio de variables para [a, b]
    return np.sum(ew * f(ex))  # Sumar los productos de pesos y evaluaciones de f

xw_7 = gaussxw(7)


#Se escalan los nodos que ya tenemos y los pesos 
xw_7escalado = gaussxwab(1,3,xw_7[0], xw_7[1])



#Igualmente se aplica para los tres
def sumatoria (xw_7escalado, func):
    result_7 = np.sum(func(xw_7escalado[0])*xw_7escalado[1])
    return result_7
print(sumatoria(xw_7escalado, func))

