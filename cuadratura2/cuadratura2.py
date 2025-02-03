#!/usr/bin/env python
import numpy as np
from scipy.integrate import quad

def func(x):
    """
    Función a integrar.
    
    Parámetros:
        x (float): Valor en el que se evalúa la función.
    
    Retorna:
        float: Resultado de la función evaluada en x.
    """
    return x**6 - x**2 * np.sin(2*x)

def gaussxw(N):
    """
    Calcula los puntos y pesos de Gauss-Legendre para la cuadratura.
    
    Parámetros:
        N (int): Número de puntos de cuadratura.
    
    Retorna:
        tuple: (x, w) donde x son los nodos y w son los pesos de Gauss.
    """
    a = np.linspace(3, 4 * (N - 1), N) / ((4 * N) + 2)
    x = np.cos(np.pi * a + 1 / (8 * N * N * np.tan(a)))
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = np.ones(N, dtype=float)
        p1 = np.copy(x)
        for k in range(1, N):
            p0, p1 = p1, ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        dp = (N + 1) * (p0 - x * p1) / (1 - x * x)
        dx = p1 / dp
        x -= dx
        delta = np.max(np.abs(dx))
    w = 2 * (N + 1) * (N + 1) / (N * N * (1 - x * x) * dp * dp)
    return x, w

def gaussxwab(a, b, x, w):
    """
    Escala los nodos y pesos de Gauss-Legendre al intervalo [a, b].
    
    Parámetros:
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        x (array): Nodos de Gauss-Legendre.
        w (array): Pesos de Gauss-Legendre.
    
    Retorna:
        tuple: (ex, ew) nodos y pesos escalados.
    """
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w

def gauss(a, b, N, f):
    """
    Calcula la integral de una función en el intervalo [a, b] usando Gauss-Legendre.
    
    Parámetros:
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        N (int): Número de puntos de cuadratura.
        f (function): Función a integrar.
    
    Retorna:
        float: Valor aproximado de la integral.
    """
    x, w = gaussxw(N)
    ex, ew = gaussxwab(a, b, x, w)
    return np.sum(ew * f(ex))

xw_7 = gaussxw(7)
xw_7escalado = gaussxwab(1, 3, xw_7[0], xw_7[1])

def sumatoria(xw_7escalado, func):
    """
    Calcula la suma ponderada de los valores de la función en los nodos escalados.
    
    Parámetros:
        xw_7escalado (tuple): Nodos y pesos escalados.
        func (function): Función a evaluar.
    
    Retorna:
        float: Resultado de la integración aproximada.
    """
    return np.sum(func(xw_7escalado[0]) * xw_7escalado[1])

print(sumatoria(xw_7escalado, func))
