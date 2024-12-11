import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
matplotlib.use('TkAgg')
#Interpolacja przyjmuje wartości x i y funkcji interpolowanej oraz x na których chcemy przeprowadzić interpolacje, oraz kernel, który determinuje w jaki sposób interpolacja będzie przebiegać
def interpolate(x_vals, y_vals, x_interp, kernel):
    #inicjalizacja tablicy na wartości y
    y_interp = []
    #iteracja poprzez wszystkie wartości x_interp, dla każdego inicjalizacja wagi
    for j in range(len(x_interp)):
        contribution = 0
        total_weight = 0
        #iteracja poprzez wszystkie wartości x_vals, dla każdego obliczanie wagi
        for i in range(len(x_vals)):
            if i < len(x_vals)-1:
                width = x_vals[i+1] - x_vals[i]
            weight = kernel(x_vals[i], x_interp[j], width)
            contribution += y_vals[i] * weight
            total_weight += weight
        if total_weight > 0:
            y_interp.append(contribution / total_weight)
    return y_interp

#Funckja nr 1
def f1(x):
    return np.sin(x)
#Funckja nr 1
def f2(x):
    return np.sin(x**(-1))
#Funckja nr 1
def f3(x):
    return np.sign(np.sin(8*x))

#sample hold
def h1(original_x, new_x, width):
    return original_x  <= new_x < original_x +width

#nearest neighbor
def h2(original_x, new_x, width):
    return (original_x - width/2 < new_x <= original_x + width/2)
#linear
def h3(original_x, new_x, width):
    t = (original_x - new_x) / width
    if 1 - abs(t) > 0:
        return 1 - abs(t)
    return 0
