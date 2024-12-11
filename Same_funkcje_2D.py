import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.transform import resize
matplotlib.use('TkAgg')

#wczytywanie obrazu
image1 = io.imread('path')
#kernel do zmniejszania obrazu poprzez średnią wartość pikseli
def average(kernel):
    return np.mean(kernel, axis=(0, 1))
#kernel do zmniejszania obrazół poprzez wybieranie najbliższego piksela
def pierwszy(kernel):
    return kernel[0, 0]
#kernele do interpolacji z zadania nr 1 (brak zmian)
def h1(original_x, new_x, width):
    return original_x <= new_x < original_x + width
def h2(original_x, new_x, width):
    return (original_x - width / 2 < new_x <= original_x + width / 2)
def h3(original_x, new_x, width):
    t = (original_x - new_x) / width
    if 1 - abs(t) > 0:
        return 1 - abs(t)
    return 0

#zmodyfikowana funkcja interpolacji z zad 1
def interpolate(x_vals, y_vals, x_interp, kernel):
    y_interp = np.zeros((len(x_interp), y_vals.shape[1]))
    #przejście przez długość nowego obrazu
    for j in range(len(x_interp)):
        contribution = np.zeros(y_vals.shape[1])
        total_weight = 0
        #przejście przez długość oryginalnego obrazu i nadanie punktom wagi
        for i in range(len(x_vals)):
            if i < len(x_vals) - 1:
                width = x_vals[i + 1] - x_vals[i]
            weight = kernel(x_vals[i], x_interp[j], width)
            contribution += y_vals[i] * weight
            total_weight += weight
        if total_weight > 0:
            y_interp[j] = contribution / total_weight
    return y_interp

#funkcja pozwalająca użyć różnych kerneli do zmniejszania obrazy
def resize_with_kernel(image, x, y, kernel, kernel_size):
    matryca = image[x:x + kernel_size, y:y + kernel_size]
    return kernel(matryca)

#zmniejszanie obrazu za pomocą splotu
def downscale(image, kernel, kernel_size):
    #parametry obrazu
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]
    #parametry nowego obrazu
    new_width = int(width / kernel_size)
    new_height = int(height / kernel_size)
    #inicjalizacja tablicy zer o wymiarach nowego obrazu
    new_image = np.zeros((new_height, new_width, channels), dtype=int)

    #Iteracja poprzez wyskość i szerokość nowego obrazu
    for i in range(new_height):
        for j in range(new_width):
            #przypisanie piksela do nowego obrazu
            new_image[i, j] = resize_with_kernel(image, i * kernel_size, j * kernel_size, kernel, kernel_size)
    return new_image

def upscale(image, kernel, kernel_size):
    #parametry obrazu
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]
    #parametry nowego obrazu
    new_width = int(width * kernel_size)
    new_height = int(height * kernel_size)
    #inicjalizacja ablicy zer o wymiarach nowego obrazu
    temp = np.zeros((height, new_width, channels), dtype=image.dtype)
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    #interpolacja pozioma
    for j in range(height):
        temp[j, :, :] = interpolate(np.arange(0, width),image[j, :, :],np.linspace(0, width - 1, new_width),kernel)
    #interpolacja pionowa
    for i in range(new_width):
        new_image[:, i, :] = interpolate(np.arange(0, height),temp[:, i, :],np.linspace(0, height - 1, new_height),kernel)
    return new_image

def bez_int(image, kernel, kernel_size):
    # parametry obrazu
    width = image.shape[1]
    height = image.shape[0]
    channels = image.shape[2]
    # parametry nowego obrazu
    new_width = int(width * kernel_size)
    new_height = int(height * kernel_size)
    # inicjalizacja tablicy zer o wymiarach nowego obrazu
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    # interpolacja pozioma
    for i in range(height):
        for j in range(width):
            new_image[i * kernel_size:(i + 1) * kernel_size, j * kernel_size:(j + 1) * kernel_size, :] = image[i, j, :]
    return new_image

#przeskalowanie obrazu o ułamek, najpirw zmniejszenie obrazu a następnie powiększenie
def by_fraction_v1(image, interpolation_kernel, downs_kernel, numerator, denominator):
    down_image = downscale(image, downs_kernel, denominator)
    up_image = upscale(down_image, interpolation_kernel, numerator)
    return up_image
#przeskalowanie obrazu o ułamek, najpirw powiększenie obrazu a następnie zmniejszenie
def by_fraction_v2(image, interpolation_kernel, downs_kernel, numerator, denominator):
    image = upscale(image, interpolation_kernel, numerator)
    image = downscale(image, downs_kernel, denominator)
    return image

def calculate_mse(original, processed):
    return mean_squared_error(original, processed)
