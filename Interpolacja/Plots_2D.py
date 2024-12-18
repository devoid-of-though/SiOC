import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.metrics import mean_squared_error
matplotlib.use('TkAgg')

#wczytywanie obrazu
image1 = io.imread('/home/userbrigh/PycharmProjects/SystemyiObrazyCyfrowe/Obrazy_cyfrowe/Obrazy/bnw1.png')
image2= io.imread("/home/userbrigh/PycharmProjects/SystemyiObrazyCyfrowe/Obrazy_cyfrowe/Obrazy/bnw2.png")
image3= io.imread("/home/userbrigh/PycharmProjects/SystemyiObrazyCyfrowe/Obrazy_cyfrowe/Obrazy/panda.jpg")
image4= io.imread("/home/userbrigh/PycharmProjects/SystemyiObrazyCyfrowe/Obrazy_cyfrowe/Obrazy/bw3.png")
image5= io.imread("/home/userbrigh/PycharmProjects/SystemyiObrazyCyfrowe/Obrazy_cyfrowe/Obrazy/bw5.png")

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

"""
#normal
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(image1)
axs[0].set_title('Obraz oryginalny')
upscaled_image1_h1 = upscale(image1, h1, 2)
axs[1].imshow(upscaled_image1_h1)
axs[1].set_title('Przeskalowanie za pomocą sample hold')
print("MSE for image1 with sample hold:", calculate_mse(image1, resize(upscaled_image1_h1, image1.shape)))
upscaled_image1_h3 = upscale(image1, h2, 2)
axs[2].imshow(upscaled_image1_h3)
axs[2].set_title('Przeskalowanie za pomocą Nearest Neighbor')
upscaled_image1_h3 = upscale(image1, h3, 2)
axs[3].imshow(upscaled_image1_h3)
axs[3].set_title('Przeskalowanie za pomocą kernela liniowego')
print("MSE for image1 with linear kernel:", calculate_mse(image1, resize(upscaled_image1_h3, image1.shape)))
print("MSE for image3 with :", calculate_mse(image1, resize(upscale(image1, h2, 2), image1.shape)))
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(image2)
axs[0].set_title('Obraz oryginalny')
upscaled_image2_h1 = upscale(image2, h1, 2)
axs[1].imshow(upscaled_image2_h1)
axs[1].set_title('Przeskalowanie za pomocą sample hold')
print("MSE for image2 with sample hold:", calculate_mse(image2, resize(upscaled_image2_h1, image2.shape)))
upscaled_image2_h3 = upscale(image2, h2, 2)
axs[2].imshow(upscaled_image2_h3)
axs[2].set_title('Przeskalowanie za pomocą Nearest Neighbor')
upscaled_image2_h3 = upscale(image2, h3, 2)
axs[3].imshow(upscaled_image2_h3)
axs[3].set_title('Przeskalowanie za pomocą kernela liniowego')
"""
"""print("MSE for image2 with linear kernel:", calculate_mse(image2, resize(upscaled_image2_h3, image2.shape)))
print("MSE for image3 with :", calculate_mse(image2, resize(upscale(image2, h2, 2), image2.shape)))
plt.figure(figsize=(15, 5))
plt.imshow(image3)
plt.title('Obraz oryginalny')
upscaled_image3_h1 = upscale(image3, h1, 2)
plt.figure(figsize=(15, 5))
plt.imshow(upscaled_image3_h1)
upscaled_image3_h1 = upscale(image3, h2, 2)
plt.figure(figsize=(15, 5))
plt.imshow(upscaled_image3_h1)
plt.title('Przeskalowanie za pomocą Nearest Neighbor')
print("MSE for image3 with sample hold:", calculate_mse(image3, resize(upscaled_image3_h1, image3.shape)))
upscaled_image3_h3 = upscale(image3, h3, 2)
plt.figure(figsize=(15, 5))
plt.imshow(upscaled_image3_h3)
plt.title('Przeskalowanie za pomocą kernela liniowego')
print("MSE for image3 with linear kernel:", calculate_mse(image3, resize(upscaled_image3_h3, image3.shape)))
print("MSE for image3 with :", calculate_mse(image3, resize(upscale(image3, h2, 2), image3.shape)))
"""
"""

#partial
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image1)
axs[0].set_title('Obraz oryginalny')
by_fraction_v1_image1 = by_fraction_v1(image1, h3, average, 3, 2)
axs[1].imshow(by_fraction_v1_image1)
print("MSE for image1 by_fraction_v1:", calculate_mse(image1, resize(by_fraction_v1_image1, image1.shape)))
by_fraction_v2_image1 = by_fraction_v2(image1, h3, average, 3, 2)
axs[2].imshow(by_fraction_v2_image1)
print("MSE for image1 by_fraction_v2:", calculate_mse(image1, resize(by_fraction_v2_image1, image1.shape)))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image2)
axs[0].set_title('Obraz oryginalny')
by_fraction_v1_image2 = by_fraction_v1(image2, h3, average, 3, 2)
axs[1].imshow(by_fraction_v1_image2)
print("MSE for image2 by_fraction_v1:", calculate_mse(image2, resize(by_fraction_v1_image2, image2.shape)))
by_fraction_v2_image2 = by_fraction_v2(image2, h3, average, 3, 2)
axs[2].imshow(by_fraction_v2_image2)
print("MSE for image2 by_fraction_v2:", calculate_mse(image2, resize(by_fraction_v2_image2, image2.shape)))

"""
"""plt.figure(figsize=(15, 5))
by_fraction_v1_image3 = by_fraction_v1(image3, h3, average, 3, 2)
plt.imshow(by_fraction_v1_image3)
plt.title("Obraz najpierw zmniejszony a następnie powiększony")
print("MSE for image3 by_fraction_v1:", calculate_mse(image3, resize(by_fraction_v1_image3, image3.shape)))
by_fraction_v2_image3 = by_fraction_v2(image3, h3, average, 3, 2)
plt.figure(figsize=(15, 5))
plt.imshow(by_fraction_v2_image3)
plt.title("Obraz najpierw powiększony a następnie zmniejszony")
print("MSE for image3 by_fraction_v2:", calculate_mse(image3, resize(by_fraction_v2_image3, image3.shape)))
"""
"""

"""

"""plt.figure(figsize=(15, 5))
downscaled_image3_avg = downscale(image3, average, 2)
plt.imshow(downscaled_image3_avg)
plt.title('Zmniejszenie za pomocą average')
print("MSE for image3 with average downscale:", calculate_mse(image3, bez_int(downscaled_image3_avg, average, 2)))
downscaled_image3_nn = downscale(image3, pierwszy, 2)
plt.figure(figsize=(15, 5))
plt.imshow(downscaled_image3_nn)
plt.title('Zmniejszenie za pomocą pierwszego')
print("MSE for image3 with pierwszy downscale:", calculate_mse(image3,bez_int(downscaled_image3_nn, nearest_neighbour, 2)))



fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image4)
axs[0].set_title('Obraz oryginalny')
downscaled_image4_avg = downscale(image4, average, 2)
axs[1].imshow(downscaled_image4_avg)
axs[1].set_title('Zmniejszenie za pomocą average')
print("MSE for image4 with average downscale:", calculate_mse(image4, bez_int(downscaled_image4_avg, average, 2)))
downscaled_image4_nn = downscale(image4, pierwszy, 2)
axs[2].imshow(downscaled_image4_nn)
axs[2].set_title('Zmniejszenie za pomocą pierwszego') 
print("MSE for image4 with pierwszy downscale:", calculate_mse(image4, bez_int(downscaled_image4_nn,average,2)))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image5)
axs[0].set_title('Obraz oryginalny')
downscaled_image5_avg = downscale(image5, average, 3)
axs[1].imshow(downscaled_image5_avg)
axs[1].set_title('Zmniejszenie za pomocą average')
print("MSE for image5 with average downscale:", calculate_mse(image5, bez_int(downscaled_image5_avg, average, 3)))
downscaled_image5_nn = downscale(image5, pierwszego, 3)
axs[2].imshow(downscaled_image5_nn)
axs[2].set_title('Zmniejszenie za pomocą pierwszego')
print("MSE for image5 with pierwszy downscale:", calculate_mse(image5, bez_int(downscaled_image5_nn, nearest_neighbour, 3)))"""
"""
"""
"""fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(image1)
axs[1].imshow(upscale(image1, h2, 16))
axs[1].set_title('Obraz przeskalowany 16 krotnie')
cztery = upscale(image1, h2, 4)
axs[2].imshow(upscale(cztery, h2, 4))
axs[2].set_title('Obraz przeskalowany 2 razy 4 krotnie')
dwarazydawa = upscale(image1, h2, 2)
dwarazydawa = upscale(dwarazydawa, h2, 2)
dwarazydawa = upscale(dwarazydawa, h2, 2)
axs[3].imshow(upscale(dwarazydawa, h2, 2))
axs[3].set_title('Obraz przeskalowany 4 razy 2 krotnie')    

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(image1)
axs[1].imshow(upscale(image1, h3, 16))
axs[1].set_title('Obraz przeskalowany 16 krotnie')
cztery = upscale(image1, h3, 4)
axs[2].imshow(upscale(cztery, h3, 4))
axs[2].set_title('Obraz przeskalowany 2 razy 4 krotnie')
dwarazydawa = upscale(image1, h3, 2)
dwarazydawa = upscale(dwarazydawa, h3, 2)
dwarazydawa = upscale(dwarazydawa, h3, 2)
axs[3].imshow(upscale(dwarazydawa, h3, 2))
axs[3].set_title('Obraz przeskalowany 4 razy 2 krotnie')"""



plt.show()