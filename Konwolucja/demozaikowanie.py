import matplotlib
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
from skimage.metrics import mean_squared_error
matplotlib.use('TkAgg')

#import obrazów
image1 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Obrazy/ghop2.jpg")
image2 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Obrazy/skull.jpg")
image3 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Obrazy/img_2.jpg")
#konwersja na skale szarości
gray = color.rgb2gray(image2)
gray2= color.rgb2gray(image3)
#kernel liniowy do interpolacji
def h3(original_x, new_x, width):
    t = (original_x - new_x) / width
    if 1 - abs(t) > 0:
        return 1 - abs(t)
    return 0
#filtr bayera
def bayer_filter():
    mask = [ [[0, 1, 0], [1, 0, 0]],             [[0, 0, 1], [0, 1, 0]]]
    return mask
#filtr fuji
def fuji_filter():
    mask = [ [[0, 1, 0], [1, 0, 0], [0, 0, 1],[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 1, 0], [0, 1, 0],[0, 0, 1], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 1], [0, 1, 0], [0, 1, 0],[1, 0, 0], [0, 1, 0], [0, 1, 0]],
             [[0, 1, 0], [1, 0, 0], [0, 0, 1],[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 1, 0], [0, 1, 0],[0, 0, 1], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 1], [0, 1, 0], [0, 1, 0],[1, 0, 0], [0, 1, 0], [0, 1, 0]]
    ]

    return mask
#kernele do wykrywania krawędzi
def sobel_x():
    mask = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]
    return mask
def sobel_y():
    mask = [[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
    return mask
def laplace_filter():
    laplace_filter = np.array(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]]
    )
    return laplace_filter
def scharr_x():
    mask = [[47, 0, -47],
            [162, 0, -162],
            [47, 0, -162]]
    return mask
def scharr_y():
    mask = [[47, 162, 47],
            [0, 0, 0],
            [-47, -162, -47]]
    return mask
def scharr_x2():
    mask = [[3, 0, -3],
            [10, 0, -10],
            [3, 0, -3]]
    return mask
def scharr_y2():
    mask = [[3, 10, 3],
            [0, 0, 0],
            [-3, -10, -3]]
    return mask
#Kernele gaussowskie
def gaussian_blur_3x3():
    mask = [[1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]]
    return mask / np.sum(mask)
def gaussian_blur_5x5():
    mask = [[1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]]
    return mask / np.sum(mask)

def gaussian_blur_7x7():
    mask = [[1, 6, 15, 20, 15, 6, 1],
            [6, 36, 90, 120, 90, 36, 6],
            [15, 90, 225, 300, 225, 90, 15],
            [20, 120, 300, 400, 300, 120, 20],
            [15, 90, 225, 300, 225, 90, 15],
            [6, 36, 90, 120, 90, 36, 6],
            [1, 6, 15, 20, 15, 6, 1]]
    return mask / np.sum(mask)
#kernel uśredniający
def blur_kernel():
    mask = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    return mask / np.sum(mask)
#kernel wyostrajaący
def W():
    mask = [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]]
    return mask
#kernel do demozaikowania filtra bayer
def average(weight):
    mask = [[0, 1*weight, 0],
            [1*weight, 4*weight, 1*weight],
            [0, 1*weight, 0]]
    return mask
#funkcja przekształcająca obraz rgb na obraz odpowiadający powstałemu z użyciem filtra bayera
def filtr(image, kernel):
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    #utworzenie nowego obrazu
    new_image = np.zeros_like(image)
    #iteracja poprzez obraz i użycie maski
    for i in range(0, height, kernel_size):
        for j in range(0, width, kernel_size):
            new_image[i:i+kernel_size, j:j+kernel_size] = image[i:i+kernel_size, j:j+kernel_size] * mask
    return new_image

#konwolucja wykorzstująca operator sobela-feldmana
def sobel_feldman(image,padding):
    #zastosowanie paddingu
    new_image = np.pad(image, ((padding, padding), (padding, padding)), mode="edge")
    height, width = image.shape
    #utworzenie tablic na gradienty
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)
    gradient_direction = np.zeros_like(image, dtype=np.float32)
    #iteracja poprzez każdy piksel obrazu i użycie maski
    for i in range(height):
        for j in range(width):
            values = new_image[i:i+3, j:j+3]
            gx = np.sum(values * sobel_x())
            gy = np.sum(values * sobel_y())
            #policzenie gradientów
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    new_image = np.clip((gradient_magnitude/gradient_direction)*255, 0, 255).astype(np.uint8)
    return [new_image, gradient_magnitude, gradient_direction]
#konwolucja wykorzstująca operator scharr'a
def scharr(image,padding):
    new_image = np.pad(image, ((padding, padding), (padding, padding)), mode="edge")
    height, width = image.shape
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)
    gradient_direction = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            values = new_image[i:i+3, j:j+3]
            gx = np.sum(values * scharr_x())
            gy = np.sum(values * scharr_y())
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    new_image = np.clip((gradient_magnitude/gradient_direction)*255, 0, 255).astype(np.uint8)
    return [new_image, gradient_magnitude, gradient_direction]

def scharr2(image,padding):
    new_image = np.pad(image, ((padding, padding), (padding, padding)), mode="edge")
    height, width = image.shape
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)
    gradient_direction = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            values = new_image[i:i+3, j:j+3]
            gx = np.sum(values * scharr_x2())
            gy = np.sum(values * scharr_y2())
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    new_image = np.clip((gradient_magnitude/gradient_direction)*255, 0, 255).astype(np.uint8)
    return [new_image, gradient_magnitude, gradient_direction]


#funkcja interpolująca
def interpolate(x_vals, y_vals, x_interp, kernel):
    y_interp = np.zeros(len(x_interp))
    for j in range(len(x_interp)):
        contribution = 0
        total_weight = 0
        for i in range(len(x_vals)):
            if i < len(x_vals) - 1:
                width = x_vals[i + 1] - x_vals[i]
            else:
                width = 1
            weight = kernel(x_vals[i], x_interp[j], width)
            contribution += y_vals[i] * weight
            total_weight += weight
        if total_weight > 0:
            y_interp[j] = contribution / total_weight
    return y_interp

#interpolacja kanałów zawierających wartości rgb
def interpolate_channel(channel,height,width,kernel):
    #utworzenie kopii kanału
    new_channel =  channel.copy()
    for row in range(height):
        #znalezienie wartości większych od zera w wierszu
        x_vals = np.where(channel[row, :] > 0)[0]
        if len(x_vals) > 1:
            y_vals = channel[row, x_vals]
            x_interp = np.arange(width)
            #interpolacja
            new_channel[row, :] = interpolate(x_vals, y_vals, x_interp, kernel)

    for col in range(width):
        #znalezienie wartości większych od zera w kolumnie
        y_vals = new_channel[:, col]
        x_vals = np.where(y_vals > 0)[0]
        if len(x_vals) > 1:
            #interpolacja
            y_interp = interpolate(x_vals, y_vals[x_vals], np.arange(height), kernel)
            new_channel[:, col] = y_interp
    return new_channel

#interpolacja kanałów zawierających wartości rgb tylko w kolumnach
def col_interpolate(channel,height,width,kernel):
    #utworzenie kopii kanału
    new_channel = channel.copy()
    for col in range(width):
        #znalezienie wartości większych od zera w kolumnie
        y_vals = new_channel[:, col]
        x_vals = np.where(y_vals > 0)[0]
        if len(x_vals) > 1:
            #interpolacja
            y_interp = interpolate(x_vals, y_vals[x_vals], np.arange(height), kernel)
            new_channel[:, col] = y_interp
    return new_channel
#demozajkowanie obrazu z zastosowanym filtrem bayer
def demosaic_bayer_interpolate(image, kernel):
    #utworzenie kanałów
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    #wypełnienie czerwonego kanału
    R[0::2, 1::2] = image[0::2, 1::2, 0]
    #wypełnienie niebieskiego kanału
    B[1::2, 0::2] = image[1::2, 0::2, 2]
    #wypełnienie zielonego kanału
    G[0::2, 0::2] = image[0::2, 0::2, 1]
    G[1::2, 1::2] = image[1::2, 1::2, 1]
    #interpolacja koloru czerwonego
    R = interpolate_channel(R,height,width,kernel)
    #interpolacja koloru zielonego
    G = col_interpolate(G,height,width,kernel)
    #interpolacja koloru niebieskiego
    B = interpolate_channel(B,height,width,kernel)
    #złożenie wszystkich kanałów
    demosaiced_image = np.stack((R, G, B), axis=-1)
    return np.clip(demosaiced_image, 0, 255).astype(np.uint8)

def demosaic_fuji(image, kernel):
    #utworzenie kanałów
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    #wypełnienie czerwonego kanału
    R[2::3, 0::3] = image[2::3, 0::3, 0]
    R[0::3, 1::3] = image[0::3, 1::3, 0]
    #wypełnienie niebieskiego kanału
    B[1::3, 0::3] = image[1::3, 0::3, 2]
    B[0::3, 2::3] = image[0::3, 2::3, 2]
    #wypełnienie zielonego kanału
    G[0::3, 0::3] = image[0::3, 0::3, 1]
    G[1::3, 1::3] = image[1::3, 1::3, 1]
    G[2::3, 1::3] = image[2::3, 1::3, 1]
    G[2::3, 2::3] = image[2::3, 2::3, 1]
    G[1::3, 2::3] = image[1::3, 2::3, 1]
    #interpolacja koloru czerwonego
    R = interpolate_channel(R,height,width,kernel)
    #interpolacja koloru zielonego
    G = col_interpolate(G,height,width,kernel)
    #interpolacja koloru niebieskiego
    B = interpolate_channel(B,height,width,kernel)
    #złożenie wszystkich kanałów
    new_image = np.stack((R, G, B), axis=-1)
    return np.clip(new_image, 0, 255).astype(np.uint8)

def demosaic_bayer_convolve(image, kernel, padding):
    #utworzenie kanałów
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    #wypełnienie czerwonego kanału
    R[0::2, 1::2] = image[0::2, 1::2, 0]
    #wypełnienie niebieskiego kanału
    B[1::2, 0::2] = image[1::2, 0::2, 2]
    #wypełnienie zielonego kanału
    G[0::2, 0::2] = image[0::2, 0::2, 1]
    G[1::2, 1::2] = image[1::2, 1::2, 1]
    #konwolucja kanału czerwonego z kernelem
    R = convolve(R, kernel(4), padding)
    #konwolucja kanału zielonego z kernelem
    G = convolve(G, kernel(2), padding)
    #konwolucja kanału niebieskiego z kernelem
    B = convolve(B, kernel(4), padding)
    #złożenie wszystkich kanałów
    new_image = np.stack((R, G, B), axis=-1)
    return np.clip(new_image, 0, 255).astype(np.uint8)

def convolve(image, kernel, padding):
    #zastosowanie paddingu
    image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    kernel = np.array(kernel, dtype=np.float32)
    kernel_size = kernel.shape[0]
    height, width = image.shape
    new_image = np.zeros((height - 2 * padding, width - 2 * padding), dtype=np.float32)
    #iteracja poprzez obraz i użycie kernela
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            region = image[i:i + kernel_size, j:j + kernel_size]
            new_image[i, j] = np.sum(region * kernel) / 8
    return new_image
def blur(image, kernel, padding):
    #zastosowanie paddingu
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    new_image = np.zeros_like(image)
    #iteracja poprzez obraz i użycie kernela
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            for c in range(channel):
                values = image[i:i+kernel_size, j:j+kernel_size, c]
                new_image[i, j, c] = np.sum(values * mask)
    return new_image
def sharpen(image, kernel, padding):
    #zastosowanie paddingu
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    new_image = np.zeros_like(image,dtype=np.float32)
    #iteracja poprzez obraz i użycie kernela
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            for c in range(channel):
                values = image[i:i+kernel_size, j:j+kernel_size, c]
                new_image[i, j, c] = np.sum(values * mask)
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image
def edge(image, kernel, padding):
    #zastosowanie paddingu
    image = np.pad(image, padding, mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width = image.shape
    new_image = np.zeros_like(image)
    #iteracja poprzez obraz i użycie kernela
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            values = image[i:i+kernel_size, j:j+kernel_size]
            new_image[i, j] = np.sum(values * mask)
    new_image = np.clip(new_image, 0, 1)
    return new_image

wyostrzony = sharpen(image3,W,1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image3,cmap="gray")
plt.title("Orginał")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(wyostrzony)
plt.title("Wyostrzony")

gauss = blur(image3,gaussian_blur_3x3,1)
gauss5 = blur(image3,gaussian_blur_5x5,2)
gauss7 = blur(image3,gaussian_blur_7x7,3)
normal_blur = blur(image3,blur_kernel,1)
plt.figure(figsize=(15, 5))
plt.imshow(image3)
plt.title("Orginał")
plt.axis("off")
plt.figure(figsize=(15, 5))
plt.imshow(normal_blur)
plt.title("Blur 3x3")
plt.axis("off")
gauss7 = blur(image3,gaussian_blur_7x7,3)
normal_blur = blur(image3,blur_kernel,1)
plt.figure(figsize=(15, 5))
plt.imshow(image3)
plt.title("Orginał")
plt.axis("off")
plt.figure(figsize=(15, 5))
plt.imshow(normal_blur)
plt.title("Blur 3x3")
plt.axis("off")

plt.figure(figsize=(15, 5))
plt.imshow(gauss)
plt.title("Gaussian Blur 3x3")
plt.axis("off")
plt.figure(figsize=(15, 5))
plt.imshow(gauss5)
plt.title("Gaussian Blur 5x5")
plt.axis("off")
plt.figure(figsize=(15, 5))
plt.imshow(gauss7)
plt.title("Gaussian Blur 7x7")
plt.axis("off")

sobel_fel = sobel_feldman(gray2,padding=1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sobel_fel[0],cmap='gray')
plt.title("Sobel Feldman")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(sobel_fel[1],cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(sobel_fel[2],cmap='gray')
plt.title("Gradient Direction")
plt.axis("off")

scharr = scharr(gray2,padding=1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(scharr[0],cmap='gray')
plt.title("Schar 47 , 162 , 47 ")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(scharr[1],cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(scharr[2],cmap='gray')
plt.title("Gradient Direction")
plt.axis("off")

scharr = scharr2(gray2,padding=1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(scharr[0],cmap='gray')
plt.title("Scharr 3,10,3")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(scharr[1],cmap='gray')
plt.title("Gradient Magnitude")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(scharr[2],cmap='gray')
plt.title("Gradient Direction")
plt.axis("off")

edge_x = edge(gray,sobel_x,padding=1)
edge_y = edge(gray,sobel_y,padding=1)
laplace = edge(gray,laplace_filter,padding=1)
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image2,cmap='gray')
plt.axis("off")
plt.title("Orginał")
plt.subplot(1,4,2)
plt.imshow(edge_x,cmap='gray')
plt.title("Sobel X")
plt.axis("off")
plt.subplot(1,4,3)
plt.imshow(edge_y,cmap='gray')
plt.title("Sobel Y")
plt.axis("off")
plt.subplot(1,4,4)
plt.imshow(laplace,cmap='gray')
plt.title("Laplace")
plt.axis("off")
"""
"""

image_bayer = filtr(image1,bayer_filter)
demosaiced_bayer1 = demosaic_bayer_convolve(image_bayer,average,0)
demosaiced_bayer2 = demosaic_bayer_interpolate(image_bayer,h3)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_bayer)
plt.title("Filtr Bayer")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(demosaiced_bayer1)
print("MSE for image5 with pierwszy downscale:", mean_squared_error(image1, demosaiced_bayer1))
plt.title("Bayer zdemozajkowany konwolucją")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(demosaiced_bayer2)
plt.title("Bayer zdemozajkowany interpolacją")
print("MSE for image5 with pierwszy downscale:", mean_squared_error(image1, demosaiced_bayer2))
plt.axis("off")


image_fuji = filtr(image1,fuji_filter)
demosaiced_image_2 = demosaic_fuji(image_fuji,h3)
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_fuji)
plt.title("Filtr Fuji")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(demosaiced_image_2)
plt.title("Fuji zdemozajkowny")
plt.axis("off")

plt.figure(figsize=(15, 5))
plt.imshow(demosaiced_image_2)
plt.figure(figsize=(15, 5))
plt.imshow(demosaiced_bayer2)

plt.show()