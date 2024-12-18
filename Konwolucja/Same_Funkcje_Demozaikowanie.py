import matplotlib
import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np
from skimage.metrics import mean_squared_error
matplotlib.use('TkAgg')


image1 = io.imread("/hopathme/userbrigh/PycharmProjects/SiOC/Obrazy/ghop2.jpg")
def h3(original_x, new_x, width):
    t = (original_x - new_x) / width
    if 1 - abs(t) > 0:
        return 1 - abs(t)
    return 0
def bayer_filter():
    mask = [ [[0, 1, 0], [1, 0, 0]],             [[0, 0, 1], [0, 1, 0]]]
    return mask
def fuji_filter():
    mask = [ [[0, 1, 0], [1, 0, 0], [0, 0, 1],[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 1, 0], [0, 1, 0],[0, 0, 1], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 1], [0, 1, 0], [0, 1, 0],[1, 0, 0], [0, 1, 0], [0, 1, 0]],
             [[0, 1, 0], [1, 0, 0], [0, 0, 1],[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 1, 0], [0, 1, 0],[0, 0, 1], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 1], [0, 1, 0], [0, 1, 0],[1, 0, 0], [0, 1, 0], [0, 1, 0]]
             ]

    return mask

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
def blur_kernel():
    mask = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    return mask / np.sum(mask)
def W():
    mask = [[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]]
    return mask
def average(weight):
    mask = [[0, 1*weight, 0],
            [1*weight, 4*weight, 1*weight],
            [0, 1*weight, 0]]
    return mask
def filtr(image, kernel):
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    new_image = np.zeros_like(image)
    for i in range(0, height, kernel_size):
        for j in range(0, width, kernel_size):
            new_image[i:i+kernel_size, j:j+kernel_size] = image[i:i+kernel_size, j:j+kernel_size] * mask
    return new_image
def sobel_feldman(image,padding):
    new_image = np.pad(image, ((padding, padding), (padding, padding)), mode="edge")
    height, width = image.shape
    gradient_magnitude = np.zeros_like(image, dtype=np.float32)
    gradient_direction = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            values = new_image[i:i+3, j:j+3]
            gx = np.sum(values * sobel_x())
            gy = np.sum(values * sobel_y())
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    new_image = np.clip((gradient_magnitude/gradient_direction)*255, 0, 255).astype(np.uint8)
    return [new_image, gradient_magnitude, gradient_direction]
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

def interpolate_channel(channel,height,width,kernel):
    interp_channel = channel.copy()
    for row in range(height):
        x_vals = np.where(channel[row, :] > 0)[0]
        if len(x_vals) > 1:
            y_vals = channel[row, x_vals]
            x_interp = np.arange(width)
            interp_channel[row, :] = interpolate(x_vals, y_vals, x_interp, kernel)

    for col in range(width):
        y_vals = interp_channel[:, col]
        x_vals = np.where(y_vals > 0)[0]
        if len(x_vals) > 1:
            y_interp = interpolate(x_vals, y_vals[x_vals], np.arange(height), kernel)
            interp_channel[:, col] = y_interp
    return interp_channel
def col_interpolate(channel,height,width,kernel):
    interp_channel = channel.copy()
    for col in range(width):
        y_vals = interp_channel[:, col]
        x_vals = np.where(y_vals > 0)[0]
        if len(x_vals) > 1:
            y_interp = interpolate(x_vals, y_vals[x_vals], np.arange(height), kernel)
            interp_channel[:, col] = y_interp
    return interp_channel

def demosaic_bayer_interpolate(image, kernel):
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    R[0::2, 1::2] = image[0::2, 1::2, 0]
    B[1::2, 0::2] = image[1::2, 0::2, 2]
    G[0::2, 0::2] = image[0::2, 0::2, 1]
    G[1::2, 1::2] = image[1::2, 1::2, 1]
    R = interpolate_channel(R,height,width,kernel)
    G = col_interpolate(G,height,width,kernel)
    B = interpolate_channel(B,height,width,kernel)
    demosaiced_image = np.stack((R, G, B), axis=-1)
    return np.clip(demosaiced_image, 0, 255).astype(np.uint8)

def demosaic_fuji(image, kernel):
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    R[2::3, 0::3] = image[2::3, 0::3, 0]
    R[0::3, 1::3] = image[0::3, 1::3, 0]
    B[1::3, 0::3] = image[1::3, 0::3, 2]
    B[0::3, 2::3] = image[0::3, 2::3, 2]
    G[0::3, 0::3] = image[0::3, 0::3, 1]
    G[1::3, 1::3] = image[1::3, 1::3, 1]
    G[2::3, 1::3] = image[2::3, 1::3, 1]
    G[2::3, 2::3] = image[2::3, 2::3, 1]
    G[1::3, 2::3] = image[1::3, 2::3, 1]
    R = interpolate_channel(R,height,width,kernel)
    G = col_interpolate(G,height,width,kernel)
    B = interpolate_channel(B,height,width,kernel)
    new_image = np.stack((R, G, B), axis=-1)
    return np.clip(new_image, 0, 255).astype(np.uint8)

def demosaic_bayer_convolve(image, kernel, padding):
    height, width, _ = image.shape
    R = np.zeros((height, width))
    G = np.zeros((height, width))
    B = np.zeros((height, width))
    R[0::2, 1::2] = image[0::2, 1::2, 0]
    B[1::2, 0::2] = image[1::2, 0::2, 2]
    G[0::2, 0::2] = image[0::2, 0::2, 1]
    G[1::2, 1::2] = image[1::2, 1::2, 1]
    R = convolve(R, kernel(4), padding)
    G = convolve(G, kernel(2), padding)
    B = convolve(B, kernel(4), padding)
    new_image = np.stack((R, G, B), axis=-1)
    return np.clip(new_image, 0, 255).astype(np.uint8)

def convolve(image, kernel, padding):
    image = np.pad(image, ((padding, padding), (padding, padding)), mode="constant")
    kernel = np.array(kernel, dtype=np.float32)
    kernel_size = kernel.shape[0]
    height, width = image.shape
    new_image = np.zeros((height - 2 * padding, width - 2 * padding), dtype=np.float32)
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            region = image[i:i + kernel_size, j:j + kernel_size]
            new_image[i, j] = np.sum(region * kernel) / 8
    return new_image
def blur(image, kernel, padding):
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    new_image = np.zeros_like(image)

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            for c in range(channel):
                values = image[i:i+kernel_size, j:j+kernel_size, c]
                new_image[i, j, c] = np.sum(values * mask)
    return new_image
def sharpen(image, kernel, padding):
    image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width, channel = image.shape
    new_image = np.zeros_like(image,dtype=np.float32)
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            for c in range(channel):
                values = image[i:i+kernel_size, j:j+kernel_size, c]
                new_image[i, j, c] = np.sum(values * mask)
    new_image = np.clip(new_image, 0, 255).astype(np.uint8)
    return new_image
def edge(image, kernel, padding):
    image = np.pad(image, padding, mode="constant")
    mask = kernel()
    kernel_size = len(mask[0])
    height, width = image.shape
    new_image = np.zeros_like(image)
    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            values = image[i:i+kernel_size, j:j+kernel_size]
            new_image[i, j] = np.sum(values * mask)
    new_image = np.clip(new_image, 0, 1)
    return new_image