import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize
from skimage import io, color
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

image = resize(io.imread("/home/userbrigh/PycharmProjects/SiOC/Obrazy/Test.png"), (100, 100))

def sample_hold_kernel2d(x: NDArray, offset: NDArray, width: float) -> NDArray:
    """Sample and hold interpolation kernel"""
    offset = offset[np.newaxis, np.newaxis, :]
    x = x - offset
    x, y = x[:, :, 0], x[:, :, 1]
    return (x >= 0) * (x < width) * (y >= 0) * (y < width)

def nearest_neighbour_kernel2d(x: NDArray, offset: NDArray, width: float) -> NDArray:
    """Nearest neighbour interpolation kernel"""
    offset = offset[np.newaxis, np.newaxis, :]
    x = x - offset
    x, y = x[:, :, 0], x[:, :, 1]
    return (x >= (-1 * width / 2)) * (x < width / 2) * (y >= (-1 * width / 2)) * (y < width / 2)

def keys_kernel2d(x: NDArray, offset: NDArray, width: float, alpha: float = -0.5) -> NDArray:
    """
    Interpolation kernel given by Keys bi-cubic function extended to 2D

    :references:
        * https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        * http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
    """

    def xy_range(xs: NDArray, ys: NDArray, xlow: float, xhigh: float, ylow: float, yhigh: float) -> NDArray:
        return (xs >= xlow) * (xs < xhigh) * (ys >= ylow) * (ys < yhigh)
    offset = offset[np.newaxis, np.newaxis, :]
    x = x - offset
    x = x / width
    x, y = x[:, :, 0], x[:, :, 1]
    x = np.abs(x)
    y = np.abs(y)

    return (
            ((alpha + 2) * x**3 - (alpha + 3) * x**2 + 1)
            * ((alpha + 2) * y**3 - (alpha + 3) * y**2 + 1)
            * xy_range(x, y, xlow=0, xhigh=1, ylow=0, yhigh=1)
            + (alpha * x**3 - 5 * alpha * x**2 + 8 * alpha * x - 4 * alpha)
            * ((alpha + 2) * y**3 - (alpha + 3) * y**2 + 1)
            * xy_range(x, y, xlow=1, xhigh=2, ylow=0, yhigh=1)
            + (alpha * y**3 - 5 * alpha * y**2 + 8 * alpha * y - 4 * alpha)
            * ((alpha + 2) * x**3 - (alpha + 3) * x**2 + 1)
            * xy_range(x, y, xlow=0, xhigh=1, ylow=1, yhigh=2)
            + (alpha * x**3 - 5 * alpha * x**2 + 8 * alpha * x - 4 * alpha)
            * (alpha * y**3 - 5 * alpha * y**2 + 8 * alpha * y - 4 * alpha)
            * xy_range(x, y, xlow=1, xhigh=2, ylow=1, yhigh=2))

def linear_kernel2d(x: NDArray, offset: NDArray, width: float) -> NDArray:
    """Nearest neighbour interpolation kernel"""
    offset = offset[np.newaxis, np.newaxis, :]
    x = x - offset
    x = x / width
    x, y = x[:, :, 0], x[:, :, 1]

    return ((1 - np.abs(x)) * (1 - np.abs(y))) * (np.abs(x) < 1) * (np.abs(y) < 1)
#funkcja interpolacji
def image_interpolate2d(image: NDArray, ratio: int, kernel: callable) -> NDArray:
    w = ratio
    #wymiary zinterpolowanego obrazu
    new_image_shape = (image.shape[0] * ratio, image.shape[1] * ratio)
    #utworzenie siatki punktów po których będziemy iterować
    image_grid = np.zeros((image.shape[0], image.shape[1], 2), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_grid[i, j] = [i, j]
    #utworzenie siatki punktów do interpolacji
    interpolate_grid = np.zeros((new_image_shape[0], new_image_shape[1], 2), dtype=float)
    for i in range(new_image_shape[0]):
        for j in range(new_image_shape[1]):
            interpolate_grid[i, j] = [i / ratio, j / ratio]
    kernels = []
    #Zastosuowanie kernela dla każdego piksela
    for point, value in zip(image_grid.reshape(-1, 2), image.ravel()):
        ker = value * kernel(interpolate_grid, offset=point, width=w)
        kernels.append(ker.reshape(new_image_shape))
    return np.sum(np.asarray(kernels), axis=0)

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image,cmap='gray')
plt.title("image")
plt.subplot(1, 2, 2)
plt.imshow(image_interpolate2d(image, 2, linear_kernel2d),cmap='gray')
plt.title("Interpolated image")
plt.show()