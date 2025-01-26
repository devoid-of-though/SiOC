import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

matplotlib.use('TkAgg')

image1 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Fourier/circle-noised.png")
image2 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Fourier/namib-noised.png")

def denoise_image_with_fourier(image):
    cutoff = 0.1
    after_fourier = np.fft.fft2(image)
    after_fourier_shifted = np.fft.fftshift(after_fourier)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(cutoff * min(rows, cols))
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol) ** 2 + (y - crow) ** 2 <= radius ** 2
    filtered_fourier = after_fourier_shifted * mask
    filtered_fourier_unshifted = np.fft.ifftshift(filtered_fourier)
    denoised_image = np.fft.ifft2(filtered_fourier_unshifted)
    return np.abs(denoised_image)

def denoise_image_with_cosine(image):
    cutoff = 200
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')
    dct_image[np.abs(dct_image) < cutoff] = 0
    denoised_image = idct(idct(dct_image.T, norm='ortho').T, norm='ortho')
    return np.clip(denoised_image, 0, 255).astype(np.uint8)

def denoise_image_color(image):
    denoised_image = np.zeros_like(image, dtype=np.float64)
    for i in range(image.shape[2]):
        denoised_image[:, :, i] = denoise_image_with_fourier(image[:, :, i])
    return np.clip(denoised_image, 0, 255).astype(np.uint8)

denoised_image1 = denoise_image_with_fourier(image1)
denoised_image2 = denoise_image_color(image2)
denoised_image3 = denoise_image_with_cosine(image1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title("Orginal Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(denoised_image1, cmap='gray')
plt.title("Denoised by FFT")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised_image3, cmap='gray')
plt.title("Denoised BY DCT")
plt.axis("off")

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Orginał")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(denoised_image2)
plt.title("Denoised by FFT")
plt.axis("off")


plt.show()
