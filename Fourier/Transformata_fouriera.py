import numpy as np
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

matplotlib.use('TkAgg')

image1 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Fourier/circle-noised.png")
image2 = io.imread("/home/userbrigh/PycharmProjects/SiOC/Fourier/namib-noised.png")

def denoise_image_with_fourier(image):
    im_fft = np.fft.fft2(image)
    keep_fraction = 0.1
    im_fft2 = im_fft.copy()
    r, c = im_fft2.shape
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    im_new = np.fft.ifft2(im_fft2)
    return np.abs(im_new)
def denoise_image_with_cosine(image):
    keep_fraction = 0.1 
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')
    r, c = dct_image.shape
    dct_image[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    dct_image[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    denoised_image = idct(idct(dct_image.T, norm='ortho').T, norm='ortho')
    return np.abs(denoised_image)

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
