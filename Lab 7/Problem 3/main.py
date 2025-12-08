import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets, ndimage

x = datasets.face(gray = True)

noise = np.random.randint(-100, 101, size = x.shape)
noisyX = x + noise

powerSignal = np.sum(noisyX ** 2)
powerNoise = np.sum(noise ** 2)

originalSNR = 10 * np.log10(powerSignal / powerNoise)

lessNoisyX = ndimage.gaussian_filter(noisyX, sigma = 0.2)
lessNoisyX = np.where(lessNoisyX < 1e-12, 1e-12, lessNoisyX)

SNR = 10 * np.log10(np.sum(lessNoisyX ** 2) / np.sum((noisyX - lessNoisyX) ** 2))

fig, axs = plt.subplots(1, 2, figsize = (10, 8))
fig.suptitle('Reduced noise')

axs[0].imshow(noisyX, cmap = 'gray')
axs[0].set_title(f'Noisy Image SNR = {originalSNR:.2f}')
axs[0].axis('off')

axs[1].imshow(lessNoisyX, cmap = 'gray')
axs[1].set_title(f'Less Noisy Image SNR = {SNR:.2f}')
axs[1].axis('off')

fig.tight_layout()
fig.savefig('3.pdf')
fig.show()