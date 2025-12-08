import numpy as np
from scipy import datasets
import matplotlib.pyplot as plt

x = datasets.face(gray=True)
y = np.fft.fft2(x)

SNR = 10
noise = np.sum(np.abs(y) ** 2) / (10 ** (SNR / 10))

flattenedY = np.abs(y).flatten()
sortedIndeces = np.argsort(flattenedY)

partialSums = np.cumulative_sum(flattenedY[sortedIndeces] ** 2)
k = np.searchsorted(partialSums, noise)
stopFreq = flattenedY[sortedIndeces[k]]

mask = (flattenedY >= stopFreq).reshape(y.shape)
y = y * mask
newX = np.real(np.fft.ifft2(y))

fig, axs= plt.subplots(1, 2)
fig.suptitle('Compression of racoon image')

axs[0].imshow(x, cmap = 'gray')
axs[0].set_title('Original image')
axs[0].axis('off')

axs[1].imshow(newX, cmap = 'gray')
axs[1].set_title('Compressed image')
axs[1].axis('off')

fig.tight_layout()
fig.savefig('2.pdf')
fig.show()