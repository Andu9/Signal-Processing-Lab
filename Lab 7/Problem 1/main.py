import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 1, 100)
n1 = n.reshape(-1, 1)
n2 = n.reshape(1, -1)

def show(x, title, nameOfFile):
    y = np.fft.fft2(x)
    freq = 20 * np.log10(np.abs(y) + 1e-12)

    fig, axs = plt.subplots(1, 2, figsize = (10, 8))
    fig.suptitle(title)

    axs[0].imshow(x, cmap = 'gray')
    axs[0].set_title('Image')

    axs[1].imshow(freq)
    axs[1].set_title('Spectrum')

    fig.tight_layout()
    fig.colorbar(axs[1].images[0], ax = axs[1])
    fig.savefig(nameOfFile)
    fig.show()

def showInverse(y, title, nameOfFile):
    x = np.real(np.fft.ifft2(y))
    freq = 20 * np.log10(np.abs(y) + 1e-12)

    fig, axs = plt.subplots(1, 2, figsize = (10, 8))
    fig.suptitle(title)

    axs[0].imshow(x, cmap = 'gray')
    axs[0].set_title('Image')

    axs[1].imshow(freq)

    axs[1].set_title('Spectrum')

    fig.tight_layout()
    fig.colorbar(axs[1].images[0], ax = axs[1])
    fig.savefig(nameOfFile)
    fig.show()

x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
show(x1, r'$x_1(n_1, n_2) = \sin(2 \pi n_1 + 3 \pi n_2)$', '1a.pdf')

x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
show(x2, r'$x_2(n_1, n_2) = \sin(4 \pi n_1) + \cos(6 \pi n_2)$', '1b.pdf')

Y = np.zeros((len(n), len(n)))
Y[0, 5] = Y[0, len(n) - 5] = 1
showInverse(Y, r'y(0, 5) = y(0, N - 5) = 1', '1c.pdf')

Y = np.zeros((len(n), len(n)))
Y[5, 0] = Y[len(n) - 5, 0] = 1
showInverse(Y, r'y(5, 0) = y(N - 5, 0) = 1', '1d.pdf')

Y = np.zeros((len(n), len(n)))
Y[5, 5] = Y[len(n) - 5, len(n) - 5] = 1
showInverse(Y, r'y(5, 5) = y(N - 5, N - 5) = 1', '1e.pdf')