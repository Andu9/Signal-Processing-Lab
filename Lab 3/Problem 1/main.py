import numpy as np
import matplotlib.pyplot as plt

N = 8

fourierMatrix = np.empty([N, N], dtype = complex)

for i in range(N):
    for j in range(N):
        fourierMatrix[i][j] = np.exp(-2 * np.pi * 1j * i * j / N)

fig, axs = plt.subplots(N, 2)
fig.suptitle('Imaginary and Real graphics for the Fourier Matrix of order N = 8')
for i in range(N):
    axs[i, 0].stem(np.arange(N), [fourierMatrix[i, j].real for j in range(N)])
    axs[i, 1].stem(np.arange(N), [fourierMatrix[i, j].imag for j in range(N)])
fig.savefig('D.pdf')
fig.show()

FHF = np.conjugate(fourierMatrix.T) @ fourierMatrix
scaledIdentityMatrix = N * np.eye(N)

print(np.allclose(scaledIdentityMatrix, FHF))