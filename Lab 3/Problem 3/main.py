import numpy as np
import matplotlib.pyplot as plt

N  = 200
time  = np.linspace(0, 1, N, endpoint = False)
signal = np.sin(2 * np.pi * 5 * time) + 2 * np.sin(2 * np.pi * 20 * time) + 4 * np.sin(2 * np.pi * 9 * time)

FourierMatrix = np.empty([N, N], dtype = complex)
for i in range(N):
    for j in range(N):
        FourierMatrix[i, j] = np.exp(-2 * np.pi * 1j * i * j / N)

values = FourierMatrix @ signal

plt.plot(time, signal)
plt.title(r'Signal $x(t) = \sin(2\pi \cdot 5 \cdot t) + 2 \cdot \sin(2\pi \cdot 20 \cdot t) + 4 \cdot \sin(2\pi \cdot 9 \cdot t)$')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('Signal in time domain.pdf')
plt.show()

plt.stem(np.arange(N)[:N // 2], np.abs(values[:N // 2]))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k]|')
plt.savefig('Signal in frequency domain.pdf')
plt.show()