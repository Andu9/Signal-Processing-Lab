import numpy as np
import matplotlib.pyplot as plt
import time

def DFT(signal):
    N = signal.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    FourierMatrix = np.exp(-2 * np.pi * 1j * k * n / N)
    return FourierMatrix @ signal

def FFT(signal):
    N = signal.shape[0]
    if N == 1:
        return signal
    if N % 2 != 0:
        return DFT(signal)

    signal_even = FFT(signal[::2])
    signal_odd = FFT(signal[1::2])
    a = np.exp(-2 * np.pi * 1j * np.arange(N) / N)
    first = signal_even + a[:N // 2] * signal_odd
    second = signal_odd - a[:N // 2] * signal_odd
    return np.concatenate([first, second])

def benchmark(sizes = (128, 256, 512, 1024, 2048, 4096, 8192)):
    DFTTimes = []
    FFTTimes = []
    NumPyFFTTimes = []

    rng = np.random.default_rng(0)

    for N in sizes:
        print('Starting with ', N)

        signal = rng.normal(size = N) + 1j * rng.normal(size = N)

        t0 = time.perf_counter()
        _ = DFT(np.asarray(signal, dtype = complex))
        t1 = time.perf_counter()
        DFTTimes.append(t1 - t0)

        t0 = time.perf_counter()
        _ = FFT(np.asarray(signal, dtype = complex))
        t1 = time.perf_counter()
        FFTTimes.append(t1 - t0)

        t0 = time.perf_counter()
        _ = np.fft.fft(signal)
        t1 = time.perf_counter()
        NumPyFFTTimes.append(t1 - t0)

        print('Ending with ', N)

    plt.plot(sizes, DFTTimes, label = 'DFT')
    plt.plot(sizes, FFTTimes, label = 'FFT')
    plt.plot(sizes, NumPyFFTTimes, label = 'NumPy FFT')
    plt.title('Execution Time Differences: DFT vs FFT vs NumPy FFT')
    plt.xlabel('N')
    plt.ylabel('Execution Time (s)')
    plt.xscale('log', base = 2)
    plt.yscale('log', base = 2)
    plt.tight_layout()
    plt.legend()
    plt.savefig('D.pdf')
    plt.show()

if __name__ == '__main__':
    benchmark()