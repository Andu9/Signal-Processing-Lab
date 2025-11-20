import numpy as np
import matplotlib.pyplot as plt

B = 1.0
minT = -3.0
maxT = 3.0
FsValues = [1.0, 1.5, 2.0, 4.0]

def squaredSinc(BParam, t):
    return np.sinc(BParam * t) ** 2

def approx(time, sampleValues, Ts, sampleTimes):
    approximation = np.zeros_like(time, dtype=float)

    for x_n, t_n in zip(sampleValues, sampleTimes):
        approximation += x_n * np.sinc((time - t_n) / Ts)

    return approximation

def solve():
    time = np.linspace(minT, maxT, 1000)
    true_sinc = squaredSinc(B, time)

    fig, axs = plt.subplots(2, 2, figsize = (16, 9))
    axs = axs.ravel()

    for ax, Fs in zip(axs, FsValues):
        Ts = 1.0 / Fs

        n = np.arange(np.floor(minT / Ts), np.ceil(maxT / Ts))
        sampleTimes = n * Ts
        sampleValues = squaredSinc(B, sampleTimes)

        approximation = approx(time, sampleValues, Ts, sampleTimes)

        ax.plot(time, true_sinc)
        ax.plot(time, approximation, linestyle = '--')
        ax.stem(sampleTimes, sampleValues, basefmt = " ")

        ax.set_title(f'Fs = {Fs:.2f} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid()

    fig.tight_layout()
    fig.savefig('1.pdf')
    fig.show()

if __name__ == '__main__':
    solve()
