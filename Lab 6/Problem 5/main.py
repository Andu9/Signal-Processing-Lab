import numpy as np
import matplotlib.pyplot as plt

def rectangularWindow(N):
    return np.ones(N)

def hanningWindow(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

def main():
    Nw = 200
    freq = 100.0
    Fs = 1000.0

    time = np.arange(Nw) / Fs
    x = np.sin(2 * np.pi * freq * time)

    rectWindow = rectangularWindow(Nw)
    hannWindow = hanningWindow(Nw)

    rectSignal = x * rectWindow
    hannSignal = x * hannWindow

    plt.subplot(3, 1, 1)
    plt.plot(time, x)
    plt.title('Sine wave')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, rectSignal)
    plt.title('Sinus with Rectangular Window')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, hannSignal)
    plt.title('Sinus with Hanning Window')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('5.pdf')
    plt.show()

if __name__ == '__main__':
    main()
