import numpy as np
import matplotlib.pyplot as plt

def solveA():
    time = np.linspace(0, 1 / 100, 1600)
    signal = np.sin(2 * np.pi * 400 * time)

    plt.plot(time, signal)
    plt.title('f = 400 Hz, 1600 samples')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('2a.pdf')
    plt.show()

def solveB():
    time = np.linspace(0, 3, 1600)
    signal = np.sin(2 * np.pi * 800 * time)

    plt.plot(time, signal)
    plt.title('f = 800 Hz, 3 seconds')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('2b.pdf')
    plt.show()

def solveC():
    freq = 240
    time = np.linspace(0, 5 / 240, 1000)
    signal = np.mod(time * freq, 1)

    plt.plot(time, signal)
    plt.title('Sawtooth signal, f = 240 Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('2c.pdf')
    plt.show()

def solveD():
    time = np.linspace(0, 3 / 300, 1000)
    signal = np.sign(np.sin(2 * np.pi * 300 * time))

    plt.plot(time, signal)
    plt.title('Square signal, f = 300 Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('2d.pdf')
    plt.show()

def solveE():
    arr = np.random.random((128, 128))

    plt.imshow(arr)
    plt.title('Random 2D signal')
    plt.axis('off')
    plt.savefig('2e.pdf')
    plt.show()

def solveF():
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    plt.imshow(R)
    plt.suptitle('2D Radial Signal')
    plt.axis('off')
    plt.savefig('2f.pdf')
    plt.show()

solveA(); solveB(); solveC()
solveD(); solveE(); solveF()