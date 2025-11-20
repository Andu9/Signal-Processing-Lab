import numpy as np
import matplotlib.pyplot as plt

def iteration(x, times = 3):
    signals = [x]
    current = x

    for _ in range(times):
        current = np.convolve(current, x)
        signals.append(current)

    return signals

def randomExperiment():
    x = np.random.randn(100)
    signals = iteration(x)

    for i, signal in enumerate(signals):
        plt.subplot(4, 1, i + 1)
        plt.plot(signal)
        plt.grid(True)
        if i == 0:
            plt.title('Iterated convolution of random signal')
        plt.ylabel(f'Iteration {i}')

    plt.xlabel('n')
    plt.tight_layout()
    plt.savefig('1 - Random.pdf')
    plt.show()

def squareExperiment():
    x = np.zeros(100)
    x[40:60] = 1.0

    signals = iteration(x)

    for i, signal in enumerate(signals):
        plt.subplot(4, 1, i + 1)
        plt.plot(signal)
        plt.grid()
        if i == 0:
            plt.title("Iterated convolution of square signal")
        plt.ylabel(f'Iteration {i}')

    plt.xlabel('n')
    plt.tight_layout()
    plt.savefig('2 - Square.pdf')
    plt.show()

if __name__ == '__main__':
    randomExperiment()
    squareExperiment()