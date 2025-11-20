import numpy as np

def solve():
    N = 20
    freq = 7
    time = np.linspace(0, 1, N, endpoint = False)
    x = np.sin(2 * np.pi * freq * time)

    d = 5
    y = np.roll(x, d)

    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    z1 = np.fft.ifft(X.conj() * Y)
    index1 = np.argmax(np.abs(z1))

    z2 = np.fft.ifft(Y / X)
    index2 = np.argmax(np.abs(z2))

    print('Original d =', d)
    print('Index 1 =', index1)
    print('Index 2 =', index2)

if __name__ == '__main__':
    solve()