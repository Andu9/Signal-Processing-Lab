import numpy as np

def polynomialProduct(p, q):
    degP = len(p) - 1
    degQ = len(q) - 1

    r = np.zeros(degP + degQ + 1, dtype = float)
    for i in range(degP + 1):
        for j in range(degQ + 1):
            r[i + j] += p[i] * q[j]

    return r

def FFTPolynomialProduct(p, q):
    n = len(p) + len(q) - 1
    next2Power = 1
    while next2Power < n:
        next2Power *= 2

    P = np.fft.fft(p, next2Power)
    Q = np.fft.fft(q, next2Power)
    R = P * Q
    r = np.fft.ifft(R)
    return np.real_if_close(r[:n])

def solve():
    N = 5
    p = np.random.randint(-5, 6, size = N + 1)
    q = np.random.randint(-5, 6, size = N + 1)

    r1 = polynomialProduct(p, q)
    r2 = FFTPolynomialProduct(p, q)

    print('p(x) coefficients:', p)
    print('q(x) coefficients:', q)
    print('r(x) direct:', r1)
    print('r(x) via FFT:', r2)

if __name__ == '__main__':
    solve()