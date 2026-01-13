import numpy as np
import matplotlib.pyplot as plt

def generateTimeSeries(N = 1000):
    n = np.linspace(0, N - 1, N)

    a2, a1, a0 = 1e-7, 1e-4, 0.0
    trend = a2 * n ** 2 + a1 * n + a0

    A1, A2 = 1.0, 0.5
    f1, f2 = 1 / 50, 1 / 100
    season = A1 * np.sin(2 * np.pi * f1 * n) + A2 * np.sin(2 * np.pi * f2 * n)

    sigma = 0.3
    noise = sigma * np.random.randn(N)

    y = trend + season + noise

    return y, n

def predictARStep(signalParam, a):
    p = len(a)
    N = len(signalParam)

    yhat = np.full(N, np.nan, dtype = np.float32)
    for i in range(p, N):
        past = signalParam[i - p:i][::-1]
        yhat[i] = np.dot(a, past)
    return yhat

def ARModel(signalParam, timeParam, p):
    N = signalParam.shape[0]
    X = np.column_stack([signalParam[p - k - 1:N - k - 1] for k in range(p)])
    y = signalParam[p:]

    a, *_ = np.linalg.lstsq(X, y, rcond = None)

    predictions = predictARStep(signalParam, a)

    plt.figure(figsize = (10, 6))
    plt.plot(timeParam, signalParam, color = 'blue', label = 'Original')
    plt.plot(timeParam, predictions, color = 'red', label = 'Prediction')
    plt.title(f'AR({p}) LS one step prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('2.pdf')
    plt.show()

    return a.astype(np.float32)

def greedySparseAR(signalParam, timeParam, p, k):
    N = signalParam.shape[0]
    X = np.column_stack([signalParam[p - i - 1:N - i - 1] for i in range(p)])
    y = signalParam[p:]

    selected = []
    remaining = list(range(p))
    aFull = np.zeros(p, dtype = np.float32)

    r = y.copy()
    aSelected = None

    for _ in range(k):
        correlations = [abs(np.dot(X[:, j], r)) for j in remaining]
        bestRemainingIndex = int(np.argmax(correlations))
        jBest = remaining[bestRemainingIndex]

        selected.append(jBest)
        remaining.remove(jBest)

        Xs = X[:, selected]
        aSelected, *_ = np.linalg.lstsq(Xs, y, rcond = None)

        r = y - Xs @ aSelected

    aFull[selected] = aSelected.astype(np.float32)
    predictions = predictARStep(signalParam, aFull)

    plt.figure(figsize = (10, 6))
    plt.plot(timeParam, signalParam, color = 'blue', label = 'Original')
    plt.plot(timeParam, predictions, color = 'red', label = 'Prediction')
    plt.title(f'Greedy sparse AR({p}) LS one step prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('3 Greedy.pdf')
    plt.show()

    return aFull

def L1RegularizationSparseAR(signalParam, timeParam, p, lam = 0.8, iterations = 500):
    N = signalParam.shape[0]
    X = np.column_stack([signalParam[p - i - 1:N - i - 1] for i in range(p)])
    y = signalParam[p:]

    sigmaMax = np.linalg.norm(X, ord = 2)
    L = sigmaMax * sigmaMax + 1e-12

    a = np.zeros(p, dtype = np.float32)

    for _ in range(iterations):
        grad = X.T @ (X @ a - y)
        a = a - (1.0 / L) * grad
        a = (np.sign(a) * np.maximum(np.abs(a) - lam / L, 0.0)).astype(np.float32)

    predictions = predictARStep(signalParam, a)

    plt.figure(figsize = (10, 6))
    plt.plot(timeParam, signalParam, color = 'blue', label = 'Original')
    plt.plot(timeParam, predictions, color = 'red', label = 'Prediction')
    plt.title(f'L1 Regularized sparse AR({p}) LS one step prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('3 L1.pdf')
    plt.show()

    return a

def polynomialRootsCompanion(coefficients):
    coefficients = np.asarray(coefficients, dtype = np.complex128)
    N = coefficients.shape[0] - 1

    b = coefficients[1:] / coefficients[0]

    C = np.zeros((N, N), dtype = np.complex128)
    C[0, :] = -b
    C[1:, :-1] = np.eye(N - 1, dtype = np.complex128)

    roots = np.linalg.eigvals(C)

    return roots

def checkStationaryAR(a, title, filename):
    coefficients = np.concatenate(([1.0], -a.astype(np.float64)))
    roots = polynomialRootsCompanion(coefficients)

    stationary = np.all(np.abs(roots) > 1.0)

    theta = np.linspace(0, 2 * np.pi, 500)
    unitCircle = np.exp(1j * theta)

    plt.figure(figsize = (10, 6))
    plt.plot(unitCircle.real, unitCircle.imag, color = 'black', label = 'Unit Circle')
    plt.scatter(roots.real, roots.imag, color = 'red', label = 'Roots')
    plt.axhline(0, color = 'black')
    plt.axvline(0, color = 'black')
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.title(title)
    plt.xlabel('Real Axis')
    plt.ylabel('Imaginary Axis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    return stationary

if __name__ == '__main__':
    signal, time = generateTimeSeries()
    # ARModel(signal, time, 20)

    # greedySparseAR(signal, time, 20, 10)
    # L1RegularizationSparseAR(signal, time, 20)

    # polynomialRootsCompanion([1.0, 0.2, -0.5, 0.1, 0.2])

    checkStationaryAR(ARModel(signal, time, 20), f'LS AR({20}) roots', '5 LS Roots.pdf')
    checkStationaryAR(greedySparseAR(signal, time, 20, 10), f'Greedy AR({20}) roots', '5 Greedy Roots.pdf')
    checkStationaryAR(L1RegularizationSparseAR(signal, time, 20), f'L1 AR({20}) roots', '5 L1 Roots.pdf')