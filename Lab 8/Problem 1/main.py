import numpy as np
import matplotlib.pyplot as plt

N = 1000
n = np.linspace(0, N - 1, N)

a2, a1, a0 = 1e-7, 1e-4, 0.0
trend = a2 * n ** 2 + a1 * n + a0

A1, A2 = 1.0, 0.5
f1, f2 = 1 / 50, 1 / 100
season = A1 * np.sin(2 * np.pi * f1 * n) + A2 * np.sin(2 * np.pi * f2 * n)

sigma = 0.3
noise = sigma * np.random.randn(N)

y = trend + season + noise

def solveA():
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex = True)
    fig.suptitle("Time Series")

    axs[0].plot(n, y)
    axs[0].set_title("Full series")

    axs[1].plot(n, trend)
    axs[1].set_title("Trend")

    axs[2].plot(n, season)
    axs[2].set_title("Seasonality")

    axs[3].plot(n, noise)
    axs[3].set_title("Noise")
    axs[3].set_xlabel("time")

    fig.tight_layout()
    fig.savefig('1a.pdf')
    fig.show()

def solveB():
    def autocorrectManual(x):
        x = np.asarray(x)
        x = x - x.mean()
        N = len(x)
        r = np.empty(N)
        for k in range(N):
            r[k] = np.dot(x[:N - k], x[k:]) / (N - k)
        return r / r[0]

    r = autocorrectManual(y)
    lags = np.linspace(0, N - 1, N)

    plt.plot(lags, r)
    plt.title('Autocorrelation of y')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    plt.savefig('1b.pdf')
    plt.show()

def fitLeastSquares(x, p):
    x = np.asarray(x)
    N = len(x)

    X = np.column_stack([x[p - 1 - k:N - 1 - k] for k in range(p)])
    targetY = x[p:]

    Xd = np.column_stack([np.ones(len(X)), X])
    coefficients, *_ = np.linalg.lstsq(Xd, targetY, rcond = None)

    b = coefficients[0]
    a = coefficients[1:]

    return a, b

def AR(x, a, b):
    p = len(a)
    N = len(x)
    y_hat = np.full(N, np.nan)

    for t in range(p, N):
        past = x[t - p:t][::-1]
        y_hat[t] = b + np.dot(a, past)

    return y_hat

def solveC():
    p = 10
    a_hat, b_hat = fitLeastSquares(y, p)
    y_pred = AR(y, a_hat, b_hat)

    plt.plot(n, y, label = 'original')
    plt.plot(n, y_pred, label = f'AR({p}) prediction')
    plt.title(f'AR({p}) in-sample prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1c.pdf')
    plt.show()

def recursiveForecast(train, p, m):
    a, b = fitLeastSquares(train, p)
    series = train.copy()
    preds = []

    for _ in range(m):
        past = series[-1:-p - 1:-1]
        nextY = b + np.dot(a, past)
        preds.append(nextY)
        series = np.append(series, nextY)

    return np.array(preds)

def MSE(x, y):
    return np.mean((x - y) ** 2)

def solveD():
    pVals = range(1, 21)
    mVals = [25, 50, 100]

    bestError = np.inf
    best = None

    for p in pVals:
        for m in mVals:
            train = y[:-m]
            test = y[-m:]
            preds = recursiveForecast(train, p, m)
            error = MSE(test, preds)
            if error < bestError:
                bestError = error
                best = (p, m)

    bestP, bestM = best
    print(f'Best model: p = {bestP}, m = {bestM}, MSE = {bestError:.4f}')

    train = y[:-bestM]
    test = y[-bestM:]
    preds = recursiveForecast(train, bestP, bestM)

    tTest = n[-bestM:]

    plt.plot(tTest, test, label = 'true')
    plt.plot(tTest, preds, label = 'predictions')
    plt.title(f'Final AR({bestP}) forecast, m = {bestM}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('1d.pdf')
    plt.show()


solveA()
solveB()
solveC()
solveD()