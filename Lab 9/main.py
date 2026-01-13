import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def generateTimeSeries(N):
    time = np.linspace(0, N - 1, N)

    a2, a1, a0 = 1e-7, 1e-4, 0.0
    trend = a2 * time ** 2 + a1 * time + a0

    A1, A2 = 1.0, 0.5
    f1, f2 = 1 / 50, 1 / 100
    season = A1 * np.sin(2 * np.pi * f1 * time) + A2 * np.sin(2 * np.pi * f2 * time)

    sigma = 0.3
    noise = sigma * np.random.randn(N)

    signal = trend + season + noise

    return signal, time

def simpleExponentiationMean(yParam, nParam, alpha = 0.6, plot = True):
    s = np.zeros_like(yParam, dtype = np.float32)
    s[0] = yParam[0]
    for i in range(1, len(yParam)):
        s[i] = alpha * yParam[i] + (1 - alpha) * s[i - 1]

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label = 'Original')
        plt.plot(nParam, s, label = 'Mean')
        plt.title(fr'Exponential Moving Average $\alpha = {alpha}$')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('1 Simple.pdf')
        plt.show()

    return s

def bestAlphaExponentiationMean(yParam, nParam, plot = True):
    best = None
    grid = np.linspace(0.1, 0.99, 1000)
    for alpha in grid:
        s = simpleExponentiationMean(yParam, nParam, alpha, False)
        yhat = np.concatenate([[s[0]], s[:-1]])
        err = np.mean((yParam[1:] - yhat[1:]) ** 2)

        if best is None or err < best[0]:
            best = (err, alpha, s)

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label = 'Original')
        plt.plot(nParam, best[2], label = r'Best $\alpha$')
        plt.legend()
        plt.title(fr'Simple Exponential Moving Average with $\alpha$ = {best[1]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig('1 Best Alpha.pdf')
        plt.show()

def doubleExponentiationMean(yParam, nParam, alpha = 0.6, beta = 0.4, plot = True):
    l = np.zeros_like(yParam, dtype = np.float32)
    b = np.zeros_like(yParam, dtype = np.float32)

    l[0] = yParam[0]
    b[0] = yParam[1] - yParam[0]

    for i in range(1, len(yParam)):
        l[i] = alpha * yParam[i] + (1 - alpha) * (l[i - 1] + b[i - 1])
        b[i] = beta * (l[i] - l[i - 1]) + (1 - beta) * b[i - 1]

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label = 'Original')
        plt.plot(nParam, l, label = 'Double Exponential Moving Average')
        plt.title(fr'Double Exponential Moving Average $\alpha = {alpha}$, $\beta = {beta}$')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('1 Double Exponential.pdf')
        plt.show()

    return l, b

def bestAlphaBetaExponentiationMean(yParam, nParam, plot = True):
    grid = np.linspace(0.1, 0.99, 100)
    best = None

    for alpha in grid:
        for beta in grid:
            l, b = doubleExponentiationMean(yParam, nParam, alpha, beta, False)
            yhat = np.concatenate([[l[0]], l[:-1] + b[:-1]])
            err = np.mean((yParam[1:] - yhat[1:]) ** 2)
            if best is None or err < best[0]:
                best = (err, alpha, beta, yhat)

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label='Original')
        plt.plot(nParam, best[3], label=r'Best $\alpha$ and $\beta$')
        plt.legend()
        plt.title(fr'Double Exponential Moving Average with $\alpha$ = {best[1]}, $\beta$ = {best[2]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig('1 Best Alpha and Beta.pdf')
        plt.show()

def tripleExponentiationMean(yParam,  nParam, P = 50, alpha = 0.6, beta = 0.4, gamma = 0.2, plot = True):
    l = np.zeros_like(yParam, dtype = np.float32)
    b = np.zeros_like(yParam, dtype = np.float32)
    s = np.zeros_like(yParam, dtype = np.float32)

    l[0] = y[0]
    b[0] = (y[P] - y[0]) / P
    s[:P] = y[:P] - np.mean(y[:P])

    for i in range(1, len(yParam)):
        sp = s[i - P] if i - P >= 0 else s[i % P]
        l[i] = alpha * (yParam[i] - sp) + (1 - alpha) * (l[i - 1] + b[i - 1])
        b[i] = beta * (l[i] - l[i - 1]) + (1 - beta) * b[i - 1]
        s[i] = gamma * (y[i] - l[i]) + (1 - gamma) * (s[i - P] if i - P >= 0 else s[i % P])

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label = 'Original')
        plt.plot(nParam, l, label = 'Triple Exponential Moving Average')
        plt.title(fr'Triple Exponential Moving Average $\alpha = {alpha}$, $\beta$ = {beta}, $\gamma$ = {gamma}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig('1 Triple Exponential.pdf')
        plt.show()

    return l, b, s

def bestAlphaBetaGammaExponentiationMean(yParam, nParam, P = 50, plot = True):
    best = None
    grid = np.linspace(0.1, 0.99, 10)

    for alpha in grid:
        for beta in grid:
            for gamma in grid:
                l, b, s = tripleExponentiationMean(yParam, nParam, P, alpha, beta, gamma, False)

                yhat = np.zeros(len(yParam), dtype = np.float32)
                yhat[0] = y[0]
                for i in range(1, len(yhat)):
                    sp = s[i - P] if i - P >= 0 else s[i % P]
                    yhat[i] = l[i - 1] + b[i - 1] + sp

                err = np.mean((yParam[1:] - yhat[1:]) ** 2)

                if best is None or err < best[0]:
                    best = (err, alpha, beta, gamma, yhat)

    if plot:
        plt.figure(figsize = (10, 6))
        plt.plot(nParam, yParam, label='Original')
        plt.plot(nParam, best[4], label=r'Best $\alpha$ and $\beta$ and $\gamma$')
        plt.legend()
        plt.title(fr'Triple Exponential Moving Average with $\alpha$ = {best[1]}, $\beta$ = {best[2]}, $\gamma$ = {best[3]}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.savefig('1 Best Alpha and Beta and Gamma.pdf')
        plt.show()

def rollingMean(yParam, q):
    m = np.full(len(yParam), np.nan)
    for i in range(q, len(yParam)):
        m[i] = np.mean(yParam[i-q:i])
    return m

def fitMARegression(yParam, nParam, q):
    fig, axs = plt.subplots(len(q), figsize = (25, 5 * len(q)))
    fig.suptitle('MA Model')

    for axs, qValue in zip(axs, q):
        m = rollingMean(yParam, qValue)
        eps = yParam - m

        A = []
        b = []
        indexList = []

        for i in range(2 * qValue, len(yParam)):
            if np.isnan(m[i]):
                continue

            A.append([eps[i - k] for k in range(1, qValue + 1)])
            b.append(yParam[i] - m[i])
            indexList.append(i)

        A = np.array(A)
        b = np.array(b)

        theta, *_ = np.linalg.lstsq(A, b, rcond = None)

        yPredictions = np.full(len(yParam), np.nan, dtype = np.float32)
        bPredictions = A @ theta

        for row, i in enumerate(indexList):
            yPredictions[i] = m[i] + bPredictions[row]

        valid = ~np.isnan(yPredictions)
        MAE = np.mean(np.abs(yParam[valid] - yPredictions[valid]))

        axs.plot(nParam, yParam, label = 'Original')
        axs.plot(nParam[valid], yPredictions[valid], label = f'MA Prediction q = {qValue}')
        axs.legend()
        axs.set_title(f'MA q = {qValue}, MAE = {MAE:.2f}')

    plt.savefig('2.pdf')
    plt.show()

def bestARIMA(yParam, nParam, pMax = 10, qMax = 10, step = 2):
    pValues = list(range(1, pMax + 1, step))
    qValues = list(range(1, qMax + 1, step))
    pairs = [(p, q) for p in pValues for q in qValues]

    bestAccuracy = np.inf
    bestP = None
    bestQ = None
    bestRes = None

    xTrain = yParam[:900]

    for p, q in pairs:
        try:
            print(f'p = {p}, q = {q}')
            model = sm.tsa.arima.ARIMA(xTrain, order = (p, 0, q))
            res = model.fit()

            if res.aic < bestAccuracy:
                bestAccuracy = res.aic
                bestP = p
                bestQ = q
                bestRes = res
        except:
            print(f'error p = {p}, q = {q}')

    pred = bestRes.forecast(steps = 1)

    plt.figure(figsize = (10, 6))
    plt.plot(nParam[:900], yParam[:900], label = 'Original')
    plt.scatter(nParam[900], yParam[900], label = 'Real Future Values')
    plt.scatter(nParam[900], pred, label = 'Predicted Values')
    plt.title(f'ARIMA(p = {bestP}, q = {bestQ})')
    plt.legend()
    plt.savefig('3.pdf')
    plt.show()

if __name__ == "__main__":
    y, n = generateTimeSeries(1000)
    # simpleExponentiationMean(y, n)
    # bestAlphaExponentiationMean(y, n)
    #
    # doubleExponentiationMean(y, n)
    # bestAlphaBetaExponentiationMean(y, n)
    #
    # tripleExponentiationMean(y, n)
    # bestAlphaBetaGammaExponentiationMean(y, n)

    # fitMARegression(y, n, [10, 20, 50])
    bestARIMA(y, n)



