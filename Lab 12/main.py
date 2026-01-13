import numpy as np
import matplotlib.pyplot as plt

def OneDimensionalGaussian(mu = 1.0, sigmaSquared = 2.0, N = 5000):
    sigma = np.sqrt(sigmaSquared)

    samples = np.random.normal(mu, sigma, N)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
    PDF = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    plt.figure()
    plt.hist(samples, bins = 50, density = True, alpha = 0.7, label = 'Samples')
    plt.plot(x, PDF, label = 'Theoretical PDF')
    plt.title('One Dimensional Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('One Dimensional.pdf')
    plt.show()

def TwoDimensionalGaussian(mu = np.array([0.0, 0.0]), sigma = np.array([[1.0, 0.5], [0.5, 1.0]]), N = 3000):
    L = np.linalg.cholesky(sigma)
    z = np.random.normal(0, 1, (N, 2))
    samples = mu + z @ L.T

    grid = np.linspace(-4.0, 4.0, 200)
    X, Y = np.meshgrid(grid, grid)

    invSigma = np.linalg.inv(sigma)
    detSigma = np.linalg.det(sigma)

    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            diff = np.array([X[i, j] - mu[0], Y[i, j] - mu[1]])
            exponent = -0.5 * diff.T @ invSigma @ diff
            Z[i, j] = (1 / (2 * np.pi * np.sqrt(detSigma))) * np.exp(exponent)

    plt.figure()
    plt.scatter(samples[:, 0], samples[:, 1], s = 8, alpha = 0.5, label = 'Samples')
    plt.contour(X, Y, Z, levels = 8)
    plt.title('Two Dimensional Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Two Dimensional.pdf')
    plt.show()

def BuildCovarianceMatrix(x, kernelFunction):
    N = len(x)
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = kernelFunction(x[i], x[j])
    return C

def SampleGaussianProcess(x, kernelFunction, N = 5):
    C = BuildCovarianceMatrix(x, kernelFunction)

    C = C + 1e-8 * np.eye(C.shape[0])
    L = np.linalg.cholesky(C)

    samples = []

    for _ in range(N):
        n = np.random.normal(0, 1, len(x))
        z = L @ n
        samples.append(z)

    return np.array(samples)

def LinearKernel(x, y):
    return x * y

def BrownianKernel(s, t):
    return min(s, t)

def SquaredExponentialKernel(x, y, alpha = 8.0):
    return np.exp(-alpha * (x - y) ** 2)

def OrnsteinUhlenbeckKernel(x, y, alpha = 3.0):
    return np.exp(-alpha * np.abs(x - y))

def PeriodicKernel(x, y, alpha = 2.0, beta = 3.0):
    return np.exp(-alpha * (np.sin(beta * np.pi * (x - y)) ** 2))

def SymmetricKernel(x, y, alpha = 8.0):
    m = min(abs(x - y), abs(x + y))
    return np.exp(-alpha * (m ** 2))

def solve2(N = 200, numberOfSamples = 5):
    x = np.linspace(-1.0, 1.0, N)

    samples = SampleGaussianProcess(x, LinearKernel, numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(x, samples[i])
    plt.title("GP - Linear kernel")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.tight_layout()
    plt.savefig("GP Linear.pdf")
    plt.show()

    t = np.linspace(0.0, 1.0, N)
    samples = SampleGaussianProcess(t, BrownianKernel, numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(t, samples[i])
    plt.title("GP - Brownian motion kernel (min{s,t})")
    plt.xlabel("t")
    plt.ylabel("z(t)")
    plt.tight_layout()
    plt.savefig("GP Brownian.pdf")
    plt.show()

    samples = SampleGaussianProcess(x, lambda a, b: SquaredExponentialKernel(a, b, alpha=8.0), numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(x, samples[i])
    plt.title("GP - Squared Exponential kernel")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.tight_layout()
    plt.savefig("GP Squared Exponential.pdf")
    plt.show()

    samples = SampleGaussianProcess(x, lambda a, b: OrnsteinUhlenbeckKernel(a, b, alpha=3.0), numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(x, samples[i])
    plt.title("GP - Ornstein-Uhlenbeck kernel")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.tight_layout()
    plt.savefig("GP Ornstein Uhlenbeck.pdf")
    plt.show()

    samples = SampleGaussianProcess(x, lambda a, b: PeriodicKernel(a, b, alpha=2.0, beta=3.0), numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(x, samples[i])
    plt.title("GP - Periodic kernel")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.tight_layout()
    plt.savefig("GP Periodic.pdf")
    plt.show()

    samples = SampleGaussianProcess(x, lambda a, b: SymmetricKernel(a, b, alpha = 8.0), numberOfSamples)
    plt.figure()
    for i in range(numberOfSamples):
        plt.plot(x, samples[i])
    plt.title("GP - Symmetric kernel")
    plt.xlabel("x")
    plt.ylabel("z(x)")
    plt.tight_layout()
    plt.savefig("GP Symmetric.pdf")
    plt.show()

def loadDataset(urlOnPath = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"):
    data = np.genfromtxt(urlOnPath, comments = '#')

    year = data[:, 0].astype(int)
    month = data[:, 1].astype(int)
    day = data[:, 2].astype(int)
    co2 = data[:, 4]

    mask = co2 > 0
    return year[mask], month[mask], day[mask], co2[mask]

def monthlyMeanSeries(year, month, values):
    ym = year * 100 + month
    uniqueYm = np.unique(ym)

    monthlyMean = np.zeros(len(uniqueYm))
    for i in range(len(uniqueYm)):
        mask = (ym == uniqueYm[i])
        monthlyMean[i] = np.mean(values[mask])
    return uniqueYm, monthlyMean

def solve3a():
    year, month, day, co2 = loadDataset()

    uniqueYm, monthlyMean = monthlyMeanSeries(year, month, co2)

    t = np.linspace(0.0, float(len(monthlyMean) - 1), len(monthlyMean))

    plt.figure()
    plt.plot(t, monthlyMean)
    plt.title('CO2 Monthly Mean')
    plt.xlabel('Time (months)')
    plt.ylabel('CO2 (ppm)')
    plt.tight_layout()
    plt.savefig('3a Monthly Mean.pdf')
    plt.show()

def linearTrend(t, y):
    A = np.column_stack((t, np.ones_like(t)))
    a, b = np.linalg.lstsq(A, y, rcond = None)[0]
    return a, b

def solve3b():
    year, month, day, co2 = loadDataset()
    uniqueYm, monthlyMean = monthlyMeanSeries(year, month, co2)

    t = np.linspace(0.0, float(len(monthlyMean) - 1), len(monthlyMean))

    a, b, = linearTrend(t, monthlyMean)
    trend = a * t + b

    detrended = monthlyMean - trend

    plt.figure()
    plt.plot(t, monthlyMean, label = 'Monthly mean CO2')
    plt.plot(t, trend, label = 'Linear trend')
    plt.title('Monthly Mean + Linear Trend')
    plt.xlabel('Time (months)')
    plt.ylabel('CO2 (ppm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('CO2 Trend.pdf')
    plt.show()

    plt.figure()
    plt.plot(t, detrended)
    plt.title('Detrended Monthly Series')
    plt.xlabel('Time (months')
    plt.ylabel('CO2 (ppm) - trend')
    plt.tight_layout()
    plt.savefig('CO2 Detrended.pdf')
    plt.show()

def RBFKernel(x, y, sigmaF = 1.0, ell = 18.0):
    return (sigmaF ** 2) * np.exp(-((x - y) ** 2) / (2.0 * (ell ** 2)))

def PeriodicKernelMonths(x, y, sigmaF = 0.7, ell = 2.0, period = 12.0):
    s = np.sin(np.pi * (x - y) / period)
    return (sigmaF ** 2) * np.exp(-2.0 * (s ** 2) / (ell ** 2))

def CombinedKernel(x, y):
    return RBFKernel(x, y, sigmaF = 1.0, ell = 18.0) + PeriodicKernelMonths(x, y, sigmaF = 0.7, ell = 2.0, period = 12.0)

def solve3c(trainFraction = 0.7, sigmaNoise = 0.8):
    year, month, day, co2 = loadDataset()
    uniqueYm, monthlyMean = monthlyMeanSeries(year, month, co2)

    t = np.linspace(0.0, float(len(monthlyMean) - 1), len(monthlyMean))

    a, b = linearTrend(t, monthlyMean)
    trend = a * t + b
    detrended = monthlyMean - trend

    splitIndex = int(trainFraction * len(t))

    xTrain = t[:splitIndex]
    yTrain = detrended[:splitIndex]

    xTest = t[splitIndex:]
    yTest = monthlyMean[splitIndex:]

    Kbb = BuildCovarianceMatrix(xTrain, CombinedKernel)
    Kaa = BuildCovarianceMatrix(xTest, CombinedKernel)

    Kab = np.zeros((len(xTest), len(xTrain)))
    for i in range(len(xTest)):
        for j in range(len(xTrain)):
            Kab[i, j] = CombinedKernel(xTest[i], xTrain[j])
    Kba = Kab.T

    Cbb = Kbb + (sigmaNoise ** 2) * np.eye(len(xTrain))
    Cbb = Cbb + 1e-8 * np.eye(len(xTrain))

    alpha = np.linalg.solve(Cbb, yTrain)

    meanPred = Kab @ alpha
    covPred = Kaa - Kab @ np.linalg.solve(Cbb, Kba)
    stdPred = np.sqrt(np.maximum(0.0, np.diag(covPred)))

    trendTest = a * xTest + b
    meanPredPpm = trendTest + meanPred

    upper = meanPredPpm + 1.96 * stdPred
    lower = meanPredPpm - 1.96 * stdPred

    plt.figure()
    plt.plot(t, monthlyMean, '--', label='Measurements')
    plt.plot(xTest, meanPredPpm, label='Gaussian process')
    plt.fill_between(xTest, lower, upper, alpha=0.25)
    plt.plot(xTest, yTest, 'k.', markersize=3)
    plt.title('Monthly average of air samples measurements from the Mauna Loa Observatory')
    plt.xlabel('Time (months)')
    plt.ylabel('Monthly average of CO2 concentration (ppm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('CO2 GP Predictions.pdf')
    plt.show()

if __name__ == '__main__':
    # OneDimensionalGaussian()
    # TwoDimensionalGaussian()
    # solve2()
    # solve3a()
    # solve3b()
    solve3c()