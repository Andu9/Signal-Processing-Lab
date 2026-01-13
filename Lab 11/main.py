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

def generateHankelMatrix(signalParam, L):
    N = signalParam.shape[0]
    K = N - L + 1

    j = np.linspace(0, K - 1, K).astype(int)
    X = np.vstack([signalParam[i + j] for i in range(L)])

    return X

def sortEigDesc(eig, UParam):
    index = np.argsort(eig)[::-1]
    return eig[index], UParam[:, index]

def diagonalAveraging(Xg):
    L, K = Xg.shape
    N = L + K - 1

    yRec = np.zeros(N)
    counts = np.zeros(N)

    for i in range(L):
        for j in range(K):
            yRec[i + j] += Xg[i, j]
            counts[i + j] += 1.0

    return yRec / counts

def reconstructFromComponents(UParam, sParam, VtParam, indices, HankelMatrixShape):
    Xg = np.zeros(HankelMatrixShape)
    for i in indices:
        Xg += sParam[i] * np.outer(UParam[:, i], VtParam[i, :])
    return diagonalAveraging(Xg)

if __name__ == "__main__":
    signal, time = generateTimeSeries()

    HankelMatrix = generateHankelMatrix(signal, 200)

    A = HankelMatrix @ HankelMatrix.T
    B = HankelMatrix.T @ HankelMatrix

    eigA, UA = np.linalg.eig(A)
    eigB, UB = np.linalg.eig(B)

    eigA, UA = sortEigDesc(eigA, UA)
    eigB, UB = sortEigDesc(eigB, UB)

    U, s, Vt = np.linalg.svd(HankelMatrix, full_matrices = False)

    # s2 = s ** 2
    # m = min(len(s2), len(eigA), len(eigB))
    #
    # maxDifferenceA = np.max(np.abs(eigA[:m] - s2[:m]))
    # maxDifferenceB = np.max(np.abs(eigB[:m] - s2[:m]))
    #
    # print(f'Max |eig(XX^T) - s ^ 2| = {maxDifferenceA}')
    # print(f'Max |eig(X^TX) - s ^ 2| = {maxDifferenceB}')

    trendGroup = [0]
    seasonalGroup = [1, 2, 3, 4]
    residualGroup = list(range(5, len(s)))

    trendReconstruction = reconstructFromComponents(U, s, Vt, trendGroup, HankelMatrix.shape)
    seasonalReconstruction = reconstructFromComponents(U, s, Vt, seasonalGroup, HankelMatrix.shape)
    residualReconstruction = reconstructFromComponents(U, s, Vt, residualGroup, HankelMatrix.shape)

    totalReconstruction = trendReconstruction + seasonalReconstruction + residualReconstruction

    error = signal - totalReconstruction

    fig, axs = plt.subplots(4, 1, figsize = (10, 6))
    axs[0].plot(time, signal, label = 'Original')
    axs[1].plot(time, trendReconstruction, label = 'Trend Reconstruction', color = 'red')
    axs[2].plot(time, seasonalReconstruction, label = 'Seasonal Reconstruction', color = 'black')
    axs[3].plot(time, residualReconstruction, label = 'Residual Reconstruction', color = 'green')
    fig.suptitle('Single Spectrum Analysis Reconstruction Components')
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
    fig.legend()
    fig.tight_layout()
    fig.savefig('4 Components.pdf')
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize = (10, 6))
    axs[0].plot(time, signal, label = 'Original')
    axs[1].plot(time, totalReconstruction, label = 'Reconstruction', color = 'red')
    fig.suptitle('Single Spectrum Analysis Reconstruction')
    for ax in axs:
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
    fig.legend()
    fig.tight_layout()
    fig.savefig('4 Reconstruction.pdf')
    plt.show()