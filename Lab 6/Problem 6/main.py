import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, freqz, filtfilt

def movingAverage(signal, windowSize):
    return np.convolve(signal, np.ones(windowSize) / windowSize, mode = 'valid')

def Butterworth(Wn, order = 5):
    return butter(order, Wn, btype = 'low')

def Chebyshev(Wn, order = 5, rp_db = 5):
    return cheby1(order, rp_db, Wn, btype = 'low')

def solve():
    data = pd.read_csv('Train.csv')
    traffic = data['Count'].values[:72]

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(traffic)), traffic, label = "Original Signal")
    plt.title("Traffic data for 3 days (72 hours)")
    plt.grid(True)
    plt.legend()

    windowSizes = [5, 9, 13, 17]

    plt.subplot(2, 1, 2)
    for windowSize in windowSizes:
        m = movingAverage(traffic, windowSize)
        rearrangedTime = np.arange(len(m)) + windowSize // 2
        plt.plot(rearrangedTime, m, label=f"w={windowSize}")

    plt.title("Moving Average Filtering")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('6 - B.pdf')
    plt.show()

    '''
    
    1 sample / hour => Fs = 1 / 3600 Hz => Nyquist frequency = 1 / 7200 Hz
    Morning/evening peaks => data has a period of about 24 hours => T_day = 24 * 3600 s => f_day = 1 / 24 * 3600 Hz
    W_n = f_c / F_Nyquist = fc / fs / 2 = (1 / 24 * 3600) / (1 / 7200) = 1 / 12
    
    '''

    fs = 1 / 3600
    fc = 1 / (24 * 3600)
    Wn = fc / (fs / 2)

    bButter, aButter = Butterworth(Wn)
    bChebyshev, aChebyshev = Chebyshev(Wn)

    print(f"Butterworth: order = 5, Wn = {Wn:.4f}")
    print(f"Chebyshev-I: order = 5, rp = 5 dB, Wn = {Wn:.4f}")

    w, hButter = freqz(bButter, aButter)
    _, hChebyshev = freqz(bChebyshev, aChebyshev)

    plt.plot(w, 20 * np.log10(np.abs(hButter)), label="Butterworth")
    plt.plot(w, 20 * np.log10(np.abs(hChebyshev)), label="Chebyshev-I")
    plt.title("Response of Designed Filters")
    plt.xlabel("Digital frequency (rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('6 - D.pdf')
    plt.show()

    xButter = filtfilt(bButter, aButter, traffic)
    xChebyshev = filtfilt(aChebyshev, aChebyshev, traffic)

    plt.plot(np.arange(len(traffic)), traffic, label="Original")
    plt.plot(np.arange(len(traffic)), xButter, label="Butterworth (order 5)")
    plt.plot(np.arange(len(traffic)), xChebyshev, label="Chebyshev-I (order 5, rp=5 dB)")
    plt.title("Filtered Traffic Signal")
    plt.xlabel("Index")
    plt.ylabel("Traffic")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('6 - E.pdf')
    plt.show()

    testOrders = [3, 5, 7]
    testRipples = [1, 3, 5, 10]

    for testOrder in testOrders:
        for rp in testRipples:
            bTest, aTest = Chebyshev(Wn, order = testOrder, rp_db = rp)
            xTest = filtfilt(bTest, aTest, traffic)

            diff = np.std(np.diff(xTest))
            print(f"Chebyshev-I: order={testOrder}, rp={rp} dB -> diff std = {diff:.3f}")

    '''
    
    Smaller diff => smoother signal, but if the order is too high or rp too big, there might appear some distortions/ripples
    
    '''

if __name__ == '__main__':
    solve()