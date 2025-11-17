import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

a. The signal seems to be sampled once every hour, so the sampling frequency is 1 / hour = 1 / 3600 Hz
b. 18288 samples with a sampling frequency of 1 / hour => 18288 hours = 762 days
c. Sampling frequency = 1 / 3600 Hz => Nyquist frequency = 1 / 7200 Hz

"""

def solveD(dataSignalParam, Fs = 1 / 3600):
    N = len(dataSignalParam)
    X = np.fft.fft(dataSignalParam)
    magX = (np.abs(X) / N)[:N // 2]
    f = Fs * np.linspace(0, N / 2, N // 2, endpoint = False) / N

    plt.plot(f * 24 * 3600, magX) # convert it cycles/day, hence the 24 * 3600 term
    plt.title('One-sided FFT Magnitude')
    plt.xlabel('Frequency (Cycles / Day)')
    plt.ylabel('|X(f)|')
    plt.show()

def solveE(dataSignalParam):
    mean = np.mean(dataSignalParam)

    if not np.isclose(mean, 0):
        print('Continuous component')
        dataSignalParam = dataSignalParam - mean

    return dataSignalParam

def solveF(dataSignalParam, Fs = 1 / 3600):
    dataSignalParam = solveE(dataSignalParam)

    N = len(dataSignalParam)
    X = np.fft.fft(dataSignalParam)
    magX = (np.abs(X) / N)[:N // 2]

    freqs = Fs * np.linspace(0, N / 2, N // 2, endpoint = False) / N

    magXNoDC = magX[1:]

    biggest4Indexes = np.argsort(magXNoDC)[-4:] + 1

    for index in biggest4Indexes[::-1]:
        freq = freqs[index]
        freqPerDay = freq * 24 * 3600
        print(f'Index = {index}, |X| = {magX[index]:.2f}, f = {freq:.6e} Hz ({freqPerDay:.3f} cycles / day)')

    """
        Index = 1, |X| = 66.85, f = 1.518907e-08 Hz (0.001 cycles / day)
        Index = 2, |X| = 35.22, f = 3.037815e-08 Hz (0.003 cycles / day)
        Index = 762, |X| = 27.10, f = 1.157407e-05 Hz (1.000 cycles / day)
        Index = 3, |X| = 25.22, f = 4.556722e-08 Hz (0.004 cycles / day)

        Index = 762 => one oscillation per day => daily human activity cycle, probably
        The other ones has longer periods, so there represent more long-term variations.
    
    """

def solveG(dataframe):
    datetimes = dataframe['Datetime'].values
    counts = dataframe['Count'].values

    startIndex = None
    for i in range(1000, len(datetimes)):
        if pd.to_datetime(datetimes[i]).weekday() == 0:
            startIndex = i
            break

    endIndex = startIndex + 30 * 24 # suppose a month is exactly 30 days

    plt.plot(datetimes[startIndex:endIndex], counts[startIndex:endIndex])
    plt.title('One month of traffic')
    plt.xlabel('Time')
    plt.ylabel('Number of cars')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

def solveH():
    """

    I would analyze the signal to find a time period where traffic seems stable and there are no peaks or lows.
    I would look at approximately a month of these data and divide them into weeks, then I would try to figure out
    what data correspond to the first 5 days of the week, which would lead me to finding the first day in the week in
    the signal. This method could fail quite badly if I can't find a period of time when the traffic is stable.

    """

    pass

def solveI(dataSignalParam):
    dataSignalParam = solveE(dataSignalParam)

    N = len(dataSignalParam)
    X = np.fft.fft(dataSignalParam)
    magX = np.abs(X) / N

    threshold = np.mean(magX) + 3 * np.std(magX) # 99.7# of values within 3 standard deviations

    filteredX = X.copy()
    for k in range(N):
        if magX[k] >= threshold:
            filteredX[k] = 0

    filtered = np.fft.ifft(filteredX).real

    time = np.linspace(0, N - 1, N) / 24

    plt.plot(time, dataSignalParam, label = 'Original (zero-mean)', alpha = 0.6)
    plt.plot(time, filtered, label = 'Filtered')
    plt.title('Filtering signal')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of cars')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('Train.csv', parse_dates=['Datetime'], dayfirst = True)

    dataID = data['ID'].values
    dataSignal = data['Count'].values

    solveD(dataSignal)
    solveD(solveE(dataSignal))
    solveF(dataSignal)
    solveG(data)
    solveI(dataSignal)





