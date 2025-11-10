import numpy as np
import matplotlib.pyplot as plt
import scipy

freq, vowels = scipy.io.wavfile.read('../Problem 5/Vowels.wav')
vowels = vowels.astype(float)
n = len(vowels)

windowSize = max(1, int(0.010 * freq))
step = max(1, windowSize // 2)
win = np.hanning(windowSize)

windows = []
for i in range(0, n - windowSize + 1, step):
    frame = vowels[i:i + windowSize] * win
    FFTValue = np.abs(np.fft.rfft(frame))
    windows.append(FFTValue)

S = np.column_stack(windows)
freqs = np.fft.rfftfreq(windowSize, d=1.0/freq)
times = (np.arange(S.shape[1]) * step) / float(freq)

S_db = 20 * np.log10(S + 1e-12)

plt.figure(figsize=(10,5))
plt.pcolormesh(times, freqs, S_db, shading='gouraud', cmap='magma')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (10 ms, 50% overlap)')
plt.colorbar(label='Magnitude (dB)')
plt.tight_layout()
plt.savefig('6.pdf')
plt.show()
