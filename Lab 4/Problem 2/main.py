import numpy as np
import matplotlib.pyplot as plt

fs = 8
f = 25

def generateSine(freq, time):
    return np.sin(2 * np.pi * freq * time)

time = np.linspace(0, 1, 2000)
sampled_time = np.linspace(0, 1, fs, endpoint = False)

signal = generateSine(f, time)
sampled_signal = generateSine(f, sampled_time)

fig, axs = plt.subplots(4, 1, figsize = (20, 10))

axs[0].plot(time, signal)
axs[0].set_title(f'Original Signal at {f} Hz')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')

axs[1].plot(time, signal)
axs[1].scatter(sampled_time, sampled_signal)
axs[1].set_title(f'Original Signal ({f} Hz) sampled at {fs} Hz')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')

axs[2].plot(time, generateSine(f + fs, time))
axs[2].scatter(sampled_time, generateSine(f + fs, sampled_time))
axs[2].set_title(f'Signal ({f + fs} Hz) sampled at {fs} Hz')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Amplitude')

axs[3].plot(time, generateSine(f - 3 * fs, time))
axs[3].scatter(sampled_time, generateSine(f - 3 * fs, sampled_time))
axs[3].set_title(f'Signal ({f - 3 * fs} Hz) sampled at {fs} Hz')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Amplitude')

fig.tight_layout()
fig.savefig('2.pdf')
fig.show()


