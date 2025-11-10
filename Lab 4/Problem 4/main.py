"""

Vrem fs >= 2 * B, cu B în intervalul [40, 200] Hz. Deci fs >= 2 * 200 = 400 Hz.

"""

import numpy as np
import matplotlib.pyplot as plt

def generateSine(freq, time):
    return np.sin(2 * np.pi * freq * time)

def generateSignal(time):
    return generateSine(40, time) + generateSine(200, time) + generateSine(100, time) + generateSine(150, time)

time = np.linspace(0, 1, 2000)
wrongly_sampled_time = np.linspace(0, 1, 100, endpoint = False)
correctly_sampled_time = np.linspace(0, 1, 400, endpoint = False)

signal = generateSignal(time)
wrongly_sampled_signal = generateSignal(wrongly_sampled_time)
correctly_sampled_signal = generateSignal(correctly_sampled_time)

fig, axs = plt.subplots(2, 1, figsize = (20, 10))

axs[0].plot(time, signal, label = 'Signal')
axs[0].scatter(correctly_sampled_time, correctly_sampled_signal, color = 'black')
axs[0].plot(time, generateSine(150, time), color = 'green', label = '150 Hz Signal')
axs[0].set_title('Correctly sampled signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].legend()

axs[1].plot(time, signal, label = 'Signal')
axs[1].scatter(wrongly_sampled_time, wrongly_sampled_signal, color = 'black')
axs[1].plot(time, generateSine(150, time), color = 'red', label = '150 Hz Signal')
axs[1].set_title('Wrong sampled signal')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()

fig.savefig('4.pdf')
fig.tight_layout()
fig.show()