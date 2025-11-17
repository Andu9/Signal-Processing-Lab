import numpy as np
import matplotlib.pyplot as plt

def generateSinus(amp, freq, phase, time):
    return amp * np.sin(2 * np.pi * freq * time + phase)

def generateCosinus(amp, freq, phase, time):
    return amp * np.cos(2 * np.pi * freq * time + phase)

time = np.linspace(0, 1, 1000)

sinus = generateSinus(amp = 2, freq = 3, phase = np.pi / 2, time = time)
cosinus = generateCosinus(amp = 2, freq = 3, phase = 0, time = time)

fig, axs = plt.subplots(2)
fig.suptitle(r'Sinus and Cosinus with phase difference $\frac{\pi}{2}$')
axs[0].plot(time, sinus); axs[0].set_title(r'Sinus with phase = $\frac{\pi}{2}$')
axs[1].plot(time, cosinus); axs[1].set_title('Cosinus with phase = 0')
plt.tight_layout()
plt.savefig('D.pdf'); plt.show()