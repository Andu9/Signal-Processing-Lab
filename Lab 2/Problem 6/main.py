import numpy as np
import matplotlib.pyplot as plt

def generateSinus(time, freq):
    return np.sin(2 * np.pi * freq * time)

time = np.linspace(0, 1, 1000)
f_s = 20

fig, axs = plt.subplots(3)
fig.suptitle('')
axs[0].plot(time, generateSinus(time, f_s / 2)); axs[0].set_title(r'$f = \frac{f_s}{2}$')
axs[1].plot(time, generateSinus(time, f_s / 4)); axs[1].set_title(r'$f = \frac{f_s}{4}$')
axs[2].plot(time, generateSinus(time, 0)); axs[2].set_title(r'$f = 0$')
fig.tight_layout()
fig.savefig('6.pdf')
fig.show()