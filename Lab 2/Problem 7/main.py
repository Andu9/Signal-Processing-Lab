import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1 / 50, 100)
signal = np.sin(2 * np.pi * 1000 * time)

fig, axs = plt.subplots(2)
fig.suptitle('Graphics of signal and sampling every 4th point')
axs[0].plot(time, signal); axs[0].set_title('Whole signal')
axs[1].plot(time[::4], signal[::4]); axs[1].set_title('Same signal with only the 4th point taken')
fig.tight_layout()
fig.savefig('7a.pdf')
fig.show()

fig, axs = plt.subplots(2)
fig.suptitle('Graphics of signal and sampling every 4th point starting from the 2nd')
axs[0].plot(time, signal); axs[0].set_title('Whole signal')
axs[1].plot(time[1::4], signal[1::4]); axs[1].set_title('Same signal with only the 4th point taken starting from the 2nd')
fig.tight_layout()
fig.savefig('7b.pdf')
fig.show()