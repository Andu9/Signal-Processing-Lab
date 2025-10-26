import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 0.03, int(1 / 0.0005))

x = np.cos(520 * np.pi * time + np.pi / 3)
y = np.cos(280 * np.pi * time - np.pi / 3)
z = np.cos(120 * np.pi * time + np.pi / 3)

fig, axs = plt.subplots(3)
fig.suptitle('x(t), y(t) and z(t)')
axs[0].plot(time, x); axs[0].set_title('x(t)')
axs[1].plot(time, y); axs[1].set_title('y(t)')
axs[2].plot(time, z); axs[2].set_title('z(t)')
plt.tight_layout()
plt.savefig('1b.pdf'); fig.show()

freq = 200
dist = 1 / freq
noSamples = 0.03 / dist

sample = np.linspace(0, 0.03, int(noSamples))

x_sampled = np.cos(520 * np.pi * sample + np.pi / 3)
y_sampled = np.cos(280 * np.pi * sample - np.pi / 3)
z_sampled = np.cos(120 * np.pi * sample + np.pi / 3)

fig, axs = plt.subplots(3)
fig.suptitle(f'Sampled x(t), y(t) and z(t) with f = {freq} Hz')
axs[0].stem(sample, x_sampled); axs[0].set_title('Sampled x(t)')
axs[1].stem(sample, y_sampled); axs[1].set_title('Sampled y(t)')
axs[2].stem(sample, z_sampled); axs[2].set_title('Sampled z(t)')
plt.tight_layout()
plt.savefig('1c.pdf'); fig.show()