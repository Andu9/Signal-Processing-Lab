import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

time = np.linspace(0, 1, 1000)
signal = np.cos(2 * np.pi * 5 * time)
y = signal * np.exp(-2 * np.pi * 1.j * time)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(time, signal)
dot_left, = axs[0].plot([], [], 'ro')
axs[0].set_title(r'Signal x(t) = $\sin(2 \pi i \cdot 5 \cdot t)$')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')

axs[1].plot(y.real, y.imag)
dot_right, = axs[1].plot([], [], 'ro')
axs[1].plot([0], [0], 'ko', markersize=3)
axs[1].set_title(r'$\operatorname{Im}(x[n] e^{-2\pi i n})$ as a function of $\operatorname{Re}(x[n] e^{-2\pi i n})$')
axs[1].set_xlabel('Real part')
axs[1].set_ylabel('Imaginary part')
fig.tight_layout()

def update(frame):
    dot_left.set_data([time[frame]], [signal[frame]])
    dot_right.set_data([y.real[frame]], [y.imag[frame]])
    return dot_left, dot_right

ani = FuncAnimation(fig, update, frames=np.arange(0, len(time)), interval=20, blit=True)
ani.save("Signal and diformed circle in C.gif", writer=PillowWriter(fps=30))

omegas = [3, 5, 12, 20]

fig2, axs2 = plt.subplots(2, 2, figsize = (10, 10))

for i, omega in enumerate(omegas):
    z = signal * np.exp(-2 * np.pi * 1.j * omega * time)
    axs2[i // 2, i % 2].plot(z.real, z.imag)
    axs2[i // 2, i % 2].set_xlabel('Real part')
    axs2[i // 2, i % 2].set_ylabel('Imaginary part')
    axs2[i // 2, i % 2].set_title(fr'$\omega = {omega}$')

fig2.tight_layout()
fig2.savefig('Different omegas.pdf')
fig2.show()

