import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 1600)
sinus = np.sin(2 * np.pi * 5 * time)
sawtooth = np.mod(time * 5, 1)

sum = sinus + sawtooth

fig, axs = plt.subplots(3)
fig.suptitle('Sinus, sawtooth and their sum')
axs[0].plot(time, sinus); axs[0].set_title('Sinus')
axs[1].plot(time, sawtooth); axs[1].set_title('Sawtooth')
axs[2].plot(time, sum); axs[2].set_title('Sum of sinus and sawtooth')
fig.tight_layout()
fig.savefig('4.pdf')
fig.show()
