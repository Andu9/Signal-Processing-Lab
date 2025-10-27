import numpy as np
import matplotlib.pyplot as plt

time = np.linspace(-np.pi / 2, np.pi / 2, 1000)
sin = np.sin(time)
linearApproximation = time
padeApproximation = (time - 7 * time ** 3 / 60) / (1 + time ** 2 / 20)

fig, axs = plt.subplots(3)
fig.suptitle('Comparison between Linear and Pade Approximation of Sinus Wave')
axs[0].plot(time, sin); axs[0].set_title('Sinus')
axs[1].plot(time, linearApproximation); axs[1].set_title(r'Linear Approximation $\sin(\alpha) \approx \alpha$')
axs[2].plot(time, padeApproximation); axs[2].set_title(r'Pade Approximation $\sin(\alpha) \approx \frac{\alpha - \frac{7\alpha^3}{60}}{1 + \frac{\alpha^2}{20}}$')
fig.tight_layout()
fig.savefig('7 Linear vs Pade Approximation of Sinus.pdf')
fig.show()

fig, axs = plt.subplots(3)
fig.suptitle('Comparison between Linear and Pade Approximation of Sinus Wave - Logarithmic y axis')
axs[0].plot(time, sin); axs[0].set_title('Sinus');
axs[1].plot(time, linearApproximation); axs[1].set_title(r'Linear Approximation $\sin(\alpha) \approx \alpha$')
axs[2].plot(time, padeApproximation); axs[2].set_title(r'Pade Approximation $\sin(\alpha) \approx \frac{\alpha - \frac{7\alpha^3}{60}}{1 + \frac{\alpha^2}{20}}$')
for ax in axs:
    ax.set_yscale('log')
fig.tight_layout()
fig.savefig('7 Linear vs Pade Approximation of Sinus - Logarithmic y axis.pdf')
fig.show()

linearError = np.abs(sin - linearApproximation)
padeError = np.abs(sin - padeApproximation)

plt.plot(time, linearError, label = 'Linear Approximation Error')
plt.plot(time, padeError, label = 'Pade Approximation Error')
plt.title('Linear vs Pade Approximation of Sinus Wave - Errors')
plt.legend()
plt.savefig('7 Linear vs Pade Approximation of Sinus - Error.pdf')
plt.show()

plt.plot(time, linearError, label = 'Linear Approximation Error')
plt.plot(time, padeError, label = 'Pade Approximation Error')
plt.title('Linear vs Pade Approximation of Sinus Wave - Errors')
plt.yscale('log')
plt.legend()
plt.savefig('7 Linear vs Pade Approximation of Sinus - Error - Logarithmic y axis.pdf')
plt.show()