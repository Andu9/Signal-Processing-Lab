import numpy as np
import matplotlib.pyplot as plt

def generateSinus(signalTime, phase, freq = 5):
    return np.sin(2 * np.pi * freq * signalTime + phase)

time = np.linspace(0, 1, 1000)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Signals with different phases and frequencies')
axs[0, 0].plot(time, generateSinus(signalTime = time, phase = 0, freq = 5))
axs[0, 0].set_title(r'$\omega = 0$, f = 5')
axs[0, 1].plot(time, generateSinus(signalTime = time, phase = np.pi / 4, freq = 4))
axs[0, 1].set_title(r'$\omega = \frac{\pi}{4}$, f = 4')
axs[1, 0].plot(time, generateSinus(signalTime = time, phase = np.pi / 2, freq = 6))
axs[1, 0].set_title(r'$\omega = \frac{\pi}{2}$, f = 6')
axs[1, 1].plot(time, generateSinus(signalTime = time, phase = np.pi, freq = 8))
axs[1, 1].set_title(r'$\omega = \pi$, f = 8')
fig.tight_layout()
fig.savefig('2 First Part.pdf')
fig.show()

chosenSignal = generateSinus(signalTime = time, phase = 0, freq = 5)
normChosenSignal = np.linalg.norm(chosenSignal)

gaussianNoise = np.random.normal(0, 1, size = len(chosenSignal))
normGaussianNoise = np.linalg.norm(gaussianNoise)

noiseSignals = []
SNRvalues = [0.1, 1, 10, 100]
for SNR in SNRvalues:
    gamma = normChosenSignal / (np.sqrt(SNR) * normGaussianNoise)
    noiseSignal = chosenSignal + gamma * gaussianNoise
    noiseSignals.append(noiseSignal)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Signals with different noise levels')
axs[0, 0].plot(time, noiseSignals[0]); axs[0, 0].set_title(f'SNR = {SNRvalues[0]}')
axs[0, 1].plot(time, noiseSignals[1]); axs[0, 1].set_title(f'SNR = {SNRvalues[1]}')
axs[1, 0].plot(time, noiseSignals[2]); axs[1, 0].set_title(f'SNR = {SNRvalues[2]}')
axs[1, 1].plot(time, noiseSignals[3]); axs[1, 1].set_title(f'SNR = {SNRvalues[3]}')
fig.tight_layout()
fig.savefig('2 Second Part.pdf')
fig.show()