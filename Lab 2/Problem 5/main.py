import numpy as np
import sounddevice

time = np.linspace(0, 1, 3 * 44100)

smallSinus = np.sin(2 * np.pi * 1000 * time)
bigSinus = np.sin(2 * np.pi * 2000 * time)

signal = np.concatenate((smallSinus, bigSinus))

print('Start listening.')
sounddevice.play(signal, samplerate = 44100)
sounddevice.wait()
print('Done listening.')