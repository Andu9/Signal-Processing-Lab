import scipy
import sounddevice
import numpy as np
import matplotlib.pyplot as plt

def getSignalA():
    time = np.linspace(0, 3, 1600 * 15)
    return np.sin(2 * np.pi * 400 * time)

def getSignalB():
    time = np.linspace(0, 3, 1600 * 15)
    return np.sin(2 * np.pi * 800 * time)

def getSignalC():
    freq = 240
    time = np.linspace(0, 3, 1600 * 15)
    return np.mod(time * freq, 1)

def getSignalD():
    time = np.linspace(0, 3, 1600 * 15)
    return np.sign(np.sin(2 * np.pi * 300 * time))

def playSignal(signal, name):
    print(f'Start listening to signal {name}')
    sounddevice.play(signal, samplerate=44100)
    sounddevice.wait()
    print(f'Stopped listening to signal {name}')

playSignal(getSignalA(), 'A'); playSignal(getSignalB(), 'B')
playSignal(getSignalC(), 'C'); playSignal(getSignalD(), 'D')

print('Saving signal A')
scipy.io.wavfile.write('signal_A.wav', int(10e5), getSignalA())
print('Done saving.')

print('Reading signal A')
rate, x = scipy.io.wavfile.read('signal_A.wav')
print(f'Rate = {rate}')
print(f'X = {x}')
print('done reading.')