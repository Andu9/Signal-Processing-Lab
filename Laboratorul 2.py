import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice

def solve1():
    def generateSinus(amp = 2, freq = 3, phase = np.pi / 2):
        x_points = np.linspace(0, 1, 1000)
        y_points = amp * np.sin(2 * np.pi * freq * x_points + phase)

        plt.plot(x_points, y_points)
        plt.suptitle(f'Sine with {amp} Amplitude, {freq} Hz, {phase} Phase')
        plt.show()

    def generateCosinus(amp = 2, freq = 3, phase = 0):
        x_points = np.linspace(0, 1, 1000)
        y_points = amp * np.cos(2 * np.pi * freq * x_points + phase)

        plt.plot(x_points, y_points)
        plt.suptitle(f'Cos with {amp} Amplitude, {freq} Hz, {phase} Phase')
        plt.show()

    generateSinus()
    generateCosinus()

def solve2():
    time = np.linspace(0, 1, 1000)
    freq = 5
    phases = [0, np.pi / 4, np.pi / 2, np.pi]
    signals = [np.sin(2 * np.pi * freq * time + phase) for phase in phases]

    plt.suptitle('Signals with different phases')
    for i, signal in enumerate(signals):
        plt.plot(time, signal, label = fr'$\phi = {phases[i] / np.pi:.2f}\pi$')
    plt.legend(loc = 'best')
    plt.show()

    x = signals[0]
    norm_x = np.linalg.norm(x)

    z = np.random.normal(0, 1, size = len(x))
    norm_z = np.linalg.norm(z)

    noise_signals = []
    SNR_value = [0.1, 1, 10, 100]
    for SNR in SNR_value:
        gamma = norm_x / (np.sqrt(SNR) * norm_z)
        noise_signal = x + gamma * z
        noise_signals.append(noise_signal)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Different noise levels')
    axs[0, 0].plot(time, noise_signals[0]); axs[0, 0].set_title(f'SNR = {SNR_value[0]}')
    axs[0, 1].plot(time, noise_signals[1]); axs[0, 1].set_title(f'SNR = {SNR_value[1]}')
    axs[1, 0].plot(time, noise_signals[2]); axs[1, 0].set_title(f'SNR = {SNR_value[2]}')
    axs[1, 1].plot(time, noise_signals[3]); axs[1, 1].set_title(f'SNR = {SNR_value[3]}')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

def solve3():
    def listen_a():
        print('Start listening to a...')
        time_a = np.linspace(0, 1, 3 * 44100)
        signal_a = np.sin(2 * np.pi * 400 * time_a)
        sounddevice.play(signal_a, 44100)
        sounddevice.wait()
        print('Done listening to a.')

    def listen_b():
        print('Start listening to b...')
        time_b = np.linspace(0, 3, 3 * 44100)
        signal_b = np.sin(2 * np.pi * 800 * time_b)
        sounddevice.play(signal_b, 44100)
        sounddevice.wait()
        print('Done listening to b.')

        print('Saving...')
        scipy.io.wavfile.write('test.wav', int(10e5), signal_b)
        print('Done saving.')

        print('Reading...')
        rate, x = scipy.io.wavfile.read('test.wav')
        print(f'Rate = {rate}')
        print(f'X = {x}')
        print('Done reading.')

    def listen_c():
        print('Start listening to c...')
        time_c = np.linspace(0, 1, 3 * 44100)
        signal_c = np.mod(time_c * 8000, 1.0)
        sounddevice.play(signal_c, 44100)
        sounddevice.wait()
        print('Done listening to c.')

    def listen_d():
        print('Start listening to d...')
        time_d = np.linspace(0, 4, 3 * 44100)
        signal_d = np.sign(np.sin(2 * np.pi * 1000 * time_d))
        sounddevice.play(signal_d, 44100)
        sounddevice.wait()
        print('Done listening to d.')

    subtasks = [listen_a, listen_b, listen_c, listen_d]
    task_to_listen = int(input())

    subtasks[task_to_listen - 1]()

def solve4():
    def sawtooth_wave(f, t):
        x = (f * t) % 1.0
        return 2.0 * (x - 0.5)

    time = np.linspace(0, 1, 1000)
    sine = np.sin(2 * np.pi * 5.0 * time - 0.5)
    saw = sawtooth_wave(5.0, time)

    Sum = sine + saw

    fig, axs = plt.subplots(3)
    axs[0].plot(time, sine); axs[0].set_title('Sine')
    axs[1].plot(time, saw); axs[1].set_title('Saw')
    axs[2].plot(time, Sum); axs[2].set_title('Sum')
    plt.show()

def solve5():
    time = np.linspace(0, 1, 3 * 44100)
    small_amplitude = np.sin(2 * np.pi * 1000 * time)
    big_amplitude = np.sin(2 * np.pi * 2000 * time)

    signal = np.concatenate((small_amplitude, big_amplitude))

    print('Start listening...')
    sounddevice.play(signal, 44100)
    sounddevice.wait()
    print('Done listening.')

def solve6():
    freq = 200.0

    sin_a = np.sin(2 * np.pi * (freq / 2) * np.linspace(0, 1, 64))
    sin_b = np.sin(2 * np.pi * (freq / 4) * np.linspace(0, 1, 64))
    sin_c = np.sin(2 * np.pi * 0.0 * np.linspace(0, 1, 64))

    fig, axs = plt.subplots(3)
    fig.suptitle('Signals with different frequencies')
    axs[0].plot(np.linspace(0, 1, 64), sin_a); axs[0].set_title(f'f = {freq / 2}')
    axs[1].plot(np.linspace(0, 1, 64), sin_b); axs[1].set_title(f'f = {freq / 4}')
    axs[2].plot(np.linspace(0, 1, 64), sin_c); axs[2].set_title(f'f = {0.0}')
    plt.show()

def solve7():
    pass

def solve8():
    pass

if __name__ == '__main__':
    n = int(input())
    functions = [solve1, solve2, solve3, solve4, solve5, solve6, solve7, solve8]
    functions[n - 1]()
