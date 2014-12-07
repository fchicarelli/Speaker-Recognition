import numpy as np
from scipy import fftpack
import math
import matplotlib.pyplot as plt


def frequency_to_mel(freq):
    """ Converts frequency to mel

    :param freq: frequency in hertz
    :return: the corresponding mel value
    """
    result = 2595 * np.log10(1 + freq / 700.0)
    return result


def mel_to_frequency(mel):
    """ Converts mel to frequency

    :param mel: mel value
    :return: the corresponding frequency value
    """
    result = 700 * (10 ** (mel / 2595.0) - 1)
    return result


def get_frame_index(n, f_sampling, t_frame=25, t_step=10):
    """ Find the indexes of the start and end of the n-th frame

    :param n: Frame number
    :param f_sampling: Sampling frequency
    :param t_frame: Total duration of each frame (in ms)
    :param t_step: The difference in time from the start of the previous frame
                    to the actual frame
    :return: Start and End indexes of the frame
    """
    start = (n * f_sampling * t_step) / 1000
    end = start + (f_sampling * t_frame) / 1000
    return start, end


def get_frame(signal, n, f_sampling, t_frame=25, t_step=10):
    """

    :param signal: A discrete signal
    :param n: Frame number
    :param f_sampling: Sampling frequency
    :param t_frame: Total duration of each frame (in ms)
    :param t_step: The difference in time from the start of the previous frame
                    to the actual frame
    :return: A portion of the signal corresponding to the nth frame
    """
    start, end = get_frame_index(n, f_sampling, t_frame, t_step)
    return signal[start:end]


def pre_emphasis(signal, alpha=0.95):
    """ Pre emphasis step. It's actually a high pass filter in time domain

    :param signal: A discrete signal
    :param alpha: Coefficient for the formula
    :return: The resulting signal
    """
    res = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return res


def hamming_windown(frame):
    """ Applies a Hamming Window to a frame of the signal

    :param frame: A portion of the signal corresponding to a frame
    :return: The signal with the window applied
    """
    w = np.hamming(frame.size)
    y = frame * w
    return y


def magnitude_spectrum(frame, nfft):
    """ Calculates the Magnitude Spectrum of a frame

    :param frame: A portion of the signal corresponding to a frame
    :param nfft: FFT size
    :return: The value of the Magnitude Spectrum
    """
    fft = np.fft.rfft(frame, nfft)
    return np.absolute(fft)


def power_spectrum(frame, nfft):
    """ Calculates the Power Spectrum of a frame

    :param frame: A portion of the signal corresponding to a frame
    :param nfft: FFT size
    :return: The value of the Power Spectrum
    """
    return 1.0/nfft * np.square(magnitude_spectrum(frame, nfft))


def create_filterbanks(lower, higher, n, nfft, f_sampling):
    """ Create a Mel Triangular Filterbank

    :param lower: Lowest frequency of the system
    :param higher: Highest frequency of the system
    :param n: Number of filters
    :param nfft: FFT size
    :param f_sampling: Sampling Frequency
    :return: The Filterbank with n Mel Triangular filters
    """
    dif = (higher - lower) / (n + 1)

    bank = np.array([lower])
    for i in xrange(1, n+2):
        bank = np.append(bank, bank[i-1] + dif)

    freq_melbank = mel_to_frequency(bank)
    f = np.floor((nfft + 1) * freq_melbank / f_sampling)

    filterbanks = np.zeros((n, nfft/2+1))

    for i in xrange(0, n):
        for j in xrange(int(f[i]), int(f[i+1])):
            filterbanks[i][j] = (j - f[i]) / (f[i+1] - f[i])
        for j in xrange(int(f[i+1]), int(f[i+2])):
            filterbanks[i][j] = (f[i+2] - j) / (f[i+2] - f[i+1])

    return filterbanks


def liftering(mfcc, l=22):
    """ Applies a liftering to the mfcc

    :param mfcc: MFCC extracted from some audio
    :param l: Parameter of the Liftering
    :return: The result MFCC
    """
    frames, num_mfcc = np.shape(mfcc)
    n = np.arange(num_mfcc)
    coef = 1 + l / 2 * np.sin(np.pi * n / l)
    return coef * mfcc


def extract_mfcc(signal, f_sampling, num_mfcc=13, debug=0):
    """ Function that extract the Mel-Frequency Cepstral Coefficient from an audio

    :param signal: A discrete signal
    :param f_sampling: Sampling frequency
    :param num_mfcc: Number of MFCC desired
    :param debug: Debug variable
    :return: The MFCCs extracted
    """
    num_samples = signal.size
    frame_size = math.floor(f_sampling * 0.025)
    step_size = math.floor(f_sampling * 0.01)
    nfft = 512

    num_frames = math.ceil((num_samples - frame_size) / step_size + 1)

    lower_freq = 50
    higher_freq = f_sampling / 2
    lower_mel = frequency_to_mel(lower_freq)
    higher_mel = frequency_to_mel(higher_freq)

    filterbanks = create_filterbanks(lower_mel, higher_mel, 26, nfft, f_sampling)

    mfcc = np.zeros((num_frames, num_mfcc))

    for i in xrange(int(num_frames)):
        pre_sig = pre_emphasis(get_frame(signal, i, f_sampling))
        h = hamming_windown(pre_sig)
        pe = power_spectrum(h, nfft)
        bank_energy = np.dot(pe[:nfft/2+1], filterbanks.transpose())
        log_bank_energy = np.log10(bank_energy)
        dct_energy = fftpack.dct(log_bank_energy)[-1*num_mfcc:]
        mfcc[i][:] = dct_energy

        if i == 28 and debug:
            plt.figure(1)
            plt.plot(get_frame(signal, i, f_sampling), color='blue')
            plt.plot(pre_sig, color='red')
            plt.plot(h, color='green')
            plt.show()

    mfcc = liftering(mfcc)

    return mfcc