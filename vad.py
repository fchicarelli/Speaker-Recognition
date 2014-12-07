import math
import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


def calculate_energy(frame):
    """Calculates the energy of a signal

    :param frame: frame of a discrete signal
    :return: the energy of the frame
    """
    energy = np.int64(0)
    # Ignoring a Runtimewarning of overflow
    with np.errstate(over='ignore'):
        for i in range(len(frame)):
            energy += frame[i]**2
        energy = math.sqrt(energy / len(frame))
    return energy


def calculate_energy2(frame):

    frame = abs(frame)
    energy = np.sum(np.square(frame, dtype=np.int64), dtype=np.int64)
    energy = math.sqrt(energy / len(frame))
    return energy


def calculate_sfm(frame):
    """Calculates the Spectral Flatness Measure of a signal

     The SFM is defined as the ratio of the geometrical mean by the
     arithmetical mean

    :param frame: frame of a discrete signal
    :return: the SFM of the frame
    """
    a = np.mean(frame)
    g = gmean(frame)
    if a == 0 or g/a <= 0:
        sfm = 0
    else:
        sfm = 10*np.log10(g/a)
    return sfm


def extract_features(signal, num_frames, frame_size, f_sampling):
    """ Given a signal, the number of frames, and the frame size, returns
     the energy, dominating frequency, and the sfm of all frames of the signal

    :param signal: A discrete signal
    :param num_frames: Number of frames of the signal
    :param frame_size: How many values are in a frame of the signal
    :param f_sampling: Sampling frequency
    :return: Returns 3 arrays of length 'num_frames' with the values of
             energy, dominating frequency, and sfm
    """
    energy = np.array(np.zeros([num_frames]), dtype=int)
    energy2 = np.array(np.zeros([num_frames]), dtype=int)
    dominating_freq = np.array(np.zeros([num_frames]))
    sfm = np.array(np.zeros([num_frames]), dtype=int)

    # Calculating features (Energy, SFM, and most dominant frequency)
    for i in xrange(int(num_frames)):
        energy[i] = calculate_energy2(get_frame(signal, frame_size, i))

        # Performs the ftt of the actual frame
        frame_fft = rfft(get_frame(signal, frame_size, i), 1024)
        #frame_fft = rfft(get_frame(signal, frame_size, i))

        power_spectrum = np.abs(frame_fft)

        #freq_fft = rfftfreq(1024, 1./f_sampling)

        # Not using the dominating frequency because the way that was implemented does not had the needed accuracy
        #max_freq_index = np.argmax(power_spectrum);
        #dominating_freq[i] = freq_fft[max_freq_index]

        #TODO: Calculate dominating_freq
        sfm[i] = calculate_sfm(power_spectrum)

    return energy, dominating_freq, sfm


def get_frame(signal, frame_size, n):
    """ Get the n-th frame of the signal

    :param signal: A discrete signal
    :param frame_size: Number of samples in a frame
    :param n: N-th frame to be gotten
    :return: An array if 'frame_size' values corresponding the n-th signal frame
    """
    return signal[frame_size*n:frame_size*(n+1)]


def remove_silence(signal, frame_size, speech):
    """ Gets a signal and remove its silence frames

    :param signal: A discrete signal
    :param frame_size: Number of samples in a frame
    :param speech: A bool array that has the info if a frame is silence or not
    :return: The resulting signal without silence frames
    """
    for i in xrange(len(speech)):
        if not speech[i]:
            signal[frame_size*i:frame_size*(i+1)] = 0

    result = signal[np.nonzero(signal)]

    return result


def compute_vad(signal, f_sampling):
    """ Receives a signal and computes the Voice Activity Detection
    based on the "A SIMPLE BUT EFFICIENT REAL-TIME VOICE ACTIVITY DETECTION
    ALGORITHM" paper [1].
    [1] http://www.eurasip.org/Proceedings/Eusipco/Eusipco2009/contents/papers/1569192958.pdf

    :param signal: A discrete signal
    :param f_sampling: Sampling frequency
    :return: The resulting signal without silence frames
    """
    # Setting the initial variables
    frame_size_time = 0.010
    frame_size_n = (f_sampling * frame_size_time)
    num_frames = int(math.ceil(signal.size / frame_size_n))
    energy_prim_thresh = 40
    f_prim_thresh = 185
    sf_prim_thresh = 5

    #print "Signal size: " + str(signal.size)
    #print "Freq Sampling: " + str(f_sampling) + " Hz"
    #print "Frame size: " + str(frame_size_n)
    #print "Number of frames: " + str(num_frames)

    # Extract features from signal
    energy, dominating_freq, sfm = extract_features(signal, num_frames, frame_size_n, f_sampling)

    # Finding minimum values of the 30 first frames
    min_energy = np.min(energy[3:29])
    #min_freq = np.min(dominating_freq[0:29])
    min_sfm = np.min(sfm[0:39])

    # Setting decision threshold
    thresh_energy = energy_prim_thresh * np.log10(min_energy)
    #thresh_freq = f_prim_thresh
    thresh_sfm = sf_prim_thresh

    speech = np.array(np.zeros([num_frames]), dtype=bool)
    silence_count = 0

    # Deciding if a frame is a speech or silence
    for i in xrange(num_frames):
        counter = 0

        if energy[i] - min_energy >= thresh_energy:
            counter += 1
        #if dominating_freq[i] - min_freq > thresh_freq:
            counter += 1
        if sfm[i] - min_sfm >= thresh_sfm:
            counter += 1

        # Not considering last frame
        # TODO: 0 padding in the last frame
        if counter > 0 and i != num_frames - 1:
            speech[i] = True
        else:
            speech[i] = False
            silence_count += 1
            min_energy = ((min_energy*silence_count)+energy[i])/(silence_count + 1)

    #print "Silence frames: " + str(silence_count) + " (before)"

    # Ignore silence run less than 10 successive frames
    # Ignore speech run less than 5 successive frames
    last = speech[0]
    sequence = 0
    start = 0
    for i in xrange(len(speech)):
        if last == speech[i]:
            sequence += 1
        else:
            if last is False and sequence < 10:
                for j in xrange(start, i):
                    speech[j] = True
            elif last is True and sequence < 5:
                for j in xrange(start, i):
                    speech[j] = False
            start = i
            sequence = 0

    #print "Silence frames: " + str(num_frames - sum(speech)) + " (after)"

    result = remove_silence(signal, frame_size_n, speech)
    result2 = np.array(result, np.int16)

    #print "Result signal size: " + str(len(result2))
    #write('result.wav', f_sampling, result2)

    return result2