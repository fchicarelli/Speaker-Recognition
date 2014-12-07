import os
from scipy.io.wavfile import read
import numpy as np
import vad
import mel
import gmm
import time
from sys import maxint


database_path = './Database/'
mfcc_path = './MFCC/'
test_path = './Test/'


def scan_files():
    """ Scan the files located at ./Database/UserName/ and save a new .mfcc
    file, containing the MFCCs of each sample read.
    """

    for user in os.listdir(database_path):
        user_path = database_path + user + '/'
        for files in os.listdir(user_path):
            if files[-4:] == '.wav':
                data = read(user_path + files)
                print 'Extracting mfcc of: ' + user_path + files

                audio = data[1]
                f_sampling = data[0]
                audio_without_silence = vad.compute_vad(audio, f_sampling)
                mfcc = mel.extract_mfcc(audio_without_silence, f_sampling)

                print 'Saving MFCC at: ' + user_path + files.split('.')[0] + '.mfcc'
                np.savetxt(user_path + files.split('.')[0] + '.mfcc', mfcc, newline='\n')


def save_all_mfcc():
    """ Scan all .mfcc files at ./Database/UserName/, group them together by
    username and save one .mfcc file for each user containing all mfcc for
    this user at ./MFCC/
    """

    for user in os.listdir(database_path):
        user_path = database_path + user + '/'
        all_values = []
        for files in os.listdir(user_path):
            if files[-5:] == '.mfcc':
                mfcc = np.loadtxt(user_path + files)
                all_values.append(mfcc)

        for i in xrange(len(all_values)):
            if i == 0:
                mfcc = np.array(all_values[0])
            else:
                mfcc = np.append(mfcc, all_values[i], axis=0)

        np.savetxt(mfcc_path + user + '.mfcc', mfcc, newline='\n')


def train_model():
    """ Read all MFCC files at ./MFCC/ and use the coefficients to train
    models for each user. The model used is a Gaussian Mixture Model
    """

    models = []
    names = []
    for files in os.listdir(mfcc_path):
        print 'Training ' + files.split('.')[0] + "'s model"
        mfcc = np.loadtxt(mfcc_path + files)
        models.append(gmm.gmm(mfcc))
        names.append(files.split('.')[0])

    return models, names


def find_speaker(models, names):
    """ Given the user models, returns the user that has the highest likelihood
    score with the input test located at ./Test/

    :param models: GMM models for each user in the database
    :param names: Name of each user

    :return: the index of the most likely user to be the one speaking in the
    input test and the mfcc from this input test.
    """

    for file in os.listdir(test_path):
        data = read(test_path + file)
        print 'Extracting mfcc of: ' + test_path + file

        audio = data[1]
        f_sampling = data[0]
        audio_without_silence = vad.compute_vad(audio, f_sampling)
        mfcc = mel.extract_mfcc(audio_without_silence, f_sampling)

        lk = -maxint - 1
        index = -1

        for i in xrange(len(models)):
            aux = gmm.get_likehood(models[i], mfcc)
            print "Score of " + names[i] + ": " + str(aux)
            if aux > lk:
                lk = aux
                index = i

        print 'The user that had the highest score was: ' + names[index]
        return index, mfcc


def test_speaker(models, nameIndex, mfcc, threshold):
    """ Do the verification process to verify if the user really is the user chosen
    or is someone else from outside the database

    :param models: All models for each user in the database
    :param nameIndex: Index of the user chosen as the most probably
    :param mfcc: MFCC of the input test
    :param threshold: Threshold for the difference
    """

    total = 0

    for i in xrange(len(models)):
        if i != nameIndex:
            total += gmm.get_likehood(models[i], mfcc)

    total /= (len(models) - 1)

    candidate = gmm.get_likehood(models[nameIndex], mfcc)

    print "Difference between the probability of being the user chosen before or any other" \
          " user in the database is: " + str(candidate - total)

    if candidate - total > threshold:
        print "Most likely the user really is the one chosen"
    else:
        print "Probably the person who is speaking is from outside the database"


if __name__ == '__main__':
    start = time.time()
    print 'Starting the MFCC extraction'
    #scan_files()
    #save_all_mfcc()
    models, names = train_model()
    nameIndex, mfcc = find_speaker(models, names)
    test_speaker(models, nameIndex, mfcc, 350)
    end = time.time()
    print str(end - start) + " seconds"