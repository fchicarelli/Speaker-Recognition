from sklearn import mixture
import numpy as np


def gmm(mfcc):
    """ Creates and train a model with some data (MFCC)

    :param mfcc: MFCC of the user to be modeled
    :return: The GMM created
    """
    g = mixture.GMM(n_components=32)
    g.fit(mfcc)
    return g


def get_likehood(model, mfcc):
    """ Gets the likelihood score of the mfcc compared with a model

    :param model: GMM for a specific user
    :param mfcc: MFCC of the input file
    :return: The likelihood score
    """
    return np.sum(model.score_samples(mfcc)[0])