#!python3

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf

def joinV(vectorA, vectorB):
    """ Custom join via vectorized concatenation. """
    return tf.concat([vectorA, vectorB], 0)

def labeler_(vectors, batch_size):
    """ Custom helper for labeling entries based on batch size. """
    if vectors is not None:
        return tf.slice(vectors, [0, 0], [batch_size, -1])
    else:
        return vectors
    
def unlabeler_(vectors, batch_size):
    """ Custom helper for unlabeling entries based on batch size. """
    if vectors is not None:
        return tf.slice(vectors, [batch_size, 0], [-1, -1])
    else:
        return vectors
    
def splitV(vectors):
    """ Custom split via helper functions performing vectorized slicing. """
    return (labeler_(vectors), unlabeler_(vectors))

class DatasetReport(object):
    """ write summary here """
    def __init__(self, dataset, labels):
        """ Initializer method. """
        self._dataset = dataset
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = dataset.shape[0]
        
    @property
    def dataset(self):
        """ Getter method to retrieve stored dataset (X). """
        return self._dataset
    
    @property
    def labels(self):
        """ Getter method to retrieve stored labels. (y) """
        return self._labels

class DatasetPartition(object):
    pass