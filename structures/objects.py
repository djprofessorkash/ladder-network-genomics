#!python3

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf

def vectorial_join(vectorA, vectorB):
    """ Custom join via vectorized concatenation. """
    return tf.concat([vectorA, vectorB], 0)

def vectorial_labeler(vectors, batch_size):
    """ Custom helper for labeling entries based on batch size. """
    if vectors is not None:
        return tf.slice(vectors, [0, 0], [batch_size, -1])
    else:
        return vectors
    
def vectorial_unlabeler(vectors, batch_size):
    """ Custom helper for unlabeling entries based on batch size. """
    if vectors is not None:
        return tf.slice(vectors, [batch_size, 0], [-1, -1])
    else:
        return vectors
    
def vectorial_split(vectors):
    """ Custom split via helper functions performing vectorized slicing. """
    return (vectorial_labeler(vectors), vectorial_unlabeler(vectors))

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