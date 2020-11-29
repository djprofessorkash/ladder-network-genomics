#!python3

"""
INITIALIZATIONS AND IMPORT STATEMENTS
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf

"""
GLOBAL FUNCTIONS AND VARIABLES
"""

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
    
def vectorial_split(vectors, batch_size):
    """ Custom split via helper functions performing vectorized slicing. """
    return (vectorial_labeler(vectors, batch_size), 
            vectorial_unlabeler(vectors, batch_size))

"""
CLASS INSTANCES
"""

class DatasetReport(object):
    """ write summary here """
    def __init__(self, dataset, labels):
        """ Initializer method. """
        self._dataset = dataset
        self._labels = labels
        self._completed_epochs = 0
        self._epoch_index = 0
        self._num_rows = dataset.shape[0]
        
    @property
    def get_dataset(self):
        """ Getter method to retrieve stored dataset (X). """
        return self._dataset
    
    @property
    def get_labels(self):
        """ Getter method to retrieve stored labels. (y) """
        return self._labels

    @property
    def get_num_rows(self):
        """ Getter method to retrieve number of samples (rows) across the stored dataset. """
        return self._num_rows

    @property
    def get_completed_epochs(self):
        """ Getter method to retrieve number of currently completed epochs (training generations). """
        return self._completed_epochs

class DatasetPartition(object):
    pass