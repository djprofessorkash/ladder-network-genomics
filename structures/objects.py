#!python38

"""
INITIALIZATIONS AND IMPORT STATEMENTS
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from sys import argv
import math, os, csv
import numpy as np
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
        self._dataset = dataset                 # Stored dataset (X-data)
        self._labels = labels                   # Stored labels (y-data)
        self._num_rows = dataset.shape[0]       # Number of rows/samples in dataset
        self._completed_epochs = 0              # Epoch progression counter
        self._epoch_index = 0                   # ???
        
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

    def sample_next_batch(self, batch_size):
        """ Method to retrieve custom (`batch_size`) number of samples from stored dataset. """
        # Define relative indices for epoch-based data sampling
        position_start = self._epoch_index
        position_end = position_start + batch_size

        # Sample dataset and labels using relative indices
        sampled_data = self._dataset[position_start:position_end]
        sampled_labels = self._labels[position_start:position_end]

        # Iteratively construct batched data and labels using random sampling
        while len(sampled_data) < batch_size:
            # Increment epoch progression counter by one
            self._completed_epochs += 1

            # Create permuted random range from stored dataset for randomized sampling
            permuted_range = np.arange(self._num_rows)
            np.random.shuffle(permuted_range)
            
            # Shuffle dataset and labels using permuted random sampling ranges
            self._dataset = self._dataset[permuted_range]
            self._labels = self._labels[permuted_range]

            # Progress to next epoch by resetting & decrementing relative indices
            position_start, position_end = 0, batch_size - len(sampled_data)

            # Construct randomly partitioned data and label segments into batch
            sampled_data = np.append(sampled_data, self._dataset[position_start:position_end], axis=0)
            sampled_labels = np.append(sampled_labels, self._labels[position_start:position_end], axis=0)

        # Update index of latest partition ending in epoch and output samples
        self._epoch_index = position_end
        return sampled_data, sampled_labels


class DatasetPartition(object):
    pass