import numpy as np

class Faces(object):
    def __init__(self, npy=None):
        self.data = np.load(npy, 'r')

    # gets random array of values
    def next_batch(self, num):
        '''
        Return a total of `num` random samples and labels.
        '''
        idx = np.arange(0, self.data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num]
        batch = self.data[idx]
        return batch
