import numpy as np


# noinspection PyPep8Naming
class DialogueTrainingData(object):
    def __init__(self, X, y, flags=None, true_length=None):
        self.X = X
        self.y = y
        self.flags = flags
        self.true_length = true_length

    def limit_training_data_to(self, max_samples):
        self.X = self.X[:max_samples]
        self.y = self.y[:max_samples]
        self.flags = self.flags[:max_samples]
        self.true_length = self.true_length[:max_samples]

    def is_empty(self):
        """Check if the training matrix does contain training samples."""
        return self.X.shape[0] == 0

    def max_history(self):
        return self.X.shape[1]

    def num_examples(self):
        return len(self.y)

    def shuffled_X_y(self):
        idx = np.arange(self.num_examples())
        np.random.shuffle(idx)
        shuffled_X = self.X[idx]
        shuffled_y = self.y[idx]
        shuffled_flags = (self.flags[idx]
                          if self.flags is not None
                          else self.flags)
        shuffled_true_length = (self.true_length[idx]
                                if self.true_length is not None
                                else self.true_length)
        return shuffled_X, shuffled_y, shuffled_flags, shuffled_true_length
