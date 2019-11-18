import torch.nn as nn
from functions import unshuffleNd, shuffleNd


class UnShuffleNd(nn.Module):
    def __init__(self, factor, N):
        super(UnShuffleNd, self).__init__()
        self.factor = factor
        self.N = N

    def forward(self, input):
        x = unshuffleNd(input, self.factor, self.N)
        return x


class ShuffleNd(nn.Module):
    def __init__(self, factor, N):
        super(ShuffleNd, self).__init__()
        self.factor = factor
        self.N = N

    def forward(self, input):
        x = shuffleNd(input, self.factor, self.N)
        return x


class UnShuffle1d(UnShuffleNd):
    def __init__(self, factor):
        super(UnShuffle1d, self).__init__(factor, 1)


class UnShuffle2d(UnShuffleNd):
    def __init__(self, factor):
        super(UnShuffle2d, self).__init__(factor, 2)


class Shuffle1d(ShuffleNd):
    def __init__(self, factor):
        super(Shuffle1d, self).__init__(factor, 1)
        self.factor = factor


class Shuffle2d(ShuffleNd):
    def __init__(self, factor):
        super(Shuffle2d, self).__init__(factor, 2)
        self.factor = factor