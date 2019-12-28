import torch.nn as nn
from functions import unshuffleNd, shuffleNd


class UnShuffleNd(nn.Module):
    def __init__(self, scale, N):
        super(UnShuffleNd, self).__init__()
        self.scale = scale
        self.N = N

    def forward(self, input):
        x = unshuffleNd(input, self.scale, self.N)
        return x


class ShuffleNd(nn.Module):
    def __init__(self, scale, N):
        super(ShuffleNd, self).__init__()
        self.scale = scale
        self.N = N

    def forward(self, input):
        x = shuffleNd(input, self.scale, self.N)
        return x


class UnShuffle1d(UnShuffleNd):
    def __init__(self, scale):
        super(UnShuffle1d, self).__init__(scale, 1)


class UnShuffle2d(UnShuffleNd):
    def __init__(self, scale):
        super(UnShuffle2d, self).__init__(scale, 2)


class UnShuffle3d(UnShuffleNd):
    def __init__(self, scale):
        super(UnShuffle3d, self).__init__(scale, 3)


class Shuffle1d(ShuffleNd):
    def __init__(self, scale):
        super(Shuffle1d, self).__init__(scale, 1)
        self.scale = scale


class Shuffle2d(ShuffleNd):
    def __init__(self, scale):
        super(Shuffle2d, self).__init__(scale, 2)
        self.scale = scale


class Shuffle3d(ShuffleNd):
    def __init__(self, scale):
        super(Shuffle3d, self).__init__(scale, 3)
        self.scale = scale