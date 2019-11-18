import numpy as np
from utils import ntuple


def unshuffleNd(input, factor, N):
    _tuple = ntuple(N)
    factor = _tuple(factor)
    input_size = input.size()
    reshape_size = [*input_size[:2]]
    for i in range(N):
        reshape_size.extend([input_size[i + 2] // factor[i], factor[i]])
    reshape_input = input.reshape(reshape_size)
    permute_order = [0, 1] + [2 * (i + 1) + 1 for i in range(N)] + [2 * (i + 1) for i in range(N)]
    permute_input = reshape_input.permute(permute_order)
    output_shape = [input_size[2:][i] // factor[i] for i in range(N)]
    output_size = [input_size[0], input_size[1] * np.prod(factor), *output_shape]
    output = permute_input.reshape(output_size)
    return output


def shuffleNd(input, factor, N):
    _tuple = ntuple(N)
    factor = _tuple(factor)
    input_size = input.size()
    reshape_size = [input_size[0], input_size[1] // np.prod(factor), *factor, *input_size[2:]]
    reshape_input = input.reshape(reshape_size)
    permute_order = [0, 1]
    for i in range(N):
        permute_order.extend([N + i + 2, i + 2])
    permute_input = reshape_input.permute(permute_order)
    output_shape = [input_size[2:][i] * factor[i] for i in range(N)]
    output_size = [input_size[0], input_size[1] // np.prod(factor), *output_shape]
    output = permute_input.reshape(output_size)
    return output