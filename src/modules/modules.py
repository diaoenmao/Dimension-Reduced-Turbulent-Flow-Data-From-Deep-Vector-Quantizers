import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(nn.Module):
    def __init__(self, embedding_size, num_embedding, vq_commit):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_size)
        # self.embedding.weight.data.uniform_(-1.0 / self.num_embedding, 1.0 / self.num_embedding)
        self.vq_commit = vq_commit

    def forward(self, input):
        input = input.transpose(1, -1).contiguous()
        flatten = input.view(-1, self.embedding_size)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(dim=1)
        )
        _, embedding_ind = dist.min(1)
        embedding_ind = embedding_ind.view(*input.shape[:-1])
        quantize = self.embedding_code(embedding_ind)
        diff = self.vq_commit * F.mse_loss(quantize.detach(), input) + F.mse_loss(quantize, input.detach())
        quantize = input + (quantize - input).detach()
        quantize = quantize.transpose(1, -1).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return self.embedding(embedding_ind)