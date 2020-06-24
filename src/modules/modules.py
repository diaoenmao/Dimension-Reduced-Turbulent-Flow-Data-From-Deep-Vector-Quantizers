import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization3d(nn.Module):
    def __init__(self, embedding_size, num_embedding, decay=0.99, eps=1e-5):
        super(VectorQuantization3d, self).__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())

    def forward(self, input):
        input = input.permute(0, 2, 3, 4, 1).contiguous()
        flatten = input.view(-1, self.embedding_size)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embedding_ind = dist.min(1)
        embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(flatten.dtype)
        embedding_ind = embedding_ind.view(*input.shape[:-1])
        quantize = self.embedding_code(embedding_ind)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embedding_onehot.sum(0)
            )
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_mean.data.mul_(self.decay).add_(1 - self.decay, embedding_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)
        diff = F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        quantize = quantize.permute(0, 4, 1, 2, 3).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1))