import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantization(nn.Module):
    def __init__(self, num_embedding, embedding_dim, ema=True, commitment=1, decay=0.99, epsilon=1e-6):
        super(Quantization, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.commitment = commitment
        self.ema = ema
        self.register_buffer('N', torch.zeros(num_embedding))
        self.ema_w = nn.Parameter(torch.Tensor(num_embedding, embedding_dim))
        self.ema_w.data.normal_()
        self.decay = decay
        self.epsilon = epsilon

    def forward(self, input):
        permutation = [0] + [2 + i for i in range(input.dim() - 2)] + [1]
        x = input.permute(permutation)
        flat_x = x.reshape(-1, self.embedding_dim)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        encoding = torch.argmin(distances, dim=1)
        if self.training:
            if self.ema:
                one_hot = F.one_hot(encoding, num_classes=self.num_embedding)
                self.N = self.N * self.decay + (1 - self.decay) * torch.sum(one_hot, dim=0)
                N_i = torch.sum(self.N)
                self.N = ((self.N + self.epsilon) / (N_i + self.epsilon * self.num_embedding) * N_i)
                dw = torch.matmul(one_hot.float().t(), flat_x)
                nw = self.ema_w * self.decay + (1 - self.decay) * dw
                self.ema_w = nn.Parameter(nw.data)
                self.embedding.weight = nn.Parameter(self.ema_w / self.N.unsqueeze(1))
        encoding = encoding.reshape(x.size()[:-1])
        quantized = self.embedding(encoding)
        unpermutation = [0, input.dim()-1] + [1 + i for i in range(input.dim() - 2)]
        quantized = quantized.permute(unpermutation)
        e_latent_loss = F.mse_loss(quantized.detach(), input)
        if self.ema:
            loss = self.commitment * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(quantized, input.detach())
            loss = q_latent_loss + self.commitment * e_latent_loss
        quantized = input + (quantized - input).detach()
        return quantized, encoding, distances, loss