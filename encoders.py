import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import utils

class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, opts):
        super(GRUEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.opts = opts

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(inputs)
        annotations = []

        for i in range(seq_len):
            x = encoded[:,i,:]
            hidden = self.gru(x, hidden)
            annotations.append(hidden)

        annotations = torch.stack(annotations, dim=1)

        return annotations, hidden

    def init_hidden(self, batch_size):
        return utils.to_var(torch.zeros(batch_size, self.hidden_size), self.opts.cuda)