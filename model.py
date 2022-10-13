# Looking at https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py for guidance on setting up the model

# Also referred to https://github.com/jeffchy/pytorch-word-embedding/blob/master/CBOW.py to figure out how to sum individual word embeddings

import torch


class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        # Initialize Variabls
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Create Embedding Layer
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)

        # Create a Liner layer that will output the predicted word
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs, labels):
        # Create embeddings for each individual context word and then sum
        embeds = torch.sum(self.embed(inputs), dim=1)

        # output is then a linear combination of these
        out = self.linear(embeds)

        return out
