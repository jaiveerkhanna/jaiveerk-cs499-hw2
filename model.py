# Looking at https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py for guidance on setting up the model

import torch


class CBOW(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        # Initialize Variabls
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Create Embedding Layer
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
