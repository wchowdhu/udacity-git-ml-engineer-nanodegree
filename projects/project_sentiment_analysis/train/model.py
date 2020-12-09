import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1, drop_prob=0.0, bidirectional=False):
        """
        Initialize the model by setting up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=drop_prob, bidirectional=bidirectional)
        if bidirectional:
            #https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            self.dense = nn.Linear(in_features=hidden_dim * 2, out_features=1) #linear transformation to input data: y = xA^T + b
        else:
            self.dense = nn.Linear(in_features=hidden_dim, out_features=1) #linear transformation to input data: y = xA^T + b
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t() #transpose input x
        lengths = x[0,:] #shape [#batch_size]
        reviews = x[1:,:] #shape [#words, #batch_size] e.g. (seq_len, batch)
        embeds = self.embedding(reviews) #(seq_len, batch, embed_size)
        lstm_out, _ = self.lstm(embeds) #(seq_len, batch, hidden_size*num_directions)
        out = self.dense(lstm_out) #(seq_len, batch, 1)
        out = out[lengths - 1, range(len(lengths))] #(batch, 1) #*
        return self.sig(out.squeeze()) #.squeeze() returns a tensor with all the dimensions of input of size 1 removed. shape: #(batch)