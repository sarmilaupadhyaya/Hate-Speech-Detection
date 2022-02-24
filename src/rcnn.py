import torch
import torch.nn as nn
import torch.nn.functional as F






class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, hidden_size_linear, class_num, dropout,padding_idx):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.cnn_layer1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 300,kernel_size=1), #1 bc text
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1))

        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.W = nn.Linear(embedding_dim + 2*hidden_size, hidden_size_linear)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size_linear, class_num)

    def forward(self, x):
        print(x.shape)
        x_emb = self.embedding(x)
        # x_emb = |batch size, seq_len, embedding_dim|
        x_emb = x_emb.permute(0,2,1) # Have to switch
        x_emb = self.cnn_layer1(x_emb) # x = |batch_size, channels (= embedding_dim), seq_length|
        x_emb = x_emb.permute(0,2,1) # Have to switch again
        output, _ = self.lstm(x_emb)
        # output = |bs, seq_len, 2*hidden_size|
        output = torch.cat([output, x_emb], 2)
        # output = |bs, seq_len, embedding_dim + 2*hidden_size|
        output = self.tanh(self.W(output)).transpose(1, 2)
        # output = |bs, seq_len, hidden_size_linear| -> |bs, hidden_size_linear, seq_len|
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        # output = |bs, hidden_size_linear|
        output = self.fc(output)
        # output = |bs, class_num|
        return output

