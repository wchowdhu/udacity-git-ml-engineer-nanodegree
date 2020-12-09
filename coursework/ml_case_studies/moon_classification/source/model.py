import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        hidden_1 = hidden_dim
        hidden_2 = hidden_dim
        # linear Hidden layer-1 (input_dim -> hidden_1)
        self.fc1 = nn.Linear(input_dim, hidden_1)
        # linear Hidden layer-2 (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # Output layer (n_hidden -> output_dim)
        self.fc3 = nn.Linear(hidden_2, output_dim)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)  
        # sigmoid layer
        self.sig = nn.Sigmoid()
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        # add hidden layer-1, with relu activation function
        hd_1 = F.relu(self.fc1(x)) #[batch_size, hidden_1]
        # add dropout layer
        hd_1_dropout = self.dropout(hd_1)
        # add hidden layer-2, with relu activation function
        hd_2 = F.relu(self.fc2(hd_1_dropout)) #[batch_size, hidden_2]
        # add dropout layer
        hd_2_dropout = self.dropout(hd_2)
        # add output layer
        out = self.fc3(hd_2_dropout) #[batch_size, output_dim]
        return self.sig(out) #[batch_size, output_dim]