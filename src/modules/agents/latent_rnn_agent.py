import torch.nn as nn
import torch.nn.functional as F


class LatentRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

                    # (bs*n,(obs+act+id)), (bs,n,hidden_size)
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs)) #(bs*n,(obs+act+id)) at time t
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
        #(bs*n,n_actions), (bs*n,hidden_dim)
# TODO: complete it