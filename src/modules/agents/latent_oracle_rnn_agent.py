import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
from modules.agents import snail_blocks as snail
import math

class LatentOracleRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentOracleRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0

        self.embed_fc_input_size = args.own_feature_size
        self.latent = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)

        self.latent_fc1 = nn.Linear(args.latent_dim, args.latent_dim * 4)
        self.latent_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim * 4)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(self.embed_fc_input_size, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(self.embed_fc_input_size, args.n_actions)

    def init_latent(self, bs):
        self.bs = bs
        loss = 0
        # end

        return loss, self.latent[:self.n_agents,:].detach()

    def forward(self, inputs, hidden_state, t=0, batch=None):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        embed_fc_input = inputs[:, - self.embed_fc_input_size:] #own features(unit_type_bits+shield_bits_ally)+id

        fc2_w = self.fc2_w_nn(embed_fc_input)
        fc2_b = self.fc2_b_nn(embed_fc_input)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        h = self.rnn(x, h_in)
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b

        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), 0
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)
