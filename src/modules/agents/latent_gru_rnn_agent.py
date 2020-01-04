import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
from tensorboardX import SummaryWriter
import time

class LatentGRURNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentGRURNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = args.batch_size_run

        # Embed Network
        self.embed_fc = nn.Linear(input_shape, args.latent_dim * 2)
        # Latent GRU
        self.latent_rnn = nn.GRUCell(args.latent_dim*2, args.latent_dim*2)

        self.latent = th.rand(self.bs*args.n_agents, args.latent_dim * 2)  # (n,mu+var)
        self.latent_hist = th.rand(self.bs*args.n_agents, args.latent_dim * 2)

        # latent -> FC2 parameter
        self.latent_fc1 = nn.Linear(args.latent_dim, args.latent_dim * 4)
        self.latent_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim * 4)
        self.fc2_w_nn = nn.Linear(args.latent_dim * 4, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(args.latent_dim * 4, args.n_actions)

        # GRU & FC1
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_latent(self, bs):
        self.bs = max(1, bs)
        loss = 0
        if self.args.runner == "episode":
            self.writer = SummaryWriter("results/tb_logs/test_latent-"+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) +"/latent")
        return loss, self.latent[:self.n_agents,:].detach(),self.latent_hist[:self.n_agents,:]

    def forward(self, inputs, hidden_state,t=0,batch=None, test_mode=None):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # Obs to role
        latent_last = self.latent.detach()
        self.latent = self.embed_fc(inputs.detach())  # (n,2*latent_dim)==(n,mu+log var)
        self.latent[:, self.latent_dim:] = th.exp(self.latent[:, self.latent_dim:])  # var

        gaussian_embed = D.Normal(self.latent[:, :self.latent_dim], self.latent[:, self.latent_dim:])
        latent = gaussian_embed.rsample()  # Sample a role

        if t==0:
            self.latent_hist=self.latent.detach()
            loss = 0
        else:
            self.latent_hist = self.latent_rnn(latent_last, self.latent_hist)
            self.latent_hist[:, self.latent_dim:] = th.exp(self.latent_hist[:, self.latent_dim:])
            gaussian_hist = D.Normal(self.latent_hist[:, :self.latent_dim], self.latent_hist[:, self.latent_dim:])
            loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed, gaussian_hist).sum()  # CE = H + KL
            loss = loss / (self.bs * self.n_agents)
            loss = th.log(1 + th.exp(loss))

        # Role -> FC2 Params
        latent = F.relu(self.latent_fc1(latent))
        latent = (self.latent_fc2(latent))
        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        # Forward pass of Q_i
        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t
        h = self.rnn(x, h_in)
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        q = th.bmm(h, fc2_w) + fc2_b

        if self.args.runner=="episode":
            self.writer.add_embedding(self.latent.reshape(-1,self.latent_dim*2),list(range(self.args.n_agents)),global_step=t,tag="latent-step")

        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss