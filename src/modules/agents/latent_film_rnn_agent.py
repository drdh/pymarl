import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D


class LatentFiLMRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentFiLMRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0
        self.embed_fc_input_size = args.own_feature_size

        self.latent = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)

        self.embed_fc = nn.Linear(self.embed_fc_input_size, args.rnn_hidden_dim * 2)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.fc1_w_nn=nn.Linear(args.latent_dim,input_shape*args.rnn_hidden_dim)
        # self.fc1_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        # self.rnn_ih_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        # self.rnn_ih_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)
        # self.rnn_hh_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        # self.rnn_hh_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        #self.fc2_w_nn = nn.Linear(args.latent_dim * 4, args.rnn_hidden_dim * args.n_actions)
        #self.fc2_b_nn = nn.Linear(args.latent_dim * 4, args.n_actions)

    def init_latent(self, bs):
        self.bs = bs
        loss = 0
        # end

        return loss, self.latent.detach()


    def forward(self, inputs, hidden_state,t=0, batch=None):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        embed_fc_input = inputs[:, - self.embed_fc_input_size:]  # own features(unit_type_bits+shield_bits_ally)+id
        latent = self.embed_fc(embed_fc_input)
        gamma_z = latent[:,:self.hidden_dim]
        beta_z = latent[:,self.hidden_dim:]

        # fc1_w=F.relu(self.fc1_w_nn(latent))
        # fc1_b=F.relu((self.fc1_b_nn(latent)))
        # fc1_w=fc1_w.reshape(-1,self.input_shape,self.args.rnn_hidden_dim)
        # fc1_b=fc1_b.reshape(-1,1,self.args.rnn_hidden_dim)

        # rnn_ih_w=F.relu(self.rnn_ih_w_nn(latent))
        # rnn_ih_b=F.relu(self.rnn_ih_b_nn(latent))
        # rnn_hh_w=F.relu(self.rnn_hh_w_nn(latent))
        # rnn_hh_b=F.relu(self.rnn_hh_b_nn(latent))
        # rnn_ih_w=rnn_ih_w.reshape(-1,self.args.rnn_hidden_dim,self.args.rnn_hidden_dim)
        # rnn_ih_b=rnn_ih_b.reshape(-1,1,self.args.rnn_hidden_dim)
        # rnn_hh_w = rnn_hh_w.reshape(-1, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        # rnn_hh_b = rnn_hh_b.reshape(-1, 1, self.args.rnn_hidden_dim)

        #fc2_w = self.fc2_w_nn(latent)
        #fc2_b = self.fc2_b_nn(latent)
        #fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        #fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        # x=F.relu(th.bmm(inputs,fc1_w)+fc1_b) #(bs*n,(obs+act+id)) at time t
        x = F.relu(self.fc1(inputs))  # (bs*n,(obs+act+id)) at time t

        # gi=th.bmm(x,rnn_ih_w)+rnn_ih_b
        # gh=th.bmm(h_in,rnn_hh_w)+rnn_hh_b
        # i_r,i_i,i_n=gi.chunk(3,2)
        # h_r,h_i,h_n=gh.chunk(3,2)

        # resetgate=th.sigmoid(i_r+h_r)
        # inputgate=th.sigmoid(i_i+h_i)
        # newgate=th.tanh(i_n+resetgate*h_n)
        # h=newgate+inputgate*(h_in-newgate)
        # h=th.tanh(gi+gh)

        # x=x.reshape(-1,self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        #h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        #q = th.bmm(h, fc2_w) + fc2_b

        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        # h = self.rnn(x, h_in)
        # q = self.fc2(h)
        q = self.fc2(gamma_z*h+beta_z) #FiLM

        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), 0
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)
