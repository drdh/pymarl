import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence
import torch.distributions as D


class LatentEvolveRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentEvolveRNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0

        # pi_param = th.rand(args.n_agents)
        # pi_param = pi_param / pi_param.sum()
        # self.pi_param = nn.Parameter(pi_param)

        # mu_param = th.randn(args.n_agents, args.latent_dim)
        # mu_param = mu_param / mu_param.norm(dim=0)
        # self.mu_param = nn.Parameter(mu_param)

        #self.embed_fc = nn.Linear(args.n_agents, args.latent_dim * 2)
        self.inference_fc1 = nn.Linear(args.rnn_hidden_dim + input_shape - args.n_agents, args.latent_dim * 4)
        self.inference_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim)

        self.latent = th.rand(args.n_agents, args.latent_dim)  # (n,mu)

        # self.latent_fc1 = nn.Linear(args.latent_dim, args.latent_dim * 4)
        # self.latent_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim * 4)
        # self.latent_fc3 = nn.Linear(args.latent_dim, args.latent_dim)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # self.fc1_w_nn=nn.Linear(args.latent_dim,input_shape*args.rnn_hidden_dim)
        # self.fc1_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        # self.rnn_ih_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        # self.rnn_ih_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)
        # self.rnn_hh_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        # self.rnn_hh_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(args.latent_dim, args.rnn_hidden_dim * args.n_actions, bias=False)
        self.fc2_b_nn = nn.Linear(args.latent_dim, args.n_actions,bias=False)

    def init_latent(self, bs):
        self.bs = bs
        loss = 0

        return loss, self.latent.detach()

    def forward(self, inputs, hidden_state):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        #self.latent = self.embed_fc(inputs[:self.n_agents, - self.n_agents:])  # (n,2*latent_dim)==(n,mu+log var)
        #self.latent[:, -self.latent_dim:] = th.exp(self.latent[:, -self.latent_dim:])  # var
        #latent_embed = self.latent.unsqueeze(0).expand(self.bs, self.n_agents, self.latent_dim * 2).reshape(
        #    self.bs * self.n_agents, self.latent_dim * 2)

        latent_infer = F.relu(self.inference_fc1(th.cat([h_in.detach(), inputs[:, :-self.n_agents]], dim=1)))
        latent_infer = self.inference_fc2(latent_infer)  # (bs*n,latent_dim)
        #latent_infer[:, -self.latent_dim:] = th.exp(latent_infer[:, -self.latent_dim:])

        # sample
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:])**(1/2))
        gaussian_infer = D.Normal(latent_infer[:, :self.latent_dim], (latent_infer[:, self.latent_dim:])**(1/2))

        #loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed, gaussian_infer).sum()  # CE = H + KL
        #loss = loss / (self.bs*self.n_agents)
        loss=0
        # handcrafted reparameterization
        # (1,n*latent_dim)                            (1,n*latent_dim)==>(bs,n*latent*dim)
        # latent_embed = self.latent[:,:self.latent_dim].reshape(1,-1)+self.latent[:,-self.latent_dim:].reshape(1,-1)*th.randn(self.bs,self.n_agents*self.latent_dim)
        # latent_embed = latent_embed.reshape(-1,self.latent_dim)  #(bs*n,latent_dim)
        # latent_infer = latent_infer[:, :self.latent_dim] + latent_infer[:, -self.latent_dim:] * th.randn_like(latent_infer[:, -self.latent_dim:])
        # loss= (latent_embed-latent_infer).norm(dim=1).sum()/(self.bs*self.n_agents)


        #latent = F.relu(self.latent_fc1(latent))
        #latent = (self.latent_fc2(latent))

        # latent=latent.reshape(-1,self.args.latent_dim)

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

        fc2_w = self.fc2_w_nn(latent)
        fc2_b = self.fc2_b_nn(latent)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

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
        h = h.reshape(-1, 1, self.args.rnn_hidden_dim)

        q = th.bmm(h, fc2_w) + fc2_b

        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        # h = self.rnn(x, h_in)
        # q = self.fc2(h)
        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)
