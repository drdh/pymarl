import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import MultivariateNormal
from torch.distributions import kl_divergence


class LatentRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentRNNAgent, self).__init__()
        self.args = args
        self.input_shape=input_shape
        self.n_agents=args.n_agents
        self.latent_dim=args.latent_dim
        self.hidden_dim=args.rnn_hidden_dim
        self.bs=0

        #pi_param = th.rand(args.n_agents)
        #pi_param = pi_param / pi_param.sum()
        #self.pi_param = nn.Parameter(pi_param)

        #mu_param = th.randn(args.n_agents, args.latent_dim)
        #mu_param = mu_param / mu_param.norm(dim=0)
        #self.mu_param = nn.Parameter(mu_param)

        self.embed_fc = nn.Linear(args.n_agents, args.latent_dim*2)
        self.inference_fc1 = nn.Linear(args.rnn_hidden_dim+input_shape-args.n_agents, args.latent_dim*4)
        self.inference_fc2 = nn.Linear(args.latent_dim*4, args.latent_dim*2)

        self.latent = th.rand(args.n_agents,args.latent_dim*2) #(n,mu+var)

        self.latent_fc1 = nn.Linear(args.latent_dim, args.latent_dim*4)
        self.latent_fc2 = nn.Linear(args.latent_dim*4, args.latent_dim*4)
        #self.latent_fc3 = nn.Linear(args.latent_dim, args.latent_dim)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        #self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        #self.fc1_w_nn=nn.Linear(args.latent_dim,input_shape*args.rnn_hidden_dim)
        #self.fc1_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        #self.rnn_ih_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        #self.rnn_ih_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)
        #self.rnn_hh_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim)
        #self.rnn_hh_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        self.fc2_w_nn=nn.Linear(args.latent_dim*4,args.rnn_hidden_dim*args.n_actions)
        self.fc2_b_nn=nn.Linear(args.latent_dim*4,args.n_actions)


    def init_latent(self,bs):
        self.bs=bs
        #self.latent=(self.mu_param + th.rand_like(self.mu_param)).unsqueeze(0).expand(bs,self.n_agents,self.latent_dim).reshape(-1,self.latent_dim)

        #KL_neg=(self.mu_param-th.tensor([-5.0]*self.latent_dim)).norm(dim=1)
        #KL_pos=(self.mu_param-th.tensor([ 5.0]*self.latent_dim)).norm(dim=1)
        #loss=th.stack([KL_neg,KL_pos]).min(dim=0)[0].sum()

        # oracle version for decoder, 3s5z
        #role_s = th.randn(3,self.latent_dim)+5.0
        #role_z = th.randn(5,self.latent_dim)-5.0
        #self.latent = th.cat([
        #    role_s,
        #    role_z
        #],dim=0).unsqueeze(0).expand(bs, self.n_agents, self.latent_dim).reshape(-1, self.latent_dim)


        #self.latent = F.relu(self.latent_fc1(self.latent))
        #self.latent = F.relu(self.latent_fc2(self.latent))
        #self.latent = F.relu(self.latent_fc3(self.latent))
        loss = 0
        # end

        return loss,self.latent.detach()

        #u = th.rand(self.n_agents, self.n_agents)
        #g = - th.log(- th.log(u))
        #c = (g + th.log(self.pi_param)).argmax(dim=1)

        #self.latent = (self.mu_param[c] + th.randn_like(self.mu_param)).unsqueeze(0).expand(bs, self.n_agents,
        #                                                                                    self.latent_dim).reshape(-1,
        #                                                                                                             self.latent_dim)
        #self.latent = self.latent / self.latent.norm(dim=0)

        #mu_distance = (self.mu_param.unsqueeze(1) - self.mu_param.unsqueeze(0)).norm(dim=2)
        #distance_weight = self.pi_param.unsqueeze(0) + self.pi_param.unsqueeze(1)
        #loss = (distance_weight * mu_distance).sum()

        # print(self.mu_param)

        #return loss

        # (bs*n,(obs+act+id)), (bs,n,hidden_dim), (bs,n,latent_dim)
    def forward(self, inputs, hidden_state):
        inputs=inputs.reshape(-1,self.input_shape)
        h_in=hidden_state.reshape(-1,self.hidden_dim)

        self.latent = self.embed_fc(inputs[:self.n_agents, - self.n_agents:]) #(n,2*latent_dim)==(n,mu+log var)
        self.latent[:,-self.latent_dim:] = th.exp(self.latent[:,-self.latent_dim:]) #var
        latent_embed = self.latent.unsqueeze(0).expand(self.bs, self.n_agents, self.latent_dim * 2).reshape(self.bs * self.n_agents, self.latent_dim * 2)

        latent_infer = F.relu(self.inference_fc1(th.cat([h_in,inputs[:,:-self.n_agents]],dim=1)))
        latent_infer = self.inference_fc2(latent_infer)                        #(n,2*latent_dim)==(n,mu+log var)
        latent_infer[:, -self.latent_dim:] = th.exp(latent_infer[:, -self.latent_dim:])

        #sample
        #gaussian_embed = MultivariateNormal(latent_embed[:, :self.latent_dim],th.diag_embed(latent_embed[:, self.latent_dim:])) #for torch 1.3.1
        #gaussian_infer = MultivariateNormal(latent_infer[:, :self.latent_dim],th.diag_embed(latent_infer[:, self.latent_dim:]))
        var_embed = th.empty(self.bs*self.n_agents,self.latent_dim,self.latent_dim)
        var_infer = th.empty(self.bs*self.n_agents,self.latent_dim,self.latent_dim)
        for i in range(self.bs*self.n_agents):
            var_embed[i]=th.diag(latent_embed[i, self.latent_dim:])
            var_infer[i]=th.diag(latent_infer[i, self.latent_dim:])
        gaussian_embed = MultivariateNormal(latent_embed[:, :self.latent_dim],var_embed)
        gaussian_infer = MultivariateNormal(latent_infer[:, :self.latent_dim],var_infer)

        latent=gaussian_embed.rsample()
        loss=gaussian_embed.entropy().sum()+kl_divergence(gaussian_embed,gaussian_infer).sum() #CE = H + KL

        #handcrafted reparameterization
                                                         #(1,n*latent_dim)                            (1,n*latent_dim)==>(bs,n*latent*dim)
        #latent_embed = self.latent[:,:self.latent_dim].reshape(1,-1)+self.latent[:,-self.latent_dim:].reshape(1,-1)*th.randn(self.bs,self.n_agents*self.latent_dim)
        #latent_embed = latent_embed.reshape(-1,self.latent_dim)  #(bs*n,latent_dim)
        #latent_infer = latent_infer[:, :self.latent_dim] + latent_infer[:, -self.latent_dim:] * th.randn_like(latent_infer[:, -self.latent_dim:])
        #loss= (latent_embed-latent_infer).norm(dim=1).sum()/(self.bs*self.n_agents)

        latent = F.relu(self.latent_fc1(latent))
        latent = F.relu(self.latent_fc2(latent))

        #latent=latent.reshape(-1,self.args.latent_dim)

        #fc1_w=F.relu(self.fc1_w_nn(latent))
        #fc1_b=F.relu((self.fc1_b_nn(latent)))
        #fc1_w=fc1_w.reshape(-1,self.input_shape,self.args.rnn_hidden_dim)
        #fc1_b=fc1_b.reshape(-1,1,self.args.rnn_hidden_dim)

        #rnn_ih_w=F.relu(self.rnn_ih_w_nn(latent))
        #rnn_ih_b=F.relu(self.rnn_ih_b_nn(latent))
        #rnn_hh_w=F.relu(self.rnn_hh_w_nn(latent))
        #rnn_hh_b=F.relu(self.rnn_hh_b_nn(latent))
        #rnn_ih_w=rnn_ih_w.reshape(-1,self.args.rnn_hidden_dim,self.args.rnn_hidden_dim)
        #rnn_ih_b=rnn_ih_b.reshape(-1,1,self.args.rnn_hidden_dim)
        #rnn_hh_w = rnn_hh_w.reshape(-1, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim)
        #rnn_hh_b = rnn_hh_b.reshape(-1, 1, self.args.rnn_hidden_dim)

        fc2_w=self.fc2_w_nn(latent)
        fc2_b=self.fc2_b_nn(latent)
        fc2_w=fc2_w.reshape(-1,self.args.rnn_hidden_dim,self.args.n_actions)
        fc2_b=fc2_b.reshape((-1,1,self.args.n_actions))

        #x=F.relu(th.bmm(inputs,fc1_w)+fc1_b) #(bs*n,(obs+act+id)) at time t
        x = F.relu(self.fc1(inputs)) #(bs*n,(obs+act+id)) at time t

        #gi=th.bmm(x,rnn_ih_w)+rnn_ih_b
        #gh=th.bmm(h_in,rnn_hh_w)+rnn_hh_b
        #i_r,i_i,i_n=gi.chunk(3,2)
        #h_r,h_i,h_n=gh.chunk(3,2)

        #resetgate=th.sigmoid(i_r+h_r)
        #inputgate=th.sigmoid(i_i+h_i)
        #newgate=th.tanh(i_n+resetgate*h_n)
        #h=newgate+inputgate*(h_in-newgate)
        #h=th.tanh(gi+gh)

        #x=x.reshape(-1,self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        h=h.reshape(-1,1,self.args.rnn_hidden_dim)

        q=th.bmm(h,fc2_w)+fc2_b


        #h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        #h = self.rnn(x, h_in)
        #q = self.fc2(h)
        return q.view(-1,self.args.n_actions), h.view(-1,self.args.rnn_hidden_dim),loss
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)