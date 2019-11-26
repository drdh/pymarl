import torch.nn as nn
import torch.nn.functional as F
import torch as th


class LatentMixtureInputRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentMixtureInputRNNAgent, self).__init__()
        self.args = args
        self.input_shape=input_shape
        self.n_agents=args.n_agents
        self.latent_dim=args.latent_dim

        pi_param = th.rand(args.n_agents)
        #pi_param = pi_param / pi_param.sum()
        self.pi_param = nn.Parameter(pi_param)

        mu_param = th.randn(args.n_agents, args.latent_dim)
        #mu_param = mu_param / mu_param.norm(dim=0)
        self.mu_param = nn.Parameter(mu_param)

        preference = th.randn(args.n_agents, args.latent_dim)
        self.preference = nn.Parameter(preference)

        self.fc1 = nn.Linear(input_shape+self.latent_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_latent(self,bs):
        #self.latent=(self.mu_param + th.rand_like(self.mu_param)).unsqueeze(0).expand(bs,self.n_agents,self.latent_dim).reshape(-1,self.latent_dim)
        # (bs*n,latent_dim)

        #KL_neg=(self.mu_param-th.tensor([-5.0]*self.latent_dim)).norm(dim=1)
        #KL_pos=(self.mu_param-th.tensor([ 5.0]*self.latent_dim)).norm(dim=1)
        #loss=th.stack([KL_neg,KL_pos]).min(dim=0)[0].sum()

        pi_param=self.pi_param / self.pi_param.sum()
        mu_param = self.mu_param / self.mu_param.norm(dim=1).unsqueeze(1)
        preference = self.preference / self.preference.norm(dim=1).unsqueeze(1)

        u = th.rand(self.n_agents, self.n_agents)
        g = - th.log(- th.log(u))
        c = (g + th.log(pi_param)).argmax(dim=1)

        task_role = (mu_param[c] + th.randn_like(mu_param)*(1.0/self.n_agents)) #(n,latent_dim)
        #task_role = mu_param[c]
                   #      (n,1,latent_dim)            # (1,n,latent_dim)
        index = (self.preference.unsqueeze(1) * task_role.unsqueeze(0)).norm(dim=2).max(dim=1)[1]
        # (n,n,2) => (n,n) ==> (n,)

        self.latent = task_role[index].unsqueeze(0).expand(bs, self.n_agents,
                                                                            self.latent_dim).reshape(-1,
                                                                                                      self.latent_dim)
        #(bs*n, latent_dim)

        loss= -(mu_param.norm(dim=1)*pi_param).sum() # KL, N(0,1)

        #self.latent = (mu_param[c] + th.randn_like(mu_param)).unsqueeze(0).expand(bs, self.n_agents,
        #                                                                                    self.latent_dim).reshape(-1,
        #                                                                                                             self.latent_dim)
        # (bs*n,latent_dim)

        #self.latent = self.latent / self.latent.norm(dim=0)

        #mu_distance = (mu_param.unsqueeze(1) - mu_param.unsqueeze(0)).norm(dim=2)
        #distance_weight = pi_param.unsqueeze(0) + pi_param.unsqueeze(1)
        #loss = (distance_weight * mu_distance).sum()

        return loss,th.cat([pi_param.data.detach().reshape(-1,1),mu_param.data.detach()],dim=1)

        # (bs*n,(obs+act+id)), (bs,n,hidden_dim)
    def forward(self, inputs, hidden_state):

        inputs_cat=th.cat([inputs,self.latent],dim=1)

        x = F.relu(self.fc1(inputs_cat)) #(bs*n,(obs+act+id)) at time t
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        #inputs = inputs.reshape(-1, 1, self.input_shape)
        #h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # (bs*n,hidden_dim)
        #latent = self.latent.reshape(-1, 1, self.args.latent_dim)  # (bs*n,1,latent_dim)

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

        #fc2_w=F.relu(self.fc2_w_nn(latent))
        #fc2_b=F.relu(self.fc2_b_nn(latent))
        #fc2_w=fc2_w.reshape(-1,self.args.rnn_hidden_dim,self.args.n_actions)
        #fc2_b=fc2_b.reshape((-1,1,self.args.n_actions))

        #x=F.relu(th.bmm(inputs,fc1_w)+fc1_b) #(bs*n,(obs+act+id)) at time t


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
        #h = self.rnn(x, h_in)
        #h=h.reshape(-1,1,self.args.rnn_hidden_dim)

        #q=F.relu(th.bmm(h,fc2_w)+fc2_b)

        return q.view(-1,self.args.n_actions), h.view(-1,self.args.rnn_hidden_dim)
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)