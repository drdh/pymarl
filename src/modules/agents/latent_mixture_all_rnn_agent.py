import torch.nn as nn
import torch.nn.functional as F
import torch as th


class LatentMixtureAllRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentMixtureAllRNNAgent, self).__init__()
        self.args = args
        self.input_shape=input_shape
        self.n_agents=args.n_agents
        self.latent_dim=args.latent_dim
        self.bs=0

        pi_param = th.rand(args.n_agents)
        # pi_param = pi_param / pi_param.sum()
        self.pi_param = nn.Parameter(pi_param)

        mu_param = th.randn(args.n_agents, args.latent_dim)
        # mu_param = mu_param / mu_param.norm(dim=0)
        self.mu_param = nn.Parameter(mu_param)

        self.latent = None

        self.fc_latent = nn.Linear(args.latent_dim * args.n_agents, args.latent_dim * args.n_agents)

        self.fc1_w_nn=nn.Linear(args.latent_dim,input_shape*args.rnn_hidden_dim)
        self.fc1_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim)

        self.rnn_ih_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim*3)
        self.rnn_ih_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*3)
        self.rnn_hh_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.rnn_hidden_dim*3)
        self.rnn_hh_b_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*3)

        self.fc2_w_nn=nn.Linear(args.latent_dim,args.rnn_hidden_dim*args.n_actions)
        self.fc2_b_nn=nn.Linear(args.latent_dim,args.n_actions)


    def init_latent(self,bs):
        self.bs=bs
        pi_param = self.pi_param / self.pi_param.sum()
        mu_param = self.mu_param / self.mu_param.norm(dim=1).unsqueeze(1)

        u = th.rand(self.n_agents, self.n_agents)
        g = - th.log(- th.log(u))
        c = (g + th.log(pi_param)).argmax(dim=1)

        task_role = (mu_param[c] + th.randn_like(mu_param) * (1.0 / self.n_agents))  # (n,latent_dim)
        # task_role = mu_param[c]

        task_role = th.tensor([
            [5, 5],
            [5, 5],
            [5, 5],
            [-5, -5],
            [-5, -5],
            [-5, -5],
            [-5, -5],
            [-5, -5]
        ], dtype=th.float)

        self.latent = task_role.reshape(self.args.n_agents, self.args.latent_dim)
        self.latent = self.latent + th.randn_like(self.latent)
        # (n,latent_dim)
        self.latent = self.latent / self.latent.norm(dim=1).unsqueeze(1)  # (n,latent_dim)

        latent = self.latent.reshape(-1, 1, self.args.latent_dim)  # (n,1,latent_dim)

        fc1_w = F.relu(self.fc1_w_nn(latent))
        fc1_b = F.relu((self.fc1_b_nn(latent)))
        self.fc1_w = fc1_w.reshape(-1, self.input_shape, self.args.rnn_hidden_dim)
        self.fc1_b = fc1_b.reshape(-1, 1, self.args.rnn_hidden_dim)

        rnn_ih_w = F.relu(self.rnn_ih_w_nn(latent))
        rnn_ih_b = F.relu(self.rnn_ih_b_nn(latent))
        rnn_hh_w = F.relu(self.rnn_hh_w_nn(latent))
        rnn_hh_b = F.relu(self.rnn_hh_b_nn(latent))
        self.rnn_ih_w = rnn_ih_w.reshape(-1, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim * 3)
        self.rnn_ih_b = rnn_ih_b.reshape(-1, 1, self.args.rnn_hidden_dim * 3)
        self.rnn_hh_w = rnn_hh_w.reshape(-1, self.args.rnn_hidden_dim, self.args.rnn_hidden_dim * 3)
        self.rnn_hh_b = rnn_hh_b.reshape(-1, 1, self.args.rnn_hidden_dim * 3)

        fc2_w = F.relu(self.fc2_w_nn(latent))
        fc2_b = F.relu(self.fc2_b_nn(latent))
        self.fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.args.n_actions)
        self.fc2_b = fc2_b.reshape((-1, 1, self.args.n_actions))

        #(1,n*latent_dim)
        #self.latent = (self.fc_latent(task_role.reshape(1,-1))).reshape(self.args.n_agents,self.args.latent_dim)
        #self.latent = task_role.reshape(self.args.n_agents,self.args.latent_dim)
        #(n,latent_dim)
        #self.latent = self.latent / self.latent.norm(dim=1).unsqueeze(1) #(n,latent_dim)
        #self.latent = self.latent.unsqueeze(0).expand(bs, self.n_agents,
        #                                                                    self.latent_dim).reshape(-1,
        #                                                                                              self.latent_dim)
            # (bs*n,latent_dim)
        #loss = -(mu_param.norm(dim=1) * pi_param).sum()  # KL, N(0,1)
        loss=0



        return loss, th.cat([pi_param.data.detach().reshape(-1, 1), mu_param.data.detach()], dim=1)

        # (bs*n,(obs+act+id)), (bs,n,hidden_dim), (bs,n,latent_dim)
    def forward(self, inputs, hidden_state):


        inputs = inputs.reshape(self.bs,self.n_agents, 1, self.input_shape)
        h_in = hidden_state.reshape(self.bs,self.n_agents, 1, self.args.rnn_hidden_dim)  # (bs*n,1, hidden_dim)

        q_all = []
        h_all = []

        for i in range(self.bs):
            x=F.relu(th.bmm(inputs[i],self.fc1_w)+self.fc1_b) #(n,(obs+act+id)) at time t
            gi=th.bmm(x,self.rnn_ih_w)+self.rnn_ih_b
            gh=th.bmm(h_in[i],self.rnn_hh_w)+self.rnn_hh_b
            i_r,i_i,i_n=gi.chunk(3,2)
            h_r,h_i,h_n=gh.chunk(3,2)

            resetgate=th.sigmoid(i_r+h_r)
            inputgate=th.sigmoid(i_i+h_i)
            newgate=th.tanh(i_n+resetgate*h_n)
            h=newgate+inputgate*(h_in[i]-newgate)
            q = F.relu(th.bmm(h, self.fc2_w) + self.fc2_b)

            h_all.append(h)
            q_all.append(q)

        # inputs = inputs.reshape(-1, self.input_shape)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # (bs*n, hidden_dim)
        # latent = self.latent.reshape(-1, self.args.latent_dim)  # (bs*n,latent_dim)

        #x=x.reshape(-1,self.args.rnn_hidden_dim)
        #h = self.rnn(x, h_in)
        #h=h.reshape(-1,1,self.args.rnn_hidden_dim)

        #x = F.relu(self.fc1(inputs)) #(bs*n,(obs+act+id)) at time t
        #h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # (bs,n,dim) ==> (bs*n, dim)
        #h = self.rnn(x, h_in)
        #h = h.reshape(-1, 1, self.args.rnn_hidden_dim)
        #q = self.fc2(h)

        q=th.stack(q_all,dim=0)
        h=th.stack(h_all,dim=0)
        return q.view(-1,self.args.n_actions), h.view(-1,self.args.rnn_hidden_dim)
        # (bs*n,n_actions), (bs*n,hidden_dim), (bs*n,latent_dim)