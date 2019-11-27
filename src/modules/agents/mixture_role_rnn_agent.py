import torch.nn as nn
import torch.nn.functional as F
import torch as th


class MixtureRoleRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MixtureRoleRNNAgent, self).__init__()
        self.args = args
        self.input_shape=input_shape
        self.n_agents=args.n_agents
        self.latent_dim=args.latent_dim
        self.hidden_dim=args.rnn_hidden_dim
        self.bs=0

        self.pi_param = nn.Parameter(th.rand(args.n_agents,args.latent_dim)) #(n,latent_dim)

        self._fc1_w = nn.Parameter(th.randn(args.latent_dim, input_shape, args.rnn_hidden_dim))
        self._fc1_b = nn.Parameter(th.randn(args.latent_dim, 1, args.latent_dim))

        self._rnn_ih_w=nn.Parameter(th.randn(args.latent_dim, args.rnn_hidden_dim, args.rnn_hidden_dim*3))
        self._rnn_ih_b=nn.Parameter(th.randn(args.latent_dim, 1, args.rnn_hidden_dim*3))
        self._rnn_hh_w=nn.Parameter(th.randn(args.latent_dim, args.rnn_hidden_dim, args.rnn_hidden_dim*3))
        self._rnn_hh_b=nn.Parameter(th.randn(args.latent_dim, 1, args.rnn_hidden_dim*3))

        self._fc2_w=nn.Parameter(th.randn(args.latent_dim, args.rnn_hidden_dim, args.n_actions))
        self._fc2_b=nn.Parameter(th.randn(args.latent_dim, 1, args.n_actions))

    def init_latent(self,bs):
        self.bs=bs
        pi_param = F.softmax(self.pi_param,dim=1) #(n,latent_dim)

        u = th.rand_like(pi_param)
        g = - th.log(- th.log(u))
        c = (g + th.log(pi_param)).argmax(dim=1) #(n,)

        self.fc1_w=self._fc1_w[c].unsqueeze(1)
        self.fc1_b=self._fc1_b[c].unsqueeze(1)

        self.rnn_ih_w=self._rnn_ih_w[c].unsqueeze(1)
        self.rnn_ih_b=self._rnn_ih_b[c].unsqueeze(1)
        self.rnn_hh_w=self._rnn_hh_w[c].unsqueeze(1)
        self.rnn_hh_b=self._rnn_hh_b[c].unsqueeze(1)

        self.fc2_w = self._fc2_w[c].unsqueeze(1)
        self.fc2_b = self._fc2_b[c].unsqueeze(1)

        loss = -(pi_param*th.log(pi_param)).sum()

        return loss, pi_param.data.detach()

        # (bs*n,(obs+act+id)), (bs,n,hidden_dim), (bs,n,latent_dim)
    def forward(self, inputs, hidden_state):
        inputs=inputs.reshape(-1,self.n_agents,self.input_shape).chunk(self.n_agents,dim=1) #[n,(bs,1,input_shape)]
        h_in=hidden_state.reshape(-1,self.n_agents,self.hidden_dim).chunk(self.n_agents,dim=1) #[n,(bs,1,hidden_dim)]

        q_all=[]
        h_all=[]
        for i in range(self.n_agents):
            x = F.relu(th.bmm(inputs[i], self.fc1_w[i]) + self.fc1_b[i])
            gi = th.bmm(x, self.rnn_ih_w[i]) + self.rnn_ih_b[i]
            gh = th.bmm(h_in[i], self.rnn_hh_w[i]) + self.rnn_hh_b[i]
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            resetgate = th.sigmoid(i_r + h_r)
            inputgate = th.sigmoid(i_i + h_i)
            newgate = th.tanh(i_n + resetgate * h_n)
            h = newgate + inputgate * (h_in - newgate)
            q = F.relu(th.bmm(h, self.fc2_w[i]) + self.fc2_b[i])

            q_all.append(q)
            h_all.append(h)

        q=th.stack(q_all,dim=1)
        h=th.stack(h_all,dim=1)

        return q.view(-1,self.args.n_actions), h.view(-1,self.args.rnn_hidden_dim)
        # (bs*n,n_actions), (bs*n,hidden_dim)