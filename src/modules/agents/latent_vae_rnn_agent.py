import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
from tensorboardX import SummaryWriter

class LatentVAERNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentVAERNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim
        self.bs = 0
        self.hg_dim = args.hg_dim
        self.rec_loss_weight = args.rec_loss_weight
        self.ce_loss_weight = args.ce_loss_weight


        # Embed Network
        self.embed_fc = nn.Linear(input_shape, args.latent_dim * 2)

        # h -> Gaussian
        self.hg_fc1 = nn.Linear(args.rnn_hidden_dim, args.hg_dim * 4)
        self.hg_fc2 = nn.Linear(args.hg_dim * 4, args.hg_dim * 2)

        # Encoder
        self.encoder_fc1 = nn.Linear(args.hg_dim, args.latent_dim * 4)
        self.encoder_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim * 2)

        # Decoder
        self.decoder_fc1 = nn.Linear(args.latent_dim, args.hg_dim)
        self.decoder_fc2 = nn.Linear(args.hg_dim, args.hg_dim * 2)

        self.latent = th.rand(args.n_agents, args.latent_dim * 2)  # (n,mu+var)

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
        self.writer = SummaryWriter("results/tb_logs/test/latent")
        return loss, self.latent[:self.n_agents,:].detach()

    def forward(self, inputs, hidden_state,t=0,batch=None, test_mode=None):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # Obs to role
        self.latent = self.embed_fc(inputs.detach())  # (n,2*latent_dim)==(n,mu+log var)
        self.latent[:, -self.latent_dim:] = th.exp(self.latent[:, -self.latent_dim:])  # var

        latent_embed = self.latent.reshape(-1, self.latent_dim * 2)
        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], latent_embed[:, self.latent_dim:])
        latent = gaussian_embed.rsample()   # Sample a role

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

        # hg Net
        hg = F.relu(self.hg_fc1(h.detach().view(-1, self.args.rnn_hidden_dim)))
        hg = self.hg_fc2(hg)
        hg[:, -self.hg_dim:] = th.exp(hg[:, -self.hg_dim:])
        h_gaussian = D.Normal(hg[:, :self.hg_dim], hg[:, self.hg_dim:])
        h_gaussian_sample = h_gaussian.rsample()

        # Encoder
        latent_encoder = F.relu(self.encoder_fc1(h_gaussian_sample))
        latent_encoder = self.encoder_fc2(latent_encoder)
        latent_encoder[:, -self.latent_dim:] = th.exp(latent_encoder[:, -self.latent_dim:])
        z_inference_gaussian = D.Normal(latent_encoder[:, :self.latent_dim], latent_encoder[:, self.latent_dim:])

        latent_encoder_d = latent_encoder.detach()
        z_inference_gaussian_d = D.Normal(latent_encoder_d[:, :self.latent_dim], latent_encoder_d[:, self.latent_dim:])
        z_inference_sample = z_inference_gaussian.rsample()

        # Decoder
        latent_decoder = F.relu(self.decoder_fc1(z_inference_sample))
        latent_decoder = self.decoder_fc2(latent_decoder)
        latent_decoder[:, -self.hg_dim:] = th.exp(latent_decoder[:, -self.hg_dim:])
        h_inference_gaussian = D.Normal(latent_decoder[:, :self.hg_dim], latent_decoder[:, self.hg_dim:])

        # Loss 1: CE loss between embed_z and inference_z
        #ce_loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed, z_inference_gaussian_d).sum()
        ce_loss =  kl_divergence(z_inference_gaussian_d,gaussian_embed).sum()
        # Loss 2: Reconstruction loss of h
        #rec_loss = D.kl_divergence(h_gaussian, h_inference_gaussian).sum()
        rec_loss = h_gaussian.entropy().sum() + kl_divergence(h_gaussian, h_inference_gaussian).sum()

        loss = self.ce_loss_weight * ce_loss + self.rec_loss_weight * rec_loss # CE = H + KL
        loss = loss / (self.bs * self.n_agents)

        if self.args.runner=="episode":
            self.writer.add_embedding(self.latent.reshape(-1,self.latent_dim*2),list(range(self.args.n_agents)),global_step=t,tag="latent-step")

        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss