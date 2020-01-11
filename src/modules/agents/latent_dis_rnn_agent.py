import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.distributions import kl_divergence
import torch.distributions as D
from tensorboardX import SummaryWriter
import time


class LatentDisRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(LatentDisRNNAgent, self).__init__()
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
        self.latent_rnn = nn.GRUCell(args.latent_dim * 2, args.latent_dim * 2)
        self.latent_bn = nn.BatchNorm1d(args.latent_dim * 2)

        self.latent = th.rand(self.bs * args.n_agents, args.latent_dim * 2)  # (n,mu+var)
        self.latent_hist = th.rand(self.bs * args.n_agents, args.latent_dim * 2)

        # latent -> FC2 parameter
        self.latent_fc1 = nn.Linear(args.latent_dim, args.latent_dim * 4)
        self.latent_fc2 = nn.Linear(args.latent_dim * 4, args.latent_dim * 4)
        self.fc2_w_nn = nn.Linear(args.latent_dim * 4, args.rnn_hidden_dim * args.n_actions)
        self.fc2_b_nn = nn.Linear(args.latent_dim * 4, args.n_actions)

        # GRU & FC1
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # Dis Net
        self.dis_net = nn.Sequential(nn.Linear(args.latent_dim * 2, args.latent_dim * 4),
                                     nn.ReLU(),
                                     nn.Linear(args.latent_dim * 4, 1))

        if args.dis_cur:
            if args.dis_net:
                self.forward = self.forward_cur_dis_net
            else:
                self.forward = self.forward_cur
        else:
            if args.dis_net:
                self.forward = self.forward_hist_dis_net
            else:
                self.forward = self.forward_hist

    def init_latent(self, bs):
        self.bs = max(1, bs)
        loss = 0
        if self.args.runner == "episode":
            self.writer = SummaryWriter(
                "results/tb_logs/test_latent-" + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
        return loss, self.latent[:self.n_agents, :].detach(), self.latent_hist[:self.n_agents, :]

    def forward_hist_dis_net(self, inputs, hidden_state, t=0, batch=None, test_mode=None, t_glob=0, train_mode=False):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # Obs to role
        latent_last = self.latent.detach()
        self.latent = self.embed_fc(inputs.detach())  # (n,2*latent_dim)==(n,mu+log var)
        self.latent[:, self.latent_dim:] = th.exp(self.latent[:, self.latent_dim:])  # var

        gaussian_embed = D.Normal(self.latent[:, :self.latent_dim], self.latent[:, self.latent_dim:])
        latent = gaussian_embed.rsample()  # Sample a role
        c_dis_loss = 0
        loss = 0

        if train_mode:
            if t == 0:
                self.latent_hist = self.latent.detach()
                loss = 0
            else:
                self.latent_hist = self.latent_rnn(latent_last, self.latent_bn(self.latent_hist))
                self.latent_hist[:, self.latent_dim:] = th.exp(self.latent_hist[:, self.latent_dim:])
                gaussian_hist = D.Normal(self.latent_hist[:, :self.latent_dim], self.latent_hist[:, self.latent_dim:])
                gaussian_hist_sample = gaussian_hist.rsample()
                loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed,
                                                                      gaussian_hist).sum()  # CE = H + KL

                # if th.isnan(loss).any():
                #     print('>>> loss')
                # if (loss > 1e6).any():
                #     print('>>> loss LARGE')

                # Dis Loss
                dis_loss = 0
                self.latent_hist_dis = self.latent_hist.clone()
                latent_hist_dis = self.latent_hist_dis.view(self.bs, self.n_agents, -1)
                latent_hist_dis_move = latent_hist_dis.clone()
                dissimilarity_cat = None
                for agent_i in range(self.n_agents):
                    latent_hist_dis_move = th.cat(
                        [latent_hist_dis_move[:, -1, :].unsqueeze(1), latent_hist_dis_move[:, :-1, :]], dim=1)
                    latent_hist_dis_pair = th.cat(
                        [latent_hist_dis[:, :, :self.latent_dim], latent_hist_dis_move[:, :, :self.latent_dim]], dim=2)
                    gaussian_hist_move = D.Normal(
                        latent_hist_dis_move[:, :, :self.latent_dim].view(self.bs * self.n_agents, -1),
                        latent_hist_dis_move[:, :, self.latent_dim:].view(self.bs * self.n_agents, -1))
                    mi = th.clamp(gaussian_hist_move.log_prob(gaussian_hist_sample), -13.9, 0).sum(dim=1,
                                                                                                   keepdim=True) / self.latent_dim / self.bs

                    # if th.isnan(mi).any():
                    #     print('>>> mi nan')
                    # if (mi > 1e6).any():
                    #     print('>>> mi LARGE')

                    dissimilarity = th.abs(self.dis_net(latent_hist_dis_pair.view(-1, 2 * self.latent_dim)))

                    # if th.isnan(dissimilarity).any():
                    #     print('>>> dissimilarity')
                    # if (dissimilarity > 1e6).any():
                    #     print('>>> dissimilarity LARGE')

                    if dissimilarity_cat is not None:
                        dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)
                    else:
                        dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                    dis_loss -= th.abs(mi - dissimilarity / self.bs).sum()

                    # if th.isnan(dis_loss).any():
                    #     print('>>>', agent_i, 'dis_loss')
                    # if (dis_loss > 1e6).any():
                    #     print('>>>', agent_i, 'dis_loss LARGE')

                    # print('mi', mi.sum())
                    # print('dis', dissimilarity.sum() / self.bs)
                    # print('dis_norm', th.norm(dissimilarity_cat, p=2, dim=1).sum() / self.bs)

                # print(dis_loss)
                dis_norm = th.norm(dissimilarity_cat, p=2, dim=1).sum() / self.bs
                # if th.isnan(dis_norm).any():
                #     print('>>> dis_norm')
                # if (dis_norm > 1e6).any():
                #     print('>>> dis_norm LARGE')

                c_dis_loss = (dis_loss + dis_norm) / self.n_agents
                loss = loss / (self.bs * self.n_agents)
                loss += self.args.dis_loss_weight * c_dis_loss
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

        if self.args.runner == "episode":
            self.writer.add_embedding(self.latent.reshape(-1, self.latent_dim * 2), list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-cur")
            self.writer.add_embedding(self.latent_hist.reshape(-1, self.latent_dim * 2),
                                      list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-hist")
        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss, c_dis_loss

    def forward_cur_dis_net(self, inputs, hidden_state, t=0, batch=None, test_mode=None, t_glob=0, train_mode=False):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # Obs to role
        latent_last = self.latent.detach()
        self.latent = self.embed_fc(inputs.detach())  # (n,2*latent_dim)==(n,mu+log var)
        self.latent[:, self.latent_dim:] = th.exp(self.latent[:, self.latent_dim:])  # var

        gaussian_embed = D.Normal(self.latent[:, :self.latent_dim], self.latent[:, self.latent_dim:])
        latent = gaussian_embed.rsample()  # Sample a role
        c_dis_loss = 0
        loss = 0

        if train_mode:
            if t == 0:
                self.latent_hist = self.latent.detach()
                loss = 0
            else:
                self.latent_hist = self.latent_rnn(latent_last, self.latent_bn(self.latent_hist))
                self.latent_hist[:, self.latent_dim:] = th.exp(self.latent_hist[:, self.latent_dim:])
                gaussian_hist = D.Normal(self.latent_hist[:, :self.latent_dim], self.latent_hist[:, self.latent_dim:])
                # gaussian_hist_sample = gaussian_hist.rsample()
                loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed,
                                                                      gaussian_hist).sum()  # CE = H + KL

                # Dis Loss
                cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
                if cur_dis_loss_weight > 0:
                    dis_loss = 0
                    dissimilarity_cat = None
                    latent_dis = latent.clone().view(self.bs, self.n_agents, -1)
                    latent_move = latent.clone().view(self.bs, self.n_agents, -1)
                    for agent_i in range(self.n_agents):
                        latent_move = th.cat(
                            [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                        latent_dis_pair = th.cat(
                            [latent_dis[:, :, :self.latent_dim], latent_move[:, :, :self.latent_dim]], dim=2)
                        mi = (13.9 + th.clamp(gaussian_embed.log_prob(latent_move.view(self.bs * self.n_agents, -1)),
                                              -13.9, 0)).sum(dim=1,
                                                             keepdim=True) / self.latent_dim

                        dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                        if dissimilarity_cat is None:
                            dissimilarity_cat = dissimilarity.view(self.bs, -1).clone()
                        else:
                            dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(self.bs, -1)], dim=1)

                        dis_loss -= (mi + dissimilarity).sum() / self.bs / self.n_agents

                    dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / self.bs

                    c_dis_loss = (dis_loss + dis_norm) / self.n_agents
                    loss = loss / (self.bs * self.n_agents)
                    loss += self.args.dis_loss_weight * c_dis_loss
                    loss = th.log(1 + th.exp(loss))
                else:
                    loss = loss / (self.bs * self.n_agents)
                    loss = th.log(1 + th.exp(loss))
                    c_dis_loss = th.zeros_like(loss)

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

        if self.args.runner == "episode":
            self.writer.add_embedding(self.latent.reshape(-1, self.latent_dim * 2), list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-cur")
            self.writer.add_embedding(self.latent_hist.reshape(-1, self.latent_dim * 2),
                                      list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-hist")
        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss, c_dis_loss

    def forward_cur(self, inputs, hidden_state, t=0, batch=None, test_mode=None, t_glob=0, train_mode=False):
        inputs = inputs.reshape(-1, self.input_shape)
        h_in = hidden_state.reshape(-1, self.hidden_dim)

        # Obs to role
        latent_last = self.latent.detach()
        self.latent = self.embed_fc(inputs.detach())  # (n,2*latent_dim)==(n,mu+log var)
        self.latent[:, self.latent_dim:] = th.exp(self.latent[:, self.latent_dim:])  # var

        gaussian_embed = D.Normal(self.latent[:, :self.latent_dim], self.latent[:, self.latent_dim:])
        latent = gaussian_embed.rsample()  # Sample a role
        c_dis_loss = 0
        loss = 0

        if train_mode:
            if t == 0:
                self.latent_hist = self.latent.detach()
                loss = 0
            else:
                self.latent_hist = self.latent_rnn(latent_last, self.latent_bn(self.latent_hist))
                self.latent_hist[:, self.latent_dim:] = th.exp(self.latent_hist[:, self.latent_dim:])
                gaussian_hist = D.Normal(self.latent_hist[:, :self.latent_dim], self.latent_hist[:, self.latent_dim:])
                # gaussian_hist_sample = gaussian_hist.rsample()
                loss = gaussian_embed.entropy().sum() + kl_divergence(gaussian_embed,
                                                                      gaussian_hist).sum()  # CE = H + KL

                # Dis Loss
                cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
                if cur_dis_loss_weight > 0:
                    dis_loss = 0
                    latent_dis = self.latent.clone().view(self.bs, self.n_agents, -1)
                    latent_dis_gaussian = D.Normal(latent_dis[:, :, :self.latent_dim],
                                                   latent_dis[:, :, self.latent_dim:])
                    latent_move = self.latent.clone().view(self.bs, self.n_agents, -1)

                    #loss_gaussian = D.Normal(th.full_like(latent_dis[:, :, :self.latent_dim],0.5),
                    #                         th.full_like(latent_dis[:, :, self.latent_dim:],0.5))
                    loss_gaussian = D.Normal(0.1,1.0)

                    for agent_i in range(self.n_agents):
                        latent_move = th.cat(
                            [latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                        latent_move_gaussian = D.Normal(latent_move[:, :, :self.latent_dim],
                                                        latent_move[:, :, self.latent_dim:])

                        #dis_kl = D.kl_divergence(latent_dis_gaussian, latent_move_gaussian) / self.bs / self.n_agents
                        dis_loss += loss_gaussian.log_prob(th.norm(latent_move[:, :, :self.latent_dim]-latent_dis[:, :, :self.latent_dim],dim=2)).sum()

                    c_dis_loss = dis_loss / self.n_agents
                    #c_dis_loss = th.zeros_like(loss)

                    loss = loss / (self.bs * self.n_agents)
                    loss += self.args.dis_loss_weight * c_dis_loss
                    loss = th.log(1 + th.exp(loss))
                else:
                    loss = loss / (self.bs * self.n_agents)
                    loss = th.log(1 + th.exp(loss))
                    c_dis_loss = th.zeros_like(loss)

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

        if self.args.runner == "episode":
            self.writer.add_embedding(self.latent.reshape(-1, self.latent_dim * 2), list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-cur")
            self.writer.add_embedding(self.latent_hist.reshape(-1, self.latent_dim * 2),
                                      list(range(self.args.n_agents)),
                                      global_step=t, tag="latent-hist")
        return q.view(-1, self.args.n_actions), h.view(-1, self.args.rnn_hidden_dim), loss, c_dis_loss

    def dis_loss_weight_schedule(self, t_glob):
        if t_glob > 0:
            return self.args.dis_loss_weight
        else:
            return 0
