import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cross_attn import CrossAttentionLayer


def print_model(model):
    # Print the model architecture and number of parameters per layer
    print(f"{'Layer':<40} {'Shape':<30} {'# Parameters':<15}")
    print("=" * 85)
    for name, param in model.named_parameters():
        print(f"{name:<40} {str(list(param.shape)):<30} {param.numel():<15}")

    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 85)
    print(f"Total number of parameters: {total_params}")


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size=400, output_size=1):
        super(Actor, self).__init__()
        self.att_dim = 400
        self.max_action = 3.0
        self.att_up = CrossAttentionLayer(6, 6, self.att_dim)
        self.att_down = CrossAttentionLayer(6, 6, self.att_dim)

        self.proj = nn.Linear(6, self.att_dim)
        self.linear2 = nn.Linear(self.att_dim, self.att_dim)
        self.linear3 = nn.Linear(self.att_dim, self.att_dim)
        self.linear4 = nn.Linear(self.att_dim, 1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def forward(self, s_list):
        # s:  initial state n x 8:   [[bus_id, ord, last last_stop, loc, hb, hf, occp, is_target] for all n buses at time t]
        x_u, x_d, x_subjects = [], [], []

        for s in s_list:
            subject_bus_idx, subject_stop_idx, subject_location = s[0,1], s[0,2], s[0,3]
            subject_location = s[0, 3]
            s[:, 1] = s[:, 1] - subject_bus_idx
            x_subject = s[0:1, 1:-1]
            x_up = s[s[:, 3] < subject_location][:, 1:-1]
            x_down = s[s[:, 3] > subject_location][:, 1:-1]

            if x_up.size(0) > 0:
                x_up = self.att_up(x_subject, x_up)
            else:
                x_up = torch.zeros(1, self.att_dim)

            if x_down.size(0) > 0:
                x_down = self.att_down(x_subject, x_down)
            else:
                x_down = torch.zeros(1, self.att_dim)
            x_u.append(x_up.squeeze())
            x_d.append(x_down.squeeze())
            x_subjects.append(x_subject.squeeze())
        x_d = torch.stack(x_d, 0)
        x_u = torch.stack(x_u, 0)
        x_subjects = torch.stack(x_subjects, 0)

        x_proj = self.proj(x_subjects)
        x = x_d + x_u + x_proj
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = (self.tanh(self.linear4(x)) + 1) * 1.5
        return x


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.hidden = 400
        self.state_dim = state_dim
        self.v_dim = 200
        self.bn = nn.BatchNorm1d(self.v_dim)
        # for ego critic
        self.fc0 = nn.Linear(state_dim + 1, self.hidden)  # state + action
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.fc3_u = nn.Linear(self.v_dim, 200)
        self.fc4_u = nn.Linear(200, 1)
        self.fc3_u_ = nn.Linear(self.v_dim, 200)
        self.fc4_u_ = nn.Linear(200, 1)

        self.fc3_d = nn.Linear(self.v_dim, 200)
        self.fc4_d = nn.Linear(200, 1)
        self.fc3_d_ = nn.Linear(self.v_dim, 200)
        self.fc4_d_ = nn.Linear(200, 1)

        self.fc3_o = nn.Linear(self.v_dim, 200)
        self.fc4_o = nn.Linear(200, 1)
        self.fc3_o_ = nn.Linear(self.v_dim, 200)
        self.fc4_o_ = nn.Linear(200, 1)

        self.aug_attention = CrossAttentionLayer(7, 14, self.v_dim)

        self.o_attention = CrossAttentionLayer(7, 7, self.v_dim)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()

        self.bn_ego_0 = nn.BatchNorm1d(self.hidden)
        self.bn_ego_1 = nn.BatchNorm1d(self.hidden)

        self.fc0 = nn.Linear(4, self.hidden)  # state + action
        self.fc1 = nn.Linear(self.hidden, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)

    def event_critic(self, x_list, fp_list, a_list):
        reg = []
        u_x_targets = []
        d_x_targets = []
        o_x_targets = []
        for x, fp, a in zip(x_list, fp_list, a_list):
            s = torch.tensor(x, dtype=torch.float32)
            # merged_s(n x 15): [ord,   last_stop,   loc,   hb,   hf,   occp,   a,
            #                    ord_2, last_stop_2, loc_2, hb_2, hf_2, occp_2, a_2, active_flag]
            merged_s = self.merge(s, a, fp)
            subject_bus_idx, subject_location = merged_s[0, 0].clone(), merged_s[0, 2].clone()
            merged_s[:, 0] = merged_s[:, 0] - subject_bus_idx
            merged_s[:, 7] = merged_s[:, 7] - subject_bus_idx

            ego_x = merged_s[0:1, :7]

            u_x = merged_s[(merged_s[:, 2] < subject_location) & (merged_s[:, -1] == 1)][:, :-1]  # active upstream
            d_x = merged_s[(merged_s[:, 2] > subject_location) & (merged_s[:, -1] == 1)][:, :-1]  # active downstream
            o_x = merged_s[merged_s[:, -1] == 0][1:, :7]  # inactive buses
            if u_x.size(0) > 0:
                u_x_target = self.aug_attention(ego_x, u_x)
            else:
                u_x_target = torch.zeros(self.v_dim)

            if d_x.size(0) > 0:
                d_x_target = self.aug_attention(ego_x, d_x)
            else:
                d_x_target = torch.zeros(self.v_dim)

            if o_x.size(0) > 0:
                o_x_target = self.o_attention(ego_x, o_x)
            else:
                o_x_target = torch.zeros(self.v_dim)

            u_x_targets.append(u_x_target.squeeze())
            d_x_targets.append(d_x_target.squeeze())
            o_x_targets.append(o_x_target.squeeze())

        u_x_targets = torch.stack(u_x_targets, 0)
        d_x_targets = torch.stack(d_x_targets, 0)
        o_x_targets = torch.stack(o_x_targets, 0)

        u_x_targets = self.bn(u_x_targets)
        d_x_targets = self.bn(d_x_targets)
        o_x_targets = self.bn(o_x_targets)

        u_x_1 = self.elu(self.fc3_u(u_x_targets))
        u_x_1 = self.fc4_u(u_x_1)
        u_x_2 = self.elu(self.fc3_u_(u_x_targets))
        u_x_2 = self.fc4_u_(u_x_2)

        d_x_1 = self.elu(self.fc3_d(d_x_targets))
        d_x_1 = self.fc4_d(d_x_1)
        d_x_2 = self.elu(self.fc3_d_(d_x_targets))
        d_x_2 = self.fc4_d_(d_x_2)

        o_x_1 = self.elu(self.fc3_o(o_x_targets))
        o_x_1 = self.fc4_o(o_x_1)
        o_x_2 = self.elu(self.fc3_o_(o_x_targets))
        o_x_2 = self.fc4_o_(o_x_2)

        G1, G2 = u_x_1 + d_x_1 + o_x_1, u_x_2 + d_x_2 + o_x_2
        if len(reg) > 0:
            reg = torch.stack(reg, 0).view(-1, 1)
        else:
            reg = torch.zeros(1)
        return G1, G2, reg

    def merge(self, s, a, fp):
        '''
        combine current state s, action (a), and active state (fp), to generate augmented state, by matching bus_ids

        parameters:
        s:  current state n x 8:   [[bus_id, ord, last_stop, loc, hb, hf, occp, is_target] for all n buses at time t]
        fp: augmented state l x 9: [[bus_id, ord, last_stop, loc, hb, hf, occp, is_target, action] for l active agents during action period of subject bus]

        returns n x 15: [ord,   last_stop,   loc,   hb,   hf,   occp,   a,
                         ord_2, last_stop_2, loc_2, hb_2, hf_2, occp_2, a_2, active_flag]
        '''
        init_bus_ids = s[:, 0]
        aug_bus_ids = fp[:, 0]

        n_bus = s.shape[0]
        a_column = np.zeros((n_bus, 1))
        a_column[0, 0] = a # Set the first value to action
        # Append the column to the original matrix
        init_state = np.hstack((s[:, 1:-1], a_column))
        init_state = torch.from_numpy(init_state).float()
        # aug_states = fp
        aug_states = torch.cat((fp[:, : -2], fp[:, -1:]), dim=1) #get rid of is_target column

        result = []
        matched_bus_ids = []
        for row in aug_states:
            bus_id_active = row[0]
            if bus_id_active in matched_bus_ids:
                continue
            matching_rows = init_bus_ids == bus_id_active
            if torch.sum(matching_rows) > 0:
                s_init = init_state[matching_rows][0:1, :]  # Get the first matching row in aug_states
                s_aug = row[1:]
                flag = torch.ones(1, 1)
                state = torch.cat((s_init, s_aug.view(1, -1), flag), dim=1)
                result.append(state.squeeze())
            matched_bus_ids.append(bus_id_active)
        result = torch.stack(result, 0)

        mask = ~init_bus_ids.unsqueeze(1).eq(aug_bus_ids).any(dim=1)
        outstanding_init = init_state[mask]
        empty_placeholder = torch.full((outstanding_init.shape[0], aug_states.shape[1]), 0, dtype=torch.float)
        outstanding_init = torch.cat((outstanding_init, empty_placeholder), dim=1)
        result = torch.cat((result, outstanding_init), dim=0)
        return result

    def ego_critic(self, x_list, fp_list, a_list):
        # x_list:  [[bus_id, ord, last_stop, loc, hb, hf, occp, is_target] for all n buses at time t]
        x_u, x_d, x_targets = [], [], []
        for s, a in zip(x_list, a_list):
            s = torch.tensor(s, dtype=torch.float32)
            x_target = s[0:1, [4, 5, 6, 7]]
            x_target[:, -1] = a
            x_targets.append(x_target.squeeze())
        ego = torch.stack(x_targets, 0)
        out1 = self.relu(self.fc0(ego))
        out1 = self.relu(self.fc1(out1))
        Q = self.fc2(out1)
        return Q

    def forward(self, xs):
        x, a, fp = xs
        Q1 = self.ego_critic(x, fp, a)
        A1, A2, reg = self.event_critic(x, fp, a)
        G1, G2 = Q1 + A1, Q1 + A2
        return Q1, A2, G1.view(-1, 1), G2.view(-1, 1), reg


class Agent():
    def __init__(self, state_dim, name, seed=123, policy_noise=0.3, noise_clip=0.2, policy_freq=1):
        random.seed(seed)
        self.seed = seed
        self.name = name
        self.gamma = 0.9
        self.state_dim = state_dim
        self.learn_step_counter = 0
        self.policy_freq = policy_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip

        self.critic = Critic(state_dim)
        self.critic_target = Critic(state_dim)
        total_params = sum(p.numel() for p in self.critic.parameters())
        print("total no of parameters for critic:", total_params)
        print_model(self.critic)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = Actor(self.state_dim)
        self.actor_target = Actor(self.state_dim)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor_target.load_state_dict(self.actor.state_dict())

    def lr_decay(self, ratio=0.99):
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] *= ratio
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] *= ratio
        self.policy_noise *= ratio

    def choose_action(self, state, noise_scale=0.3):
        state = torch.tensor(state, dtype=torch.float)
        a = self.actor([state])[0].detach().numpy()
        # Add Gaussian noise for exploration
        a = a + np.random.normal(0, noise_scale, size=a.shape)

        # Clip the action to be within the valid action range
        a = np.clip(a, 0, 3.0)
        return a

    def distill_from_others(self, memories, batch=1024, epochs=20):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.actor.train()

        # Sample a batch from the memories
        batch_s, batch_a = [], []
        memory = random.sample(memories, batch)
        for s, fp, a, r, ns, nfp in memory:
            batch_s.append(s)
            batch_a.append(a)

        # Convert batch to tensors
        batch_a = torch.tensor(batch_a, dtype=torch.float).view(-1, 1)

        for epoch in range(epochs):
            total_loss = 0.0
            student_actions = []

            for state in batch_s:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                # Forward pass for distilled agent
                student_action = self.actor(state_tensor)
                student_actions.append(student_action)

            # Stack student actions into a single tensor
            student_actions = torch.cat(student_actions, dim=0)

            # Calculate distillation loss
            loss = F.mse_loss(student_actions, batch_a)

            # Backpropagation
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(batch_s)
            print(f"Distill Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        return avg_loss

    def learn(self, memories, batch=16):
        if len(memories) < batch:
            return 0, 0

        batch_s, batch_fp, batch_a, batch_r, batch_ns, batch_nfp = [], [], [], [], [], []
        memory = random.sample(memories, batch)

        batch_mask = []
        batch_mask_n = []
        batch_fp_critic_t = []
        batch_actor_a = []
        for s, fp, a, r, ns, nfp, in memory:
            batch_s.append(s)
            _fp_ = np.array(copy.deepcopy(fp))
            _fp_ = torch.tensor(_fp_, dtype=torch.float32)
            # _fp_[0, self.state_dim] = self.actor([torch.tensor(s, dtype=torch.float32)])[0].detach()
            batch_fp_critic_t.append(_fp_)
            batch_actor_a.append(self.actor([torch.tensor(s, dtype=torch.float32)])[0])
            batch_fp.append(torch.FloatTensor(fp))
            batch_mask.append(len(fp) - 1)
            batch_mask_n.append(len(nfp) - 1)
            batch_a.append(a)
            batch_r.append(r)
            batch_ns.append(ns)
            batch_nfp.append(torch.FloatTensor(nfp))
        b_fp_pad = batch_fp
        b_nfp_pad = batch_nfp

        batch_actor_a = torch.stack(batch_actor_a, 0)
        b_a = torch.tensor(np.array(batch_a), dtype=torch.float).view(-1, 1)
        b_r = torch.tensor(np.array(batch_r), dtype=torch.float).view(-1, 1)

        def critic_learn():
            Q, A1, G1, G2, reg = self.critic([batch_s, b_a, b_fp_pad])
            batch_ns_tensor = [torch.tensor(state, dtype=torch.float) for state in batch_ns]
            nb_a = self.actor_target(batch_ns_tensor).detach().view(-1, 1)
            noise = (torch.randn_like(nb_a) * self.policy_noise).clamp(0, self.noise_clip)
            nb_a = (nb_a + noise).clamp(0, 3.0)
            # nb_a = (nb_a + noise)
            Q_, A1_, G1_, G2_, _ = self.critic_target(
                [batch_ns, nb_a, b_nfp_pad])
            G_ = torch.min(G1_, G2_)
            q_target = b_r + self.gamma * (G_.detach()).view(-1, 1)

            loss_fn = nn.MSELoss()
            qloss = loss_fn(G1, q_target) + loss_fn(G2, q_target) + 0.1 * reg.mean()
            self.critic_optim.zero_grad()
            qloss.backward()
            self.critic_optim.step()
            return qloss.item()

        def actor_learn():
            policy_loss, _, _, _, _ = self.critic([batch_s, batch_actor_a.view(-1, 1), batch_fp_critic_t])
            policy_loss = -torch.mean(policy_loss)
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            return policy_loss.item()

        def soft_update(net_target, net, tau=0.02):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        qloss = critic_learn()
        if self.learn_step_counter % self.policy_freq == 0:
            policy_loss = actor_learn()
            soft_update(self.critic_target, self.critic, tau=0.02)
            soft_update(self.actor_target, self.actor, tau=0.02)
        else:
            policy_loss = torch.zeros(1)

        self.learn_step_counter += 1

        return policy_loss, qloss

    def save(self, model):
        abspath = os.path.abspath(os.path.dirname(__file__))

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
        torch.save(self.actor.state_dict(), path)

        path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_critic.pth"
        torch.save(self.critic.state_dict(), path)

    def load(self, model):
        try:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "/save/" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
        except:
            abspath = os.path.abspath(os.path.dirname(__file__))
            print('Load: ' + abspath + "/save/" + str(self.name) + '_' + str(model))
            path = abspath + "\\save\\" + str(self.name) + '_' + str(model) + str(self.seed) + "_actor.pth"
            state_dict = torch.load(path)
            self.actor.load_state_dict(state_dict)
