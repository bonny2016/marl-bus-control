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
        self.hidden = 200
        self.state_dim = state_dim
        self.v_dim = 200

        # Batch normalization
        self.bn = nn.BatchNorm1d(self.v_dim)

        # Ego critic MLP
        self.ego_mlp = nn.Sequential(
            nn.Linear(4, self.hidden),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden),
            nn.Linear(self.hidden, 1)
        )

        # Upstream, Downstream, and Other MLPs
        self.upstream_mlp_1 = self.create_target_mlp()

        self.downstream_mlp_1 = self.create_target_mlp()

        self.passive_mlp_1 = self.create_target_mlp()

        # Attention Layers
        self.upstream_attention = CrossAttentionLayer(7, 14, self.v_dim)
        self.downstream_attention = CrossAttentionLayer(7, 14, self.v_dim)
        self.passive_attention = CrossAttentionLayer(7, 7, self.v_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def create_target_mlp(self):
        """Helper function to create target MLPs (upstream, downstream, other)."""
        return nn.Sequential(
            nn.Linear(self.v_dim, self.v_dim),
            nn.ELU(),
            nn.Linear(self.v_dim, 1)
        )

    def compute_attention_target(self, attention_layer, ego_x, x):
        """Helper to compute attention targets."""
        if x.size(0) > 0:
            return attention_layer(ego_x, x).squeeze()
        else:
            return torch.zeros(self.v_dim)

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
        a_column[0, 0] = a  # Set the first value to action
        # Append the column to the original matrix
        init_state = np.hstack((s[:, 1:-1], a_column))
        init_state = torch.from_numpy(init_state).float()
        # aug_states = fp
        aug_states = torch.cat((fp[:, : -2], fp[:, -1:]), dim=1)  # get rid of is_target column

        result = []
        for row in aug_states:
            bus_id_active = row[0]
            matching_rows = init_bus_ids == bus_id_active
            if torch.sum(matching_rows) > 0:
                s_init = init_state[matching_rows][0:1, :]  # Get the first matching row in aug_states
                s_aug = row[1:]
                flag = torch.ones(1, 1)
                state = torch.cat((s_init, s_aug.view(1, -1), flag), dim=1)
                result.append(state.squeeze())
        result = torch.stack(result, 0)

        mask = ~init_bus_ids.unsqueeze(1).eq(aug_bus_ids).any(dim=1)
        outstanding_init = init_state[mask]
        empty_placeholder = torch.full((outstanding_init.shape[0], aug_states.shape[1]), 0, dtype=torch.float)
        outstanding_init = torch.cat((outstanding_init, empty_placeholder), dim=1)
        result = torch.cat((result, outstanding_init), dim=0)
        return result

    def forward(self, xs):
        x_list, a_list, fp_list = xs
        u_x_targets, d_x_targets, p_x_targets, ego_targets = [], [], [], []

        for x, fp, a in zip(x_list, fp_list, a_list):
            s = torch.tensor(x, dtype=torch.float32)

            # Merge state, action, and feature points
            merged_s = self.merge(s, a, fp)

            # keep the original tensor of a for computation graph, useful for actor learning
            merged_s[0, 6] = a
            subject_bus_idx, subject_location = merged_s[0, 0].clone(), merged_s[0, 2].clone()

            # Normalize positional indices relative to the subject bus
            merged_s[:, 0] -= subject_bus_idx  # Normalize the "ord" column
            merged_s[:, 7] -= subject_bus_idx  # Normalize the second "ord_2" column

            # Separate ego, upstream, downstream, and other states
            ego_x = merged_s[0:1, :7]
            active_up_x = merged_s[(merged_s[:, 2] < subject_location) & (merged_s[:, -1] == 1)][:, :-1]
            active_down_x = merged_s[(merged_s[:, 2] > subject_location) & (merged_s[:, -1] == 1)][:, :-1]
            passive_x = merged_s[merged_s[:, -1] == 0][1:, :7]

            # Compute attention targets
            u_x_targets.append(self.compute_attention_target(self.upstream_attention, ego_x, active_up_x))
            d_x_targets.append(self.compute_attention_target(self.downstream_attention, ego_x, active_down_x))
            p_x_targets.append(self.compute_attention_target(self.passive_attention, ego_x, passive_x))

            ego_target = merged_s[0:1, [3, 4, 5, 6]].clone()
            ego_targets.append(ego_target.squeeze())

        # Stack and normalize targets
        u_x_targets = self.bn(torch.stack(u_x_targets, dim=0))
        d_x_targets = self.bn(torch.stack(d_x_targets, dim=0))
        o_x_targets = self.bn(torch.stack(p_x_targets, dim=0))
        ego_targets = torch.stack(ego_targets, dim=0)

        # Compute final MLP outputs for attention and ego critic
        u_x_1 = self.upstream_mlp_1(u_x_targets)
        d_x_1 = self.downstream_mlp_1(d_x_targets)
        p_x_1 = self.passive_mlp_1(o_x_targets)
        Q1 = self.ego_mlp(ego_targets)

        # Aggregate outputs
        G1 = Q1 + u_x_1 + d_x_1 + p_x_1
        return G1.view(-1, 1)


class Agent():
    def __init__(self, state_dim, name, seed=123, policy_noise=0.3, noise_clip=0.2, policy_freq=1):
        random.seed(seed)
        self.seed = seed
        self.name = name
        # self.gamma = 0.9
        self.gamma = 1
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

    def lr_decay(self, ratio=0.999):
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] *= ratio
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] *= ratio
        self.policy_noise *= ratio

    def choose_action(self, state, noise_scale=0.3):
        state = torch.tensor(state, dtype=torch.float)
        a = self.actor([state])[0].detach().numpy()
        # Add Gaussian noise for exploration

        # a = a + np.random.normal(0, noise_scale, size=a.shape)
        a = a + np.random.normal(0, self.policy_noise, size=a.shape)
        # Clip the action to be within the valid action range
        a = np.clip(a, 0, 3.0)
        return a

    def actor_output_variance(self, memories, agents_pool, batch_size=1024):
        """
        Calculates the variance of the outputs from a pool of actor models.

        Parameters:
            memories (list): List of replay memory tuples (s, fp, a, r, ns, nfp).
            batch_size (int): Number of samples to draw from the memories.
            agents_pool (list): List of agents, each having an actor model.

        Returns:
            torch.Tensor: Variance of outputs across agents for each state in the batch.
        """
        # Sample a batch from the memories
        batch_s = []
        # memory = random.sample(memories, batch_size)
        for s, _, _, _, _, _ in memories:
            batch_s.append(s)

        # Set all actors to evaluation mode
        for agent in agents_pool:
            agent.actor.eval()

        # Collect results from each actor
        teacher_results = []
        for s in batch_s:
            outputs_per_state = []
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            for agent in agents_pool:
                with torch.no_grad():  # Disable gradients for inference
                    result = agent.actor(state_tensor) * 180.0  # Ensure correct shape for input
                    outputs_per_state.append(result.squeeze(0))  # Remove batch dimension
            teacher_results.append(torch.stack(outputs_per_state))  # Shape: [num_agents, output_dim]

        # Calculate variance of outputs across agents for each state
        teacher_results = torch.stack(teacher_results)  # Shape: [batch_size, num_agents, output_dim], i.e [2000,4,1]
        output_variances = teacher_results.var(dim=1)  # Variance across agents for each state
        output_avg = teacher_results.mean(dim=1)
        return output_avg.mean().item(), output_variances.mean().item()

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

    def learn(self, memories, batch=256):
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
            Q = self.critic([batch_s, b_a, b_fp_pad])
            batch_ns_tensor = [torch.tensor(state, dtype=torch.float) for state in batch_ns]
            nb_a = self.actor_target(batch_ns_tensor).detach().view(-1, 1)
            noise = (torch.randn_like(nb_a) * self.policy_noise).clamp(0, self.noise_clip)
            nb_a = (nb_a + noise).clamp(0, 3.0)
            Q_ = self.critic_target([batch_ns, nb_a, b_nfp_pad])
            q_target = b_r + self.gamma * (Q_.detach()).view(-1, 1)

            loss_fn = nn.MSELoss()
            qloss = loss_fn(Q, q_target)
            self.critic_optim.zero_grad()
            qloss.backward()
            self.critic_optim.step()
            return qloss.item()

        def actor_learn():
            Q= self.critic([batch_s, batch_actor_a.view(-1, 1), batch_fp_critic_t])
            policy_loss = -torch.mean(Q)
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
