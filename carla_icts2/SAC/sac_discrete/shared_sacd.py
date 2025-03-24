import os

import numpy as np
import torch
from torch.optim import Adam

from SAC.sac_discrete.sacd.model import CateoricalPolicy, DQNBase, TwinnedQNetwork
from SAC.sac_discrete.sacd.utils import disable_gradients

from .base import BaseAgent


class SharedSacdAgent(BaseAgent):
    def __init__(
        self,
        env,
        test_env,
        log_dir,
        num_steps=100000,
        batch_size=64,
        lr=0.0003,
        memory_size=1000000,
        gamma=0.99,
        multi_step=1,
        target_entropy_ratio=0.98,
        start_steps=20000,
        update_interval=4,
        target_update_interval=8000,
        use_per=False,
        dueling_net=False,
        num_eval_steps=125000,
        save_interval=100000,
        max_episode_steps=27000,
        log_interval=10,
        eval_interval=1000,
        cuda=True,
        seed=0,
        path=None,
        max_grad_norm=None,
    ):
        super().__init__(
            env,
            test_env,
            log_dir,
            num_steps,
            batch_size,
            memory_size,
            gamma,
            multi_step,
            target_entropy_ratio,
            start_steps,
            update_interval,
            target_update_interval,
            use_per,
            num_eval_steps,
            max_episode_steps,
            save_interval,
            log_interval,
            eval_interval,
            cuda,
            seed,
            lr=lr,
            max_grad_norm=max_grad_norm,
        )

        print(num_steps)
        # Define networks.
        self.conv = DQNBase(self.env.observation_space.shape[2]).to(self.device)
        self.policy = CateoricalPolicy(
            self.env.observation_space.shape[2], self.env.action_space.n, shared=True,
        ).to(self.device)
        self.online_critic = TwinnedQNetwork(
            self.env.observation_space.shape[2],
            self.env.action_space.n,
            dueling_net=dueling_net,
            shared=True,
        ).to(device=self.device)
        self.target_critic = (
            TwinnedQNetwork(
                self.env.observation_space.shape[2],
                self.env.action_space.n,
                dueling_net=dueling_net,
                shared=True,
            )
            .to(device=self.device)
            .eval()
        )

        # Copy parameters of the learning network to the target network.
        self.target_critic.load_state_dict(self.online_critic.state_dict())

        if path:
            self.resume = True
            self.conv.load_state_dict(torch.load(path + "conv.pth"))
            self.policy.load_state_dict(torch.load(path + "policy.pth"))
            self.online_critic.load_state_dict(torch.load(path + "online_critic.pth"))
            self.target_critic.load_state_dict(torch.load(path + "target_critic.pth"))

        # Disable gradient calculations of the target network.
        disable_gradients(self.target_critic)

        self.policy_optim = Adam(
            list(self.conv.parameters()) + list(self.policy.parameters()), lr=lr,
        )
        self.q1_optim = Adam(
            list(self.conv.parameters()) + list(self.online_critic.Q1.parameters()), lr=lr,
        )
        self.q2_optim = Adam(self.online_critic.Q2.parameters(), lr=lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy = -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha = torch.tensor(-0.69, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def explore(self, state):
        # Act with randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.0
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action, _, _ = self.policy.sample(state)
            curr_q1 = self.online_critic.Q1(state)
            curr_q2 = self.online_critic.Q2(state)
            q = torch.min(curr_q1, curr_q2)
            critic_action = torch.argmax(q, dim=1)
        return action.item(), critic_action.item()

    def exploit(self, state):
        # Act without randomness.
        state, t = state
        state = torch.ByteTensor(state[None, ...]).to(self.device).float() / 255.0
        t = torch.FloatTensor(t[None, ...]).to(self.device)
        with torch.no_grad():
            state = self.conv(state)
            state = torch.cat([state, t], dim=1)
            action = self.policy.act(state)
        return action.item()

    def update_target(self):
        # taken from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
        polyak = 0.005
        # self.target_critic.load_state_dict(self.online_critic.state_dict())
        with torch.no_grad():
            for p, p_targ in zip(self.online_critic.parameters(), self.target_critic.parameters(), strict=False):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        states, t = states
        states = self.conv(states)
        states = torch.cat([states, t], dim=-1)
        curr_q1 = self.online_critic.Q1(states).gather(1, actions.long())
        curr_q2 = self.online_critic.Q2(states.detach()).gather(1, actions.long())
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_states, t_new = next_states
            next_states = self.conv(next_states)
            next_states = torch.cat([next_states, t_new], dim=1)
            _, action_probs, log_action_probs = self.policy.sample(next_states)
            next_q1, next_q2 = self.target_critic(next_states)
            next_q = (
                action_probs * (torch.min(next_q1, next_q2) - self.alpha * log_action_probs)
            ).sum(dim=1, keepdim=True)

        assert rewards.shape == next_q.shape
        return rewards + (1.0 - dones) * self.gamma_n * next_q

    def calc_critic_loss(self, batch, weights):
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        states, t = states

        with torch.no_grad():
            states = self.conv(states)
        states = torch.cat([states, t], dim=1)

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs = self.policy.sample(states)

        with torch.no_grad():
            # Q for every actions to calculate expectations of Q.
            q1, q2 = self.online_critic(states)
            q = torch.min(q1, q2)

        # Expectations of entropies.
        entropies = -torch.sum(action_probs * log_action_probs, dim=1, keepdim=True)

        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (-q - self.alpha * entropies)).mean()

        return policy_loss, entropies.detach()

    def calc_entropy_loss(self, entropies, weights):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropies) * weights)
        return entropy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self.conv.save(os.path.join(save_dir, "conv.pth"))
        self.policy.save(os.path.join(save_dir, "policy.pth"))
        self.online_critic.save(os.path.join(save_dir, "online_critic.pth"))
        self.target_critic.save(os.path.join(save_dir, "target_critic.pth"))
