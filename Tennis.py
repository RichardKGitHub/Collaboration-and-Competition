#!/usr/bin/env python
# coding: utf-8

"""

COPYRIGHT NOTICE

This File contains Code provided by Udacity to be used by its students in order to solve the given projects.
You need to ask Udacity Inc. for permission in case you like to use that sections in your Software.
The functions and classes concerned are indicated by a comment.
Link to Udacity: https://eu.udacity.com/
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import commentjson
import datetime
import copy
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
from unityagents import UnityEnvironment


def hidden_init(layer):
    '''
    this function was provided by Udacity Inc.
    '''
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class MyAppLookupError(LookupError):
    """raise this when there's a lookup error for my app"""
    # source of this class:
    # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python/24065533#24065533


class EnvUtils:
    def __init__(self):
        self.states = np.empty(shape=(1, admin.number_of_agents, admin.state_size))
        self.next_states = np.empty(shape=(1, admin.number_of_agents, admin.state_size))
        self.states_normalized = np.empty(shape=(1, admin.number_of_agents, admin.state_size))
        self.next_states_normalized = np.empty(shape=(1, admin.number_of_agents, admin.state_size))

    def get_random_start_state(self):
        env_info_tr = env.reset(train_mode=admin.env_train_mode)[brain_name]
        for _ in range(admin.number_of_random_actions):
            actions = np.clip(np.random.randn(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
            env_info_tr = env.step(actions)[brain_name]
        self.states = env_info_tr.vector_observations
        self.normalize_states()
        return None

    def normalize_states(self):
        if admin.normalize_states:
            iInterpolationMinOrig = np.array(admin.lInterpolParam[0])
            iInterpolationMaxOrig = np.array(admin.lInterpolParam[1])
            iInterpolationMinNew = np.ones(admin.state_size) * -1
            iInterpolationMaxNew = np.ones(admin.state_size)
            fAnstieg = (iInterpolationMaxNew - iInterpolationMinNew) / (iInterpolationMaxOrig - iInterpolationMinOrig)
            fOffset = (iInterpolationMaxOrig * iInterpolationMinNew - iInterpolationMinOrig * iInterpolationMaxNew) / (
                    iInterpolationMaxOrig - iInterpolationMinOrig)
            # clip resulting normalized states if requested
            if admin.lInterpolParam[2]:
                self.states_normalized = np.clip(fAnstieg * self.states + fOffset, -1, 1)
                self.next_states_normalized = np.clip(fAnstieg * self.next_states + fOffset, -1, 1)
            else:
                self.states_normalized = fAnstieg * self.states + fOffset
                self.next_states_normalized = fAnstieg * self.next_states + fOffset
        else:
            self.states_normalized = self.states.copy()
            self.next_states_normalized = self.next_states.copy()
        return None


class Actor1(nn.Module):
    """Actor (Policy) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fcs1_units, fcs2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fcs2_units)
        self.fc3 = nn.Linear(fcs2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic1(nn.Module):
    """Critic (Value) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fcs1_units, fc2_units, fc3_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Actor2(nn.Module):
    """Actor (Policy) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return F.tanh(self.fc3(x))


class Critic2(nn.Module):
    """Critic (Value) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.fc1(state)
        xs = F.relu(x)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    '''
    this class was provided by Udacity Inc.
    '''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.state_noise = np.zeros(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state_noise = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state_noise
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state_noise = x + dx
        return self.state_noise


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, actions, environment_info):
        """Add a new experience to memory."""
        # Save experience / reward
        for agent_count in range(admin.number_of_agents):
            state = env_utils.states_normalized[agent_count]
            action = actions[agent_count]
            reward = environment_info.rewards[agent_count]
            next_state = env_utils.next_states_normalized[agent_count]
            is_done = environment_info.local_done[agent_count]
            e = self.experience(state, action, reward, next_state, is_done)
            self.memory.append(e)
        return None

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Administration:
    """defines looped interactions of the Agent (use case and training)"""

    def __init__(self, config_data_interact):
        self.load_indices = config_data_interact['load_indices']
        self.save_indices = config_data_interact['save_indices']
        self.path_load = config_data_interact['path_load']
        self.path_save = config_data_interact['path_save']
        self.load_parameters_from_file = config_data_interact['load_parameters_from_file']
        self.save_weights = config_data_interact['save_weights']
        self.save_plot = config_data_interact['save_plot']
        self.show_plot = config_data_interact['show_plot']
        self.episodes_train = config_data_interact['episodes_train']
        self.episodes_test = config_data_interact['episodes_test']
        self.target_reward = config_data_interact['target_reward']
        self.consecutive_episodes_required = config_data_interact['consecutive_episodes_required']
        self.network_type = config_data_interact['network_type']
        self.actor_fcs1_units = config_data_interact['actor_fcs1_units']
        self.actor_fcs2_units = config_data_interact['actor_fcs2_units']
        self.critic_fcs1_units = config_data_interact['critic_fcs1_units']
        self.critic_fcs2_units = config_data_interact['critic_fcs2_units']
        self.critic_fcs3_units = config_data_interact['critic_fcs3_units']
        self.add_noise = config_data_interact['add_noise']
        self.noise_scale_best_small = config_data_interact['noise_scale_best_small']
        self.noise_scale_best_big = config_data_interact['noise_scale_best_big']
        self.noise_scale_worst = config_data_interact['noise_scale_worst']
        self.sigma = config_data_interact['sigma']
        self.noise_theta = config_data_interact['noise_theta']
        self.noise_sigma = config_data_interact['noise_sigma']
        self.random_seed = config_data_interact['random_seed']
        self.epsilon_start = config_data_interact['epsilon_start']
        self.epsilon_end = config_data_interact['epsilon_end']
        self.epsilon_decay = config_data_interact['epsilon_decay']
        self.epsilon_test = config_data_interact['epsilon_test']
        self.epsilon = self.epsilon_test
        self.buffer_size_admin = config_data_interact['buffer_size_admin']
        self.batch_size_admin = config_data_interact['batch_size_admin']
        self.gamma = config_data_interact['gamma']
        self.tau = config_data_interact['tau']
        self.learning_rate_actor = config_data_interact['learning_rate_actor']
        self.learning_rate_critic = config_data_interact['learning_rate_critic']
        self.weight_decay = config_data_interact['weight_decay']
        self.learn_every = config_data_interact['learn_every']
        self.consecutive_learning_steps = config_data_interact['consecutive_learning_steps']
        self.lInterpolParam = config_data_interact['lInterpolParam']
        self.normalize_states = config_data_interact['normalize_states']
        self.number_of_agents = config_data_interact['number_of_agents']  # 20
        self.agents_duplication_factor = config_data_interact['agents_duplication_factor']
        self.number_of_random_actions = config_data_interact['number_of_random_actions']
        self.max_steps_per_training_episode = config_data_interact['max_steps_per_training_episode']
        self.env_train_mode = config_data_interact['env_train_mode']
        self.environment_path = config_data_interact['environment_path']

        if train is True:
            self.scores_all_episodes_and_NW = np.empty(shape=(1, 3, self.episodes_train))
        else:
            self.scores_all_episodes_and_NW = np.empty(shape=(1, 3, self.episodes_test))
        self.rewards_all_networks = np.empty(shape=(3, 1))
        self.state_size = 24
        self.action_size = 2

        self.i_update = 0
        self.q_loss_loss_one_episode = np.zeros(shape=(5, int(self.max_steps_per_training_episode/self.learn_every*self.consecutive_learning_steps)))
        self.q_loss_loss = np.zeros(shape=(1, 5, self.episodes_train))
        self.epsilon_sigma_noise = np.zeros(shape=(1, 3, self.episodes_train))
        self.sigma_noiseMean = np.zeros(shape=(1, 2, self.max_steps_per_training_episode))
        self.step_counter = 0
        self.max_scores = []
        '''
        up from here this Function may contain Code provided by Udacity Inc.
        '''
        if self.network_type == "DDPG_1":
            self.actor_local = Actor1(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor1(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_target = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_2":
            self.actor_local = Actor2(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor2(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units).to(device)
            self.critic_target = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"DDPG_1\" or \"DDPG_2\"")
        self.noise = OUNoise(self.action_size, self.random_seed, theta=self.noise_theta, sigma=self.noise_sigma)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size_admin, self.batch_size_admin, self.random_seed)
        self.soft_update_started = False

    def train_ddpg(self):
        if self.load_parameters_from_file:
            self.load_parameter()
        self.soft_update_started = False
        self.epsilon = self.epsilon_start
        solved = False                          # if mean over 100 episodes of max score > 0.5
        solved1 = False                         # if max score of one Episode is reached
        time_new = time_start = datetime.datetime.now()
        for i in range(self.episodes_train):
            for j in range(1):        # first approach was evolutionary with j multiple weights tested one after another
                self.noise.reset()
                min_reward, mean_reward, max_reward = admin.get_rewards_ddpg(trainmode=self.env_train_mode)
                self.scores_all_episodes_and_NW[j, 0, i] = min_reward
                self.scores_all_episodes_and_NW[j, 1, i] = mean_reward
                self.scores_all_episodes_and_NW[j, 2, i] = max_reward
                # for one episode: calc mean of <reward, Q_target, Q_expected, critic_loss and actor_loss> over all
                # updates and agents in this episode
                mean_of_q_loss_loss = self.q_loss_loss_one_episode.mean(axis=1)
                self.q_loss_loss[j, 0, i] = mean_of_q_loss_loss[0]      # for reward
                self.q_loss_loss[j, 1, i] = mean_of_q_loss_loss[1]      # for Q_target
                self.q_loss_loss[j, 2, i] = mean_of_q_loss_loss[2]      # for Q_expected
                self.q_loss_loss[j, 3, i] = mean_of_q_loss_loss[3]      # for critic_loss
                self.q_loss_loss[j, 4, i] = mean_of_q_loss_loss[4]      # for actor_loss
                self.epsilon_sigma_noise[j, 0, i] = self.epsilon
                # for one episode: calc mean of <sigma and max of noise> over all steps and agents in this episode
                mean_of_sigma_noise = self.sigma_noiseMean[j].mean(axis=1)
                self.epsilon_sigma_noise[j, 1, i] = mean_of_sigma_noise[0]      # for sigma
                self.epsilon_sigma_noise[j, 2, i] = mean_of_sigma_noise[1]      # for max noise
            if i >= self.consecutive_episodes_required and not solved:
                # if max_reward in one Episode reaches 2 or higher
                if max_reward >= 2.5 and not solved1:
                    print(f"\n\nscore >= 2.5 in episode {i - self.consecutive_episodes_required + 1}\t"
                          f"max_score: {max_reward}\n")
                    if self.save_weights:
                        self.save_parameter('s1_')
                    solved1 = True
                # if mean of Results reaches the goal value
                # (mean over episodes of max of every episode Position [2] of scores_all_episodes)
                if self.scores_all_episodes_and_NW[0][:, i - self.consecutive_episodes_required:i + 1].mean(axis=1)[2] >= self.target_reward:
                    print(f"\n\ntarget reward reached in episode: "
                          f"{i - self.consecutive_episodes_required + 1}: mean_of_max_of_rewards: "
                          f"{self.scores_all_episodes_and_NW[0][:, i - self.consecutive_episodes_required:i + 1].mean(axis=1)[2]}\n")
                    if self.save_weights:
                        self.save_parameter('s_')       # save weights at moment of solving
                    solved = True                       # <target reached> if-statement gets only visited once if break is commented out
                    # break
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            # print some information's about the current scores
            if (i + 1) % 1 == 0:
                time_old = time_new
                time_new = datetime.datetime.now()
                if i > 99:
                    print('\rscores: <mean over last 100 Episodes> | <last Episode>: min: {:.5f} | {:.5f}\tmean: {:.5f} | {:.5f}\t'
                          'max: {:.5f} | {:.5f}\tEpisode {}/{}\tTime since start: {}\tdeltaTime: '
                          '{}'.format(self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[0], min_reward,
                                      self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[1], mean_reward,
                                      self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[2], max_reward,
                                      i+1, self.episodes_train, str(time_new-time_start).split('.')[0],
                                      str(time_new-time_old).split('.')[0]), end="")
                else:
                    print('\rscores last episode: min_Score {:.5f} \tAverage_Score: {:.5f} \tMax_Score {:.5f} \tEpisode {}/{}\t'
                          'Time since start: {}\tdeltaTime: {}'.format(min_reward, mean_reward, max_reward, i + 1,
                                                                        self.episodes_train,
                                                                        str(time_new - time_start).split('.')[0],
                                                                        str(time_new - time_old).split('.')[0]), end="")
        if self.save_weights:
            self.save_parameter('g_')       # save weights at end of training
        admin.plot_results()                # plot results
        return None

    def test(self):
        # load policy if needed
        self.load_parameter()
        self.epsilon = self.episodes_test
        self.max_scores = []
        self.add_noise = False
        time_new = time_start = datetime.datetime.now()
        for i in range(self.episodes_test):
            j = 0                     # first approach was evolutionary with j multiple weights tested one after another
            min_reward, mean_reward, max_reward = admin.get_rewards_ddpg(trainmode=False)
            self.scores_all_episodes_and_NW[j, 0, i] = min_reward
            self.scores_all_episodes_and_NW[j, 1, i] = mean_reward
            self.scores_all_episodes_and_NW[j, 2, i] = max_reward
            time_old = time_new
            time_new = datetime.datetime.now()
            if i > self.consecutive_episodes_required - 1:
                print('\rscores: <mean over last 100 Episodes> | <last Episode>: min: {:.5f} | {:.5f}\t'
                      'mean: {:.5f} | {:.5f}\tmax: {:.5f} | {:.5f}\tEpisode {}/{}\tTime since start: {}\tdeltaTime:'
                      ' {}'.format(self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[0], min_reward,
                                  self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[1], mean_reward,
                                  self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[2], max_reward,
                                  i+1, self.episodes_test, str(time_new-time_start).split('.')[0],
                                  str(time_new-time_old).split('.')[0]), end="")
                self.max_scores.append(self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[1])
            else:
                print('\rscores last episode: min_Score {:.5f} \tAverage_Score: {:.5f} \tMax_Score {:.5f} \t'
                      'Episode {}/{}\tTime since start: {}\t'
                      'deltaTime: {}'.format(min_reward, mean_reward, max_reward,
                                             i + 1, self.episodes_test, str(time_new - time_start).split('.')[0],
                                             str(time_new - time_old).split('.')[0]), end="")
        admin.plot_results()
        return None


    def get_rewards_ddpg(self, trainmode=False):
        score_one_episode = np.zeros(self.number_of_agents)
        self.i_update = 0                                           # used for q and loss documentation
        self.sigma_noiseMean = np.zeros(shape=(1, 2, self.max_steps_per_training_episode))
        env_utils.get_random_start_state()
        for step in range(self.max_steps_per_training_episode):
            self.step_counter = step                                # used for sigma and Noise documentation
            actions = self.act()
            env_info = env.step(actions)[brain_name]
            score_one_episode += np.array(env_info.rewards)
            if np.any(env_info.local_done):
                print(f"\tstep: {step}")
                break
            env_utils.next_states = env_info.vector_observations
            env_utils.normalize_states()
            if trainmode:
                self.memory.add(actions, env_info)
                if step % self.learn_every == 0:
                    for _ in range(self.consecutive_learning_steps):
                        self.step()
            env_utils.states = env_utils.next_states.copy()
            env_utils.states_normalized = env_utils.next_states_normalized.copy()
        return score_one_episode.min(), score_one_episode.mean(), score_one_episode.max()

    def plot_results(self):
        """plot graphs of score and additional information"""
        '''plot scores'''
        x_plot = np.arange(self.scores_all_episodes_and_NW.shape[-1])
        list_of_names = ['min_scores', 'mean_scores', 'max_scores']
        for row in range(1):
            for column in range(3):
                plt.subplot(1, 3, (column + 1) * (row + 1))
                plt.plot(x_plot, self.scores_all_episodes_and_NW[1 - 1, column, :], '-')
                plt.title(list_of_names[column])
                plt.xlabel('episodes')
                plt.ylabel('scores')
        if self.save_plot:
            # save the plot
            plt.savefig(self.path_save + "scores_" + self.save_indices + ".png")
        if self.show_plot:
            # plot the scores
            plt.show()
        # plot training specific plots
        if train:
            '''plot reward, Q_target, Q_expected, critic_loss and actor_loss'''
            x2_plot = np.arange(self.q_loss_loss.shape[-1])
            list_of_names = ['reward_one_step', 'Q_target', 'Q_expected', 'critic_loss',
                             'actor_loss']
            for row in range(1):
                for column in range(5):
                    plt.subplot(1, 5, (column + 1) * (row + 1))
                    plt.plot(x2_plot, self.q_loss_loss[1 - 1, column, :], '-')
                    plt.title(list_of_names[column])
            if self.save_plot:
                # save the plot
                plt.savefig(self.path_save + "losses_" + self.save_indices + ".png")
            if self.show_plot:
                # plot the losses
                plt.show()
            '''plot epsilon, sigma_noise and max_noise'''
            list_of_names = ['epsilon_if_noise', 'sigma_noise', 'max_noise']
            x3_plot = np.arange(self.epsilon_sigma_noise.shape[-1])
            for row in range(1):
                for column in range(3):
                    plt.subplot(1, 3, (column + 1) * (row + 1))
                    plt.plot(x3_plot, self.epsilon_sigma_noise[1 - 1, column, :], '-')
                    plt.title(list_of_names[column])
            if self.save_plot:
                # save the plot
                plt.savefig(self.path_save + "noise_" + self.save_indices + ".png")
            if self.show_plot:
                # plot the noise information
                plt.show()
        # plot testing specific plots
        else:
            x4_plot = np.arange(self.consecutive_episodes_required + 1,
                                self.consecutive_episodes_required + 1 + len(self.max_scores))
            plt.subplot(1, 3, 2)
            plt.plot(x4_plot, self.max_scores, '-')
            plt.title('mean over 100 consecutive scores')
            plt.xlabel('episodes')
            plt.ylabel('mean of scores over 100 episodes')
            if self.save_plot:
                # save the plot
                plt.savefig(self.path_save + "mean_score_" + self.save_indices + ".png")
            if self.show_plot:
                # plot the noise information
                plt.show()
        return None

    def init_agent(self):
        '''
        this Function may contain Code provided by Udacity Inc.
        '''
        # reset the environment
        env_info = env.reset(train_mode=self.env_train_mode)[brain_name]
        # number of agents
        self.number_of_agents = len(env_info.agents)
        print('Number of agents:', self.number_of_agents)
        # size of each action
        self.action_size = brain.vector_action_space_size
        print('Size of each action:', self.action_size)
        # examine the state space
        states = env_info.vector_observations
        self.state_size = states.shape[1]
        print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], self.state_size))
        if self.network_type == "DDPG_1":
            self.actor_local = Actor1(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor1(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_target = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_2":
            self.actor_local = Actor2(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor2(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units).to(device)
            self.critic_target = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"DDPG_1\" or \"DDPG_2\"")
        # Noise process
        self.noise = OUNoise(self.action_size, self.random_seed, theta=self.noise_theta, sigma=self.noise_sigma)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size_admin, self.batch_size_admin, self.random_seed)
        return

    def step(self):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        '''
        this function contains some changes but was mainly provided by Udacity Inc.
        '''
        # Learn, if enough samples are available in memory
        if len(self.memory) > max(self.batch_size_admin, 255):
            experiences = self.memory.sample()
            self.learn(experiences)
            self.i_update += 1
        return None

    def act(self):
        """Returns actions for given state as per current policy."""
        '''
        this function contains changes but was previously provided by Udacity Inc.
        '''
        state = torch.from_numpy(env_utils.states_normalized).float().to(device)
        # action by network
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        self.sigma_noiseMean[0, 0, self.step_counter] = self.noise_sigma            # for plot of sigma
        # only add noise during training and if requested by the config
        if self.add_noise and train:
            if random.random() < self.epsilon:      # add noise with probability epsilon --> no noise if epsilon-greedy
                noise = self.noise.sample()
                action += noise
                self.sigma_noiseMean[0, 1, self.step_counter] = noise.max()         # for plot of max noise
            else:
                self.sigma_noiseMean[0, 1, self.step_counter] = 0                   # for plot of max noise, if no noise
        else:
            self.sigma_noiseMean[0, 1, self.step_counter] = 0                       # for plot of max noise, if no noise
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        '''
        this function contains some changes but was mainly provided by Udacity Inc.
        '''
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        '''suggestet clipping'''
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update_started = True
        # save reward Q_target Q_expected critic_loss and actor_loss values for plot #
        self.q_loss_loss_one_episode[0, self.i_update] = rewards[0]
        self.q_loss_loss_one_episode[1, self.i_update] = Q_targets[0].detach().cpu().numpy()
        self.q_loss_loss_one_episode[2, self.i_update] = Q_expected[0].detach().cpu().numpy()
        self.q_loss_loss_one_episode[3, self.i_update] = critic_loss
        self.q_loss_loss_one_episode[4, self.i_update] = actor_loss
        return None

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        '''
        this function contains some changes but was mainly provided by Udacity Inc.
        '''
        if self.soft_update_started:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        else:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)

    def save_parameter(self, affix):
        torch.save(self.actor_local.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_actor_local.pt')
        torch.save(self.actor_target.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_actor_target.pt')
        torch.save(self.critic_local.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_critic_local.pt')
        torch.save(self.critic_target.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_critic_target.pt')
        return None

    def load_parameter(self):
        self.actor_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_local.pt'))
        self.actor_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_target.pt'))
        self.critic_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_local.pt'))
        self.critic_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_target.pt'))
        return None


if __name__ == "__main__":
    # Idea of parser: https://docs.python.org/2/howto/argparse.html
    parser = argparse.ArgumentParser(description='Interacting Agent')
    parser.add_argument('--train', type=str, default='False', help='True: train the agent; '
                                                                  'default=False: test the agent')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='Name of config_file in root of Continuous_Control')
    args = parser.parse_args()

    # convert argparse arguments to bool since argparse doesn't treat booleans as expected:
    if args.train == 'True' or args.train == 'true' or args.train == 'TRUE':
        train = True
    elif args.train == 'False' or args.train == 'false' or args.train == 'FALSE':
        train = False
    else:
        raise MyAppLookupError('--train can only be True or False | default: False')

    # load config_file.json
    # Idea: https://commentjson.readthedocs.io/en/latest/
    with open(args.config_file, 'r') as f:
        config_data = commentjson.load(f)

    '''
    from here on this function may contain some Code provided by Udacity Inc.
    '''
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # initialize configuration
    admin = Administration(config_data)

    # initialize Environment
    env = UnityEnvironment(file_name=admin.environment_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # initialize Agent
    admin.init_agent()

    # initialize Environment Utilities
    env_utils = EnvUtils()

    # train or test the Agent
    if train is True:
        print(f"\nTrain the Network using config_file <{args.config_file}> on device <{device}> "
              f"with weights-save-index <{admin.save_indices}>")
        admin.train_ddpg()
    else:
        print(f"\nTest the Network with fixed weights from <checkpoint_{admin.load_indices}.pth> "
              f"using config_file <{args.config_file}> on device <{device}>")
        admin.env_train_mode = False
        admin.test()
    env.close()

