import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import commentjson
import datetime
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from unityagents import UnityEnvironment
from collections import deque



class NetworkFullyConnected(nn.Module):
    """Actor (Policy) Model."""
    ''' 
    this class was provided by Udacity Inc.
    '''

    def __init__(self, state_size, action_size, seed=1203, fc1_units=21, fc2_units=10):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(NetworkFullyConnected, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)  # self.sig = nn.Sigmoid()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # return self.sig(self.fc2(x))

    # from Udacity REINFORCE in Policy Gradient Methods
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class NetworkOneHiddenLayer(nn.Module):
    '''from CEM.py  Lesson2 Nr.9 Workspace'''

    def __init__(self, s_size, a_size, h_size=8):
        # 00: h_size=16
        # 01: h_size=16
        # 02: h_size=8

        super(NetworkOneHiddenLayer, self).__init__()
        # state, hidden layer, action sizes
        self.s_size = s_size
        self.h_size = h_size
        self.a_size = a_size
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        # print(f"weights.shape: {weights.shape}")
        # print(f"weights[:s_size * h_size].shape: {weights[:s_size * h_size].shape}")
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = 2*x - 1
        return x.cpu().data

    # def evaluate(self, weights, gamma=1.0, max_t=5000):
    #     self.set_weights(weights)
    #     episode_return = 0.0
    #     state = self.env.reset()
    #     for t in range(max_t):
    #         state = torch.from_numpy(state).float().to(device)
    #         action = self.forward(state)
    #         state, reward, done, _ = self.env.step(action)
    #         episode_return += reward * math.pow(gamma, t)
    #         if done:
    #             break
    #     return episode_return


class EnvUtils:
    def __init__(self):
        self.states = np.empty(shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))
        self.normalized_states = np.empty(
            shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))

    def set_states(self, states):
        self.states = states
        self.normalize_states()
        return None

    def get_random_start_state(self):
        for _ in range(admin.number_of_random_actions):
            actions = np.clip(np.random.randn(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
            env_info_tr = env.step(actions)[brain_name]
        self.states = env_info_tr.vector_observations
        self.normalize_states()
        return

    def get_states_min_max_Values(self):
        # perform some random steps to get a random starting point
        _ = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = []
        actions_ = []
        for _ in range(admin.episodes_test):
            for _ in range(400):
                actions = np.clip(np.random.randn(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
                # print(f"actions_1:{actions_1}\nbrain_name={brain_name}")
                env_info_tr = env.step(actions)[brain_name]
                actions_.append(actions)
                states.append(env_info_tr.vector_observations)  # get next state (for each agent)
            _ = env.reset(train_mode=True)[brain_name]
        states = np.array(states)
        # print(f"states: {states}")
        min1 = states.min(axis=1)
        # print(f"min1: {min1}")
        min2 = min1.min(axis=0)
        # print(f"min2: {min2}")
        max2 = states.max(axis=1).max(axis=0)
        # med=states.mean(axis=1).mean(axis=0)
        print(f"stateMin: {min2}\n\nstateMax: {max2}")
        return None

    def normalize_states(self):
        iInterpolationMinOrig = np.array(admin.lInterpolParam[0])
        iInterpolationMaxOrig = np.array(admin.lInterpolParam[1])
        iInterpolationMinNew = np.ones(admin.state_size) * -1
        iInterpolationMaxNew = np.ones(admin.state_size)
        fAnstieg = (iInterpolationMaxNew - iInterpolationMinNew) / (iInterpolationMaxOrig - iInterpolationMinOrig)
        fOffset = (iInterpolationMaxOrig * iInterpolationMinNew - iInterpolationMinOrig * iInterpolationMaxNew) / (
                iInterpolationMaxOrig - iInterpolationMinOrig)
        if admin.lInterpolParam[2]:  # clip resulting normalized states if requested
            self.normalized_states = np.clip(fAnstieg * env_utils.states + fOffset, -1, 1)
        else:
            self.normalized_states = fAnstieg * env_utils.states + fOffset
        # print(f"aInterpolatedData: {self.normalized_states}")
        return None


class MyAppLookupError(LookupError):
    """raise this when there's a lookup error for my app"""
    # source of this class:
    # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python/24065533#24065533


class Administration:
    """defines looped interactions of the Agent (use case and training)"""

    def __init__(self, config_data_interact):
        self.load_indices = config_data_interact['load_indices']
        self.save_indices = config_data_interact['save_indices']
        self.path_load = config_data_interact['path_load']
        self.path_save = config_data_interact['path_save']
        self.load_scores_version = config_data_interact['load_scores_version']
        self.save_weights = config_data_interact['save_weights']
        self.save_plot = config_data_interact['save_plot']
        self.show_plot = config_data_interact['show_plot']
        self.episodes_train = config_data_interact['episodes_train']
        self.episodes_test = config_data_interact['episodes_test']
        self.target_reward = config_data_interact['target_reward']
        self.consecutive_episodes_required = config_data_interact['consecutive_episodes_required']
        self.network_type = config_data_interact['network_type']
        self.num_of_parallel_networks = config_data_interact['num_of_parallel_networks']  # 50
        self.keep_weights_n_best_min = config_data_interact['keep_weights_n_best_min']
        self.keep_weights_n_best_mean = config_data_interact['keep_weights_n_best_mean']
        self.keep_weights_n_best_max = config_data_interact['keep_weights_n_best_max']
        self.keep_weights_n_best_min_small_change = config_data_interact['keep_weights_n_best_min_small_change']
        self.keep_weights_n_best_mean_small_change = config_data_interact['keep_weights_n_best_mean_small_change']
        self.keep_weights_n_best_max_small_change = config_data_interact['keep_weights_n_best_max_small_change']
        self.keep_weights_n_best_min_big_change = config_data_interact['keep_weights_n_best_min_big_change']
        self.keep_weights_n_best_mean_big_change = config_data_interact['keep_weights_n_best_mean_big_change']
        self.keep_weights_n_best_max_big_change = config_data_interact['keep_weights_n_best_max_big_change']
        self.keep_weights_n_worst_min_big_change = config_data_interact['keep_weights_n_worst_min_big_change']
        self.keep_weights_n_worst_mean_big_change = config_data_interact['keep_weights_n_worst_mean_big_change']
        self.keep_weights_n_worst_max_big_change = config_data_interact['keep_weights_n_worst_max_big_change']
        self.noise_scale_best_small = config_data_interact['noise_scale_best_small']
        self.noise_scale_best_big = config_data_interact['noise_scale_best_big']
        self.noise_scale_worst = config_data_interact['noise_scale_worst']
        self.sigma = config_data_interact['sigma']
        # self.epsilon_start = config_data_interact['epsilon_start']
        # self.epsilon_end = config_data_interact['epsilon_end']
        # self.epsilon_decay = config_data_interact['epsilon_decay']
        # self.epsilon_test = config_data_interact['epsilon_test']
        # self.buffer_size = config_data_interact['buffer_size']
        # self.batch_size = config_data_interact['batch_size']
        # self.gamma = config_data_interact['gamma']
        # self.tau = config_data_interact['tau']
        # self.learning_rate = config_data_interact['learning_rate']
        # self.update_target_every = config_data_interact['update_target_every']
        self.lInterpolParam = config_data_interact['lInterpolParam']
        self.number_of_agents = config_data_interact['number_of_agents']  # 20
        self.agents_duplication_factor = config_data_interact['agents_duplication_factor']
        self.number_of_random_actions = config_data_interact['number_of_random_actions']
        self.max_steps_per_training_episode = config_data_interact['max_steps_per_training_episode']
        self.num_of_same_act_repetition = config_data_interact['num_of_same_act_repetition']
        self.env_train_mode = config_data_interact['env_train_mode']
        self.environment_path = config_data_interact['environment_path']
        if train is True:
            self.rewards_all_episodes = np.empty(shape=(3, self.episodes_train))
        else:
            self.rewards_all_episodes = np.empty(shape=(3, self.episodes_test))
        self.rewards_all_networks = np.empty(shape=(3, self.num_of_parallel_networks))
        self.weightslist = np.empty(shape=(self.num_of_parallel_networks, 612))
        self.nextweightslist = np.empty(shape=(self.num_of_parallel_networks, 612))
        self.state_size = 33
        self.weights_dim = 612
        # check correctness of keep_weights inputs
        sum_of_keep_weights = self.keep_weights_n_best_min + self.keep_weights_n_best_mean + \
                              self.keep_weights_n_best_max + self.keep_weights_n_best_min_small_change + \
                              self.keep_weights_n_best_mean_small_change + self.keep_weights_n_best_max_small_change + \
                              self.keep_weights_n_best_min_big_change + self.keep_weights_n_best_mean_big_change + \
                              self.keep_weights_n_best_max_big_change + self.keep_weights_n_worst_min_big_change + \
                              self.keep_weights_n_worst_mean_big_change + self.keep_weights_n_worst_max_big_change
        if self.num_of_parallel_networks < sum_of_keep_weights:
            raise MyAppLookupError(f"\nthe Number of parallel Networks ({self.num_of_parallel_networks}) is smaller than "
                                   f"the Number of intended weights to keep ({sum_of_keep_weights})\n"
                                   f"please change the <keep_weights...> or <num_of_parallel_networks> parameter "
                                   f"in your config file ({args.config_file})")

    def init_agent(self):
        # examine the state space
        env_info_observation = env.reset(train_mode=self.env_train_mode)[brain_name]
        states_observation = env_info_observation.vector_observations
        self.state_size = states_observation.shape[1]
        action_size = brain.vector_action_space_size
        # print(f" init_agent: s_size: {self.state_size}\n a_size: {action_size}")  # s_size: 33;  a_size: 4
        # agent_ = NetworkFullyConnected(state_size=state_size, action_size=action_size).to(device)
        # optimizer = optim.Adam(agent_.parameters(), lr=1e-4)        # policy.parameters()

        if self.network_type == "NetworkFullyConnected":
            agent_ = NetworkFullyConnected(state_size=self.state_size, action_size=action_size).to(device)
        elif self.network_type == "NetworkOneHiddenLayer":
            agent_ = NetworkOneHiddenLayer(s_size=self.state_size, a_size=action_size).to(device)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"QNetwork\" or \"DoubleQNetwork\"")
        return agent_

    def update_weightslist(self):
        """ pick and reuse weights that delivered high rewards
        add some noise to the weights
        refill rest with random weights"""
        # <self.nextweightslist> was initialized at train()
        # print(f"weights_pre: {self.weightslist}")
        # idx: array with indexes that deliver rewards (rising rewards from start to end of list)
        idx = np.array(self.rewards_all_networks.argsort(axis=1))
        startidx = 0

        # keep weights (amount: <self.keep_weights_n_best_min>) with best min_reward
        # min_reward = self.rewards_all_networks[0]
        # example:
        # fill <nextweightslist> at position 0 and 1 (n=2) with weights that delivered the highest min_reward:
        # n = 2 = <self.keep_weights_n_best_min> (keep the weights with 2 best min_rewards)
        # weightslist_new[startidx:startidx+n, :] = weightslist[idx[:, -n:][0]]
        # weightslist_new[0:2, :] = weightslist[idx[: ,-2][0]
        # startidx=startidx+n
        if self.keep_weights_n_best_min > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_min, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_min:][0]]
        startidx = startidx + self.keep_weights_n_best_min

        # keep weights with best mean_reward
        # mean_reward = self.rewards_all_networks[1]
        if self.keep_weights_n_best_mean > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_mean, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_mean:][1]]
        startidx = startidx + self.keep_weights_n_best_mean

        # keep weights with best max_reward
        # max_reward = self.rewards_all_networks[2]
        if self.keep_weights_n_best_max > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_max, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_max:][2]]
        startidx = startidx + self.keep_weights_n_best_max

        # keep weights with best min_reward and add a small noise
        if self.keep_weights_n_best_min_small_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_min_small_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_min_small_change:][0]] \
                    + self.noise_scale_best_small * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_min_small_change

        # keep weights with best mean_reward and add a small noise
        if self.keep_weights_n_best_mean_small_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_mean_small_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_mean_small_change:][1]] \
                    + self.noise_scale_best_small * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_mean_small_change

        # keep weights with best max_reward and add a small noise
        if self.keep_weights_n_best_max_small_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_max_small_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_max_small_change:][2]] \
                    + self.noise_scale_best_small * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_max_small_change

        # keep weights with best min_reward and add a lot of noise
        if self.keep_weights_n_best_min_big_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_min_big_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_min_big_change:][0]] \
                    + self.noise_scale_best_big * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_min_big_change

        # keep weights with best mean_reward and add a lot of noise
        if self.keep_weights_n_best_mean_big_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_mean_big_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_mean_big_change:][1]] \
                    + self.noise_scale_best_big * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_mean_big_change

        # keep weights with best max_reward and add a lot of noise
        if self.keep_weights_n_best_max_big_change > 0:
            self.nextweightslist[startidx:startidx + self.keep_weights_n_best_max_big_change, :] = \
                    self.weightslist[idx[:, -self.keep_weights_n_best_max_big_change:][2]] \
                    + self.noise_scale_best_big * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_best_max_big_change

        # keep weights with worst min_reward and add a lot of noise
        self.nextweightslist[startidx:startidx + self.keep_weights_n_worst_min_big_change, :] = \
                self.weightslist[idx[:, :self.keep_weights_n_worst_min_big_change][0]] \
                + self.noise_scale_worst * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_worst_min_big_change

        # keep weights with worst mean_reward and add a lot of noise
        self.nextweightslist[startidx:startidx + self.keep_weights_n_worst_mean_big_change, :] = \
                self.weightslist[idx[:, :self.keep_weights_n_worst_mean_big_change][1]] \
                + self.noise_scale_worst * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_worst_mean_big_change

        # keep weights with worst max_reward and add a lot of noise
        self.nextweightslist[startidx:startidx + self.keep_weights_n_worst_max_big_change, :] = \
                self.weightslist[idx[:, :self.keep_weights_n_worst_max_big_change][2]] \
                + self.noise_scale_worst * np.random.rand(self.weights_dim)
        startidx = startidx + self.keep_weights_n_worst_max_big_change

        # fill the rest with random weights
        self.nextweightslist[startidx:, :] = self.sigma * np.random.rand(self.weights_dim)
        self.weightslist = self.nextweightslist.copy()
        # print(f"weights_post: {self.weightslist}")
        return None

    def train(self):
        self.weights_dim = agent.get_weights_dim()
        self.weightslist = self.sigma * np.random.randn(self.num_of_parallel_networks, agent.get_weights_dim())
        # self.weightslist = np.load(self.path_load + 'weights_' + self.load_indices + '.npy')
        self.nextweightslist = np.empty(shape=(self.num_of_parallel_networks, agent.get_weights_dim()))
        # print(
        #     f"weights: {self.weightslist}\nlenWeights: {len(self.weightslist)}\nweights_dim: {agent.get_weights_dim()}")
        saved = False
        time_new = time_start = datetime.datetime.now()
        for i in range(self.episodes_train):
            for j in range(self.num_of_parallel_networks):
                agent.set_weights(self.weightslist[j])
                env_utils.get_random_start_state()
                min_reward, mean_reward, max_reward = admin.get_rewards(trainmode=True)
                self.rewards_all_networks[0, j] = min_reward
                self.rewards_all_networks[1, j] = mean_reward
                self.rewards_all_networks[2, j] = max_reward
            # print(f"episode={i}\nrew_all_ep_0={self.rewards_all_networks.max(axis=1)[0]}\nrew_all_ep_1={self.rewards_all_networks.max(axis=1)[1]}\nrew_all_ep_2={self.rewards_all_networks.max(axis=1)[2]}")
            for k in range(3):
                self.rewards_all_episodes[k, i] = self.rewards_all_networks.max(axis=1)[k]
            if i >= self.consecutive_episodes_required:
                '''next 3 only for testing'''
                # rewards_deque = self.rewards_all_episodes[:, i - 100:i + 1]
                # rewards_deque_mean = self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)
                # rewards_deque_max = self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1).max()
                # print(f"ci check_indexing in train(): rewards_deque_shape={rewards_deque.shape()} | should be (3,100)")
                # print(f"ci rewards_deque_mean={rewards_deque_mean} | should be of shape (3))")
                # print(f"ci rewards_deque_max={rewards_deque_max} | should be one value")

                # if self.rewards_all_episodes[:,i-100:i+1].mean(axis=1).max() >= self.target_reward: # if either min or mean or max of Results reaches the goal value
                # if self.rewards_all_episodes[:,i-100:i+1].mean(axis=1)[0] >= self.target_reward:    # if min of Results reaches the goal value
                if self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)[
                    1] >= self.target_reward:  # if mean of Results reaches the goal value
                    print(f"target reward reached in episode: {i - self.consecutive_episodes_required}: "
                          f"mean_of_means_of_rewards={self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)[1]}")
                    # last_max_reward_positions = np.argmax(self.rewards_all_networks,
                    #                                       axis=1)  # np.argmax gives first max position (even if there are multiple max positions)
                    # print(f"ci last_max_reward_positions= {last_max_reward_positions}")
                    # print(
                    #     f"ci corresponding rewards={[self.rewards_all_networks[z] for z in last_max_reward_positions]}")
                    # max_reward_weights_min = self.weightslist[last_max_reward_positions[0]]
                    # max_reward_weights_mean = self.weightslist[last_max_reward_positions[1]]
                    # max_reward_weights_max = self.weightslist[last_max_reward_positions[2]]
                    if self.save_weights and not saved:
                        np.save(self.path_save + 'weights_s' + self.save_indices, self.weightslist)
                        np.save(self.path_save + 'scores_s' + self.save_indices, self.rewards_all_networks)
                        saved = True
                    break
            # print(f"rewards_all_nw: {self.rewards_all_networks}")
            # print(f"\n\nrewards_all_ep: {self.rewards_all_episodes}")
            self.update_weightslist()
            if (i + 1) % 25 == 0:
                time_old = time_new
                time_new = datetime.datetime.now()
                if i > 99:
                    print('\rMin_Score {}\tAverage_Score: {:.2f}\tMax_Score {}\tEpisode {}/{}\tTime since start: {}'
                          '\tdeltaTime: {}'.format(self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)[0],
                                                   self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)[1],
                                                   self.rewards_all_episodes[:, i - 100:i + 1].mean(axis=1)[2],
                                                   i+1, self.episodes_train, str(time_new-time_start).split('.')[0],
                                                   str(time_new-time_old).split('.')[0]), end="")
                else:
                    print('\rMin_Score - \tAverage_Score: - \tMax_Score - \tEpisode {}/{}\tTime since start: {}'
                          '\tdeltaTime: {}'.format(i + 1, self.episodes_train, str(time_new - time_start).split('.')[0],
                                                   str(time_new - time_old).split('.')[0]), end="")
        admin.plot_results()
        if self.save_weights:
            # last_max_reward_positions = np.argmax(self.rewards_all_networks,
            #                                       axis=1)  # np.argmax gives first max position (even if there are multiple max positions)
            # print(f"ci Ende: last_max_reward_positions= {last_max_reward_positions}")
            # # doesn't work yet: print(f"ci Ende: corresponding rewards={[self.rewards_all_networks[z] for z in last_max_reward_positions]}")
            # # max_reward_weights_min = self.weightslist[last_max_reward_positions[0]]
            # # max_reward_weights_mean = self.weightslist[last_max_reward_positions[1]]
            # # max_reward_weights_max = self.weightslist[last_max_reward_positions[2]]
            np.save(self.path_save + 'weights_g' + self.save_indices, self.weightslist)
            np.save(self.path_save + 'scores_g' + self.save_indices, self.rewards_all_networks)
        return None

    def test(self):
        self.weights_dim = agent.get_weights_dim()
        self.weightslist = np.load(self.path_load + 'weights_' + self.load_indices + '.npy')
        self.rewards_all_networks = np.load(self.path_load + 'scores_' + self.load_indices)
        # self.update_weightslist()
        agent.set_weights(self.weightslist[self.load_scores_version])
        # initialize
        # env = environment
        # agent = Agent()
        # choose mean_weights
        rewards_test = []
        rewards_deque = deque(maxlen=self.consecutive_episodes_required)
        means_of_means_of_sum_of_rewards = []
        for i in range(self.episodes_test):
            env_utils.states = env.reset(train_mode=self.env_train_mode)[brain_name].vector_observations
            env_utils.normalize_states()
            reward_min, reward_mean, reward_max = agent.get_rewards(env, trainmode=False)
            rewards_test.append(reward)
            if i >= self.consecutive_episodes_required:
                means_of_means_of_sum_of_rewards.append(np.mean(rewards_deque))
        agent.plot_results(rewards_test)
        return None

    # def test(self):
    #     """"""
    #     '''
    #     this Function may contain Code provided by Udacity Inc.
    #     '''
    #     agent.qnetwork_local.load_state_dict(torch.load(self.path_load + "checkpoint_" + self.load_indices + ".pth"))
    #     time_new = time_start = datetime.datetime.now()
    #     score = 0
    #     scores = []
    #     scores_window = deque(maxlen=100)   # last 100 scores
    #
    #     for i_episode in range(self.episodes_test):
    #         env_info = env.reset(train_mode=True)[brain_name]
    #         state, _, _ = get_infos(env_info)
    #         while True:
    #             action = agent.act(state, self.epsilon_test)
    #             env_info = env.step(action)[brain_name]
    #             next_state, reward, done = get_infos(env_info)
    #             score += reward
    #             state = next_state
    #             if done:
    #                 env.reset()
    #                 break
    #         scores_window.append(score)
    #         scores.append(score)
    #         if (i_episode + 1) % 25 == 0:
    #             time_old = time_new
    #             time_new = datetime.datetime.now()
    #             print('\rMin_Score {}\tAverage_Score: {:.2f}\tMax_Score {}\tEpisode {}/{}\tTime since start: {}'
    #                   '\tdeltaTime: {}'.format(np.min(scores_window), np.mean(scores_window), np.max(scores_window),
    #                                            i_episode, self.episodes_test-1, str(time_new-time_start).split('.')[0],
    #                                            str(time_new-time_old).split('.')[0]), end="")
    #         score = 0
    #     env.close()
    #     print("\n")
    #     plot_scores(scores)
    #     return None

    def get_rewards(self, trainmode=True):
        '''test_networks()'''
        # get trajectories and discount them --> new Rewards --> not necessary
        # --> 20 Robots per weight for n episodes --> get weights (raw)
        rewards_sum = np.zeros(self.number_of_agents)
        actions_list = []
        # if trainmode:
        #     for _ in range(self.number_of_random_actions):
        #         actions = np.clip(np.random.randn(self.number_of_agents, 4) / 4, a_min=-1, a_max=1)
        #         env_info_tr = env.step(actions)[brain_name]
        #     env_utils.states = env_info_tr.vector_observations  # get next state (for each agent)
        # else:
        #     env_utils.states = env.reset(train_mode=self.env_train_mode)[brain_name].vector_observations
        # env_utils.normalize_states()
        for _ in range(self.max_steps_per_training_episode):
            actions = agent(
                torch.from_numpy(env_utils.normalized_states).float().to(device)).squeeze().cpu().detach().numpy()
            actions_list.append(actions)
            '''use same action multiple times'''
            for i_same_act in range(self.num_of_same_act_repetition):
                env_info = env.step(actions)[brain_name]
                rewards_sum += np.array(env_info.rewards)
                # print(f"env_info.rewards: {env_info.rewards}")
                if env_info.local_done:  # if is_done .... from Udacity
                    break
            env_utils.states = env_info.vector_observations
            env_utils.normalize_states()
        # print(f"actions= {actions_list}")
        # print(f"actions_min={min(actions_list())}")
        # print(f"actions_max={max(actions_list())}")
        return rewards_sum.min(), rewards_sum.mean(), rewards_sum.max()

    def plot_results(self):
        '''
        this Function may contain Code provided by Udacity Inc.
        '''
        # create figure
        # fig = plt.figure()
        # if self.rewards_all_episodes.size()[0] == 3:

        ''' Training:
        max of min_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode
        max of mean_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode
        max of max_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode 
            Testing:
        same plots, but always from the same Network
        '''
        # print(f"plot_results() rewards_all_episodes={self.rewards_all_episodes}")
        # print(f"plot_results() rewards_all_episodes[0]={self.rewards_all_episodes[0]}")

        x_plot = np.arange(len(self.rewards_all_episodes[0]))
        numOfPlots = 3

        for plot_number in range(3):
            plt.subplot(numOfPlots, 1, plot_number + 1)
            plt.plot(x_plot, self.rewards_all_episodes[plot_number], '.-')
            # plt.title('A tale of 2 subplots')
            # plt.ylabel('Damped oscillation')


        # elif self.rewards_all_episodes.size()[0] == 1:
        #     ''' Test:
        #     tested Network: score: max of the 20 agents (of score in one Episode) over Episodes
        #     tested Network: score: mean of the 20 agents (of score in one Episode) over Episodes
        #     tested Network: score: min of the 20 agents (of score in one Episode) over Episodes '''
        #     x_plot = np.arange(len(self.rewards_all_episodes[0]))
        #     plt.subplot(1, 1, 1)
        #     plt.plot(x_plot, self.rewards_all_episodes[0], '.-')
        #
        #     # fig.add_subplot(212)
        #     # plt.plot(np.arange(len(epsilones)), epsilones)
        #     # plt.ylabel('epsilon')
        #     # plt.xlabel('Episode #')
        #     # fig.add_subplot(211)
        # else:
        #     print("plot_results(): rewards of wrong Dimension")

        # ''' Training:
        # max of min_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode
        # max of mean_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode
        # max of max_reward of all Network over Episodes (min_reward: min score of the 20 agents in one Episode
        #     Testing:
        # same plots, but always from the same Network
        # '''
        # x_plot = np.arange(len(self.rewards_all_episodes[0]))
        # print(f"doesent count up:  {self.rewards_all_episodes.size()[0]}")
        # for plot_number in range(self.rewards_all_episodes.size()[0]):
        #     plt.subplot(self.rewards_all_episodes.size()[0], 1, plot_number + 1)
        #     plt.plot(x_plot, self.rewards_all_episodes[plot_number], '.-')
        #     # plt.title('A tale of 2 subplots')
        #     # plt.ylabel('Damped oscillation')
        if self.save_plot:
            # save the plot
            plt.savefig(self.path_save + "graph_" + self.save_indices + ".png")
        if self.show_plot:
            # plot the scores
            plt.show()
        return None


if __name__ == "__main__":
    # Idea of parser: https://docs.python.org/2/howto/argparse.html
    parser = argparse.ArgumentParser(description='Interacting Agent')
    parser.add_argument('--train', type=str, default='True', help='True: train the agent; '
                                                                  'default=False: test the agent')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='Name of config_file in root of Continuous_Control')
    parser.add_argument('--getminmax', type=str, default='False',
                        help='True: get min and max Values of state; default=False: do nothing')
    args = parser.parse_args()

    # convert argparse arguments to bool since argparse doesn't treat booleans as expected:
    if args.train == 'True' or args.train == 'true' or args.train == 'TRUE':
        train = True
    elif args.train == 'False' or args.train == 'false' or args.train == 'FALSE':
        train = False
    else:
        raise MyAppLookupError('--train can only be True or False | default: False')
    if args.getminmax == 'True' or args.getminmax == 'true' or args.getminmax == 'TRUE':
        getminmax = True
    elif args.getminmax == 'False' or args.getminmax == 'false' or args.getminmax == 'FALSE':
        getminmax = False
    else:
        raise MyAppLookupError('--getminmax can only be True or False | default: False')

    # load config_file.json
    # Idea: https://commentjson.readthedocs.io/en/latest/
    with open(args.config_file, 'r') as f:
        config_data = commentjson.load(f)
    # initialize configuration
    admin = Administration(config_data)

    '''
    from here on this function may contain some Code provided by Udacity Inc.
    '''
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize Environment
    env = UnityEnvironment(file_name=admin.environment_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # initialize Agent
    agent = admin.init_agent()

    # initialize Environment Utilities
    env_utils = EnvUtils()

    # get min and max Values of state if selected
    if getminmax is True:
        env_utils.get_states_min_max_Values()

    # train or test the Agent
    if train is True:
        print(f"\nTrain the Network using config_file <{args.config_file}> on device <{device}> "
              f"with weights-save-index <{admin.save_indices}>")
        admin.train()
    else:
        print(f"\nTest the Network with fixed weights from <checkpoint_{admin.load_indices}.pth> "
              f"using config_file <{args.config_file}> on device <{device}>")
        admin.test()
    env.close()
