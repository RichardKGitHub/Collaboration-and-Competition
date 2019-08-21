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
from torch.distributions import Categorical
from unityagents import UnityEnvironment
from collections import deque

# class NetworkFullyConnected(nn.Module):
#     """Actor (Policy) Model."""
#     '''
#     this class was provided by Udacity Inc.
#     '''
#
#     def __init__(self, state_size, action_size, seed=1203, fc1_units=21, fc2_units=10):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_units (int): Number of nodes in first hidden layer
#             fc2_units (int): Number of nodes in second hidden layer
#         """
#         super(NetworkFullyConnected, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, action_size)  # self.sig = nn.Sigmoid()
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)  # return self.sig(self.fc2(x))
#
#     # from Udacity REINFORCE in Policy Gradient Methods
#     def act(self, state):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         probs = self.forward(state).cpu()
#         m = Categorical(probs)
#         action = m.sample()
#         return action.item(), m.log_prob(action)
#
# class NetworkOneHiddenLayer(nn.Module):
#     '''from CEM.py  Lesson2 Nr.9 Workspace'''
#
#     def __init__(self, s_size, a_size, h_size=8):
#         # 00: h_size=16
#         # 01: h_size=16
#         # 02: h_size=8
#
#         super(NetworkOneHiddenLayer, self).__init__()
#         # state, hidden layer, action sizes
#         self.s_size = s_size
#         self.h_size = h_size
#         self.a_size = a_size
#         # define layers
#         self.fc1 = nn.Linear(self.s_size, self.h_size)
#         self.fc2 = nn.Linear(self.h_size, self.a_size)
#
#     def set_weights(self, weights):
#         s_size = self.s_size
#         h_size = self.h_size
#         a_size = self.a_size
#         # separate the weights for each layer
#         fc1_end = (s_size * h_size) + h_size
#         # print(f"weights.shape: {weights.shape}")
#         # print(f"weights[:s_size * h_size].shape: {weights[:s_size * h_size].shape}")
#         fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
#         fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
#         fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
#         fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
#         # set the weights for each layer
#         self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
#         self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
#         self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
#         self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))
#
#     def get_weights_dim(self):
#         return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.tanh(self.fc2(x))
#         x = 2*x - 1
#         return x.cpu().data
#
#     # def evaluate(self, weights, gamma=1.0, max_t=5000):
#     #     self.set_weights(weights)
#     #     episode_return = 0.0
#     #     state = self.env.reset()
#     #     for t in range(max_t):
#     #         state = torch.from_numpy(state).float().to(device)
#     #         action = self.forward(state)
#     #         state, reward, done, _ = self.env.step(action)
#     #         episode_return += reward * math.pow(gamma, t)
#     #         if done:
#     #             break
#     #     return episode_return


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
        self.states = np.empty(shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))
        self.next_states = np.empty(shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))
        self.states_normalized = np.empty(
            shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))
        self.next_states_normalized = np.empty(
            shape=(admin.num_of_parallel_networks, admin.number_of_agents, admin.state_size))

    def set_states(self, states, next_states):
        self.states = states
        self.next_states = next_states
        self.normalize_states()
        return None

    def get_random_start_state(self):
        # actions_list = []   # only for testing
        # state_list = []     # only for testing
        env_info_tr = env.reset(train_mode=admin.env_train_mode)[brain_name]
        for _ in range(admin.number_of_random_actions):
            actions = np.clip(np.random.randn(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
            # print(f"random_actions={actions}")
            env_info_tr = env.step(actions)[brain_name]
            # self.states = env_info_tr.vector_observations        # for test
            # self.normalize_states()                             #for test
            # state_list.append(self.states_normalized)                       # for test
            # actions_list.append(actions)                                    # for test
        # print(f"actions={np.array(actions_list)}")
        # print(f"actions_min={actions_list.min()}")
        # print(f"actions_mean={actions_list.mean()}")
        # print(f"actions_max={actions_list.max()}")
        # print(f"state_list1= {state_list}")
        # print(f"state1={np.array(state_list).min()}")
        # print(f"state_mean1={np.array(state_list).mean()}")
        # print(f"state_max1={np.array(state_list).max()}")
        self.states = env_info_tr.vector_observations
        self.normalize_states()
        return env_info_tr

    def get_states_min_max_Values(self):
        # perform some random steps to get a random starting point
        _ = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = []
        actions_ = []
        for _ in range(admin.episodes_test):
            for _ in range(400):
                actions = np.clip(np.random.rand(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
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
            self.states_normalized = np.clip(fAnstieg * self.states + fOffset, -1, 1)
            self.next_states_normalized = np.clip(fAnstieg * self.next_states + fOffset, -1, 1)
        else:
            self.states_normalized = fAnstieg * self.states + fOffset
            self.next_states_normalized = fAnstieg * self.next_states + fOffset
        # print(f"aInterpolatedData: {self.normalized_states}")
        # self.states_normalized = self.states.copy()
        # self.next_states_normalized = self.next_states.copy()
        # print(f"states_norm: {self.states_normalized}\tstates: {self.states}\nnsn: {self.next_states_normalized}\tns: {self.next_states}")
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
        # fc_units = 256
        super(Actor1, self).__init__()
        # self.seed = torch.manual_seed(seed)
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
        # x = 2 * x - 1
        # x = F.relu(self.fc3(x))
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
        # fcs1_units = 256, fc2_units = 256, fc3_units = 128
        super(Critic1, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units+action_size, fc3_units)
        # self.fc4 = nn.Linear(fc3_units, fc4_units)
        # self.fc5 = nn.Linear(fc4_units, 1)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        # self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        xs = F.leaky_relu(self.fc2(xs))
        # xs = 2 * xs - 1
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc3(x))
        # x = F.leaky_relu(self.fc4(x))
        # # x = torch.tanh(self.fc5(x))
        # x = self.fc5(x)
        x = self.fc4(x)
        return x


class Actor2(nn.Module):
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
        # fc_units = 256
        super(Actor2, self).__init__()
        # self.seed = torch.manual_seed(seed)
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
        # x = 2 * x - 1
        # x = F.relu(self.fc3(x))
        return x


class Critic2(nn.Module):
    """Critic (Value) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fcs1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        # fcs1_units = 256, fc2_units = 256, fc3_units = 128
        super(Critic2, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        # self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # self.fc4 = nn.Linear(fc3_units, fc4_units)
        # self.fc5 = nn.Linear(fc4_units, 1)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        # self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        # xs = F.leaky_relu(self.fc2(xs))
        # xs = 2 * xs - 1
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc4(x))
        # # x = torch.tanh(self.fc5(x))
        # x = self.fc5(x)
        x = self.fc3(x)
        return x


class Actor3(nn.Module):
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
        # fc_units = 256
        super(Actor3, self).__init__()
        # self.seed = torch.manual_seed(seed)
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
        # x = 2 * x - 1
        # x = F.relu(self.fc3(x))
        return x


class Critic3(nn.Module):
    """Critic (Value) Model."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed, fcs1_units, fc2_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        # fcs1_units = 256, fc2_units = 256, fc3_units = 128
        super(Critic3, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fcs1_units)
        # self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        # self.fc4 = nn.Linear(fc3_units, fc4_units)
        # self.fc5 = nn.Linear(fc4_units, 1)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        # self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        # self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        # xs = F.leaky_relu(self.fc2(xs))
        # xs = 2 * xs - 1
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc4(x))
        # # x = torch.tanh(self.fc5(x))
        # x = self.fc5(x)
        x = self.fc3(x)
        return x


class Actor4(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor4, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_bn:
            x = self.fc1(self.bn1(state))
        else:
            x = self.fc1(state)

        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn3(x)
        # print(f"forward actor bn: {self.use_bn}")
        return F.tanh(self.fc3(x))


class Critic4(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, use_bn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic4, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_bn = use_bn
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.fc1(state)
        xs = F.relu(x)
        if self.use_bn:
            x = self.bn1(x)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.relu(x)
        if self.use_bn:
            x = self.bn2(x)
        # print(f"forward critic bn: {self.use_bn}")
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    '''
    this class contains some changes but was mainly provided by Udacity Inc.
    '''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state_noise = copy.copy(self.mu)
        # print(f"resetNoise: {self.state_noise}")

    def sample(self):
        # """Update internal state and return it as a noise sample."""
        # x = self.state
        # # dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random()*2 -1 for i in range(len(x))])
        # # dx = self.theta * (self.mu - x) * self.sigma * np.array([random.random() * 2 - 1 for i in range(len(x))])
        # dx = self.sigma * np.array([random.random() * 2 - 1 for i in range(len(x))])
        # # print(f"noise x: {self.state}\tnoise mu-x {self.mu - x}\tdx: {dx}")
        # # self.state = x +dx
        # self.state = dx
        """Update internal state and return it as a noise sample."""
        x = self.state_noise
        # print(f"Noise: {self.state_noise}\ttheta: {self.theta}\tmu: {self.mu}\tsigma: {self.sigma}")
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state_noise = x + dx
        # print(f"noise: {self.state_noise}")
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
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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
            # print(f"add_memory inputs: s:{state}\na:{action}\nr:{reward}\nns:{next_state}\nd:{is_done}")
            # self.memory.add(state, action, reward, next_state, is_done)
            e = self.experience(state, action, reward, next_state, is_done)
            self.memory.append(e)
        return None

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
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
        self.load_scores_version = config_data_interact['load_scores_version']
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
        # self.actor_fcs_units = config_data_interact['actor_fc_units']
        self.critic_fcs1_units = config_data_interact['critic_fcs1_units']
        self.critic_fcs2_units = config_data_interact['critic_fcs2_units']
        self.critic_fcs3_units = config_data_interact['critic_fcs3_units']
        # self.critic_fcs4_units = config_data_interact['critic_fcs4_units']
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
        # self.update_target_every = config_data_interact['update_target_every']
        self.lInterpolParam = config_data_interact['lInterpolParam']
        self.number_of_agents = config_data_interact['number_of_agents']  # 20
        self.agents_duplication_factor = config_data_interact['agents_duplication_factor']
        self.number_of_random_actions = config_data_interact['number_of_random_actions']
        self.max_steps_per_training_episode = config_data_interact['max_steps_per_training_episode']
        # self.num_of_same_act_repetition = config_data_interact['num_of_same_act_repetition']
        self.env_train_mode = config_data_interact['env_train_mode']
        self.environment_path = config_data_interact['environment_path']
        if train is True:
            self.scores_all_episodes_and_NW = np.empty(shape=(self.num_of_parallel_networks, 3, self.episodes_train))
        else:
            self.scores_all_episodes_and_NW = np.empty(shape=(self.num_of_parallel_networks, 3, self.episodes_test))
        self.rewards_all_networks = np.empty(shape=(3, self.num_of_parallel_networks))
        self.weightslist = np.empty(shape=(self.num_of_parallel_networks, 612))
        self.nextweightslist = np.empty(shape=(self.num_of_parallel_networks, 612))
        self.state_size = 33
        self.action_size = 4
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
        self.i_update = 0
        self.q_loss_loss_one_episode = np.zeros(shape=(5, int(self.max_steps_per_training_episode/self.learn_every*self.consecutive_learning_steps)))
        self.q_loss_loss = np.zeros(shape=(self.num_of_parallel_networks, 5, self.episodes_train))
        self.epsilon_sigma_noise = np.zeros(shape=(self.num_of_parallel_networks, 3, self.episodes_train))
        self.sigma_noiseMean = np.zeros(shape=(self.num_of_parallel_networks, 2, self.max_steps_per_training_episode))
        # print(f"init: max_steps_per_training_episode {self.max_steps_per_training_episode}\nsigma_noiseMean{self.sigma_noiseMean}")
        self.step_counter = 0
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
        elif self.network_type == "DDPG_3":
            self.actor_local = Actor3(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor3(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic3(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units).to(device)
            self.critic_target = Critic3(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_4":
            self.actor_local = Actor4(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor4(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic4(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                       self.critic_fcs2_units).to(device)
            self.critic_target = Critic4(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"DDPG_1\" to \"DDPG_4\"")
        self.noise = OUNoise(self.action_size, self.random_seed, theta=self.noise_theta, sigma=self.noise_sigma)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size_admin, self.batch_size_admin, self.random_seed)
        self.soft_update_started = False

    def train_ddpg(self):
        # self.weights_dim = agent.get_weights_dim()
        # self.weightslist = self.sigma * np.random.randn(self.num_of_parallel_networks, agent.get_weights_dim())
        # # self.weightslist = np.load(self.path_load + 'weights_' + self.load_indices + '.npy')
        # self.nextweightslist = np.empty(shape=(self.num_of_parallel_networks, agent.get_weights_dim()))
        # print(
        #     f"weights: {self.weightslist}\nlenWeights: {len(self.weightslist)}\nweights_dim: {agent.get_weights_dim()}")

        self.load_parameter()

        self.soft_update_started=False
        print("start training")
        self.epsilon = self.epsilon_start
        saved = False
        time_new = time_start = datetime.datetime.now()
        for i in range(self.episodes_train):
            for j in range(self.num_of_parallel_networks):
                # agent.set_weights(self.weightslist[j])
                # env_utils.get_random_start_state()
                self.noise.reset()
                min_reward, mean_reward, max_reward = admin.get_rewards_ddpg(trainmode=self.env_train_mode)
                self.scores_all_episodes_and_NW[j, 0, i] = min_reward
                self.scores_all_episodes_and_NW[j, 1, i] = mean_reward
                self.scores_all_episodes_and_NW[j, 2, i] = max_reward
                mean_of_q_loss_loss = self.q_loss_loss_one_episode.mean(axis=1)
                # print(f"mean_of_ll: {mean_of_q_loss_loss}")
                self.q_loss_loss[j, 0, i] = mean_of_q_loss_loss[0]
                self.q_loss_loss[j, 1, i] = mean_of_q_loss_loss[1]
                self.q_loss_loss[j, 2, i] = mean_of_q_loss_loss[2]
                self.q_loss_loss[j, 3, i] = mean_of_q_loss_loss[3]
                self.q_loss_loss[j, 4, i] = mean_of_q_loss_loss[4]
                self.epsilon_sigma_noise[j, 0, i] = self.epsilon
                mean_of_sigma_noise = self.sigma_noiseMean[j].mean(axis=1)
                # print(f"mean_of_sigma_noise: {self.sigma_noiseMean}")
                self.epsilon_sigma_noise[j, 1, i] = mean_of_sigma_noise[0]
                self.epsilon_sigma_noise[j, 2, i] = mean_of_sigma_noise[1]
                # self.epsilon_sigma_noise[j, 0, i] = self.epsilon
            # print(f"g_loss: {self.q_loss_loss}")
            if i >= self.consecutive_episodes_required:
                # rewards_deque_episodes = self.rewards_all_episodes[:, i - 100:i + 1]
                for m in range(self.num_of_parallel_networks):
                    # if mean of Results reaches the goal value
                    # print(f"hello Value: {self.scores_all_episodes_and_NW[m, :, i - 100:i + 1].mean(axis=1)[1]}")
                    if self.scores_all_episodes_and_NW[m, :, i - 100:i + 1].mean(axis=1)[1] >= self.target_reward:
                        print(f"\ntarget reward reached by Network No. {m} in episode: "
                              f"{i - self.consecutive_episodes_required}: mean_of_means_of_rewards="
                              f"{self.scores_all_episodes_and_NW[m][:, i - 100:i + 1].mean(axis=1)[1]}\n")
                        if self.save_weights and not saved:
                            # np.save(self.path_save + 'weights_s' + self.save_indices, self.weightslist)
                            self.save_parameter('s_')
                            saved = True
                        break
            # self.update_weightslist()
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)
            if (i + 1) % 1 == 0:
                time_old = time_new
                time_new = datetime.datetime.now()
                if i > 99:
                    print('\nscores: mean over last 100 Episodes | last Episode: min: {} | {}\tmean: {:.2f} | {}\t'
                          'max: {} | {}\tEpisode {}/{}\tTime since start: {}\tdeltaTime: '
                          '{}'.format(self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[0], min_reward,
                                      self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[1], mean_reward,
                                      self.scores_all_episodes_and_NW[0][:, i - 100:i + 1].mean(axis=1)[2], max_reward,
                                      i+1, self.episodes_train, str(time_new-time_start).split('.')[0],
                                      str(time_new-time_old).split('.')[0]), end="")
                else:
                    print('\nscores last episode: min_Score {} \tAverage_Score: {} \tMax_Score {} \tEpisode {}/{}\t'
                          'Time since start: {}\tdeltaTime: {}'.format(min_reward, mean_reward, max_reward, i + 1,
                                                                        self.episodes_train,
                                                                        str(time_new - time_start).split('.')[0],
                                                                        str(time_new - time_old).split('.')[0]), end="")
            if self.save_weights:
                # save your policy!
                self.save_parameter('g_')
                # np.save(self.path_save + 'weights_g' + self.save_indices, self.weightslist)
        admin.plot_results()
        return None

    def test(self):
        # load policy if needed
        self.load_parameter()
        self.epsilon = self.episodes_test

        min_reward, mean_reward, max_reward = admin.get_rewards_ddpg(trainmode=False)
        print(f"test: reward: {mean_reward}")

        # self.weights_dim = agent.get_weights_dim()
        # self.weightslist = np.load(self.path_load + 'weights_' + self.load_indices + '.npy')

        # self.rewards_all_networks = np.load(self.path_load + 'scores_' + self.load_indices)
        # self.update_weightslist()
        # agent.set_weights(self.weightslist[self.load_scores_version])
        # initialize
        # env = environment
        # agent = Agent()
        # choose mean_weights
        rewards_test = []
        rewards_deque = deque(maxlen=self.consecutive_episodes_required)
        means_of_means_of_sum_of_rewards = []

        # for i in range(self.episodes_test):
        #     env_utils.states = env.reset(train_mode=self.env_train_mode)[brain_name].vector_observations
        #     env_utils.normalize_states()
        #     reward_min, reward_mean, reward_max = agent.get_rewards(env, trainmode=False)
        #     rewards_test.append(reward)
        #     if i >= self.consecutive_episodes_required:
        #         means_of_means_of_sum_of_rewards.append(np.mean(rewards_deque))
        # agent.plot_results(rewards_test)
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

    # def test_2(self):
    #     '''
    #     this function contains some changes but was mainly provided by Udacity Inc.
    #     '''
    #     self.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    #     self.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    #
    #     state = env.reset()
    #     self.reset()
    #     while True:
    #         action = self.act(state)
    #         env.render()
    #         next_state, reward, done, _ = env.step(action)
    #         state = next_state
    #         if done:
    #             break
    #
    #     env.close()

    def get_rewards_ddpg(self, trainmode=True):
        '''test_networks()'''
        # get trajectories and discount them --> new Rewards --> not necessary
        # --> 20 Robots per weight for n episodes --> get weights (raw)
        score_one_episode = np.zeros(self.number_of_agents)
        actions_list = []   #   only to check behavior
        # state_list = []     # only for testing
        self.i_update = 0   # for q and loss documentation and soft_update
        self.sigma_noiseMean = np.zeros(shape=(self.num_of_parallel_networks, 2, self.max_steps_per_training_episode))
        env_info = env_utils.get_random_start_state()
        test = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                False, False, False, True, False]
        for step in range(self.max_steps_per_training_episode):
            self.step_counter = step    # for sigma and Noise documentation
            actions = self.act()
            # actions = agent(
            #     torch.from_numpy(env_utils.normalized_states).float().to(device)).squeeze().cpu().detach().numpy()
            actions_list.append(actions)        # only to test actionspace


            '''use same action multiple times'''
            # for i_same_act in range(self.num_of_same_act_repetition):
            env_info = env.step(actions)[brain_name]
            score_one_episode += np.array(env_info.rewards)
            # print(f"i_same_act should always be 0: {i_same_act}")
            # print(f"at step {step}\tis_done: {env_info.local_done}")
            # if env_info.local_done:  # if is_done .... from Udacity
            #     print(f"first break at step {step}")
            #     break
            if np.any(env_info.local_done):
                break
            env_utils.next_states = env_info.vector_observations
            env_utils.normalize_states()
            # state_list.append(env_utils.next_states_normalized)
            self.memory.add(actions, env_info)
            # learn every <self.learn_every> step
            if step % self.learn_every == 0:
                for loerning_step in range(self.consecutive_learning_steps):
                    self.step()

            # if step % self.learn_every == 0:
            #     self.step()
            # print(f"bbbbbbbbbbbbbbbbbnextstate: {env_utils.next_states_normalized}")
            env_utils.states = env_utils.next_states.copy()
            env_utils.states_normalized = env_utils.next_states_normalized.copy()
            # print(f"dddddddddddddddddddstate {env_utils.states_normalized}")
        # print(f"state_list2= {state_list}")
        # print(f"state2={np.array(state_list).min()}")
        # print(f"state_mean2={np.array(state_list).mean()}")
        # print(f"state_max2={np.array(state_list).max()}")


        # print(f"actions_list= {actions_list}")
        # print(f"actions_min={np.array(actions_list).min()}")
        # print(f"actions_mean={np.array(actions_list).mean()}")
        # print(f"actions_max={np.array(actions_list).max()}")
        # print(f"actions= {actions}")
        # print(f"actions_min={np.array(actions).min()}")
        # print(f"actions_mean={np.array(actions).mean()}")
        # print(f"actions_max={np.array(actions).max()}")
        return score_one_episode.min(), score_one_episode.mean(), score_one_episode.max()

    def plot_results(self):
        '''
        this Function may contain Code provided by Udacity Inc.
        '''
        x_plot = np.arange(self.scores_all_episodes_and_NW.shape[-1])
        '''plot scores'''
        list_of_names = ['min_scores', 'mean_scores', 'max_scores']
        for row in range(self.num_of_parallel_networks):
            for column in range(3):
                plt.subplot(self.num_of_parallel_networks, 3, (column+1) * (row+1))
                plt.plot(x_plot, self.scores_all_episodes_and_NW[self.num_of_parallel_networks-1, column, :], '-')
                plt.title(list_of_names[column])
                plt.xlabel('episodes')
                plt.ylabel('scores')
        if self.save_plot:
            # save the plot
            plt.savefig(self.path_save + "scores_" + self.save_indices + ".png")
        if self.show_plot:
            # plot the scores
            plt.show()
        '''plot reward, Q_target, Q_expected, critic_loss and actor_loss'''
        x2_plot = np.arange(self.q_loss_loss.shape[-1])
        """ !!! in learn: only last network gets saved into first Position [0]"""
        list_of_names = ['reward_lastNW', 'Q_target_lastNW', 'Q_expected_lastNW', 'critic_loss_lastNW',
                         'actor_loss_lastNW']
        for row in range(self.num_of_parallel_networks):
            for column in range(5):
                plt.subplot(self.num_of_parallel_networks, 5, (column+1) * (row+1))
                plt.plot(x2_plot, self.q_loss_loss[self.num_of_parallel_networks-1, column, :], '-')
                plt.title(list_of_names[column])
                # plt.ylabel('episodes')
                # plt.ylabel('scores')
        if self.save_plot:
            # save the plot
            plt.savefig(self.path_save + "losses_" + self.save_indices + ".png")
        if self.show_plot:
            # plot the scores
            plt.show()
        # '''plot noise_sigma and epsilon'''
        # list_of_names = ['epsilon', 'sigma_noise', 'max_noise']
        # x3_plot = np.arange(self.epsilon_sigma_noise.shape[-1])
        # for row in range(self.num_of_parallel_networks):
        #     for column in range(3):
        #         plt.subplot(self.num_of_parallel_networks, 3, (column+1) * (row+1))
        #         plt.plot(x3_plot, self.epsilon_sigma_noise[self.num_of_parallel_networks-1, column, :], '-')
        #         plt.title(list_of_names[column])
        #         # plt.ylabel('episodes')
        #         # plt.ylabel('scores')
        # if self.save_plot:
        #     # save the plot
        #     plt.savefig(self.path_save + "noise_" + self.save_indices + ".png")
        # if self.show_plot:
        #     # plot the scores
        #     plt.show()
        return None

    def init_agent(self):
        # examine the state space
        env_info_observation = env.reset(train_mode=self.env_train_mode)[brain_name]
        states_observation = env_info_observation.vector_observations
        self.state_size = states_observation.shape[1]
        self.action_size = brain.vector_action_space_size
        # print(f" init_agent: s_size: {self.state_size}\n a_size: {action_size}")  # s_size: 33;  a_size: 4
        # agent_ = NetworkFullyConnected(state_size=state_size, action_size=action_size).to(device)
        # optimizer = optim.Adam(agent_.parameters(), lr=1e-4)        # policy.parameters()
        '''
        up from here this Function may contain Code provided by Udacity Inc.
        '''
        if self.network_type == "DDPG_1":
            # Actor Network (w/ Target Network)
            self.actor_local = Actor1(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor1(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

            # Critic Network (w/ Target Network)
            self.critic_local = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_target = Critic1(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units, self.critic_fcs3_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_2":
            # Actor Network (w/ Target Network)
            self.actor_local = Actor2(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor2(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

            # Critic Network (w/ Target Network)

            self.critic_local = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units).to(device)
            self.critic_target = Critic2(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_3":
            # Actor Network (w/ Target Network)
            self.actor_local = Actor3(self.state_size, self.action_size, self.random_seed,
                                     self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor3(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)

            # Critic Network (w/ Target Network)
            self.critic_local = Critic3(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units).to(device)
            self.critic_target = Critic3(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                    self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        elif self.network_type == "DDPG_4":
            self.actor_local = Actor4(self.state_size, self.action_size, self.random_seed,
                                      self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_target = Actor4(self.state_size, self.action_size, self.random_seed,
                                       self.actor_fcs1_units, self.actor_fcs2_units).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.learning_rate_actor)
            self.critic_local = Critic4(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                        self.critic_fcs2_units).to(device)
            self.critic_target = Critic4(self.state_size, self.action_size, self.random_seed, self.critic_fcs1_units,
                                         self.critic_fcs2_units).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.learning_rate_critic,
                                               weight_decay=self.weight_decay)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"DDPG_1\" to \"DDPG_4\"")
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
        if len(self.memory) > max(self.batch_size_admin, 255):      # wait till a buffer of at least 256
            # print(f"size memory: {len(self.memory)}")
            experiences = self.memory.sample()
            # print(f"experiences: {experiences}")
            self.learn(experiences)
            self.i_update += 1
        return None

    def act(self):
        """Returns actions for given state as per current policy."""
        '''
        this function contains some changes but was mainly provided by Udacity Inc.
        '''
        state = torch.from_numpy(env_utils.states_normalized).float().to(device)
        # action by network
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        self.sigma_noiseMean[0, 0, self.step_counter] = self.noise_sigma
        if self.add_noise:
            if random.random() < self.epsilon:
                # noise = np.random.randn(self.number_of_agents, self.action_size) * self.noise_sigma
                # print(f"no_NOISE_action = {action}")
                noise = self.noise.sample()
                action += noise
                # print(f"no_clip_action = {action}")
                # print(f"noise_mean {noise.mean()}")
                self.sigma_noiseMean[0, 1, self.step_counter] = noise.max()
            else:
                self.sigma_noiseMean[0, 1, self.step_counter] = 0
        else:
            self.sigma_noiseMean[0, 1, self.step_counter] = 0
        # print(f"noise_sigma {self.sigma_noiseMean}")
        return np.clip(action, -1, 1)

    def actV1(self, add_noise=False):
        """Returns actions for given state as per current policy."""
        '''
        this function contains some changes but was mainly provided by Udacity Inc.
        '''
        state = torch.from_numpy(env_utils.states_normalized).float().to(device)
        if random.random() > self.epsilon:
            # action by network
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            action = np.clip(action, -1, 1)
        else:
            # random action
            action = np.clip(np.random.randn(admin.number_of_agents, 4) / 4, a_min=-1, a_max=1)
        if add_noise:
            # print(f"noise: {(2*(np.random.randn(self.number_of_agents, self.action_size))-1)*self.noise_sigma}\taction_0: {action}")
            # # action += self.noise.sample()
            # action += (2*(np.random.randn(self.number_of_agents, self.action_size))-1)*self.noise_sigma
            # print(f"noise: {np.random.randn(self.number_of_agents, self.action_size)* self.noise_sigma}\taction_0: {action}")
            # action += self.noise.sample()
            # print(f"noise= {np.random.randn(self.number_of_agents, self.action_size) * self.noise_sigma}")
            action += np.random.randn(self.number_of_agents, self.action_size) * self.noise_sigma
            # print(f"action in act: {action}")
            # print(f"action_1: {action}")
        return np.clip(action, -1, 1)

    def reset(self):
        '''
        this function was provided by Udacity Inc.
        '''
        self.noise.reset()

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
        # print(f"states_in_learn: {states}")
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        # print(f"actions: {actions}\trewards: {rewards}\tnext_states: {next_states}\tdones: {dones}")

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Q_targets = rewards
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # print(f"q_exp {Q_expected}")
        # critic_loss = F.mse_loss(Q_expected, Q_targets, reduction='none')
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # print(f"fcritic_loss {critic_loss}")
        # print(f"critic_loss: {critic_loss}")
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # critic_loss.backward(torch.Tensor([1,10]))
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)

        '''suggestet clipping'''
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # print(f"actor_loss {actor_loss}")
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update_started = True
        # save reward Q_target Q_expected critic_loss and actor_loss values for plot #
        # print(f"Q_ex: {Q_expected}\tcr: {critic_loss}\tac: {actor_loss}")
        value_q_loss_loss = Q_expected.detach().cpu().numpy()
        # print(f"i_update: {self.i_update} value_q: {value_q_loss_loss}")
        # print(f"rewards. {rewards}\ntarget. {Q_targets}")
        # print(f"rewards in learn: {rewards}")
        # print(f"critic_loss: {critic_loss}\t actor_loss: {actor_loss}")
        self.q_loss_loss_one_episode[0, self.i_update] = rewards[0]
        self.q_loss_loss_one_episode[1, self.i_update] = Q_targets[0].detach().cpu().numpy()
        self.q_loss_loss_one_episode[2, self.i_update] = Q_expected[0].detach().cpu().numpy()
        self.q_loss_loss_one_episode[3, self.i_update] = critic_loss
        self.q_loss_loss_one_episode[4, self.i_update] = actor_loss
        # self.q_loss_loss_one_episode[0, 0, self.i_update] = rewards
        # self.q_loss_loss_one_episode[0, 1, self.i_update] = Q_targets.detach().cpu().numpy()
        # self.q_loss_loss_one_episode[0, 2, self.i_update] = Q_expected.detach().cpu().numpy()
        # self.q_loss_loss_one_episode[0, 3, self.i_update] = critic_loss
        # self.q_loss_loss_one_episode[0, 4, self.i_update] = actor_loss
        # print(f"q_loss_ep {self.q_loss_loss_one_episode}")

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        # if self.i_update != 0:
        # print(f"sustarted: {self.soft_update_started}")
        if self.soft_update_started:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
                # print(f'soft Tau: {self.tau}')
        else:
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(local_param.data)
                # print('copy instead softupdate')

    def save_parameter(self, affix):
        # policy_name = self.path_save + 'policy_' + affix + self.save_indices + '_actor_local.pt'
        # torch.save(self.actor_local.parameters(), policy_name)
        torch.save(self.actor_local.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_actor_local.pt')
        torch.save(self.actor_target.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_actor_target.pt')
        torch.save(self.critic_local.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_critic_local.pt')
        torch.save(self.critic_target.state_dict(), self.path_save + 'policy_' + affix + self.save_indices + '_critic_target.pt')
        # self.soft_update(self.critic_local, self.critic_target)
        # self.soft_update(self.actor_local, self.actor_target)
        # for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        #     target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        return None

    def load_parameter(self):
        # torch.save(model.state_dict(), PATH)
        # Load:
        #
        # device = torch.device('cpu')
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH, map_location=device))

        # self.actor_local = torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_local.pt')
        # self.actor_target = torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_target.pt')
        # self.critic_local = torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_local.pt')
        # self.critic_target = torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_target.pt')
        self.actor_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_local.pt'))
        self.actor_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_target.pt'))
        self.critic_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_local.pt'))
        self.critic_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_target.pt'))
        # self.actor_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_local.pt'))
        # self.actor_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_actor_target.pt',
        #                                             map_location=device))
        # self.critic_local.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_local.pt',
        #                                             map_location=device))
        # self.critic_target.load_state_dict(torch.load(self.path_load + 'policy_' + self.load_indices + '_critic_target.pt',
        #                                             map_location=device))

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

    # get min and max Values of state if selected
    if getminmax is True:
        env_utils.get_states_min_max_Values()

    # train or test the Agent
    if train is True:
        print(f"\nTrain the Network using config_file <{args.config_file}> on device <{device}> "
              f"with weights-save-index <{admin.save_indices}>")
        admin.train_ddpg()
    else:
        print(f"\nTest the Network with fixed weights from <checkpoint_{admin.load_indices}.pth> "
              f"using config_file <{args.config_file}> on device <{device}>")
        # admin.env_train_mode = False
        admin.test()
    env.close()

