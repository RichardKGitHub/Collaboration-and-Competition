#!/usr/bin/env python
# coding: utf-8

# # Continuous Control Project

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision.transforms.functional as TF    # used for Normalization
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import progressbar as pb
import numpy as np
from parallelEnv import parallelEnv
from unityagents import UnityEnvironment
from collections import deque

policy_name = 'PPO.policy'


# check device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

env = UnityEnvironment(file_name='/home/user2/Documents/github/udacity/DeepReinforcementLearning/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86_64')

# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
# states = env_info.vector_observations                  # get the current state (for each agent)
# scores = np.zeros(num_agents)                          # initialize the score (for each agent)
# count = 0
# while True:
#
#     actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#     actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#     print(f"actions: {actions}\nbrain_name={brain_name}") if count == 0 else None
#     count += 1
#     env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#     next_states = env_info.vector_observations         # get next state (for each agent)
#     rewards = env_info.rewards                         # get reward (for each agent)
#     dones = env_info.local_done                        # see if episode finished
#     scores += env_info.rewards                         # update the score (for each agent)
#     states = next_states                               # roll over states to next time step
#     if np.any(dones):                                  # exit loop if episode finished
#         break
# print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

# based on udacity pong exercise structure
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        # # Calculation of outputsize
        # 80x80x2 to outputsize x outputsize
        # outputsize = (inputsize - kernel_size + stride)/stride
        # (round up if not an integer)

        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)
        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size = 9 * 9 * 16

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()

        # # output = 20x20 here
        # self.conv = nn.Conv2d(2, 1, kernel_size=4, stride=4)
        # self.size=1*20*20
        #
        # # 1 fully connected layer
        # self.fc = nn.Linear(self.size, 1)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten the tensor
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))

        # x = F.relu(self.conv(x))
        # # flatten the tensor
        # x = x.view(-1,self.size)
        # return self.sig(self.fc(x))

class PolicyFullyConnected(nn.Module):
    """Actor (Policy) Model."""
    ''' 
    this class was provided by Udacity Inc.
    '''
    def __init__(self, state_size, action_size, seed=1234, fc1_units=21, fc2_units=10):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PolicyFullyConnected, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)    # self.sig = nn.Sigmoid()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)      # return self.sig(self.fc2(x))

    # from Udacity REINFORCE in Policy Gradient Methods
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


# class train:
#
#     def __init__(self):
#         super(train, self).__init__()

# from Udacity REINFORCE in Policy Gradient Methods
def reinforce(self, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    policy = PolicyFullyConnected().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores


# from Udacity pong example pong-PPO.py:
def train(env=env, policy_name='PPO.policy', device=device):

    print(f" state_size={state_size}; action_size={action_size}")
    policy = PolicyFullyConnected(state_size=state_size, action_size=action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    # training loop max iterations
    episode = 10
    n_agents = 32

    # widget bar to display progress
    # get_ipython().system('pip install progressbar')
    widget = ['training loop: ', pb.Percentage(), ' ',
              pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

    # # envs = parallelEnv('PongDeterministic-v4', n=n_agents, seed=1234)
    # envs = parallelEnv(env, n=n_agents, seed=1234)

    discount_rate = .99
    epsilon = 0.1
    beta = .01
    tmax = 320
    SGD_epoch = 4

    # keep track of progress
    mean_rewards = []
    all_total_rewards = []
    all_acrions = []
    for e in range(episode):

        # collect trajectories
        # old_probs, states_, actions_, rewards_ = train.collect_trajectories(envs, policy, tmax=tmax)
        old_probs, states_, actions_, rewards_ = collect_trajectories(env, policy, tmax=tmax)
        # print('old_probs={}\nstates_={}\nactions_={}\nrewards_={}'.format(old_probs, states_, actions_, rewards_))
        total_rewards = np.sum(rewards_, axis=0)

        # from clipped_surrogate (till rewards....)
        discount = 0.995
        discount = discount ** np.arange(len(rewards_))
        rewards = np.asarray(rewards_) * discount[:, np.newaxis]

        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:, np.newaxis]) / std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions_cs = torch.tensor(actions_, dtype=torch.int8, device=device)
        old_probs_cs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards_cs = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # gradient ascent step
        for _ in range(SGD_epoch):
            # L = -train.clipped_surrogate(policy, old_probs, states_, actions_, rewards_, epsilon=epsilon, beta=beta)
            L = -clipped_surrogate(policy, old_probs_cs, states_, actions_cs, rewards_cs, epsilon=epsilon, beta=beta)
            # print(f"L: {L}")
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= .999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta *= .995

        # get the average reward of the parallel environments
        mean_rewards.append(np.mean(total_rewards))
        all_total_rewards.append(total_rewards)
        all_acrions.append(actions_)

        # display some progress every 20 iterations
        if (e + 1) % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
            print(total_rewards)

        # update progress widget bar
        timer.update(e + 1)

    timer.finish()

    # save your policy!
    # torch.save(policy, policy_name)

    rewards = np.array(all_total_rewards)
    # print(f"rewards: {rewards}")
    # print(f"actions= {all_acrions}")
    x_plot = np.arange(len(rewards))
    colums = 4
    numOfPlots = n_agents
    for plot_number in range(numOfPlots):
        plt.subplot(numOfPlots / colums, colums, plot_number + 1)
        plt.plot(x_plot, rewards[:, plot_number], '.-')
        # plt.title('A tale of 2 subplots')
        # plt.ylabel('Damped oscillation')
    plt.show()

# from udacity pong exercise pong_utils.py
# collect trajectories for a parallelized parallelEnv object
# def collect_trajectories(envs, policy, tmax=200, nrand=5):
def collect_trajectories(env, policy, tmax=200, nrand=5):

    # number of parallel instances
    # n = len(envs.ps)
    n = 20

    # initialize returning lists and start the game!
    states_list = []
    rewards_list = []
    probs_list = []
    actions_list = []

    # envs.reset()
    #
    # # start all parallel agents
    # envs.step([1] * n)

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

    # perform nrand random steps to get a random starting point
    for _ in range(nrand):
        actions_1 = np.clip(np.random.randn(n, 4)/4, a_min=-1, a_max=1)
        # states_1, rewards_1, _, _ = envs.step(actions_1)
        # print(f"actions_1:{actions_1}\nbrain_name={brain_name}")
        env_info_ = env.step(actions_1)[brain_name]
        states_2 = env_info_.vector_observations  # get next state (for each agent)
        rewards_2 = env_info_.rewards  # get reward (for each agent)
        dones = env_info_.local_done  # see if episode finished
        # states_3, rewards_3, _, _ = envs.step(actions_1)
        # print(f"states_2={states_2}")

    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        # batch_input = preprocess_batch([fr1, fr2])

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        # probs = policy(state).squeeze().cpu().detach().numpy()

        #  Expected object of type torch.cuda.FloatTensor but found type torch.DoubleTensor for argument #4 'mat1'
        states_2 = torch.tensor(states_2)
        states_2 = states_2.float().to(device)
        # print(f"states_2={states_2}")
        actions_1 = policy(states_2).squeeze().cpu().detach().numpy()
        # actions_1 = policy(states_2).squeeze().detach().cpu().numpy()

        # print(f"actions_1 with states_2 input={actions_1}")

        '''here we do actions = probs --> use actions later on'''
        probs_1 = (actions_1 + 1) / 2
        # print(f"probs_1 in traject = {probs_1}") if t == 0 else None
        # action_1 = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        # probs = np.where(action == RIGHT, probs, 1.0 - probs)'

        '''use same action multiple times'''
        # advance the game (0=no action)
        # we take one action and skip game forward
        env_info_1 = env.step(actions_1)[brain_name]
        env_info_2 = env.step(actions_1)[brain_name]
        # states_3, rewards_3, is_done, _ = envs.step(actions_1)
        states_1 = env_info_1.vector_observations  # get next state (for each agent)
        rewards_1 = env_info_1.rewards  # get reward (for each agent)
        # states_2 = env_info_2.vector_observations  # get next state (for each agent)
        rewards_2 = env_info_2.rewards  # get reward (for each agent)
        is_done = env_info_2.local_done  # see if episode finished

        # print(f"rewards_1: {len(rewards_1)} rewards_2: {len(rewards_2)} ") if t == 0 else None
        rewards = np.array(rewards_1) + np.array(rewards_2) # + rewards_3
        # print(f"rewards: {rewards.shape}") if t == 0 else None

        # store the result
        states_list.append(states_1)
        rewards_list.append(rewards)
        probs_list.append(probs_1)
        actions_list.append(actions_1)

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if is_done:
            print("break with is_done") if t <= 50 else None
            break

    # return pi_theta, states, actions, rewards, probability
    # return prob_list, state_list, action_list, reward_list
    return probs_list, states_list, actions_list, rewards_list

def clipped_surrogate(policy, old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):

    # # convert states to policy (or probability)
    # # new_probs = train.states_to_prob(policy, states)
    # new_probs = states_to_prob(policy, states)
    # # new_probs = torch.where(actions == RIGHT, new_probs, 1.0 - new_probs)
    # # new_probs = actions
    # # ratio for clipping

    states = torch.tensor(states)
    states = states.float().to(device)
    # print(f"states_2={states_2}")
    # actions_cs_ = policy(states).squeeze().cpu().detach().numpy()
    # actions_cs_ = policy(states)
    # print(f"actions_CS={actions_cs_}")
    # new_probs_cs = (torch.tensor(actions_cs_, dtype=torch.float, device=device) + 1) / 2
    # print(f"new_probs_cs={new_probs_cs}")
    # ratio = new_probs_cs / old_probs

    actions_cs_ = (policy(states) + 1) / 2
    ratio = actions_cs_ / old_probs

    # clipped function
    clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    # print(f"ratio: {ratio}\nclip: {clip}\nrewards: {rewards}")
    # print(f"dimratio: {ratio.size()}\ndimclip: {clip.size()}\ndimrewards: {rewards.size()}")
    clipped_surrogate = torch.min(ratio * torch.tensor(rewards).view(1,20,1), clip * torch.tensor(rewards).view(1,20,1))
    # print(f"clipped_surrogate = {clipped_surrogate}")
    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    # print(f"new_probs={new_probs_cs}\nold_probs={old_probs}")

    # entropy = -(new_probs_cs * torch.log(old_probs + 1.e-10) + \
    #             (1.0 - new_probs_cs) * torch.log(1.0 - old_probs + 1.e-10))
    entropy = -(actions_cs_ * torch.log(old_probs + 1.e-10) + \
                (1.0 - actions_cs_) * torch.log(1.0 - old_probs + 1.e-10))
    # # print(f"entropy = {entropy}\nbeta = {beta}")
    # # this returns an average of all the entries of the tensor
    # # effective computing L_sur^clip / T
    # # averaged over time-step and number of trajectories
    # # this is desirable because we have normalized our rewards
    # c_L = clipped_surrogate + beta * entropy
    # # print(f"c_L={c_L}\nc_L.size()={c_L.size()}")
    # L=c_L.mean([-2]).view(4)
    # # L=c_L.mean()
    # # print(f"L={L}\nL.size()={L.size()}")
    # return L
    return torch.mean(clipped_surrogate + beta * entropy)

# from udacity pong exercise pong_utils.py
# convert states to probability, passing through the policy
def states_to_prob(policy, states):
    # states = torch.stack(torch.tensor(states))
    states=torch.tensor(states)
    # states = torch.stack(list(states))
    policy_input = states.view(-1, *states.shape[-3:])
    # policy_input= policy_input.
    print(f"policy_input: {policy_input}")
    policy_input = torch.tensor(policy_input)
    policy_input = policy_input.float().to(device)
    policy_ = policy(policy_input)
    print(f"policy_={policy_}")
    print(f"states.shape={states.shape}\npolicy_.shape={policy_.shape}")
    policy_ = policy_.view(states.shape[:-3])
    print(f"policy_view={policy_}")
    return policy(policy_input).view(states.shape[:-3])

# # from ShangtongZhang Examples:
# # A2C
# def a2c_feature(**kwargs):
#     generate_tag(kwargs)
#     kwargs.setdefault('log_level', 0)
#     config = Config()
#     config.merge(kwargs)
#
#     config.num_workers = 5
#     config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
#     config.eval_env = Task(config.game)
#     config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
#     config.network_fn = lambda: CategoricalActorCriticNet(
#         config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
#     config.discount = 0.99
#     config.use_gae = True
#     config.gae_tau = 0.95
#     config.entropy_weight = 0.01
#     config.rollout_length = 5
#     config.gradient_clip = 0.5
#     run_steps(A2CAgent(config))

train()

env.close()
