import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import commentjson
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

class NetworkOneHiddenLayer(nn.Module):
    '''from CEM.py  Lesson2 Nr.9 Workspace'''
    def __init__(self, s_size, a_size, h_size=16):
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

class Agent:
    def __init__(self):
        # self.number_of_agents=20
        # self.number_of_random_actions=5
        self.weights = Agent.choose_random_weights(self, self.number_of_agents)
        self.rewards = Agent.get_rewards()


class EnvUtils:
    def __init__(self):
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[brain_name]
        return None
    def setstates(self, states):
        self.states = states
    def get_min_max(self):
        return None
    def normalize_state(self):
        return None

class MyAppLookupError(LookupError):
    """raise this when there's a lookup error for my app"""
    # source of this class:
    # https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python/24065533#24065533

class Interaction:
    """defines looped interactions of the Agent (use case and training)"""

    def __init__(self, config_data_interact):
        self.load_indices = config_data_interact['load_indices']
        self.save_indices = config_data_interact['save_indices']
        # self.path_load = config_data_interact['path_load']
        # self.path_save = config_data_interact['path_save']
        # self.save_weights = config_data_interact['save_weights']
        # self.save_plot = config_data_interact['save_plot']
        # self.show_plot = config_data_interact['show_plot']
        self.network_type = config_data_interact['network_type']
        self.num_of_parallel_networks = config_data_interact['num_of_parallel_networks']
        self.episodes_train = config_data_interact['episodes_train']
        self.episodes_test = config_data_interact['episodes_test']
        self.target_reward = config_data_interact['target_reward']
        self.consecutive_episodes_required = config_data_interact['consecutive_episodes_required']
        # self.epsilon_start = config_data_interact['epsilon_start']
        # self.epsilon_end = config_data_interact['epsilon_end']
        # self.epsilon_decay = config_data_interact['epsilon_decay']
        # self.epsilon_test = config_data_interact['epsilon_test']
        # self.buffer_size = config_data_interact['buffer_size']
        # self.batch_size = config_data_interact['batch_size']
        # self.gamma = config_data_interact['gamma']
        # self.tau = config_data_interact['tau']
        self.sigma = config_data_interact['sigma']
        # self.learning_rate = config_data_interact['learning_rate']
        # self.update_target_every = config_data_interact['update_target_every']
        self.number_of_agents = config_data_interact['number_of_agents']
        self.env_train_mode = config_data_interact['env_train_mode']
        self.environment_path = config_data_interact['environment_path']

    def init_agent(self):
        # examine the state space
        env_info_observation = env.reset(train_mode=self.env_train_mode)[brain_name]
        states_observation = env_info_observation.vector_observations
        state_size = states_observation.shape[1]
        action_size = brain.vector_action_space_size

        # agent_ = NetworkFullyConnected(state_size=state_size, action_size=action_size).to(device)
        # optimizer = optim.Adam(agent_.parameters(), lr=1e-4)        # policy.parameters()

        if self.network_type == "NetworkFullyConnected":
            agent_ = NetworkFullyConnected(state_size=state_size, action_size=action_size).to(device)
        elif self.network_type == "NetworkOneHiddenLayer":
            agent_ = NetworkOneHiddenLayer(s_size=state_size, a_size=action_size).to(device)
        else:
            raise MyAppLookupError(f"No valid network_type specified | given: \"{self.network_type}\" | expected: "
                                   f"\"QNetwork\" or \"DoubleQNetwork\"")
        return agent_

    def train(self):
        print(f"train-fct\nagent: {agent}")
        for _ in range(50):
            actions = np.clip(np.random.randn(self.number_of_agents, 4) / 4, a_min=-1, a_max=1)

            env_info_tr = env.step(actions)[brain_name]
        states = env_info_tr.vector_observations
        # print(f"states: {states}")
        # print(f"\n\n\nrewards: {env_info_tr.rewards}")
        # initialize
        # env = environment
        # agent = Agent()
        weights = self.sigma * np.random.randn(agent.get_weights_dim())
        # print(f"weights: {weights}\nlenWeights: {len(weights)}")
        weights_pop = [weights + (self.sigma * np.random.randn(agent.get_weights_dim())) for i in range(self.num_of_parallel_networks)]
        # print(f"weights_pop: {weights_pop}\nlenWeights: {len(weights_pop)}")
        rewards = np.array([config.get_rewards(weights) for weights in weights_pop])
            '''to do ....'''
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]

        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
        # weights = agent.choose_random_weights()
        rewards_deque = deque(maxlen=self.consecutive_episodes_required)
        means_of_means_of_sum_of_rewards = []
        saved = False
        for i in range(self.episodes_train):
            min_reward, mean_reward, max_reward = agent.get_rewards(env, weights)  # list of 50 for all weights
            rewards_deque.append(min_reward, mean_reward, max_reward)
            agent.update_weights()
            if i >= self.consecutive_episodes_required:
                mean_of_means_of_sum_of_rewards = np.mean(rewards_deque)
                means_of_means_of_sum_of_rewards.append(mean_of_means_of_sum_of_rewards)
                if mean_of_means_of_sum_of_rewards >= self.target_reward and saved == False:
                    print(
                        f"task solved in episode {i - self.consecutive_episodes_required}: mean_of_means_of_rewards={mean_of_means_of_sum_of_rewards}")
                    # save_best_mean_weights
                    saved = True
                    # break
        # save
        # all_weights
        # save
        # best
        # mean_weights
        agent.plot_results()
        return None

    # def train(self):
    #     """"""
    #     '''
    #     this Function may contain Code provided by Udacity Inc.
    #     '''
    #     time_new = time_start = datetime.datetime.now()
    #     score = 0
    #     scores = []
    #     epsilones = []
    #     solved = False
    #     scores_window = deque(maxlen=100)   # last 100 scores
    #     epsilon = self.epsilon_start
    #     for i_episode in range(self.episodes_train):
    #         env_info = env.reset(train_mode=True)[brain_name]
    #         state, _, _ = get_infos(env_info)
    #         while True:
    #             action = agent.act(state, epsilon)
    #             env_info = env.step(action)[brain_name]
    #             next_state, reward, done = get_infos(env_info)
    #             agent.step(state, action, reward, next_state, done)
    #             score += reward
    #             state = next_state
    #             if done:                    # if done = True
    #                 env.reset()
    #                 break
    #         scores_window.append(score)
    #         scores.append(score)
    #         epsilones.append(epsilon)
    #         if (i_episode + 1) % 25 == 0:
    #             time_old = time_new
    #             time_new = datetime.datetime.now()
    #             print('\rMin_Score {}\tAverage_Score: {:.2f}\tMax_Score {}\tEpisode {}/{}\tTime since start: {}'
    #                   '\tdeltaTime: {}'.format(np.min(scores_window), np.mean(scores_window), np.max(scores_window),
    #                                            i_episode, self.episodes_train-1, str(time_new-time_start).split('.')[0],
    #                                            str(time_new-time_old).split('.')[0]), end="")
    #         if np.mean(scores_window) >= self.end_training_score and solved is False:
    #             print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
    #                                                                    i_episode - 100, np.mean(scores_window)),
    #                                                                    end="\n\n")
    #             if self.save_weights:
    #                 torch.save(agent.qnetwork_local.state_dict(), self.path_save + "checkpoint_s_" +
    #                            self.save_indices + ".pth")
    #             solved = True
    #             # break
    #         epsilon = max(epsilon * self.epsilon_decay, self.epsilon_end)
    #         score = 0
    #     if self.save_weights:
    #         torch.save(agent.qnetwork_local.state_dict(), self.path_save + "checkpoint_g_" + self.save_indices + ".pth")
    #     env.close()
    #     print(f'\n')
    #     plot_scores(scores, epsilones)
    #     return None

    def test(self, episodes=10,):
        # initialize
        # env = environment
        # agent = Agent()
        weight = load()
        # choose mean_weights
        rewards_test = []
        rewards_deque = deque(maxlen=self.consecutive_episodes_required)
        means_of_means_of_sum_of_rewards = []
        for i in range(episodes):
            reward = agent.get_rewards(env, weight, trainmode=False)
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

    def choose_random_weights(self, number_of_networks):
        '''to_do'''
        weights_crw=0
        return weights_crw
    def get_rewards(self, env=env, policy=agent, max_steps=200, trainmode=True, n_same_act=1):
        '''test_networds()'''
        # get trajectories and discount them --> new Rewards --> not necessary
        # --> 20 Robots per weight for n episodes --> get weights (raw)
        rewards_sum = np.zeros(self.number_of_agents)
        if trainmode==True:
            for _ in range(self.number_of_random_actions):
                actions = np.clip(np.random.randn(self.number_of_agents, 4) / 4, a_min=-1, a_max=1)
                env_info_tr = env.step(actions)[brain_name]
            states = env_info_tr.vector_observations  # get next state (for each agent)
        else:
            states = env.reset(train_mode=True)[brain_name].vector_observations
        for _ in range(max_steps):
            actions = policy(states).squeeze().cpu().detach().numpy()
            '''use same action multiple times'''
            for i_same_act in range(n_same_act):
                env_info = env.step(actions)[brain_name]
                rewards_sum += np.array(env_info.rewards)
                if env_info.local_done.any():               # if is_done .... from Udacity
                    break
            states = env_info.vector_observations
        return rewards_sum.min(), rewards_sum.mean(), rewards_sum.max()

    def update_weights(self):
        Agent.calc_mean_rewards()
        # --> sort weights
        # build new weights using best, worst and random
        return None

    def plot_results(self, rewards):
        if rewards.size()[0]==3:
            ''' Training:
            5 best Networks: reward: max of the 20 agents (of Sum in one Episode) over Episodes
            5 best Networks: reward: mean of the 20 agents (of Sum in one Episode) over Episodes
            5 best Networks: reward: min of the 20 agents (of Sum in one Episode) over Episodes '''
        elif  rewards.size()[0]==1:
            ''' Test:
            tested Network: reward: max of the 20 agents (of Sum in one Episode) over Episodes
            tested Network: reward: mean of the 20 agents (of Sum in one Episode) over Episodes
            tested Network: reward: min of the 20 agents (of Sum in one Episode) over Episodes '''
        else:
            print("plot_results(): rewards of wrong Dimension")
        return None


if __name__ == "__main__":
    # Idea of parser: https://docs.python.org/2/howto/argparse.html
    parser = argparse.ArgumentParser(description='Interacting Agent')
    parser.add_argument('--train', type=str, default='True', help='True: train the agent; '
                                                                   'default=False: test the agent')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='Name of config_file in root of Navigation')
    args = parser.parse_args()

    # convert to bool since argparse doesn't treat booleans as expected:
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
    # initialize configuration
    config = Interaction(config_data)

    '''
    from here on this function may contain some Code provided by Udacity Inc.
    '''
    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize Environment
    env = UnityEnvironment(file_name=config.environment_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # initialize Agent
    agent = config.init_agent()

    # train or test the Agent
    if train is True:
        print(f"\nTrain the Network using config_file <{args.config_file}> on device <{device}> "
              f"with weights-save-index <{config.save_indices}>")
        config.train()
    else:
        print(f"\nTest the Network with fixed weights from <checkpoint_{config.load_indices}.pth> "
              f"using config_file <{args.config_file}> on device <{device}>")
        config.test()
    env.close()
