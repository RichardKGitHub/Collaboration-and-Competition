import numpy as np
import torch
from collections import deque

class Model:

class Agent:
    def __init__(self):
        self.number_of_agents=20
        self.weights = Agent.choose_random_weights(self, self.number_of_agents)
        self.rewards = Agent.test_networks()
    def choose_random_weights(self, number_of_networks):
        return weights_crw
    def test_networks(self, env, weights_tn, trainmode=True):
        # get trajectories and discount them --> new Rewards --> not necessary
        # --> 20 Robots per weight for n episodes --> get weights (raw)
        if trainmode==True:
            random actions

        env.step(action)
        calc mean_rewards_tn
        self.mean_rewards_tn=mean_rewards_tn
        return mean_rewards_tn
    def update_weights(self):
        Agent.calc_mean_rewards()
        --> sort weights
        build new weights using best, worst and random
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

class StateUtils:
    # def __init__(self):
    #     self.state =
    #     return None
    def setstates(self, states):
        self.states = states
    def get_min_max(self):
        return None
    def normalize_state(self):
        return None

def train(episodes=10, consecutive_episodes=100, target_reward=30):
    initialize
    env=environment
    agent=Agent()
    weights = agent.choose_random_weights()
    rewards_deque = deque(maxlen=consecutive_episodes)
    means_of_means_of_sum_of_rewards = []
    saved = False
    for i in range(episodes):
        mean_rewards = agent.test_networks(env, weights)                 # list of 50 for all weights
        rewards_deque.append(mean_rewards)
        agent.update_weights()
        if i >= consecutive_episodes:
            mean_of_means_of_sum_of_rewards= np.mean(rewards_deque)
            means_of_means_of_sum_of_rewards.append(mean_of_means_of_sum_of_rewards)
            if mean_of_means_of_sum_of_rewards >= target_reward and saved == False:
                print(f"task solved in episode {i-consecutive_episodes}: mean_of_means_of_rewards={mean_of_means_of_sum_of_rewards}")
                save_best_mean_weights
                saved = True
                # break
    save all_weights
    save best mean_weights
    agent.plot_results()
    return None

def test(episodes=10, consecutive_episodes=100):
    initialize
    env = environment
    agent = Agent()
    weight = load mean_weights
    rewards_test = []
    rewards_deque = deque(maxlen=consecutive_episodes)
    means_of_means_of_sum_of_rewards = []
    for i in range(episodes):
        reward = agent.test_networks(env, weight, trainmode=False)
        rewards_test.append(reward)
        if i >= consecutive_episodes:
            means_of_means_of_sum_of_rewards.append(np.mean(rewards_deque))
    agent.plot_results(rewards_test)

    return None
