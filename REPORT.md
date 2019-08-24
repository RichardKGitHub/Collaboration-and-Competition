[//]: # (Image References)

[image1]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/scores_35.png "training scores DDPG_1"
[image2]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/noise_35.png "loss DDPG_1"
[image3]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/scores_35.png "noise DDPG_1"
[image4]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/mean_score_36.png "test scores DDPG_1 consecutive mean"
[image5]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/scores_31.png "training scores DDPG_2"
[image6]: https://github.com/RichardKGitHub/Collaboration-and-Competition/blob/col_01/archive/mean_score_37.png "test scores DDPG_2 consecutive mean"

## Learning Algorithm
The Project was solved by a ddpg algorithm ( Deep Deterministic Policy Gradient) \
This algorithm utilises an actor to determine the next action and a critic to provide a Q-Value for a given state-action-combination

- four neural networks:
  - local_actor: network to determine action (in response to the state from the environment) (only network needed during test)
  - target_actor: network to determine future actions during update process of local_critic
  - local_critic: network to determine loss for the update of local_actor
  - target_critic: network to determine future Q-Value for the calculation of the "discounted Reward" during update of local_critic
- after each environmental step a replay buffer gets filled with the 20 <state, action, reward, next state> information's from the 20 agents
- after each environmental step the weights of the local networks get updated using an batch_size of 256 randomly picked from the replay buffer
- after each environmental step the target networks are getting updated via soft_update:
  ```
  target_param_new = tau * copy(local_param) + (1.0 - tau) * target_param
  ```
#### Network architecture
The Project was solved by two different network architectures
###### Network 1 (DDPG_1)
- actor:
  - input layer: 24 Neurones for the state-space of 24
  - first hidden layer: 128 Neurones   |   activation function: Rectified Linear Unit (ReLU)
  - second hidden layer: 128 Neurones   |   activation function: Rectified Linear Unit (ReLU)
  - output layer: 2 Neurones for the action-space of 2   |   activation function: tanh (to reach action values between -1 and 1
- critic:
  - input layer one: 24 Neurones for the state-space of 24
  - input layer two: 2 Neurones for the action-space of 2
  - first "hidden" layer: combination of 128 Neurones that are connected to the input layer 1 (leakyReLu) and the input layer 2
  - second hidden layer: 128 Neurones   |   activation function: leaky Rectified Linear Unit (leakyReLU)
  - third hidden layer: 9 Neurones   |   activation function: leaky Rectified Linear Unit (leakyReLU)
  - output layer: 1 Neuron corresponding to one Q-Value
###### Network 2 (DDPG_2)
- actor:
  - input layer: 24 Neurones for the state-space of 24
  - first hidden layer: 128 Neurones   |   activation function: Rectified Linear Unit (ReLU)
  - second hidden layer: 128 Neurones   |   activation function: Rectified Linear Unit (ReLU)
  - output layer: 2 Neurones for the action-space of 2   |   activation function: tanh (to reach action values between -1 and 1
- critic:
  - input layer 1: 24 Neurones for the state-space of 24
  - input layer 2: 2 Neurones for the action-space of 2
  - first "hidden" layer: combination of 128 Neurones that are connected to the input layer 1 (ReLu) and the input layer 2
  - second hidden layer: 128 Neurones   |   activation function: Rectified Linear Unit (ReLU)
  - output layer: 1 Neuron corresponding to one Q-Value
#### Hyperparameters
- both algorithms use the same parameters:
  - maximal Number of episodes `if --train==True` (network gets trained): 4000 (Network 1) / 5000 (Network 2)
  - Number of episodes `if --train==False` (network gets tested): 400
  - epsilon: 1.0                    (epsilon is used in an different approach: 1.0 means: always add Noise to action)
  - epsilon during test mode: 0.0   (epsilon is used in an different approach: 0.0 means: no Noise added to action)
  - replay buffer size: 2e6
  - batch size": 512
  - discount factor gamma: 0.9
  - tau: 1e-3 (for soft update of target parameters)
  - learning_rate: 1e-3
## Plot of Rewards
The network architecture of DDPG_1 outperforms DDPG_2 in respect to the learning speed (it needs half as much episodes to reach the target value) and the mean_score over 100 consecutive episodes during training. 
This is quite remarkable, considering that the only difference lays in the critics architecture with an additional layer of 9 neurons before the 
output neuron and a different activation function for the critic network (leaky_ReLu instead of ReLu).
#### DDPG_1
- task solved in episode 1171 (reaching a mean score over 100 consecutive episodes of 0.502475 in episode 1271)
  - graph max_scores shows the score of the best agent per episode and min_scores the score of the worst agent
![training scores DDPG_1][image1]
  - loss during training (mean over all steps in one episode):
![loss DDPG_1][image2]
  - noise for the action during training(mean over all steps in one episode):
![noise DDPG_1][image3]
- the test was performed over 400 episodes with the weights that where saved at episode 1271 of training. No Noise was added to the actions during the test
  - Min_consecutive_Score: 1,72
  - Max_consecutive_Score: 1,99
  - graph shows the mean Score over the last 100 episodes:
![test scores DDPG_1 consecutive mean][image4]
#### DDPG_2
- task solved in episode 2466 (reaching a mean score over 100 consecutive episodes of 0.501485 in episode 2566)
  - graph max_scores shows the score of the best agent per episode and min_scores the score of the worst agent
![training scores DDPG_2][image5]
- the test was performed over 400 episodes with the weights that where saved at episode 2568 of training. No Noise was added to the actions during the test
  - Min_consecutive_Score: 0,77
  - Max_consecutive_Score: 1,16
  - graph shows the mean Score over the last 100 episodes:
![test scores DDPG_2 consecutive mean][image6]
## Ideas for Future Work
- In the next step, the parameters for both networks and algorithms could be further adjusted to see if the task can be solved in fewer episodes.
- The implementation of a Proximal Policy Optimisation algorithm (PPO) is an additional next step in order to compare different learning strategy's.
