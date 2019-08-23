## Project Details
This Project was completed in the course of the Deep Reinforcement Learning Nanodegree Program from Udacity Inc. \
In this Project 2 Agents play tennis together
- Action space: 2
- state space: 24
- A reward of +0.1 is provided while hand in target location
- one Episode takes maximum 1000 steps
- the environment is solved when the agent gets an average score of +30 over 100 consecutive episodes: 
  - After each episode, the rewards of each agent are summarized (without discounting)
  - The mean of the 20 resulting values is the score of one episode
## Getting Started - dependencies

#### Python version
- python3.6
#### Packages
- Install the required pip packages:
  ```
  pip install -r requirements.txt
  ```

- Only if your hardware supports it: install pytorch_gpu (otherwise skip it since torch will be installed with the environment anyway)  
  ```
  conda install pytorch-gpu
  ```
#### Environment
- Install gym 
  - [gym](https://github.com/openai/gym) 
  - follow the given instructions to set up gym (instructions can be found in README.md at the root of the repository)
  - make `gym` a Sources Root of your Project
- The environment for this project is included in the following Git-repository
  - [Git-repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
  - follow the given instructions to set up the Environment (instructions can be found in `README.md` at the root of the repository)
  - make the included `python` folder a Sources Root of your Project
- Insert the below provided Unity environment into the `p2_continuous-control/` folder of your `deep-reinforcement-learning/` folder from the previous step and unzip (or decompress) the file
  - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
## Instructions - Run the Script
In your shell run:
```
python3.6 Tennis_Project.py
```
For specification of interaction-mode and -config-file run:
```
python3.6 Tennis_Project.py --train False --config_file config.json
```
Info: \
The UnityEnvironment is expected at `"environment_path":"/data/Tennis_Linux_NoVis/Tennis"`. \
This can be changed in the `config.json` file if necessary.
