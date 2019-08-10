## Project Details
This Project was completed in the course of the Deep Reinforcement Learning Nanodegree Program from Udacity Inc. \
In this Project a 20 Agents have to follow a target location
- Action space: 4
- state space: 33
- A reward of +0.1 is provided while hand in target location
- one Episode takes XXX frames
- the environment is solved when the agent gets an average score of +30 over 100 consecutive episodes

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



## Instructions - Run the Script
In your shell run:
```
python3.6 Continuous_Control.py
```
For specification of interaction-mode and -config-file run:
```
python3.6 Continuous_Control.py --train True --config_file config.json
```
Info: \
The UnityEnvironment is expected at `" XXX "`. \
This can be changed in the `config.json` file if necessary.
