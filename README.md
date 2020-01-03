# deep_rl_project

This is the repository of a deep reinforcement learning project of the Lyon University.

There are two parts:
* Cartpole: The point was to master the cartpole environment from the gym library to make sure we had a working implementation and a good understanding of the DQN algorithm
* Atari: The point of the second part was to master the breakout environment by using more advanced methods such as Dueling, Double Q, etc.

To execute the code you first need to load the requirements, run: pip3 install -r requirements.txt

Then, if you are in cartpole or atari you can execute the following:
* python3 main.py: launches a new learning process with the parameters defined in config.py
* python3 run_trained_model.py: launches a pre-trained model to visualize its performance, parameters are defined in the file
* python3 display_log_plot.py: used to visualize a log from a learning phase as a plot, parameters are defined in the file
