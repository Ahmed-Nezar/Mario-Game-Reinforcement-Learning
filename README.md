# Mario-Game-Reinforcement-Learning

## Introduction
This project is a reinforcement learning project that uses the PPO algorithm to train an agent to play the game Super Mario Bros. The game is played using the gym-super-mario-bros environment. The agent is trained using the stable-baselines3 library.
The agent was trained for 5 million steps and was able to achieve a Total score of 21650.

### Mario Game after training the agent:

![](./Others/Mario-Reinforcment-Learning.gif)

## Training Parameters

- Total Timesteps: 5,000,000
- Learning Rate: 0.0000003
- number of steps per iteration: 2048

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To prepare the environment, run the following notebook:
- prepare_environment.ipynb

To train the agent, run the following notebook:
- train.ipynb

To test the agent, run the following notebook:
- test.ipynb

## Results
The agent was able to learn to play the game and achieve a Total score of 21650.
