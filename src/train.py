import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import os
from stable_baselines3 import PPO # load PPO (Proximal Policy Optimization) Algorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
# GrayScale the environment
env = GrayScaleObservation(env,keep_dim=True)
# Wrap inside a DummyVecEnv to support vectorized environments
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env,n_stack=4, channels_order="last")

class TrainingAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(self.save_path + f"/new_model_{self.n_calls}")
        return True
    
CHECKPOINT_DIR = "train/"
LOG_DIR = "tensorboard_logs/"

# setups the callback function to save the model every 500000 steps
callback = TrainingAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0000003, n_steps=2048)
# Train the agent for 5M steps meaning 5M frames the model will be trained on
model.learn(total_timesteps=5000000, callback=callback)
