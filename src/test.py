import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO # load PPO (Proximal Policy Optimization) Algorithm

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
env = GrayScaleObservation(env,keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env,n_stack=4, channels_order="last")

model = PPO.load("train/new_model_3500000.zip", env=env)

state = env.reset()
# loop through the game
while True:
    action, _= model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()

