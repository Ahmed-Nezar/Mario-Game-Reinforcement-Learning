import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt

SIMPLE_MOVEMENT # This is the list of possible actions that our character can take

env = gym_super_mario_bros.make('SuperMarioBros-v0')
actions_before_wrapping = env.action_space # This is the number of possible actions that our character can take berfore wrapping
env = JoypadSpace(env,SIMPLE_MOVEMENT)

print("Number of actions before wrapping: ", actions_before_wrapping)
print("Number of actions after wrapping: ", env.action_space)

print("Observation space: ", env.observation_space.shape)

print("Action List: ", SIMPLE_MOVEMENT)

done = True
for step in range(5): # remember to change this to 1000000
    if done:
        # starts the game or resets it if it is over
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample()) # random action
    env.render() # render the game to the screen
env.close() # closes the game window

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
env.reset()

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env,SIMPLE_MOVEMENT)
initial_state = env.reset()
# GrayScale the environment
env = GrayScaleObservation(env,keep_dim=True)
initial_state_grayscale = env.reset()
plt.subplot(1,2,1)
plt.imshow(initial_state)
plt.title("RGB Observation")
plt.subplot(1,2,2)
plt.imshow(initial_state_grayscale)
plt.title("Grayscale Observation")
plt.show()

print("Shape of initial state: ", initial_state.shape)
print("Shape of grayscale state: ", initial_state_grayscale.shape)

# Wrap inside a DummyVecEnv to support vectorized environments
env = DummyVecEnv([lambda: env])

initial_state_grayscale_wrapped = env.reset()
# compare the shapes
print("Shape of initial state: ", initial_state.shape)
print("Shape of initial state grayscale: ", initial_state_grayscale.shape)
print("Shape of initial state grayscale wrapped: ", initial_state_grayscale_wrapped.shape)

env = VecFrameStack(env,n_stack=4, channels_order="last")

initial_state_grayscale_wrapped_stacked = env.reset()
# compare the shapes
print("Shape of initial state: ", initial_state.shape)
print("Shape of initial state grayscale: ", initial_state_grayscale.shape)
print("Shape of initial state grayscale wrapped: ", initial_state_grayscale_wrapped.shape)
print("Shape of initial state grayscale wrapped stacked: ", initial_state_grayscale_wrapped_stacked.shape)

def Visualizing_Frames(initial_state, initial_state_grayscale, initial_state_grayscale_wrapped, initial_state_grayscale_wrapped_stacked):
    plt.figure(figsize=(15,15))
    plt.subplot(1,4,1)
    plt.imshow(initial_state)
    plt.title("RGB Observation")
    plt.subplot(1,4,2)
    plt.imshow(initial_state_grayscale)
    plt.title("Grayscale Observation")
    plt.subplot(1,4,3)
    plt.imshow(initial_state_grayscale_wrapped[0])
    plt.title("GrayScale Wrapped")
    plt.subplot(1,4,4)
    plt.imshow(initial_state_grayscale_wrapped_stacked[0])
    plt.title("GrayScale Wrapped Stacked")
    plt.show()
    
Visualizing_Frames(initial_state, initial_state_grayscale, initial_state_grayscale_wrapped, initial_state_grayscale_wrapped_stacked)

# each step we take the stacked frames change & thus we will have new state
for i in range(3):
    initial_state_grayscale_wrapped_stacked,_,__,___ = env.step([env.action_space.sample()])
    # Visualizing all frames 
    Visualizing_Frames(initial_state, initial_state_grayscale, initial_state_grayscale_wrapped, initial_state_grayscale_wrapped_stacked)
