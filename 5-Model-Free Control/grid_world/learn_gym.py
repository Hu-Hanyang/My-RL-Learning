# test windy_grid_world
import gym
from gym import Env
from gridworld import WindyGridWorld
from core import Agent
from agents import SarsaAgent, SarsaLambdaAgent, QAgent
from utils import learning_curve

# Sarsa doesn't work for Q of get_dict() is 'float' rather than 'dictionary'
# env = WindyGridWorld()
# agent = SarsaAgent(env, capacity=10000)
#
# statistics = agent.learning(gamma = 1.0,
#                             epsilon = 1,
#                             decaying_epsilon = True,
#                             alpha = 0.5,
#                             max_episode_num = 800,
#                             display = False)

# Sarsa
env = WindyGridWorld()
agent = SarsaAgent(env, capacity=10000)

print(agent.obs_space)
# data = agent.learning(max_episode_num = 180, display = False)
# statistics = agent.learning(gamma = 0.9, epsilon = 0.1, decaying_epsilon = True, alpha = 0.5, max_episode_num = 800, display = False )

# agent.learning_method(epsilon=0.01, display=True)

# learning_curve(statistics, x_index=2, y1_index=1, y2_index=None)

env.reset()
env.render()
env.close()