import random

from core import State, Transition, Experience, Episode
import gym
from gym import Env
from utils import str_key, set_dict, get_dict
from utils import epsilon_greedy_pi, epsilon_greedy_policy
from utils import greedy_policy, learning_curve
from approximator import Approximator
import random

class Agent(object):
    '''
    Attributes:
        env: environment
        obs_space = observation_space
        action_space = action_space
    Methods:
        policy: the input is state and the output is action
        learning_method: how to build Q table
    '''
    def __init__(self, env=None, capacity=10000):
        self.env = env
        self.obs_space = env.observation_space if env is not None else None
        self.action_space = env.action_space if env is not None else None
        self.States = [i for i in range(self.obs_space.n)]
        self.Actions = [i for i in range(self.action_space.n)]
        self.Q = {}
        self.experience = Experience(capacity=capacity)
        self.state = None  # current state

    def policy(self, A, s=None, Q=None, epsilon=None):
        return random.sample(self.A, k=1)[0]

    def perform_policy(self, s, epsilon=0.05):
        Q = self.Q
        action = self.policy(self, s, Q, epsilon)
        

