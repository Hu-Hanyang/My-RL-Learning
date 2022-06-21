from play21 import Player, Dealer, Arena
from utils import str_key, set_dict, get_dict
from utils import epsilon_greedy_policy
import math


class MC_Player(Player):
    def __init__(self, name = "", A = None, display = False):
        super(MC_Player, self).__init__(name, A, display)
        self.Q = {}
        self.Nsa = {}  # (s,a)出现的次数
        self.total_learning_times = 0
        self.policy = self.epsilon_greedy_policy
        self.learning_method = self.learn_Q

    def learn_Q(self, episode, r):  # 从每个episode中学习Q(s,a)
        for s, a in  episode:
            nsa = get_dict(self.Nsa, s, a)
            set_dict(self.Nsa, nsa+1, s, a)
            q = get_dict(self.Q, s, a)
            set_dict(self.Q, q+(r-q)/(nsa+1), s, a)
        self.total_learning_times += 1

    def reset_memory(self):
        self.Q.clear()
        self.Nsa.clear()
        self.total_learning_times = 0

    def epsilon_greedy_policy(self, dealer, epsilon=None):
        player_points, _ = self.get_points()
        if player_points >= 21:
            return self.A[1]
        if player_points < 12:
            return self.A[0]
        else:
            A, Q = self.A, self.Q
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0 / (1 + 4 * math.log10(1 + player.total_learning_times))
                return epsilon_greedy_policy(A, s, Q, epsilon)



