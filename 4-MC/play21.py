from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import str_key, set_dict, get_dict


# define a father class Game
class Game():
    def __init__(self, name='', A=None, display=False):
        self.name = name
        self.cards = []  # cards in hand
        self.display = display  # whether display the details of the game
        self.policy = None
        self.learning_method = None
        self.A = A  # action space

    def __str__(self):
        return self.name

    def _value_of(self, card):
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J', 'Q', 'K']:
                v = 10
            else:
                v = 0
        finally:
            return v

    def get_points(self):  # return total points and whether to use ace=1
        num_of_usable_ace = 0
        total_points = 0
        cards = self.cards
        if cards is None:
            return 0, False
