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
                v = 0  # 这里我有问题
        finally:
            return v

    def get_points(self):  # return total points and whether to use ace=1
        num_of_usable_ace = 0
        total_points = 0
        cards = self.cards
        if cards is None:
            return 0, False
        for card in cards:
            v = self._value_of(card)
            if v==1:
                num_of_usable_ace += 1
                v = 11
            total_points += v
            while total_points > 21 and num_of_usable_ace > 0:
                total_points -= 10
                num_of_usable_ace -= 1
        return total_points, bool(num_of_usable_ace)

    def receive(self, cards = []):
        cards = list(cards)
        for card in cards:
            self.cards.append(card)

    def discharge_cards(self):
        self.cards.clear()

    def cards_info(self):
        self.cards_info('{}{}现在的牌：{}\n'.format(self.role, self, self.cards))

    def _info(self, msg):
        if self.display:
            print(msg, end="")

