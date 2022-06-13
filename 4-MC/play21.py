from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import str_key, set_dict, get_dict


# define a father class Game
class Game:
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
            if v==1:  # 这里很巧妙，先默认A=11，然后在后面用总点数进行判断与修改
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

# player1 = Game()
# cards1 = ['A', 'Q', '9']
# player1.receive(cards1)
# print(player1.cards)
# print(player1.get_points())


class Dealer(Game):
    def __init__(self, name = '', A = None, display = False):
        super(Dealer, self).__init__(name, A, display)
        self.role = '庄家'
        self.policy = self.dealer_policy

    def first_card_value(self):
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self, Dealer = None):
        action = ''
        dealer_points, _ = self.get_points()
        if dealer_points >= 17:
            action = self.A[1]  # stop
        else:
            action = self.A[0]  # continue
        return action


class Player(Game):
    def __init__(self, name = '', A = None, display = False):
        super(Player, self).__init__(name, A, display)
        self.policy =  self.naive_policy
        self.role = '玩家'

    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return dealer_first_card_value, player_points, useable_ace

    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))

    def naive_policy(self, dealer = None):  # naive policy和dealer无关
        player_points, _ = self.get_points()
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action


class Arena():  # 游戏管理者
    def __init__(self, display = None, A = None):
        self.cards = ['A', '2', '3','4','5','6','7','8','9', '10', 'J', 'Q', 'K']*4
        self.card_q = Queue(maxsize=52)  # 洗好的牌
        self.cards_in_pool = []  # 已经用过的牌
        self.display = display
        self.episodes = []  # 产生的对局信息列表
        self.load_cards(self.cards)  # 把初始状态的52张牌装入发牌器
        self.A = A

    def load_cards(self, cards):  # 洗牌并且把牌装进发牌器
        shuffle(cards)
        for card in cards:
            self.card_q.put(card)
        cards.clear()
        return

    def serve_card_to(self, gamer, n = 1):  # 给dealer或者player发1张牌
        cards = []  # 将要发出去的牌
        for _ in range(n):
            if self.card_q.empty():
                self._info('\n 发牌器没牌了，整理废牌，重新洗牌；')
                shuffle(self.cards_in_pool)
                self._info('一共整理了{}张已使用的牌，重新放入发牌器\n'.format(len(self.cards_in_pool)))
                assert(len(self.cards_in_pool) > 20)
                self.load_cards(self.cards_in_pool)
            cards.append(self.card_q.get())
        self._info('发了{}张牌({})给{}{}'.format(n, cards, gamer.role, gamer))
        gamer.reveive(cards)
        gamer.cards_info()

    def reward_of(self, dealer, player):  # 判断谁赢了
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, player_points, dealer_points, useable_ace

    def _info(self, message):
        if self.display:
            print(message, end='')

    def recycle_cards(self, *players):
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards()

    def play_game(self, dealer, player):
        self._info("========= 开始新的一局 =========\n")
        self.serve_card_to(player, n=2)
        self.serve_card_to(dealer, n=2)
        episode = []
        if player.policy is None:
            self._info('玩家得有一个策略啊！')
            return
        if dealer.policy is None:
            self._info('庄家也得整一个策略')
            return
        while True:  # 先是玩家决定动作
            action = player.policy(dealer)
            self._info('{}{}选择：{}'.format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action))  # add (s, a)
            if action == self.A[0]:
                self.serve_card_to(player)
            else:
                break
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        # 玩家已经不能再叫牌，庄家可以叫牌，先判断谁输谁赢
        if player_points > 21:
            self._info('玩家手牌点数合计为{}超过21，他输了hhh，得分为：{}\n'.format(player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward))
            self._info('========= 本局结束 ==========\n')
            return episode, reward
        #  玩家没有超过21点，现在看庄家要不要再抽卡
        self._info('\n')
        while True:
            action = dealer.policy()
            self._info("{}{}选择:{};".format(dealer.role, dealer, action))
            if action == self.A[0]:  # 庄家继续叫牌
                self.serve_card_to(dealer)
            else:
                break
        # 双方都不能再叫牌了，开始清算准备结束这个episode
        self._info("\n双方均停止叫牌;\n")
        reward, player_points, dealer_points, useable_ace =self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()
        if reward == 1:
            self._info('玩家居然赢了！')
        elif reward == -1:
            self._info('玩家还是输了啊！')
        else:
            self._info('打平打平啦～')
        self._info(" 玩家{}点,庄家{}点\n".format(player_points, dealer_points))
        self._info("========= 本局结束 ==========\n")
        self.recycle_cards(player, dealer)  # 回收洗牌
        self.episodes.append((episode, reward))  # 将这个episode放在episodes中
        return episode, reward

    def play_games(self, dealer, player, num = 2, show_statistic = True):
        player_results = [0, 0, 0]  # 玩家输-平局-胜利 的局数
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            player_results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        if show_statistic:
            print('一共玩了{}局，其中玩家赢了{}局，打平{}局， 玩家输了{}局，胜率为：{:.2f}， 不输掉牌局的概率为：{:.2f}'.format(
                num, player_results[2], player_results[1], player_results[0], player_results[2]/num,
                (player_results[2]+player_results[1])/num))
            return

    def _info(self, message):
            if self.display:
                print(message, end='')






