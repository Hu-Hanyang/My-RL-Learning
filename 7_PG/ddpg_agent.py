from gym import Env, spaces
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from core import Transition, Experience, Agent
from utils import soft_update, hard_update
from utils import OrnsteinUhlenbeckActionNoise
from network import Actor, Critic


class DDPGAgent(Agent):
    def __init__(self, env,
                 capacity=2e-6,
                 batch_size=128,
                 action_lim=1,
                 learning_rate=0.001,
                 epochs=2):
        '''
        DDPG uses both actor and critic network
        :param env: the environment that the agent interacts with
        :param capacity: the capacity of the replay buffer
        :param batch_size: the size of training batch
        :param action_lim: the upper and lower bounds
        :param learning_rate: the update learning rate
        :param epochs: I don't know what it is
        '''
        if env is None:
            raise "agent should have one environment"
        super(DDPGAgent, self).__init__(env, capacity)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_lim = action_lim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = 0.999
        self.epochs = epochs
        self.tau = 0.001  # the coefficient of soft copy
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)
        # actor network and optimizer
        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)
        # critic network and optimier
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        # total copy(hard copy) and soft copy
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        '''
        得到给定状态下依据目标actor network计算出的action，不探索
        :param state: numpy array
        :return: numpy array
        '''
        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        '''
        得到给定状态下根据演员网络计算出的带噪声的行为，模拟一定的探索
        :param state: numpy array
        :return: numpy array
        '''
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        new_action = new_action.clip(min=-1 * self.action_lim,
                                     max=self.action_lim)
        return new_action

    def _learn_from_memory(self):
        trans_pieces = self.sample(self.batch_size)
        s0 = np.vstack([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        s1 = np.vstack([x.s1 for x in trans_pieces])

        # optimize the critic network
        a1 = self.target_actor.forward(s1).detach()
        next_val = torch.squeeze(self.target_critic.forward(s1, a1).detach())
        y_expected = r1 + self.gamma * next_val
        y_expected = y_expected.type(torch.FloatTensor)
        a0 = torch.from_numpy(a0)
        y_predicted = torch.squeeze(self.critic.forward(s0, a0))
        ## compute loss and back-propagation
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        # optimize the actor network
        pred_a0 = self.actor.forward(s0)
        ## compute the loss and gradient ascent
        loss_actor = -1 * torch.sum(self.critic.forward(s0, pred_a0))
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        # soft updating parameters
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        return loss_critic.item(), loss_actor.item()

    def learning_method(self, display=False, explore = True):
        self.state = np.float64(self.env.reset())
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss_critic, loss_actor = 0.0, 0.0
        while not is_done:
            s0 = self.state
            if explore:
                a0 = self.get_exploitation_action(s0)
            else:
                a0 = self.actor.forward(s0).detach().data.numpy()
            s1, r1, is_done, info, total_reward = self.act(a0)

            if display:
                self.env.render()

            if self.total_trans > self.batch_size:
                loss_c, loss_a = self._learn_from_memory()
                loss_critic += loss_c
                loss_actor += loss_a

            time_in_episode += 1
        loss_critic /= time_in_episode
        loss_actor /= time_in_episode
        if display:
            print("{}".format(self.experience.last_episode))
        return time_in_episode, total_reward, loss_critic, loss_actor

    def learning(self, max_episode_num=800, display=False, explore=True):
        total_time, episode_reward, num_episode = 0, 0, 0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)):
            time_in_episode, episode_reward, loss_critic, loss_actor = \
                self.learning_method(display=display, explore=explore)
            total_time += time_in_episode
            num_episode += 1
            total_times.append(total_time)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
            print("episode:{:3}：loss critic:{:4.3f}, J_actor:{:4.3f}". \
                  format(num_episode - 1, loss_critic, -loss_actor))
            if explore and num_episode % 100 == 0:
                self.save_models(num_episode)
        return total_times, episode_rewards, num_episodes

    def save_models(self, episode_count):
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print("Models saved successfully")

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print("Models loaded succesfully")


