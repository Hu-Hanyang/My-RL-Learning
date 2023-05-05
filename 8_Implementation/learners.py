import numpy as np
import torch
from replay_buffer import ReplayBuffer
from networks import Qnet
from torch.utils.tensorboard import SummaryWriter

class DQN_Agent():
    def __init__(self, gamma, epsilon, eps_end, eps_decay, lr, n_state, hidden, n_action, batch_size, max_size, device):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.n_state = n_state
        self.n_action = n_action
        self.hidden = hidden
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(max_size)
        self.device = device
        self.steps = 0  # to document the training loss

        self.policy_net = Qnet(self.n_state, self.n_action, self.hidden).to(self.device)
        self.target_net = Qnet(self.n_state, self.n_action, self.hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        
    def choose_action(self, state):
        sample = np.random.random_sample()  # [0.0, 1.0)
        # eps_threshold = self.eps_end + (self.epsilon - self.eps_end) * np.exp(-1.* step/self.eps_decay)
        eps_threshold = self.epsilon
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.policy_net(torch.tensor(state, device=self.device, dtype=torch.float32)).argmax().item()
                return action
        else:  # randomly
            return np.random.randint(0, self.n_action)
        
    def train(self):
        if len(self.replay_buffer.storage) < self.batch_size:
            return  # replay buffer is not enough
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_Q = self.target_net(next_states).max(1)[0]  # the maximum next Q value
        expected_Q = rewards + self.gamma * next_Q * (1 - dones)

        loss = self.loss_fn(Q, expected_Q.detach()) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
    
    def updatae_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, model_path):
        torch.save(self.target_net.state_dict(), model_path)