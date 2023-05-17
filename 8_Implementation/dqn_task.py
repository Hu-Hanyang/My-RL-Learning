import gymnasium as gym
import torch
import random
import numpy as np
from learners import DQN_Agent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# some (hyper)parameters 
hidden = 128
gamma = 0.99
epsilon = 0.1
eps_end = 0.05
eps_decay = 1000
tau = 0.005  # update rate of the target network
lr = 1e-4  # learning rate of the optimizer
n_update = 100  # the update number of the target work
batch_size = 64
max_size = 800
num_episode = 1500

# initialize the env
env = gym.make("CartPole-v1", render_mode="rgb_array")
n_state = env.observation_space.shape[0]  # n_state = 4
n_action = env.action_space.n  # n_action = 2

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The current device is {device}.")
# initilize the agent
agent = DQN_Agent(gamma, epsilon, eps_end, eps_decay, lr, n_state, hidden, n_action, batch_size, max_size, device)
writer = SummaryWriter('./dqn_records')


# training
for episode in tqdm(range(num_episode)):
    rewards = 0.0
    state, info = env.reset()
    for _ in range(n_update):
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        rewards += reward
        agent.replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        agent.train()
    
        if episode % 100 == 0:
            env.render()

        if done:
            break
    agent.updatae_target()
    epsilon = np.maximum(epsilon*eps_decay, eps_end)
    writer.add_scalar("Rewards", rewards, episode)

print("Saving the model.............")
model_path = "./dqn_model.pt"
agent.save_model(model_path)  