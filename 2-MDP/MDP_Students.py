import numpy as np

# MRP
num_states = 7
# {"0":"C1", "1":"C2", "2":"C3", "3":"Pass", "4":"Pub", "5":"FB", "6":"Sleep"}
i_to_n = {"0": "C1", "1": "C2", "2": "C3", "3": "Pass", "4": "Pub", "5": "FB",
          "6": "Sleep"}  # dictionary that contains index to states

n_to_i = {}  # dictionary that contains states to index
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

# states transition matrix : C1, C2, C3, Pass, Pub, FB, Sleep
Pss = [
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]
Pss = np.array(Pss)

# rewards
rewards = [-2, -2, -2, 10, 1, -1, 0]  # here one state corresponds one reward
gamma = 0.5  # discount factor


# calculation of returns Gt
def compute_return(start_index=0, chain=None, gamma=0.5):
    retrn, power, gamma = 0.0, 0, gamma
    if start_index > len(chain):
        print('The state index is wrong!')
    else:
        for i in range(start_index, len(chain)):
            retrn += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
            power += 1
    return retrn


chains = [
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB", "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

test = compute_return(2, chains[0], gamma=0.5)


# print(test)


# analytical solution: V = (I - \gamma P)^(-1)R
def compute_value(Pss, rewards, num_states=7, gamma=0.05):
    rewards = np.array(rewards).reshape((-1, 1))  # convert rewards to np.array and column vector
    values = np.dot(np.linalg.inv(np.eye(num_states, num_states) - gamma * Pss), rewards)
    return values


values1 = compute_value(Pss, rewards, gamma=0.99999)
# print(values1)

# verify the q\pi(s,a)
from utils import str_key, display_dict
from utils import set_prob, set_reward, get_prob, get_reward
from utils import set_value, set_pi, get_value, get_pi