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

# MDP
# verify the q\pi(s,a)
from utils import str_key, display_dict
from utils import set_prob, set_reward, get_prob, get_reward
from utils import set_value, set_pi, get_value, get_pi

# construct MDP
S = ['Phone', 'C1', 'C2', 'C3', 'Break']  # 5 states
A = ['Play Phone', 'Study', 'Hold Phone', 'in Bar', 'Leave Study']  # 5 actions
R = {}  # Rsa dictionary
P = {}  # Pss'a dictionary
gamma1 = 1.0
# set transition probability based on the s, a and s'
set_prob(P, S[0], A[0], S[0])
set_prob(P, S[0], A[2], S[1])
set_prob(P, S[1], A[0], S[0])
set_prob(P, S[1], A[1], S[2])
set_prob(P, S[2], A[1], S[3])
set_prob(P, S[2], A[4], S[4])
set_prob(P, S[3], A[1], S[4])
set_prob(P, S[3], A[3], S[1], p=0.2)
set_prob(P, S[3], A[3], S[2], p=0.4)
set_prob(P, S[3], A[3], S[3], p=0.4)
# set transition rewards based on the s and a
set_reward(R, S[0], A[0], -1)
set_reward(R, S[0], A[2], 0)
set_reward(R, S[1], A[0], -1)
set_reward(R, S[1], A[1], -2)
set_reward(R, S[2], A[1], -2)
set_reward(R, S[2], A[4], 0)
set_reward(R, S[3], A[1], 10)
set_reward(R, S[3], A[3], +1)
# display_dict(R)
MDP = (S, A, R, P, gamma)  # define MDP as a tuple
# set random policy: pi(a|s) = 0.5
Pi = {}
set_pi(Pi, S[0], A[0], 0.5)
set_pi(Pi, S[0], A[2], 0.5)
set_pi(Pi, S[1], A[0], 0.5)
set_pi(Pi, S[1], A[1], 0.5)
set_pi(Pi, S[2], A[1], 0.5)
set_pi(Pi, S[2], A[4], 0.5)
set_pi(Pi, S[3], A[1], 0.5)
set_pi(Pi, S[3], A[3], 0.5)
# display_dict(Pi)
# initialize value function
V = {}
# display_dict(V)


# qpi(s,a) = r(s,a) + gamma \sum p vpi(s')
def compute_q(MDP, V, s, a):
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)  # p vpi(s')
        q_sa = get_reward(R, s, a) + gamma * q_sa
        return q_sa


# vpi(s) = \sum pi(a|s) * qpi(s,a)
def compute_v(MDP, V, Pi, s):
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)
    return v_s


# use policy iteration to calculate the value function v
def v_bellman(MDP, V, Pi):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime


def policy_evaluation(MDP, V, Pi, n):
    for i in range(n):
        V = v_bellman(MDP, V, Pi)
    return V

V = policy_evaluation(MDP, V, Pi, 100)
# display_dict(V)






