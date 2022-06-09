# state space
S = [i for i in range(16)]
# action space
A = ['n', 'e', 's', 'w']
ds_actions = {'n': -4, 'e': 1, 's': 4, 'w': -1}


# environment dynamics
def dynamics(s,a):
    s_prime = s
    if(s%4 == 0 and a == 'w') or (s<4 and a == 'n') or ((s+1)%4 == 0 and a == "e") or (s > 11 and a == "s") \
            or s in [0, 15]:  # 最左侧向左动 or 第一排向上动 or 最右侧向右动 or 第四排向下动 or 在首尾两个位置 这里留个？
        pass
    else:
        ds = ds_actions[a]
        s_prime = s + ds
    reward = 0 if s in [0, 15] else -1
    done = True if s in [0, 15] else False
    return s_prime, reward, done


# states transition probability function
def P(s, a, s1):
    s_prime, _, _ = dynamics(s, a)
    return s1 == s_prime


def R(s,a):
    _, r, _ = dynamics(s, a)
    return r

gamma = 1.00
MDP = (S, A, R, P, gamma)



