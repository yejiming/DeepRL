import numpy as np
import numpy.random as nr
from gym.spaces import prng
import matplotlib.pyplot as plt

from DeepRL.hw2.frozen_lake import FrozenLakeEnv


def random_P():

    def random_reward():
        prob = nr.random(3)
        prob /= prob.sum()
        reward = nr.random(3)
        return [(prob[0], 0, reward[0]), (prob[1], 1, reward[1]), (prob[2], 2, reward[2])]

    return {
        0: {0: random_reward(), 1: random_reward()},
        1: {0: random_reward(), 1: random_reward()},
        2: {0: random_reward(), 1: random_reward()}
    }


class MDP(object):

    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
        self.iter = 0

    def next_iter(self):
        self.iter += 1


class MYMDP(object):

    def __init__(self):
        self.P = random_P()
        self.nS = 3
        self.nA = 2
        self.iter = 0

    def next_iter(self):
        self.iter += 1
        if self.iter % 50 == 0:
            self.P = random_P()


def begin_grading():
    print("\x1b[43m")


def end_grading():
    print("\x1b[0m")


env = FrozenLakeEnv()
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
env.seed(0)
prng.seed(10)


def value_iteration(mdp, gamma, nIt):
    """
    Inputs:
        mdp: MDP
        gamma: discount factor
        nIt: number of iterations, corresponding to n above
    Outputs:
        (value_functions, policies)

    len(value_functions) == nIt+1 and len(policies) == n
    """
    print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)]  # list of value functions contains the initial value function V^{(0)}, which is zero
    pis = []
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None  # \pi^{(it)} = Greedy[V^{(it-1)}]. Just used for printout
        Vprev = Vs[-1]  # V^{(it)}
        # YOUR CODE HERE
        V = np.zeros(mdp.nS)
        pi = np.zeros(mdp.nS)
        for state, action_map in mdp.P.items():
            maps = [action_map[i] for i in range(mdp.nA)]
            values = [sum([p * (r + gamma * Vprev[s]) for p, s, r in map_]) for map_ in maps]
            V[state] = np.max(values)
            pi[state] = np.argmax(values)
        pi = pi.astype(int)

        max_diff = np.abs(V - Vprev).max()
        nChgActions = "N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f" % (it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
        mdp.next_iter()
    return Vs, pis


def compute_vpi(pi, mdp, gamma):
    # YOUR CODE HERE
    a = []
    b = []
    for state, action_map in mdp.P.items():
        map_ = action_map[pi[state]]
        a_i = np.zeros(mdp.nS)
        a_i[state] = 1
        b_i = 0
        for p, s, r in map_:
            a_i[s] -= p * gamma
            b_i += p * r
        a.append(a_i)
        b.append(b_i)
    V = np.linalg.solve(a, b)
    return V


def compute_qpi(vpi, mdp, gamma):
    # YOUR CODE HERE
    Qpi = np.zeros((mdp.nS, mdp.nA))
    for state, action_map in mdp.P.items():
        for action, map_ in action_map.items():
            value = sum([p * (r + gamma * vpi[s]) for p, s, r in map_])
            Qpi[state, action] = value
    return Qpi


def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis


mdp = MDP({s: {a: [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)

# Problem 1
GAMMA = 0.95
begin_grading()
Vs_VI, pis_VI = value_iteration(mdp, gamma=GAMMA, nIt=20)
end_grading()

# Problem 2
chg_iter = 50
mymdp = MYMDP()
begin_grading()
Vs, pis = value_iteration(mymdp, gamma=GAMMA, nIt=chg_iter+1)
end_grading()

# Problem 3a
begin_grading()
print(compute_vpi(np.ones(16), mdp, gamma=GAMMA))
end_grading()

# Problem 3b
begin_grading()
Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Qpi:\n", Qpi)
end_grading()

Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
for s in range(5):
    plt.figure()
    plt.plot(np.array(Vs_VI)[:,s])
    plt.plot(np.array(Vs_PI)[:,s])
    plt.ylabel("value of state %i"%s)
    plt.xlabel("iteration")
    plt.legend(["value iteration", "policy iteration"], loc='best')
plt.show()