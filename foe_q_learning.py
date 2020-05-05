from utils import *
from base_learner import *
from cvxpy import Variable, Problem, Minimize


class FoeQLearning(BaseLearner):

    def __init__(self):
        super(FoeQLearning, self).__init__()
        self.no_iter = 1e6
        # self.epsilon_decay = 0
        # self.epsilon = 0
        self.alpha = 1.0
        self.alpha_decay = (self.alpha_min / self.alpha) ** (1. / self.no_iter)

    def epsilon_greedy(self, pi, state, epsilon):
        state_a, state_b, state_ball = state
        temp = np.random.random()
        if temp < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            prob = np.abs(pi[state_a, state_b, state_ball]) / np.sum(np.abs(pi[state_a, state_b, state_ball]))
            action = np.random.choice([0, 1, 2, 3, 4], 1, p=prob)[0]
        return action

    def max_min(self, A):
        num_action = A.shape[0]
        w = Variable(num_action + 1)
        eye = np.identity(num_action)
        additional_1 = np.ones((1, num_action))
        additional_2 = np.zeros((1, num_action))
        additional = np.concatenate([additional_1, additional_2], axis=1)
        ori_eye = np.concatenate([-A, -eye], axis=1)
        total = np.concatenate([additional, ori_eye], axis=0)
        total_func = w.T * total

        object_vec = np.zeros(((num_action + 1), 1))
        object_vec[0, 0] = -1
        object_func = w.T * object_vec
        sum_vec = np.ones(((num_action + 1), 1))
        sum_vec[0, 0] = 0
        sum_func = w.T * sum_vec

        prob = Problem(Minimize(object_func),
                       [sum_func == 1, total_func <= 0])
        prob.solve()
        weights = w.value
        return weights[1:], weights[0]

    def update_weight(self, Q, state):
        state_a, state_b, state_ball = state
        A = Q[state_a, state_b, state_ball]
        return self.max_min(A)

    def learn(self):
        errors = []
        Q_a = np.zeros((8, 8, 2, 5, 5))
        Q_b = np.zeros((8, 8, 2, 5, 5))
        pi_a = np.ones((8, 8, 2, 5)) * 1 / 5
        pi_b = np.ones((8, 8, 2, 5)) * 1 / 5
        V_a = np.ones((8, 8, 2))
        V_b = np.ones((8, 8, 2))
        epsilon = epsilon_decay = 10**(np.log10(self.epsilon_min)/self.no_iter)
        alpha = alpha_decay = 10**(np.log10(self.alpha_min)/self.no_iter)
        gamma = self.gamma
        env = SoccerGame()
        i = 0
        while i < self.no_iter:
            env.reset()
            state = env.state_encode()
            pi, val = self.update_weight(Q_a, state)
            pi_a[state[0]][state[1]][state[2]] = pi
            V_a[state[0]][state[1]][state[2]] = val
            pi, val = self.update_weight(Q_b, state)
            pi_b[state[0]][state[1]][state[2]] = pi
            V_b[state[0]][state[1]][state[2]] = val
            while True:
                if i % 1000 == 1:
                    print(str(errors[-1]))
                before_value = Q_a[2][1][1][2][4]
                actions = [self.epsilon_greedy(pi_a, state, epsilon), self.epsilon_greedy(pi_b, state, epsilon)]
                state_new, rewards, done = env.step(actions)
                pi, val = self.update_weight(Q_a, state_new)
                pi_a[state_new[0]][state_new[1]][state_new[2]] = pi
                V_a[state_new[0]][state_new[1]][state_new[2]] = val
                pi, val = self.update_weight(Q_b, state_new)
                pi_b[state_new[0]][state_new[1]][state_new[2]] = pi
                V_b[state_new[0]][state_new[1]][state_new[2]] = val
                i += 1
                if done:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                        alpha * (rewards[0] + gamma * V_a[state_new[0], state_new[1], state_new[2]] -
                                 Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                        alpha * (rewards[1] + gamma * V_b[state_new[0], state_new[1], state_new[2]] -
                                 Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    break
                else:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                            alpha * (rewards[0] + gamma * V_a[state_new[0], state_new[1], state_new[2]] -
                            Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                            alpha * (rewards[1] + gamma * V_b[state_new[0], state_new[1], state_new[2]] -
                            Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    state = state_new
                # epsilon *= self.epsilon_decay
                # epsilon = max(self.epsilon_min, epsilon)
                # alpha *= self.alpha_decay
                # alpha = max(self.alpha_min, alpha)
                alpha = alpha_decay ** i
                epsilon = epsilon_decay ** i
        plot_error(errors, "foe_q_learning_final_2")
        return


if __name__ == "__main__":
    learner = FoeQLearning()
    learner.learn()

