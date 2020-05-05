from utils import *
from base_learner import *


class FriendQLearning(BaseLearner):

    def __init__(self):
        super(FriendQLearning, self).__init__()
        self.no_iter = 200000
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999995


    def epsilon_greedy(self, Q, state, epsilon):
        state_a, state_b, state_ball = state
        temp = np.random.random()
        if temp < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            actions = np.where(Q[state_a, state_b, state_ball, :, :] == np.max(Q[state_a, state_b, state_ball, :, :]))
            action = actions[0][np.random.choice(range(len(actions[0])), 1)[0]]
        return action

    def learn(self):
        errors = []
        Q_a = np.zeros((8, 8, 2, 5, 5))
        Q_b = np.zeros((8, 8, 2, 5, 5))
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        env = SoccerGame()
        i = 0
        while i < self.no_iter:
            env.reset()
            state = env.state_encode()
            while True:
                if i % 10000 == 1:
                    print(str(errors[-1]))
                before_value = Q_a[2][1][1][2][4]
                actions = [self.epsilon_greedy(Q_a, state, epsilon), self.epsilon_greedy(Q_b, state, epsilon)]
                state_new, rewards, done = env.step(actions)
                i += 1
                if done:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                        alpha * (rewards[0] - Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                        alpha * (rewards[1] - Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    break
                else:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                        alpha * (rewards[0] + gamma * np.max(Q_a[state_new[0], state_new[1], state_new[2], :, :]) -
                        Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                        alpha * (rewards[1] + gamma * np.max(Q_b[state_new[0], state_new[1], state_new[2], :, :]) -
                        Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    state = state_new
                epsilon *= self.epsilon_decay
                epsilon = max(self.epsilon_min, epsilon)
                # alpha *= self.alpha_decay
                # alpha = max(self.alpha_min, alpha)
                alpha = 1 / (i / self.alpha_min / self.no_iter + 1)
        plot_error(errors, "friend_q_learning_3_2")
        return


if __name__ == "__main__":
    learner = FriendQLearning()
    learner.learn()
