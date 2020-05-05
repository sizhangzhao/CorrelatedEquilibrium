from utils import *
from base_learner import *


class Sarsa(BaseLearner):

    def epsilon_greedy(self, Q, state, epsilon):
        state_a, state_b, state_ball = state
        temp = np.random.random()
        if temp < epsilon:
            action = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        else:
            action = np.argmax(Q[state_a, state_b, state_ball, :])
        return action

    def learn(self):
        errors = []
        Q_a = np.zeros((8, 8, 2, 5))
        Q_b = np.zeros((8, 8, 2, 5))
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.gamma
        env = SoccerGame()
        i = 0
        while i < self.no_iter:
            env.reset()
            state = env.state_encode()
            actions = [self.epsilon_greedy(Q_a, state, epsilon), self.epsilon_greedy(Q_b, state, epsilon)]
            while True:
                if i % 10000 == 1:
                    print(str(errors[-1]))
                before_value = Q_a[2][1][1][2]
                state_new, rewards, done = env.step(actions)
                actions_new = [self.epsilon_greedy(Q_a, state_new, epsilon), self.epsilon_greedy(Q_b, state_new, epsilon)]
                i += 1
                if done:
                    Q_a[state[0], state[1], state[2], actions[0]] = Q_a[state[0], state[1], state[2], actions[0]] + \
                        alpha * (rewards[0] - Q_a[state[0], state[1], state[2], actions[0]])
                    Q_b[state[0], state[1], state[2], actions[1]] = Q_b[state[0], state[1], state[2], actions[1]] + \
                        alpha * (rewards[1] - Q_b[state[0], state[1], state[2], actions[1]])
                    after_value = Q_a[2][1][1][2]
                    errors.append(abs(before_value - after_value))
                    break
                else:
                    Q_a[state[0], state[1], state[2], actions[0]] = Q_a[state[0], state[1], state[2], actions[0]] + \
                        alpha * (rewards[0] + gamma * Q_a[state_new[0], state_new[1], state_new[2], actions_new[0]] -
                        Q_a[state[0], state[1], state[2], actions[0]])
                    Q_b[state[0], state[1], state[2], actions[1]] = Q_b[state[0], state[1], state[2], actions[1]] + \
                        alpha * (rewards[1] + gamma * Q_b[state_new[0], state_new[1], state_new[2], actions_new[1]] -
                        Q_b[state[0], state[1], state[2], actions[1]])
                    after_value = Q_a[2][1][1][2]
                    errors.append(abs(before_value - after_value))
                    state = state_new
                    actions = actions_new
                epsilon *= self.epsilon_decay
                epsilon = max(self.epsilon_min, epsilon)
                alpha *= self.alpha_decay
                alpha = max(self.alpha_min, alpha)
        plot_error(errors, "sarsa")
        return


if __name__ == "__main__":
    learner = Sarsa()
    learner.learn()

