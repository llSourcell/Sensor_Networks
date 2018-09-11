import numpy as np
import random as rand


class QLearner:
    '''A generic implementation of Q-Learning and Dyna-Q'''

    def __init__(self, *,
                 num_states,
                 num_actions,
                 learning_rate,
                 discount_rate=1.0,
                 random_action_prob=0.5,
                 random_action_decay_rate=0.99,
                 dyna_iterations=0):

        self._num_states = num_states
        self._num_actions = num_actions
        self._learning_rate = learning_rate
        self._discount_rate = discount_rate
        self._random_action_prob = random_action_prob
        self._random_action_decay_rate = random_action_decay_rate
        self._dyna_iterations = dyna_iterations

        self._experiences = []

        # Initialize Q to small random values.
        self._Q = np.zeros((num_states, num_actions), dtype=np.float)
        self._Q += np.random.normal(0, 0.3, self._Q.shape)

    def initialize(self, state):
        '''Set the initial state and return the learner's first action'''
        self._decide_next_action(state)
        self._stored_state = state
        return self._stored_action

    def learn(self, initial_state, experience_func, iterations=100):
        '''Iteratively experience new states and rewards'''
        all_policies = np.zeros((self._num_states, iterations))
        all_utilities = np.zeros_like(all_policies)
        for i in range(iterations):
            done = False
            self.initialize(initial_state)
            for j in range(iterations):
                state, reward, done = experience_func(self._stored_state,
                                                      self._stored_action)
                self.experience(state, reward)
                if done:
                    break

            policy, utility = self.get_policy_and_utility()
            all_policies[:, i] = policy
            all_utilities[:, i] = utility
        return all_policies, all_utilities

    def experience(self, state, reward):
        '''The learner experiences state and receives a reward'''
        self._update_Q(self._stored_state, self._stored_action, state, reward)

        if self._dyna_iterations > 0:
            self._experiences.append(
                (self._stored_state, self._stored_action, state, reward)
            )
            exp_idx = np.random.choice(len(self._experiences),
                                       self._dyna_iterations)
            for i in exp_idx:
                self._update_Q(*self._experiences[i])

        # determine an action and update the current state
        self._decide_next_action(state)
        self._stored_state = state

        self._random_action_prob *= self._random_action_decay_rate

        return self._stored_action

    def get_policy_and_utility(self):
        policy = np.argmax(self._Q, axis=1)
        utility = np.max(self._Q, axis=1)
        return policy, utility

    def _update_Q(self, s, a, s_prime, r):
        best_reward = self._Q[s_prime, self._find_best_action(s_prime)]
        self._Q[s, a] *= (1 - self._learning_rate)
        self._Q[s, a] += (self._learning_rate
                          * (r + self._discount_rate * best_reward))

    def _decide_next_action(self, state):
        if rand.random() <= self._random_action_prob:
            self._stored_action = rand.randint(0, self._num_actions - 1)
        else:
            self._stored_action = self._find_best_action(state)

    def _find_best_action(self, state):
        return int(np.argmax(self._Q[state, :]))
