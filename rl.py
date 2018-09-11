from gridworld import GridWorldMDP
from qlearn import QLearner

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(utility_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in Best Policy', color='r')


if __name__ == '__main__':
    shape = (3, 4)
    goal = (0, -1)
    trap = (1, -1)
    obstacle = (1, 1)
    start = (2, 0)
    default_reward = -0.1
    goal_reward = 1
    trap_reward = -1

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal] = goal_reward
    reward_grid[trap] = trap_reward
    reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    terminal_mask[goal] = True
    terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)
    obstacle_mask[1, 1] = True

    gw = GridWorldMDP(reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      action_probabilities=[
                          (-1, 0.1),
                          (0, 0.8),
                          (1, 0.1),
                      ],
                      no_action_probability=0.0)

    mdp_solvers = {'Value Iteration': gw.run_value_iterations,
                   'Policy Iteration': gw.run_policy_iterations}

    for solver_name, solver_fn in mdp_solvers.items():
        print('Final result of {}:'.format(solver_name))
        policy_grids, utility_grids = solver_fn(iterations=25, discount=0.5)
        print(policy_grids[:, :, -1])
        print(utility_grids[:, :, -1])
        plt.figure()
        gw.plot_policy(utility_grids[:, :, -1])
        plot_convergence(utility_grids, policy_grids)
        plt.show()

    ql = QLearner(num_states=(shape[0] * shape[1]),
                  num_actions=4,
                  learning_rate=0.8,
                  discount_rate=0.9,
                  random_action_prob=0.5,
                  random_action_decay_rate=0.99,
                  dyna_iterations=0)

    start_state = gw.grid_coordinates_to_indices(start)

    iterations = 1000
    flat_policies, flat_utilities = ql.learn(start_state,
                                             gw.generate_experience,
                                             iterations=iterations)

    new_shape = (gw.shape[0], gw.shape[1], iterations)
    ql_utility_grids = flat_utilities.reshape(new_shape)
    ql_policy_grids = flat_policies.reshape(new_shape)
    print('Final result of QLearning:')
    print(ql_policy_grids[:, :, -1])
    print(ql_utility_grids[:, :, -1])

    plt.figure()
    gw.plot_policy(ql_utility_grids[:, :, -1], ql_policy_grids[:, :, -1])
    plot_convergence(ql_utility_grids, ql_policy_grids)
    plt.show()
